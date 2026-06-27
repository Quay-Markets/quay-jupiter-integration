//! Jupiter `Amm` adapter for the Quay program.
//!
//! Mirrors the contract in
//! `jup-ag/jupiter-amm-implementation/jupiter-core/src/amms/spl_token_swap_amm.rs`
//! against `jupiter-amm-interface = "0.6"`. Lives in its own crate so the
//! Jupiter dependency cone doesn't infect `quay-sdk`'s build.
//!
//! Each `QuayAmm` instance represents **one Strategy** — a pricing curve
//! bound to a single `(base_mint, quote_mint)` pair on one MarketMaker.
//! Jupiter's resolver holds many of these (one per active strategy) and
//! shares quote distribution across them.
//!
//! # Account-tracking model
//!
//! `get_accounts_to_update` returns eight pubkeys:
//! 1. the **Strategy** itself (bytecode + userspace + frozen flags + fee bps),
//! 2. the **MarketMaker** the strategy is anchored to (asset table inventories
//!    + halt flags),
//! 3. the strategy's bound **Quotes** account (live quote slots + freshness),
//! 4. **GlobalConfig** (halt flags),
//! 5. and 6. the base and quote **vault** token accounts (swap-ix accounts
//!    only — the VM no longer prices off vault balances),
//! 7. and 8. the base and quote **mints** (decimals + Token-2022 owner +
//!    `TransferFeeConfig` detection for the `is_active()` gate).
//!
//! Clock is **not** in this list — `AmmContext.clock_ref` is the live
//! source for `current_slot` / `current_unix_sec`; the resolver updates
//! the shared atomics every slot.
//!
//! Heavy decoding lives in `update`; `quote` only re-runs the VM. This
//! follows the Jupiter integration guideline: "Move everything that needs
//! to be done only when the state changes to update, rather than quote".
//!
//! # `Swap` enum
//!
//! `jupiter-amm-interface 0.6` ships a closed enum of supported swap
//! variants. When Jupiter merges Quay into Metis they add `Swap::Quay` and
//! patch the one return value below. Until then we return `Swap::TokenSwap`
//! as a placeholder — `account_metas` is correct in either case, and callers
//! building the swap ix outside the trait can use `QuayAmm::swap_account_metas()`
//! together with `quay_sdk::ix::swap()`.

use std::sync::atomic::Ordering;

use anyhow::{anyhow, Result as JupiterResult};
use jupiter_amm_interface::{
    single_program_amm, try_get_account_data, AccountMap, Amm, AmmContext, KeyedAccount, Quote,
    QuoteParams, SingleProgramAmm, Swap, SwapAndAccountMetas, SwapMode, SwapParams,
};
use solana_program::instruction::AccountMeta;
use solana_program::pubkey::Pubkey;

use quay_sdk::consts::{
    MAX_USERSPACE_LEN, ROUTE_JUPITER, SIDE_BUY_BASE, SIDE_SELL_BASE,
    SWAP_LOADED_ACCOUNTS_DATA_SIZE_LIMIT,
};
use quay_sdk::ix;
use quay_sdk::pda;
use quay_sdk::simulate::{simulate_swap_in, SwapSimulationInputs};
use quay_sdk::state::{GlobalConfig, MarketMakerHeader, StrategyHeader};

/// Swap account count for a **same-token-program** market. Mirrors the
/// on-chain layout in `instructions/swap.rs`: 11 program-read positional
/// accounts (cfg + strategy + mm + quotes + 2 vaults + 2 taker ATAs +
/// taker + 2 mints), the Instructions sysvar at index 11, and one
/// trailing token-program slot the program's parser skips but the
/// runtime must load for the transfer CPIs.
const SWAP_ACCOUNTS_LEN_SAME_PROGRAM: usize = 13;

/// Swap account count for a **mixed-token-program** market — adds one more
/// trailing program (Token-2022 base alongside SPL Token quote, or vice
/// versa). The SDK's `push_token_programs` dedup decides which case we're
/// in; `get_accounts_len` mirrors that.
const SWAP_ACCOUNTS_LEN_MIXED_PROGRAM: usize = 14;

/// SPL Token program id. Default `(base|quote)_token_program` until `update()`
/// can resolve the actual owner from the mint account — practically all
/// markets use it, and a wrong placeholder is corrected before the first quote.
const SPL_TOKEN_PROGRAM_ID: Pubkey = Pubkey::new_from_array([
    0x06, 0xdd, 0xf6, 0xe1, 0xd7, 0x65, 0xa1, 0x93, 0xd9, 0xcb, 0xe1, 0x46, 0xce, 0xeb, 0x79, 0xac,
    0x1c, 0xb4, 0x85, 0xed, 0x5f, 0x5b, 0x37, 0x91, 0x3a, 0x8c, 0xf5, 0x85, 0x7e, 0xff, 0x00, 0xa9,
]);

/// Token-2022 program id (`TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb`).
/// Used to gate the TLV-extension parser below — SPL Token accounts can't
/// carry extensions so the parser is skipped when the mint owner is the
/// classic program.
const TOKEN_2022_PROGRAM_ID: Pubkey = Pubkey::new_from_array([
    0x06, 0xdd, 0xf6, 0xe1, 0xee, 0x75, 0x8f, 0xde, 0x18, 0x42, 0x5d, 0xbc, 0xe4, 0x6c, 0xcd, 0xda,
    0xb6, 0x1a, 0xfc, 0x4d, 0x83, 0xb9, 0x0d, 0x27, 0xfe, 0xbd, 0xf9, 0x28, 0xd8, 0xa1, 0x8b, 0xfc,
]);

/// Token-2022 Mint extension TLV starts at byte 166: 82-byte base Mint +
/// 83-byte multisig padding + 1-byte `AccountType::Mint` discriminator.
const TOKEN_2022_TLV_OFFSET: usize = 166;

/// `ExtensionType::TransferFeeConfig` discriminator (u16 LE = `0x0001`).
const EXT_TRANSFER_FEE_CONFIG: u16 = 1;

/// Does this Token-2022 mint carry a `TransferFeeConfig`? Returns `false` for
/// SPL Token mints (they cannot have extensions) and for malformed TLV.
///
/// The on-chain `swap` does **not** subtract the transfer-fee from
/// `amount_in` before pricing, so a transfer-fee'd mint would mis-quote.
/// Refusing the venue from `is_active()` is the safe gate until the on-chain
/// program is taught to net the fee.
fn mint_has_transfer_fee(token_program: &Pubkey, data: &[u8]) -> bool {
    if *token_program != TOKEN_2022_PROGRAM_ID {
        return false;
    }
    let mut i = TOKEN_2022_TLV_OFFSET;
    while i + 4 <= data.len() {
        let ext_type = u16::from_le_bytes([data[i], data[i + 1]]);
        let length = u16::from_le_bytes([data[i + 2], data[i + 3]]) as usize;
        if ext_type == EXT_TRANSFER_FEE_CONFIG {
            return true;
        }
        // TLV value follows the 4-byte header; advance past it. Saturating
        // add guards against a length that would overflow `usize` on 32-bit
        // hosts — malformed TLV breaks the loop cleanly.
        let Some(next) = i.checked_add(4).and_then(|n| n.checked_add(length)) else {
            return false;
        };
        i = next;
    }
    false
}


/// Quay's mainnet program id (placeholder until deployment locks it in).
///
/// Devnet uses a different deploy key; integrators should set this via
/// `QuayAmm::with_program_id` if they're targeting devnet/testnet, but
/// `single_program_amm!` requires a `const Pubkey`, so we anchor the
/// label-resolution machinery on the deployed mainnet id.
pub const QUAY_PROGRAM_ID: Pubkey = solana_program::pubkey!("QUayE6nexQWYNZAEqfN8FxoNwQDSu3CAzT2qq9J1ArG");

#[derive(Clone)]
pub struct QuayAmm {
    program_id: Pubkey,
    /// The Strategy account is the Amm's primary key — Jupiter polls it.
    strategy_key: Pubkey,
    strategy_data: Vec<u8>,
    /// The MarketMaker the strategy is anchored to (derived from `strategy.owner`).
    mm_key: Pubkey,
    mm_data: Vec<u8>,
    /// Strategy's bound quotes account (`strategy.quotes_account`).
    quotes_key: Pubkey,
    quotes_data: Vec<u8>,
    global_config_key: Pubkey,
    global_config_data: Vec<u8>,
    base_mint: Pubkey,
    quote_mint: Pubkey,
    /// Token program owning each mint. Tracked separately so a mixed
    /// SPL/Token-2022 market settles correctly. Resolved from each mint
    /// account's `owner` on `update()`.
    base_token_program: Pubkey,
    quote_token_program: Pubkey,
    /// Mint decimals — both SPL Token and Token-2022 store this at offset 44
    /// of the mint account. Read once per update and fed to `simulate_swap`.
    base_decimals: u8,
    quote_decimals: u8,
    /// Cached vault PDAs. The pricing VM no longer reads vault balances
    /// (the `LoadVault*` opcodes were removed in the DSL-v1 redesign), so no
    /// vault data is cached — these keys exist only because the on-chain
    /// `swap` ix still takes the vaults as positional accounts, so they stay
    /// in `get_accounts_to_update` for the router.
    vault_base_key: Pubkey,
    vault_quote_key: Pubkey,
    /// Cached `AmmContext.clock_ref` so `quote` can read current slot +
    /// unix timestamp without re-threading them through the API.
    clock: jupiter_amm_interface::ClockRef,
    /// `StrategyHeader.routing_flags` — surfaced only when `ROUTE_JUPITER` is
    /// set, so an MM opts a curve into Jupiter explicitly. `0` (not routed)
    /// before the first decode.
    routing_flags: u8,
    /// Cached header bytes for the hot `is_active` path — every halt /
    /// freeze flag that gates `swap` on-chain.
    strategy_frozen: u8,
    strategy_frozen_admin: u8,
    mm_frozen: u8,
    mm_frozen_admin: u8,
    mm_halted_admin: u8,
    cfg_swap_halted: u8,
    cfg_protocol_halted: u8,
    /// Did `update()` find a `TransferFeeConfig` extension on either mint?
    /// The on-chain `swap` prices `amount_in` gross, so a transfer-fee mint
    /// would settle fewer atoms than the curve was quoted on. `is_active()`
    /// folds these in so the route planner skips the venue.
    base_has_transfer_fee: bool,
    quote_has_transfer_fee: bool,
}

// Anchor `QuayAmm` to its single program id. Routers iterating known AMM
// programs will pick up Quay via this trait; the macro also auto-generates
// the `AmmProgramIdToLabel` impl Jupiter uses for label dispatch.
single_program_amm!(QuayAmm, QUAY_PROGRAM_ID, "Quay");

impl QuayAmm {
    /// Account list the on-chain `swap` ix expects, in the order locked by
    /// `instructions/swap.rs`. Delegated to `quay_sdk::ix::swap` so the
    /// trailing token-program dedup matches the live builder.
    pub fn swap_account_metas(
        &self,
        taker: &Pubkey,
        taker_ata_base: &Pubkey,
        taker_ata_quote: &Pubkey,
        side: u8,
    ) -> Vec<AccountMeta> {
        self.swap_instruction(taker, taker_ata_base, taker_ata_quote, side, 0, 0)
            .accounts
    }

    /// Full on-chain `swap` instruction (data + metas), built from this
    /// strategy's cached PDAs. Mirrors the Titan adapter's
    /// `generate_swap_instruction` so callers that go around the closed
    /// `jupiter-amm-interface 0.6` `Swap` enum don't have to rebuild the
    /// data side themselves.
    ///
    /// `side`: `0 = SELL_BASE`, `1 = BUY_BASE`. The taker's two ATAs are
    /// supplied in canonical `(base, quote)` order regardless of side —
    /// `quay_sdk::ix::swap` resolves IN/OUT internally.
    #[allow(clippy::too_many_arguments)]
    pub fn swap_instruction(
        &self,
        taker: &Pubkey,
        taker_ata_base: &Pubkey,
        taker_ata_quote: &Pubkey,
        side: u8,
        amount_in: u64,
        min_amount_out: u64,
    ) -> solana_program::instruction::Instruction {
        ix::swap(
            &self.program_id,
            &self.strategy_key,
            &self.mm_key,
            &self.quotes_key,
            &self.base_mint,
            &self.quote_mint,
            taker,
            taker_ata_base,
            taker_ata_quote,
            &self.base_token_program,
            &self.quote_token_program,
            amount_in,
            min_amount_out,
            side,
        )
    }

    /// Override the Quay program id baked in by `single_program_amm!`
    /// (mainnet by default). Routers targeting devnet/testnet — where the
    /// deploy key differs — call this on the `Amm` returned by
    /// `from_keyed_account` if their `KeyedAccount.account.owner` couldn't
    /// be set to the live id ahead of time.
    ///
    /// The four PDAs (`mm_key`, `global_config_key`, `vault_base_key`,
    /// `vault_quote_key`) are re-derived against the new program id —
    /// otherwise the next `update()` would fetch accounts under the wrong
    /// program graph and `try_from_account` would fail on the
    /// discriminator.
    ///
    /// Fails if `strategy_data` no longer decodes (which only happens if
    /// `from_keyed_account` succeeded against a now-corrupt buffer — not
    /// reachable in practice, but plumbed as `Result` to keep the
    /// invariant explicit).
    pub fn with_program_id(mut self, program_id: Pubkey) -> JupiterResult<Self> {
        // Source of truth for `strategy_owner` is the cached strategy
        // header. We could carry the owner as a field instead, but
        // re-decoding here keeps the struct flat and the rederive
        // self-contained.
        let strategy = StrategyHeader::try_from_account(&self.strategy_data)
            .map_err(|e| anyhow!("with_program_id: re-decode StrategyHeader: {e}"))?;
        let strategy_owner = Pubkey::new_from_array(strategy.owner);
        let (mm_key, _) = pda::market_maker_pda(&program_id, &strategy_owner);
        let (global_config_key, _) = pda::global_config_pda(&program_id);
        let (vault_base_key, _) = pda::vault_pda(&program_id, &mm_key, &self.base_mint);
        let (vault_quote_key, _) = pda::vault_pda(&program_id, &mm_key, &self.quote_mint);

        self.program_id = program_id;
        self.mm_key = mm_key;
        self.global_config_key = global_config_key;
        self.vault_base_key = vault_base_key;
        self.vault_quote_key = vault_quote_key;
        Ok(self)
    }

    /// Routers should set this on every Quay swap tx (architecture KPI:
    /// 32 KiB / 1 page → ~16K CU saved per swap vs the 64 MiB default).
    pub const RECOMMENDED_LOADED_ACCOUNTS_DATA_SIZE_LIMIT: u32 =
        SWAP_LOADED_ACCOUNTS_DATA_SIZE_LIMIT;
}

impl Amm for QuayAmm {
    fn from_keyed_account(
        keyed_account: &KeyedAccount,
        amm_context: &AmmContext,
    ) -> JupiterResult<Self> {
        let strategy = StrategyHeader::try_from_account(&keyed_account.account.data)
            .map_err(|e| anyhow!("decode StrategyHeader: {e}"))?;

        let program_id = keyed_account.account.owner;
        let base_mint = Pubkey::new_from_array(strategy.base_mint);
        let quote_mint = Pubkey::new_from_array(strategy.quote_mint);
        let strategy_owner = Pubkey::new_from_array(strategy.owner);
        let quotes_key = Pubkey::new_from_array(strategy.quotes_account);
        let (mm_key, _) = pda::market_maker_pda(&program_id, &strategy_owner);
        let (global_config_key, _) = pda::global_config_pda(&program_id);
        let (vault_base_key, _) = pda::vault_pda(&program_id, &mm_key, &base_mint);
        let (vault_quote_key, _) = pda::vault_pda(&program_id, &mm_key, &quote_mint);

        Ok(Self {
            program_id,
            strategy_key: keyed_account.key,
            strategy_data: keyed_account.account.data.clone(),
            mm_key,
            mm_data: Vec::new(),
            quotes_key,
            quotes_data: Vec::new(),
            global_config_key,
            global_config_data: Vec::new(),
            base_mint,
            quote_mint,
            // Default to SPL Token; `update()` corrects this from each mint
            // account's `owner` before the first quote.
            base_token_program: SPL_TOKEN_PROGRAM_ID,
            quote_token_program: SPL_TOKEN_PROGRAM_ID,
            base_decimals: 0,
            quote_decimals: 0,
            vault_base_key,
            vault_quote_key,
            clock: amm_context.clock_ref.clone(),
            routing_flags: strategy.routing_flags,
            strategy_frozen: strategy.frozen,
            strategy_frozen_admin: strategy.frozen_admin,
            // Default to 1 (active-halt) until update fills them — refuses
            // quotes during the first-slot warmup, which is safer than
            // serving routes against half-decoded state.
            mm_frozen: 1,
            mm_frozen_admin: 1,
            mm_halted_admin: 1,
            cfg_swap_halted: 1,
            cfg_protocol_halted: 1,
            // Default false; `update()` flips these on if either mint
            // carries a `TransferFeeConfig` extension. Inverted polarity
            // (vs the halt bytes) because the safe default is "no fee".
            base_has_transfer_fee: false,
            quote_has_transfer_fee: false,
        })
    }

    fn label(&self) -> String {
        "Quay".to_string()
    }

    fn program_id(&self) -> Pubkey {
        self.program_id
    }

    fn key(&self) -> Pubkey {
        self.strategy_key
    }

    fn get_reserve_mints(&self) -> Vec<Pubkey> {
        vec![self.base_mint, self.quote_mint]
    }

    fn get_accounts_to_update(&self) -> Vec<Pubkey> {
        vec![
            self.strategy_key,
            self.mm_key,
            self.quotes_key,
            self.global_config_key,
            self.vault_base_key,
            self.vault_quote_key,
            self.base_mint,
            self.quote_mint,
        ]
    }

    fn update(&mut self, account_map: &AccountMap) -> JupiterResult<()> {
        // Strategy — re-decode to refresh the frozen flags.
        let strategy_data = try_get_account_data(account_map, &self.strategy_key)?;
        self.strategy_data = strategy_data.to_vec();
        let strategy = StrategyHeader::try_from_account(&self.strategy_data)
            .map_err(|e| anyhow!("decode StrategyHeader on update: {e}"))?;
        self.routing_flags = strategy.routing_flags;
        self.strategy_frozen = strategy.frozen;
        self.strategy_frozen_admin = strategy.frozen_admin;

        // MarketMaker — refresh halt flags.
        let mm_data = try_get_account_data(account_map, &self.mm_key)?;
        self.mm_data = mm_data.to_vec();
        let mm = MarketMakerHeader::try_from_account(&self.mm_data)
            .map_err(|e| anyhow!("decode MarketMakerHeader on update: {e}"))?;
        self.mm_frozen = mm.frozen;
        self.mm_frozen_admin = mm.frozen_admin;
        self.mm_halted_admin = mm.halted_admin;

        // Quotes — full data (the simulator reads the blob).
        let quotes_data = try_get_account_data(account_map, &self.quotes_key)?;
        self.quotes_data = quotes_data.to_vec();

        // GlobalConfig — halt flags.
        let cfg_data = try_get_account_data(account_map, &self.global_config_key)?;
        self.global_config_data = cfg_data.to_vec();
        let cfg = GlobalConfig::try_from_account(&self.global_config_data)
            .map_err(|e| anyhow!("decode GlobalConfig on update: {e}"))?;
        self.cfg_swap_halted = cfg.swap_halted;
        self.cfg_protocol_halted = cfg.protocol_halted;

        // Vaults are no longer priced (the VM dropped `LoadVault*`), but the
        // `swap` ix still takes them on-chain, so we keep them in
        // `get_accounts_to_update` and only verify they're present.
        try_get_account_data(account_map, &self.vault_base_key)?;
        try_get_account_data(account_map, &self.vault_quote_key)?;

        // Mints — owner = token program (mixed-program correctness), and
        // `data[44]` = decimals (both SPL Token and Token-2022 store it there).
        // `mint_has_transfer_fee` walks the Token-2022 TLV; SPL Token mints
        // short-circuit on the program-id check.
        if let Some(acc) = account_map.get(&self.base_mint) {
            self.base_token_program = acc.owner;
            if acc.data.len() > 44 {
                self.base_decimals = acc.data[44];
            }
            self.base_has_transfer_fee = mint_has_transfer_fee(&acc.owner, &acc.data);
        }
        if let Some(acc) = account_map.get(&self.quote_mint) {
            self.quote_token_program = acc.owner;
            if acc.data.len() > 44 {
                self.quote_decimals = acc.data[44];
            }
            self.quote_has_transfer_fee = mint_has_transfer_fee(&acc.owner, &acc.data);
        }

        Ok(())
    }

    fn quote(&self, quote_params: &QuoteParams) -> JupiterResult<Quote> {
        if quote_params.swap_mode == SwapMode::ExactOut {
            // Quay's swap ix is exact-in only and the DSL is a forward
            // pricer (no inverse). Jupiter degrades gracefully when an
            // `Amm` rejects ExactOut.
            return Err(anyhow!("Quay does not support ExactOut quotes"));
        }

        let side = if quote_params.input_mint == self.base_mint
            && quote_params.output_mint == self.quote_mint
        {
            SIDE_SELL_BASE
        } else if quote_params.input_mint == self.quote_mint
            && quote_params.output_mint == self.base_mint
        {
            SIDE_BUY_BASE
        } else {
            return Err(anyhow!(
                "input/output mints don't match this strategy's pair (base={} quote={})",
                self.base_mint,
                self.quote_mint
            ));
        };

        // Read current slot + unix timestamp from the ClockRef the resolver
        // updates each slot. Curves that use `LoadNowSlot` or rely on the
        // staleness gate get the same numbers a real swap would see.
        let current_slot = self.clock.slot.load(Ordering::Relaxed);
        let current_unix_sec = self.clock.unix_timestamp.load(Ordering::Relaxed);

        // Price into a stack scratch buffer so `quote()` performs no heap
        // allocation, for every curve — stateful or not. `MAX_USERSPACE_LEN`
        // (16 KiB) is the program's hard cap on userspace, so it fits any
        // strategy.
        let mut scratch = [0u8; MAX_USERSPACE_LEN as usize];
        let sim = simulate_swap_in(
            SwapSimulationInputs {
                strategy_data: &self.strategy_data,
                market_maker_data: &self.mm_data,
                quotes_data: &self.quotes_data,
                global_config_data: &self.global_config_data,
                current_slot,
                current_unix_sec,
                side,
                amount_in: quote_params.amount,
                min_amount_out: 0,
                base_decimals: self.base_decimals,
                quote_decimals: self.quote_decimals,
            },
            &mut scratch,
        )
        .map_err(|e| anyhow!("simulate_swap: {e}"))?;

        // Protocol cut is skimmed from the INPUT (DSL-v1 fee model): the fee
        // accrues to the IN-side asset, denominated in IN-token atoms.
        // SELL_BASE (0) → IN is base; BUY_BASE (1) → IN is quote.
        let fee_mint = if side == SIDE_SELL_BASE {
            self.base_mint
        } else {
            self.quote_mint
        };
        Ok(Quote {
            in_amount: quote_params.amount,
            out_amount: sim.out_to_taker,
            fee_amount: sim.protocol_cut,
            fee_mint,
            ..Quote::default()
        })
    }

    fn get_swap_and_account_metas(
        &self,
        swap_params: &SwapParams,
    ) -> JupiterResult<SwapAndAccountMetas> {
        let SwapParams {
            source_mint,
            destination_mint,
            source_token_account,
            destination_token_account,
            token_transfer_authority,
            ..
        } = swap_params;

        let (taker_ata_base, taker_ata_quote, side) = if *source_mint == self.base_mint
            && *destination_mint == self.quote_mint
        {
            (*source_token_account, *destination_token_account, SIDE_SELL_BASE)
        } else if *source_mint == self.quote_mint && *destination_mint == self.base_mint {
            (*destination_token_account, *source_token_account, SIDE_BUY_BASE)
        } else {
            return Err(anyhow!(
                "neither source nor destination mint matches this strategy's pair"
            ));
        };

        Ok(SwapAndAccountMetas {
            // PLACEHOLDER — Jupiter forks this SDK and patches in
            // `Swap::Quay`. Until then `Swap::TokenSwap` keeps the type
            // checker happy; the `account_metas` below are the load-bearing
            // payload and are correct in either variant.
            swap: Swap::TokenSwap,
            account_metas: self.swap_account_metas(
                token_transfer_authority,
                &taker_ata_base,
                &taker_ata_quote,
                side,
            ),
        })
    }

    fn get_accounts_len(&self) -> usize {
        // Mirror `quay_sdk::ix::swap`'s `push_token_programs` dedup so
        // Jupiter's transaction sizing matches the actual meta count it
        // gets back from `get_swap_and_account_metas`. During warmup both
        // fields default to `SPL_TOKEN_PROGRAM_ID`, so we report the
        // same-program count until `update()` flips one of them.
        if self.base_token_program == self.quote_token_program {
            SWAP_ACCOUNTS_LEN_SAME_PROGRAM
        } else {
            SWAP_ACCOUNTS_LEN_MIXED_PROGRAM
        }
    }

    fn is_active(&self) -> bool {
        // The MM must opt this strategy into Jupiter via
        // `routing_flags & ROUTE_JUPITER` (stored on-chain, enforced off-chain).
        // Then mirror the on-chain `execute_swap` halt-gate set: every protocol
        // / MM / strategy halt or freeze flag must be clear for swaps to land.
        // Plus refuse Token-2022 transfer-fee mints — the on-chain swap prices
        // `amount_in` gross and would settle short of the curve's quoted output.
        self.routing_flags & ROUTE_JUPITER != 0
            && self.cfg_swap_halted == 0
            && self.cfg_protocol_halted == 0
            && self.strategy_frozen == 0
            && self.strategy_frozen_admin == 0
            && self.mm_frozen == 0
            && self.mm_frozen_admin == 0
            && self.mm_halted_admin == 0
            && !self.base_has_transfer_fee
            && !self.quote_has_transfer_fee
    }

    fn unidirectional(&self) -> bool {
        false
    }

    fn supports_exact_out(&self) -> bool {
        false
    }

    fn clone_amm(&self) -> Box<dyn Amm + Send + Sync> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: panic on assertion failure is the desired behavior"
)]
mod tests {
    use super::*;

    /// Build a `QuayAmm` with every halt / freeze byte cleared and no
    /// transfer-fee mints — `is_active()` should return true. Per-flag
    /// tests below flip one bit at a time and assert the gate flips.
    /// Empty data buffers are fine — `is_active()` doesn't read them.
    fn all_active_amm() -> QuayAmm {
        let zero = Pubkey::new_from_array([0; 32]);
        QuayAmm {
            program_id: zero,
            strategy_key: zero,
            strategy_data: Vec::new(),
            mm_key: zero,
            mm_data: Vec::new(),
            quotes_key: zero,
            quotes_data: Vec::new(),
            global_config_key: zero,
            global_config_data: Vec::new(),
            base_mint: zero,
            quote_mint: zero,
            base_token_program: SPL_TOKEN_PROGRAM_ID,
            quote_token_program: SPL_TOKEN_PROGRAM_ID,
            base_decimals: 0,
            quote_decimals: 0,
            vault_base_key: zero,
            vault_quote_key: zero,
            clock: jupiter_amm_interface::ClockRef::default(),
            routing_flags: ROUTE_JUPITER,
            strategy_frozen: 0,
            strategy_frozen_admin: 0,
            mm_frozen: 0,
            mm_frozen_admin: 0,
            mm_halted_admin: 0,
            cfg_swap_halted: 0,
            cfg_protocol_halted: 0,
            base_has_transfer_fee: false,
            quote_has_transfer_fee: false,
        }
    }

    #[test]
    fn is_active_true_when_all_flags_clear() {
        assert!(all_active_amm().is_active());
    }

    /// One row of the halt-gate property table: a human-readable name and
    /// a setter that flips a single flag on a fresh `QuayAmm`.
    type HaltCase = (&'static str, fn(&mut QuayAmm));

    /// Property: setting any one halt / freeze byte to non-zero must flip
    /// `is_active()` to false. Drives a closure over every field name we
    /// care about so that adding a new flag forces a code change here.
    #[test]
    fn is_active_false_when_any_single_halt_set() {
        let cases: &[HaltCase] = &[
            ("cfg_swap_halted", |a| a.cfg_swap_halted = 1),
            ("cfg_protocol_halted", |a| a.cfg_protocol_halted = 1),
            ("strategy_frozen", |a| a.strategy_frozen = 1),
            ("strategy_frozen_admin", |a| a.strategy_frozen_admin = 1),
            ("mm_frozen", |a| a.mm_frozen = 1),
            ("mm_frozen_admin", |a| a.mm_frozen_admin = 1),
            ("mm_halted_admin", |a| a.mm_halted_admin = 1),
        ];
        for (name, set) in cases {
            let mut amm = all_active_amm();
            set(&mut amm);
            assert!(!amm.is_active(), "flag {name}=1 should disable is_active");
        }
    }

    #[test]
    fn is_active_false_when_transfer_fee_present() {
        let mut amm = all_active_amm();
        amm.base_has_transfer_fee = true;
        assert!(!amm.is_active(), "base transfer-fee mint should disable is_active");
        let mut amm = all_active_amm();
        amm.quote_has_transfer_fee = true;
        assert!(!amm.is_active(), "quote transfer-fee mint should disable is_active");
    }

    #[test]
    fn is_active_false_when_not_jupiter_routed() {
        // The MM must opt a strategy into Jupiter via `routing_flags & ROUTE_JUPITER`.
        let mut amm = all_active_amm();
        amm.routing_flags = 0;
        assert!(!amm.is_active(), "no routing bits set must disable is_active");
        amm.routing_flags = 0x01; // ROUTE_TITAN only — not Jupiter.
        assert!(!amm.is_active(), "another router's bit alone must not enable Jupiter");
    }

    #[test]
    fn get_accounts_len_branches_on_mixed_program() {
        let mut amm = all_active_amm();
        assert_eq!(amm.get_accounts_len(), SWAP_ACCOUNTS_LEN_SAME_PROGRAM);
        amm.quote_token_program = TOKEN_2022_PROGRAM_ID;
        assert_eq!(amm.get_accounts_len(), SWAP_ACCOUNTS_LEN_MIXED_PROGRAM);
    }

    /// Build a Token-2022 mint blob with a single TLV extension at
    /// `TOKEN_2022_TLV_OFFSET`. Length-only — we don't care about the value
    /// for the discriminator-walk test.
    fn token2022_mint_with(ext_type: u16, ext_value_len: usize) -> Vec<u8> {
        let mut data = vec![0u8; TOKEN_2022_TLV_OFFSET + 4 + ext_value_len];
        data[TOKEN_2022_TLV_OFFSET..TOKEN_2022_TLV_OFFSET + 2]
            .copy_from_slice(&ext_type.to_le_bytes());
        data[TOKEN_2022_TLV_OFFSET + 2..TOKEN_2022_TLV_OFFSET + 4]
            .copy_from_slice(&(ext_value_len as u16).to_le_bytes());
        data
    }

    #[test]
    fn transfer_fee_detected_on_token_2022() {
        let data = token2022_mint_with(EXT_TRANSFER_FEE_CONFIG, 108);
        assert!(mint_has_transfer_fee(&TOKEN_2022_PROGRAM_ID, &data));
    }

    #[test]
    fn other_extensions_do_not_trigger() {
        // ExtensionType::MintCloseAuthority = 3; skipped to expose the
        // `length` advance — if we miscalculate the TLV stride, the next
        // iteration may land inside the previous extension's value and
        // randomly match `EXT_TRANSFER_FEE_CONFIG`.
        let data = token2022_mint_with(3, 32);
        assert!(!mint_has_transfer_fee(&TOKEN_2022_PROGRAM_ID, &data));
    }

    #[test]
    fn spl_token_short_circuits() {
        // 82-byte base mint with no padding — even if the bytes happened to
        // spell `0x0001` somewhere, the program-id check must keep us out
        // of the TLV walker.
        let data = vec![0xFFu8; 82];
        assert!(!mint_has_transfer_fee(&SPL_TOKEN_PROGRAM_ID, &data));
    }

    #[test]
    fn malformed_tlv_returns_false() {
        // Half-written extension header (only 2 bytes after the offset).
        let mut data = vec![0u8; TOKEN_2022_TLV_OFFSET + 2];
        data[TOKEN_2022_TLV_OFFSET..TOKEN_2022_TLV_OFFSET + 2]
            .copy_from_slice(&EXT_TRANSFER_FEE_CONFIG.to_le_bytes());
        assert!(!mint_has_transfer_fee(&TOKEN_2022_PROGRAM_ID, &data));
    }
}
