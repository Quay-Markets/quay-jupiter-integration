//! Jupiter `Amm` adapter for the Quay program.
//!
//! Targets `jupiter-amm-interface = "0.6"`. Lives in its own crate so
//! `quay-sdk` doesn't inherit the Jupiter dependency tree.
//!
//! One `QuayAmm` is one Strategy: a pricing curve bound to a single
//! (base_mint, quote_mint) pair on one MarketMaker. Jupiter's resolver
//! constructs one instance per active strategy and polls the accounts
//! from `get_accounts_to_update`: the strategy, its market maker, the
//! bound quotes account, the global config, both vaults (the swap ix
//! takes them, the VM doesn't price off their balances) and both mints
//! (decimals and owning token program). The clock is not tracked;
//! `AmmContext.clock_ref` already carries the current slot and unix time.
//!
//! Heavy decoding happens in `update`; `quote` only re-runs the pricing
//! VM against the cached account data.
//!
//! The 0.6 `Swap` enum is closed, so `get_swap_and_account_metas`
//! returns `Swap::TokenSwap` until Jupiter adds a `Swap::Quay` variant
//! on their side. The account metas are the part that matters and are
//! correct either way.

use std::sync::atomic::Ordering;

use anyhow::{anyhow, Result as JupiterResult};
use jupiter_amm_interface::{
    single_program_amm, try_get_account_data, AccountMap, Amm, AmmContext, KeyedAccount, Quote,
    QuoteParams, SingleProgramAmm, Swap, SwapAndAccountMetas, SwapMode, SwapParams,
};
use solana_program::pubkey::Pubkey;

use quay_sdk::consts::{MAX_USERSPACE_LEN, ROUTE_JUPITER, SIDE_BUY_BASE, SIDE_SELL_BASE};
use quay_sdk::ix;
use quay_sdk::pda::{self, SPL_TOKEN_PROGRAM_ID};
use quay_sdk::simulate::{simulate_swap_in, SwapSimulationInputs};
use quay_sdk::state::{GlobalConfig, MarketMakerHeader, StrategyHeader};
use quay_sdk::TxContext;

/// Account count of the on-chain swap ix when both mints share one token
/// program: 11 positional accounts (cfg, strategy, mm, quotes, two vaults,
/// two taker ATAs, taker, two mints), the Instructions sysvar, and the
/// token program.
const SWAP_ACCOUNTS_LEN_SAME_PROGRAM: usize = 13;

/// A mixed SPL / Token-2022 market appends the second token program.
const SWAP_ACCOUNTS_LEN_MIXED_PROGRAM: usize = 14;

/// Token program and decimals of a mint, from the account map. A missing
/// or truncated mint is a hard error: quoting with stale decimals or a
/// wrong token program would misprice the venue.
///
/// No Token-2022 extension scanning here. The program rejects everything
/// but metadata extensions at asset registration, and a mint's extension
/// set is fixed at creation, so transfer-fee mints can't appear in a Quay
/// market.
fn read_mint(account_map: &AccountMap, mint: &Pubkey) -> JupiterResult<(Pubkey, u8)> {
    let acc = account_map
        .get(mint)
        .ok_or_else(|| anyhow!("update: mint account {mint} missing from account map"))?;
    // Both SPL Token and Token-2022 store decimals at offset 44.
    let decimals = *acc.data.get(44).ok_or_else(|| {
        anyhow!(
            "update: mint {mint} data too short for a Mint ({} bytes)",
            acc.data.len()
        )
    })?;
    Ok((acc.owner, decimals))
}

/// Quay's mainnet program id (placeholder until deployment locks it in).
///
/// `single_program_amm!` needs a const program id for label resolution.
/// `from_keyed_account` takes the live id from the strategy account's
/// owner, so devnet/testnet deploys resolve correctly on their own.
pub const QUAY_PROGRAM_ID: Pubkey =
    solana_program::pubkey!("QUayE6nexQWYNZAEqfN8FxoNwQDSu3CAzT2qq9J1ArG");

/// Jupiter's mainnet aggregator program id — the top-level program a routed
/// swap executes under. `quote()` simulates with this entrypoint at CPI
/// depth 2 so context-gated curves (`EntrypointIs` / `LoadIxDepth`) price
/// the branch a Jupiter route actually takes.
pub const JUPITER_ROUTER_ID: Pubkey =
    solana_program::pubkey!("JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4");

#[derive(Clone)]
pub struct QuayAmm {
    program_id: Pubkey,
    /// Strategy account pubkey, the Amm's primary key.
    strategy_key: Pubkey,
    strategy_data: Vec<u8>,
    /// MarketMaker PDA, derived from `strategy.owner`.
    mm_key: Pubkey,
    mm_data: Vec<u8>,
    /// `strategy.quotes_account`.
    quotes_key: Pubkey,
    quotes_data: Vec<u8>,
    global_config_key: Pubkey,
    global_config_data: Vec<u8>,
    base_mint: Pubkey,
    quote_mint: Pubkey,
    /// Owner of each mint, resolved in `update`. Tracked per side so
    /// mixed SPL / Token-2022 markets build the right transfer CPIs.
    base_token_program: Pubkey,
    quote_token_program: Pubkey,
    /// Mint decimals, fed to the swap simulator.
    base_decimals: u8,
    quote_decimals: u8,
    /// Vault PDAs. The VM doesn't price off vault balances, but the swap
    /// ix takes them, so they stay in `get_accounts_to_update`.
    vault_base_key: Pubkey,
    vault_quote_key: Pubkey,
    /// `AmmContext.clock_ref`, read at quote time.
    clock: jupiter_amm_interface::ClockRef,
    /// `StrategyHeader.routing_flags`. The venue only surfaces when the
    /// `ROUTE_JUPITER` bit is set.
    routing_flags: u8,
    /// Halt and freeze bytes mirrored from the on-chain swap gates.
    /// Seeded to 1 (halted) at construction so the venue stays inactive
    /// until the first successful `update`.
    strategy_frozen: u8,
    strategy_frozen_admin: u8,
    mm_frozen: u8,
    mm_frozen_admin: u8,
    mm_halted_admin: u8,
    cfg_swap_halted: u8,
    cfg_protocol_halted: u8,
}

// Registers the program-id-to-"Quay" label mapping Jupiter uses for
// dispatch.
single_program_amm!(QuayAmm, QUAY_PROGRAM_ID, "Quay");

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
            // Corrected from the mint owners on the first update.
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
            // Halted until the first successful update; see the field docs.
            mm_frozen: 1,
            mm_frozen_admin: 1,
            mm_halted_admin: 1,
            cfg_swap_halted: 1,
            cfg_protocol_halted: 1,
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
        let strategy_data = try_get_account_data(account_map, &self.strategy_key)?.to_vec();
        let strategy = StrategyHeader::try_from_account(&strategy_data)
            .map_err(|e| anyhow!("decode StrategyHeader on update: {e}"))?;
        let routing_flags = strategy.routing_flags;
        let strategy_frozen = strategy.frozen;
        let strategy_frozen_admin = strategy.frozen_admin;

        let mm_data = try_get_account_data(account_map, &self.mm_key)?.to_vec();
        let mm = MarketMakerHeader::try_from_account(&mm_data)
            .map_err(|e| anyhow!("decode MarketMakerHeader on update: {e}"))?;
        let mm_frozen = mm.frozen;
        let mm_frozen_admin = mm.frozen_admin;
        let mm_halted_admin = mm.halted_admin;

        let quotes_data = try_get_account_data(account_map, &self.quotes_key)?.to_vec();

        let cfg_data = try_get_account_data(account_map, &self.global_config_key)?.to_vec();
        let cfg = GlobalConfig::try_from_account(&cfg_data)
            .map_err(|e| anyhow!("decode GlobalConfig on update: {e}"))?;
        let cfg_swap_halted = cfg.swap_halted;
        let cfg_protocol_halted = cfg.protocol_halted;

        // Vaults aren't priced; just check they were supplied.
        try_get_account_data(account_map, &self.vault_base_key)?;
        try_get_account_data(account_map, &self.vault_quote_key)?;

        let (base_token_program, base_decimals) = read_mint(account_map, &self.base_mint)?;
        let (quote_token_program, quote_decimals) = read_mint(account_map, &self.quote_mint)?;

        self.strategy_data = strategy_data;
        self.routing_flags = routing_flags;
        self.strategy_frozen = strategy_frozen;
        self.strategy_frozen_admin = strategy_frozen_admin;
        self.mm_data = mm_data;
        self.mm_frozen = mm_frozen;
        self.mm_frozen_admin = mm_frozen_admin;
        self.mm_halted_admin = mm_halted_admin;
        self.quotes_data = quotes_data;
        self.global_config_data = cfg_data;
        self.cfg_swap_halted = cfg_swap_halted;
        self.cfg_protocol_halted = cfg_protocol_halted;
        self.base_token_program = base_token_program;
        self.base_decimals = base_decimals;
        self.quote_token_program = quote_token_program;
        self.quote_decimals = quote_decimals;

        Ok(())
    }

    fn quote(&self, quote_params: &QuoteParams) -> JupiterResult<Quote> {
        if quote_params.swap_mode == SwapMode::ExactOut {
            // The swap ix is exact-in only and the DSL has no inverse.
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

        // Same clock a real swap would see; the resolver keeps these
        // atomics fresh every slot.
        let current_slot = self.clock.slot.load(Ordering::Relaxed);
        let current_unix_sec = self.clock.unix_timestamp.load(Ordering::Relaxed);

        // Simulate calling by Jupiter router (top-level ix -> CPI, depth 2).
        let tx = TxContext {
            ix_depth: 2,
            tx_flags: 0,
            entrypoint_program: JUPITER_ROUTER_ID.to_bytes(),
            signers: &[],
        };

        // Stack buffer sized to the program's userspace cap, so quoting
        // never allocates, stateful curves included.
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
                tx,
            },
            &mut scratch,
        )
        .map_err(|e| anyhow!("simulate_swap: {e}"))?;

        // The protocol fee is skimmed from the input, so it's denominated
        // in the input mint.
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

        // The taker's ATAs go to `ix::swap` in (base, quote) order
        // regardless of direction; the builder resolves in/out from `side`.
        let (taker_ata_base, taker_ata_quote, side) =
            if *source_mint == self.base_mint && *destination_mint == self.quote_mint {
                (
                    *source_token_account,
                    *destination_token_account,
                    SIDE_SELL_BASE,
                )
            } else if *source_mint == self.quote_mint && *destination_mint == self.base_mint {
                (
                    *destination_token_account,
                    *source_token_account,
                    SIDE_BUY_BASE,
                )
            } else {
                return Err(anyhow!(
                    "neither source nor destination mint matches this strategy's pair"
                ));
            };

        // Amounts are irrelevant here, only the metas are consumed.
        let metas = ix::swap(
            &self.program_id,
            &self.strategy_key,
            &self.mm_key,
            &self.quotes_key,
            &self.base_mint,
            &self.quote_mint,
            token_transfer_authority,
            &taker_ata_base,
            &taker_ata_quote,
            &self.base_token_program,
            &self.quote_token_program,
            0,
            0,
            side,
        )
        .accounts;

        Ok(SwapAndAccountMetas {
            // The 0.6 Swap enum has no Quay variant yet; Jupiter patches
            // this to `Swap::Quay` when they merge the integration. The
            // metas are correct either way.
            swap: Swap::TokenSwap,
            account_metas: metas,
        })
    }

    fn get_accounts_len(&self) -> usize {
        // Must match the meta count `ix::swap` produces, including its
        // trailing token-program dedup. Before the first update both
        // programs default to SPL Token, so this reports the same-program
        // count during warmup.
        if self.base_token_program == self.quote_token_program {
            SWAP_ACCOUNTS_LEN_SAME_PROGRAM
        } else {
            SWAP_ACCOUNTS_LEN_MIXED_PROGRAM
        }
    }

    fn is_active(&self) -> bool {
        self.routing_flags & ROUTE_JUPITER != 0
            && self.cfg_swap_halted == 0
            && self.cfg_protocol_halted == 0
            && self.strategy_frozen == 0
            && self.strategy_frozen_admin == 0
            && self.mm_frozen == 0
            && self.mm_frozen_admin == 0
            && self.mm_halted_admin == 0
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
    use quay_sdk::pda::TOKEN_2022_PROGRAM_ID;

    /// Every halt/freeze byte cleared and the Jupiter routing bit set, so
    /// `is_active()` returns true. Data buffers stay empty; `is_active`
    /// doesn't read them.
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
        }
    }

    #[test]
    fn is_active_true_when_all_flags_clear() {
        assert!(all_active_amm().is_active());
    }

    type HaltCase = (&'static str, fn(&mut QuayAmm));

    /// Any single halt/freeze byte set must disable the venue. One case
    /// per field, so adding a flag forces an update here.
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
    fn is_active_false_when_not_jupiter_routed() {
        let mut amm = all_active_amm();
        amm.routing_flags = 0;
        assert!(
            !amm.is_active(),
            "no routing bits set must disable is_active"
        );
        amm.routing_flags = 0x01; // ROUTE_TITAN only.
        assert!(
            !amm.is_active(),
            "another router's bit alone must not enable Jupiter"
        );
    }

    #[test]
    fn get_accounts_len_branches_on_mixed_program() {
        let mut amm = all_active_amm();
        assert_eq!(amm.get_accounts_len(), SWAP_ACCOUNTS_LEN_SAME_PROGRAM);
        amm.quote_token_program = TOKEN_2022_PROGRAM_ID;
        assert_eq!(amm.get_accounts_len(), SWAP_ACCOUNTS_LEN_MIXED_PROGRAM);
    }
}
