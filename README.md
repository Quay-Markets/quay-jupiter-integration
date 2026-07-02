# quay-aggregator-jupiter

Jupiter `Amm` adapter for the Quay program, against
`jupiter-amm-interface = "=0.6.1"`.

A Quay **Strategy** holds the pricing logic for one
`(base_mint, quote_mint)` pair. This crate maps one Strategy to one
`QuayAmm` and quotes by running the same pricing VM off-chain
(`quay-sdk` / `quay-vm` from crates.io), so a quote is the amount the
program itself would pay in that slot. The public surface is the `Amm`
trait plus `QUAY_PROGRAM_ID`; instruction builders, PDAs and the
simulator live in `quay-sdk`.

```toml
[dependencies]
quay-aggregator-jupiter = { git = "<this repo>" }
```

## Usage

```rust
use jupiter_amm_interface::{Amm, AmmContext, FeeMode, KeyedAccount, QuoteParams, SwapMode};
use quay_aggregator_jupiter::QuayAmm;

// keyed_strategy: the Strategy account, e.g. from getProgramAccounts
// filtered on QUAY_PROGRAM_ID.
let mut amm = QuayAmm::from_keyed_account(&keyed_strategy, &amm_context)?;

// Every slot: fetch the tracked accounts and refresh.
let account_map = fetch(amm.get_accounts_to_update()); // 8 pubkeys
amm.update(&account_map)?;

if amm.is_active() {
    let quote = amm.quote(&QuoteParams {
        amount: 1_000_000, // input atoms
        input_mint: base_mint,
        output_mint: quote_mint,
        swap_mode: SwapMode::ExactIn,
        fee_mode: FeeMode::default(),
    })?;
    // quote.out_amount   — atoms the taker receives
    // quote.fee_amount   — protocol cut, in input-mint atoms
}
```

## Tracked accounts

`get_accounts_to_update` returns eight pubkeys:

| Account | Used for |
| --- | --- |
| Strategy | curve bytecode + userspace, fee bps, freeze flags, routing bits |
| MarketMaker | asset-table inventories, freeze/halt flags |
| Quotes | published price inputs + publish timestamp |
| GlobalConfig | protocol-wide halt flags |
| base / quote vault | swap-ix accounts only; balances are not priced |
| base / quote mint | decimals + owning token program |

## Quoting semantics

- Exact-in only. `supports_exact_out()` is false and `quote()` rejects
  `SwapMode::ExactOut`.
- `fee_amount` is the protocol fee, skimmed from the input before the
  curve prices the remainder; `fee_mint` is therefore the input mint.
- A size the curve refuses (side disabled, over inventory, stale inputs)
  comes back as `Err`, not a zero quote.
- Amounts are atoms on both sides.

## Activity gating

`is_active()` returns true only when:

- the strategy's `routing_flags` has the Jupiter bit set. The program
  stores this byte but does not enforce it; each aggregator adapter is
  expected to check its own bit, which is what makes routing opt-in per
  venue;
- every halt/freeze byte the on-chain swap checks is clear:
  `GlobalConfig.{swap_halted,protocol_halted}`,
  `Strategy.{frozen,frozen_admin}`,
  `MarketMaker.{frozen,frozen_admin,halted_admin}`.

## Swap instruction

`get_swap_and_account_metas` builds the metas via `quay_sdk::ix::swap`,
matching the on-chain account order:

```
0  global config
1  strategy            (writable)
2  market maker        (writable)
3  quotes
4  vault (in side)     (writable)
5  vault (out side)    (writable)
6  taker ATA (in)      (writable)
7  taker ATA (out)     (writable)
8  taker               (signer)
9  mint (in)
10 mint (out)
11 Instructions sysvar
12 token program       (+ second program at 13 for mixed markets)
```

`get_accounts_len()` reports 13, or 14 when base and quote mints are
owned by different token programs (SPL Token vs Token-2022).

`SwapAndAccountMetas.swap` is `Swap::TokenSwap` for now: the 0.6 `Swap`
enum is closed and has no Quay variant, so Jupiter patches this one
return value when merging the integration. Only the metas are consumed
downstream; they are correct in either case.

## Tests

```
cargo test
```

## Dependency notes

`jupiter-amm-interface 0.6.1` declares its modular Solana dependencies
as `>=2` with no upper bound, so a fresh resolve floats them to 3.x/4.x
while `solana-program = "2.1"` (required to match `quay-sdk`'s
re-exported types) keeps our side at 2.x — two majors of the same types,
which doesn't compile. `Cargo.toml` therefore declares
`solana-account`, `solana-account-decoder`, `solana-clock`,
`solana-instruction` and `solana-pubkey` directly with 2.x caps, forcing
the whole graph onto one 2.x line. Drop those extra dependencies once
upstream bounds its ranges.
