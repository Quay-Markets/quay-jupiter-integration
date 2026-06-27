# quay-aggregator-jupiter

Jupiter `Amm` adapter for the Quay program. Depends on the off-chain
`quay-sdk` (and its pricing VM `quay-vm`) from crates.io, so the repo
builds with no external Quay sources.

```toml
quay-aggregator-jupiter = { git = "<this repo>" }
```

```rust
use quay_aggregator_jupiter::QuayAmm;
use jupiter_amm_interface::{Amm, KeyedAccount, AmmContext};

let amm = QuayAmm::from_keyed_account(&keyed_strategy, &amm_context)?;
// Jupiter's resolver pulls these every slot via `update`:
//   amm.get_accounts_to_update() ==
//     [strategy, mm, quotes, global_config, vault_base, vault_quote,
//      base_mint, quote_mint]
```

Each `QuayAmm` instance represents **one Strategy** — a pricing curve
bound to a single `(base_mint, quote_mint)` pair on one MarketMaker.
Jupiter's resolver typically holds many (one per active strategy) and
shares quote distribution across them.

## What this crate does

- `single_program_amm!` macro registers Quay's program id with Jupiter's
  `AmmProgramIdToLabel` resolver.
- `from_keyed_account` decodes a `StrategyHeader` (`quay_sdk::state`),
  derives the MarketMaker PDA from `strategy.owner`, and caches the
  bound quotes account, both vault PDAs, the `GlobalConfig` PDA, and
  both mints.
- Heavy decoding lives in `update`; `quote` only re-runs the VM (calls
  through to `quay_sdk::simulate::simulate_swap`).
- `update` resolves each mint's owning token program from its `owner`
  field — mixed SPL / Token-2022 markets settle correctly.
- `AmmContext.clock_ref` is cached so curves using `LoadNowSlot` /
  `LoadNowUnixSec` / `LoadArenaTimestampSec` / `LoadLastUpdateSlot` see
  the same numbers a real swap would. (Protocol-side staleness is gone
  — curves now emit `REJECT StaleFeed` (`Custom(0xC002)`) when their
  per-strategy freshness policy fails.)
- `is_active()` mirrors the full on-chain halt-gate set:
  `!cfg.swap_halted` ∧ `!cfg.protocol_halted` ∧
  `!strategy.{frozen,frozen_admin}` ∧ `!mm.{frozen,frozen_admin,halted_admin}`,
  plus refuses Token-2022 mints carrying a `TransferFeeConfig` extension
  (the on-chain swap prices `amount_in` gross and would short-fill against
  the curve's quoted output). The route planner skips disabled strategies
  entirely.
- `get_accounts_len()` returns the exact account count the on-chain
  swap ix takes (12 for a same-token-program market).
- `swap_instruction(taker, ata_base, ata_quote, side, amount_in, min_out)`
  returns the full on-chain swap `Instruction` (data + metas). Useful for
  callers that go around the closed `jupiter-amm-interface 0.6` `Swap`
  enum and need the data side as well as the metas.
- `with_program_id(pid) -> Result<Self>` overrides the mainnet
  `QUAY_PROGRAM_ID` const for devnet / testnet integrators and re-derives
  the four cached PDAs (`mm_key`, `global_config_key`, `vault_*_key`)
  against the new program id. Routers that construct via
  `from_keyed_account` get the live program id off
  `KeyedAccount.account.owner` automatically and don't need this.

## `jupiter-amm-interface` is pinned to `=0.6.0`

`0.6.0` is the version Jupiter's own `Cargo.lock` resolves. `0.6.1` is on
crates.io but its dep ranges let Cargo independently pick
`solana-account-decoder 3.x` and `solana-account 4.x` for the jupiter
sub-graph while our `solana-program = "2.1"` keeps the rest at 2.x. The
two majors coexist inside `jupiter-amm-interface` itself and fail to
compile. Pin held at `=0.6.0` until the upstream range is tightened.

## `Swap` enum placeholder

`SwapAndAccountMetas.swap` returns `Swap::TokenSwap` as a placeholder.
The 0.6.x `Swap` enum is closed and there's no `Swap::Quay` variant yet.
When Jupiter merges the integration into Metis they add `Swap::Quay` and
patch the one return value. The `account_metas` we return are
load-bearing and correct in either case. Callers building swap
instructions outside the `Amm` trait can use
`QuayAmm::swap_account_metas()` together with `quay_sdk::ix::swap()`.
