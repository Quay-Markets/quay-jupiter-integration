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

One `QuayAmm` is one Strategy: a pricing curve bound to a single
`(base_mint, quote_mint)` pair on one MarketMaker. Jupiter's resolver
holds one instance per active strategy.

The crate exposes exactly the `Amm` trait surface plus the
`QUAY_PROGRAM_ID` constant. Anything else an integrator might need
(instruction builders, PDAs, the swap simulator) lives in `quay-sdk`.

## Implementation notes

- `from_keyed_account` decodes the `StrategyHeader`, derives the
  MarketMaker, GlobalConfig and vault PDAs, and reads the program id off
  the account owner, so devnet/testnet deploys work without special
  handling.
- Heavy decoding lives in `update`; `quote` only re-runs the VM
  (`quay_sdk::simulate::simulate_swap_in`) on a stack buffer, so quoting
  never allocates.
- `update` commits atomically: a failed update leaves the previous state
  untouched. Freshly constructed venues carry halted defaults, so a
  strategy is never quotable before its first successful update.
- `update` resolves each mint's owning token program from the account
  owner, so mixed SPL / Token-2022 markets settle correctly. A missing
  or truncated account fails the update outright.
- `is_active()` requires the strategy's Jupiter routing bit
  (`routing_flags & ROUTE_JUPITER`, stored on-chain, enforced by each
  adapter) and mirrors the on-chain halt gates: `cfg.swap_halted`,
  `cfg.protocol_halted`, `strategy.{frozen,frozen_admin}`,
  `mm.{frozen,frozen_admin,halted_admin}` all clear. There is no
  transfer-fee gate because the program rejects transfer-affecting
  Token-2022 extensions at asset registration.
- `get_accounts_len()` returns the exact account count of the on-chain
  swap ix: 13 for a same-token-program market, 14 for a mixed
  SPL / Token-2022 market.
- ExactOut is unsupported: the swap ix is exact-in only and the DSL has
  no inverse pricer.

## `jupiter-amm-interface` is pinned to `=0.6.0`

`0.6.0` is the version Jupiter's own `Cargo.lock` resolves. `0.6.1` is on
crates.io but its dep ranges let Cargo independently pick
`solana-account-decoder 3.x` and `solana-account 4.x` for the jupiter
sub-graph while our `solana-program = "2.1"` keeps the rest at 2.x. The
two majors coexist inside `jupiter-amm-interface` itself and fail to
compile. Pin held at `=0.6.0` until the upstream range is tightened.

## `Swap` enum placeholder

`SwapAndAccountMetas.swap` returns `Swap::TokenSwap` as a placeholder.
The 0.6.x `Swap` enum is closed and there's no `Swap::Quay` variant yet;
Jupiter adds one when they merge the integration and patches the single
return value. The `account_metas` are what actually gets consumed and
are correct in either case. Callers building swap instructions outside
the `Amm` trait can use `quay_sdk::ix::swap()` directly.
