# Tradeagent Agent Brief

This file is the fastest entry point for coding agents working on this repo.

## What This Project Is

Tradeagent is a compact paper/live trading runtime for a 20-instrument universe. It combines:

- Kronos candle forecasts for entry ranking and exit forecasts.
- Round-trip risk filtering for new entries.
- Particle-style Kronos path tracking for exits.
- A Moscow trading-session guard that keeps the bot cash-flat near session close.
- A LightGBM meta-selector path that is still present and tested.
- Saved-candle runtime backtests for the May 1-14 dataset.

The current implemented entry mode in the main config is `kronos_rank`. Ideas discussed later, such as `kronos_single_top` or rebound-only entry, are not implemented unless you add them explicitly.

## Repository Map

- `arena_bot/types.py` defines config dataclasses and shared domain types.
- `arena_bot/config.py` loads YAML configs into `RuntimeConfig`.
- `arena_bot/runtime.py` is the main decision engine: session guard, exit pass, risk cap pass, entry pass, order planning, diagnostics.
- `arena_bot/kronos_provider.py` wraps the local Kronos model and cache, including `score()` and `forecast_paths()`.
- `arena_bot/runtime_backtest.py` replays saved candles through the full runtime and exports trades/account/ranking diagnostics.
- `arena_bot/storage.py` owns SQLite tables for state, decisions, cached Kronos forecasts, paper positions, and particle trackers.
- `arena_bot/market_data.py` provides saved-candle market data and computed market metrics.
- `arena_bot/meta_selector.py` keeps the LightGBM selector model path.
- `configs/universe_v1_may1_14.yaml` is the main current config.
- `data/universe-v1-may1-14/candles/` contains the compact saved candle dataset for the current run.
- `data/universe-v1-may1-14/models/lightgbm_meta/latest/` contains the latest LightGBM model.
- `kronos/model/` and `kronos/weights/` contain local Kronos code and LFS-tracked weights.
- `tests/test_runtime.py` is the best map of intended runtime behavior.

## Current Runtime Flow

`RuntimeEngine.run_once()` is the main cycle:

1. Load market snapshots, candles, metrics, positions, and account state.
2. Compute trading session state.
3. If session is pre-open/closed, block trading and Kronos calls.
4. If `force_flat`, close all positions and remove closed trackers.
5. Run exit logic first.
6. Apply risk cap reductions if enabled.
7. Build entry targets.
8. Plan and execute paper/live orders.
9. Sync Kronos particle trackers after position changes.
10. Persist a decision payload with diagnostics.

## Entry Logic

Current main config uses:

```yaml
trade_lifecycle:
  entry:
    mode: kronos_rank
```

`kronos_rank` behavior:

- Calls Kronos `score()` for the current universe.
- Converts `pred_return` sign into long/short side.
- Ranks candidates by configured metric, currently `abs_edge`.
- Applies top-N slot logic before post-selection cost filtering.
- Filters selected candidates by round-trip cost:
  - `slippage_one_way = spread_pct * risk.slippage_spread_multiplier`
  - `one_way_cost = spread_pct / 2 + commission_rate + slippage_one_way`
  - `round_trip_cost = 2 * one_way_cost`
  - `net_edge = abs(pred_return) - round_trip_cost`
- If `net_edge <= 0`, candidate reason is `cost_exceeds_pred_return` and no lower-ranked replacement is backfilled.

## Exit Logic

If `trade_lifecycle.exit.particle_enabled=true`, exits use particle tracking when possible:

- A tracker stores raw Kronos sample paths, weights, current step, confidence, planned exit, and extension count.
- Weights update by comparing actual candle movement to each path.
- Exit plan ranks future steps by `expected_net * probability_plus`.
- Dynamic session horizon clips particle forecasts to `force_flat_time`.
- If fewer than 3 future candles fit, the tracker uses only the next candle.
- If tracker creation/refresh is impossible, runtime can fall back to legacy edge exit.

Legacy edge exit still exists and compares directional `pred_return` against one-way exit cost.

## Trading Session Rules

Defaults are in `TradingSessionConfig`:

- Timezone: `Europe/Moscow`.
- Main session: `10:00-18:40`.
- Entries start at `11:00`.
- New entries stop at `17:40`.
- Force-flat runs at `18:30`.
- `flat_all_asset_classes=true`, so equities, futures, and crypto are all closed in the v1 MOEX-style session.

Kronos forecasts must fit inside session limits. Entry uses `kronos_cutoff`; particle exit clips to `force_flat_time`.

## Commands

Install and test:

```powershell
python -m pip install -e ".[test]"
git lfs pull
python -m pytest -q
```

Run the May 1-14 runtime backtest:

```powershell
python -m arena_bot.runtime_backtest `
  --config configs/universe_v1_may1_14.yaml `
  --from 2026-05-01T00:00:00 `
  --till 2026-05-14T23:59:59 `
  --run-dir data/universe-v1-may1-14/runtime-runs/manual
```

Run one paper decision:

```powershell
python -m arena_bot.cli run-once --config configs/universe_v1_may1_14.yaml --as-of 2026-05-04T11:00:00
```

Train LightGBM from accumulated SQLite history:

```powershell
python -m arena_bot.cli train-lightgbm --config configs/universe_v1_may1_14.yaml
```

## Working Rules For Agents

- Prefer reading `tests/test_runtime.py` before changing runtime behavior.
- Keep config paths relative to repo root.
- Do not commit runtime outputs, SQLite state, logs, or `.env` files.
- Do not remove Kronos LFS weights unless the config and docs are updated to fetch them another way.
- When touching `runtime.py`, run at least `python -m pytest tests/test_runtime.py -q`.
- When touching Kronos provider behavior, run `python -m pytest tests/test_kronos_provider.py -q`.
- For broad changes, run `python -m pytest -q`.
- Preserve existing behavior unless the user explicitly asks to change strategy.

## Known Strategy Direction

Recent analysis showed `kronos_rank` is a weak but nonzero directional signal, not a calibrated high-confidence ranker. Future work may move entry toward fewer trades, single-top allocation, or rebound-only setups, but those are design directions, not current behavior.
