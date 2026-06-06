# Tradeagent Agent Brief

This file is the fastest entry point for coding agents working on this repo.

## What This Project Is

Tradeagent is a compact paper/live trading runtime for a 20-instrument universe. It combines:

- Kronos candle forecasts for entry ranking.
- Round-trip risk filtering for new entries.
- Giveback trailing exits based on MFE inside the open position.
- Optional particle-style Kronos path tracking for exits.
- A Moscow trading-session guard that keeps the bot cash-flat near session close.
- A LightGBM meta-selector path that is still present and tested.
- Saved-candle runtime backtests for the May 1-14 dataset.

The main May 1-14 config and `configs/default.yaml` currently use the `kronos_single_top` baseline with giveback trailing exits. `kronos_rank` remains implemented and covered by tests, but is no longer the default config strategy.

## Repository Map

- `arena_bot/types.py` defines config dataclasses and shared domain types.
- `arena_bot/config.py` loads YAML configs into `RuntimeConfig`.
- `arena_bot/runtime.py` is the main decision engine: session guard, exit pass, risk cap pass, entry pass, order planning, diagnostics.
- `arena_bot/kronos_provider.py` wraps the local Kronos model and cache, including `score()` and `forecast_paths()`.
- `arena_bot/runtime_backtest.py` replays saved candles through the full runtime and exports trades/account/ranking diagnostics.
- `arena_bot/storage.py` owns SQLite tables for state, decisions, cached Kronos forecasts, paper positions, giveback state, and particle trackers.
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

The May 1-14 experiment config uses:

```yaml
trade_lifecycle:
  max_total_positions: 1
  exit:
    enabled: true
    giveback_enabled: true
    giveback_min_arm_profit: 0.012
    giveback_ratio: 0.60
    edge_enabled: false
    particle_enabled: false
  entry:
    mode: kronos_single_top
```

`kronos_single_top` behavior:

- Calls Kronos `score()` for the full selected universe each trading hour.
- Entry is gated to the configured `decision_interval_minutes`; minute runtime ticks between those slots are exit-only.
- Ranks candidates by `gross_pred_return = abs(pred_return)`.
- Takes rank-1 only; no lower-ranked replacement is backfilled.
- Computes spread, commission, slippage, round-trip cost, and `net_edge`.
- Enters only if `net_edge >= single_top_min_net_edge`, `gross_pred_return <= single_top_max_gross_pred_return`, and `rank1.gross_pred_return - rank2.gross_pred_return >= single_top_min_rank_gap`.
- The current experiment sets `single_top_min_rank_gap: 0.0004`, i.e. `0.04%`.
- Entries are also blocked unless `as_of + decision_interval_minutes` fits before both `kronos_cutoff` and `force_flat_time`, so there is a full next Kronos reassessment interval available.
- Targets near full capital using `cash_buffer_pct` and `sizing_safety_pct`.
- If the current position is the same `secid` and same side as rank-1, it holds without close/reopen.
- If rank-1 changes, flips side, or fails filters, rebalance orders close the old position; a passing new rank-1 is opened immediately.
- `entry_diagnostics.ranked_candidates` is always populated when Kronos entry is allowed, so `runtime_backtest.py` writes hourly `ranked_top.jsonl` rows.

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

The active single-top baseline uses giveback trailing:

- Each held position stores `entry_price`, side, `mfe_pct`, and `last_pnl_pct`.
- `run-live` calls the runtime every `rebalance.exit_interval_minutes`, currently every minute, so giveback updates independently of hourly entry reassessment.
- Existing positions without giveback state are bootstrapped from the current mark price with `mfe_pct=0`, so startup does not immediately close them.
- A position arms at `giveback_min_arm_profit: 0.012`.
- It closes when `(mfe_pct - current_pnl_pct) / mfe_pct >= giveback_ratio`, currently `0.60`.
- A giveback close blocks same-tick reopening of that `secid` in `kronos_single_top`.

If `trade_lifecycle.exit.particle_enabled=true`, exits use particle tracking when possible:

- A tracker stores raw Kronos sample paths, weights, current step, confidence, planned exit, and extension count.
- Weights update by comparing actual candle movement to each path.
- Exit plan ranks future steps by `expected_net * probability_plus`.
- Dynamic session horizon clips particle forecasts to `force_flat_time`.
- If fewer than 3 future candles fit, the tracker uses only the next candle.
- If tracker creation/refresh is impossible, runtime can fall back to legacy edge exit.

Legacy edge exit still exists and compares directional `pred_return` against one-way exit cost when `edge_enabled=true`.

## Trading Session Rules

Defaults are in `TradingSessionConfig`:

- Timezone: `Europe/Moscow`.
- Main session: `10:00-18:40`.
- Entries start at `11:00`.
- New entries stop at `17:40`.
- Force-flat runs at `18:30`.
- `flat_all_asset_classes=true`, so equities, futures, and crypto are all closed in the v1 MOEX-style session.

Kronos forecasts and the next entry reassessment interval must fit inside session limits. Entry uses `kronos_cutoff`; particle exit clips to `force_flat_time`.

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

Recent analysis showed `kronos_rank` is a weak but nonzero directional signal, not a calibrated high-confidence ranker. The current active experiment is fewer trades via `kronos_single_top`; a rebound-only detector is still a design direction, not implemented behavior.
