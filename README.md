# Tradeagent

Trading runtime built around Kronos forecasts, LightGBM selector metadata, risk filters, trading-session guards, and giveback trailing exits.

For coding agents, start with [AGENTS.md](AGENTS.md). It explains the current architecture, runtime flow, implemented strategy modes, commands, and safe working rules.

## What Is Implemented

- `kronos_rank` entry mode: Kronos ranks candidates, then the runtime applies strict round-trip cost filtering.
- `kronos_single_top` entry mode: hourly full-capital baseline that ranks the universe, trades only rank-1 if strict cost, gross-return, and top1/top2 gap filters pass, and blocks entries unless a full next Kronos reassessment interval remains.
- Giveback trailing exits: open positions track MFE on every runtime tick; `run-live` ticks every minute while entries remain hourly.
- Particle exit tracking: still implemented for configs that enable Kronos sample-path exits.
- Session-safe trading: venue-aware calendar sessions with MOEX force-flat scoped to MOEX instruments and crypto kept 24/7 when Binance status is `TRADING`.
- LightGBM meta-selector support: model loading/training code and latest compact model are kept in the repo.
- Saved-candle backtests: the current May 1-14 dataset is included for reproducible runtime checks.

## Layout

- `arena_bot/` - runtime, storage, risk/session logic, Kronos provider, backtest runner, CLI.
- `configs/` - paper/runtime YAML configs with relative paths.
- `kronos/model/` - Kronos model code used by `RealKronosSignalProvider`.
- `kronos/weights/` - local Kronos base model and tokenizer weights. Tensor files are tracked with Git LFS.
- `data/universe-v1-may1-14/candles/` - compact May 1-14 candle set.
- `data/universe-v1-may1-14/models/lightgbm_meta/` - latest LightGBM meta-selector model.
- `tests/` - regression tests for runtime, Kronos provider, storage, session logic, and LightGBM behavior.
- `AGENTS.md` - detailed repo guide for future coding agents.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
git lfs pull
```

Install the optional optimizer tools only when running offline Optuna studies:

```powershell
python -m pip install -e ".[test,optimize]"
```

## Verify

```powershell
python -m pytest -q
```

## Run Simulation / Backtest

Simulations use saved candles and the same runtime path as paper/live trading. Orders are executed in paper mode inside the backtest loop, cash/lots are updated locally, and outputs are written under `--run-dir`.

General command:

```powershell
python -m arena_bot.runtime_backtest `
  --config configs/universe_v1_may1_14.yaml `
  --from 2026-05-01T00:00:00 `
  --till 2026-05-14T23:59:59 `
  --run-dir data/universe-v1-may1-14/runtime-runs/manual
```

Current May 1-14 baseline:

```powershell
python -m arena_bot.runtime_backtest `
  --config configs/universe_v1_may1_14.yaml `
  --from 2026-05-01T00:00:00 `
  --till 2026-05-14T23:59:59 `
  --run-dir data/universe-v1-may1-14/runtime-runs/manual
```

For another window, use a config whose `market_data.saved_candles.directories` points at that candle set, then change `--from`, `--till`, and `--run-dir`.

Main outputs:

- `summary.json` - final equity, return, commissions, open positions, order counts.
- `trades.csv` - executed paper orders.
- `account_curve.csv` - equity/cash/positions after each runtime tick.
- `ranked_top.jsonl` - Kronos-ranked candidates and filter reasons.
- `risk_blocked_orders.jsonl` - orders blocked by risk/leverage constraints.
- `arena_state.sqlite3` - SQLite state, including cached Kronos forecasts.
- `logs/*.jsonl` - full decision payloads with entry/exit diagnostics.

For fair A/B strategy comparisons, reuse the same `arena_state.sqlite3` Kronos forecast cache. Fresh Kronos sampling can produce a different ranking even with the same code and dates.

Runtime outputs, SQLite state, logs, and backtest exports are intentionally ignored by git.

## How The Current Strategy Works

- `run-live` calls the runtime every minute via `rebalance.exit_interval_minutes: 1`.
- Entry/Kronos reassessment is gated to `rebalance.decision_interval_minutes: 60`, so minute ticks between hourly slots are exit-only.
- `kronos_single_top` asks Kronos to forecast the next candle, then ranks instruments by `gross_pred_return = abs(pred_close / last_close - 1)`.
- The sign of `pred_return` selects side: positive is long, negative is short.
- Only rank-1 can be traded. It must pass net-edge, top1/top2 gap, gross-prediction cap, session, sizing, and risk checks.
- Giveback trailing tracks raw directional PnL per open position, updates MFE on every runtime tick, arms at `+1.2%`, and closes after `60%` giveback from MFE.
- Force-flat is venue-scoped: a MOEX close only closes MOEX instruments, while Binance spot remains eligible when its symbol status is `TRADING`.

Kronos predicts `open/high/low/close/volume/amount`, but the current entry ranking uses only predicted close return. Other predicted candle fields are not yet part of ranking.

## Vector Positive Research

`kronos_vector_research` is an experimental entry mode for offline research. It builds long/short candidates, computes a positive vector and a risk vector for each candidate, scores the positive vector, and ranks allowed candidates by positive score. The runtime now supports a per-instrument `positive_threshold` in vector weights YAML:

```yaml
instrument_weights:
  VTBR:
    positive_weights:
      net_edge_score: 0.22
      edge_z_score: 0.08
      rr_score: 0.14
      mae_score: 0.12
      close_score: 0.16
      body_score: 0.03
      wick_score: 0.05
      candle_quality: 0.07
      edge_risk_quality: 0.13
    positive_threshold: 0.0
    risk_weights:
      false_breakout_risk: 0.25
      wide_spread_risk: 0.25
      late_entry_risk: 0.15
      high_mae_risk: 0.20
      direction_conflict_risk: 0.15
    risk_threshold: 1.0
```

`positive_threshold` defaults to `0.0` for old YAML files. Candidates below the threshold stay in diagnostics with reason `positive_score_below_threshold`, but they are not ranked among allowed entries. `risk_threshold: 1.0` keeps the risk vector effectively disabled for stage-1 positive-only experiments.

The vector-positive config example is:

```powershell
python -m arena_bot.runtime_backtest `
  --config configs/universe_v1_may1_14_vector_positive.yaml `
  --from 2026-05-01T00:00:00 `
  --till 2026-05-14T23:59:59 `
  --run-dir data/universe-v1-may1-14/runtime-runs/vector-positive
```

## Stage-1 Positive Optimizer

The offline optimizer learns one global positive-weight vector and one global `positive_threshold`. It does not optimize risk weights, does not use `risk_score` in the objective, and does not use risk as a filter or penalty.

The optimizer uses the economic positive prior as the softmax center:

```text
net_edge_score       0.22
close_score          0.16
rr_score             0.14
edge_risk_quality    0.13
mae_score            0.12
edge_z_score         0.08
candle_quality       0.07
wick_score           0.05
body_score           0.03
```

`pos_raw_*` Optuna parameters are deviations from `log(prior)`, not free raw weights. The optimized threshold defaults to the bounded search range `0.35..0.80`.

Example:

```powershell
python -m arena_bot.optimization.positive_weights `
  --config configs/universe_v1_may1_14.yaml `
  --from 2026-05-01T00:00:00 `
  --till 2026-05-14T23:59:59 `
  --trials 200 `
  --top-k 1 `
  --output data/universe-v1-may1-14/optimized/global_positive_weights.yaml `
  --study-db data/universe-v1-may1-14/optimized/positive_optuna.sqlite3 `
  --report data/universe-v1-may1-14/optimized/global_positive_report.json
```

The output weights YAML contains only `instrument_weights`, duplicated for all enabled instruments. Optimizer metadata is written to the JSON report. Generated optimizer outputs, SQLite files, candidate caches, and runtime runs should remain uncommitted.

## Positive Candidate Cache

Optuna trials and offline replays should not call Kronos repeatedly. Build or extend a JSONL candidate cache first:

```powershell
python -m arena_bot.optimization.build_positive_cache `
  --config configs/universe_v1_may1_14.yaml `
  --candles-dir data/universe-v1-may1-14/candles `
  --from 2026-05-01T00:00:00 `
  --till 2026-05-14T23:59:59 `
  --output data/universe-v1-may1-14/optimized/positive_candidates_full_session.jsonl `
  --base-cache data/universe-v1-may1-14/optimized/positive_candidates.jsonl
```

Rows include current bid/ask/mid, predicted OHLC, positive vector, raw vector metrics, costs, future bid/ask/mid, and realized net return for the next candle. Future data is used only as the target, never as a candidate feature.

## Fixed Positive Replay

`arena_bot.optimization.fixed_positive_replay` is an offline replay that re-ranks cached candidates without running Kronos again. Risk is recomputed/logged with fixed weights but is not used for ranking or filtering:

```text
false_breakout_risk      0.16
wide_spread_risk         0.30
late_entry_risk          0.20
high_mae_risk            0.24
direction_conflict_risk  0.10
```

The latest research mode ranks candidates by raw, non-normalized `net_edge`:

```powershell
python -m arena_bot.optimization.fixed_positive_replay `
  --config configs/universe_v1_may1_14.yaml `
  --candles-dir data/universe-v1-may1-14/candles `
  --candidate-cache data/universe-v1-may1-14/optimized/positive_candidates_crypto_shared_session.jsonl `
  --from 2026-05-01T00:00:00 `
  --till 2026-05-14T23:59:59 `
  --output-dir data/universe-v1-may1-14/runtime-runs/fixed-net-edge-single-top `
  --strategy single-top-net-edge-runtime-like `
  --max-positive-score 0.8 `
  --min-raw-mae-pct 0 `
  --max-raw-rr 40 `
  --close-before-session-gap
```

Current replay mechanics:

- Rank all cached candidates by raw `raw_vector_metrics.net_edge`, descending.
- Consider only rank-1. If rank-1 fails a filter, do not backfill with rank-2.
- Enter only if all filters pass: `positive_score <= 0.8`, `raw_mae_pct > 0`, `raw_rr < 40`, `gross_pred_return <= 0.75%`, `net_edge >= 0.15%`, and top1/top2 gap `>= 0.04%`.
- Hold if the same `secid + side` remains rank-1.
- Close and switch if a different rank-1 passes filters.
- Close to cash if rank-1 fails filters.
- Close before nights or non-trading gaps when the next decision timestamp is not the next hour.

This replay writes `ranked_decisions.jsonl`, `position_events.jsonl`, `trades.csv`, and `summary.json` under `--output-dir`. These outputs are ignored by git.
