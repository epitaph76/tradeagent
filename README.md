# Tradeagent

Trading runtime built around Kronos forecasts, LightGBM selector metadata, risk filters, trading-session guards, and giveback trailing exits.

For coding agents, start with [AGENTS.md](AGENTS.md). It explains the current architecture, runtime flow, implemented strategy modes, commands, and safe working rules.

## What Is Implemented

- `kronos_rank` entry mode: Kronos ranks candidates, then the runtime applies strict round-trip cost filtering.
- `kronos_single_top` entry mode: hourly full-capital baseline that ranks the universe, trades only rank-1 if strict cost, gross-return, and top1/top2 gap filters pass, and blocks entries unless a full next Kronos reassessment interval remains.
- Giveback trailing exits: open positions track MFE on every runtime tick; `run-live` ticks every minute while entries remain hourly.
- Particle exit tracking: still implemented for configs that enable Kronos sample-path exits.
- Session-safe trading: v1 uses a single `Europe/Moscow` window and forces all assets cash-flat near session close.
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
- `force_flat_time: 18:30` closes remaining positions to cash.

Kronos predicts `open/high/low/close/volume/amount`, but the current entry ranking uses only predicted close return. Other predicted candle fields are not yet part of ranking.
