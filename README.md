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

## Run May 1-14 Runtime Backtest

```powershell
python -m arena_bot.runtime_backtest `
  --config configs/universe_v1_may1_14.yaml `
  --from 2026-05-01T00:00:00 `
  --till 2026-05-14T23:59:59 `
  --run-dir data/universe-v1-may1-14/runtime-runs/manual
```

Runtime outputs, SQLite state, logs, and backtest exports are intentionally ignored by git.
