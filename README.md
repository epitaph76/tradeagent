# Tradeagent

Compact trading runtime built around Kronos forecasts, LightGBM selector metadata, risk filters, trading-session guards, and particle-based exit tracking.

## Layout

- `arena_bot/` - runtime, storage, risk/session logic, Kronos provider, backtest runner, CLI.
- `configs/` - paper/runtime configs with relative paths.
- `kronos/model/` - Kronos model code used by `RealKronosSignalProvider`.
- `kronos/weights/` - local Kronos base model and tokenizer weights. Large tensor files are tracked with Git LFS.
- `data/universe-v1-may1-14/candles/` - compact May 1-14 candle set used by the current saved-candle config.
- `data/universe-v1-may1-14/models/lightgbm_meta/` - latest LightGBM meta-selector model.
- `tests/` - regression tests for runtime, Kronos provider, storage, session logic, and LightGBM behavior.

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
