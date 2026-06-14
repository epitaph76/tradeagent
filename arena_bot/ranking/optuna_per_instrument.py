from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..config import load_config
from ..runtime_backtest import run as run_runtime_backtest
from .scoring import POSITIVE_METRICS, RISK_METRICS, save_instrument_weights_yaml


@dataclass(frozen=True)
class SingleInstrumentBacktestResult:
    net_pnl: float
    max_drawdown: float = 0.0
    trade_count: int = 0
    selected_candidates_log: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    run_dir: str = ""


def sample_normalized_weights(
    trial: Any,
    prefix: str,
    metrics: list[str],
    low: float = 0.0,
    high: float = 2.0,
) -> dict[str, float]:
    raw = {metric: trial.suggest_float(f"{prefix}_{metric}", low, high) for metric in metrics}
    total = sum(raw.values()) + 1e-9
    return {metric: raw[metric] / total for metric in metrics}


def extract_normalized_weights_from_best_params(
    best_params: dict[str, float],
    prefix: str,
    metrics: list[str],
) -> dict[str, float]:
    raw = {metric: best_params[f"{prefix}_{metric}"] for metric in metrics}
    total = sum(raw.values()) + 1e-9
    return {metric: raw[metric] / total for metric in metrics}


def optimize_instrument_weights(
    secid: str,
    n_trials: int = 200,
    *,
    config_path: str = "configs/universe_v1_may1_14.yaml",
    from_date: str | None = None,
    till_date: str | None = None,
    run_root: str | Path | None = None,
    keep_runs: bool = False,
):
    import optuna

    def objective(trial: Any) -> float:
        positive_weights = sample_normalized_weights(
            trial=trial,
            prefix=f"{secid}_positive",
            metrics=POSITIVE_METRICS,
        )
        risk_weights = sample_normalized_weights(
            trial=trial,
            prefix=f"{secid}_risk",
            metrics=RISK_METRICS,
        )
        risk_threshold = trial.suggest_float(f"{secid}_risk_threshold", 0.10, 0.80)
        result = run_single_instrument_backtest_with_weights(
            secid=secid,
            positive_weights=positive_weights,
            risk_weights=risk_weights,
            risk_threshold=risk_threshold,
            config_path=config_path,
            from_date=from_date,
            till_date=till_date,
            run_root=run_root,
            keep_run=keep_runs,
        )
        score = result.net_pnl
        if hasattr(result, "max_drawdown"):
            score -= 0.25 * result.max_drawdown
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


def optimize_all_instruments(
    instruments: list[str],
    n_trials_per_instrument: int = 200,
    *,
    config_path: str = "configs/universe_v1_may1_14.yaml",
    from_date: str | None = None,
    till_date: str | None = None,
    output_path: str | Path = "configs/optimized_per_instrument_weights.yaml",
    run_root: str | Path | None = None,
    keep_runs: bool = False,
) -> dict[str, dict[str, Any]]:
    optimized: dict[str, dict[str, Any]] = {}
    for secid in instruments:
        study = optimize_instrument_weights(
            secid=secid,
            n_trials=n_trials_per_instrument,
            config_path=config_path,
            from_date=from_date,
            till_date=till_date,
            run_root=run_root,
            keep_runs=keep_runs,
        )
        best_params = study.best_params
        positive_weights = extract_normalized_weights_from_best_params(
            best_params=best_params,
            prefix=f"{secid}_positive",
            metrics=POSITIVE_METRICS,
        )
        risk_weights = extract_normalized_weights_from_best_params(
            best_params=best_params,
            prefix=f"{secid}_risk",
            metrics=RISK_METRICS,
        )
        optimized[secid] = {
            "positive_weights": positive_weights,
            "risk_weights": risk_weights,
            "risk_threshold": float(best_params[f"{secid}_risk_threshold"]),
            "objective_value": float(study.best_value),
        }
        save_instrument_weights_yaml(optimized, output_path)
    return optimized


def run_single_instrument_backtest_with_weights(
    secid: str,
    positive_weights: dict[str, float],
    risk_weights: dict[str, float],
    risk_threshold: float,
    *,
    config_path: str = "configs/universe_v1_may1_14.yaml",
    from_date: str | None = None,
    till_date: str | None = None,
    run_root: str | Path | None = None,
    keep_run: bool = False,
) -> SingleInstrumentBacktestResult:
    config_path_obj = Path(config_path)
    base_config = load_config(config_path_obj)
    from_s, till_s = _resolve_backtest_window(config_path_obj, secid, from_date=from_date, till_date=till_date)
    root = Path(run_root) if run_root is not None else Path(tempfile.mkdtemp(prefix="kronos-vector-optuna-"))
    root.mkdir(parents=True, exist_ok=True)
    trial_dir = Path(tempfile.mkdtemp(prefix=f"{secid}-", dir=root))
    config_out = trial_dir / "config.yaml"
    weights_out = trial_dir / "weights.yaml"
    run_dir = trial_dir / "run"

    try:
        save_instrument_weights_yaml(
            {
                secid: {
                    "positive_weights": positive_weights,
                    "risk_weights": risk_weights,
                    "risk_threshold": float(risk_threshold),
                }
            },
            weights_out,
        )
        _write_single_instrument_config(
            config_path=config_path_obj,
            out_path=config_out,
            secid=secid,
            weights_path=weights_out,
        )
        args = argparse.Namespace(
            config=str(config_out),
            from_date=from_s,
            till_date=till_s,
            run_dir=str(run_dir),
            progress_every=0,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            run_runtime_backtest(args)

        summary = _read_json(run_dir / "summary.json")
        starting_cash = float(summary.get("starting_cash", base_config.risk.starting_cash) or base_config.risk.starting_cash)
        final_equity = float(summary.get("final_equity_liquidation", starting_cash) or starting_cash)
        selected_log = tuple(_read_selected_candidates(run_dir / "ranked_top.jsonl"))
        result = SingleInstrumentBacktestResult(
            net_pnl=final_equity - starting_cash,
            max_drawdown=_max_drawdown(run_dir / "account_curve.csv"),
            trade_count=_trade_count(run_dir / "trades.csv"),
            selected_candidates_log=selected_log,
            run_dir=str(run_dir),
        )
        if keep_run:
            return result
        return result
    finally:
        if not keep_run:
            shutil.rmtree(trial_dir, ignore_errors=True)
            if run_root is None:
                shutil.rmtree(root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="optuna-per-instrument")
    parser.add_argument("--config", default="configs/universe_v1_may1_14.yaml")
    parser.add_argument("--instruments", nargs="*", default=None)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--from", dest="from_date", default=None)
    parser.add_argument("--till", dest="till_date", default=None)
    parser.add_argument("--output", default="configs/optimized_per_instrument_weights.yaml")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    instruments = list(args.instruments or [])
    if not instruments:
        config = load_config(args.config)
        instruments = [instrument.secid for instrument in config.instruments if instrument.enabled]
    optimized = optimize_all_instruments(
        instruments=instruments,
        n_trials_per_instrument=int(args.n_trials),
        config_path=str(args.config),
        from_date=args.from_date,
        till_date=args.till_date,
        output_path=args.output,
        run_root=args.run_root,
        keep_runs=bool(args.keep_runs),
    )
    print(json.dumps({"instrument_weights": optimized}, ensure_ascii=False, indent=2))
    return 0


def _write_single_instrument_config(*, config_path: Path, out_path: Path, secid: str, weights_path: Path) -> None:
    raw = _read_yaml(config_path)
    instruments = list(raw.get("instruments") or [])
    filtered = [row for row in instruments if str(row.get("secid") if isinstance(row, Mapping) else row) == secid]
    if not filtered:
        raise ValueError(f"instrument {secid} not found in {config_path}")
    raw["instruments"] = filtered
    _absolutize_saved_candle_directories(raw, config_path=config_path)
    lifecycle = dict(raw.get("trade_lifecycle") or {})
    entry = dict(lifecycle.get("entry") or {})
    entry["mode"] = "kronos_vector_research"
    entry["instrument_weights_path"] = str(weights_path)
    lifecycle["entry"] = entry
    raw["trade_lifecycle"] = lifecycle
    out_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _absolutize_saved_candle_directories(raw: dict[str, Any], *, config_path: Path) -> None:
    market = raw.get("market_data")
    if not isinstance(market, Mapping):
        return
    saved = market.get("saved_candles")
    if not isinstance(saved, Mapping):
        return
    directories = []
    for value in list(saved.get("directories") or []):
        path = Path(str(value))
        if path.is_absolute():
            directories.append(str(path))
            continue
        candidates = [
            config_path.parent / path,
            config_path.parent.parent / path,
            Path.cwd() / path,
        ]
        resolved = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        directories.append(str(resolved.resolve()))
    market = dict(market)
    saved = dict(saved)
    saved["directories"] = directories
    market["saved_candles"] = saved
    raw["market_data"] = market


def _resolve_backtest_window(
    config_path: Path,
    secid: str,
    *,
    from_date: str | None,
    till_date: str | None,
) -> tuple[str, str]:
    if from_date and till_date:
        return from_date, till_date
    candle_path = _find_saved_candle_file(config_path, secid)
    if candle_path is None:
        if not from_date or not till_date:
            raise ValueError("--from and --till are required when saved candles cannot be inferred")
        return from_date, till_date
    import pandas as pd

    df = pd.read_csv(candle_path, usecols=["timestamps"])
    ts = pd.to_datetime(df["timestamps"], errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        raise ValueError(f"not enough timestamps in {candle_path}")
    resolved_from = from_date or ts.iloc[0].to_pydatetime().isoformat(timespec="seconds")
    resolved_till = till_date or ts.iloc[-1].to_pydatetime().isoformat(timespec="seconds")
    return resolved_from, resolved_till


def _find_saved_candle_file(config_path: Path, secid: str) -> Path | None:
    raw = _read_yaml(config_path)
    saved = (((raw.get("market_data") or {}).get("saved_candles")) or {}) if isinstance(raw, Mapping) else {}
    directories = list(saved.get("directories") or [])
    patterns = list(saved.get("filename_patterns") or ("candles_{secid}.csv", "candles_1m_{secid}.csv"))
    for directory in directories:
        directory_path = Path(str(directory))
        if not directory_path.is_absolute():
            candidates = [
                config_path.parent / directory_path,
                config_path.parent.parent / directory_path,
                Path.cwd() / directory_path,
            ]
            directory_path = next((candidate for candidate in candidates if candidate.exists()), Path(str(directory)))
        for pattern in patterns:
            path = directory_path / str(pattern).format(secid=secid)
            if path.exists():
                return path
    return None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _max_drawdown(path: Path) -> float:
    if not path.exists():
        return 0.0
    peak: float | None = None
    max_dd = 0.0
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            equity = float(row.get("equity_liquidation", 0.0) or 0.0)
            peak = equity if peak is None else max(peak, equity)
            max_dd = max(max_dd, (peak or equity) - equity)
    return max_dd


def _trade_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _read_selected_candidates(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, Mapping) and bool(row.get("selected")):
                out.append(dict(row))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
