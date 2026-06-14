from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import optuna
import pandas as pd

from ..config import load_config
from ..market_data import SavedCandleMarketDataProvider
from .optuna_per_instrument import extract_normalized_weights_from_best_params, sample_normalized_weights
from .scoring import (
    BASELINE_RISK_WEIGHTS,
    POSITIVE_METRICS,
    RISK_METRICS,
    compute_candidate_vectors,
    instrument_scales_from_history,
    save_instrument_weights_yaml,
    score_candidate,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="optuna-cached-vectors")
    parser.add_argument("--config", default="configs/universe_v1_may1_14.yaml")
    parser.add_argument("--cache-db", required=True)
    parser.add_argument("--candles-dir", required=True)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--till", dest="till_date", required=True)
    parser.add_argument("--instruments", nargs="*", default=None)
    parser.add_argument("--n-trials", type=int, default=5000)
    parser.add_argument("--output", default="configs/optimized_per_instrument_weights.yaml")
    parser.add_argument("--run-dir", default="data/optuna-cached-vector-run")
    parser.add_argument("--study-storage", default="")
    parser.add_argument("--study-prefix", default="cached-vector")
    parser.add_argument("--checkpoint-every-trials", type=int, default=250)
    parser.add_argument("--starting-cash", type=float, default=100000.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", choices=("debug", "info", "warning"), default="warning")
    parser.add_argument("--ignore-risk-threshold", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.log_level == "warning":
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    elif args.log_level == "info":
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

    runner = CachedVectorOptunaRunner(args)
    runner.run()
    return 0


class CachedVectorOptunaRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = Path(args.config)
        self.cache_db = Path(args.cache_db)
        self.candles_dir = Path(args.candles_dir)
        self.output_path = Path(args.output)
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.run_dir / "summary.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.config = load_config(self.config_path)
        self.provider = SavedCandleMarketDataProvider(directories=[self.candles_dir], history_rows=512)
        selected = set(args.instruments or [])
        self.instruments = [
            instrument
            for instrument in self.config.instruments
            if instrument.enabled and (not selected or instrument.secid in selected)
        ]
        self.instruments_by_secid = {instrument.secid: instrument for instrument in self.instruments}
        self.commission = float(self.config.risk.commission_rate)
        self.slippage_multiplier = float(self.config.risk.slippage_spread_multiplier)
        storage = str(args.study_storage or "").strip()
        if storage:
            self.storage_url = storage
        else:
            self.storage_url = "sqlite:///" + (self.run_dir / "optuna_studies.sqlite3").resolve().as_posix()
        self.optimized: dict[str, dict[str, Any]] = {}
        self.summary: dict[str, Any] = {
            "mode": "cached_vector_research",
            "config": str(self.config_path),
            "cache_db": str(self.cache_db),
            "candles_dir": str(self.candles_dir),
            "from": str(args.from_date),
            "till": str(args.till_date),
            "n_trials": int(args.n_trials),
            "checkpoint_every_trials": int(args.checkpoint_every_trials),
            "ignore_risk_threshold": bool(args.ignore_risk_threshold),
            "study_storage": self.storage_url,
            "output": str(self.output_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": "",
            "instruments": {},
        }

    def run(self) -> None:
        self._event("run_start", instruments=[instrument.secid for instrument in self.instruments])
        for instrument in self.instruments:
            self._optimize_one(instrument.secid)
        self.summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_checkpoint()
        self._event("run_done", optimized_count=len(self.optimized))

    def _optimize_one(self, secid: str) -> None:
        dataset = self._build_dataset(secid)
        if not dataset:
            self.summary["instruments"][secid] = {"status": "skipped", "reason": "empty_dataset"}
            self._write_checkpoint()
            self._event("instrument_skipped", secid=secid, reason="empty_dataset")
            return

        study_name = f"{self.args.study_prefix}-{secid}"
        sampler = optuna.samplers.TPESampler(seed=self.args.seed)
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=self.storage_url,
            load_if_exists=True,
            sampler=sampler,
        )
        existing_trials = len([trial for trial in study.trials if trial.value is not None])
        remaining_trials = max(int(self.args.n_trials) - existing_trials, 0)
        self.summary["instruments"].setdefault(
            secid,
            {
                "status": "running",
                "candidate_rows": len(dataset),
                "timestamps": len({row["as_of"] for row in dataset}),
                "existing_trials": existing_trials,
                "target_trials": int(self.args.n_trials),
            },
        )
        self._event(
            "instrument_start",
            secid=secid,
            candidate_rows=len(dataset),
            timestamps=len({row["as_of"] for row in dataset}),
            existing_trials=existing_trials,
            remaining_trials=remaining_trials,
        )

        def objective(trial: optuna.Trial) -> float:
            weights = {
                "positive_weights": sample_normalized_weights(trial, f"{secid}_positive", POSITIVE_METRICS),
                "risk_weights": (
                    dict(BASELINE_RISK_WEIGHTS)
                    if self.args.ignore_risk_threshold
                    else sample_normalized_weights(trial, f"{secid}_risk", RISK_METRICS)
                ),
                "risk_threshold": (
                    1.0
                    if self.args.ignore_risk_threshold
                    else trial.suggest_float(f"{secid}_risk_threshold", 0.10, 0.80)
                ),
            }
            return self._evaluate(dataset, weights)["objective"]

        def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            completed = len([item for item in study.trials if item.value is not None])
            checkpoint_every = max(int(self.args.checkpoint_every_trials), 1)
            if completed % checkpoint_every == 0:
                self._checkpoint_study(secid, dataset, study, status="running")

        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, callbacks=[callback], show_progress_bar=False)
        self._checkpoint_study(secid, dataset, study, status="done")
        self._event("instrument_done", secid=secid, best_value=float(study.best_value), trials=len(study.trials))

    def _checkpoint_study(self, secid: str, dataset: list[dict[str, Any]], study: optuna.Study, *, status: str) -> None:
        try:
            weights = self._weights_from_study(secid, study)
            eval_result = self._evaluate(dataset, weights)
            self.optimized[secid] = {**weights, "objective_value": float(study.best_value)}
            self.summary["instruments"][secid] = {
                "status": status,
                "candidate_rows": len(dataset),
                "timestamps": len({row["as_of"] for row in dataset}),
                "completed_trials": len([trial for trial in study.trials if trial.value is not None]),
                "target_trials": int(self.args.n_trials),
                "best_value": float(study.best_value),
                **eval_result,
            }
            self._write_checkpoint()
            self._event(
                "checkpoint",
                secid=secid,
                status=status,
                completed_trials=self.summary["instruments"][secid]["completed_trials"],
                best_value=float(study.best_value),
                net_pnl=eval_result["net_pnl"],
                max_drawdown=eval_result["max_drawdown"],
                trade_count=eval_result["trade_count"],
                selected_long=eval_result["selected_long"],
                selected_short=eval_result["selected_short"],
            )
        except ValueError:
            return

    def _weights_from_study(self, secid: str, study: optuna.Study) -> dict[str, Any]:
        best_params = study.best_params
        if self.args.ignore_risk_threshold:
            return {
                "positive_weights": extract_normalized_weights_from_best_params(
                    best_params,
                    f"{secid}_positive",
                    POSITIVE_METRICS,
                ),
                "risk_weights": dict(BASELINE_RISK_WEIGHTS),
                "risk_threshold": 1.0,
            }
        return {
            "positive_weights": extract_normalized_weights_from_best_params(
                best_params,
                f"{secid}_positive",
                POSITIVE_METRICS,
            ),
            "risk_weights": extract_normalized_weights_from_best_params(
                best_params,
                f"{secid}_risk",
                RISK_METRICS,
            ),
            "risk_threshold": float(best_params[f"{secid}_risk_threshold"]),
        }

    def _write_checkpoint(self) -> None:
        if self.optimized:
            save_instrument_weights_yaml(self.optimized, self.output_path)
        self.summary_path.write_text(json.dumps(self.summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_dataset(self, secid: str) -> list[dict[str, Any]]:
        instrument = self.instruments_by_secid[secid]
        raw = self.provider._load(secid)
        dataset = []
        for row in self._load_forecasts(secid):
            as_of = datetime.fromisoformat(row["as_of"])
            next_ts = _next_timestamp(raw, as_of)
            if next_ts is None:
                continue
            snapshot = self.provider.snapshots(as_of, [instrument]).get(secid)
            next_snapshot = self.provider.snapshots(next_ts, [instrument]).get(secid)
            candles = self.provider.candles(as_of, [instrument]).get(secid)
            metric = self.provider.metrics(as_of, [instrument]).get(secid)
            if snapshot is None or next_snapshot is None or metric is None:
                continue
            bid = float(snapshot.bid or 0.0)
            ask = float(snapshot.ask or 0.0)
            exit_bid = float(next_snapshot.bid or next_snapshot.last_price or 0.0)
            exit_ask = float(next_snapshot.ask or next_snapshot.last_price or 0.0)
            if bid <= 0 or ask <= 0 or exit_bid <= 0 or exit_ask <= 0 or ask < bid:
                continue
            spread_rate = float(metric.spread_pct or snapshot.spread_pct or 0.0)
            slippage_rate = max(spread_rate, 0.0) * self.slippage_multiplier
            scales = instrument_scales_from_history(candles, snapshot=snapshot, metric=metric)
            pred = row["pred_ohlcv"]
            for side in ("long", "short"):
                candidate = {
                    "secid": secid,
                    "direction": side,
                    "side": side,
                    "bid": bid,
                    "ask": ask,
                    "pred_open": float(pred.get("open", 0.0) or 0.0),
                    "pred_high": float(pred.get("high", 0.0) or 0.0),
                    "pred_low": float(pred.get("low", 0.0) or 0.0),
                    "pred_close": float(pred.get("close", 0.0) or 0.0),
                    "commission_rate": self.commission,
                    "slippage_rate": slippage_rate,
                }
                try:
                    vectors = compute_candidate_vectors(candidate, scales)
                except Exception:
                    continue
                dataset.append(
                    {
                        "as_of": row["as_of"],
                        "side": side,
                        "positive_vector": vectors["positive_vector"],
                        "risk_vector": vectors["risk_vector"],
                        "net_return": self._execution_net_return(
                            side=side,
                            entry_bid=bid,
                            entry_ask=ask,
                            exit_bid=exit_bid,
                            exit_ask=exit_ask,
                            slippage_rate=slippage_rate,
                        ),
                    }
                )
        return dataset

    def _load_forecasts(self, secid: str) -> list[dict[str, Any]]:
        con = sqlite3.connect(self.cache_db)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                """
                SELECT as_of, secid, pred_return, metadata_json
                FROM kronos_forecasts
                WHERE secid = ? AND as_of >= ? AND as_of <= ?
                ORDER BY as_of
                """,
                (secid, self.args.from_date, self.args.till_date),
            ).fetchall()
        finally:
            con.close()
        out = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"])
            except Exception:
                metadata = {}
            pred = metadata.get("pred_ohlcv") if isinstance(metadata, Mapping) else None
            if isinstance(pred, Mapping):
                out.append({"as_of": row["as_of"], "pred_return": float(row["pred_return"] or 0.0), "pred_ohlcv": dict(pred)})
        return out

    def _execution_net_return(
        self,
        *,
        side: str,
        entry_bid: float,
        entry_ask: float,
        exit_bid: float,
        exit_ask: float,
        slippage_rate: float,
    ) -> float:
        cost = 2.0 * self.commission + 2.0 * slippage_rate
        if side == "long":
            return ((exit_bid - entry_ask) / entry_ask) - cost if entry_ask > 0 else 0.0
        return ((entry_bid - exit_ask) / entry_bid) - cost if entry_bid > 0 else 0.0

    def _evaluate(self, dataset: list[dict[str, Any]], weights: dict[str, Any]) -> dict[str, Any]:
        grouped = defaultdict(list)
        for row in dataset:
            grouped[row["as_of"]].append(row)
        equity = float(self.args.starting_cash)
        peak = equity
        max_drawdown = 0.0
        trade_count = 0
        selected_long = 0
        selected_short = 0
        for as_of in sorted(grouped):
            allowed = []
            for candidate in grouped[as_of]:
                scores = score_candidate(candidate, weights)
                if self.args.ignore_risk_threshold or scores["risk_score"] <= float(weights["risk_threshold"]):
                    allowed.append((scores["positive_score"], candidate))
            if not allowed:
                continue
            _, selected = max(allowed, key=lambda item: item[0])
            equity *= 1.0 + float(selected["net_return"])
            trade_count += 1
            if selected["side"] == "long":
                selected_long += 1
            else:
                selected_short += 1
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
        net_pnl = equity - float(self.args.starting_cash)
        return {
            "net_pnl": net_pnl,
            "max_drawdown": max_drawdown,
            "trade_count": trade_count,
            "selected_long": selected_long,
            "selected_short": selected_short,
            "objective": net_pnl - 0.25 * max_drawdown,
        }

    def _event(self, event: str, **payload: Any) -> None:
        row = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


def _next_timestamp(raw: pd.DataFrame, as_of: datetime) -> datetime | None:
    if raw.empty or "timestamps" not in raw:
        return None
    later = raw["timestamps"][raw["timestamps"] > pd.Timestamp(as_of)]
    if later.empty:
        return None
    return pd.Timestamp(later.iloc[0]).to_pydatetime()


if __name__ == "__main__":
    raise SystemExit(main())
