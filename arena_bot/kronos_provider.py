from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .storage import StateStore
from .types import Instrument, KronosConfig, SignalRow

KLINE_COLS = ["open", "high", "low", "close", "volume", "amount"]


class RealKronosSignalProvider:
    name = "kronos"

    def __init__(
        self,
        *,
        config: KronosConfig,
        state: StateStore | None = None,
        refresh_cache: bool = False,
    ):
        self.config = config
        self.state = state
        self.refresh_cache = refresh_cache
        self._predictor = None
        self._model_name = str(config.model or "base")
        self._params_key = _params_key(config)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def params_key(self) -> str:
        return self._params_key

    def score(
        self,
        as_of: datetime,
        instruments: Sequence[Instrument],
        candles: Mapping[str, pd.DataFrame],
    ) -> Sequence[SignalRow]:
        if not self.config.enabled:
            return []
        eligible = list(instruments)
        if not eligible:
            return []

        as_of_s = as_of.isoformat(timespec="seconds")
        secids = [instrument.secid for instrument in eligible]
        cached: dict[str, dict[str, Any]] = {}
        if self.config.cache_enabled and self.state is not None and not self.refresh_cache:
            cached = self.state.load_kronos_forecasts(
                as_of=as_of_s,
                model=self.model_name,
                params_key=self.params_key,
                secids=tuple(secids),
            )

        forecasts: dict[str, dict[str, Any]] = dict(cached)
        missing = [instrument for instrument in eligible if instrument.secid not in forecasts]
        for instrument in missing:
            row = self._forecast_one(as_of, instrument, candles.get(instrument.secid, pd.DataFrame()))
            if row is not None:
                forecasts[instrument.secid] = row

        if not forecasts:
            return []

        scores = predicted_returns_to_bullish_scores({secid: row["pred_return"] for secid, row in forecasts.items()})
        rows = {}
        out = []
        for instrument in eligible:
            row = forecasts.get(instrument.secid)
            if row is None:
                continue
            bullish_score = float(scores.get(instrument.secid, 0.5))
            confidence = float(row.get("confidence", 0.0) or 0.0)
            saved = {**row, "bullish_score": bullish_score, "confidence": confidence}
            saved_metadata = dict(saved.get("metadata") or {})
            rows[instrument.secid] = saved
            out.append(
                SignalRow(
                    as_of=as_of,
                    secid=instrument.secid,
                    signal_name=self.name,
                    bullish_score=bullish_score,
                    confidence=confidence,
                    reason="real_kronos_forecast",
                    metadata={
                        **saved_metadata,
                        "model": self.model_name,
                        "last_close": saved["last_close"],
                        "pred_close": saved["pred_close"],
                        "pred_return": saved["pred_return"],
                        "cached": instrument.secid in cached and instrument.secid not in {item.secid for item in missing},
                    },
                )
            )

        if rows and self.config.cache_enabled and self.state is not None:
            self.state.save_kronos_forecasts(
                as_of=as_of_s,
                model=self.model_name,
                params_key=self.params_key,
                rows=rows,
            )
        return tuple(out)

    def forecast_paths(
        self,
        as_of: datetime,
        instruments: Sequence[Instrument],
        candles: Mapping[str, pd.DataFrame],
        *,
        pred_len: int | None = None,
        sample_count: int | None = None,
        max_target_time: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not self.config.enabled:
            return {}
        horizon = max(int(pred_len if pred_len is not None else self.config.pred_len), 1)
        samples_requested = max(int(sample_count if sample_count is not None else self.config.sample_count), 1)
        predictor = self._ensure_predictor()
        out: dict[str, dict[str, Any]] = {}
        for instrument in instruments:
            df = _prepare_candles(candles.get(instrument.secid, pd.DataFrame()), context_rows=max(int(self.config.context_rows), 2))
            if df.empty or len(df) < 2:
                continue
            last_close = _safe_float(df["close"].iloc[-1])
            if last_close <= 0:
                continue
            y_timestamp = _future_timestamps(df, as_of, pred_len=horizon)
            if max_target_time is not None and _future_exceeds_limit(y_timestamp, max_target_time):
                continue
            try:
                if hasattr(predictor, "predict_samples") and samples_requested > 1:
                    samples = predictor.predict_samples(
                        df=df[KLINE_COLS],
                        x_timestamp=df["timestamps"],
                        y_timestamp=y_timestamp,
                        pred_len=horizon,
                        T=float(self.config.temperature),
                        top_p=float(self.config.top_p),
                        sample_count=samples_requested,
                        verbose=False,
                    )
                    paths = [
                        [
                            {col: _safe_float(samples[path_idx, step_idx, col_idx]) for col_idx, col in enumerate(KLINE_COLS)}
                            for step_idx in range(horizon)
                        ]
                        for path_idx in range(int(samples.shape[0]))
                    ]
                else:
                    pred = predictor.predict(
                        df=df[KLINE_COLS],
                        x_timestamp=df["timestamps"],
                        y_timestamp=y_timestamp,
                        pred_len=horizon,
                        T=float(self.config.temperature),
                        top_p=float(self.config.top_p),
                        sample_count=samples_requested,
                        verbose=False,
                    )
                    paths = [
                        [
                            {col: _safe_float(row[col]) if col in pred.columns else 0.0 for col in KLINE_COLS}
                            for _, row in pred.tail(horizon).iterrows()
                        ]
                    ]
            except Exception:
                continue
            if not paths:
                continue
            out[instrument.secid] = {
                "secid": instrument.secid,
                "as_of": as_of.isoformat(timespec="seconds"),
                "last_close": last_close,
                "horizon": horizon,
                "sample_count": len(paths),
                "timestamps": [pd.Timestamp(value).isoformat() for value in y_timestamp],
                "paths": paths,
            }
        return out

    def _forecast_one(self, as_of: datetime, instrument: Instrument, candles: pd.DataFrame) -> dict[str, Any] | None:
        df = _prepare_candles(candles, context_rows=max(int(self.config.context_rows), 2))
        if df.empty or len(df) < 2:
            return None
        last_close = _safe_float(df["close"].iloc[-1])
        if last_close <= 0:
            return None
        pred_len = max(int(self.config.pred_len), 1)
        y_timestamp = _future_timestamps(df, as_of, pred_len=pred_len)
        predictor = self._ensure_predictor()
        sample_count = max(int(self.config.sample_count), 1)
        try:
            if hasattr(predictor, "predict_samples") and sample_count > 1:
                samples = predictor.predict_samples(
                    df=df[KLINE_COLS],
                    x_timestamp=df["timestamps"],
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=float(self.config.temperature),
                    top_p=float(self.config.top_p),
                    sample_count=sample_count,
                    verbose=False,
                )
                close_samples = [float(value) for value in samples[:, pred_len - 1, KLINE_COLS.index("close")]]
                pred_ohlcv = {
                    col: float(sum(_safe_float(value) for value in samples[:, pred_len - 1, idx]) / len(samples))
                    for idx, col in enumerate(KLINE_COLS)
                }
                pred_close = pred_ohlcv["close"]
                confidence = _direction_confidence(close_samples, last_close)
            else:
                pred = predictor.predict(
                    df=df[KLINE_COLS],
                    x_timestamp=df["timestamps"],
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=float(self.config.temperature),
                    top_p=float(self.config.top_p),
                    sample_count=sample_count,
                    verbose=False,
                )
                last_pred = pred.iloc[-1]
                pred_ohlcv = {col: _safe_float(last_pred[col]) if col in pred.columns else 0.0 for col in KLINE_COLS}
                pred_close = pred_ohlcv["close"]
                confidence = _move_confidence(pred_close / last_close - 1.0)
        except Exception:
            return None
        pred_return = pred_close / last_close - 1.0 if last_close > 0 else 0.0
        return {
            "last_close": last_close,
            "pred_close": pred_close,
            "pred_return": pred_return,
            "bullish_score": 0.5,
            "confidence": confidence,
            "metadata": {
                "code_dir": str(self.config.code_dir),
                "weights_dir": str(self.config.weights_dir),
                "pred_ohlcv": pred_ohlcv,
                "target_timestamp": y_timestamp.iloc[-1].isoformat(),
            },
        }

    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor

        code_dir = Path(self.config.code_dir)
        repo_root = code_dir.parent if code_dir.name == "model" else code_dir
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

        model_dir = _model_dir(Path(self.config.weights_dir), self.model_name)
        tokenizer_dir = _tokenizer_dir(Path(self.config.weights_dir), self.model_name)
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir))
        model = Kronos.from_pretrained(str(model_dir))
        tokenizer.eval()
        model.eval()
        device = _resolve_device(self.config.device)
        self._predictor = KronosPredictor(
            model,
            tokenizer,
            device=device,
            max_context=min(_native_context(self.model_name), max(int(self.config.context_rows), 2)),
        )
        return self._predictor


def predicted_returns_to_bullish_scores(predicted_returns: Mapping[str, float]) -> dict[str, float]:
    clean = {
        secid: float(value)
        for secid, value in predicted_returns.items()
        if _is_finite(value)
    }
    if not clean:
        return {}
    ordered = sorted(clean.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 0.5}
    denom = len(ordered) - 1
    return {secid: idx / denom for idx, (secid, _) in enumerate(ordered)}


def _prepare_candles(candles: pd.DataFrame, *, context_rows: int) -> pd.DataFrame:
    if candles is None or candles.empty or "timestamps" not in candles.columns:
        return pd.DataFrame()
    df = candles.copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce")
    df = df.dropna(subset=["timestamps"]).sort_values("timestamps").tail(context_rows).reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0) * pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    for col in KLINE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.dropna(subset=["open", "high", "low", "close"])


def _next_timestamp(df: pd.DataFrame, as_of: datetime) -> pd.Timestamp:
    ts = pd.to_datetime(df["timestamps"])
    if len(ts) >= 2:
        delta = ts.iloc[-1] - ts.iloc[-2]
        if pd.notna(delta) and delta.total_seconds() > 0:
            return pd.Timestamp(ts.iloc[-1]) + delta
    return pd.Timestamp(as_of) + pd.Timedelta(hours=1)


def _future_timestamps(df: pd.DataFrame, as_of: datetime, *, pred_len: int) -> pd.Series:
    first = _next_timestamp(df, as_of)
    ts = pd.to_datetime(df["timestamps"])
    if len(ts) >= 2:
        delta = ts.iloc[-1] - ts.iloc[-2]
        if pd.notna(delta) and delta.total_seconds() > 0:
            return pd.Series([first + idx * delta for idx in range(max(int(pred_len), 1))])
    return pd.Series([first + pd.Timedelta(hours=idx) for idx in range(max(int(pred_len), 1))])


def _future_exceeds_limit(timestamps: pd.Series, max_target_time: datetime) -> bool:
    limit = pd.Timestamp(max_target_time)
    for value in timestamps:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None and limit.tzinfo is not None:
            ts = ts.tz_localize(limit.tzinfo)
        elif ts.tzinfo is not None and limit.tzinfo is not None:
            ts = ts.tz_convert(limit.tzinfo)
        elif ts.tzinfo is not None and limit.tzinfo is None:
            limit = limit.tz_localize(ts.tzinfo)
        if ts > limit:
            return True
    return False


def _direction_confidence(pred_closes: Sequence[float], last_close: float) -> float:
    directions = [1 if value > last_close else (-1 if value < last_close else 0) for value in pred_closes if _is_finite(value)]
    if not directions:
        return 0.0
    best = max(directions.count(-1), directions.count(0), directions.count(1)) / len(directions)
    move = _move_confidence(sum(pred_closes) / len(pred_closes) / last_close - 1.0 if last_close > 0 else 0.0)
    return max(float(best), move)


def _move_confidence(pred_return: float) -> float:
    return min(abs(float(pred_return)) / 0.01, 1.0) if _is_finite(pred_return) else 0.0


def _model_dir(weights_dir: Path, model_name: str) -> Path:
    return weights_dir / f"NeoQuasar__Kronos-{model_name}"


def _tokenizer_dir(weights_dir: Path, model_name: str) -> Path:
    suffix = "2k" if model_name == "mini" else "base"
    return weights_dir / f"NeoQuasar__Kronos-Tokenizer-{suffix}"


def _native_context(model_name: str) -> int:
    return 2048 if model_name == "mini" else 512


def _resolve_device(value: str) -> str:
    if value and value != "auto":
        return value
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _params_key(config: KronosConfig) -> str:
    payload = {
        "model": config.model,
        "context_rows": int(config.context_rows),
        "pred_len": int(config.pred_len),
        "sample_count": int(config.sample_count),
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False
