from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    as_of TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    secid TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS selector_returns (
                    as_of TEXT NOT NULL,
                    selector TEXT NOT NULL,
                    return_value REAL NOT NULL,
                    PRIMARY KEY(as_of, selector)
                );
                CREATE TABLE IF NOT EXISTS market_features (
                    as_of TEXT PRIMARY KEY,
                    features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_positions (
                    secid TEXT PRIMARY KEY,
                    lots INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS account_state (
                    account_id TEXT PRIMARY KEY,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    gross REAL NOT NULL,
                    net REAL NOT NULL,
                    margin_used REAL NOT NULL,
                    available_cash REAL NOT NULL DEFAULT 0,
                    available_gross REAL NOT NULL DEFAULT 0,
                    cash_buffer REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS selector_paper_positions (
                    selector TEXT NOT NULL,
                    secid TEXT NOT NULL,
                    weight REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(selector, secid)
                );
                CREATE TABLE IF NOT EXISTS kronos_forecasts (
                    as_of TEXT NOT NULL,
                    secid TEXT NOT NULL,
                    model TEXT NOT NULL,
                    params_key TEXT NOT NULL,
                    last_close REAL NOT NULL,
                    pred_close REAL NOT NULL,
                    pred_return REAL NOT NULL,
                    bullish_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(as_of, secid, model, params_key)
                );
                CREATE TABLE IF NOT EXISTS kronos_exit_trackers (
                    secid TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated_at TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL,
                    current_step INTEGER NOT NULL,
                    planned_exit_at TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL,
                    state_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    def insert_decision(self, decision_id: str, as_of: str, payload: Mapping[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO decisions(decision_id, as_of, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (decision_id, as_of, _json(payload), _now()),
            )

    def insert_order(
        self,
        *,
        decision_id: str,
        as_of: str,
        secid: str,
        direction: str,
        quantity: int,
        status: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any] | None = None,
        error: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO orders(
                    decision_id, as_of, secid, direction, quantity, status,
                    request_json, response_json, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (decision_id, as_of, secid, direction, int(quantity), status, _json(request), _json(response or {}), error, _now()),
            )

    def save_market_features(self, as_of: str, features: Mapping[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO market_features(as_of, features_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(as_of) DO UPDATE SET
                    features_json=excluded.features_json,
                    created_at=excluded.created_at
                """,
                (as_of, _json(features), _now()),
            )

    def append_selector_returns(self, as_of: str, returns: Mapping[str, float]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO selector_returns(as_of, selector, return_value)
                VALUES (?, ?, ?)
                ON CONFLICT(as_of, selector) DO UPDATE SET return_value=excluded.return_value
                """,
                [(as_of, str(selector), float(value or 0.0)) for selector, value in returns.items()],
            )

    def clear_training_history(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM selector_returns")
            conn.execute("DELETE FROM market_features")

    def save_kronos_forecasts(
        self,
        *,
        as_of: str,
        model: str,
        params_key: str,
        rows: Mapping[str, Mapping[str, Any]],
    ) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO kronos_forecasts(
                    as_of, secid, model, params_key, last_close, pred_close,
                    pred_return, bullish_score, confidence, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(as_of, secid, model, params_key) DO UPDATE SET
                    last_close=excluded.last_close,
                    pred_close=excluded.pred_close,
                    pred_return=excluded.pred_return,
                    bullish_score=excluded.bullish_score,
                    confidence=excluded.confidence,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at
                """,
                [
                    (
                        as_of,
                        str(secid),
                        model,
                        params_key,
                        float(row.get("last_close", 0.0) or 0.0),
                        float(row.get("pred_close", 0.0) or 0.0),
                        float(row.get("pred_return", 0.0) or 0.0),
                        float(row.get("bullish_score", 0.5) or 0.5),
                        float(row.get("confidence", 0.0) or 0.0),
                        _json(dict(row.get("metadata") or {})),
                        _now(),
                    )
                    for secid, row in rows.items()
                ],
            )

    def load_kronos_forecasts(
        self,
        *,
        as_of: str,
        model: str,
        params_key: str,
        secids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, Any]]:
        params: list[Any] = [as_of, model, params_key]
        where = "as_of = ? AND model = ? AND params_key = ?"
        if secids:
            placeholders = ",".join("?" for _ in secids)
            where += f" AND secid IN ({placeholders})"
            params.extend(secids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT secid, last_close, pred_close, pred_return, bullish_score,
                       confidence, metadata_json, created_at
                FROM kronos_forecasts
                WHERE {where}
                """,
                params,
            ).fetchall()
        out = {}
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"])
            except Exception:
                metadata = {}
            out[row["secid"]] = {
                "last_close": float(row["last_close"]),
                "pred_close": float(row["pred_close"]),
                "pred_return": float(row["pred_return"]),
                "bullish_score": float(row["bullish_score"]),
                "confidence": float(row["confidence"]),
                "metadata": metadata,
                "created_at": row["created_at"],
            }
        return out

    def save_kronos_exit_tracker(
        self,
        *,
        secid: str,
        side: str,
        created_at: str,
        last_updated_at: str,
        horizon: int,
        sample_count: int,
        current_step: int,
        planned_exit_at: str,
        confidence: float,
        state: Mapping[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO kronos_exit_trackers(
                    secid, side, created_at, last_updated_at, horizon, sample_count,
                    current_step, planned_exit_at, confidence, state_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(secid) DO UPDATE SET
                    side=excluded.side,
                    created_at=excluded.created_at,
                    last_updated_at=excluded.last_updated_at,
                    horizon=excluded.horizon,
                    sample_count=excluded.sample_count,
                    current_step=excluded.current_step,
                    planned_exit_at=excluded.planned_exit_at,
                    confidence=excluded.confidence,
                    state_json=excluded.state_json
                """,
                (
                    str(secid),
                    str(side),
                    str(created_at),
                    str(last_updated_at),
                    int(horizon),
                    int(sample_count),
                    int(current_step),
                    str(planned_exit_at or ""),
                    float(confidence),
                    _json(dict(state or {})),
                ),
            )

    def load_kronos_exit_tracker(self, secid: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT secid, side, created_at, last_updated_at, horizon, sample_count,
                       current_step, planned_exit_at, confidence, state_json
                FROM kronos_exit_trackers
                WHERE secid = ?
                """,
                (str(secid),),
            ).fetchone()
        if row is None:
            return None
        try:
            state = json.loads(row["state_json"])
        except Exception:
            state = {}
        return {
            "secid": row["secid"],
            "side": row["side"],
            "created_at": row["created_at"],
            "last_updated_at": row["last_updated_at"],
            "horizon": int(row["horizon"]),
            "sample_count": int(row["sample_count"]),
            "current_step": int(row["current_step"]),
            "planned_exit_at": row["planned_exit_at"],
            "confidence": float(row["confidence"]),
            "state": state,
        }

    def load_kronos_exit_trackers(self) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT secid, side, created_at, last_updated_at, horizon, sample_count,
                       current_step, planned_exit_at, confidence, state_json
                FROM kronos_exit_trackers
                """
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                state = json.loads(row["state_json"])
            except Exception:
                state = {}
            out[row["secid"]] = {
                "secid": row["secid"],
                "side": row["side"],
                "created_at": row["created_at"],
                "last_updated_at": row["last_updated_at"],
                "horizon": int(row["horizon"]),
                "sample_count": int(row["sample_count"]),
                "current_step": int(row["current_step"]),
                "planned_exit_at": row["planned_exit_at"],
                "confidence": float(row["confidence"]),
                "state": state,
            }
        return out

    def delete_kronos_exit_tracker(self, secid: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM kronos_exit_trackers WHERE secid = ?", (str(secid),))

    def load_selector_return_history(self, *, limit: int = 512) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT as_of, selector, return_value
                FROM selector_returns
                ORDER BY as_of DESC
                LIMIT ?
                """,
                (limit * 32,),
            ).fetchall()
        grouped: dict[str, dict[str, Any]] = {}
        for row in reversed(rows):
            item = grouped.setdefault(row["as_of"], {"as_of": row["as_of"], "returns": {}})
            item["returns"][row["selector"]] = float(row["return_value"])
        return list(grouped.values())[-limit:]

    def load_lightgbm_training_rows(self, *, limit: int = 512) -> list[dict[str, Any]]:
        history = {row["as_of"]: row for row in self.load_selector_return_history(limit=limit)}
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT as_of, features_json
                FROM market_features
                ORDER BY as_of DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for row in reversed(rows):
            ret = history.get(row["as_of"])
            if not ret:
                continue
            out.append({"as_of": row["as_of"], "features": json.loads(row["features_json"]), "returns": ret["returns"]})
        return out

    def load_paper_positions(self) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT secid, lots, weight, opened_at, updated_at FROM paper_positions").fetchall()
        return {
            row["secid"]: {
                "lots": int(row["lots"]),
                "weight": float(row["weight"]),
                "opened_at": row["opened_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    def upsert_paper_position(self, secid: str, lots: int, weight: float, as_of: str) -> None:
        with self.connect() as conn:
            if int(lots) == 0:
                conn.execute("DELETE FROM paper_positions WHERE secid = ?", (secid,))
                return
            existing = conn.execute("SELECT opened_at FROM paper_positions WHERE secid = ?", (secid,)).fetchone()
            opened_at = existing["opened_at"] if existing else as_of
            conn.execute(
                """
                INSERT INTO paper_positions(secid, lots, weight, opened_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(secid) DO UPDATE SET
                    lots=excluded.lots,
                    weight=excluded.weight,
                    updated_at=excluded.updated_at
                """,
                (secid, int(lots), float(weight), opened_at, as_of),
            )

    def load_account_state(self, account_id: str = "paper") -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT account_id, cash, equity, gross, net, margin_used,
                       available_cash, available_gross, cash_buffer, updated_at
                FROM account_state
                WHERE account_id = ?
                """,
                (account_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "account_id": row["account_id"],
            "cash": float(row["cash"]),
            "equity": float(row["equity"]),
            "gross": float(row["gross"]),
            "net": float(row["net"]),
            "margin_used": float(row["margin_used"]),
            "available_cash": float(row["available_cash"]),
            "available_gross": float(row["available_gross"]),
            "cash_buffer": float(row["cash_buffer"]),
            "updated_at": row["updated_at"],
        }

    def save_account_state(self, account: Mapping[str, Any], *, account_id: str = "paper", as_of: str | None = None) -> None:
        updated_at = as_of or _now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO account_state(
                    account_id, cash, equity, gross, net, margin_used,
                    available_cash, available_gross, cash_buffer, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    cash=excluded.cash,
                    equity=excluded.equity,
                    gross=excluded.gross,
                    net=excluded.net,
                    margin_used=excluded.margin_used,
                    available_cash=excluded.available_cash,
                    available_gross=excluded.available_gross,
                    cash_buffer=excluded.cash_buffer,
                    updated_at=excluded.updated_at
                """,
                (
                    account_id,
                    float(account.get("cash", 0.0) or 0.0),
                    float(account.get("equity", 0.0) or 0.0),
                    float(account.get("gross", 0.0) or 0.0),
                    float(account.get("net", 0.0) or 0.0),
                    float(account.get("margin_used", 0.0) or 0.0),
                    float(account.get("available_cash", 0.0) or 0.0),
                    float(account.get("available_gross", 0.0) or 0.0),
                    float(account.get("cash_buffer", 0.0) or 0.0),
                    updated_at,
                ),
            )

    def save_selector_positions(
        self,
        selector: str,
        weights: Mapping[str, float],
        prices: Mapping[str, float],
        as_of: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM selector_paper_positions WHERE selector = ?", (selector,))
            conn.executemany(
                """
                INSERT INTO selector_paper_positions(selector, secid, weight, entry_price, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (selector, secid, float(weight), float(prices.get(secid, 0.0)), as_of)
                    for secid, weight in weights.items()
                    if abs(float(weight)) > 1e-12 and float(prices.get(secid, 0.0)) > 0
                ],
            )

    def load_selector_positions(self) -> dict[str, dict[str, dict[str, float]]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT selector, secid, weight, entry_price, updated_at FROM selector_paper_positions"
            ).fetchall()
        out: dict[str, dict[str, dict[str, float]]] = {}
        for row in rows:
            out.setdefault(row["selector"], {})[row["secid"]] = {
                "weight": float(row["weight"]),
                "entry_price": float(row["entry_price"]),
                "updated_at": row["updated_at"],
            }
        return out


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
