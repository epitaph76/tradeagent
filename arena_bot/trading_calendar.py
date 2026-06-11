from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .storage import StateStore
from .types import Instrument, TradingSessionConfig

MOEX_BASE = "https://iss.moex.com/iss"
BINANCE_BASE = "https://api.binance.com"


@dataclass(frozen=True)
class InstrumentSessionState:
    secid: str
    venue: str
    boardid: str
    enabled: bool
    timezone: str
    trade_date: str
    local_as_of: str
    session_state: str
    session_type: str
    session_open: str
    entry_start: str
    new_entry_cutoff: str
    kronos_cutoff: str
    force_flat_time: str
    session_close: str
    session_open_at: str
    entry_start_at: str
    new_entry_cutoff_at: str
    kronos_cutoff_at: str
    force_flat_at: str
    session_close_at: str
    kronos_allowed: bool
    entry_allowed: bool
    exit_allowed: bool
    allow_new_trackers: bool
    force_flat_required: bool
    action_reason: str
    source: str
    status: str = ""
    is_traded: bool = True
    _entry_start_dt: datetime | None = None
    _kronos_cutoff_dt: datetime | None = None
    _force_flat_dt: datetime | None = None

    def to_runtime_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_public_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if not key.startswith("_")}


class SessionCalendarProvider:
    def session_state(
        self,
        as_of: datetime,
        instruments: Sequence[Instrument],
        *,
        candles: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class CacheFirstSessionCalendarProvider(SessionCalendarProvider):
    def __init__(self, *, config: TradingSessionConfig, state: StateStore | None = None):
        self.config = config
        self.state = state

    def session_state(
        self,
        as_of: datetime,
        instruments: Sequence[Instrument],
        *,
        candles: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not bool(self.config.enabled):
            by_secid = {
                instrument.secid: _disabled_state(as_of, instrument, self.config).to_runtime_dict()
                for instrument in instruments
            }
            return _aggregate_session(by_secid, self.config)

        rows_by_key = self._load_rows(as_of, instruments)
        by_secid: dict[str, dict[str, Any]] = {}
        for instrument in instruments:
            rows = rows_by_key.get((str(instrument.venue), instrument.secid), [])
            state = _state_for_instrument(
                as_of=as_of,
                instrument=instrument,
                config=self.config,
                cached_rows=rows,
                candles=(candles or {}).get(instrument.secid),
            )
            by_secid[instrument.secid] = state.to_runtime_dict()
        return _aggregate_session(by_secid, self.config)

    def _load_rows(
        self,
        as_of: datetime,
        instruments: Sequence[Instrument],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        if self.state is None or not bool(getattr(self.config, "calendar_cache_enabled", True)):
            return {}
        trade_dates = {_local_date(as_of, _template_timezone(self.config, str(instrument.venue))) for instrument in instruments}
        venues = sorted({str(instrument.venue) for instrument in instruments})
        secids = sorted({instrument.secid for instrument in instruments})
        out: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for trade_date in trade_dates:
            for row in self.state.load_trading_sessions(trade_date=trade_date.isoformat(), secids=secids, venues=venues):
                out.setdefault((str(row["venue"]), str(row["secid"])), []).append(row)
        return out


def sync_trading_sessions(
    *,
    config: TradingSessionConfig,
    state: StateStore,
    instruments: Sequence[Instrument],
    from_dt: datetime,
    till_dt: datetime,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_errors: dict[str, str] = {}
    try:
        rows.extend(_fetch_moex_session_rows(config=config, instruments=instruments, from_dt=from_dt, till_dt=till_dt))
    except Exception as exc:
        source_errors["moex"] = str(exc)
    try:
        rows.extend(_fetch_binance_status_rows(instruments=instruments, from_dt=from_dt, till_dt=till_dt))
    except Exception as exc:
        source_errors["binance"] = str(exc)
    if source_errors or not rows:
        rows.extend(_fallback_rows_for_range(config=config, instruments=instruments, from_dt=from_dt, till_dt=till_dt))
    state.save_trading_sessions(rows)
    return {"rows": len(rows), "source_errors": source_errors}


def session_force_flat_times(
    *,
    config: TradingSessionConfig,
    state: StateStore | None,
    instruments: Sequence[Instrument],
    from_dt: datetime,
    till_dt: datetime,
) -> list[datetime]:
    out: set[datetime] = set()
    current = from_dt.date()
    while current <= till_dt.date():
        cached_rows = state.load_trading_sessions(trade_date=current.isoformat()) if state is not None else []
        rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in cached_rows:
            rows_by_key.setdefault((str(row.get("venue")), str(row.get("secid"))), []).append(row)
        for instrument in instruments:
            venue = str(instrument.venue)
            if venue == "binance_spot":
                continue
            tz = ZoneInfo(_template_timezone(config, venue))
            rows = rows_by_key.get((venue, instrument.secid), [])
            if rows and any(str(row.get("session_type")) == "trade_day" and not bool(row.get("is_traded")) for row in rows):
                continue
            session_rows = [
                row
                for row in rows
                if bool(row.get("is_traded", False)) and str(row.get("time_from") or "") and str(row.get("time_till") or "")
            ]
            intervals = _intervals_from_rows(session_rows, current, tz) if session_rows else _template_intervals(config, venue, current, tz)
            if not intervals and _venue_weekday_only(venue) and current.weekday() >= 5:
                continue
            for _, _, session_close in intervals:
                dt = (session_close - timedelta(minutes=max(int(config.force_flat_minutes_before_close), 0))).replace(tzinfo=None)
                if from_dt <= dt <= till_dt:
                    out.add(dt)
        current += timedelta(days=1)
    return sorted(out)


def parse_moex_off_days(js: Mapping[str, Any], *, venue: str) -> list[dict[str, Any]]:
    rows = _iss_rows(js, "off_days")
    out = []
    for row in rows:
        trade_date = _first_value(row, "tradedate", "TRADEDATE", "date")
        if not trade_date:
            continue
        is_traded = _bool_int(_first_value(row, "is_traded", "IS_TRADED", "stock_workday", "futures_workday"))
        out.append(
            {
                "venue": venue,
                "secid": "*",
                "trade_date": str(trade_date)[:10],
                "session_type": "trade_day",
                "is_traded": is_traded,
                "source": f"{venue}_off_days",
                "raw": row,
            }
        )
    return out


def parse_moex_securities_boards(js: Mapping[str, Any], *, venue: str) -> list[dict[str, Any]]:
    rows = _iss_rows(js, "securities_boards")
    out = []
    for row in rows:
        trade_date = _first_value(row, "tradedate", "TRADEDATE")
        secid = _first_value(row, "secid", "SECID")
        boards = str(_first_value(row, "boards", "BOARDS") or "")
        if not trade_date or not secid:
            continue
        out.append(
            {
                "venue": venue,
                "secid": str(secid),
                "trade_date": str(trade_date)[:10],
                "session_type": "boards",
                "is_traded": bool(boards),
                "boardid": boards,
                "source": f"{venue}_securities_boards",
                "raw": row,
            }
        )
    return out


def parse_moex_suspended_details(js: Mapping[str, Any], *, venue: str) -> list[dict[str, Any]]:
    rows = _iss_rows(js, "suspended_details")
    out = []
    for row in rows:
        secid = _first_value(row, "secid", "SECID")
        trade_date = _first_value(row, "tradedate", "TRADEDATE", "date", "DATE")
        if not secid or not trade_date:
            continue
        out.append(
            {
                "venue": venue,
                "secid": str(secid),
                "trade_date": str(trade_date)[:10],
                "session_type": "suspended",
                "is_traded": False,
                "status": "SUSPENDED",
                "source": f"{venue}_suspended_details",
                "raw": row,
            }
        )
    return out


def parse_moex_futures_session(js: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _iss_rows(js, "session_schedule")
    out = []
    for row in rows:
        trade_date = _first_value(row, "tradedate", "TRADEDATE")
        secid = _first_value(row, "secid", "SECID")
        if not trade_date or not secid:
            continue
        out.append(
            {
                "venue": "moex_futures",
                "secid": str(secid),
                "trade_date": str(trade_date)[:10],
                "session_type": str(_first_value(row, "type", "TYPE") or "session"),
                "time_from": str(_first_value(row, "time_from", "TIME_FROM") or ""),
                "time_till": str(_first_value(row, "time_till", "TIME_TILL") or ""),
                "is_traded": True,
                "boardid": str(_first_value(row, "boardid", "BOARDID") or ""),
                "source": "moex_futures_session",
                "raw": row,
            }
        )
    return out


def parse_binance_exchange_info(js: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    symbols = js.get("symbols") if isinstance(js, Mapping) else None
    if not isinstance(symbols, list):
        return out
    today = datetime.now().date().isoformat()
    for row in symbols:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "")
        if not symbol:
            continue
        status = str(row.get("status") or "")
        out.append(
            {
                "venue": "binance_spot",
                "secid": symbol,
                "trade_date": today,
                "session_type": "status",
                "is_traded": status == "TRADING" and bool(row.get("isSpotTradingAllowed", True)),
                "status": status,
                "source": "binance_exchange_info",
                "raw": dict(row),
            }
        )
    return out


def _state_for_instrument(
    *,
    as_of: datetime,
    instrument: Instrument,
    config: TradingSessionConfig,
    cached_rows: Sequence[Mapping[str, Any]],
    candles: Any = None,
) -> InstrumentSessionState:
    venue = str(instrument.venue)
    if venue == "binance_spot":
        return _crypto_state(as_of=as_of, instrument=instrument, config=config, cached_rows=cached_rows, candles=candles)
    return _scheduled_state(as_of=as_of, instrument=instrument, config=config, cached_rows=cached_rows)


def _crypto_state(
    *,
    as_of: datetime,
    instrument: Instrument,
    config: TradingSessionConfig,
    cached_rows: Sequence[Mapping[str, Any]],
    candles: Any = None,
) -> InstrumentSessionState:
    tz = ZoneInfo(_template_timezone(config, "binance_spot"))
    local_as_of = _as_local(as_of, tz)
    status_rows = [row for row in cached_rows if str(row.get("session_type")) == "status"]
    status = str(status_rows[-1].get("status") or "TRADING") if status_rows else "TRADING"
    is_traded = status == "TRADING" and _has_historical_crypto_candle(candles, local_as_of)
    if not is_traded:
        return _closed_state(
            as_of=local_as_of,
            instrument=instrument,
            config=config,
            reason="binance_symbol_not_trading" if status != "TRADING" else "binance_history_missing_candle",
            source=str(status_rows[-1].get("source") or "binance_exchange_info") if status_rows else "crypto_24_7",
            status=status,
            is_traded=False,
        )
    start = datetime.combine(local_as_of.date(), time(0, 0), tzinfo=tz)
    close = datetime.combine(local_as_of.date(), time(23, 59, 59), tzinfo=tz)
    return _active_state(
        as_of=local_as_of,
        instrument=instrument,
        config=config,
        session_type="crypto_24_7",
        session_open=start,
        session_close=close,
        source=str(status_rows[-1].get("source") or "crypto_24_7") if status_rows else "crypto_24_7",
        status=status,
        is_traded=True,
        force_flat_enabled=False,
    )


def _scheduled_state(
    *,
    as_of: datetime,
    instrument: Instrument,
    config: TradingSessionConfig,
    cached_rows: Sequence[Mapping[str, Any]],
) -> InstrumentSessionState:
    venue = str(instrument.venue)
    tz = ZoneInfo(_template_timezone(config, venue))
    local_as_of = _as_local(as_of, tz)
    trade_date = local_as_of.date()
    source = "fallback_template"
    if cached_rows:
        source = ",".join(sorted({str(row.get("source") or "cache") for row in cached_rows}))
        if any(str(row.get("session_type")) == "suspended" for row in cached_rows):
            return _closed_state(
                as_of=local_as_of,
                instrument=instrument,
                config=config,
                reason="session_suspended",
                source=source,
                status="SUSPENDED",
                is_traded=False,
            )
        if any(str(row.get("session_type")) == "trade_day" and not bool(row.get("is_traded")) for row in cached_rows):
            return _closed_state(
                as_of=local_as_of,
                instrument=instrument,
                config=config,
                reason="session_non_trading_day",
                source=source,
                is_traded=False,
            )

    session_rows = [
        row
        for row in cached_rows
        if bool(row.get("is_traded", False)) and str(row.get("time_from") or "") and str(row.get("time_till") or "")
    ]
    intervals = _intervals_from_rows(session_rows, trade_date, tz) if session_rows else _template_intervals(config, venue, trade_date, tz)
    if not intervals and not cached_rows and _venue_weekday_only(venue) and local_as_of.weekday() >= 5:
        return _closed_state(
            as_of=local_as_of,
            instrument=instrument,
            config=config,
            reason="session_non_trading_day",
            source=source,
            is_traded=False,
        )
    if not intervals:
        return _closed_state(
            as_of=local_as_of,
            instrument=instrument,
            config=config,
            reason="session_no_intervals",
            source=source,
            is_traded=False,
        )

    for session_type, session_open, session_close in intervals:
        if session_open <= local_as_of < session_close:
            return _active_state(
                as_of=local_as_of,
                instrument=instrument,
                config=config,
                session_type=session_type,
                session_open=session_open,
                session_close=session_close,
                source=source,
                status="TRADING",
                is_traded=True,
                force_flat_enabled=True,
            )

    if local_as_of < intervals[0][1]:
        reason = "session_pre_open"
        state = "pre_session"
    elif any(prev_close <= local_as_of < next_open for (_, _, prev_close), (_, next_open, _) in zip(intervals, intervals[1:])):
        reason = "session_between_sessions"
        state = "closed"
    else:
        reason = "session_closed"
        state = "closed"
    first_type, first_open, first_close = intervals[0]
    return _inactive_interval_state(
        as_of=local_as_of,
        instrument=instrument,
        config=config,
        session_type=first_type,
        session_open=first_open,
        session_close=first_close,
        state=state,
        reason=reason,
        source=source,
    )


def _active_state(
    *,
    as_of: datetime,
    instrument: Instrument,
    config: TradingSessionConfig,
    session_type: str,
    session_open: datetime,
    session_close: datetime,
    source: str,
    status: str,
    is_traded: bool,
    force_flat_enabled: bool,
) -> InstrumentSessionState:
    venue = str(instrument.venue)
    entry_start = _entry_start(config, as_of, session_open, force_flat_enabled=force_flat_enabled)
    new_entry_cutoff = session_close - timedelta(minutes=max(int(config.new_entry_cutoff_minutes_before_close), 0))
    kronos_cutoff = new_entry_cutoff
    force_flat_at = session_close - timedelta(minutes=max(int(config.force_flat_minutes_before_close), 0))
    if not force_flat_enabled:
        entry_start = as_of.replace(hour=0, minute=0, second=0, microsecond=0)
        new_entry_cutoff = session_close
        kronos_cutoff = session_close
        force_flat_at = session_close

    if as_of < entry_start:
        state = "warmup"
        reason = "session_warmup_no_entry"
    elif as_of < new_entry_cutoff:
        state = "trade"
        reason = ""
    elif force_flat_enabled and as_of < force_flat_at:
        state = "no_new_entries"
        reason = "session_entry_cutoff"
    elif force_flat_enabled and as_of < session_close:
        state = "force_flat"
        reason = "session_force_flat"
    else:
        state = "trade" if not force_flat_enabled else "closed"
        reason = "" if not force_flat_enabled else "session_closed"

    return InstrumentSessionState(
        secid=instrument.secid,
        venue=venue,
        boardid=str(instrument.boardid or ""),
        enabled=True,
        timezone=str(session_open.tzinfo),
        trade_date=as_of.date().isoformat(),
        local_as_of=as_of.isoformat(timespec="seconds"),
        session_state=state,
        session_type=session_type,
        session_open=_hhmm(session_open),
        entry_start=_hhmm(entry_start),
        new_entry_cutoff=_hhmm(new_entry_cutoff),
        kronos_cutoff=_hhmm(kronos_cutoff),
        force_flat_time=_hhmm(force_flat_at),
        session_close=_hhmm(session_close),
        session_open_at=session_open.isoformat(timespec="seconds"),
        entry_start_at=entry_start.isoformat(timespec="seconds"),
        new_entry_cutoff_at=new_entry_cutoff.isoformat(timespec="seconds"),
        kronos_cutoff_at=kronos_cutoff.isoformat(timespec="seconds"),
        force_flat_at=force_flat_at.isoformat(timespec="seconds"),
        session_close_at=session_close.isoformat(timespec="seconds"),
        kronos_allowed=state == "trade",
        entry_allowed=state == "trade",
        exit_allowed=state in {"trade", "no_new_entries"},
        allow_new_trackers=state == "trade",
        force_flat_required=force_flat_enabled and state == "force_flat",
        action_reason=reason,
        source=source,
        status=status,
        is_traded=is_traded,
        _entry_start_dt=entry_start,
        _kronos_cutoff_dt=kronos_cutoff if force_flat_enabled else None,
        _force_flat_dt=force_flat_at if force_flat_enabled else None,
    )


def _inactive_interval_state(
    *,
    as_of: datetime,
    instrument: Instrument,
    config: TradingSessionConfig,
    session_type: str,
    session_open: datetime,
    session_close: datetime,
    state: str,
    reason: str,
    source: str,
) -> InstrumentSessionState:
    active = _active_state(
        as_of=session_open,
        instrument=instrument,
        config=config,
        session_type=session_type,
        session_open=session_open,
        session_close=session_close,
        source=source,
        status="CLOSED",
        is_traded=True,
        force_flat_enabled=True,
    )
    data = active.to_runtime_dict()
    data.update(
        {
            "local_as_of": as_of.isoformat(timespec="seconds"),
            "session_state": state,
            "kronos_allowed": False,
            "entry_allowed": False,
            "exit_allowed": False,
            "allow_new_trackers": False,
            "force_flat_required": False,
            "action_reason": reason,
            "status": "CLOSED",
        }
    )
    return InstrumentSessionState(**data)


def _closed_state(
    *,
    as_of: datetime,
    instrument: Instrument,
    config: TradingSessionConfig,
    reason: str,
    source: str,
    status: str = "CLOSED",
    is_traded: bool = False,
) -> InstrumentSessionState:
    tz = as_of.tzinfo or ZoneInfo(str(config.timezone or "Europe/Moscow"))
    close = datetime.combine(as_of.date(), time(0, 0), tzinfo=tz)
    return InstrumentSessionState(
        secid=instrument.secid,
        venue=str(instrument.venue),
        boardid=str(instrument.boardid or ""),
        enabled=True,
        timezone=str(tz),
        trade_date=as_of.date().isoformat(),
        local_as_of=as_of.isoformat(timespec="seconds"),
        session_state="closed",
        session_type="closed",
        session_open="",
        entry_start="",
        new_entry_cutoff="",
        kronos_cutoff="",
        force_flat_time="",
        session_close="",
        session_open_at="",
        entry_start_at="",
        new_entry_cutoff_at="",
        kronos_cutoff_at="",
        force_flat_at="",
        session_close_at="",
        kronos_allowed=False,
        entry_allowed=False,
        exit_allowed=False,
        allow_new_trackers=False,
        force_flat_required=False,
        action_reason=reason,
        source=source,
        status=status,
        is_traded=is_traded,
        _entry_start_dt=close,
        _kronos_cutoff_dt=close,
        _force_flat_dt=close,
    )


def _disabled_state(as_of: datetime, instrument: Instrument, config: TradingSessionConfig) -> InstrumentSessionState:
    tz = ZoneInfo(str(config.timezone or "Europe/Moscow"))
    local_as_of = _as_local(as_of, tz)
    return InstrumentSessionState(
        secid=instrument.secid,
        venue=str(instrument.venue),
        boardid=str(instrument.boardid or ""),
        enabled=False,
        timezone=str(config.timezone),
        trade_date=local_as_of.date().isoformat(),
        local_as_of=local_as_of.isoformat(timespec="seconds"),
        session_state="disabled",
        session_type="disabled",
        session_open=str(config.session_open),
        entry_start=str(config.entry_start),
        new_entry_cutoff=str(config.new_entry_cutoff),
        kronos_cutoff=str(config.kronos_cutoff),
        force_flat_time=str(config.force_flat_time),
        session_close=str(config.session_close),
        session_open_at="",
        entry_start_at="",
        new_entry_cutoff_at="",
        kronos_cutoff_at="",
        force_flat_at="",
        session_close_at="",
        kronos_allowed=True,
        entry_allowed=True,
        exit_allowed=True,
        allow_new_trackers=True,
        force_flat_required=False,
        action_reason="",
        source="disabled",
    )


def _aggregate_session(by_secid: Mapping[str, Mapping[str, Any]], config: TradingSessionConfig) -> dict[str, Any]:
    public_by_secid = {
        secid: {key: value for key, value in dict(row).items() if not str(key).startswith("_")}
        for secid, row in by_secid.items()
    }
    states = [dict(row) for row in by_secid.values()]
    force_flat = sorted(row["secid"] for row in states if bool(row.get("force_flat_required")))
    entryable = sorted(row["secid"] for row in states if bool(row.get("entry_allowed")))
    exitable = sorted(row["secid"] for row in states if bool(row.get("exit_allowed")))
    if force_flat:
        state = "force_flat"
        reason = "session_force_flat"
    elif entryable:
        state = "trade"
        reason = ""
    elif states:
        state = str(states[0].get("session_state") or "closed")
        reason = str(states[0].get("action_reason") or "session_closed")
    else:
        state = "closed"
        reason = "session_no_instruments"
    first = states[0] if states else {}
    return {
        "enabled": bool(config.enabled),
        "timezone": str(config.timezone),
        "session_state": state,
        "local_as_of": first.get("local_as_of", ""),
        "session_open": first.get("session_open", str(config.session_open)),
        "entry_start": first.get("entry_start", str(config.entry_start)),
        "new_entry_cutoff": first.get("new_entry_cutoff", str(config.new_entry_cutoff)),
        "kronos_cutoff": first.get("kronos_cutoff", str(config.kronos_cutoff)),
        "force_flat_time": first.get("force_flat_time", str(config.force_flat_time)),
        "session_close": first.get("session_close", str(config.session_close)),
        "session_by_secid": public_by_secid,
        "_session_by_secid": dict(by_secid),
        "entryable_secids": entryable,
        "exitable_secids": exitable,
        "force_flat_secids": force_flat,
        "kronos_allowed": bool(entryable),
        "entry_allowed": bool(entryable),
        "exit_allowed": bool(exitable),
        "allow_new_trackers": bool(entryable),
        "force_flat_required": bool(force_flat),
        "flat_all_asset_classes": bool(getattr(config, "flat_all_asset_classes", False)),
        "action_reason": reason,
        "_entry_start_dt": first.get("_entry_start_dt"),
        "_kronos_cutoff_dt": first.get("_kronos_cutoff_dt"),
        "_force_flat_dt": first.get("_force_flat_dt"),
    }


def _fetch_moex_session_rows(
    *,
    config: TradingSessionConfig,
    instruments: Sequence[Instrument],
    from_dt: datetime,
    till_dt: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for venue, endpoint in (("moex_stock", "stock"), ("moex_futures", "futures")):
        venue_instruments = [instrument for instrument in instruments if str(instrument.venue) == venue]
        if not venue_instruments:
            continue
        url = f"{MOEX_BASE}/calendars/{endpoint}.json"
        js = _get_json(
            url,
            params={
                "iss.meta": "off",
                "show_all_days": 1,
                "from": from_dt.date().isoformat(),
                "till": till_dt.date().isoformat(),
            },
        )
        day_rows = parse_moex_off_days(js, venue=venue)
        traded_by_date = {row["trade_date"]: bool(row["is_traded"]) for row in day_rows}
        current = from_dt.date()
        while current <= till_dt.date():
            is_traded = traded_by_date.get(current.isoformat(), current.weekday() < 5)
            for instrument in venue_instruments:
                rows.extend(
                    _session_cache_rows_for_day(
                        config=config,
                        instrument=instrument,
                        day=current,
                        source=f"{venue}_off_days" if current.isoformat() in traded_by_date else "fallback_template",
                        is_traded=is_traded,
                        raw={"calendar": traded_by_date.get(current.isoformat())},
                    )
                )
            current += timedelta(days=1)
    return rows


def _fetch_binance_status_rows(
    *,
    instruments: Sequence[Instrument],
    from_dt: datetime,
    till_dt: datetime,
) -> list[dict[str, Any]]:
    symbols = [instrument.secid for instrument in instruments if str(instrument.venue) == "binance_spot"]
    if not symbols:
        return []
    js = _get_json(f"{BINANCE_BASE}/api/v3/exchangeInfo", params={"symbols": _json_symbols(symbols)})
    parsed = parse_binance_exchange_info(js)
    by_symbol = {row["secid"]: row for row in parsed}
    rows = []
    current = from_dt.date()
    while current <= till_dt.date():
        for symbol in symbols:
            base = dict(by_symbol.get(symbol) or {})
            if not base:
                continue
            base["trade_date"] = current.isoformat()
            rows.append(base)
        current += timedelta(days=1)
    return rows


def _fallback_rows_for_range(
    *,
    config: TradingSessionConfig,
    instruments: Sequence[Instrument],
    from_dt: datetime,
    till_dt: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current = from_dt.date()
    while current <= till_dt.date():
        for instrument in instruments:
            is_traded = current.weekday() < 5 if _venue_weekday_only(str(instrument.venue)) else True
            rows.extend(
                _session_cache_rows_for_day(
                    config=config,
                    instrument=instrument,
                    day=current,
                    source="fallback_template",
                    is_traded=is_traded,
                    raw={},
                )
            )
        current += timedelta(days=1)
    return rows


def _session_cache_rows_for_day(
    *,
    config: TradingSessionConfig,
    instrument: Instrument,
    day: date,
    source: str,
    is_traded: bool,
    raw: Mapping[str, Any],
) -> list[dict[str, Any]]:
    venue = str(instrument.venue)
    if venue == "binance_spot":
        return [
            {
                "venue": venue,
                "secid": instrument.secid,
                "trade_date": day.isoformat(),
                "session_type": "status",
                "is_traded": True,
                "status": "TRADING",
                "source": source,
                "raw": dict(raw),
            }
        ]
    tz = ZoneInfo(_template_timezone(config, venue))
    if not is_traded:
        return [
            {
                "venue": venue,
                "secid": instrument.secid,
                "trade_date": day.isoformat(),
                "session_type": "trade_day",
                "is_traded": False,
                "boardid": str(instrument.boardid or ""),
                "source": source,
                "raw": dict(raw),
            }
        ]
    return [
        {
            "venue": venue,
            "secid": instrument.secid,
            "trade_date": day.isoformat(),
            "session_type": session_type,
            "time_from": _hhmm(start),
            "time_till": _hhmm(end),
            "is_traded": True,
            "boardid": str(instrument.boardid or ""),
            "source": source,
            "raw": dict(raw),
        }
        for session_type, start, end in _template_intervals(config, venue, day, tz)
    ]


def _template_intervals(config: TradingSessionConfig, venue: str, day: date, tz: ZoneInfo) -> list[tuple[str, datetime, datetime]]:
    template = _venue_template(config, venue)
    sessions = template.get("sessions") if isinstance(template, Mapping) else None
    if not sessions:
        sessions = [{"type": "main", "open": config.session_open, "close": config.session_close}]
    out = []
    for idx, row in enumerate(sessions):
        if not isinstance(row, Mapping):
            continue
        session_type = str(row.get("type") or f"session_{idx + 1}")
        start = datetime.combine(day, _parse_time(row.get("open") or row.get("time_from") or config.session_open), tzinfo=tz)
        end = datetime.combine(day, _parse_time(row.get("close") or row.get("time_till") or config.session_close), tzinfo=tz)
        if end <= start:
            end += timedelta(days=1)
        out.append((session_type, start, end))
    return sorted(out, key=lambda item: item[1])


def _intervals_from_rows(rows: Sequence[Mapping[str, Any]], day: date, tz: ZoneInfo) -> list[tuple[str, datetime, datetime]]:
    out = []
    for row in rows:
        try:
            start = datetime.combine(day, _parse_time(row.get("time_from")), tzinfo=tz)
            end = datetime.combine(day, _parse_time(row.get("time_till")), tzinfo=tz)
        except Exception:
            continue
        if end <= start:
            end += timedelta(days=1)
        out.append((str(row.get("session_type") or "session"), start, end))
    return sorted(out, key=lambda item: item[1])


def _entry_start(config: TradingSessionConfig, as_of: datetime, session_open: datetime, *, force_flat_enabled: bool) -> datetime:
    if not force_flat_enabled:
        return session_open
    warmup = session_open + timedelta(minutes=max(int(config.entry_warmup_minutes_after_open), 0))
    try:
        legacy = datetime.combine(as_of.date(), _parse_time(config.entry_start), tzinfo=session_open.tzinfo)
    except Exception:
        legacy = warmup
    return max(warmup, legacy)


def _has_historical_crypto_candle(candles: Any, as_of: datetime) -> bool:
    if candles is None or not hasattr(candles, "columns"):
        return True
    if "timestamps" not in candles.columns:
        return True
    if getattr(candles, "empty", True):
        return False
    ts = pd.to_datetime(candles["timestamps"], errors="coerce").dropna()
    if ts.empty:
        return False
    latest = pd.Timestamp(ts.iloc[-1]).to_pydatetime()
    latest = latest.replace(tzinfo=as_of.tzinfo) if latest.tzinfo is None else latest.astimezone(as_of.tzinfo)
    return latest <= as_of and (as_of - latest) <= timedelta(hours=25)


def _get_json(url: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
    response = requests.get(url, params=dict(params), timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "json" not in content_type and response.text.lstrip().startswith("<"):
        raise RuntimeError(f"non-json response from {url}")
    return response.json()


def _iss_rows(js: Mapping[str, Any], block_name: str) -> list[dict[str, Any]]:
    block = js.get(block_name) if isinstance(js, Mapping) else None
    if not isinstance(block, Mapping):
        return []
    columns = block.get("columns")
    data = block.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        return []
    return [dict(zip(columns, row)) for row in data if isinstance(row, list)]


def _first_value(row: Mapping[str, Any], *keys: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row:
            return row[key]
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _bool_int(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return int(value) != 0
    except Exception:
        return str(value).strip().lower() in {"true", "yes", "y", "on"}


def _venue_template(config: TradingSessionConfig, venue: str) -> Mapping[str, Any]:
    raw = getattr(config, "session_templates", {}) or {}
    if isinstance(raw, Mapping):
        template = raw.get(venue)
        if isinstance(template, Mapping):
            return template
    return {}


def _template_timezone(config: TradingSessionConfig, venue: str) -> str:
    template = _venue_template(config, venue)
    return str(template.get("timezone") or config.timezone or "Europe/Moscow")


def _venue_weekday_only(venue: str) -> bool:
    return venue in {"moex_stock", "moex_futures"}


def _local_date(value: datetime, timezone: str) -> date:
    return _as_local(value, ZoneInfo(timezone)).date()


def _as_local(value: datetime, tz: ZoneInfo) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=tz)
    return value.astimezone(tz)


def _parse_time(value: Any) -> time:
    raw = str(value or "00:00").strip()
    parts = raw.split(":")
    if len(parts) < 2:
        raise ValueError(f"invalid time: {value}")
    return time(hour=int(parts[0]), minute=int(parts[1]), second=int(parts[2]) if len(parts) > 2 else 0)


def _hhmm(value: datetime) -> str:
    if value.second:
        return value.strftime("%H:%M:%S")
    return value.strftime("%H:%M")


def _json_symbols(symbols: Sequence[str]) -> str:
    return "[" + ",".join(f'"{symbol}"' for symbol in symbols) + "]"
