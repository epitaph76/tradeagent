from __future__ import annotations

from arena_bot.storage import StateStore


def test_kronos_exit_tracker_round_trip_and_delete(tmp_path):
    state = StateStore(tmp_path / "state.sqlite3")

    assert state.load_kronos_exit_tracker("SBER") is None

    state.save_kronos_exit_tracker(
        secid="SBER",
        side="long",
        created_at="2026-06-03T12:00:00",
        last_updated_at="2026-06-03T12:00:00",
        horizon=8,
        sample_count=20,
        current_step=0,
        planned_exit_at="2026-06-03T15:00:00",
        confidence=0.9,
        state={"weights": [0.5, 0.5], "paths": [[{"close": 101.0}], [{"close": 99.0}]]},
    )

    tracker = state.load_kronos_exit_tracker("SBER")
    assert tracker is not None
    assert tracker["side"] == "long"
    assert tracker["horizon"] == 8
    assert tracker["sample_count"] == 20
    assert tracker["current_step"] == 0
    assert tracker["planned_exit_at"] == "2026-06-03T15:00:00"
    assert tracker["state"]["weights"] == [0.5, 0.5]
    assert "SBER" in state.load_kronos_exit_trackers()

    state.delete_kronos_exit_tracker("SBER")

    assert state.load_kronos_exit_tracker("SBER") is None
