import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import espn

JS = json.loads((Path(__file__).parent / "fixtures" / "espn_league.json").read_text())
CATS = json.loads((Path(__file__).parent / "fixtures" / "espn_league_cats.json").read_text())  # live 16-team H2H categories, trimmed
OLD = {"season": 2027, "my_slot": 1, "adds_per_day": 3, "strategy": "greedy", "slots": {"G": 9}}
SCHED = pd.DataFrame({"team_id": [1, 2], "date": ["2026-10-20", "2026-10-21"], "week": [1, 1]})


def test_auth(tmp_path):
    assert espn.auth(tmp_path) is None
    (tmp_path / "espn_auth.json").write_text('{"espn_s2": "a", "SWID": "{B}"}')
    assert espn.auth(tmp_path) == {"espn_s2": "a", "SWID": "{B}"}


def test_league_repeats_view_and_sends_cookies(monkeypatch):
    seen = {}
    monkeypatch.setattr(espn.fetch, "get", lambda url, cookies=None: seen.update(url=url, cookies=cookies) or JS)
    assert espn.league(["mSettings", "mTeam"], 7, 2027, {"c": 1}) is JS
    assert seen["url"].endswith("/seasons/2027/segments/0/leagues/7?view=mSettings&view=mTeam")
    assert seen["cookies"] == {"c": 1}


def test_settings_from_league():
    s = espn.settings_from_league(JS, OLD, "{test-swid}")
    assert s["slots"] == {"G": 2, "F": 1, "SG/SF": 1, "PF/C": 1, "UTIL": 1, "BE": 6, "IR": 2}
    assert (s["teams"], s["rounds"], s["my_slot"], s["espn_team_id"], s["my_team_name"]) == (14, 12, 9, 13, "Raj")
    assert (s["playoff_teams"], s["playoff_round_periods"], s["playoff_weeks"], s["league_id"]) == (4, 2, 4, 178869420)
    assert len(s["scoring"]) == 19 and s["scoring"]["FGMI"] == -0.5 and s["scoring"]["QD"] == 100 and s["scoring"]["3PM"] == 2
    assert s["strategy"] == "greedy" and OLD["slots"] == {"G": 9}
    assert s["format"] == "points" and "categories" not in s and "adds_per_day" not in s
    assert (s["adds_per_week"], s["pick_seconds"], s["draft_time"]) == (3, 75, datetime.fromtimestamp(1790825400).isoformat(timespec="minutes"))


def test_settings_from_league_categories():
    old = {**OLD, "scoring": {"PTS": 1.0}, "adds_per_day": 2}
    s = espn.settings_from_league(CATS, old, "{test-swid}")
    assert s["format"] == "categories" and "scoring" not in s and "adds_per_day" not in s
    assert s["categories"] == [{"stat": n, "reverse": r} for n, r in [
        ("FT%", False), ("OREB", False), ("3P%", False), ("DREB", False), ("TW", False), ("PF", True),
        ("PTS", False), ("BLK", False), ("STL", False), ("TO", True), ("AST", False), ("FG%", False)]]
    assert s["slots"] == {"G": 2, "F": 1, "SG/SF": 2, "PF/C": 1, "UTIL": 1, "BE": 4, "IR": 2}
    assert (s["teams"], s["rounds"], s["my_slot"], s["espn_team_id"], s["my_team_name"]) == (16, 11, 8, 13, "Raj")
    assert (s["adds_per_week"], s["pick_seconds"], s["draft_time"]) == (4, 60, datetime.fromtimestamp(1790825400).isoformat(timespec="minutes"))
    assert "scoring" in old  # input untouched


def test_settings_from_league_stale_keys_follow_format():
    assert "categories" not in espn.settings_from_league(JS, {**OLD, "categories": [], "scoring": {}}, None)
    assert espn.settings_from_league(CATS, {**OLD, "adds_per_day": 2}, None)["adds_per_week"] == 4
    nodate = json.loads(json.dumps(CATS))
    del nodate["settings"]["draftSettings"]["date"]
    assert espn.settings_from_league(nodate, {**OLD, "draft_time": "x"}, None)["draft_time"] == "x"


def test_settings_from_league_unknown_category_raises():
    js = json.loads(json.dumps(CATS))
    js["settings"]["scoringSettings"]["scoringItems"].append({"statId": 99, "points": 1.0})
    with pytest.raises(ValueError, match="99"):
        espn.settings_from_league(js, OLD, None)


def test_settings_from_league_defaults_and_no_swid():
    s = espn.settings_from_league(JS, {**OLD, "espn_team_id": 5, "adds_per_day": 1}, None)
    assert s["espn_team_id"] == 5 and s["my_slot"] == 2  # kept id, slot re-derived from pickOrder
    assert espn.settings_from_league(JS, {"season": 2027}, "{TEST-SWID}")["adds_per_week"] == 3


def test_matchups_use_espn_calendar():
    m = espn.matchups(JS, SCHED)
    assert list(m.columns) == ["date", "matchup", "playoff"]
    assert m.iloc[0].tolist() == ["2026-10-20", 1, False]
    assert m[m.matchup == 20].date.tolist() == ["2026-11-08", "2026-11-09"]
    assert m[m.playoff].matchup.tolist() == [20, 20, 21, 21] and not m[m.matchup == 19].playoff.any()


def test_is_real_calendar():
    assert not espn.is_real_calendar(espn.matchups(JS, SCHED), JS)  # fixture = 1-day-per-matchup placeholder
    js = json.loads(json.dumps(JS))
    js["settings"]["scheduleSettings"].update(matchupPeriodCount=1, matchupPeriods={"1": list(range(1, 15))})
    assert espn.is_real_calendar(espn.matchups(js, SCHED), js)


def test_rosters():
    df = espn.rosters(JS)
    assert list(df.columns) == ["team_id", "team_name", "player_id", "lineup_slot"]
    assert df[df.team_id == 13].lineup_slot.tolist() == ["G", "BE", "IR"]
    assert df[df.team_id == 5].iloc[0].tolist() == [5, "Team Five", 3945274, "UTIL"]


def test_scores():
    assert espn.scores(JS, 1, 13) == {"me": 123.0, "opp": 98.5, "opp_id": 5}
    assert espn.scores(JS, 2, 13) == {"me": 10.0, "opp": 20.0, "opp_id": 1}
    assert espn.scores(JS, 3, 13) is None


def run_main(tmp_path, monkeypatch, js):
    (tmp_path / "data").mkdir(exist_ok=True)
    SCHED.to_csv(tmp_path / "data" / "schedule.csv", index=False)
    (tmp_path / "settings.json").write_text(json.dumps({**OLD, "league_id": 1}))
    monkeypatch.setattr(espn, "ROOT", tmp_path)
    monkeypatch.setattr(espn, "DATA", tmp_path / "data")
    monkeypatch.setattr(espn, "auth", lambda home: {"espn_s2": "x", "SWID": "{TEST-SWID}"})
    monkeypatch.setattr(espn, "league", lambda *a: js)
    espn.main()
    return json.loads((tmp_path / "settings.json").read_text())


def test_main_writes_settings_and_matchups(tmp_path, monkeypatch):
    weekly = json.loads(json.dumps(JS))
    sch = weekly["settings"]["scheduleSettings"]
    sch["matchupPeriodCount"], sch["matchupPeriods"] = 2, {"1": list(range(1, 8)), "2": list(range(8, 15)), "3": [15, 16]}
    s = run_main(tmp_path, monkeypatch, weekly)
    assert s["my_slot"] == 9 and s["rounds"] == 12 and not (tmp_path / "settings.tmp").exists()
    assert len(pd.read_csv(tmp_path / "data" / "matchups.csv")) == 16


def test_main_skips_placeholder_calendar(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "matchups.csv").write_text("stale")
    run_main(tmp_path, monkeypatch, JS)  # fixture = ESPN's 1-day-per-matchup preseason calendar
    assert not (tmp_path / "data" / "matchups.csv").exists()


def test_main_syncs_category_league(tmp_path, monkeypatch):
    s = run_main(tmp_path, monkeypatch, CATS)
    assert (s["format"], s["teams"], s["rounds"], s["my_slot"]) == ("categories", 16, 11, 8)
    assert len(s["categories"]) == 12 and "scoring" not in s
