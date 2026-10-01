import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "projections.csv").exists(),
                                reason="needs data: run fetch.py and model.py first")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated DRAFT_HOME: copies of settings.json + the two CSVs app.py needs. Tests never touch
    the repo's real settings.json or data/draft.json (M8)."""
    (tmp_path / "data").mkdir()
    shutil.copy(ROOT / "data" / "projections.csv", tmp_path / "data" / "projections.csv")
    shutil.copy(ROOT / "data" / "schedule.csv", tmp_path / "data" / "schedule.csv")
    shutil.copy(ROOT / "settings.json", tmp_path / "settings.json")
    monkeypatch.setenv("DRAFT_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def at(home):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def saved_picks(home):
    p = home / "data" / "draft.json"
    return json.loads(p.read_text()) if p.exists() else []


def test_loads_with_recommendations(at):
    assert not at.exception
    assert len(at.dataframe) >= 1


def test_draft_unlisted_and_undo(at, home):
    pid = int(pd.read_csv(home / "data" / "projections.csv").player_id.iloc[0])
    at.selectbox(key="pick_0").set_value(pid).run()
    at.button(key="drafted").click().run()
    assert saved_picks(home) == [pid] and not at.exception
    at.selectbox(key="pick_1").set_value(-1).run()
    at.button(key="drafted").click().run()
    assert saved_picks(home) == [pid, None]
    at.button(key="undo").click().run()
    assert saved_picks(home) == [pid] and not at.exception


def test_draft_tab_follows_draft_json_written_by_the_extension(at, home):
    pid = int(pd.read_csv(home / "data" / "projections.csv").player_id.iloc[0])
    assert "pick 1:" in at.header[0].value
    (home / "data" / "draft.json").write_text(json.dumps([pid]))  # what server.py writes
    at.run()
    assert "pick 2:" in at.header[0].value
    at.selectbox(key="pick_1").set_value(-1).run()
    at.button(key="drafted").click().run()
    assert saved_picks(home) == [pid, None]  # the app appends to what the server wrote


def test_changing_team_count_keeps_working(at):
    at.selectbox(key="teams").set_value(10).run()
    assert not at.exception
    assert at.selectbox(key="teams").value == 10


def test_reducing_teams_below_playoff_teams_no_error(at):
    """Reducing "Teams" below "Playoff teams" should not error, and the widget should reflect it."""
    at.selectbox(key="teams").set_value(12).run()
    assert not at.exception
    at.selectbox(key="teams").set_value(8).run()
    assert not at.exception
    assert at.selectbox(key="teams").value == 8


def test_missing_data_shows_error_message(tmp_path, monkeypatch):
    """Fresh clone with settings.json but no data/ shows the fetch.py message, not a traceback."""
    from streamlit.testing.v1 import AppTest
    shutil.copy(ROOT / "settings.json", tmp_path / "settings.json")
    monkeypatch.setenv("DRAFT_HOME", str(tmp_path))
    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at2.exception
    assert any("fetch.py" in e.value for e in at2.error)


def test_draft_persists_across_restart(at, home):
    """Restarting the app mid-draft reloads picks from data/draft.json."""
    pid = int(pd.read_csv(home / "data" / "projections.csv").player_id.iloc[0])
    at.selectbox(key="pick_0").set_value(pid).run()
    at.button(key="drafted").click().run()
    assert saved_picks(home) == [pid] and not at.exception

    from streamlit.testing.v1 import AppTest
    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert saved_picks(home) == [pid] and not at2.exception


def test_my_slot_persists_across_restart(at, home):
    """F1: changing "My draft slot" autosaves settings.json, so a fresh session picks it up."""
    at.selectbox(key="my_slot").set_value(3).run()
    assert not at.exception

    from streamlit.testing.v1 import AppTest
    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at2.exception
    assert at2.selectbox(key="my_slot").value == 3


def test_season_tab_without_espn_auth_explains_setup(at):
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Draft", "Season"]
    assert any("espn_auth.json" in i.value for i in at.info)


def test_sync_button_reloads_synced_settings(at, home, monkeypatch):
    """Sync overwrites settings.json; the sidebar widgets must pick the new values up, not autosave the old ones back."""
    import subprocess
    from types import SimpleNamespace

    def fake_sync(cmd, **kw):
        s = json.loads((home / "settings.json").read_text())
        (home / "settings.json").write_text(json.dumps({**s, "teams": 12, "my_slot": 5, "slots": {**s["slots"], "G": 3}}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_sync)
    at.button(key="sync").click().run()
    assert not at.exception
    assert (at.selectbox(key="teams").value, at.selectbox(key="my_slot").value, at.number_input(key="slot_G").value) == (12, 5, 3)
    assert json.loads((home / "settings.json").read_text())["teams"] == 12


def test_sync_failure_shows_message(at, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="need espn_auth.json"))
    at.button(key="sync").click().run()
    assert any("need espn_auth.json" in e.value for e in at.error)


def test_season_tab_uses_espn_matchup_calendar(at, home, monkeypatch):
    """One long ESPN matchup (id 3, no data/matchups.csv): its scores are looked up by ESPN's id, not by period + 1."""
    import espn
    ids = pd.read_csv(home / "data" / "projections.csv").player_id.astype(int).tolist()
    ent = lambda pids, slots: [{"playerId": p, "lineupSlotId": sl} for p, sl in zip(pids, slots)]
    js = {"teams": [{"id": 13, "name": "Raj", "roster": {"entries": ent(ids[20:34], [5] * 2 + [12] * 10 + [13] * 2)}},
                    {"id": 1, "name": "Rivals", "roster": {"entries": ent(ids[40:45], [5] * 5)}}],
          "settings": {"scheduleSettings": {"matchupPeriodCount": 3, "matchupPeriods": {"3": list(range(1, 300))}}},
          "schedule": [{"matchupPeriodId": 3, "home": {"teamId": 13, "totalPoints": 11.0},
                        "away": {"teamId": 1, "totalPoints": 22.0}}]}
    (home / "espn_auth.json").write_text('{"espn_s2": "x", "SWID": "{y}"}')
    monkeypatch.setattr(espn, "league", lambda *a, **k: js)
    from streamlit.testing.v1 import AppTest
    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at2.exception and not at2.error
    assert [d.value["Points so far"].tolist() for d in at2.dataframe if "Points so far" in d.value] == [[11, 22]]


def test_season_tab_shows_matchup_and_pickups(at, home, monkeypatch):
    """A synthetic ESPN league: me (team 13) with 14 players vs team 1 with 5; everyone else in the pool is free."""
    import espn
    ids = pd.read_csv(home / "data" / "projections.csv").player_id.astype(int).tolist()
    ent = lambda pids, slots: [{"playerId": p, "lineupSlotId": sl} for p, sl in zip(pids, slots)]
    js = {"teams": [{"id": 13, "name": "Raj", "roster": {"entries": ent(ids[20:34], [5] * 2 + [12] * 10 + [13] * 2)}},
                    {"id": 1, "name": "Rivals", "roster": {"entries": ent(ids[40:45], [5] * 5)}}],
          "settings": {"scheduleSettings": {"matchupPeriodCount": 19, "matchupPeriods": {"1": [1]}}},  # placeholder
          "schedule": [{"matchupPeriodId": 1, "home": {"teamId": 13, "totalPoints": 10.0},
                        "away": {"teamId": 1, "totalPoints": 20.0}}]}
    (home / "espn_auth.json").write_text('{"espn_s2": "x", "SWID": "{y}"}')
    monkeypatch.setattr(espn, "league", lambda *a, **k: js)
    from streamlit.testing.v1 import AppTest
    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at2.exception and not at2.error
    assert at2.metric[0].label == "Win chance"
    tables = [d.value for d in at2.dataframe]
    assert list(tables[-2].columns) == ["Team", "Points so far", "Projected final"]
    assert tables[-2]["Points so far"].tolist() == [10, 20]
    assert list(tables[-1].columns[:3]) == ["Player", "Pos", "Games left this matchup"] and len(tables[-1]) == 15
    assert (tables[-1]["Drop"] != "").all()


CATS = [{"stat": s, "reverse": s in ("PF", "TO")} for s in
        ("PTS", "BLK", "STL", "AST", "OREB", "DREB", "PF", "TO", "FG%", "FT%", "3P%", "TW")]


def test_category_league_labels_value_and_flags_sim(home):
    s = json.loads((home / "settings.json").read_text())
    s.pop("scoring")
    (home / "settings.json").write_text(json.dumps({**s, "format": "categories", "categories": CATS, "teams": 16,
                                                    "rounds": 11, "my_slot": 8}))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert "Value/g" in at.dataframe[0].value.columns and "Pts/game" not in at.dataframe[0].value.columns
    assert any("approximate in category leagues" in c.value for c in at.caption)
    assert not any("season points" in c.value for c in at.caption)
    from sim import LEAGUE_KEYS
    assert {"format", "categories"} <= set(LEAGUE_KEYS)  # a saved sim from another category setup is stale
    assert "scoring" not in json.loads((home / "settings.json").read_text())


def test_points_league_has_no_category_caption(at):
    assert "Pts/game" in at.dataframe[0].value.columns
    assert not any("approximate in category leagues" in c.value for c in at.caption)
