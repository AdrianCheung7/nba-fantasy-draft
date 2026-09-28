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
