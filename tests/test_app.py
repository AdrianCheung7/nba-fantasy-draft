import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DRAFT = ROOT / "data" / "draft.json"
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "projections.csv").exists(),
                                reason="needs data: run fetch.py and model.py first")


@pytest.fixture
def at():
    from streamlit.testing.v1 import AppTest
    backup = DRAFT.read_text() if DRAFT.exists() else None
    DRAFT.unlink(missing_ok=True)
    yield AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    DRAFT.unlink(missing_ok=True)
    if backup is not None:
        DRAFT.write_text(backup)


def saved_picks():
    return json.loads(DRAFT.read_text()) if DRAFT.exists() else []


def test_loads_with_recommendations(at):
    assert not at.exception
    assert len(at.dataframe) >= 1


def test_draft_unlisted_and_undo(at):
    pid = int(pd.read_csv(ROOT / "data" / "projections.csv").player_id.iloc[0])
    at.selectbox(key="pick_0").set_value(pid).run()
    at.button(key="drafted").click().run()
    assert saved_picks() == [pid] and not at.exception
    at.selectbox(key="pick_1").set_value(-1).run()
    at.button(key="drafted").click().run()
    assert saved_picks() == [pid, None]
    at.button(key="undo").click().run()
    assert saved_picks() == [pid] and not at.exception


def test_changing_team_count_keeps_working(at):
    at.selectbox(key="teams").set_value(10).run()
    assert not at.exception
