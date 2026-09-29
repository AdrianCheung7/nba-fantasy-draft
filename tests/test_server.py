import json
import os
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

import draft
import server

ROOT = Path(__file__).parent.parent
STATS = ["PTS", "FGM", "FGA", "FTM", "FTA", "3PM", "3PA", "OREB", "DREB", "AST", "STL", "BLK", "TO", "MIN",
         "EJ", "FF", "TF", "TD", "QD", "TW"]
# id, name, ESPN proTeamId
PLAYERS = [(3112335, "Nikola Jokic", 7), (3032977, "Giannis Antetokounmpo", 14), (4432166, "Cade Cunningham", 8),
           (5217746, "Yang Hansen", 22), (1, "Jalen Williams", 2), (2, "Jalen Williams", 25)]

SEEN = []  # settings each fake advise call received


@pytest.fixture
def home(tmp_path):
    (tmp_path / "data").mkdir()
    shutil.copy(ROOT / "settings.json", tmp_path / "settings.json")
    rows = [dict(player_id=i, name=n, team_id=t, positions="C|UTIL|BE", adp=float(k), injury_status="ACTIVE",
                 age=25.0, gp=70.0, cv=0.3, **dict.fromkeys(STATS, 1.0)) for k, (i, n, t) in enumerate(PLAYERS)]
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "projections.csv", index=False)
    pd.DataFrame({"team_id": [7, 8, 14, 22, 2, 25], "date": ["2026-10-20"] * 6, "week": 1}
                 ).to_csv(tmp_path / "data" / "schedule.csv", index=False)
    return tmp_path


@pytest.fixture
def api(home, monkeypatch):
    calls = []
    SEEN.clear()

    def fake_advise(b, ev, picks, settings, strategy="greedy"):
        calls.append(picks)
        SEEN.append(settings)
        top = pd.DataFrame({"player_id": [1], "name": ["X"], "pos": ["C"], "fp_pg": [1.5], "games": [70.0],
                            "value": [100.0], "p_back": [0.5]})
        top.attrs["fallback"] = False
        return {"on_clock": 2, "my_next": 9, "top": top, "roster": [], "needs": ["G"]}

    monkeypatch.setattr(draft, "advise", fake_advise, raising=False)
    srv = server.make_server(home, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None, method=None, origin=None):
        req = urllib.request.Request(base + path, method=method or ("POST" if body is not None else "GET"),
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Origin": origin} if origin else {})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or b"null"), r.headers

    yield call, calls
    srv.shutdown()
    srv.server_close()


def pick(pk, name, espn_id=None, team="XXX"):
    return {"round": 1, "pick": pk, "overall": pk, "espn_id": espn_id, "name": name, "nba_team": team,
            "pos": "C", "fantasy_team": "T"}


def test_state_before_picks(api):
    call, _ = api
    assert call("/state")[0]["status"] == "no picks yet"


def test_post_maps_persists_and_responds(api, home):
    call, calls = api
    picks = [pick(1, "Nikola Jokic", 3112335), pick(2, "Giánnis Antetokounmpo", None, "MIA"),
             pick(3, "Cade Cunningham"), pick(4, "Nobody Here"), pick(6, "Zzz", 5217746),
             pick(7, "Jalen Williams", None, "OKC")]
    r, _ = call("/picks", {"picks": picks[::-1], "url": "u"})  # order on the page is not guaranteed
    assert r["mapped"] == 5 and r["unmatched"] == ["Nobody Here"]
    assert r["top"][0]["name"] == "X" and r["my_next"] == 9 and r["needs"] == ["G"]
    # advise sees board row indexes, gaps filled with None
    assert [p["player"] for p in calls[-1]] == [0, 1, 2, None, None, 3, 5]
    assert [p["overall"] for p in calls[-1]] == [1, 2, 3, 4, 5, 6, 7]
    saved = json.loads((home / "data" / "draft.json").read_text())
    assert saved == [3112335, 3032977, 4432166, None, None, 5217746, 2]
    assert call("/picks", {"picks": picks})[0] == r  # idempotent
    assert call("/state")[0] == r


def test_cors_preflight(api):
    call, _ = api
    _, h = call("/picks", method="OPTIONS")
    assert h["Access-Control-Allow-Origin"] == "*" and h["Access-Control-Allow-Private-Network"] == "true"


def test_bad_body(api):
    call, _ = api
    with pytest.raises(urllib.error.HTTPError) as e:
        call("/picks", {"picks": "nope"})
    assert e.value.code == 400


def test_only_extension_origin_allowed(api):
    call, _ = api
    for path, body in (("/picks", {"picks": [pick(1, "Nikola Jokic", 3112335)]}), ("/state", None)):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(path, body, origin="https://evil.example")
        assert e.value.code == 403
    assert call("/state", origin="chrome-extension://abc")[0]["status"] == "no picks yet"


def players(calls):
    return [p["player"] for p in calls[-1]]


def test_partial_repost_never_forgets_picks(api, home):
    call, calls = api
    call("/picks", {"picks": [pick(1, "Nikola Jokic", 3112335), pick(2, "Giannis Antetokounmpo", 3032977),
                              pick(3, "Cade Cunningham", 4432166)]})
    call("/picks", {"picks": [pick(3, "Cade Cunningham", 4432166)]})  # lazy-loaded / filtered Picks panel
    assert players(calls) == [0, 1, 2]
    assert json.loads((home / "data" / "draft.json").read_text()) == [3112335, 3032977, 4432166]
    call("/picks", {"picks": [pick(1, "Nikola Jokic", 3112335), pick(4, "Yang Hansen", 5217746)]})
    assert players(calls) == [0, 1, 2, 3]


def test_new_draft_resets_when_pick_one_differs(api, home):
    call, calls = api
    call("/picks", {"picks": [pick(1, "Nikola Jokic", 3112335), pick(2, "Giannis Antetokounmpo", 3032977)]})
    call("/picks", {"picks": [pick(1, "Cade Cunningham", 4432166)]})
    assert players(calls) == [2]
    assert json.loads((home / "data" / "draft.json").read_text()) == [4432166]


def test_new_draft_resets_when_league_changes_even_if_pick_one_same(api, home):
    # practice drafts: SGA/Jokic go 1-2 every time, but each practice draft has its own leagueId
    call, calls = api
    old = [pick(1, "Nikola Jokic", 3112335), pick(2, "Giannis Antetokounmpo", 3032977), pick(3, "Cade Cunningham", 4432166)]
    call("/picks", {"picks": old, "url": "https://fantasy.espn.com/basketball/draft?leagueId=111&teamId=13"})
    call("/picks", {"picks": old[:1], "url": "https://fantasy.espn.com/basketball/draft?leagueId=222&teamId=13"})
    assert players(calls) == [0]


def test_new_draft_resets_when_a_known_player_reappears_elsewhere(api):
    call, calls = api
    call("/picks", {"picks": [pick(1, "Nikola Jokic", 3112335), pick(2, "Giannis Antetokounmpo", 3032977)]})
    call("/picks", {"picks": [pick(1, "Nikola Jokic", 3112335), pick(2, "Cade Cunningham", 4432166),
                              pick(3, "Giannis Antetokounmpo", 3032977)]})
    assert players(calls) == [0, 2, 1]


def test_pick_number_beyond_team_count_is_overall(api):
    call, calls = api
    p = pick(17, "Yang Hansen", 5217746)
    call("/picks", {"picks": [pick(1, "Nikola Jokic", 3112335), {**p, "round": 2}]})
    assert [x["overall"] for x in calls[-1]] == list(range(1, 18)) and calls[-1][16]["player"] == 3


def test_my_slot_derived_from_my_team_name(api):
    call, calls = api
    mine = {**pick(3, "Nikola Jokic", 3112335), "round": 2, "fantasy_team": "Raj"}  # round 2, pick 3 of 14 -> slot 12
    r, _ = call("/picks", {"picks": [mine]})
    assert SEEN[-1]["my_slot"] == 12 and r["slot"] == 12 and r["me"] == 11
    assert "12" in r["warning"] and "9" in r["warning"]
    r, _ = call("/picks", {"picks": [{**pick(9, "Nikola Jokic", 3112335), "fantasy_team": "Raj"}]})  # odd round: pick 9
    assert SEEN[-1]["my_slot"] == 9 and r["slot"] == 9 and "warning" not in r
    r, _ = call("/picks", {"picks": [pick(1, "Nikola Jokic", 3112335)]})  # none of mine on the page: settings slot
    assert r["slot"] == 9 and "warning" not in r


def test_recompute_on_settings_change_and_restart(api, home):
    call, calls = api
    call("/picks", {"picks": [pick(1, "Nikola Jokic", 3112335)]})
    n = len(calls)
    f = home / "settings.json"
    f.write_text(json.dumps({**json.loads(f.read_text()), "my_slot": 5}))
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 5))
    r, _ = call("/state")
    assert len(calls) == n + 1 and r["me"] == 4 and r["top"][0]["name"] == "X"
    srv = server.make_server(home, 0)  # restart: last result comes back without a new POST
    try:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/state") as resp:
            assert json.loads(resp.read())["me"] == 4
    finally:
        srv.shutdown()
        srv.server_close()
