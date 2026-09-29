import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import fetch
from fetch import parse_games, parse_players, parse_proj, parse_schedule, save

GAME = {"statSplitTypeId": 5, "statSourceId": 0, "seasonId": 2026, "scoringPeriodId": 7,        "stats": {"0": 30.0, "4": 3.0, "7": 1.0, "13": 11.0, "14": 20.0, "40": 36.0, "43": 1.0}}
DNP = {**GAME, "scoringPeriodId": 8, "stats": {}}
PROJ = {"statSplitTypeId": 0, "statSourceId": 1, "seasonId": 2027, "stats": {"0": 1640.0, "42": 82.0}}
PLAYER = {"id": 1, "fullName": "Test Guy", "active": True, "proTeamId": 5,
          "eligibleSlots": [1, 5, 7, 11, 12, 13], "injuryStatus": "OUT",
          "ownership": {"averageDraftPosition": 12.5}, "stats": [GAME, DNP, PROJ],
          "draftRanksByRankType": {"STANDARD": {"rank": 40}, "POINTS": {"rank": 33}}}


def test_parse_games_keeps_played_games_only():
    df = parse_games([PLAYER, {"id": 2}], 2026)
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.player_id, r.period, r.PTS, r.OREB, r.EJ, r.FGA, r.TW, r.BLK) == (1, 7, 30.0, 3.0, 1.0, 20.0, 1.0, 0.0)

def test_parse_players_positions_and_defaults():
    players = [PLAYER, {"id": 2, "active": True}, {"id": 3, "active": True, "eligibleSlots": [4, 11],
                                                   "ownership": {"averageDraftPosition": 0.0}},
               {"id": 4, "active": True, "eligibleSlots": [4], "draftRanksByRankType": {"STANDARD": {"rank": 77}}}]
    df = parse_players(players, {1: 27})
    assert list(df.player_id) == [1, 3, 4]
    a, b = df.iloc[0], df.iloc[1]
    assert a.positions == "SG|G|SG/SF|UTIL|BE|IR" and a.adp == 12.5 and a.injury_status == "OUT" and a.age == 27
    assert (a.espn_rank, b.espn_rank, df.iloc[2].espn_rank) == (33, 999, 77)  # POINTS, none, STANDARD fallback
    assert b.positions == "C|UTIL" and b.adp == 999.0 and b.injury_status == "ACTIVE" and b.team_id == 0


def test_parse_proj_per_game():
    r = parse_proj([PLAYER, {"id": 9}], 2027).iloc[0]
    assert (r.player_id, r.gp, r.PTS, r.REB) == (1, 82.0, 20.0, 0.0)


def test_parse_proj_keeps_espn_reb():
    proj = {**PROJ, "stats": {"0": 1640.0, "6": 820.0, "42": 82.0}}  # ESPN projects REB (id 6), not OREB/DREB
    assert parse_proj([{**PLAYER, "stats": [proj]}], 2027).iloc[0].REB == 10.0


def test_parse_schedule_eastern_dates_and_weeks():
    t = lambda d: int(pd.Timestamp(d, tz="America/New_York").timestamp() * 1000)
    js = {"settings": {"proTeams": [
        {"id": 1, "proGamesByScoringPeriod": {"1": [{"date": t("2026-10-19 19:30")}],
                                               "7": [{"date": t("2026-10-25 19:30")}]}},
        {"id": 2, "proGamesByScoringPeriod": {"8": [{"date": t("2026-10-26 22:30")}]}},
        {"id": 0, "proGamesByScoringPeriod": {"1": [{"date": t("2026-10-19 19:30")}]}}]}}
    df = parse_schedule(js)
    assert list(df.columns) == ["team_id", "date", "week"]
    assert df.values.tolist() == [[1, "2026-10-19", 1], [1, "2026-10-25", 1], [2, "2026-10-26", 2]]


def test_save_is_atomic(tmp_path):
    p = tmp_path / "x.csv"
    save(pd.DataFrame({"a": [1]}), p)
    assert p.read_text().splitlines() == ["a", "1"]
    assert not (tmp_path / "x.tmp").exists()


def test_get_requests_full_player_list(monkeypatch):
    # Live ESPN behavior (discovered 2026-09-27): a bare GET on the players endpoint
    # returns only 50 players (its default page size), not "all players" as assumed.
    # fetch.get must send fetch.PLAYER_HEADERS so player pulls aren't silently truncated.
    seen = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def fake_get(url, headers=None, timeout=None, cookies=None):
        seen["headers"], seen["cookies"] = headers, cookies
        return FakeResp()

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    fetch.get(fetch.PLAYERS.format(season=2026), headers=fetch.PLAYER_HEADERS, cookies={"a": "b"})
    assert seen["headers"] == fetch.PLAYER_HEADERS and seen["cookies"] == {"a": "b"}
    assert '"limit": 5000' in fetch.PLAYER_HEADERS["x-fantasy-filter"]
