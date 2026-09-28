import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim import bracket_order, playoffs, regular_season, round_robin, validate


@pytest.mark.parametrize("n", [4, 5, 14])
def test_round_robin_covers_every_pair(n):
    games = [tuple(sorted(p)) for rnd in round_robin(n) for p in rnd if min(p) >= 0]
    assert len(games) == len(set(games)) == len(list(combinations(range(n), 2)))


def test_bracket_order():
    assert bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]


def test_regular_season_strongest_team_first():
    weekly = np.tile(np.arange(4.0)[:, None], (1, 3))
    rank, wins = regular_season(weekly, 3)
    assert rank.tolist() == [3, 2, 1, 0] and wins[3] == 3


def test_regular_season_ties():
    rank, wins = regular_season(np.ones((4, 3)), 3)
    assert wins.tolist() == [1.5, 1.5, 1.5, 1.5]


def test_playoffs_six_teams_with_byes():
    weekly = np.tile(np.arange(6.0)[:, None], (1, 3))
    assert playoffs([0, 1, 2, 3, 4, 5], weekly, [0, 1, 2]) == 5
    assert playoffs([5, 4, 3, 2, 1, 0], weekly, [0, 1, 2]) == 5
    with pytest.raises(ValueError):
        playoffs(list(range(8)), np.ones((8, 2)), [0, 1])


def test_validate():
    ok = {"teams": 14, "playoff_teams": 6, "playoff_weeks": 3}
    validate(ok, 14)
    for bad, slot in [(ok, 15), (ok, 0), ({**ok, "playoff_teams": 9}, 1), ({**ok, "playoff_teams": 15}, 1)]:
        with pytest.raises(ValueError):
            validate(bad, slot)


from draft import board, day_matrix, ev_matrix, starters, team_value
from model import RATES, STATS
from sim import adp_pick, sim_draft, sim_season


def make_board(rows, settings):
    """rows: (pts_per_game, positions, injury_status, team_id, cv)"""
    proj = pd.DataFrame([{"player_id": 100 + i, "name": f"P{i}", "team_id": t, "positions": pos, "adp": i + 1.0,
                          "injury_status": inj, "age": 25, "gp": 82.0, "cv": cv,
                          **{c: 0.0 for c in STATS + RATES}, "PTS": pts}
                         for i, (pts, pos, inj, t, cv) in enumerate(rows)])
    return board(proj, settings)


def make_schedule(teams=4, weeks=4):
    days = pd.date_range("2026-10-19", periods=7 * weeks)
    return pd.DataFrame([{"team_id": t, "date": d.strftime("%Y-%m-%d"), "week": i // 7 + 1}
                         for i, d in enumerate(days) for t in range(1, teams + 1) if (i + t) % 2 == 0])


TINY = {"teams": 3, "rounds": 2, "slots": {"UTIL": 1, "BE": 1}, "scoring": {"PTS": 1},
        "playoff_teams": 2, "playoff_weeks": 1}


def test_season_matches_team_value():
    s = {**TINY, "slots": {"UTIL": 2, "BE": 1}}
    b = make_board([(30, "UTIL|BE", "ACTIVE", 1, 0.0), (20, "UTIL|BE", "ACTIVE", 2, 0.0),
                    (10, "UTIL|BE", "ACTIVE", 1, 0.0)], s)
    sched = make_schedule()
    G, days = day_matrix(b, sched)
    week = sched.drop_duplicates("date").set_index("date").week.reindex(days).values
    weekly = sim_season([[0, 1, 2]], b, G, week - 1, int(week.max()), 2, np.random.default_rng(0))
    assert weekly.sum() == pytest.approx(team_value(ev_matrix(b, G)[[0, 1, 2]], 2))


def test_sim_draft_unique_full_rosters():
    b = make_board([(40 - 3 * i, "UTIL|BE", "ACTIVE", 1 + i % 4, 0.3) for i in range(10)], TINY)
    G, _ = day_matrix(b, make_schedule())
    ev = ev_matrix(b, G)
    for opp in ("greedy", "adp"):
        rosters = sim_draft(b, ev, TINY, 0, "greedy", opp, None, np.random.default_rng(1))
        picks = [p for r in rosters for p in r]
        assert [len(r) for r in rosters] == [2, 2, 2] and len(set(picks)) == 6
        assert rosters[0][0] == 0  # I pick first, greedy, no noise on my own picks


def test_adp_pick_unranked_after_ranked_then_by_value():
    # ESPN gives every undrafted player adp ~140: rank them after real ADPs, best season_fp first
    b = make_board([(5, "UTIL|BE", "ACTIVE", 1, 0.3), (10, "UTIL|BE", "ACTIVE", 1, 0.3),
                    (30, "UTIL|BE", "ACTIVE", 1, 0.3)], TINY)
    b["adp"] = [130.0, 140.0, 140.0]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        assert adp_pick(b, [], set(), TINY, rng) == 0
        assert adp_pick(b, [], {0}, TINY, rng) == 2


from draft import STRATEGIES
from sim import best, run


def test_run_smoke():
    s = {**TINY, "teams": 4}
    b = make_board([(40 - 3 * i, "UTIL|BE|IR", "OUT" if i == 5 else "ACTIVE", 1 + i % 4, 0.3)
                    for i in range(12)], s)
    df = run(s, 2, n_drafts=2, n_seasons=3, b=b, schedule=make_schedule())
    assert len(df) == len(STRATEGIES) * 2
    assert df[["p_champion", "p_playoffs", "p_bottom3"]].stack().between(0, 1).all()
    assert best(df) in STRATEGIES
