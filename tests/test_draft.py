import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from draft import (board, day_matrix, ev_matrix, feasible, matched, picks_before_next, snake, starters,
                   team_value)
from model import RATES, STATS

LEAGUE = {"slots": {"G": 2, "F": 1, "SG/SF": 1, "PF/C": 1, "UTIL": 1, "BE": 6, "IR": 2}, "rounds": 14,
          "scoring": {"PTS": 1}}
SCHED = pd.DataFrame({"team_id": [1, 1, 2, 2], "date": ["2026-10-20", "2026-10-21"] * 2, "week": [1] * 4})


def make_board(rows, settings=LEAGUE):
    """rows: (pts_per_game, positions, injury_status, team_id)"""
    proj = pd.DataFrame([{"player_id": 100 + i, "name": f"P{i}", "team_id": t, "positions": pos, "adp": i + 1.0,
                          "injury_status": inj, "age": 25, "gp": 82.0, "cv": 0.3,
                          **{c: 0.0 for c in STATS + RATES}, "PTS": pts}
                         for i, (pts, pos, inj, t) in enumerate(rows)])
    return board(proj, settings)


def test_snake_and_next_pick():
    order = snake(3, 2)
    assert order == [0, 1, 2, 2, 1, 0]
    assert picks_before_next(order, 0, 0) == 4
    assert picks_before_next(order, 2, 2) == 0
    assert picks_before_next(order, 5, 0) is None


def test_starters_expand_slots():
    assert starters(LEAGUE) == ["G", "G", "F", "SG/SF", "PF/C", "UTIL"]


def test_board_columns():
    b = make_board([(20, "G|UTIL|BE|IR", "OUT", 1)])
    r = b.iloc[0]
    assert (r.fp_pg, r.avail, r.season_fp, r.ir_ok) == (20, 1.0, 1640, True)
    assert r.pos == {"G", "UTIL", "BE", "IR"} and r.sd == pytest.approx(6.0)


def test_day_matrix_free_agent():
    b = make_board([(20, "G|UTIL", "ACTIVE", 1), (20, "G|UTIL", "ACTIVE", 0)])
    G, days = day_matrix(b, SCHED)
    assert list(days) == ["2026-10-20", "2026-10-21"]
    assert G.tolist() == [[True, True], [False, False]]
    assert ev_matrix(b, G).tolist() == [[20, 20], [0, 0]]


def test_team_value_daily_best_lineup():
    ev = np.array([[10, 0], [5, 5], [1, 8]], float)
    assert team_value(ev, 1) == 18
    assert team_value(ev, 2) == 28
    assert team_value(np.stack([ev, ev]), 1).tolist() == [18, 18]
    assert team_value(ev[[]], 2) == 0


def test_matched_uses_augmenting_paths():
    assert matched([{"PF/C", "UTIL"}, {"PF/C"}], ["PF/C", "UTIL"]) == 2
    assert matched([{"C", "PF/C", "UTIL"}, {"C", "PF/C", "UTIL"}], ["G", "PF/C", "UTIL"]) == 2


def test_feasible_healthy_cap():
    s = {"slots": {"UTIL": 1, "BE": 1, "IR": 1}, "rounds": 3, "scoring": {"PTS": 1}}
    b = make_board([(30, "UTIL|BE|IR", "ACTIVE", 1)] * 3 + [(5, "UTIL|BE|IR", "OUT", 1)], s)
    assert feasible(b, [0, 1], [2, 3], s) == [3]


def test_feasible_starter_coverage():
    s = {"slots": {"G": 1, "PF/C": 1}, "rounds": 2, "scoring": {"PTS": 1}}
    b = make_board([(30, "C|PF/C|UTIL", "ACTIVE", 1), (25, "C|PF/C|UTIL", "ACTIVE", 1),
                    (10, "G|UTIL", "ACTIVE", 1)], s)
    assert feasible(b, [0], [1, 2], s) == [2]


from draft import STRATEGIES, recommend, survive

ROWS = [(30, "C|PF/C|UTIL|BE|IR", "ACTIVE", 1), (20, "G|UTIL|BE|IR", "ACTIVE", 1), (10, "G|UTIL|BE|IR", "ACTIVE", 2)]


def rec(rows, mine=(), taken=(), strategy="greedy", settings=LEAGUE, po_mask=None):
    b = make_board(rows, settings)
    G, _ = day_matrix(b, SCHED)
    return recommend(b, ev_matrix(b, G), list(mine), set(taken), settings, strategy, po_mask)


def test_recommend_ranks_and_excludes_taken_and_none():
    r = rec(ROWS, taken={0, None})
    assert list(r.index) == [1, 2]
    assert r.marginal.iloc[0] == 40  # 20 pts x 2 game days


def test_recommend_falls_back_when_nothing_legal():
    s = {"slots": {"UTIL": 1, "IR": 1}, "rounds": 2, "scoring": {"PTS": 1}}
    r = rec([(30, "UTIL|IR", "ACTIVE", 1), (20, "UTIL|IR", "ACTIVE", 1)], mine=[0], settings=s)
    assert list(r.index) == [1]
    assert r.attrs["fallback"] is True  # M7: flagged so the UI can warn the user


def test_recommend_no_fallback_flag_when_legal_picks_exist():
    assert rec(ROWS).attrs["fallback"] is False


def test_ir_early_boosts_injured_player():
    rows = [(20, "UTIL|BE|IR", "ACTIVE", 1), (18, "UTIL|BE|IR", "OUT", 1)]
    assert rec(rows).index[0] == 0
    assert rec(rows, strategy="ir_early").index[0] == 1


def test_playoffs_strategy_prefers_playoff_days():
    rows = [(10, "UTIL|BE", "ACTIVE", 1), (10, "UTIL|BE", "ACTIVE", 2)]
    sched_split = pd.DataFrame({"team_id": [1, 2], "date": ["2026-10-20", "2026-10-21"], "week": [1, 1]})
    b = make_board(rows)
    G, _ = day_matrix(b, sched_split)
    r = recommend(b, ev_matrix(b, G), [], set(), LEAGUE, "playoffs", np.array([False, True]))
    assert r.index[0] == 1


def test_every_strategy_scores():
    for name in STRATEGIES:
        assert rec(ROWS, strategy=name).score.notna().all(), name


def test_survive():
    b = make_board([(100, "UTIL", "ACTIVE", 1)] + [(10, "UTIL", "ACTIVE", 1)] * 4)
    rng = np.random.default_rng(0)
    assert survive(b, [0, 4], set(), 0, rng).tolist() == [1.0, 1.0]
    p = survive(b, [0, 4], set(), 1, rng)
    assert p[0] == 0.0 and p[4] > 0.5
    assert survive(b, [0], set(), None, rng).tolist() == [0.0]
