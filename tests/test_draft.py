import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from draft import (board, day_matrix, ev_matrix, feasible, lineup, matched, picks_before_next, slot_fit, snake,
                   starters, team_value)
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


def util_fit(n_players, n_util):
    b = make_board([(1, "UTIL", "ACTIVE", 1)] * n_players, {**LEAGUE, "slots": {"UTIL": n_util}})
    return slot_fit(b, {"slots": {"UTIL": n_util}})


def test_team_value_daily_best_lineup():
    ev = np.array([[10, 0], [5, 5], [1, 8]], float)
    assert team_value(ev, *util_fit(3, 1)) == 18
    assert team_value(ev, *util_fit(3, 2)) == 28
    fit, cap = util_fit(3, 1)
    assert team_value(np.stack([ev, ev]), np.stack([fit, fit]), cap).tolist() == [18, 18]
    assert team_value(ev[[]], fit[[]], cap) == 0


def test_team_value_respects_positions():
    s = {"slots": {"G": 1, "UTIL": 1, "BE": 3}}
    b = make_board([(10, "C|PF/C|UTIL", "ACTIVE", 1), (9, "C|PF/C|UTIL", "ACTIVE", 1),
                    (5, "PG|G|UTIL", "ACTIVE", 1), (8, "PG|G|UTIL", "ACTIVE", 2)], {**LEAGUE, **s})
    fit, cap = slot_fit(b, s)
    ev = np.array([[10], [9], [5], [0]], float)  # player 3 is off that day
    assert team_value(ev, fit, cap) == 15  # two Cs can't both start: C in UTIL + G in G
    assert lineup(ev, fit, cap)[:, 0].tolist() == [True, False, True, False]


def test_team_value_reassigns_slots():
    s = {"slots": {"G": 1, "UTIL": 1}}
    b = make_board([(10, "PG|G|UTIL", "ACTIVE", 1), (9, "G", "ACTIVE", 1), (8, "C|UTIL", "ACTIVE", 1)],
                   {**LEAGUE, **s})
    fit, cap = slot_fit(b, s)
    assert team_value(np.array([[10.0], [9.0], [8.0]]), fit, cap) == 19  # G-only takes G, first G moves to UTIL
    assert team_value(np.array([[10.0], [0.0], [8.0]]), fit, cap) == 18


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


import json

from draft import STRATEGIES, advise, recommend, survive

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


def test_punt_strategies_value_without_their_categories(monkeypatch):
    import model
    from draft import offered
    cats = [{"stat": s, "reverse": s in ("TO", "PF")} for s in ("PTS", "AST", "TO", "PF", "FT%")]
    s = {**LEAGUE, "format": "categories", "categories": cats}
    shim = lambda df, st: sum((-1 if c["reverse"] else 1) * df[c["stat"].rstrip("%")] for c in st["categories"])
    monkeypatch.setattr(model, "value_pg", shim, raising=False)  # P0-B's value_pg: z-scores; plain sums here
    b = make_board([(10, "G|UTIL|BE", "ACTIVE", 1), (11, "G|UTIL|BE", "ACTIVE", 1)])
    b[["AST", "TO", "PF", "FT"]] = [[10.0, 4.0, 4.0, -5.0], [0.0, 0.0, 0.0, 0.0]]
    b["fp_pg"] = shim(b, s)  # 7 vs 11
    G, _ = day_matrix(b, SCHED)

    def top(strategy):
        return recommend(b, ev_matrix(b, G), [], set(), s, strategy).index[0]

    assert top("greedy") == 1 and top("punt_AST") == 1 and top("punt_BLK") == 1  # BLK isn't a category here
    assert top("punt_TO_PF") == 0 and top("punt_FT") == 0  # 15 vs 11, 12 vs 11
    assert {"punt_TO_PF", "punt_FT", "punt_3P", "punt_BLK", "punt_AST"} <= set(offered(s))
    assert not any(n.startswith("punt_") for n in offered(LEAGUE))


def test_survive():
    b = make_board([(100, "UTIL", "ACTIVE", 1)] + [(10, "UTIL", "ACTIVE", 1)] * 4)
    rng = np.random.default_rng(0)
    assert survive(b, [0, 4], set(), [], {}, LEAGUE, rng).tolist() == [1.0, 1.0]
    p = survive(b, [0, 4], set(), [1], {}, LEAGUE, rng)
    assert p[0] < 0.25 and p[4] > 0.5
    assert survive(b, [0], set(), None, {}, LEAGUE, rng).tolist() == [0.0]


def test_survive_follows_espn_rank_over_value():
    b = make_board([(30, "UTIL", "ACTIVE", 1), (20, "UTIL", "ACTIVE", 1)])  # adp 1, 2 (fallback rank)
    rng = np.random.default_rng(0)
    p = survive(b, [0, 1], set(), [0], {}, LEAGUE, rng)
    assert p[0] < p[1]
    b["espn_rank"] = [50.0, 1.0]
    p = survive(b, [0, 1], set(), [0], {}, LEAGUE, rng)
    assert p[0] > p[1]


def test_survive_opponent_skips_players_it_cannot_start():
    s = {**LEAGUE, "slots": {"G": 1, "PF/C": 1, "BE": 2}}
    b = make_board([(30, "C|PF/C", "ACTIVE", 1), (29, "PG|G", "ACTIVE", 1), (40, "C|PF/C", "ACTIVE", 1)], s)
    rng = np.random.default_rng(0)
    free_c = survive(b, [0], {2}, [0], {}, s, rng)[0]
    blocked_c = survive(b, [0], {2}, [0], {0: [2]}, s, rng)[0]  # team 0 already starts a C at PF/C
    assert blocked_c > free_c + 0.2


def test_advise():
    s = {**LEAGUE, "teams": 3, "rounds": 2, "my_slot": 2, "slots": {"UTIL": 1, "BE": 1}}
    b = make_board(ROWS + [(5, "UTIL|BE|IR", "ACTIVE", 1)] * 3, s)
    G, _ = day_matrix(b, SCHED)
    ev = ev_matrix(b, G)
    a = advise(b, ev, [{"overall": 1, "player": 0}], s)
    assert (a["on_clock"], a["my_next"]) == (1, 5)  # I'm on the clock; p_back refers to my pick after this
    top = a["top"]
    assert list(top.columns) == ["player_id", "name", "pos", "fp_pg", "games", "value", "p_back", "status"]
    assert set(top.status) <= {"", "OUT", "DTD", "IR", "SUSP"}
    assert top.index[0] == 1 and 0 not in top.index and top.attrs["fallback"] is False
    assert top.pos.iloc[0] == "G" and top.p_back.between(0, 1).all()
    assert a["roster"] == [] and a["needs"] == ["UTIL"]
    json.dumps({**a, "top": top.to_dict("records")})  # server-ready

    picks = [{"overall": 1, "player": 0}, {"overall": 2, "player": None}, {"overall": 3, "player": 1}]
    a = advise(b, ev, picks, s)
    assert (a["on_clock"], a["my_next"]) == (2, 5)
    assert a["roster"][0]["player_id"] is None and a["roster"][0]["name"] == "(player not listed)"
    assert a["needs"] == ["UTIL"]
    assert 1 not in a["top"].index

    a = advise(b, ev, [{"overall": i, "player": i - 1} for i in range(1, 7)], s)
    assert (a["on_clock"], a["my_next"], len(a["top"])) == (None, None, 0)
    assert [r["player_id"] for r in a["roster"]] == [101, 104]


def test_advise_status_tags():
    s = {**LEAGUE, "teams": 2, "rounds": 2, "my_slot": 1, "slots": {"UTIL": 1, "IR": 1}}
    b = make_board([(30, "UTIL|BE|IR", "OUT", 1), (20, "UTIL|BE|IR", "DAY_TO_DAY", 1), (10, "UTIL|BE|IR", "ACTIVE", 2)], s)
    G, _ = day_matrix(b, SCHED)
    top = advise(b, ev_matrix(b, G), [], s)["top"]
    assert dict(zip(top.name, top.status)) == {"P0": "OUT", "P1": "DTD", "P2": ""}
