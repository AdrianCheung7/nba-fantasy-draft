import json
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


def test_playoff_rounds_sum_matchup_periods():
    weekly = np.array([[5, 0, 4, 5], [3, 3, 0, 3], [0, 0, 0, 0], [0, 6, 1, 1]], float)
    # 2-period rounds: semis (1v4: 5 < 6, 2v3) on periods 0+1, final (4v2: 2 < 3) on periods 2+3
    assert playoffs([0, 1, 2, 3], weekly, [0, 1, 2, 3], 2) == 1
    assert playoffs([0, 1, 2, 3], weekly, [0, 1, 2, 3]) == 0  # 1-period rounds: semis on 2, final on 3
    with pytest.raises(ValueError):
        playoffs([0, 1, 2, 3], weekly, [1, 2, 3], 2)


def test_validate():
    ok = {"teams": 14, "playoff_teams": 6, "playoff_weeks": 3}
    validate(ok, 14)
    validate({**ok, "playoff_teams": 4, "playoff_weeks": 4, "playoff_round_periods": 2}, 9)  # the real league
    real_short = {**ok, "playoff_teams": 4, "playoff_weeks": 3, "playoff_round_periods": 2}
    for bad, slot in [(ok, 15), (ok, 0), ({**ok, "playoff_teams": 9}, 1), ({**ok, "playoff_teams": 15}, 1),
                      (real_short, 1)]:
        with pytest.raises(ValueError):
            validate(bad, slot)


from draft import board, day_matrix, ev_matrix, slot_fit, starters, team_value
from model import RATES, STATS
from sim import availability, back_to_back, calendar, espn_pick, sim_draft, sim_season


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
    fit, cap = slot_fit(b, s)
    week = sched.drop_duplicates("date").set_index("date").week.reindex(days).values
    weekly = sim_season([[0, 1, 2]], b, G, week - 1, int(week.max()), fit, cap, np.random.default_rng(0))
    assert weekly.sum() == pytest.approx(team_value(ev_matrix(b, G)[[0, 1, 2]], fit[[0, 1, 2]], cap))


def test_season_lineup_respects_positions():
    s = {**TINY, "slots": {"PF/C": 1, "G": 1, "BE": 1}}
    b = make_board([(30, "C|PF/C", "ACTIVE", 1, 0.0), (20, "C|PF/C", "ACTIVE", 1, 0.0),
                    (10, "PG|G", "ACTIVE", 1, 0.0)], s)
    sched = make_schedule(teams=1, weeks=1)
    G, _ = day_matrix(b, sched)
    weekly = sim_season([[0, 1, 2]], b, G, np.zeros(G.shape[1], int), 1, *slot_fit(b, s), np.random.default_rng(0))
    assert weekly[0, 0] == 40 * G[0].sum()  # the second C sits


def test_calendar_weeks_and_matchups():
    days = np.array(["2026-10-20", "2026-10-27", "2026-11-03", "2026-11-10", "2026-11-17"])
    sched = pd.DataFrame({"team_id": 1, "date": days, "week": [1, 2, 3, 4, 5]})
    s = {"playoff_weeks": 2}
    period, n_reg, po = calendar(days, sched, s)
    assert period.tolist() == [0, 1, 2, 3, 4] and (n_reg, po) == (3, [3, 4])
    m = pd.DataFrame({"date": days[1:], "matchup": [1, 1, 2, 3], "playoff": [False, False, False, True]})
    period, n_reg, po = calendar(days, sched, s, m)
    assert period.tolist() == [-1, 0, 0, 1, 2] and (n_reg, po) == (2, [2])


def test_sim_draft_unique_full_rosters():
    b = make_board([(40 - 3 * i, "UTIL|BE", "ACTIVE", 1 + i % 4, 0.3) for i in range(10)], TINY)
    G, _ = day_matrix(b, make_schedule())
    ev = ev_matrix(b, G)
    for opp in ("greedy", "espn"):
        rosters = sim_draft(b, ev, TINY, 0, "greedy", opp, None, np.random.default_rng(1))
        picks = [p for r in rosters for p in r]
        assert [len(r) for r in rosters] == [2, 2, 2] and len(set(picks)) == 6
        assert rosters[0][0] == 0  # I pick first, greedy, no noise on my own picks


def test_espn_pick_unranked_after_ranked_then_by_value():
    # ESPN gives every undrafted player adp ~140: rank them after real ADPs, best season_fp first
    b = make_board([(5, "UTIL|BE", "ACTIVE", 1, 0.3), (10, "UTIL|BE", "ACTIVE", 1, 0.3),
                    (30, "UTIL|BE", "ACTIVE", 1, 0.3)], TINY)
    b["adp"] = [130.0, 140.0, 140.0]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        assert espn_pick(b, [], set(), TINY, rng) == 0
        assert espn_pick(b, [], {0}, TINY, rng) == 2
    b["espn_rank"] = [999.0, 2.0, 999.0]  # espn_rank wins over adp; 999 = unranked
    assert espn_pick(b, [], set(), TINY, np.random.default_rng(0)) == 1
    assert espn_pick(b, [], {1}, TINY, np.random.default_rng(0)) == 2


from draft import STRATEGIES
from sim import best, from_draft, report, report_from_rosters, run


def test_run_smoke():
    s = {**TINY, "teams": 4}
    b = make_board([(40 - 3 * i, "UTIL|BE|IR", "OUT" if i == 5 else "ACTIVE", 1 + i % 4, 0.3)
                    for i in range(12)], s)
    df = run(s, 2, n_drafts=2, n_seasons=3, b=b, schedule=make_schedule())
    assert len(df) == len(STRATEGIES) * 2
    assert df[["p_champion", "p_playoffs", "p_bottom3"]].stack().between(0, 1).all()
    assert "obj_se" in df.columns and (df.obj_se >= 0).all()
    assert best(df) in STRATEGIES


def test_report_smoke():
    s = {**TINY, "teams": 4}
    b = make_board([(40 - 3 * i, "UTIL|BE", "ACTIVE", 1 + i % 4, 0.3) for i in range(12)], s)
    df = report(s, 2, n_drafts=2, n_seasons=3, b=b, schedule=make_schedule())
    assert df.team.tolist() == [1, 2, 3, 4] and df.us.tolist() == [False, True, False, False]
    assert df.p_champion.sum() == pytest.approx(1) and df.p_playoffs.sum() == pytest.approx(2)
    assert (df.avg_points > 0).all() and df.avg_wins.sum() == pytest.approx(6)  # 3 reg weeks x 2 games


def test_report_from_rosters_stronger_team_wins_more():
    s = {**TINY, "teams": 2, "my_slot": 1}
    b = make_board([(40, "UTIL|BE", "ACTIVE", 1, 0.1), (30, "UTIL|BE", "ACTIVE", 2, 0.1),
                    (5, "UTIL|BE", "ACTIVE", 1, 0.1), (4, "UTIL|BE", "ACTIVE", 2, 0.1)], s)
    df = report_from_rosters([[0, 1], [2, 3]], b, s, n_seasons=10, schedule=make_schedule())
    assert df.us.tolist() == [True, False] and df.p_champion.tolist() == [1.0, 0.0]
    assert df.avg_wins.iloc[0] == 3


def test_from_draft_builds_snake_rosters_and_marks_my_team(tmp_path):
    s = {**TINY, "teams": 2, "my_slot": 1}
    b = make_board([(40, "UTIL|BE", "ACTIVE", 1, 0.1), (30, "UTIL|BE", "ACTIVE", 2, 0.1),
                    (5, "UTIL|BE", "ACTIVE", 1, 0.1), (4, "UTIL|BE", "ACTIVE", 2, 0.1)], s)
    (tmp_path / "draft.json").write_text("[100, 101, 102, 103]")  # snake: team 1 gets 100 + 103, team 2 gets 101 + 102
    df = from_draft(s, 10, tmp_path, b=b, schedule=make_schedule())
    assert df.us.tolist() == [True, False] and df.p_champion.tolist() == [1.0, 0.0]  # 40 + 4 beats 30 + 5
    assert json.loads((tmp_path / "report_from_draft.json").read_text())["slot"] == 1
    (tmp_path / "last_picks.json").write_text('{"picks": [], "slot": 2}')  # the page said I am team 2
    (tmp_path / "draft.json").write_text("[100, null, 102, 103]")  # an unlisted pick just leaves a roster short
    df = from_draft(s, 10, tmp_path, b=b, schedule=make_schedule())
    assert df.us.tolist() == [False, True]


def test_best_keeps_greedy_when_gap_within_noise():
    df = pd.DataFrame([
        {"strategy": "greedy", "opponents": "greedy", "p_champion": 0.20, "p_bottom3": 0.10, "obj_se": 0.05},
        {"strategy": "greedy", "opponents": "adp", "p_champion": 0.20, "p_bottom3": 0.10, "obj_se": 0.05},
        {"strategy": "durable", "opponents": "greedy", "p_champion": 0.22, "p_bottom3": 0.11, "obj_se": 0.05},
        {"strategy": "durable", "opponents": "adp", "p_champion": 0.22, "p_bottom3": 0.11, "obj_se": 0.05},
    ])
    assert best(df) == "greedy"


def test_best_switches_when_gap_clearly_beyond_noise():
    df = pd.DataFrame([
        {"strategy": "greedy", "opponents": "greedy", "p_champion": 0.10, "p_bottom3": 0.20, "obj_se": 0.01},
        {"strategy": "greedy", "opponents": "adp", "p_champion": 0.10, "p_bottom3": 0.20, "obj_se": 0.01},
        {"strategy": "durable", "opponents": "greedy", "p_champion": 0.40, "p_bottom3": 0.05, "obj_se": 0.01},
        {"strategy": "durable", "opponents": "adp", "p_champion": 0.40, "p_bottom3": 0.05, "obj_se": 0.01},
    ])
    assert best(df) == "durable"


def test_availability_is_one_contiguous_spell():
    G = np.tile(np.arange(60) % 3 != 0, (500, 1))                      # 40 game days, every third day off
    plays = availability(G, np.full(500, 0.6), np.random.default_rng(0))
    assert not (plays & ~G).any()
    for row, g in zip(plays[:50], G[:50]):
        gap = np.flatnonzero(~row[g])                                  # missed games, by game number
        assert len(gap) == 0 or gap[-1] - gap[0] + 1 == len(gap)
    assert plays.sum(axis=1).mean() == pytest.approx(0.6 * 40, abs=1.0)
    assert availability(G, np.ones(500), np.random.default_rng(0)).sum() == G.sum()   # avail 1: everyone plays


def test_availability_beta_binomial_clumps_around_the_same_mean():
    G = np.ones((4000, 82), bool)
    plain = availability(G, np.full(4000, 0.75), np.random.default_rng(0)).mean(axis=1)
    clumpy = availability(G, np.full(4000, 0.75), np.random.default_rng(0), k=np.full(4000, 2.0)).mean(axis=1)
    assert plain.mean() == pytest.approx(0.75, abs=0.01) and clumpy.mean() == pytest.approx(0.75, abs=0.02)
    assert plain.std() < 0.06 and clumpy.std() > 0.2                   # binomial ~0.05; beta(1.5, .5) ~0.25
    always = availability(G, np.ones(4000), np.random.default_rng(0), k=np.full(4000, 50.0)).mean()
    assert 0.96 < always < 0.995                                       # m is clipped to .98: nobody is a lock
    assert availability(G, np.ones(4000), np.random.default_rng(0), k=np.full(4000, np.nan)).all()  # NaN: old way


def test_back_to_back_is_second_consecutive_night():
    days = np.array(["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-23", "2026-10-25", "2026-10-26"])
    G = np.array([[1, 1, 1, 0, 0, 1], [1, 0, 1, 1, 1, 1]], bool)       # team 2: 21st -> 23rd is not consecutive
    assert back_to_back(G, days).astype(int).tolist() == [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]]


def test_availability_rests_older_players_on_back_to_backs_but_keeps_the_season_total():
    G = np.ones((3000, 80), bool)
    b2b = G & (np.arange(80) % 2 == 1)                                 # every other night is a 2nd night
    pen = np.full(3000, 0.14)
    plays = availability(G, np.full(3000, 0.8), np.random.default_rng(0), b2b=b2b, pen=pen)
    assert plays.mean() == pytest.approx(0.8, abs=0.01)                # GP model already counts these misses
    assert plays[:, 1::2].mean() < plays[:, ::2].mean() - 0.1
    none = availability(G, np.full(3000, 0.8), np.random.default_rng(0), b2b=b2b, pen=np.zeros(3000))
    assert abs(none[:, 1::2].mean() - none[:, ::2].mean()) < 0.03


def test_sim_season_rests_only_rotation_veterans_on_back_to_backs():
    s = {**TINY, "slots": {"UTIL": 3, "BE": 0}}
    b = make_board([(30, "UTIL", "ACTIVE", 1, 0.0)] * 3, s)
    b["MIN"], b["age"] = [30.0, 30.0, 15.0], [24.0, 44.0, 44.0]        # pen: 0, .14, 0 (bench minutes)
    G = np.ones((3, 1000), bool)
    b2b = G.copy()
    b2b[:, 0] = False
    fit, cap = slot_fit(b, s)
    weekly = sim_season([[0], [1], [2]], b, G, np.zeros(1000, int), 1, fit, cap, np.random.default_rng(0), b2b)
    assert weekly[:, 0] == pytest.approx([30 * 1000, 30 * 1000 * 0.86, 30 * 1000], rel=0.04)
