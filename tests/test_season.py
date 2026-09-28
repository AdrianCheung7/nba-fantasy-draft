import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from draft import day_matrix, ev_matrix, slot_fit
from season import drop_candidate, matchup_projection, pickups, rest_of_matchup, split_roster, week_ev, window
from test_draft import make_board

S = {"slots": {"UTIL": 2, "BE": 1, "IR": 1}, "rounds": 4, "scoring": {"PTS": 1}}
DAYS = ["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"]
SCHED = pd.DataFrame({"team_id": [1] * 4, "date": DAYS, "week": [1, 1, 2, 2]})
WK = np.array([True, True, False, False])
ROS = np.ones(4, bool)
# 0 A 30, 1 B 20, 2 C 5 (bench), 3 I 50 (IR slot); free: 4 X 25, 5 Y 10, 6 Z 40 (out), 7 W 50 (day-to-day)
ROWS = [(30, "UTIL", "ACTIVE", 1), (20, "UTIL", "ACTIVE", 1), (5, "UTIL", "ACTIVE", 1), (50, "UTIL", "ACTIVE", 1),
        (25, "UTIL", "ACTIVE", 1), (10, "UTIL", "ACTIVE", 1), (40, "UTIL", "OUT", 1), (50, "UTIL", "DAY_TO_DAY", 1)]


@pytest.fixture
def world():
    b = make_board(ROWS, S)
    G, _ = day_matrix(b, SCHED)
    return b, ev_matrix(b, G), G


def test_drop_candidate_is_least_valuable(world):
    b, ev, G = world
    fit, cap = slot_fit(b, S)
    assert drop_candidate(ev, fit, cap, [0, 1, 2]) == 2      # the bench player costs nothing
    assert drop_candidate(ev, fit, cap, [0, 1]) == 1         # both start: the weaker one
    assert drop_candidate(ev, fit, cap, []) is None


def test_pickups_gains_drop_and_order(world):
    b, ev, G = world
    p = pickups(b, ev, G, mine=[0, 1, 2, 3], active=[0, 1, 2], free=[4, 5, 6, 7], wk=WK, ros=ROS, settings=S)
    assert p.index[0] == 7 and (p["drop"] == "P2").all()     # never P3, who sits in the IR slot
    x, y, z, w = (p.loc[i] for i in (4, 5, 6, 7))
    assert (x.week_gain, x.ros_gain, x.games) == (10, 20, 2)     # 25 replaces 20 in the lineup: +5/day
    assert w.week_gain == 10 and w.games == 1                    # day-to-day: half the expected points, half the games
    assert y.week_gain == 0 and y.ros_gain == 0 and not y.star
    assert (z.games, z.week_gain, z.ros_gain, z.star) == (0, 0, 80, True)  # out this matchup, a star for the season
    assert list(p.week_gain) == sorted(p.week_gain, reverse=True)


def test_pickups_open_roster_spot_needs_no_drop(world):
    b, ev, G = world
    p = pickups(b, ev, G, mine=[0], active=[0], free=[4], wk=WK, ros=ROS, settings=S)
    assert p.loc[4, "drop"] == "" and p.loc[4, "ros_gain"] == 100  # 25/day x 4 days


def test_week_ev_scales_injuries(world):
    b, ev, G = world
    e = week_ev(b, ev, WK)
    assert e[6].tolist() == [0, 0] and e[7].tolist() == [25, 25] and e[0].tolist() == [30, 30]


def test_rest_of_matchup_mean_and_variance(world):
    b, ev, G = world
    fit, cap = slot_fit(b, S)
    mean, var = rest_of_matchup(b, week_ev(b, ev, WK), fit, cap, [0, 1, 2])
    assert mean == 100 and var == pytest.approx(2 * (9 ** 2 + 6 ** 2))  # sd = 0.3 * fp_pg; two starters, two days


def test_split_roster_skips_ir_and_unknown_players():
    rows = pd.DataFrame({"player_id": [1, 2, 3, 99], "lineup_slot": ["G", "IR", "BE", "G"]})
    assert split_roster(rows, {1: 10, 2: 11, 3: 12}) == ([10, 11, 12], [10, 12])


def test_window():
    days, period = np.array(DAYS), np.array([0, 0, 1, 1])
    p, wk, ros = window(days, period, "2026-10-21")
    assert p == 0 and wk.tolist() == [False, True, False, False] and ros.tolist() == [False, True, True, True]
    p, wk, _ = window(days, period, "2026-01-01")  # before the season: the first matchup, whole
    assert p == 0 and wk.sum() == 2
    assert window(days, period, "2027-01-01") is None


def test_matchup_projection_win_pct():
    even = matchup_projection(100, 100, 300, 300, 400, 400)
    assert even["win"] == pytest.approx(0.5) and even["me"] == 400 and even["opp"] == 400
    assert matchup_projection(100, 100, 350, 300, 400, 400)["win"] > 0.7
    assert matchup_projection(100, 100, 300, 300, 0, 0)["win"] == 0.5
    assert matchup_projection(100, 90, 300, 300, 0, 0)["win"] == 1.0
