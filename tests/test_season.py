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


def test_pickups_full_active_roster_with_empty_ir_suggests_a_drop(world):
    b, ev, G = world
    p = pickups(b, ev, G, mine=[0, 1, 2], active=[0, 1, 2], free=[4], wk=WK, ros=ROS, settings=S)  # 3 = slots minus IR
    assert p.loc[4, "drop"] == "P2" and p.loc[4, "ros_gain"] == 20  # 25 replaces the bench 5 and displaces B: +5 x 4 days


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


# ---- category (H2H categories) mode ----
from season import adds_left, cat_lines, cat_matchup, cat_pickups, cat_scores, so_far_from_scores  # noqa: E402

PTS = [{"stat": "PTS"}]


def test_cat_matchup_equal_teams_is_a_coin_flip_per_category():
    cats = [{"stat": "PTS"}, {"stat": "TO", "reverse": True}, {"stat": "FG%"}]
    L = {"PTS": (100, 400), "TO": (20, 9), "FGM": (40, 0), "FGA": (90, 0)}
    df, won = cat_matchup(L, L, {}, {}, cats)
    assert list(df.columns) == ["cat", "me", "opp", "p_win"] and list(df.cat) == ["PTS", "TO", "FG%"]
    assert df.p_win.tolist() == pytest.approx([0.5] * 3) and won == pytest.approx(1.5)


def test_cat_matchup_reverse_category_lower_wins():
    L = lambda to: {"PTS": (100, 100), "TO": (to, 9)}
    cats = [{"stat": "PTS"}, {"stat": "TO", "reverse": True}]
    df, _ = cat_matchup(L(10), L(20), {}, {}, cats)   # I turn it over half as often
    assert df.p_win[1] > 0.9 and df.p_win[0] == pytest.approx(0.5)
    assert cat_matchup(L(20), L(10), {}, {}, cats)[0].p_win[1] < 0.1
    assert cat_matchup(L(20), L(10), {}, {}, [{"stat": "TO"}])[0].p_win[0] > 0.9  # without reverse, more is better


def test_cat_matchup_ratio_uses_made_over_attempted_and_the_score_so_far():
    cats = [{"stat": "FG%"}]
    mine = {"FGM": (50, 0), "FGA": (100, 0)}      # 50 made, but only 50 %
    opp = {"FGM": (40, 0), "FGA": (70, 0)}        # fewer made, 57 %
    df, _ = cat_matchup(mine, opp, {}, {}, cats)
    assert df.me[0] == pytest.approx(0.5) and df.opp[0] == pytest.approx(40 / 70) and df.p_win[0] < 0.5
    df, _ = cat_matchup(mine, opp, {"FGM": 60, "FGA": 100}, {}, cats)   # 60 % head start: 110/200 vs 40/70
    assert df.me[0] == pytest.approx(0.55) and 0.3 < df.p_win[0] < 0.5
    assert cat_matchup({}, {}, {}, {}, cats)[0].p_win[0] == 0.5     # nothing played or projected: a tie


def test_cat_matchup_so_far_counts():
    df, won = cat_matchup({"PTS": (50, 0)}, {"PTS": (100, 0)}, {"PTS": 60}, {"PTS": 0}, PTS)
    assert (df.me[0], df.opp[0], df.p_win[0]) == (110, 100, 1.0) and won == 1.0


def test_cat_lines_starters_injuries_and_variance(world):
    b, ev, G = world
    fit, cap = slot_fit(b, S)
    evw = week_ev(b, ev + 1, WK)                      # +1: every player "plays"
    scale = (b.avail * b.injury_status.map({"DAY_TO_DAY": 0.5}).fillna(1.0)).values
    mean, var = cat_lines(b, evw, fit, cap, [0, 1, 2], scale, PTS)["PTS"]
    assert mean == 100 and var == pytest.approx(2 * (9 ** 2 + 6 ** 2))  # cv .3 x PTS, two starters, two days
    assert cat_lines(b, evw, fit, cap, [7], scale, PTS)["PTS"][0] == 50   # day-to-day: half of 50 x 2 days


def test_cat_pickups_rank_by_expected_categories_then_season_value(world):
    b, ev, G = world
    p = cat_pickups(b, b.PTS, G, mine=[0, 1, 2, 3], active=[0, 1, 2], free=[4, 5, 6, 7], wk=WK, ros=ROS,
                    settings={**S, "categories": PTS}, opp_lines={"PTS": (100, 100)}, so_me={}, so_opp={})
    assert (p["drop"] == "P2").all()                          # never P3, who sits in the IR slot
    assert list(p.index) == [4, 7, 6, 5]                      # X and W both add this week (W's 50 is more volatile, hurt by DTD)
    assert p.loc[4].cat_gain >= p.loc[7].cat_gain > 0 and list(p.cat_gain) == sorted(p.cat_gain, reverse=True)
    assert p.loc[6].cat_gain == 0 and p.loc[6].star and not p.loc[5].star
    assert (p.loc[7].ros_gain, p.loc[4].ros_gain, p.loc[6].ros_gain) == (120, 20, 80)
    assert p.loc[7].games == 1 and p.loc[6].games == 0


def test_cat_scores_and_so_far_from_espn_scoreboard():
    side = lambda t, by: {"teamId": t, "totalPoints": 0.0, "cumulativeScore": {"scoreByStat": by}}
    js = {"schedule": [{"matchupPeriodId": 2, "home": side(13, {"0": {"score": 90.0}, "19": {"score": 0.45}}),
                        "away": side(5, {"0": {"score": 80.0}})}]}
    sc = cat_scores(js, 2, 13)
    assert sc["me"] == {"PTS": 90.0, "FG%": 0.45} and sc["opp"] == {"PTS": 80.0} and sc["opp_id"] == 5
    assert cat_scores(js, 3, 13) is None
    pre = {"schedule": [{"matchupPeriodId": 1, "home": {"teamId": 13, "totalPoints": 0.0},
                         "away": {"teamId": 5, "totalPoints": 0.0}}]}
    assert cat_scores(pre, 1, 13)["me"] == {}                    # pre-season: no per-category data
    sf = so_far_from_scores(sc["me"], {"FGA": (60, 0)}, [{"stat": "PTS"}, {"stat": "FG%"}], 0.5)
    assert sf == {"PTS": 90.0, "FGA": 30.0, "FGM": pytest.approx(13.5)}   # % turned back into made/attempted


def test_adds_left():
    js = {"teams": [{"id": 13, "transactionCounter": {"matchupAcquisitionTotals": {"3": 1.0}}},
                    {"id": 2, "transactionCounter": {}}]}
    assert adds_left(js, 13, 3, 4) == 3 and adds_left(js, 13, 4, 4) == 4 and adds_left(js, 2, 3, 4) == 4
    assert adds_left(js, 99, 3, 4) is None
