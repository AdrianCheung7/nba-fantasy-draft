import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import FEATURES, GP_FEATURES, RATES, STATS, backtest, blend_espn, features, moved, primary_team, espn_split, fit, pick_weight, points, predict, season_table, shrunk_rates, team_budget

SCORING = {"FGM": 0.5, "FGMI": -0.5, "FTM": 1, "FTMI": -1, "3PM": 2, "3PMI": -1, "OREB": 1.5, "DREB": 1,
           "AST": 1, "STL": 2, "BLK": 2, "TO": -1, "EJ": -100, "FF": -2, "TF": -3, "TD": 10, "QD": 100,
           "PTS": 1, "TW": 1}
LINE = dict(FGM=10, FGA=18, FTM=5, FTA=6, **{"3PM": 2, "3PA": 4}, OREB=3, DREB=9, AST=10, STL=2, BLK=1,
            TO=3, PTS=27, EJ=0, FF=0, TF=0, TD=1, QD=0, TW=1)


def fake_games(n_players=40, seasons=(2022, 2023, 2024, 2025, 2026), games=30, skip=()):
    rows = []
    for p in range(n_players):
        for s in seasons:
            if (p, s) in skip:
                continue
            for g in range(games):
                rows.append({**{c: float(p % 10 + 1) for c in STATS}, **{c: 0.0 for c in RATES},
                             "season": s, "player_id": p, "period": g, "team_id": p % 3 + 1})
    return pd.DataFrame(rows)


def test_points_triple_double_line():
    assert points(pd.DataFrame([LINE]), SCORING).iloc[0] == pytest.approx(71.5)


def test_points_uses_edited_weights():
    assert points(pd.DataFrame([LINE]), {**SCORING, "PTS": 2}).iloc[0] == pytest.approx(98.5)


def test_regression_recovers_stable_players():
    tab = season_table(fake_games())
    stat_m, gp_m, _ = fit(tab, {}, [2023, 2024, 2025, 2026], 2027)
    pred = predict(stat_m, gp_m, tab, {}, 2027, 2027).set_index("player_id")
    for p in (0, 5, 9):
        assert pred.loc[p, "PTS"] == pytest.approx(p % 10 + 1, abs=0.6)
    assert pred.gp.between(27, 33).all()


def test_missing_season_player_projected():
    tab = season_table(fake_games(skip={(3, 2026)}))
    stat_m, gp_m, _ = fit(tab, {}, [2023, 2024, 2025, 2026], 2027)
    pred = predict(stat_m, gp_m, tab, {}, 2027, 2027).set_index("player_id")
    assert 3 in pred.index and pred.loc[3, "PTS"] > 0


def test_shrunk_rates_pull_toward_league():
    g = fake_games(n_players=2, seasons=(2026,), games=10)
    g.loc[(g.player_id == 0) & (g.period == 0), "EJ"] = 1.0
    r = shrunk_rates(g, 2027).set_index("player_id").EJ
    assert r[0] == pytest.approx(5 / 70) and r[1] == pytest.approx(2 / 70)


from model import build

COLS = ["player_id", "name", "team_id", "positions", "adp", "espn_rank", "injury_status", "age", "gp", "gp_k", "cv", *STATS, *RATES]


def test_build_rookie_fallback_and_injury_scaling():
    games = fake_games()
    players = pd.DataFrame({"player_id": [0, 1, 999], "name": ["A", "B", "Rook"], "team_id": [1, 2, 3],
                            "positions": ["G|UTIL", "F|UTIL", "C|UTIL"], "adp": [1.0, 2.0, 3.0],
                            "injury_status": ["OUT", "ACTIVE", "ACTIVE"], "age": [25, 30, 19]})
    espn = pd.DataFrame([{"player_id": 999, "gp": 70.0, **{c: 5.0 for c in STATS}, **{c: 0.0 for c in RATES}, "REB": 8.0},
                         {"player_id": 1, "gp": 10.0, **{c: 99.0 for c in STATS}, **{c: 0.0 for c in RATES}, "REB": 8.0}])
    settings = {"season": 2027, "history_seasons": [2022, 2023, 2024, 2025, 2026], "scoring": SCORING}
    proj = build(games, players, espn, {0: 25, 1: 30}, settings).set_index("player_id")
    assert list(proj.reset_index().columns) == COLS
    assert set(proj.index) == {0, 1, 999}
    # rookie's ESPN gp is scaled by the model/ESPN gp ratio for player 1, the only veteran in both frames
    # (player 1 is ACTIVE, so its own gp is untouched by injury scaling and reflects the ratio directly)
    factor = proj.loc[1, "gp"] / 10.0
    assert proj.loc[999, "gp"] == pytest.approx(70.0 * factor) and proj.loc[999, "PTS"] == 5.0
    assert proj.loc[999, "OREB"] == pytest.approx(8 * 0.27) and proj.loc[999, "DREB"] == pytest.approx(8 * 0.73)
    assert 40 < proj.loc[1, "PTS"] < 60                                         # veteran: 50/50 with ESPN's 99
    assert proj.loc[0, "gp"] == pytest.approx(0.75 * proj.loc[1, "gp"], rel=0.1)  # OUT scaled
    assert proj.cv.notna().all() and proj[RATES].notna().all().all()
    assert proj.gp_k.isna().all()                                               # no rotation players in the fake league


from model import blend


def test_blend_weights_current_season_by_games_played():
    proj = pd.DataFrame({"player_id": [0, 1, 2], "gp": [70.0] * 3, "cv": [0.3] * 3,
                         **{c: [10.0] * 3 for c in STATS + RATES}})
    cur = pd.DataFrame([{"player_id": p, **{c: 20.0 for c in STATS + RATES}} for p in [0] * 20 + [1] * 60])
    out = blend(proj, cur).set_index("player_id")
    assert out.loc[0, "PTS"] == pytest.approx(15.0) and out.loc[0, "TD"] == pytest.approx(15.0)  # 20/(20+20) = 0.5
    assert out.loc[1, "PTS"] == pytest.approx(17.5)                                               # 60/80
    assert out.loc[2, "PTS"] == 10.0                                                              # no games: untouched
    assert (out.gp == 70).all() and (out.cv == 0.3).all()


def espn_frame(bump=0.0, players=range(40)):
    """ESPN preseason projection = the (constant) truth of fake_games + bump."""
    return pd.DataFrame([{"player_id": p, "gp": 30.0, **{c: p % 10 + 1.0 + bump for c in STATS},
                          "REB": 2 * (p % 10 + 1.0) + bump} for p in players])  # OREB = DREB = the stat


def test_espn_split_uses_prior_season_oreb_share():
    tab = pd.DataFrame({"player_id": [1, 2], "season": [2026, 2026], "OREB": [1.0, 0.0], "DREB": [3.0, 0.0]})
    e = espn_split(pd.DataFrame({"player_id": [1, 2, 3], "REB": [8.0, 8.0, 10.0]}), tab, 2026).set_index("player_id")
    assert e.OREB.tolist() == pytest.approx([2.0, 8 * 0.27, 2.7]) and (e.OREB + e.DREB).tolist() == [8.0, 8.0, 10.0]


def test_blend_espn_mixes_stats_not_gp():
    proj = pd.DataFrame({"player_id": [1, 2], "gp": [60.0, 60.0], **{c: [10.0, 10.0] for c in STATS}})
    e = pd.DataFrame({"player_id": [1], "gp": [82.0], **{c: [20.0] for c in STATS}})
    out = blend_espn(proj, e, 0.25).set_index("player_id")
    assert out.PTS.tolist() == [17.5, 10.0] and out.gp.tolist() == [60.0, 60.0]  # 2 has no ESPN line


def test_pick_weight_follows_the_better_source():
    tab = season_table(fake_games())
    exact = {t: espn_frame() for t in (2025, 2026)}
    junk = {t: espn_frame(bump=30) for t in (2025, 2026)}
    assert pick_weight(tab, {}, exact, SCORING, 2027) == 0.0
    assert pick_weight(tab, {}, junk, SCORING, 2027) == 1.0
    assert pick_weight(tab, {}, {2026: exact[2026]}, SCORING, 2027) == 0.5  # < 2 seasons: default


def test_backtest_reports_each_stage_per_season():
    r = backtest(fake_games(), {}, {"season": 2027, "scoring": SCORING}, {t: espn_frame() for t in (2025, 2026)})
    assert list(r.stage.unique()) == ["last season", "baseline", "+A", "+A+B", "+A+B+C"] and set(r.season) == {2025, 2026}
    assert r.attrs["w"] == 0.0 and (r[r.stage == "+A"].mae < 1e-9).all()  # ESPN was exact, so all weight goes to it


def team_frame(**cols):
    return pd.DataFrame({**{c: 0.0 for c in STATS}, "MIN": 30.0, **cols})


def test_team_budget_pulls_team_rates_toward_league_more_for_movers():
    p = team_frame(PTS=[20.0, 20.0, 10.0, 10.0])  # team 1 scores 20 per 30 min, team 2 10; league 15
    out = team_budget(p, [1, 1, 2, 2], [False, True, False, False])
    assert out.PTS.tolist() == pytest.approx([20 * 0.75 ** 0.5, 20 * 0.75, 10 * 1.5 ** 0.5, 10 * 1.5 ** 0.5])
    assert out.MIN.tolist() == [30.0] * 4                                    # only the shot budget moves
    out = team_budget(p, [0, 1, 2, float("nan")], [False] * 4)  # rows without a team stay as they are
    assert out.PTS.tolist() == pytest.approx([20, 20 * 0.75 ** 0.5, 10 * 1.5 ** 0.5, 10])


def test_team_budget_keeps_made_within_attempted():
    p = team_frame(FGM=[10.0, 10.0, 30.0, 30.0], FGA=[10.0, 10.0, 20.0, 20.0])  # odd inputs: FGM scales x2, FGA x1.5
    out = team_budget(p, [1, 1, 2, 2], [True] * 4)
    assert (out.FGM <= out.FGA).all() and out.FGM[0] == pytest.approx(15.0)


def test_primary_team_and_moved():
    g = pd.DataFrame({"player_id": [1, 1, 1, 2], "team_id": [5, 6, 6, 7], "MIN": [20.0, 10.0, 12.0, 20.0]})
    old = primary_team(g)
    assert old.to_dict() == {1: 6, 2: 7}                                     # most minutes
    assert moved([1, 2, 3], np.array([6, 9, 4]), old).tolist() == [False, True, False]  # 3 is a rookie: not moved


def test_gp_model_uses_lean_features_with_age_squared():
    tab = season_table(fake_games())
    ages = {p: 20.0 + p for p in range(40)}
    _, gp_m, _ = fit(tab, ages, [2023, 2024, 2025, 2026], 2027)
    assert GP_FEATURES == ["GP_1", "GP_2", "GP_3", "MIN_1", "MIN_2", "MIN_3", "age", "age2"]
    assert gp_m.n_features_in_ == 8
    assert fit(tab, ages, [2023, 2024, 2025, 2026], 2027, FEATURES)[1].n_features_in_ == len(FEATURES)  # v2 (backtest)
    f = features(tab, ages, 2026, 2027)
    assert (f.age2 == f.age ** 2).all() and f.age.iloc[0] == 20.0 - 1


def test_fit_kappa_by_method_of_moments_on_rotation_players():
    rng = np.random.default_rng(0)
    tab = pd.DataFrame([{"player_id": p, "season": s, **{c: 30.0 for c in STATS}, "GP": float(rng.binomial(82, 0.75))}
                        for p in range(300) for s in range(2022, 2027)])
    kappa = fit(tab, {}, [2023, 2024, 2025, 2026], 2027)[2]
    assert 60 < kappa < 110          # plain binomial luck (no extra injury clumping): kappa ~ games - 1 = 81
    assert np.isnan(fit(season_table(fake_games()), {}, [2023, 2024, 2025, 2026], 2027)[2])  # nobody plays 24+ MIN


def test_blend_with_no_games_yet_leaves_projection_alone():
    proj = pd.DataFrame({"player_id": [0], "gp": [70.0], "cv": [0.3], **{c: [10.0] for c in STATS + RATES}})
    out = blend(proj, pd.DataFrame(columns=["season", "player_id", "period", *STATS, *RATES]))
    assert out.PTS.dtype == float and out.PTS.tolist() == [10.0]  # fetch.py writes a header-only file before tip-off


from model import cat_value, value_pg

CATS = {"format": "categories", "teams": 2, "rounds": 2, "categories": [
    {"stat": s, "reverse": s in ("PF", "TO")} for s in
    ("PTS", "BLK", "STL", "AST", "OREB", "DREB", "PF", "TO", "FG%", "FT%", "3P%", "TW")]}


def cat_frame(n=8, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({c: r.uniform(1, 10, n) for c in STATS + RATES + ["PF"]})
    for m, a, k in (("FGM", "FGA", 2), ("FTM", "FTA", 1.3), ("3PM", "3PA", 2.8)):
        d[a] = d[m] * k * r.uniform(0.8, 1.2, n)
    d["TW"] = r.uniform(0.3, 0.7, n)
    return d.assign(gp=70.0)


def test_cat_value_reverse_and_volume_weighted_percentages():
    d = cat_frame()
    v = cat_value(d, CATS)
    worse = d.copy()
    worse.loc[0, "TO"] += 3  # more turnovers: lower value
    assert cat_value(worse, CATS)[0] < v[0]
    hot = d.copy()  # 90% on 10 shots beats 90% on 1 shot
    hot.loc[0, ["FGM", "FGA"]] = [9, 10]
    lo = d.copy()
    lo.loc[0, ["FGM", "FGA"]] = [0.9, 1]
    assert cat_value(hot, CATS)[0] > cat_value(lo, CATS)[0]
    neg = d.copy()  # pct impact is relative to the league rate: 10% on 10 shots hurts
    neg.loc[0, ["FGM", "FGA"]] = [1, 10]
    assert cat_value(neg, CATS)[0] < cat_value(lo, CATS)[0]


def test_cat_value_replacement_player_is_zero():
    v = cat_value(cat_frame(), CATS)
    n = CATS["teams"] * CATS["rounds"]
    top = np.sort(v.values)[::-1]
    assert top[n] == pytest.approx(0)  # the best undrafted player (equal gp)
    assert (top[:n] > 0).all()


def test_cat_value_skips_missing_pf_with_warning():
    d = cat_frame().drop(columns="PF")
    with pytest.warns(UserWarning, match="PF"):
        v = cat_value(d, CATS)
    assert v.notna().all()


def test_value_pg_dispatches_on_format():
    d = cat_frame()
    assert value_pg(d, {"scoring": {"PTS": 1, "TO": -1}}).equals(points(d, {"PTS": 1, "TO": -1}))
    assert value_pg(d, {"format": "points", "scoring": {"PTS": 1}}).equals(points(d, {"PTS": 1}))
    assert value_pg(d, CATS).equals(cat_value(d, CATS))
