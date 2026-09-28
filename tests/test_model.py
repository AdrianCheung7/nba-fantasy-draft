import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import RATES, STATS, fit, points, predict, season_table, shrunk_rates

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
                             "season": s, "player_id": p, "period": g})
    return pd.DataFrame(rows)


def test_points_triple_double_line():
    assert points(pd.DataFrame([LINE]), SCORING).iloc[0] == pytest.approx(71.5)


def test_points_uses_edited_weights():
    assert points(pd.DataFrame([LINE]), {**SCORING, "PTS": 2}).iloc[0] == pytest.approx(98.5)


def test_regression_recovers_stable_players():
    tab = season_table(fake_games())
    stat_m, gp_m = fit(tab, {}, [2023, 2024, 2025, 2026], 2027)
    pred = predict(stat_m, gp_m, tab, {}, 2027, 2027).set_index("player_id")
    for p in (0, 5, 9):
        assert pred.loc[p, "PTS"] == pytest.approx(p % 10 + 1, abs=0.6)
    assert pred.gp.between(27, 33).all()


def test_missing_season_player_projected():
    tab = season_table(fake_games(skip={(3, 2026)}))
    stat_m, gp_m = fit(tab, {}, [2023, 2024, 2025, 2026], 2027)
    pred = predict(stat_m, gp_m, tab, {}, 2027, 2027).set_index("player_id")
    assert 3 in pred.index and pred.loc[3, "PTS"] > 0


def test_shrunk_rates_pull_toward_league():
    g = fake_games(n_players=2, seasons=(2026,), games=10)
    g.loc[(g.player_id == 0) & (g.period == 0), "EJ"] = 1.0
    r = shrunk_rates(g, 2027).set_index("player_id").EJ
    assert r[0] == pytest.approx(5 / 70) and r[1] == pytest.approx(2 / 70)


from model import build

COLS = ["player_id", "name", "team_id", "positions", "adp", "espn_rank", "injury_status", "age", "gp", "cv", *STATS, *RATES]


def test_build_rookie_fallback_and_injury_scaling():
    games = fake_games()
    players = pd.DataFrame({"player_id": [0, 1, 999], "name": ["A", "B", "Rook"], "team_id": [1, 2, 3],
                            "positions": ["G|UTIL", "F|UTIL", "C|UTIL"], "adp": [1.0, 2.0, 3.0],
                            "injury_status": ["OUT", "ACTIVE", "ACTIVE"], "age": [25, 30, 19]})
    espn = pd.DataFrame([{"player_id": 999, "gp": 70.0, **{c: 5.0 for c in STATS}, **{c: 0.0 for c in RATES}},
                         {"player_id": 1, "gp": 10.0, **{c: 99.0 for c in STATS}, **{c: 0.0 for c in RATES}}])
    settings = {"season": 2027, "history_seasons": [2022, 2023, 2024, 2025, 2026], "scoring": SCORING}
    proj = build(games, players, espn, {0: 25, 1: 30}, settings).set_index("player_id")
    assert list(proj.reset_index().columns) == COLS
    assert set(proj.index) == {0, 1, 999}
    # rookie's ESPN gp is scaled by the model/ESPN gp ratio for player 1, the only veteran in both frames
    # (player 1 is ACTIVE, so its own gp is untouched by injury scaling and reflects the ratio directly)
    factor = proj.loc[1, "gp"] / 10.0
    assert proj.loc[999, "gp"] == pytest.approx(70.0 * factor) and proj.loc[999, "PTS"] == 5.0
    assert proj.loc[1, "PTS"] < 50                                              # veteran ignores ESPN
    assert proj.loc[0, "gp"] == pytest.approx(0.75 * proj.loc[1, "gp"], rel=0.1)  # OUT scaled
    assert proj.cv.notna().all() and proj[RATES].notna().all().all()
