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
