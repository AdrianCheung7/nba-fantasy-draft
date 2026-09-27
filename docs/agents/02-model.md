# 02 — Projection Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Invoke the `ponytail` skill before writing code.

**Agent model:** `sonnet`

**Goal:** Project every active player's 2026-27 per-game stats, games played and game-to-game volatility, and write `data/projections.csv`.

**Architecture:** `model.py` — pure functions over pandas DataFrames. Ridge regression (multi-output) predicts per-game box-score stats from the prior 3 seasons + age; a second Ridge predicts games played; rare events (EJ/FF/TF/TD/QD) and team wins (TW) use shrunk weighted rates. Fantasy points are never stored — `points(df, scoring)` computes them on demand so scoring edits need no retraining.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn (`Ridge`, `StandardScaler`, `make_pipeline`), pytest.

**Spec:** [../spec.md](../spec.md) — section 3.1.

## Global Constraints

- Inputs are the CSVs from 01-data (contracts in [README.md](README.md#shared-contracts-all-agents-rely-on-these-do-not-rename)); output `data/projections.csv` must have exactly the contract columns.
- ESPN season ids: `2027` = 2026-27 (target), `2022..2026` = history.
- Scoring keys: `FGM FGMI FTM FTMI 3PM 3PMI OREB DREB AST STL BLK TO EJ FF TF TD QD PTS TW`.
- No new dependencies. No classes. Knobs are module constants with a `ponytail:` comment.

## Review Focus

1. Player who missed an entire season (injury) → still projected (from older seasons), with lower GP (Task 1 test `test_missing_season_player_projected`).
2. Rookie with no NBA history → uses ESPN projection, not dropped (Task 2 test).
3. Scoring weights edited → `points()` reflects new weights with no refit (Task 1 test).
4. Player with missing age → median age, not NaN crash (Task 1 test uses empty `ages`).
5. Injured-at-draft player → GP scaled by `INJURY_GP` (Task 2 test).

---

### Task 1: Scoring, features, regression

**Files:**
- Create: `model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `data/games_*.csv` columns `season, player_id, period, PTS, BLK, STL, AST, OREB, DREB, EJ, FF, TF, TO, FGM, FGA, FTM, FTA, 3PM, 3PA, TD, QD, MIN, TW`.
- Produces: `STATS: list[str]`, `RATES: list[str]`, `points(df, scoring: dict) -> pd.Series`, `season_table(games) -> DataFrame[player_id, season, *STATS, GP]`, `features(tab, ages: dict, target: int, season: int) -> DataFrame`, `fit(tab, ages, targets: list[int], season: int) -> (stat_model, gp_model)`, `predict(stat_model, gp_model, tab, ages, target, season) -> DataFrame[player_id, *STATS, gp]`, `shrunk_rates(games, target) -> DataFrame[player_id, *RATES]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_model.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Implement `model.py` (part 1)**

```python
"""Regression projections for the 2026-27 season.  Run: python model.py [--backtest]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data"
STATS = ["PTS", "FGM", "FGA", "FTM", "FTA", "3PM", "3PA", "OREB", "DREB", "AST", "STL", "BLK", "TO", "MIN"]
RATES = ["EJ", "FF", "TF", "TD", "QD", "TW"]  # rare or team-driven: shrunk rates, not regression
FEATURES = [f"{c}_{k}" for c in STATS + ["GP"] for k in (1, 2, 3)] + ["age"]
SHRINK_K = 40  # ponytail: prior strength in weighted games; tune if rates look too flat/noisy
MIN_GP_TRAIN = 10  # seasons shorter than this are too noisy to be stat targets


def points(df, scoring):
    d = df.assign(FGMI=df.FGA - df.FGM, FTMI=df.FTA - df.FTM, **{"3PMI": df["3PA"] - df["3PM"]})
    return sum(w * d[k] for k, w in scoring.items())


def season_table(games):
    g = games.groupby(["player_id", "season"])
    t = g[STATS].mean()
    t["GP"] = g.size()
    return t.reset_index()


def features(tab, ages, target, season):
    """One row per player seen in the 3 seasons before `target`."""
    base = pd.DataFrame({"player_id": tab[tab.season.between(target - 3, target - 1)].player_id.unique()})
    for k in (1, 2, 3):
        lag = tab[tab.season == target - k].drop(columns="season")
        base = base.merge(lag.rename(columns={c: f"{c}_{k}" for c in STATS + ["GP"]}), on="player_id", how="left")
    for c in STATS:  # missing season -> copy nearest known season
        base[f"{c}_1"] = base[f"{c}_1"].fillna(base[f"{c}_2"]).fillna(base[f"{c}_3"])
        base[f"{c}_2"] = base[f"{c}_2"].fillna(base[f"{c}_1"])
        base[f"{c}_3"] = base[f"{c}_3"].fillna(base[f"{c}_2"])
    for k in (1, 2, 3):
        base[f"GP_{k}"] = base[f"GP_{k}"].fillna(0)  # missed season = 0 games: that IS the signal
    age_now = base.player_id.map(ages).astype(float)  # ages are as of the draft season
    base["age"] = (age_now.fillna(age_now.median()) - (season - target)).fillna(27)
    return base


def fit(tab, ages, targets, season):
    X, Y, Xg, Yg = [], [], [], []
    for t in targets:
        f = features(tab, ages, t, season)
        y = f[["player_id"]].merge(tab[tab.season == t], on="player_id", how="left")
        Xg.append(f[FEATURES])
        Yg.append(y.GP.fillna(0))
        played = (y.GP >= MIN_GP_TRAIN).values
        X.append(f[FEATURES][played])
        Y.append(y[STATS][played])
    ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10))
    return ridge().fit(pd.concat(X), pd.concat(Y)), ridge().fit(pd.concat(Xg), pd.concat(Yg))


def predict(stat_m, gp_m, tab, ages, target, season):
    f = features(tab, ages, target, season)
    out = pd.DataFrame(stat_m.predict(f[FEATURES]), columns=STATS).clip(lower=0)
    for made, att in (("FGM", "FGA"), ("FTM", "FTA"), ("3PM", "3PA")):
        out[made] = out[[made, att]].min(axis=1)
    out.insert(0, "player_id", f.player_id.values)
    out["gp"] = np.clip(gp_m.predict(f[FEATURES]), 0, 82)
    return out


def shrunk_rates(games, target, k=SHRINK_K):
    w = games.season.map({target - 1: 3, target - 2: 2, target - 3: 1}).fillna(0)
    league = games.loc[w > 0, RATES].mean()
    num = games[RATES].mul(w, axis=0).groupby(games.player_id).sum()
    den = w.groupby(games.player_id).sum()
    return num.add(k * league, axis=1).div(den + k, axis=0).reset_index()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: 5 passed. If `test_regression_recovers_stable_players` misses by a little, lower `Ridge(alpha=...)` to 3 — do not loosen the test.

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: ridge stat projections and shrunk event rates"
```

### Task 2: Build projections, backtest, CLI

**Files:**
- Modify: `model.py` (append)
- Test: `tests/test_model.py` (append)

**Interfaces:**
- Consumes: `data/players.csv`, `data/espn_proj.csv`, `data/ages.csv`, `settings.json`.
- Produces: `INJURY_GP: dict`, `fp_cv(games, scoring, target) -> DataFrame[player_id, cv]`, `build(games, players, espn_proj, ages, settings) -> DataFrame` (projections contract), `backtest(games, ages, settings) -> dict[n, mae_model, mae_last]`, `data/projections.csv`.

- [ ] **Step 1: Append failing test**

```python
from model import build

COLS = ["player_id", "name", "team_id", "positions", "adp", "injury_status", "age", "gp", "cv", *STATS, *RATES]


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
    assert proj.loc[999, "gp"] == 70.0 and proj.loc[999, "PTS"] == 5.0          # rookie uses ESPN
    assert proj.loc[1, "PTS"] < 50                                              # veteran ignores ESPN
    assert proj.loc[0, "gp"] == pytest.approx(0.75 * proj.loc[1, "gp"], rel=0.1)  # OUT scaled
    assert proj.cv.notna().all() and proj[RATES].notna().all().all()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_model.py::test_build_rookie_fallback_and_injury_scaling -v`
Expected: FAIL — `ImportError: cannot import name 'build'`

- [ ] **Step 3: Append implementation to `model.py`**

```python
INJURY_GP = {"OUT": 0.75, "INJURY_RESERVE": 0.5, "SUSPENSION": 0.9}  # ponytail: calibration knob
OUT_COLS = ["player_id", "name", "team_id", "positions", "adp", "injury_status", "age", "gp", "cv", *STATS, *RATES]


def fp_cv(games, scoring, target):
    g = games[games.season >= target - 2]
    g = g.assign(fp=points(g, scoring))
    s = g.groupby("player_id").fp.agg(["mean", "std", "size"])
    cv = (s["std"] / s["mean"].clip(lower=5)).where(s["size"] >= 15)
    return cv.fillna(cv.median()).rename("cv").reset_index()


def build(games, players, espn_proj, ages, settings):
    season = settings["season"]
    tab = season_table(games)
    seasons = set(tab.season)
    stat_m, gp_m = fit(tab, ages, [s for s in settings["history_seasons"] if s - 1 in seasons], season)
    proj = predict(stat_m, gp_m, tab, ages, season, season)
    rookies = espn_proj[~espn_proj.player_id.isin(proj.player_id)][["player_id", "gp", *STATS]]
    proj = pd.concat([proj, rookies], ignore_index=True)
    proj = proj.merge(shrunk_rates(games, season), on="player_id", how="left")
    proj[RATES] = proj[RATES].fillna(games[RATES].mean())
    cv = fp_cv(games, settings["scoring"], season)
    proj = proj.merge(cv, on="player_id", how="left")
    proj["cv"] = proj.cv.fillna(cv.cv.median())
    out = players.merge(proj, on="player_id")
    out["gp"] = out.gp * out.injury_status.map(INJURY_GP).fillna(1.0)
    return out[OUT_COLS]


def backtest(games, ages, settings):
    """Fit on seasons before the last one, predict it, compare with 'repeat last season'."""
    tab = season_table(games)
    test = int(tab.season.max())
    seasons = set(tab.season)
    stat_m, gp_m = fit(tab, ages, [s for s in range(test - 3, test) if s - 1 in seasons], settings["season"])
    pred = predict(stat_m, gp_m, tab, ages, test, settings["season"])
    last = features(tab, ages, test, settings["season"]).rename(columns={f"{c}_1": c for c in STATS})
    actual = tab[(tab.season == test) & (tab.GP >= 20)]
    fp = lambda d: pd.Series(points(d.assign(**{c: 0.0 for c in RATES}), settings["scoring"]).values,
                             index=d.player_id.values)
    df = pd.concat({"actual": fp(actual), "model": fp(pred), "last": fp(last)}, axis=1, join="inner")
    return {"season": test, "n": len(df),
            "mae_model": round((df.model - df.actual).abs().mean(), 2),
            "mae_last": round((df["last"] - df.actual).abs().mean(), 2)}


def main():
    s = json.loads((ROOT / "settings.json").read_text())
    games = pd.concat([pd.read_csv(DATA / f"games_{y}.csv") for y in s["history_seasons"]], ignore_index=True)
    a = pd.read_csv(DATA / "ages.csv")
    ages = dict(zip(a.player_id, a.age))
    if "--backtest" in sys.argv:
        print(backtest(games, ages, s))
        return
    proj = build(games, pd.read_csv(DATA / "players.csv"), pd.read_csv(DATA / "espn_proj.csv"), ages, s)
    proj.to_csv(DATA / "projections.csv", index=False)
    top = proj.assign(fp=points(proj, s["scoring"])).nlargest(15, "fp")
    print(top[["name", "positions", "gp", "fp"]].to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all model tests**

Run: `pytest tests/test_model.py -v`
Expected: 6 passed.

- [ ] **Step 5: Backtest on real data** (requires 01-data output)

Run: `python model.py --backtest`
Expected: a dict like `{'season': 2026, 'n': 3xx, 'mae_model': X, 'mae_last': Y}` with **X < Y**. If X ≥ Y, try `Ridge(alpha=1)` and `alpha=30` and keep the best; report all three results. Do not add features or models beyond that without asking the orchestrator.

- [ ] **Step 6: Build projections**

Run: `python model.py`
Expected: writes `data/projections.csv` and prints a top-15 list of well-known stars (e.g. Jokic near the top — he's a points-league monster under this scoring). Sanity: no `gp` > 82, no negative stats.

- [ ] **Step 7: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: build projections with rookie fallback, injury scaling, backtest"
```

Report back: backtest dict(s), top-15 printout, row count of `projections.csv`.
