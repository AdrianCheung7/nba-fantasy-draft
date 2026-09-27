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
