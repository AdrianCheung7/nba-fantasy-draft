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
    rookies = espn_proj[~espn_proj.player_id.isin(proj.player_id)][["player_id", "gp", *STATS]].copy()
    # ESPN's own gp projections run ~20% higher than the model's for players in both frames (F2): scale
    # rookies (who have no history, so only ESPN gp is available) to the model's scale by that ratio.
    vets = proj[["player_id", "gp"]].merge(
        espn_proj[["player_id", "gp"]].rename(columns={"gp": "espn_gp"}), on="player_id")
    factor = (vets.gp / vets.espn_gp).median() if len(vets) else 1.0
    rookies["gp"] = rookies.gp * factor
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
