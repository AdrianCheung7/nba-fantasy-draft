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
SHOT = ["PTS", "FGM", "FGA", "FTM", "FTA", "3PM", "3PA", "TO"]  # stats a team's shot budget is shared out of
OREB_SHARE = 0.27  # ponytail: league OREB/REB, for ESPN players with no prior-season line


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


def clip_made(d):
    for made, att in (("FGM", "FGA"), ("FTM", "FTA"), ("3PM", "3PA")):
        d[made] = d[[made, att]].min(axis=1)
    return d


def predict(stat_m, gp_m, tab, ages, target, season):
    f = features(tab, ages, target, season)
    out = clip_made(pd.DataFrame(stat_m.predict(f[FEATURES]), columns=STATS).clip(lower=0))
    out.insert(0, "player_id", f.player_id.values)
    out["gp"] = np.clip(gp_m.predict(f[FEATURES]), 0, 82)
    return out


def fpg(d, scoring):
    """Fantasy points per game of a per-game stat frame, indexed by player_id (rare-event rates ignored)."""
    return pd.Series(points(d.assign(**{c: 0.0 for c in RATES}), scoring).values, index=d.player_id.values)


def model_pred(tab, ages, test, season):
    """The regression's projection of season `test`, trained only on seasons before it."""
    seasons = set(tab.season)
    stat_m, gp_m = fit(tab, ages, [s for s in range(test - 3, test) if s - 1 in seasons], season)
    return predict(stat_m, gp_m, tab, ages, test, season)


def espn_split(e, tab, prev):
    """ESPN projects total rebounds only: split REB into OREB/DREB by the player's `prev`-season OREB share."""
    p = tab[tab.season == prev].set_index("player_id")
    share = (p.OREB / (p.OREB + p.DREB)).reindex(e.player_id).fillna(OREB_SHARE).values
    return e.assign(OREB=e.REB * share, DREB=e.REB * (1 - share))


def blend_espn(proj, e, w):
    """Per-game STATS = w * model + (1 - w) * ESPN where ESPN has the player; gp stays the model's."""
    x = e.set_index("player_id")[STATS].reindex(proj.player_id).values
    out = proj.copy()
    out[STATS] = np.where(np.isnan(x), proj[STATS].values, w * proj[STATS].values + (1 - w) * np.nan_to_num(x))
    return out


def primary_team(g):
    """player_id -> the team he played the most minutes for in these game logs."""
    return g.groupby(["player_id", "team_id"]).MIN.sum().groupby(level=0).idxmax().map(lambda i: i[1])


def moved(pids, team, old):
    """True where the player's `team` differs from `old` (pid -> last season's primary team); rookies: False."""
    o = pd.Series(pids).map(old).values
    return ~np.isnan(o) & (o != team)


def team_budget(p, team, mv):
    """Usage competition: SHOT stats *= (league rate / team rate) ** lam, rate = stat per projected minute over the
    team's players, lam = 1 for players who changed teams (a new roster to share with), else 0.5.
    `team`, `mv`: per-row team id and moved flag; rows without a team are left alone."""
    p, t = p.copy(), pd.Series(np.asarray(team, float), index=p.index)
    ok = (t > 0).values
    q, t, lam = p[ok], t[ok], np.where(np.asarray(mv)[ok], 1.0, 0.5)
    for c in SHOT:
        q[c] = q[c] * ((q[c].sum() / q.MIN.sum()) / (q[c].groupby(t).transform("sum") / q.MIN.groupby(t).transform("sum"))) ** lam
    p.loc[ok, SHOT] = q[SHOT]
    return clip_made(p)


def score(pred, tab, test, scoring, top=150):
    """FP/g MAE of a projection vs the season's actuals (players with 20+ games): all, and the `top` by projected
    season FP."""
    actual = tab[(tab.season == test) & (tab.GP >= 20)]
    d = pd.concat({"a": fpg(actual, scoring), "p": fpg(pred, scoring)}, axis=1, join="inner")
    d["val"] = d.p * pred.set_index("player_id").gp.reindex(d.index)
    e = (d.p - d.a).abs()
    return {"n": len(d), "mae": e.mean(), "mae_top": e[d.val.nlargest(top).index].mean()}


def pick_weight(tab, ages, hist, scoring, season):
    """Model weight w in {0, .1 .. 1} with the lowest FP/g MAE for w * model + (1 - w) * ESPN over the past seasons
    that have an ESPN preseason projection in `hist` ({season: frame}); 0.5 when fewer than 2 seasons."""
    ts = [t for t in hist if t - 2 in set(tab.season)]  # need a season to train on and one before it for lags
    if len(ts) < 2:
        return 0.5
    ws = np.arange(11) / 10
    err = np.zeros(11)
    for t in ts:
        pred, e = model_pred(tab, ages, t, season), espn_split(hist[t], tab, t - 1)
        err += [score(blend_espn(pred, e, w), tab, t, scoring)["mae"] for w in ws]
    return ws[err.argmin()]


def shrunk_rates(games, target, k=SHRINK_K):
    w = games.season.map({target - 1: 3, target - 2: 2, target - 3: 1}).fillna(0)
    league = games.loc[w > 0, RATES].mean()
    num = games[RATES].mul(w, axis=0).groupby(games.player_id).sum()
    den = w.groupby(games.player_id).sum()
    return num.add(k * league, axis=1).div(den + k, axis=0).reset_index()


INJURY_GP = {"OUT": 0.75, "INJURY_RESERVE": 0.5, "SUSPENSION": 0.9}  # ponytail: calibration knob
OUT_COLS = ["player_id", "name", "team_id", "positions", "adp", "espn_rank", "injury_status", "age", "gp", "cv", *STATS, *RATES]


def fp_cv(games, scoring, target):
    g = games[games.season >= target - 2]
    g = g.assign(fp=points(g, scoring))
    s = g.groupby("player_id").fp.agg(["mean", "std", "size"])
    cv = (s["std"] / s["mean"].clip(lower=5)).where(s["size"] >= 15)
    return cv.fillna(cv.median()).rename("cv").reset_index()


def build(games, players, espn_proj, ages, settings, espn_hist=None):
    season = settings["season"]
    tab = season_table(games)
    seasons = set(tab.season)
    stat_m, gp_m = fit(tab, ages, [s for s in settings["history_seasons"] if s - 1 in seasons], season)
    espn_proj = espn_split(espn_proj, tab, season - 1)
    w = pick_weight(tab, ages, espn_hist or {}, settings["scoring"], season)
    proj = blend_espn(predict(stat_m, gp_m, tab, ages, season, season), espn_proj, w)
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
    last = primary_team(games[games.season == season - 1])
    out = team_budget(out, out.team_id, moved(out.player_id, out.team_id.values, last))
    if "espn_rank" not in out:  # players.csv from before fetch.py added the column
        out["espn_rank"] = 999
    out["gp"] = out.gp * out.injury_status.map(INJURY_GP).fillna(1.0)
    return out[OUT_COLS]


def blend(proj, cur_games, k=20):
    """D9: per-game rates = w * this season's + (1 - w) * projected, w = n / (n + k), n = games played."""
    cols = STATS + RATES
    g = cur_games.groupby("player_id")
    n = g.size().reindex(proj.player_id).fillna(0).values[:, None]
    cur = g[cols].mean().reindex(proj.player_id).values
    w = n / (n + k)
    out = proj.copy()
    out[cols] = np.where(n > 0, w * np.nan_to_num(cur) + (1 - w) * proj[cols].values, proj[cols].values)
    return out


def backtest(games, ages, settings, hist):
    """Per test season: FP/g MAE of each stage (all players with 20+ games, and the top 150 by projected season FP)
    vs 'repeat last season'. hist = {season: ESPN preseason projection}. attrs['w'] = the ESPN blend weight."""
    tab, sc, season = season_table(games), settings["scoring"], settings["season"]
    w = pick_weight(tab, ages, hist, sc, season)
    rows = []
    for t in sorted(hist):
        if t - 2 not in set(tab.season):
            continue
        base = model_pred(tab, ages, t, season)
        last = features(tab, ages, t, season).rename(columns={f"{c}_1": c for c in STATS}).assign(gp=lambda d: d.GP_1)
        a = blend_espn(base, espn_split(hist[t], tab, t - 1), w)
        old = primary_team(games[games.season == t - 1])
        first = games[games.season == t].sort_values("period").groupby("player_id").team_id.first()
        team = base.player_id.map(first).fillna(base.player_id.map(old)).values  # preseason team, roughly
        stages = {"last season": last, "baseline": base, "+A": a,
                  "+A+B": team_budget(a, team, moved(base.player_id, team, old))}
        rows += [{"season": t, "stage": k, **score(v, tab, t, sc)} for k, v in stages.items()]
    out = pd.DataFrame(rows)
    out.attrs["w"] = w
    return out


def main():
    s = json.loads((ROOT / "settings.json").read_text())
    games = pd.concat([pd.read_csv(DATA / f"games_{y}.csv") for y in s["history_seasons"]], ignore_index=True)
    a = pd.read_csv(DATA / "ages.csv")
    ages = dict(zip(a.player_id, a.age))
    hist = {y: pd.read_csv(DATA / f"espn_proj_{y}.csv") for y in s["history_seasons"]
            if (DATA / f"espn_proj_{y}.csv").exists()}
    if "--backtest" in sys.argv:
        r = backtest(games, ages, s, hist)
        print(f"ESPN blend weight on the model: {r.attrs['w']}")
        for c in ("mae", "mae_top"):
            t = r.pivot(index="stage", columns="season", values=c).reindex(r.stage.unique())
            print(f"\nFP/g {c}" + (" (top 150)" if c == "mae_top" else ""))
            print(t.assign(avg=t.mean(axis=1)).round(3).to_string())
        return
    proj = build(games, pd.read_csv(DATA / "players.csv"), pd.read_csv(DATA / "espn_proj.csv"), ages, s, hist)
    cur = DATA / f"games_{s['season']}.csv"  # this season's game logs (fetch.py), once it has any
    if cur.exists():
        proj = blend(proj, pd.read_csv(cur))
    proj.to_csv(DATA / "projections.csv", index=False)
    top = proj.assign(fp=points(proj, s["scoring"])).nlargest(15, "fp")
    print(top[["name", "positions", "gp", "fp"]].to_string(index=False))


if __name__ == "__main__":
    main()
