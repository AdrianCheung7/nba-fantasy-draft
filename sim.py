"""Strategy simulator: simulated drafts x simulated H2H seasons.
Run: python sim.py --slot 5 [--drafts 20 --seasons 20]           best strategy per slot
     python sim.py --report --slot 9 [--drafts 20 --seasons 20]  our best strategy vs 13 near-optimal teams"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from draft import (STRATEGIES, board, day_matrix, espn_rank, ev_matrix, feasible, lineup, recommend, slot_fit,
                   snake)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
# ponytail: opponent pick noise (lognormal sigma); 0 = clones of the greedy recommender. Calibrated so
# greedy-vs-greedy slot 7 (14 teams, 20x20) p_playoffs lands in 6/14 +/- 0.10: 0.1 -> 0.755, 0.03 -> 0.342,
# 0.02 -> 0.285, 0.01 -> 0.188. Re-calibrate if league size or projections change a lot.
NOISE = 0.03
OPPONENTS = ("greedy", "espn")
LEAGUE_KEYS = ("teams", "rounds", "slots", "scoring", "playoff_teams", "playoff_weeks", "playoff_round_periods")


def round_robin(n):
    """Circle method. Each round is a list of (a, b) pairs; -1 marks a bye."""
    t = list(range(n)) + ([-1] if n % 2 else [])
    m = len(t)
    rounds = []
    for _ in range(m - 1):
        rounds.append([(t[i], t[m - 1 - i]) for i in range(m // 2)])
        t = [t[0], t[-1]] + t[1:-1]
    return rounds


def bracket_order(size):
    order = [1]
    while len(order) < size:
        n = len(order) * 2
        order = [x for s in order for x in (s, n + 1 - s)]
    return order


def regular_season(weekly, n_reg):
    wins = np.zeros(weekly.shape[0])
    rr = round_robin(weekly.shape[0])
    for w in range(n_reg):
        for a, c in rr[w % len(rr)]:
            if a < 0 or c < 0:
                continue
            pa, pc = weekly[a, w], weekly[c, w]
            wins[a] += (pa > pc) + 0.5 * (pa == pc)
            wins[c] += (pc > pa) + 0.5 * (pa == pc)
    return np.lexsort((-weekly[:, :n_reg].sum(axis=1), -wins)), wins  # wins, then points-for


def playoffs(seeds, weekly, weeks, round_periods=1):
    """Seeded single elimination; top seeds get byes; each round sums `round_periods` matchup periods and the
    rounds use the last periods of `weeks`."""
    size = 1 << (len(seeds) - 1).bit_length()
    rounds = size.bit_length() - 1
    need = rounds * round_periods
    if need > len(weeks):
        raise ValueError(f"{len(seeds)} playoff teams need {need} playoff weeks, have {len(weeks)}")
    alive = [seeds[s - 1] if s <= len(seeds) else -1 for s in bracket_order(size)]
    use = list(weeks[len(weeks) - need:])
    for r in range(rounds):
        pts = weekly[:, use[r * round_periods:(r + 1) * round_periods]].sum(axis=1)
        alive = [c if a < 0 else a if c < 0 else (a if pts[a] >= pts[c] else c)
                 for a, c in zip(alive[::2], alive[1::2])]
    return alive[0]


def validate(settings, slot):
    teams, p, w = settings["teams"], settings["playoff_teams"], settings["playoff_weeks"]
    need = (p - 1).bit_length() * settings.get("playoff_round_periods", 1)
    if not 1 <= slot <= teams:
        raise ValueError(f"draft slot must be 1..{teams}")
    if not 2 <= p <= teams:
        raise ValueError(f"playoff teams must be 2..{teams}")
    if need > w:
        raise ValueError(f"{p} playoff teams need {need} playoff weeks, have {w}")


def calendar(days, schedule, settings, matchups=None):
    """Matchup period per day (-1 = outside the fantasy season), regular-season period count, playoff periods.
    ESPN's matchup calendar (date, matchup, playoff) when given, else calendar weeks with the last
    playoff_weeks as playoffs (spec A5)."""
    if matchups is None:
        period = schedule.drop_duplicates("date").set_index("date").week.reindex(days).values - 1
        n = int(period.max()) + 1
        return period, n - settings["playoff_weeks"], list(range(n - settings["playoff_weeks"], n))
    m = matchups.assign(date=pd.to_datetime(matchups.date).dt.strftime("%Y-%m-%d")).drop_duplicates("date")
    ids = np.sort(m.matchup.unique())
    period = pd.Series(np.searchsorted(ids, m.matchup), index=m.date).reindex(days).fillna(-1).astype(int).values
    po = sorted(set(np.searchsorted(ids, m.matchup[m.playoff.astype(bool)]).tolist()))
    return period, len(ids) - len(po), po


def back_to_back(G, days):
    """G (teams' rows x days) -> True where the team also played the previous calendar day."""
    d = pd.to_datetime(days)
    prev = pd.Series(np.arange(len(d)), index=d).reindex(d - pd.Timedelta(days=1)).values  # NaN: no game day then
    has = ~np.isnan(prev)
    B = np.zeros_like(G)
    B[:, has] = G[:, has] & G[:, prev[has].astype(int)]
    return B


def availability(G, avail, rng, k=None, b2b=None, pen=None):
    """One contiguous injury spell per player-season; length ~ Binomial(games, 1 - a). a = avail, or with k
    (fitted in model.py, output as gp_k) a ~ Beta(m*k, (1-m)*k) around m = avail clipped to [.02, .98]: real
    availability clumps (stars miss 10 games or 60, rarely 20).
    b2b (same shape as G) + pen (per player): on those 2nd nights he also sits with probability pen. The GP model
    already counts these misses, so a is raised to keep the expected season total."""
    n = G.sum(axis=1)
    if b2b is not None:
        avail = np.minimum(avail / (1 - pen * b2b.sum(axis=1) / np.maximum(n, 1)), 1)
    if k is not None and np.isfinite(k).all():
        m = np.clip(avail, 0.02, 0.98)
        avail = rng.beta(m * k, (1 - m) * k)
    miss = rng.binomial(n, 1 - avail)
    start = (rng.random(len(G)) * (n - miss + 1)).astype(int)
    game_no = np.cumsum(G, axis=1) - 1
    plays = G & ~((game_no >= start[:, None]) & (game_no < (start + miss)[:, None]))
    return plays if b2b is None else plays & ~(b2b & (rng.random(G.shape) < pen[:, None]))


def sim_season(rosters, b, G, period, n_periods, fit, cap, rng, b2b=None):
    """Fantasy points (teams, matchup periods); daily lineup = draft.lineup on expected points of who plays.
    b2b: (players, days) 2nd-night flags; rotation veterans (MIN > 22) rest there, more with age."""
    fp, sd, avail = b.fp_pg.values, b.sd.values, b.avail.values
    k = b.gp_k.values if "gp_k" in b else None  # projections.csv from before model.py fitted it: plain binomial
    pen = np.where(b.MIN.values > 22, np.clip(0.012 * (np.nan_to_num(b.age.values, nan=24) - 24), 0, 0.14), 0)
    weekly = np.zeros((len(rosters), n_periods))
    for t, r in enumerate(rosters):
        r = np.asarray(r, int)
        plays = availability(G[r], avail[r], rng, k[r] if k is not None else None,
                             None if b2b is None else b2b[r], pen[r])
        start = lineup(np.where(plays, fp[r, None], 0.0), fit[r], cap)
        real = np.where(start, rng.normal(fp[r, None], sd[r, None], plays.shape), 0.0)
        np.add.at(weekly[t], period, real.sum(axis=0))
    return weekly


def espn_pick(b, mine, taken, settings, rng):
    """Opponent following ESPN's list (espn_rank, else adp) with small noise; unranked players by value after."""
    free = b[~b.index.isin(list(taken))]
    noisy = espn_rank(free).values * rng.lognormal(0, NOISE, len(free))
    order = free.index[np.lexsort((-free.season_fp.values, noisy))]
    return next((c for c in order[:100] if feasible(b, mine, [c], settings)), order[0])


def sim_draft(b, ev, settings, me, strategy, opp, po_mask, rng):
    rosters = [[] for _ in range(settings["teams"])]
    taken = set()
    for t in snake(settings["teams"], settings["rounds"]):
        if t == me:
            pick = recommend(b, ev, rosters[t], taken, settings, strategy,
                             po_mask if strategy == "playoffs" else None).index[0]
        elif opp == "espn":
            pick = espn_pick(b, rosters[t], taken, settings, rng)
        else:
            r = recommend(b, ev, rosters[t], taken, settings, "greedy")
            pick = r.index[np.argmax(r.score.values * rng.lognormal(0, NOISE, len(r)))]
        rosters[t].append(pick)
        taken.add(pick)
    return rosters


def trim(b, settings):
    """Speed: drop players nobody drafts, but keep injured stars (IR stash candidates) and ESPN's ranked list."""
    n = settings["teams"] * settings["rounds"] + 80
    keep = ((b.season_fp.rank(ascending=False) <= n) | (b.ir_ok & (b.fp_pg.rank(ascending=False) <= 150))
            | (espn_rank(b) <= n))
    return b[keep].reset_index(drop=True)


def prep(settings, b=None, schedule=None, matchups=None, draftable=True):
    """Board (trimmed to draftable players) + fantasy-season calendar; defaults read data/."""
    if schedule is None:
        schedule = pd.read_csv(DATA / "schedule.csv")
        if (DATA / "matchups.csv").exists():
            matchups = pd.read_csv(DATA / "matchups.csv")
    b = board(pd.read_csv(DATA / "projections.csv"), settings) if b is None else b
    b = trim(b, settings) if draftable else b
    G, days = day_matrix(b, schedule)
    B = back_to_back(G, days)
    period, n_reg, po = calendar(days, schedule, settings, matchups)
    G, B, period = G[:, period >= 0], B[:, period >= 0], period[period >= 0]
    fit, cap = slot_fit(b, settings)
    return dict(b=b, G=G, b2b=B, ev=ev_matrix(b, G), period=period, n_periods=int(period.max()) + 1, n_reg=n_reg, po=po,
                po_mask=np.isin(period, po), fit=fit, cap=cap)


def season_stats(rosters, c, settings, rng, n_seasons):
    """Per team, summed over seasons: [champion, playoffs, bottom 3, wins, regular-season points]."""
    teams, p, rp = len(rosters), settings["playoff_teams"], settings.get("playoff_round_periods", 1)
    m = np.zeros((teams, 5))
    for _ in range(n_seasons):
        weekly = sim_season(rosters, c["b"], c["G"], c["period"], c["n_periods"], c["fit"], c["cap"], rng, c["b2b"])
        rank, wins = regular_season(weekly, c["n_reg"])
        place = np.argsort(rank)
        m[playoffs(list(rank[:p]), weekly, c["po"], rp), 0] += 1
        m[:, 1] += place < p
        m[:, 2] += place >= teams - 3
        m[:, 3] += wins
        m[:, 4] += weekly[:, :c["n_reg"]].sum(axis=1)
    return m


def one_draft(job, c, settings, me, seed, n_seasons):
    """One draft + its seasons -> season_stats for every team."""
    strat, opp, d = job
    # common random numbers: draft d / its seasons use the same seeds for every strategy
    rosters = sim_draft(c["b"], c["ev"], settings, me, strat, opp, c["po_mask"], np.random.default_rng([seed, d, 0]))
    return season_stats(rosters, c, settings, np.random.default_rng([seed, d, 1]), n_seasons)


def drafts(jobs, c, settings, me, seed, n_seasons):
    # drafts are pandas-bound (~10 s each); independent + seeded per job, so results match a serial run exactly
    with ProcessPoolExecutor() as pool:
        return list(pool.map(partial(one_draft, c=c, settings=settings, me=me, seed=seed, n_seasons=n_seasons), jobs))


def run(settings, slot, n_drafts=20, n_seasons=20, seed=0, strategies=None, b=None, schedule=None, matchups=None):
    validate(settings, slot)
    jobs = [(strat, opp, d) for strat in strategies or list(STRATEGIES) for opp in OPPONENTS for d in range(n_drafts)]
    sums = [m[slot - 1, :4] for m in drafts(jobs, prep(settings, b, schedule, matchups), settings, slot - 1, seed,
                                            n_seasons)]
    rows = []
    for i in range(0, len(jobs), n_drafts):
        rates = np.array(sums[i:i + n_drafts]) / n_seasons  # per-draft rates: (n_drafts, 4)
        m = rates.mean(axis=0)
        obj = rates[:, 0] - rates[:, 2]  # per-draft (p_champion - p_bottom3)
        obj_se = obj.std(ddof=1) / np.sqrt(len(obj)) if len(obj) > 1 else 0.0
        rows.append({"strategy": jobs[i][0], "opponents": jobs[i][1], "p_champion": m[0], "p_playoffs": m[1],
                     "p_bottom3": m[2], "avg_wins": m[3], "obj_se": obj_se})
    return pd.DataFrame(rows)


def team_table(m, me):
    return pd.DataFrame({"team": np.arange(1, len(m) + 1), "us": np.arange(len(m)) == me, "avg_points": m[:, 4],
                         "avg_wins": m[:, 3], "p_playoffs": m[:, 1], "p_champion": m[:, 0]})


def report(settings, slot, n_drafts=20, n_seasons=20, seed=0, strategy="greedy", b=None, schedule=None,
           matchups=None):
    """D14: `strategy` at `slot` vs noisy-greedy opponents; per-team (= draft slot) season averages."""
    validate(settings, slot)
    jobs = [(strategy, "greedy", d) for d in range(n_drafts)]
    sums = sum(drafts(jobs, prep(settings, b, schedule, matchups), settings, slot - 1, seed, n_seasons))
    return team_table(sums / (n_drafts * n_seasons), slot - 1)


def report_from_rosters(rosters, b, settings, n_seasons=200, seed=0, schedule=None, matchups=None):
    """Season-simulate given rosters (lists of board indices of b, one per draft slot, e.g. a practice draft)."""
    c = prep(settings, b, schedule, matchups, draftable=False)
    m = season_stats(rosters, c, settings, np.random.default_rng(seed), n_seasons)
    return team_table(m / n_seasons, settings["my_slot"] - 1)


def from_draft(settings, n_seasons=200, data=DATA, b=None, schedule=None, matchups=None):
    """Season-simulate the real (practice) draft in data/draft.json (player ids in overall order, null = unlisted,
    which leaves that roster a player short). My team = the slot last_picks.json read off the page, else settings."""
    ids = json.loads((data / "draft.json").read_text())
    if b is None:
        b = board(pd.read_csv(data / "projections.csv"), settings)
    if schedule is None:
        schedule = pd.read_csv(data / "schedule.csv")
        matchups = pd.read_csv(data / "matchups.csv") if (data / "matchups.csv").exists() else None
    idx = dict(zip(b.player_id, b.index))
    rosters = [[] for _ in range(settings["teams"])]
    for t, i in zip(snake(settings["teams"], settings["rounds"]), ids):
        if i in idx:
            rosters[t].append(idx[i])
    try:
        slot = json.loads((data / "last_picks.json").read_text()).get("slot")
    except (OSError, ValueError):
        slot = None
    slot = slot or settings["my_slot"]
    df = report_from_rosters(rosters, b, {**settings, "my_slot": slot}, n_seasons, schedule=schedule, matchups=matchups)
    out = {"slot": slot, "n_seasons": n_seasons, "table": df.to_dict("records")}
    (data / "report_from_draft.json").write_text(json.dumps(out, indent=2))
    return df


def best(df):
    # ["obj"], not .obj: GroupBy.obj is the whole grouped frame
    g = df.assign(obj=df.p_champion - df.p_bottom3).groupby("strategy")
    mean_obj, se = g["obj"].mean(), g["obj_se"].mean()  # se averaged over opponent rows
    top = mean_obj.idxmax()
    if top == "greedy" or "greedy" not in mean_obj.index:
        return top
    gap = mean_obj[top] - mean_obj["greedy"]
    # only switch off greedy if the gap clearly exceeds noise (F3): otherwise it's not worth the risk
    return top if gap > 2 * np.sqrt(se[top] ** 2 + se["greedy"] ** 2) else "greedy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int)
    ap.add_argument("--drafts", type=int, default=20)
    ap.add_argument("--seasons", type=int, help="default 20 (200 with --from-draft)")
    ap.add_argument("--from-draft", action="store_true", help="season-simulate the rosters in data/draft.json")
    ap.add_argument("--report", action="store_true", help="our best strategy vs 13 near-optimal opponents")
    ap.add_argument("--settings", type=Path, default=ROOT / "settings.json")
    a = ap.parse_args()
    if not (a.slot or a.from_draft):
        ap.error("--slot is required")
    settings = json.loads(a.settings.read_text())
    if a.from_draft:
        print(from_draft(settings, a.seasons or 200).round(3).to_string(index=False))
        return
    a.seasons = a.seasons or 20
    league = {k: settings.get(k) for k in LEAGUE_KEYS}
    if a.report:
        f = DATA / f"strategy_slot{a.slot}.json"
        saved = json.loads(f.read_text()) if f.exists() else {}
        strat = saved["best"] if saved.get("league") == league else "greedy"
        df = report(settings, a.slot, a.drafts, a.seasons, strategy=strat)
        print(f"strategy: {strat}  ({a.drafts} drafts x {a.seasons} seasons)")
        print(df.round(3).to_string(index=False))
        out = {"strategy": strat, "slot": a.slot, "n_drafts": a.drafts, "n_seasons": a.seasons,
               "table": df.to_dict("records")}
        (DATA / f"report_slot{a.slot}.json").write_text(json.dumps(out, indent=2))
        return
    df = run(settings, a.slot, a.drafts, a.seasons)
    print(df.round(3).to_string(index=False))
    out = {"best": best(df), "table": df.to_dict("records"), "n_drafts": a.drafts, "n_seasons": a.seasons,
           "league": league}
    (DATA / f"strategy_slot{a.slot}.json").write_text(json.dumps(out, indent=2))
    print("best:", out["best"])


if __name__ == "__main__":
    main()
