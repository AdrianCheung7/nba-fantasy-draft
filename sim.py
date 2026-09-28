"""Strategy simulator: simulated drafts x simulated H2H seasons.
Run: python sim.py --slot 5 [--drafts 20 --seasons 20]"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from draft import STRATEGIES, board, day_matrix, ev_matrix, feasible, recommend, snake, starters

ROOT = Path(__file__).parent
DATA = ROOT / "data"
# ponytail: opponent pick noise (lognormal sigma); 0 = clones of the greedy recommender. Calibrated so
# greedy-vs-greedy slot 7 (14 teams, 20x20) p_playoffs lands in 6/14 +/- 0.10: 0.1 -> 0.755, 0.03 -> 0.342,
# 0.02 -> 0.285, 0.01 -> 0.188. Re-calibrate if league size or projections change a lot.
NOISE = 0.03
OPPONENTS = ("greedy", "adp")
ADP_UNRANKED = 139  # ESPN's default adp (~140) for players nobody drafts


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


def playoffs(seeds, weekly, weeks):
    """Seeded single elimination; top seeds get byes; rounds use the last weeks of `weeks`."""
    size = 1 << (len(seeds) - 1).bit_length()
    rounds = size.bit_length() - 1
    if rounds > len(weeks):
        raise ValueError(f"{len(seeds)} playoff teams need {rounds} playoff weeks, have {len(weeks)}")
    alive = [seeds[s - 1] if s <= len(seeds) else -1 for s in bracket_order(size)]
    for w in weeks[len(weeks) - rounds:]:
        alive = [c if a < 0 else a if c < 0 else (a if weekly[a, w] >= weekly[c, w] else c)
                 for a, c in zip(alive[::2], alive[1::2])]
    return alive[0]


def validate(settings, slot):
    teams, p, w = settings["teams"], settings["playoff_teams"], settings["playoff_weeks"]
    if not 1 <= slot <= teams:
        raise ValueError(f"draft slot must be 1..{teams}")
    if not 2 <= p <= teams:
        raise ValueError(f"playoff teams must be 2..{teams}")
    if (p - 1).bit_length() > w:
        raise ValueError(f"{p} playoff teams need {(p - 1).bit_length()} playoff weeks, have {w}")


def availability(G, avail, rng):
    """One contiguous injury spell per player-season; length ~ Binomial(games, 1 - avail)."""
    plays = G.copy()
    for i in range(len(G)):
        game_days = np.flatnonzero(G[i])
        miss = rng.binomial(len(game_days), 1 - avail[i])
        if miss:
            start = rng.integers(0, len(game_days) - miss + 1)
            plays[i, game_days[start:start + miss]] = False
    return plays


def sim_season(rosters, b, G, week_idx, n_weeks, n_start, rng):
    """Weekly fantasy points (teams, weeks); daily lineup = top-n expected scorers who play."""
    fp, sd, avail = b.fp_pg.values, b.sd.values, b.avail.values
    weekly = np.zeros((len(rosters), n_weeks))
    for t, r in enumerate(rosters):
        r = np.asarray(r)
        plays = availability(G[r], avail[r], rng)
        lineup = np.argsort(-np.where(plays, fp[r, None], -np.inf), axis=0)[:n_start]
        real = np.where(plays, rng.normal(fp[r, None], sd[r, None], plays.shape), 0.0)
        np.add.at(weekly[t], week_idx, np.take_along_axis(real, lineup, axis=0).sum(axis=0))
    return weekly


def adp_pick(b, mine, taken, settings, rng):
    free = b[~b.index.isin(list(taken))]
    # ESPN gives undrafted players adp ~140: they go after every ranked player, best season_fp first
    noisy = np.where(free.adp < ADP_UNRANKED, free.adp * rng.lognormal(0, NOISE, len(free)), np.inf)
    order = free.index[np.lexsort((-free.season_fp.values, noisy))]
    return next((c for c in order[:100] if feasible(b, mine, [c], settings)), order[0])


def sim_draft(b, ev, settings, me, strategy, opp, po_mask, rng):
    rosters = [[] for _ in range(settings["teams"])]
    taken = set()
    for t in snake(settings["teams"], settings["rounds"]):
        if t == me:
            pick = recommend(b, ev, rosters[t], taken, settings, strategy,
                             po_mask if strategy == "playoffs" else None).index[0]
        elif opp == "adp":
            pick = adp_pick(b, rosters[t], taken, settings, rng)
        else:
            r = recommend(b, ev, rosters[t], taken, settings, "greedy")
            pick = r.index[np.argmax(r.score.values * rng.lognormal(0, NOISE, len(r)))]
        rosters[t].append(pick)
        taken.add(pick)
    return rosters


def trim(b, settings):
    """Speed: drop players nobody drafts, but keep injured stars (IR stash candidates)."""
    n = settings["teams"] * settings["rounds"] + 80
    keep = (b.season_fp.rank(ascending=False) <= n) | (b.ir_ok & (b.fp_pg.rank(ascending=False) <= 150))
    return b[keep].reset_index(drop=True)


def one_draft(job, b, ev, G, week_idx, n_weeks, n_reg, po_mask, po_weeks, settings, me, seed, n_seasons):
    """One draft + its seasons -> summed [champion, playoffs, bottom3, wins] for my team."""
    strat, opp, d = job
    p, teams, n_start = settings["playoff_teams"], settings["teams"], len(starters(settings))
    # common random numbers: draft d / its seasons use the same seeds for every strategy
    rosters = sim_draft(b, ev, settings, me, strat, opp, po_mask, np.random.default_rng([seed, d, 0]))
    rng = np.random.default_rng([seed, d, 1])
    m = np.zeros(4)
    for _ in range(n_seasons):
        weekly = sim_season(rosters, b, G, week_idx, n_weeks, n_start, rng)
        rank, wins = regular_season(weekly, n_reg)
        place = int(np.flatnonzero(rank == me)[0])
        m += [playoffs(list(rank[:p]), weekly, po_weeks) == me, place < p, place >= teams - 3, wins[me]]
    return m


def run(settings, slot, n_drafts=20, n_seasons=20, seed=0, strategies=None, b=None, schedule=None):
    validate(settings, slot)
    b = board(pd.read_csv(DATA / "projections.csv"), settings) if b is None else b
    schedule = pd.read_csv(DATA / "schedule.csv") if schedule is None else schedule
    b = trim(b, settings)
    G, days = day_matrix(b, schedule)
    ev = ev_matrix(b, G)
    week = schedule.drop_duplicates("date").set_index("date").week.reindex(days).values
    n_weeks = int(week.max())
    n_reg = n_weeks - settings["playoff_weeks"]
    po_mask, po_weeks = week > n_reg, list(range(n_reg, n_weeks))
    ctx = dict(b=b, ev=ev, G=G, week_idx=week - 1, n_weeks=n_weeks, n_reg=n_reg, po_mask=po_mask,
               po_weeks=po_weeks, settings=settings, me=slot - 1, seed=seed, n_seasons=n_seasons)
    jobs = [(strat, opp, d) for strat in strategies or list(STRATEGIES) for opp in OPPONENTS for d in range(n_drafts)]
    # drafts are pandas-bound (~10 s each); independent + seeded per job, so results match a serial run exactly
    with ProcessPoolExecutor() as pool:
        sums = list(pool.map(partial(one_draft, **ctx), jobs))
    rows = []
    for i in range(0, len(jobs), n_drafts):
        m = np.sum(sums[i:i + n_drafts], axis=0) / (n_drafts * n_seasons)
        rows.append({"strategy": jobs[i][0], "opponents": jobs[i][1], "p_champion": m[0], "p_playoffs": m[1],
                     "p_bottom3": m[2], "avg_wins": m[3]})
    return pd.DataFrame(rows)


def best(df):
    # ["obj"], not .obj: GroupBy.obj is the whole grouped frame
    return df.assign(obj=df.p_champion - df.p_bottom3).groupby("strategy")["obj"].mean().idxmax()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--drafts", type=int, default=20)
    ap.add_argument("--seasons", type=int, default=20)
    a = ap.parse_args()
    settings = json.loads((ROOT / "settings.json").read_text())
    df = run(settings, a.slot, a.drafts, a.seasons)
    print(df.round(3).to_string(index=False))
    out = {"best": best(df), "table": df.to_dict("records")}
    (DATA / f"strategy_slot{a.slot}.json").write_text(json.dumps(out, indent=2))
    print("best:", out["best"])


if __name__ == "__main__":
    main()
