"""Strategy simulator: simulated drafts x simulated H2H seasons.
Run: python sim.py --slot 5 [--drafts 20 --seasons 20]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from draft import STRATEGIES, board, day_matrix, ev_matrix, feasible, recommend, snake, starters

ROOT = Path(__file__).parent
DATA = ROOT / "data"
NOISE = 0.1  # ponytail: opponent pick noise (lognormal sigma); 0 = clones of the greedy recommender
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
