"""Draft engine: snake order, value board, roster legality, marginal-value recommendations."""
import numpy as np
import pandas as pd

import model
from model import points, value_pg

IR_OK = {"OUT", "INJURY_RESERVE"}
NON_STARTING = {"BE", "IR"}


def snake(teams, rounds):
    return [i if r % 2 == 0 else teams - 1 - i for r in range(rounds) for i in range(teams)]


def picks_before_next(order, overall, team):
    nxt = [j for j in range(overall + 1, len(order)) if order[j] == team]
    return nxt[0] - overall - 1 if nxt else None


def starters(settings):
    return [name for name, n in settings["slots"].items() if name not in NON_STARTING for _ in range(n)]


def board(proj, settings):
    b = proj.reset_index(drop=True).copy()
    b["fp_pg"] = value_pg(b, settings)
    b["avail"] = (b.gp / 82).clip(0, 1)
    b["sd"] = b.cv * b.fp_pg.abs()
    b["season_fp"] = b.fp_pg * b.gp
    b["pos"] = b.positions.str.split("|").apply(set)
    b["ir_ok"] = b.injury_status.isin(IR_OK)
    return b


def day_matrix(b, schedule):
    days = np.sort(schedule.date.unique())
    col = {d: i for i, d in enumerate(days)}
    team_days = np.zeros((max(int(schedule.team_id.max()), int(b.team_id.max())) + 1, len(days)), bool)
    team_days[schedule.team_id.values, schedule.date.map(col).values] = True
    return team_days[b.team_id.values], days  # team 0 (free agent) never plays


def ev_matrix(b, G):
    return (b.fp_pg * b.avail).values[:, None] * G


def slot_fit(b, settings):
    """The starter slots as Hall's condition: a set of players can fill them iff, for every set Q of slot names,
    the players eligible only for slots in Q number <= the slots in Q. fit[p, q] = player p counts toward Q q."""
    slots = starters(settings)
    names = sorted(set(slots))
    bits = np.array([sum(1 << i for i, n in enumerate(names) if n in pos) for pos in b.pos], dtype=np.int64)
    # ponytail: 2^(slot names) conditions; fine for the handful of starter slot names leagues use
    qs = np.arange(1 << len(names))
    fit = (bits[:, None] & ~qs) == 0
    cap = np.array([sum(slots.count(n) for i, n in enumerate(names) if q >> i & 1) for q in qs])
    keep = np.flatnonzero(fit.any(axis=0))  # a Q nobody fits inside can't bind
    keep = keep[np.argsort(cap[keep], kind="stable")]
    _, first = np.unique(fit[:, keep], axis=1, return_index=True)  # same players: keep the tightest cap
    keep = keep[first]
    return fit[:, keep].astype(np.int8), cap[keep]


def lineup(ev, fit, cap):
    """Daily starters (bool, shaped like ev: (..., players, days)): each day add players best-first while the
    set still fits the starter slots. A player's value doesn't depend on his slot, so the sets that fit form a
    transversal matroid and this greedy is optimal. fit: (..., players, conditions) from slot_fit."""
    order = np.argsort(-ev, axis=-2)
    plays = np.take_along_axis(ev, order, axis=-2) > 0
    start = np.zeros(ev.shape, bool)
    used = np.zeros(ev.shape[:-2] + (ev.shape[-1], len(cap)), np.int16)
    for k in range(ev.shape[-2]):
        i = order[..., k:k + 1, :]
        f = np.take_along_axis(fit, np.swapaxes(i, -1, -2), axis=-2)  # (..., days, conditions)
        ok = plays[..., k, :] & (used + f <= cap).all(axis=-1)
        used += f * ok[..., None]
        np.put_along_axis(start, i, ok[..., None, :], axis=-2)
    return start


def team_value(ev, fit, cap):
    """Season total of daily best legal lineups. ev: (..., players, days)."""
    return (ev * lineup(ev, fit, cap)).sum(axis=(-2, -1))


def match(pos_sets, slots):
    """Player filling each starter slot (-1 = open); bipartite matching by augmenting paths."""
    owner = [-1] * len(slots)

    def augment(p, seen):
        for s, name in enumerate(slots):
            if name in pos_sets[p] and s not in seen:
                seen.add(s)
                if owner[s] == -1 or augment(owner[s], seen):
                    owner[s] = p
                    return True
        return False

    for p in range(len(pos_sets)):
        augment(p, set())
    return owner


def matched(pos_sets, slots):
    """Max starter slots fillable by these players."""
    return sum(o >= 0 for o in match(pos_sets, slots))


def feasible(b, mine, cand, settings):
    """Candidates that keep my roster completable (spec A4 + starter coverage)."""
    slots = starters(settings)
    ir = settings["slots"].get("IR", 0)
    healthy_cap = sum(settings["slots"].values()) - ir
    picks_left = settings["rounds"] - len(mine) - 1
    ir_ok, pos, pats = b.ir_ok.values, b.pos.values, b.positions.values
    mine_ir = int(ir_ok[mine].sum())
    covers = {}  # starter coverage depends only on the candidate's positions
    out = []
    for c in cand:
        if len(mine) + 1 - min(ir, mine_ir + int(ir_ok[c])) > healthy_cap:
            continue
        if len(slots) > picks_left:
            if pats[c] not in covers:
                covers[pats[c]] = len(slots) - matched([pos[i] for i in mine] + [pos[c]], slots) <= picks_left
            if not covers[pats[c]]:
                continue
        out.append(c)
    return out


TOP_K = 60  # ponytail: only the best 60 legal players are scored; ample for 8-20 teams

# ponytail: coefficients are knobs; sim.py measures which strategy actually wins
STRATEGIES = {
    "greedy": lambda c, ctx: c.marginal,
    "durable": lambda c, ctx: c.marginal * np.sqrt(c.avail),
    "low_variance": lambda c, ctx: c.marginal - 2 * c.sd * np.sqrt(c.gp),
    "playoffs": lambda c, ctx: c.marginal + 2 * c.marginal_po,
    "ir_early": lambda c, ctx: c.marginal * np.where(c.ir_ok & ctx["ir_open"], 1.3, 1.0),
    # category leagues only: recommend() values players without these categories ("FT" = FT%), then greedy
    **{f"punt_{p}": lambda c, ctx: c.marginal for p in ("TO_PF", "FT", "3P", "BLK", "AST")},
}


def offered(settings):
    """Strategy names for this league: the punts only in category leagues."""
    return [n for n in STRATEGIES if not n.startswith("punt_") or settings.get("format") == "categories"]


def recommend(b, ev, mine, taken, settings, strategy="greedy", po_mask=None, k=TOP_K):
    mine = list(mine)
    fit, cap = slot_fit(b, settings)
    free = b[~b.index.isin(list(taken) + mine)].sort_values("season_fp", ascending=False).index
    cand = feasible(b, mine, list(free), settings)[:k]  # best k legal players
    fallback = not cand  # M7: nothing legal (e.g. no IR-eligible left): best available, flagged for the UI
    cand = cand or list(free[:k])
    out = b.loc[cand].copy()
    if strategy.startswith("punt_") and settings.get("format") == "categories":
        drop = strategy.split("_")[1:]
        s = {**settings, "categories": [x for x in settings["categories"] if x["stat"].rstrip("%") not in drop]}
        # ev = value * avail on game days; ponytail: a player valued exactly 0 loses his game days here
        ev = (np.asarray(model.value_pg(b, s), float) * b.avail.values)[:, None] * (ev != 0)
    rosters = [mine + [c] for c in cand]
    out["marginal"] = team_value(ev[rosters], fit[rosters], cap) - team_value(ev[mine], fit[mine], cap)
    if po_mask is None:
        out["marginal_po"] = 0.0
    else:
        evp = ev[:, po_mask]
        out["marginal_po"] = team_value(evp[rosters], fit[rosters], cap) - team_value(evp[mine], fit[mine], cap)
    ctx = {"ir_open": settings["slots"].get("IR", 0) > int(b.ir_ok.values[mine].sum())}
    out["score"] = STRATEGIES[strategy](out, ctx)
    out = out.sort_values("score", ascending=False)
    out.attrs["fallback"] = fallback
    return out


ADP_UNRANKED = 139  # ESPN's default adp (~140) for players nobody drafts
RANK_UNRANKED = 999  # fetch.py's espn_rank for players ESPN doesn't rank


def espn_rank(b):
    """Position on ESPN's draft list (espn_rank, else adp); unranked = inf."""
    if "espn_rank" in b:
        return b.espn_rank.where(b.espn_rank < RANK_UNRANKED, np.inf)
    return b.adp.where(b.adp < ADP_UNRANKED, np.inf)


# D2 opponent model: each opponent takes the lowest noisy blend of ESPN list position and our value position.
# Calibrated 2026-09-28: all 14 teams draft with the model (12 rounds, 150 sims); mean pick vs ESPN ADP over the
# 109 players with ADP < 120. MAE in picks: weight 1.0/noise 0.4 -> 5.1, 0.95/0.4 -> 5.9, 0.9/0.35 -> 7.8,
# 0.7/0.45 -> 14.5, 0.5/0.45 -> 20.0 (espn_rank alone: 8.3). MISFIT 1.0-3.0 moved MAE < 0.01: eligibility is
# broad, so it rarely binds. Re-run if ADP drifts from ESPN's list during draft season.
# Re-checked 2026-09-30 for 16 teams x 11 rounds, STANDARD rank + category-league ADP (60 sims): 1.0/0.4 -> 5.4,
# 0.95/0.4 -> 5.6, 0.9/0.4 -> 7.4, 0.8/0.45 -> 10.0 (value was points; rank dominates). Constants unchanged.
RANK_WEIGHT = 0.95  # ADP says ESPN's list (autopick + queues) dominates; keep a little value for sharper owners
PICK_NOISE = 0.4  # lognormal sigma on the blend: humans reach/slide ~40% of a player's list position
MISFIT = 2.0  # blend multiplier for a player the team couldn't start in an open starter slot (needs-based drafting)


def survive(b, cand, taken, upcoming, rosters, settings, rng, sims=200):
    """P(candidate still on the board at my next pick). upcoming: teams picking before it, in order (None = I have
    no next pick); rosters: team -> its board indices."""
    if upcoming is None:
        return pd.Series(0.0, index=cand)
    free = b[~b.index.isin(list(taken))]
    rank_pos = np.lexsort((-free.season_fp.values, espn_rank(free).values)).argsort() + 1
    key = RANK_WEIGHT * rank_pos + (1 - RANK_WEIGHT) * free.season_fp.rank(ascending=False, method="first").values
    top = np.argsort(key)[:len(cand) + 3 * len(upcoming) + 10]
    pool, key, pats = free.index[top], key[top], free.positions.values[top]
    slots = starters(settings)
    cache = {}

    def fits(roster):  # sorted positions strings -> which pool players would fill an open starter slot
        if roster not in cache:
            sets = [set(p.split("|")) for p in roster]
            base = matched(sets, slots)
            ok = {p: base == len(slots) or matched(sets + [set(p.split("|"))], slots) > base for p in set(pats)}
            cache[roster] = np.array([ok[p] for p in pats])
        return cache[roster]

    base = {t: b.positions.values[rosters.get(t, [])].tolist() for t in upcoming}
    noisy = key * rng.lognormal(0, PICK_NOISE, (sims, len(pool)))
    gone = np.zeros(noisy.shape, bool)
    for s in range(sims):
        got = {t: [] for t in upcoming}
        for t in upcoming:
            score = noisy[s] * np.where(fits(tuple(sorted(base[t] + got[t]))), 1.0, MISFIT)
            j = np.where(gone[s], np.inf, score).argmin()
            gone[s, j] = True
            got[t].append(pats[j])
    return pd.Series(1 - gone.mean(axis=0), index=pool).reindex(cand).fillna(1.0)


def short_pos(p):
    return ", ".join(x for x in p.split("|") if x not in ("BE", "IR", "UTIL"))


UNLISTED = "(player not listed)"
TOP_COLS = ["player_id", "name", "pos", "fp_pg", "games", "value", "p_back", "status"]
STATUS_TAG = {"OUT": "OUT", "DAY_TO_DAY": "DTD", "INJURY_RESERVE": "IR", "SUSPENSION": "SUSP"}


def advise(b, ev, picks, settings, strategy="greedy", po_mask=None):
    """Single entry point for the draft UIs. picks: [{"overall": 1-based pick, "player": board index or None
    (unlisted)}]. Teams are 0-based snake indices (my_slot - 1 = me). my_next (1-based) is the pick p_back
    refers to: my pick after this one when I'm on the clock. Plain types plus one DataFrame (index = board
    index), so a server can JSON it."""
    order = snake(settings["teams"], settings["rounds"])
    me = settings["my_slot"] - 1
    picks = [p for p in picks if p["overall"] <= len(order)]  # e.g. rounds lowered mid-draft
    rosters, u = {}, 0
    for p in picks:
        t = order[p["overall"] - 1]
        if p["player"] is not None:
            rosters.setdefault(t, []).append(p["player"])
        u += p["player"] is None and t == me  # my unlisted picks still use a round and a bench spot
    mine = rosters.get(me, [])
    taken = {p["player"] for p in picks if p["player"] is not None}
    roster = [{"player_id": int(b.at[i, "player_id"]), "name": b.at[i, "name"], "pos": short_pos(b.at[i, "positions"]),
               "fp_pg": float(b.at[i, "fp_pg"]), "games": float(b.at[i, "gp"])} for i in mine]
    roster += [{**dict.fromkeys(["player_id", "pos", "fp_pg", "games"]), "name": UNLISTED} for _ in range(u)]
    slots = starters(settings)
    needs = [s for s, o in zip(slots, match([b.pos.iat[i] for i in mine], slots)) if o < 0]
    n = max((p["overall"] for p in picks), default=0)  # 0-based index of the pick on the clock
    if n >= len(order):
        top = pd.DataFrame(columns=TOP_COLS)
        top.attrs["fallback"] = False
        return {"on_clock": None, "my_next": None, "top": top, "roster": roster, "needs": needs}
    start = n + (order[n] == me)
    nxt = next((j for j in range(start, len(order)) if order[j] == me), None)
    rs = settings if not u else {**settings, "rounds": settings["rounds"] - u,
                                 "slots": {**settings["slots"], "BE": max(0, settings["slots"].get("BE", 0) - u)}}
    rec = recommend(b, ev, mine, taken, rs, strategy, po_mask).head(10)
    p_back = survive(b, list(rec.index), taken, None if nxt is None else order[start:nxt], rosters, settings,
                     np.random.default_rng(0))
    top = pd.DataFrame({"player_id": rec.player_id, "name": rec["name"], "pos": rec.positions.map(short_pos),
                        "fp_pg": rec.fp_pg, "games": rec.gp, "value": rec.marginal, "p_back": p_back.values,
                        "status": rec.injury_status.map(STATUS_TAG).fillna("")})
    top.attrs["fallback"] = bool(rec.attrs["fallback"])
    return {"on_clock": order[n], "my_next": None if nxt is None else nxt + 1, "top": top, "roster": roster,
            "needs": needs}
