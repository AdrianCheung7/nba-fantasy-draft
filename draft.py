"""Draft engine: snake order, value board, roster legality, marginal-value recommendations."""
import numpy as np
import pandas as pd

from model import points

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
    b["fp_pg"] = points(b, settings["scoring"])
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


def team_value(ev, n_start):
    """Season total of daily best-n lineups. ev: (..., players, days)."""
    # ponytail: lineup from expected values ignores position eligibility per day (spec A3)
    top = -np.sort(-ev, axis=-2)[..., :n_start, :]
    return np.maximum(top, 0).sum(axis=(-2, -1))


def matched(pos_sets, slots):
    """Max starter slots fillable by these players (bipartite matching, augmenting paths)."""
    owner = [-1] * len(slots)

    def augment(p, seen):
        for s, name in enumerate(slots):
            if name in pos_sets[p] and s not in seen:
                seen.add(s)
                if owner[s] == -1 or augment(owner[s], seen):
                    owner[s] = p
                    return True
        return False

    return sum(augment(p, set()) for p in range(len(pos_sets)))


def feasible(b, mine, cand, settings):
    """Candidates that keep my roster completable (spec A4 + starter coverage)."""
    slots = starters(settings)
    ir = settings["slots"].get("IR", 0)
    healthy_cap = sum(settings["slots"].values()) - ir
    picks_left = settings["rounds"] - len(mine) - 1
    mine_ir = int(b.ir_ok.values[mine].sum())
    out = []
    for c in cand:
        if len(mine) + 1 - min(ir, mine_ir + int(b.ir_ok.iat[c])) > healthy_cap:
            continue
        if len(slots) > picks_left and len(slots) - matched([b.pos.iat[i] for i in mine + [c]], slots) > picks_left:
            continue
        out.append(c)
    return out
