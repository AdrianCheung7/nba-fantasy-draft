"""In-season logic (pure, no Streamlit/network): pickups, drop candidate, matchup outlook."""
import math

import numpy as np
import pandas as pd

from draft import lineup, short_pos, slot_fit, team_value

INJURY = {"OUT": 0.0, "INJURY_RESERVE": 0.0, "DAY_TO_DAY": 0.5}  # D10: share of the matchup's expected points left


def window(days, period, today):
    """(matchup period, mask of its days from today on, mask of the season's days from today on); None once the
    season is over. Before the season it is the first matchup, whole."""
    live = (period >= 0) & (days >= today)
    if not live.any():
        return None
    p = period[live.argmax()]  # ponytail: a matchup with no game days left is skipped early; ESPN still shows it
    return p, (period == p) & (days >= today), live  # today counts whole: games already played are also in "so far"


def week_ev(b, ev, wk):
    """Expected points per player and remaining matchup day, scaled for injuries (D10)."""
    return ev[:, wk] * b.injury_status.map(INJURY).fillna(1.0).values[:, None]


def split_roster(rows, idx):
    """(all, active) board indices of a team's roster rows; active = not in an IR slot. Unknown players are skipped."""
    rows = rows[rows.player_id.isin(idx)]
    ids = rows.player_id.map(idx)
    return ids.tolist(), ids[rows.lineup_slot != "IR"].tolist()


def drop_candidate(ev, fit, cap, active):
    """D11: the active (non-IR) player whose removal costs the least lineup value, the lower scorer on ties."""
    if not active:
        return None
    a = np.array(active, int)
    rest = np.array([np.delete(a, i) for i in range(len(a))]).reshape(len(a), len(a) - 1)
    loss = (team_value(ev[a], fit[a], cap) - team_value(ev[rest], fit[rest], cap)).round(6)
    return int(a[np.lexsort((ev[a].sum(axis=1), loss))[0]])


def pickups(b, ev, G, mine, active, free, wk, ros, settings, n=15, k=50):
    """Free agents ranked by lineup points gained over the rest of this matchup (week_gain) and the rest of the
    season (ros_gain), each replacing the drop candidate. mine = my whole roster, active = without IR slots."""
    fit, cap = slot_fit(b, settings)
    evw, evr = week_ev(b, ev, wk), ev[:, ros]
    free = np.asarray(free, int)
    cand = np.unique(np.r_[free[np.argsort(-evw[free].sum(axis=1))[:k]], free[np.argsort(-evr[free].sum(axis=1))[:k]]])
    drop = drop_candidate(evr, fit, cap, active) if len(active) >= sum(settings["slots"].values()) - settings["slots"].get("IR", 0) else None
    a = np.array(active, int)
    rosters = np.array([[i for i in a if i != drop] + [c] for c in cand], int).reshape(len(cand), -1)

    def gain(e):
        return team_value(e[rosters], fit[rosters], cap) - team_value(e[a], fit[a], cap)

    out = pd.DataFrame({"name": b.name[cand].values, "pos": b.positions[cand].map(short_pos).values,
                        "games": G[cand][:, wk].sum(axis=1) * b.injury_status[cand].map(INJURY).fillna(1.0).values,
                        "week_gain": gain(evw), "ros_gain": gain(evr),
                        "drop": "" if drop is None else b.at[drop, "name"]}, index=cand)
    out["star"] = out.ros_gain > 0
    return out.sort_values(["week_gain", "ros_gain"], ascending=False).head(n)


def rest_of_matchup(b, evw, fit, cap, players):
    """(expected points, variance) of the daily best lineups over the remaining days. evw from week_ev."""
    p = np.array(players, int)
    start = lineup(evw[p], fit[p], cap)
    return (evw[p] * start).sum(), ((b.sd.values[p] ** 2)[:, None] * start).sum()


def matchup_projection(my_so_far, opp_so_far, my_rest_ev, opp_rest_ev, my_rest_var, opp_rest_var):
    """D12: projected totals and my win chance, Normal approximation of the remaining points."""
    me, opp = my_so_far + my_rest_ev, opp_so_far + opp_rest_ev
    sd = math.sqrt(my_rest_var + opp_rest_var)
    win = 0.5 * (1 + math.erf((me - opp) / (sd * math.sqrt(2)))) if sd else (me > opp) + 0.5 * (me == opp)
    return {"me": me, "opp": opp, "win": float(win)}
