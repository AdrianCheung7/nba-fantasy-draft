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


# ---- H2H categories ----
RATIO = {"FG%": ("FGM", "FGA"), "FT%": ("FTM", "FTA"), "3P%": ("3PM", "3PA")}
STAT_ID = {0: "PTS", 1: "BLK", 2: "STL", 3: "AST", 4: "OREB", 5: "DREB", 9: "PF", 11: "TO", 19: "FG%", 20: "FT%",
           21: "3P%", 43: "TW"}  # ESPN scoring stat id -> category name


def _cols(cats):
    return sorted({c for k in cats for c in RATIO.get(k["stat"], (k["stat"],))})


def value_series(b, settings):
    """Per-game category value of every player: model.value_pg when it exists, else a plain z-score sum."""
    import model
    vp = getattr(model, "value_pg", None)
    if vp:
        return pd.Series(np.asarray(vp(b, settings), float), index=b.index)
    # ponytail: stand-in until model.value_pg lands; pool = the 200 most-played players, no iteration
    pool, tot = b.nlargest(200, "MIN"), pd.Series(0.0, index=b.index)
    for k in settings["categories"]:
        st = k["stat"]
        if st in RATIO:
            m, a = RATIO[st]
            f = lambda d: d[m] - pool[m].sum() / pool[a].sum() * d[a]
        else:
            f = lambda d: d[st] if st in d else d["MIN"] * 0
        z = (f(b) - f(pool).mean()) / (f(pool).std() or 1.0)
        tot += -z if k.get("reverse") else z
    return tot


def cat_lines(b, evw, fit, cap, players, scale, cats):
    """{column: (mean, variance)} of the roster's totals over the remaining days (starters of the daily best lineup).
    evw from week_ev (must be > 0 for players who play); scale = availability x injury share per player.
    Counting variance: per-game sd = cv x per-game mean; TW is a coin flip per start."""
    p = np.array(players, int)
    start = lineup(evw[p], fit[p], cap)
    w, n = (start * scale[p][:, None]).sum(axis=1), start.sum(axis=1)
    out = {}
    for c in _cols(cats):
        x = b[c].values[p] if c in b else np.zeros(len(p))
        out[c] = (float(w @ x), float(w @ (x * (1 - x)).clip(0)) if c == "TW" else float(n @ (b.cv.values[p] * x) ** 2))
    return out


def cat_matchup(my_lines, opp_lines, so_far_me, so_far_opp, cats):
    """(DataFrame cat, me, opp, p_win; expected categories I win). Lines from cat_lines, so-far = raw column totals
    (missing = 0). Normal approximation; a ratio is made / attempted with binomial variance on the attempts left."""
    def total(lines, so, st):
        if st not in RATIO:
            return so.get(st, 0) + lines.get(st, (0, 0))[0], lines.get(st, (0, 0))[1]
        m, a = RATIO[st]
        made, att = so.get(m, 0) + lines.get(m, (0, 0))[0], so.get(a, 0) + lines.get(a, (0, 0))[0]
        r = made / att if att else 0.0
        return r, r * (1 - r) * lines.get(a, (0, 0))[0] / att ** 2 if att else 0.0

    rows = []
    for k in cats:
        (x, vx), (y, vy) = total(my_lines, so_far_me, k["stat"]), total(opp_lines, so_far_opp, k["stat"])
        d, sd = (y - x if k.get("reverse") else x - y), math.sqrt(vx + vy)
        rows.append((k["stat"], x, y, 0.5 * (1 + math.erf(d / (sd * math.sqrt(2)))) if sd else (d > 0) + 0.5 * (d == 0)))
    df = pd.DataFrame(rows, columns=["cat", "me", "opp", "p_win"])
    return df, float(df.p_win.sum())


def cat_pickups(b, val, G, mine, active, free, wk, ros, settings, opp_lines, so_me, so_opp, n=15, k=50):
    """Category-mode pickups: free agents ranked by the gain in expected categories won this matchup (cat_gain) and
    then by rest-of-season value (ros_gain, from the per-game value Series `val`), each replacing the drop candidate."""
    cats = settings["categories"]
    fit, cap = slot_fit(b, settings)
    avail, inj = b.avail.values, b.injury_status.map(INJURY).fillna(1.0).values
    v = np.asarray(val, float)
    ev_u = (v * avail)[:, None] * G
    ev_s = ((v - v.min() + 1) * avail)[:, None] * G  # shifted so every player on a game day can start; ranking only
    evw, evr_s, evr_u = week_ev(b, ev_s, wk), ev_s[:, ros], ev_u[:, ros]
    free = np.asarray(free, int)
    cand = np.unique(np.r_[free[np.argsort(-(ev_u[:, wk].sum(axis=1) * inj)[free])[:k]],
                           free[np.argsort(-evr_u[free].sum(axis=1))[:k]]])
    drop = drop_candidate(evr_s, fit, cap, active) if len(active) >= sum(settings["slots"].values()) - settings["slots"].get("IR", 0) else None

    def won(players):
        return cat_matchup(cat_lines(b, evw, fit, cap, players, avail * inj, cats), opp_lines, so_me, so_opp, cats)[1]

    def ros_val(players):
        p = np.array(players, int)
        return (evr_u[p] * lineup(evr_s[p], fit[p], cap)).sum()

    base_won, base_ros = won(active), ros_val(active)
    rosters = [[i for i in active if i != drop] + [int(c)] for c in cand]
    out = pd.DataFrame({"name": b.name[cand].values, "pos": b.positions[cand].map(short_pos).values,
                        "games": G[cand][:, wk].sum(axis=1) * inj[cand],
                        "cat_gain": [won(r) - base_won for r in rosters],
                        "ros_gain": [ros_val(r) - base_ros for r in rosters],
                        "drop": "" if drop is None else b.at[drop, "name"]}, index=cand)
    out["star"] = out.ros_gain > 0
    return out.sort_values(["cat_gain", "ros_gain"], ascending=False).head(n)


def cat_scores(js, matchup, me):
    """Per-category scores so far in ESPN matchup `matchup`: {"me": {cat: score}, "opp": {...}, "opp_id"}; None if I
    have no game. The dicts are empty when the scoreboard has no cumulativeScore.scoreByStat (pre-season)."""
    for g in js["schedule"]:
        sides = [s for s in (g.get("home"), g.get("away")) if s]
        if g["matchupPeriodId"] == matchup and any(s["teamId"] == me for s in sides) and len(sides) == 2:
            mine, opp = sorted(sides, key=lambda s: s["teamId"] != me)
            by = lambda s: {STAT_ID[int(i)]: v["score"] for i, v in s.get("cumulativeScore", {}).get("scoreByStat", {}).items()
                            if int(i) in STAT_ID and v.get("score") is not None}
            return {"me": by(mine), "opp": by(opp), "opp_id": opp["teamId"]}


def so_far_from_scores(scores, lines, cats, frac):
    """Raw column totals for cat_matchup from ESPN category scores. ESPN gives a % not made/attempted, so the attempts
    so far are guessed as frac (days played / days left) of the attempts still projected in `lines`."""
    out = {}
    for k in cats:
        st, v = k["stat"], scores.get(k["stat"])
        if v is None:
            continue
        if st in RATIO:
            m, a = RATIO[st]
            out[a] = lines.get(a, (0, 0))[0] * frac
            out[m] = (v / 100 if v > 1 else v) * out[a]
        else:
            out[st] = v
    return out


def adds_left(js, me, matchup, limit):
    """Adds left in ESPN matchup `matchup` (team.transactionCounter.matchupAcquisitionTotals); None if I'm unknown."""
    t = next((t for t in js["teams"] if t["id"] == me), None)
    return None if t is None else int(limit - t.get("transactionCounter", {}).get("matchupAcquisitionTotals", {}).get(str(matchup), 0))
