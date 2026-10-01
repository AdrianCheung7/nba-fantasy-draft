"""Draft-day assistant.  Run: streamlit run app.py"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import espn
from draft import advise, board, day_matrix, ev_matrix, offered, slot_fit, snake, starters
from fetch import SLOT_NAMES
from season import matchup_projection, pickups, rest_of_matchup, split_roster, week_ev, window
from season import INJURY, adds_left, cat_lines, cat_matchup, cat_pickups, cat_scores, so_far_from_scores, value_series
from sim import LEAGUE_KEYS, calendar, validate

ROOT = Path(__file__).parent
# DRAFT_HOME lets tests point settings/picks/data at a tmp dir instead of the real repo (M8); sim.py's
# heavy simulation always runs against the real repo regardless (see the "simulate" button below).
HOME = Path(os.environ.get("DRAFT_HOME", ROOT))
DATA = HOME / "data"
SETTINGS = HOME / "settings.json"
DRAFT = DATA / "draft.json"
UNLISTED = "(player not listed)"

st.set_page_config(page_title="Fantasy Draft Assistant", layout="wide")


@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA / name)


def load_picks():
    try:
        return json.loads(DRAFT.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_picks(picks):
    tmp = DRAFT.with_suffix(".tmp")  # write-then-rename: a crash never corrupts the pick log
    tmp.write_text(json.dumps(picks))
    tmp.replace(DRAFT)


missing = [f for f in ("projections.csv", "schedule.csv") if not (DATA / f).exists()]
if missing:
    st.error(f"Missing data files: {missing}. Run `python fetch.py` then `python model.py`, then reload.")
    st.stop()

s = json.loads(SETTINGS.read_text())
CATS = s.get("format") == "categories"
PER_GAME = "Value/g" if CATS else "Pts/game"  # category value: summed z-scores per game

with st.sidebar:
    s["teams"] = st.selectbox("Teams", list(range(8, 21)), index=min(max(s["teams"], 8), 20) - 8, key="teams")
    # F1: clamp to the session's current my_slot (if the widget already has one) so reducing "Teams"
    # can't leave a stale, out-of-range value in session_state for the selectbox below.
    cur_slot = st.session_state.get("my_slot", s["my_slot"])
    if cur_slot > s["teams"]:
        st.session_state.my_slot = s["teams"]
        cur_slot = s["teams"]
    s["my_slot"] = st.selectbox("My draft slot", list(range(1, s["teams"] + 1)),
                                index=min(cur_slot, s["teams"]) - 1, key="my_slot")
    with st.expander("League settings"):
        s["rounds"] = st.number_input("Rounds", 1, 30, s["rounds"], key="rounds")
        st.caption("Roster slots")
        slots = {n: st.number_input(n, 0, 15, s["slots"].get(n, 0), key=f"slot_{n}") for n in SLOT_NAMES.values()}
        s["slots"] = {n: v for n, v in slots.items() if v}
        if "scoring" in s:  # category leagues have no point weights
            st.caption("Scoring")
            s["scoring"] = {k: st.number_input(k, value=float(v), step=0.5, key=f"sc_{k}")
                            for k, v in s["scoring"].items()}
        st.caption("Playoffs")
        s["playoff_teams"] = st.number_input("Playoff teams", 2, s["teams"], min(s["playoff_teams"], s["teams"]),
                                             key="po_teams")
        s["playoff_weeks"] = st.number_input("Playoff weeks", 1, 6, s["playoff_weeks"], key="po_weeks")
    if s["rounds"] > sum(s["slots"].values()):
        st.warning("More rounds than roster spots: extra picks can't be rostered.")

    strat_file = DATA / f"strategy_slot{s['my_slot']}.json"
    sim = json.loads(strat_file.read_text()) if strat_file.exists() else None
    default = s.get("strategy", "greedy")
    if sim is not None:
        if sim.get("league") == {k: s.get(k) for k in LEAGUE_KEYS}:
            default = sim["best"]
        else:  # M10: league settings changed since this simulation ran; don't trust it
            st.caption("Strategy results are for different league settings - rerun.")
            sim = None
            default = "greedy"
    strat_names = offered(s)
    if st.button("Sync league from ESPN", key="sync"):  # rewrites settings.json + data/matchups.csv (needs espn_auth.json)
        r = subprocess.run([sys.executable, str(ROOT / "espn.py")], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            st.error((r.stderr.strip().splitlines() or ["sync failed"])[-1])
        else:
            load_csv.clear()  # matchups.csv may have changed
            for k in [k for k in st.session_state if k in ("teams", "my_slot", "rounds", "po_teams", "po_weeks")
                      or k.startswith(("slot_", "sc_"))]:
                del st.session_state[k]  # widgets would otherwise keep the old values and autosave them back
            st.rerun()
    with st.expander("Strategy"):
        s["strategy"] = st.selectbox("Strategy", strat_names,
                                     index=strat_names.index(default) if default in strat_names else 0, key="strategy")
        if CATS:
            st.caption("Simulated odds are approximate in category leagues until the category H2H sim lands.")
        if st.button("Save settings + run strategy simulation (~4 min)", key="simulate"):
            SETTINGS.write_text(json.dumps(s, indent=2))  # sim.py reads settings.json
            with st.spinner("Simulating drafts and seasons..."):
                r = subprocess.run([sys.executable, str(ROOT / "sim.py"), "--slot", str(s["my_slot"])],
                                   cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                st.error((r.stderr.strip().splitlines() or ["simulation failed"])[-1])
            else:
                st.rerun()
        if sim:
            st.caption(f"Simulated best strategy for slot {s['my_slot']}: **{sim['best']}**")
            st.dataframe(pd.DataFrame(sim["table"]).round(3), hide_index=True)
    with st.expander("Reset draft"):
        if st.checkbox("Yes, clear every pick", key="confirm_reset") and st.button("Clear picks", key="reset"):
            save_picks([])
            (DATA / "last_picks.json").unlink(missing_ok=True)  # else the pick server replays the old draft
            st.rerun()

# F1: autosave whatever the sidebar landed on, so a refresh/restart resumes from it instead of resetting
# to the settings.json snapshot that existed before this session started editing.
if s != json.loads(SETTINGS.read_text()):
    SETTINGS.write_text(json.dumps(s, indent=2))

# F5: reject settings the engine can't handle before computing any recommendations.
try:
    validate(s, s["my_slot"])
    if len(starters(s)) > s["rounds"]:
        raise ValueError(f"roster needs {len(starters(s))} starter slots but only {s['rounds']} rounds are set")
except ValueError as e:
    st.error(str(e))
    st.stop()

b = board(load_csv("projections.csv"), s)
sched = load_csv("schedule.csv")
G, days = day_matrix(b, sched)
ev = ev_matrix(b, G)
period, _, po = calendar(days, sched, s, load_csv("matchups.csv") if (DATA / "matchups.csv").exists() else None)
po_mask = np.isin(period, po)

@st.cache_data(ttl=600)
def espn_league(home, league_id, season):
    return espn.league(espn.VIEWS, league_id, season, espn.auth(home))


def render_season():
    if st.button("Refresh data (stats + projections, a few minutes)", key="refresh"):
        with st.spinner("Fetching stats and rebuilding projections..."):
            for script in ("fetch.py", "model.py"):
                r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, capture_output=True, text=True)
                if r.returncode:
                    return st.error(f"{script} failed: " + (r.stderr.strip().splitlines() or ["see console"])[-1])
        st.cache_data.clear()
        st.rerun()
    try:
        cookies = espn.auth(HOME)
    except ValueError:
        return st.error("espn_auth.json is not valid JSON.")
    if not cookies or not s.get("league_id"):
        return st.info(f"The Season tab reads your ESPN league. Create {HOME / 'espn_auth.json'} containing "
                       '{"espn_s2": "...", "SWID": "{...}"} (the two cookies of a logged-in fantasy.espn.com '
                       "session) and make sure settings.json has your league_id, then reload.")
    try:
        js = espn_league(str(HOME), s["league_id"], s["season"])
    except RuntimeError as e:
        return st.error(f"ESPN refused the request - the cookies in espn_auth.json are probably expired. ({e})")
    m = espn.matchups(js, sched)  # ESPN's own calendar; the placeholder (pre-season) keeps the calendar-week fallback
    real = espn.is_real_calendar(m, js)
    per = calendar(days, sched, s, m)[0] if real else period
    win = window(days, per, pd.Timestamp.now("America/New_York").strftime("%Y-%m-%d"))
    if win is None:
        return st.info("The season is over.")
    p, wk, ros = win
    mid = int(np.sort(m.matchup.unique())[p]) if real else int(p) + 1  # ESPN's matchup id
    idx = dict(zip(b.player_id, b.index))
    rows = espn.rosters(js)
    mine, active = split_roster(rows[rows.team_id == s["espn_team_id"]], idx)
    fit, cap = slot_fit(b, s)
    evw = week_ev(b, ev, wk)
    if s.get("format") == "categories":
        return render_cat_season(js, rows, idx, mine, active, wk, ros, mid, int((per == p).sum()))
    st.subheader(f"This week's matchup (matchup {mid}, {int(wk.sum())} game days left)")
    sc = espn.scores(js, mid, s["espn_team_id"])
    if sc is None:
        st.info("No head-to-head game found for this matchup.")
    else:
        opp = split_roster(rows[rows.team_id == sc["opp_id"]], idx)[1]
        (m_ev, m_var), (o_ev, o_var) = (rest_of_matchup(b, evw, fit, cap, t) for t in (active, opp))
        m = matchup_projection(sc["me"], sc["opp"], m_ev, o_ev, m_var, o_var)
        names = {t["id"]: t["name"] for t in js["teams"]}
        st.dataframe(pd.DataFrame({"Team": [names[s["espn_team_id"]] + " (you)", names.get(sc["opp_id"], "Opponent")],
                                   "Points so far": [sc["me"], sc["opp"]],
                                   "Projected final": [m["me"], m["opp"]]}).round(0), hide_index=True)
        st.metric("Win chance", f"{m['win']:.0%}")
    st.subheader("Pickups")
    if not mine:
        return st.info("Your roster is empty (the draft hasn't happened yet).")
    free = b.index[~b.player_id.isin(rows.player_id)]
    pk = pickups(b, ev, G, mine, active, free, wk, ros, s)
    st.dataframe(pd.DataFrame({"Player": pk.name, "Pos": pk.pos, "Games left this matchup": pk.games.round(1),
                               "+Pts this matchup": pk.week_gain.round(0), "+Pts rest of season": pk.ros_gain.round(0),
                               "★": pk.star.map({True: "★", False: ""}), "Drop": pk["drop"]}), hide_index=True)
    st.caption("Points added to your daily best lineup if you add the player and drop the named one (never an IR "
               "player). Out players count 0 games this matchup, day-to-day players half. ★ = adds points over the "
               "rest of the season too. Injury statuses come from the last Refresh.")


def render_cat_season(js, rows, idx, mine, active, wk, ros, mid, days_total):
    """Season tab for H2H-category leagues: per-category projection, expected categories won, pickups."""
    cats, me = s["categories"], s["espn_team_id"]
    fit, cap = slot_fit(b, s)
    val = value_series(b, s).values
    avail, inj = b.avail.values, b.injury_status.map(INJURY).fillna(1.0).values
    evw = week_ev(b, ((val - val.min() + 1) * avail)[:, None] * G, wk)  # shifted so every player on a game day can start
    st.subheader(f"This week's matchup (matchup {mid}, {int(wk.sum())} game days left)")
    sc = cat_scores(js, mid, me)
    if sc is None:
        return st.info("No head-to-head game found for this matchup.")
    opp = split_roster(rows[rows.team_id == sc["opp_id"]], idx)[1]
    lines = [cat_lines(b, evw, fit, cap, t, avail * inj, cats) for t in (active, opp)]
    frac = (days_total - wk.sum()) / max(wk.sum(), 1)  # days played / days left, to guess attempts so far
    so_me, so_opp = (so_far_from_scores(sc[k], l, cats, frac) for k, l in zip(("me", "opp"), lines))
    df, won = cat_matchup(lines[0], lines[1], so_me, so_opp, cats)
    fmt = lambda c, x: f"{x:.3f}" if c.endswith("%") else f"{x:.1f}"
    st.dataframe(pd.DataFrame({"Cat": df.cat, "Me": [fmt(c, x) for c, x in zip(df.cat, df.me)],
                               "Opp": [fmt(c, x) for c, x in zip(df.cat, df.opp)],
                               "Win %": (df.p_win * 100).round(0)}), hide_index=True)
    st.markdown(f"**Expected categories: {won:.1f} – {len(cats) - won:.1f}**")
    st.subheader("Pickups")
    if not mine:
        return st.info("Your roster is empty (the draft hasn't happened yet).")
    free = b.index[~b.player_id.isin(rows.player_id)]
    pk = cat_pickups(b, val, G, mine, active, free, wk, ros, s, lines[1], so_me, so_opp)
    st.dataframe(pd.DataFrame({"Player": pk.name, "Pos": pk.pos, "Games left this matchup": pk.games.round(1),
                               "+Cats this matchup": pk.cat_gain.round(2), "+Value ROS": pk.ros_gain.round(1),
                               "★": pk.star.map({True: "★", False: ""}), "Drop": pk["drop"]}), hide_index=True)
    limit = s.get("adds_per_week", 4)
    left = adds_left(js, me, mid, limit)
    st.caption(f"Adds left this week: {left} of {limit}." if left is not None else f"{limit} adds per week.")
    st.caption("+Cats = expected categories won this matchup gained by adding the player and dropping the named one "
               "(never an IR player). +Value ROS = rest-of-season lineup value gained. ★ = helps over the rest of "
               "the season too. Out players count 0 games this matchup, day-to-day players half.")


draft_tab, season_tab = st.tabs(["Draft", "Season"])
with draft_tab:
    picks = load_picks()  # every rerun: the extension's server rewrites draft.json while a draft is running
    idx = dict(zip(b.player_id, b.index))
    pname = dict(zip(b.player_id, b["name"]))
    pos = dict(zip(b.player_id, b.positions))
    order = snake(s["teams"], s["rounds"])
    me = s["my_slot"] - 1
    n = len(picks)
    taken = {idx.get(p) for p in picks}  # None = unlisted player
    # advise counts my "not listed"/unknown picks (F4) as spent rounds and bench spots
    adv = advise(b, ev, [{"overall": i + 1, "player": idx.get(p)} for i, p in enumerate(picks)], s,
                 s["strategy"], po_mask if s["strategy"] == "playoffs" else None)

    done = n >= len(order)
    if done:
        st.header("Draft complete")
    else:
        on_clock = order[n]
        rnd, k = divmod(n, s["teams"])
        st.header(f"Round {rnd + 1}, pick {k + 1}: " + ("YOUR PICK" if on_clock == me else f"Team {on_clock + 1} on the clock"))
        avail = b[~b.index.isin(list(taken))].sort_values("season_fp", ascending=False)
        choice = st.selectbox("Who was just picked?", [-1] + avail.player_id.tolist(), index=None,
                              placeholder="Start typing a name...", key=f"pick_{n}",
                              format_func=lambda p: UNLISTED if p == -1 else f"{pname[p]} ({pos[p]})")
    c1, c2 = st.columns(2)
    if not done and c1.button("Drafted", type="primary", disabled=choice is None, key="drafted"):
        picks.append(None if choice == -1 else int(choice))
        save_picks(picks)
        st.rerun()
    if c2.button("Undo last pick", disabled=not picks, key="undo"):
        picks.pop()
        save_picks(picks)
        st.rerun()
    if not done:
        rec = adv["top"]
        st.subheader("Take now" if on_clock == me else "Best options for your next pick")
        st.dataframe(pd.DataFrame({
            "Player": rec["name"] + b.loc[rec.index, "injury_status"].map(lambda x: "" if x == "ACTIVE" else f" ({x})"),
            "Pos": rec.pos,
            PER_GAME: rec.fp_pg.round(1),
            "Games": rec.games.round(0),
            "Value": rec.value.round(0),
            "% still there": (rec.p_back * 100).round(0),
        }), hide_index=True)
        st.caption(("Value = season category value (summed G-scores over replacement) the player adds to your lineup. "
                    if CATS else "Value = season points the player adds to your lineup. ") +
                   "% still there = chance they are still available at your next pick.")
        if rec.attrs["fallback"]:  # M7
            st.caption("No fully legal pick left - showing best available.")

    left, right = st.columns(2)
    left.subheader("My roster")
    roster = pd.DataFrame(adv["roster"], columns=["name", "pos", "fp_pg", "games"]).astype({"fp_pg": float, "games": float})
    left.dataframe(roster.round({"fp_pg": 1, "games": 0}).rename(
        columns={"name": "Player", "pos": "Pos", "fp_pg": PER_GAME, "games": "Games"}), hide_index=True)
    if adv["needs"]:
        left.caption("Still needed: " + ", ".join(adv["needs"]))
    right.subheader("Pick log")
    right.dataframe(pd.DataFrame({
        "pick": list(range(n, 0, -1)),
        "team": [str(order[i] + 1) if i < len(order) else "?" for i in range(n - 1, -1, -1)],
        "player": [pname.get(p, UNLISTED) for p in reversed(picks)]}), hide_index=True)
with season_tab:
    render_season()
