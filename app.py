"""Draft-day assistant.  Run: streamlit run app.py"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from draft import STRATEGIES, advise, board, day_matrix, ev_matrix, snake, starters
from fetch import SLOT_NAMES
from sim import calendar, validate

ROOT = Path(__file__).parent
# DRAFT_HOME lets tests point settings/picks/data at a tmp dir instead of the real repo (M8); sim.py's
# heavy simulation always runs against the real repo regardless (see the "simulate" button below).
HOME = Path(os.environ.get("DRAFT_HOME", ROOT))
DATA = HOME / "data"
SETTINGS = HOME / "settings.json"
DRAFT = DATA / "draft.json"
UNLISTED = "(player not listed)"
LEAGUE_KEYS = ("teams", "rounds", "slots", "scoring", "playoff_teams", "playoff_weeks")

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
        if sim.get("league") == {k: s[k] for k in LEAGUE_KEYS}:
            default = sim["best"]
        else:  # M10: league settings changed since this simulation ran; don't trust it
            st.caption("Strategy results are for different league settings - rerun.")
            sim = None
            default = "greedy"
    strat_names = list(STRATEGIES)
    with st.expander("Strategy"):
        s["strategy"] = st.selectbox("Strategy", strat_names,
                                     index=strat_names.index(default) if default in strat_names else 0, key="strategy")
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
            st.session_state.picks = []
            save_picks([])
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

if "picks" not in st.session_state:
    st.session_state.picks = load_picks()
picks = st.session_state.picks
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
        "Pts/game": rec.fp_pg.round(1),
        "Games": rec.games.round(0),
        "Value": rec.value.round(0),
        "% still there": (rec.p_back * 100).round(0),
    }), hide_index=True)
    st.caption("Value = season points the player adds to your lineup. "
               "% still there = chance they are still available at your next pick.")
    if rec.attrs["fallback"]:  # M7
        st.caption("No fully legal pick left - showing best available.")

left, right = st.columns(2)
left.subheader("My roster")
roster = pd.DataFrame(adv["roster"], columns=["name", "pos", "fp_pg", "games"]).astype({"fp_pg": float, "games": float})
left.dataframe(roster.round({"fp_pg": 1, "games": 0}).rename(
    columns={"name": "Player", "pos": "Pos", "fp_pg": "Pts/game", "games": "Games"}), hide_index=True)
if adv["needs"]:
    left.caption("Still needed: " + ", ".join(adv["needs"]))
right.subheader("Pick log")
right.dataframe(pd.DataFrame({
    "pick": list(range(n, 0, -1)),
    "team": [str(order[i] + 1) if i < len(order) else "?" for i in range(n - 1, -1, -1)],
    "player": [pname.get(p, UNLISTED) for p in reversed(picks)]}), hide_index=True)
