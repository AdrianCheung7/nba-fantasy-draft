"""Draft-day assistant.  Run: streamlit run app.py"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from draft import STRATEGIES, board, day_matrix, ev_matrix, picks_before_next, recommend, snake, survive
from fetch import SLOT_NAMES

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SETTINGS = ROOT / "settings.json"
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


def short_pos(p):
    return ", ".join(x for x in p.split("|") if x not in ("BE", "IR", "UTIL"))


missing = [f for f in ("projections.csv", "schedule.csv") if not (DATA / f).exists()]
if missing:
    st.error(f"Missing data files: {missing}. Run `python fetch.py` then `python model.py`, then reload.")
    st.stop()

s = json.loads(SETTINGS.read_text())

with st.sidebar:
    s["teams"] = st.selectbox("Teams", list(range(8, 21)), index=min(max(s["teams"], 8), 20) - 8, key="teams")
    s["my_slot"] = st.selectbox("My draft slot", list(range(1, s["teams"] + 1)),
                                index=min(s["my_slot"], s["teams"]) - 1, key="my_slot")
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
        if st.button("Save settings", key="save"):
            SETTINGS.write_text(json.dumps(s, indent=2))
            st.success("Saved to settings.json")
    if s["rounds"] > sum(s["slots"].values()):
        st.warning("More rounds than roster spots: extra picks can't be rostered.")

    strat_file = DATA / f"strategy_slot{s['my_slot']}.json"
    sim = json.loads(strat_file.read_text()) if strat_file.exists() else None
    strat_names = list(STRATEGIES)
    default = sim["best"] if sim else s.get("strategy", "greedy")
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

b = board(load_csv("projections.csv"), s)
sched = load_csv("schedule.csv")
G, days = day_matrix(b, sched)
ev = ev_matrix(b, G)
week = sched.drop_duplicates("date").set_index("date").week.reindex(days).values
po_mask = week > week.max() - s["playoff_weeks"]

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
mine = [idx[p] for p, t in zip(picks, order) if t == me and p in idx]

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
    gap = picks_before_next(order, n if on_clock == me else n - 1, me)
    rec = recommend(b, ev, mine, taken, s, s["strategy"], po_mask if s["strategy"] == "playoffs" else None).head(10)
    rec["survive"] = survive(b, list(rec.index), taken, gap, np.random.default_rng(0)).values
    st.subheader("Take now" if on_clock == me else "Best options for your next pick")
    st.dataframe(pd.DataFrame({
        "Player": rec["name"] + rec.injury_status.map(lambda x: "" if x == "ACTIVE" else f" ({x})"),
        "Pos": rec.positions.map(short_pos),
        "Pts/game": rec.fp_pg.round(1),
        "Games": rec.gp.round(0),
        "Value": rec.marginal.round(0),
        "% still there": (rec.survive * 100).round(0),
    }), hide_index=True)
    st.caption("Value = season points the player adds to your lineup. "
               "% still there = chance they are still available at your next pick.")

left, right = st.columns(2)
left.subheader("My roster")
left.dataframe(pd.DataFrame({"Player": b.loc[mine, "name"], "Pos": b.loc[mine, "positions"].map(short_pos),
                             "Pts/game": b.loc[mine, "fp_pg"].round(1), "Games": b.loc[mine, "gp"].round(0)}),
               hide_index=True)
right.subheader("Pick log")
right.dataframe(pd.DataFrame({
    "pick": list(range(n, 0, -1)),
    "team": [str(order[i] + 1) if i < len(order) else "?" for i in range(n - 1, -1, -1)],
    "player": [pname.get(p, UNLISTED) for p in reversed(picks)]}), hide_index=True)
