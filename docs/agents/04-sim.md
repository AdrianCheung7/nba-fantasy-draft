# 04 — Strategy Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Invoke the `ponytail` skill before writing code.

**Agent model:** `opus` — this is the hardest-reasoning part of the project (Monte Carlo design, correctness of season/bracket logic, performance tuning). Think about statistical validity, not just passing tests.

**Goal:** For my draft slot, simulate many full drafts (opponents draft "optimally", or by ESPN ADP) followed by many H2H seasons, and report which draft strategy gives the highest P(champion) with the lowest P(bottom-3 finish).

**Architecture:** `sim.py` reuses `draft.recommend` for every pick (so the simulator and the live recommendations can never disagree about value). Seasons are simulated day by day: each player gets one randomized contiguous injury spell per season, per-game fantasy points ~ Normal(fp_pg, sd), the manager starts the top-N expected scorers who play that day. Round-robin regular season (circle method), seeded single-elimination playoffs with byes. Common random numbers across strategies (same seeds per draft index) so strategy differences aren't noise.

**Tech Stack:** Python 3.12, numpy, pandas, argparse, pytest.

**Spec:** [../spec.md](../spec.md) — section 3.3, assumptions A1–A3.

## Global Constraints

- Consumes only `draft.py` public functions: `STRATEGIES, board, day_matrix, ev_matrix, feasible, recommend, snake, starters` (signatures in [03-draft.md](03-draft.md)). Do not change `draft.py`; if you must, stop and report why.
- `run(settings, slot, ...)` — `slot` is **1-based** (what the user sees). Internally team index = `slot - 1`.
- Output file: `data/strategy_slot{slot}.json` = `{"best": <strategy name>, "table": [rows]}`.
- Objective for "best": mean over opponent models of `p_champion − p_bottom3`.
- Performance budget: `run(settings, slot, n_drafts=20, n_seasons=20)` for 14 teams × 14 rounds must finish in **< 5 minutes** on a laptop. Measure it (Task 3). If too slow, optimize inside `sim.py` (e.g. pass a smaller `k` to `recommend` for opponents, vectorize seasons) — never by weakening the model silently; report what you changed.
- No new dependencies.

## Review Focus

1. Invalid league settings (slot > teams, 9 playoff teams with 3 playoff weeks) → clear `ValueError`, not an index crash (Task 1 `test_validate`).
2. Odd team count (e.g. 13) → round-robin with byes works (Task 1 `test_round_robin_covers_every_pair`).
3. Zero-noise season must exactly equal the draft engine's `team_value` — proves sim and recommender agree (Task 2 `test_season_matches_team_value`).
4. Strategies compared on different random draws would produce fake differences → common random numbers by draft index (Task 3 code; reviewer checks).
5. Tied weekly scores → half-win each, no crash (Task 1 `test_regular_season_ties`).

---

### Task 1: League mechanics — schedule, standings, playoffs, validation

**Files:**
- Create: `sim.py`
- Test: `tests/test_sim.py`

**Interfaces:**
- Produces: `round_robin(n) -> list[list[tuple[int,int]]]` (−1 = bye), `bracket_order(size) -> list[int]`, `regular_season(weekly: ndarray (teams, weeks), n_reg) -> (rank: ndarray best-first, wins: ndarray)`, `playoffs(seeds: list[int], weekly, weeks: list[int]) -> int` champion team, `validate(settings, slot) -> None`.

- [ ] **Step 1: Write the failing tests** — `tests/test_sim.py`

```python
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim import bracket_order, playoffs, regular_season, round_robin, validate


@pytest.mark.parametrize("n", [4, 5, 14])
def test_round_robin_covers_every_pair(n):
    games = [tuple(sorted(p)) for rnd in round_robin(n) for p in rnd if min(p) >= 0]
    assert len(games) == len(set(games)) == len(list(combinations(range(n), 2)))


def test_bracket_order():
    assert bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]


def test_regular_season_strongest_team_first():
    weekly = np.tile(np.arange(4.0)[:, None], (1, 3))
    rank, wins = regular_season(weekly, 3)
    assert rank.tolist() == [3, 2, 1, 0] and wins[3] == 3


def test_regular_season_ties():
    rank, wins = regular_season(np.ones((4, 3)), 3)
    assert wins.tolist() == [1.5, 1.5, 1.5, 1.5]


def test_playoffs_six_teams_with_byes():
    weekly = np.tile(np.arange(6.0)[:, None], (1, 3))
    assert playoffs([0, 1, 2, 3, 4, 5], weekly, [0, 1, 2]) == 5
    assert playoffs([5, 4, 3, 2, 1, 0], weekly, [0, 1, 2]) == 5
    with pytest.raises(ValueError):
        playoffs(list(range(8)), np.ones((8, 2)), [0, 1])


def test_validate():
    ok = {"teams": 14, "playoff_teams": 6, "playoff_weeks": 3}
    validate(ok, 14)
    for bad, slot in [(ok, 15), (ok, 0), ({**ok, "playoff_teams": 9}, 1), ({**ok, "playoff_teams": 15}, 1)]:
        with pytest.raises(ValueError):
            validate(bad, slot)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sim'`

- [ ] **Step 3: Implement league mechanics in `sim.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sim.py -v`
Expected: 8 passed (round-robin is parametrized ×3).

- [ ] **Step 5: Commit**

```bash
git add sim.py tests/test_sim.py
git commit -m "feat: league mechanics for strategy simulator"
```

### Task 2: Season and draft simulation

**Files:**
- Modify: `sim.py` (append), `tests/test_sim.py` (append)

**Interfaces:**
- Consumes: `draft.board, day_matrix, ev_matrix, feasible, recommend, snake, starters, team_value` (test only).
- Produces: `availability(G, avail, rng) -> ndarray[bool]`, `sim_season(rosters, b, G, week_idx, n_weeks, n_start, rng) -> ndarray (teams, weeks)`, `adp_pick(b, mine, taken, settings, rng) -> int`, `sim_draft(b, ev, settings, me: int (0-based), strategy, opp, po_mask, rng) -> list[list[int]]`.

- [ ] **Step 1: Append failing tests**

```python
from draft import board, day_matrix, ev_matrix, starters, team_value
from model import RATES, STATS
from sim import sim_draft, sim_season


def make_board(rows, settings):
    """rows: (pts_per_game, positions, injury_status, team_id, cv)"""
    proj = pd.DataFrame([{"player_id": 100 + i, "name": f"P{i}", "team_id": t, "positions": pos, "adp": i + 1.0,
                          "injury_status": inj, "age": 25, "gp": 82.0, "cv": cv,
                          **{c: 0.0 for c in STATS + RATES}, "PTS": pts}
                         for i, (pts, pos, inj, t, cv) in enumerate(rows)])
    return board(proj, settings)


def make_schedule(teams=4, weeks=4):
    days = pd.date_range("2026-10-19", periods=7 * weeks)
    return pd.DataFrame([{"team_id": t, "date": d.strftime("%Y-%m-%d"), "week": i // 7 + 1}
                         for i, d in enumerate(days) for t in range(1, teams + 1) if (i + t) % 2 == 0])


TINY = {"teams": 3, "rounds": 2, "slots": {"UTIL": 1, "BE": 1}, "scoring": {"PTS": 1},
        "playoff_teams": 2, "playoff_weeks": 1}


def test_season_matches_team_value():
    s = {**TINY, "slots": {"UTIL": 2, "BE": 1}}
    b = make_board([(30, "UTIL|BE", "ACTIVE", 1, 0.0), (20, "UTIL|BE", "ACTIVE", 2, 0.0),
                    (10, "UTIL|BE", "ACTIVE", 1, 0.0)], s)
    sched = make_schedule()
    G, days = day_matrix(b, sched)
    week = sched.drop_duplicates("date").set_index("date").week.reindex(days).values
    weekly = sim_season([[0, 1, 2]], b, G, week - 1, int(week.max()), 2, np.random.default_rng(0))
    assert weekly.sum() == pytest.approx(team_value(ev_matrix(b, G)[[0, 1, 2]], 2))


def test_sim_draft_unique_full_rosters():
    b = make_board([(40 - 3 * i, "UTIL|BE", "ACTIVE", 1 + i % 4, 0.3) for i in range(10)], TINY)
    G, _ = day_matrix(b, make_schedule())
    ev = ev_matrix(b, G)
    for opp in ("greedy", "adp"):
        rosters = sim_draft(b, ev, TINY, 0, "greedy", opp, None, np.random.default_rng(1))
        picks = [p for r in rosters for p in r]
        assert [len(r) for r in rosters] == [2, 2, 2] and len(set(picks)) == 6
        assert rosters[0][0] == 0  # I pick first, greedy, no noise on my own picks
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_sim.py -v`
Expected: FAIL — `ImportError: cannot import name 'sim_draft'`

- [ ] **Step 3: Append implementation**

```python
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
    order = (free.adp * rng.lognormal(0, NOISE, len(free))).sort_values().index
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sim.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add sim.py tests/test_sim.py
git commit -m "feat: season and draft simulation"
```

### Task 3: `run`, best strategy, CLI, timing

**Files:**
- Modify: `sim.py` (append), `tests/test_sim.py` (append)

**Interfaces:**
- Produces: `trim(b, settings) -> DataFrame`, `run(settings, slot: int (1-based), n_drafts=20, n_seasons=20, seed=0, strategies=None, b=None, schedule=None) -> DataFrame[strategy, opponents, p_champion, p_playoffs, p_bottom3, avg_wins]`, `best(df) -> str`, CLI writing `data/strategy_slot{slot}.json`.

- [ ] **Step 1: Append failing test**

```python
from draft import STRATEGIES
from sim import best, run


def test_run_smoke():
    s = {**TINY, "teams": 4}
    b = make_board([(40 - 3 * i, "UTIL|BE|IR", "OUT" if i == 5 else "ACTIVE", 1 + i % 4, 0.3)
                    for i in range(12)], s)
    df = run(s, 2, n_drafts=2, n_seasons=3, b=b, schedule=make_schedule())
    assert len(df) == len(STRATEGIES) * 2
    assert df[["p_champion", "p_playoffs", "p_bottom3"]].stack().between(0, 1).all()
    assert best(df) in STRATEGIES
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sim.py::test_run_smoke -v`
Expected: FAIL — `ImportError: cannot import name 'best'`

- [ ] **Step 3: Append implementation**

```python
def trim(b, settings):
    """Speed: drop players nobody drafts, but keep injured stars (IR stash candidates)."""
    n = settings["teams"] * settings["rounds"] + 80
    keep = (b.season_fp.rank(ascending=False) <= n) | (b.ir_ok & (b.fp_pg.rank(ascending=False) <= 150))
    return b[keep].reset_index(drop=True)


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
    n_start, me, p, teams = len(starters(settings)), slot - 1, settings["playoff_teams"], settings["teams"]
    rows = []
    for strat in strategies or list(STRATEGIES):
        for opp in OPPONENTS:
            m = np.zeros(4)
            for d in range(n_drafts):
                # common random numbers: draft d / its seasons use the same seeds for every strategy
                rosters = sim_draft(b, ev, settings, me, strat, opp, po_mask, np.random.default_rng([seed, d, 0]))
                rng = np.random.default_rng([seed, d, 1])
                for _ in range(n_seasons):
                    weekly = sim_season(rosters, b, G, week - 1, n_weeks, n_start, rng)
                    rank, wins = regular_season(weekly, n_reg)
                    place = int(np.flatnonzero(rank == me)[0])
                    m += [playoffs(list(rank[:p]), weekly, po_weeks) == me, place < p, place >= teams - 3, wins[me]]
            m /= n_drafts * n_seasons
            rows.append({"strategy": strat, "opponents": opp, "p_champion": m[0], "p_playoffs": m[1],
                         "p_bottom3": m[2], "avg_wins": m[3]})
    return pd.DataFrame(rows)


def best(df):
    return df.assign(obj=df.p_champion - df.p_bottom3).groupby("strategy").obj.mean().idxmax()


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
```

- [ ] **Step 4: Run all sim tests**

Run: `pytest tests/test_sim.py -v`
Expected: 11 passed.

- [ ] **Step 5: Timing and sanity on real data** (requires 01–03 done)

Run: `python -c "import time,subprocess,sys; t=time.time(); subprocess.run([sys.executable,'sim.py','--slot','7','--drafts','3','--seasons','5']); print(round(time.time()-t),'s')"`
Expected: a 10-row table, `best: <name>`, and a time. Extrapolate to `--drafts 20 --seasons 20`; if > 5 min, optimize `sim.py` (profile first with `python -m cProfile -s cumtime sim.py --slot 7 --drafts 2 --seasons 2 | head -30`) and re-measure. Sanity: `greedy` vs `greedy` opponents should have p_playoffs ≈ playoff_teams/teams (≈0.43) — a big deviation means a bug, investigate before continuing.

- [ ] **Step 6: Full run for one slot**

Run: `python sim.py --slot 7`
Expected: table + `data/strategy_slot7.json`. Record the table in your report.

- [ ] **Step 7: Commit**

```bash
git add sim.py tests/test_sim.py
git commit -m "feat: strategy simulator run, best strategy selection, CLI"
```

Report back: test output, timing before/after any optimization, the slot-7 table, and your judgment on whether differences between strategies exceed simulation noise (compute the standard error of p_champion: sqrt(p(1-p)/(drafts·seasons))).
