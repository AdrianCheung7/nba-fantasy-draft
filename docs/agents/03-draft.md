# 03 — Draft Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Invoke the `ponytail` skill before writing code.

**Agent model:** `sonnet`

**Goal:** Given projections, league settings, my picks and everyone's picks, recommend the best legal pick under a chosen strategy, and estimate who will still be there at my next turn.

**Architecture:** `draft.py` — pure functions, no I/O. A player's value to *my* team is the marginal gain in a season of ESPN-style daily lineups (each day, the top-N expected scorers who have a game). Legality = healthy-player cap (IR spots only for OUT/INJURY_RESERVE players) + remaining picks can still fill every starter slot (bipartite matching). Strategies are small score functions in a dict so the simulator (04) can compare them.

**Tech Stack:** Python 3.12, numpy, pandas, pytest.

**Spec:** [../spec.md](../spec.md) — section 3.2, assumptions A3–A4.

## Global Constraints

- Input: `data/projections.csv` (contract in [README.md](README.md#shared-contracts-all-agents-rely-on-these-do-not-rename)), `data/schedule.csv`, `settings.json` shape from 01-data.
- `settings["slots"]` keys are ESPN slot names (`PG SG SF PF C G F SG/SF G/F PF/C F/C UTIL BE IR`); `BE`/`IR` are non-starting.
- Board indices are positional (`RangeIndex`); every function takes/returns board indices, never player_ids.
- `taken` may contain `None` (a "player not listed" pick from the UI) — must be ignored.
- No I/O in `draft.py`. No new dependencies. Recommendation for 14 teams must take < 1 s.

## Review Focus

1. Late draft, my healthy spots full and no IR-eligible player left → must still return a recommendation, not an empty table (Task 2 test `test_recommend_falls_back_when_nothing_legal`).
2. Roster of only centers late → guards must be forced so starters can be filled (Task 1 `test_feasible_starter_coverage`).
3. `None` in `taken` (unlisted player) → ignored (Task 2 test).
4. Free agent (`team_id` 0) → plays no games, gets zero marginal value (Task 1 `test_day_matrix_free_agent`).
5. Scoring edited → `board()` recomputes `fp_pg` from stats (covered by `model.points` tests in 02; `board` calls it).

---

### Task 1: Order, board, lineup value, feasibility

**Files:**
- Create: `draft.py`
- Test: `tests/test_draft.py`

**Interfaces:**
- Consumes: `model.points(df, scoring) -> Series`, `model.STATS`, `model.RATES`.
- Produces:
  - `snake(teams: int, rounds: int) -> list[int]` — team index (0-based) for each overall pick.
  - `picks_before_next(order: list[int], overall: int, team: int) -> int | None` — opponent picks between pick `overall` and `team`'s next pick; `None` if no next pick.
  - `starters(settings) -> list[str]` — expanded starting slot names, e.g. `["G","G","F","SG/SF","PF/C","UTIL"]`.
  - `board(proj, settings) -> DataFrame` — adds `fp_pg, avail, sd, season_fp, pos (set[str]), ir_ok (bool)`.
  - `day_matrix(b, schedule) -> (np.ndarray[bool] (players, days), np.ndarray days)`.
  - `ev_matrix(b, G) -> np.ndarray (players, days)` expected points per player-day.
  - `team_value(ev, n_start) -> float | np.ndarray` — works on `(players, days)` or `(k, players, days)`.
  - `matched(pos_sets: list[set], slots: list[str]) -> int`.
  - `feasible(b, mine: list[int], cand: list[int], settings) -> list[int]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_draft.py`

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from draft import (board, day_matrix, ev_matrix, feasible, matched, picks_before_next, snake, starters,
                   team_value)
from model import RATES, STATS

LEAGUE = {"slots": {"G": 2, "F": 1, "SG/SF": 1, "PF/C": 1, "UTIL": 1, "BE": 6, "IR": 2}, "rounds": 14,
          "scoring": {"PTS": 1}}
SCHED = pd.DataFrame({"team_id": [1, 1, 2, 2], "date": ["2026-10-20", "2026-10-21"] * 2, "week": [1] * 4})


def make_board(rows, settings=LEAGUE):
    """rows: (pts_per_game, positions, injury_status, team_id)"""
    proj = pd.DataFrame([{"player_id": 100 + i, "name": f"P{i}", "team_id": t, "positions": pos, "adp": i + 1.0,
                          "injury_status": inj, "age": 25, "gp": 82.0, "cv": 0.3,
                          **{c: 0.0 for c in STATS + RATES}, "PTS": pts}
                         for i, (pts, pos, inj, t) in enumerate(rows)])
    return board(proj, settings)


def test_snake_and_next_pick():
    order = snake(3, 2)
    assert order == [0, 1, 2, 2, 1, 0]
    assert picks_before_next(order, 0, 0) == 4
    assert picks_before_next(order, 2, 2) == 0
    assert picks_before_next(order, 5, 0) is None


def test_starters_expand_slots():
    assert starters(LEAGUE) == ["G", "G", "F", "SG/SF", "PF/C", "UTIL"]


def test_board_columns():
    b = make_board([(20, "G|UTIL|BE|IR", "OUT", 1)])
    r = b.iloc[0]
    assert (r.fp_pg, r.avail, r.season_fp, r.ir_ok) == (20, 1.0, 1640, True)
    assert r.pos == {"G", "UTIL", "BE", "IR"} and r.sd == pytest.approx(6.0)


def test_day_matrix_free_agent():
    b = make_board([(20, "G|UTIL", "ACTIVE", 1), (20, "G|UTIL", "ACTIVE", 0)])
    G, days = day_matrix(b, SCHED)
    assert list(days) == ["2026-10-20", "2026-10-21"]
    assert G.tolist() == [[True, True], [False, False]]
    assert ev_matrix(b, G).tolist() == [[20, 20], [0, 0]]


def test_team_value_daily_best_lineup():
    ev = np.array([[10, 0], [5, 5], [1, 8]], float)
    assert team_value(ev, 1) == 18
    assert team_value(ev, 2) == 28
    assert team_value(np.stack([ev, ev]), 1).tolist() == [18, 18]
    assert team_value(ev[[]], 2) == 0


def test_matched_uses_augmenting_paths():
    assert matched([{"PF/C", "UTIL"}, {"PF/C"}], ["PF/C", "UTIL"]) == 2
    assert matched([{"C", "PF/C", "UTIL"}, {"C", "PF/C", "UTIL"}], ["G", "PF/C", "UTIL"]) == 2


def test_feasible_healthy_cap():
    s = {"slots": {"UTIL": 1, "BE": 1, "IR": 1}, "rounds": 3, "scoring": {"PTS": 1}}
    b = make_board([(30, "UTIL|BE|IR", "ACTIVE", 1)] * 3 + [(5, "UTIL|BE|IR", "OUT", 1)], s)
    assert feasible(b, [0, 1], [2, 3], s) == [3]


def test_feasible_starter_coverage():
    s = {"slots": {"G": 1, "PF/C": 1}, "rounds": 2, "scoring": {"PTS": 1}}
    b = make_board([(30, "C|PF/C|UTIL", "ACTIVE", 1), (25, "C|PF/C|UTIL", "ACTIVE", 1),
                    (10, "G|UTIL", "ACTIVE", 1)], s)
    assert feasible(b, [0], [1, 2], s) == [2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'draft'`

- [ ] **Step 3: Implement `draft.py` (part 1)**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add draft.py tests/test_draft.py
git commit -m "feat: draft board, daily-lineup team value, roster legality"
```

### Task 2: Recommend, strategies, survival odds

**Files:**
- Modify: `draft.py` (append)
- Test: `tests/test_draft.py` (append)

**Interfaces:**
- Consumes: Task 1 functions.
- Produces:
  - `STRATEGIES: dict[str, Callable[[DataFrame, dict], Series]]` with keys `greedy, durable, low_variance, playoffs, ir_early`.
  - `recommend(b, ev, mine: list[int], taken: set, settings, strategy="greedy", po_mask: np.ndarray[bool] | None = None, k=60) -> DataFrame` — board rows plus `marginal, marginal_po, score`, sorted by `score` desc. Never empty while any player is available.
  - `survive(b, cand: list[int], taken: set, n_picks: int | None, rng: np.random.Generator, sims=200) -> Series` — P(still available at my next pick), indexed like `cand`; `n_picks=None` → 0.0.

- [ ] **Step 1: Append failing tests**

```python
from draft import STRATEGIES, recommend, survive

ROWS = [(30, "C|PF/C|UTIL|BE|IR", "ACTIVE", 1), (20, "G|UTIL|BE|IR", "ACTIVE", 1), (10, "G|UTIL|BE|IR", "ACTIVE", 2)]


def rec(rows, mine=(), taken=(), strategy="greedy", settings=LEAGUE, po_mask=None):
    b = make_board(rows, settings)
    G, _ = day_matrix(b, SCHED)
    return recommend(b, ev_matrix(b, G), list(mine), set(taken), settings, strategy, po_mask)


def test_recommend_ranks_and_excludes_taken_and_none():
    r = rec(ROWS, taken={0, None})
    assert list(r.index) == [1, 2]
    assert r.marginal.iloc[0] == 40  # 20 pts x 2 game days


def test_recommend_falls_back_when_nothing_legal():
    s = {"slots": {"UTIL": 1, "IR": 1}, "rounds": 2, "scoring": {"PTS": 1}}
    r = rec([(30, "UTIL|IR", "ACTIVE", 1), (20, "UTIL|IR", "ACTIVE", 1)], mine=[0], settings=s)
    assert list(r.index) == [1]


def test_ir_early_boosts_injured_player():
    rows = [(20, "UTIL|BE|IR", "ACTIVE", 1), (18, "UTIL|BE|IR", "OUT", 1)]
    assert rec(rows).index[0] == 0
    assert rec(rows, strategy="ir_early").index[0] == 1


def test_playoffs_strategy_prefers_playoff_days():
    rows = [(10, "UTIL|BE", "ACTIVE", 1), (10, "UTIL|BE", "ACTIVE", 2)]
    sched_split = pd.DataFrame({"team_id": [1, 2], "date": ["2026-10-20", "2026-10-21"], "week": [1, 1]})
    b = make_board(rows)
    G, _ = day_matrix(b, sched_split)
    r = recommend(b, ev_matrix(b, G), [], set(), LEAGUE, "playoffs", np.array([False, True]))
    assert r.index[0] == 1


def test_every_strategy_scores():
    for name in STRATEGIES:
        assert rec(ROWS, strategy=name).score.notna().all(), name


def test_survive():
    b = make_board([(100, "UTIL", "ACTIVE", 1)] + [(10, "UTIL", "ACTIVE", 1)] * 4)
    rng = np.random.default_rng(0)
    assert survive(b, [0, 4], set(), 0, rng).tolist() == [1.0, 1.0]
    p = survive(b, [0, 4], set(), 1, rng)
    assert p[0] == 0.0 and p[4] > 0.5
    assert survive(b, [0], set(), None, rng).tolist() == [0.0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_draft.py -v`
Expected: FAIL — `ImportError: cannot import name 'STRATEGIES'`

- [ ] **Step 3: Append implementation to `draft.py`**

```python
TOP_K = 60  # ponytail: only the best 60 legal players are scored; ample for 8-20 teams

# ponytail: coefficients are knobs; sim.py measures which strategy actually wins
STRATEGIES = {
    "greedy": lambda c, ctx: c.marginal,
    "durable": lambda c, ctx: c.marginal * np.sqrt(c.avail),
    "low_variance": lambda c, ctx: c.marginal - 2 * c.sd * np.sqrt(c.gp),
    "playoffs": lambda c, ctx: c.marginal + 2 * c.marginal_po,
    "ir_early": lambda c, ctx: c.marginal * np.where(c.ir_ok & ctx["ir_open"], 1.3, 1.0),
}


def recommend(b, ev, mine, taken, settings, strategy="greedy", po_mask=None, k=TOP_K):
    mine = list(mine)
    n_start = len(starters(settings))
    free = b[~b.index.isin(list(taken) + mine)].sort_values("season_fp", ascending=False).index
    cand = []
    for c in free:  # best-first until k legal players
        if feasible(b, mine, [c], settings):
            cand.append(c)
            if len(cand) == k:
                break
    cand = cand or list(free[:k])  # nothing legal (e.g. no IR-eligible left): best available
    out = b.loc[cand].copy()
    rosters = [mine + [c] for c in cand]
    out["marginal"] = team_value(ev[rosters], n_start) - team_value(ev[mine], n_start)
    if po_mask is None:
        out["marginal_po"] = 0.0
    else:
        evp = ev[:, po_mask]
        out["marginal_po"] = team_value(evp[rosters], n_start) - team_value(evp[mine], n_start)
    ctx = {"ir_open": settings["slots"].get("IR", 0) > int(b.ir_ok.values[mine].sum())}
    out["score"] = STRATEGIES[strategy](out, ctx)
    return out.sort_values("score", ascending=False)


def survive(b, cand, taken, n_picks, rng, sims=200):
    """P(candidate still on the board at my next pick); opponents take by noisy season value."""
    if n_picks is None:
        return pd.Series(0.0, index=cand)
    free = b[~b.index.isin(list(taken))]
    pool = free.nlargest(len(cand) + 3 * n_picks + 10, "season_fp")
    noisy = pool.season_fp.values * rng.lognormal(0, 0.15, (sims, len(pool)))
    hit = np.zeros(noisy.shape, bool)
    np.put_along_axis(hit, np.argsort(-noisy, axis=1)[:, :n_picks], True, axis=1)
    return pd.Series(1 - hit.mean(axis=0), index=pool.index).reindex(cand).fillna(1.0)
```

- [ ] **Step 4: Run all draft tests**

Run: `pytest tests/test_draft.py -v`
Expected: 14 passed.

- [ ] **Step 5: Timing check on real data** (requires 01 + 02 outputs)

Run:
```bash
python -c "import json,time,numpy as np,pandas as pd; from draft import *; s=json.load(open('settings.json')); b=board(pd.read_csv('data/projections.csv'),s); G,_=day_matrix(b,pd.read_csv('data/schedule.csv')); ev=ev_matrix(b,G); t=time.time(); r=recommend(b,ev,[],set(),s); print(round(time.time()-t,2),'s'); print(r[['name','positions','fp_pg','gp','marginal']].head(10).to_string())"
```
Expected: under 1 s; top 10 are top-tier fantasy players with sensible positions.

- [ ] **Step 6: Commit**

```bash
git add draft.py tests/test_draft.py
git commit -m "feat: recommendations, strategies, survival odds"
```

Report back: test output, timing, top-10 printout.
