# 06 — Integration & Over-Engineering Review Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. Invoke the `ponytail` skill first, then `ponytail:ponytail-review` in Task 2.

**Agent model:** `haiku` — checklist-driven verification. You verify and report; you only fix things this document explicitly allows. Anything else goes in the report for the orchestrator.

**Goal:** Prove the whole system works end to end against the spec's success criteria, remove over-engineering without changing behavior, and leave the repo runnable from its README.

**Architecture:** No new modules. Runs every command a user would run, checks each spec success criterion with evidence, runs the ponytail over-engineering review, and pushes.

**Tech Stack:** pytest, Streamlit AppTest, git, gh.

**Spec:** [../spec.md](../spec.md) — sections 4 and 7.

## Global Constraints

- Allowed edits: deleting dead code or duplication flagged by `ponytail-review` **only if all tests still pass afterwards**; fixing README commands that don't work verbatim; fixing a crash that has a one-line obvious cause (add a test for it first).
- Not allowed: changing formulas, strategy coefficients, model features, file formats, or widget keys. Report those instead.
- Don't delete `data/` (the ESPN fetch takes ~10 min).

## Review Focus

1. Accented player names (Jokić, Dončić, Nurkić) show correctly in the UI and CSVs (UTF-8 end to end on Windows).
2. Reducing "Teams" below the current "Playoff teams" value in the sidebar → no Streamlit error.
3. Fresh clone with no `data/` → app shows the "run fetch.py" message, not a traceback.
4. Restarting the app mid-draft reloads picks from `data/draft.json`.
5. Scoring edit in the sidebar (e.g. BLK 2 → 4) visibly re-ranks recommendations without re-running `model.py`.

---

### Task 1: Success criteria with evidence

- [ ] **Step 1: Full test suite**

Run: `pytest -v`
Expected: all pass (fetch 5, model 6, draft 14, sim 11, app 3 = 39). Paste the summary line.

- [ ] **Step 2: Criterion 1: data + model**

Run: `python model.py --backtest` then `python model.py`
Expected: `mae_model < mae_last`; top-15 printout of recognizable stars.

- [ ] **Step 3: Criterion 2: recommendation speed**

Run:
```bash
python -c "import json,time,numpy as np,pandas as pd; from draft import *; s=json.load(open('settings.json')); b=board(pd.read_csv('data/projections.csv'),s); G,_=day_matrix(b,pd.read_csv('data/schedule.csv')); ev=ev_matrix(b,G); mine=list(range(7)); t=time.time(); recommend(b,ev,mine,set(range(100)),s); print(round(time.time()-t,2),'s')"
```
Expected: < 2 s.

- [ ] **Step 4: Criterion 3: simulator**

Run: `python sim.py --slot 3 --drafts 3 --seasons 3`
Expected: 10-row table, `best: ...`, `data/strategy_slot3.json` exists.

- [ ] **Step 5: Criterion 5: scoring edit changes ranking without retraining**

Run:
```bash
python -c "import json,pandas as pd; from draft import *; s=json.load(open('settings.json')); p=pd.read_csv('data/projections.csv'); sc=pd.read_csv('data/schedule.csv'); top=lambda s: (lambda b: recommend(b, ev_matrix(b, day_matrix(b, sc)[0]), [], set(), s)['name'].head(10).tolist())(board(p, s)); a=top(s); s['scoring']['BLK']=6; s['scoring']['PTS']=0.5; print(a); print(top(s)); print('changed:', a != top(s))"
```
Expected: `changed: True`.

- [ ] **Step 6: Review Focus 1–4 manual checks**

- UTF-8: `python -c "import pandas as pd; p=pd.read_csv('data/players.csv'); print(p[p.name.str.contains('Jok|Don', regex=True)].name.tolist())"` should print the diacritics correctly. If they are mangled, report which step mangles them.
- For focus items 2–4, write an AppTest check in `tests/test_app.py` for each (same fixture style), run it, and report pass/fail. Fix only if the cause is a one-line obvious fix.

### Task 2: Over-engineering review

- [ ] **Step 1:** Invoke `ponytail:ponytail-review` on `fetch.py model.py draft.py sim.py app.py`.
- [ ] **Step 2:** For each finding that is a pure deletion or simplification: apply it, run `pytest -q`, and keep it only if green. Revert otherwise.
- [ ] **Step 3:** List findings you did not apply, with one line each on why.
- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: ponytail review cleanups"
```

### Task 3: README and push

- [ ] **Step 1:** Follow `README.md` "Run it" commands verbatim in a fresh terminal. Fix any command that doesn't work as written.
- [ ] **Step 2:** Run `pytest -q` one last time.
- [ ] **Step 3: Commit and push**

```bash
git add -A
git commit -m "docs: verified run instructions"
git push
```

Report back: the evidence for each success criterion (command + key output line), review-focus results, applied and skipped ponytail findings, and anything the orchestrator must decide.
