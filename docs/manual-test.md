# Manual test checklist (v2)

Start: double-click `Start Draft App.bat` (or have Claude start it). App = http://localhost:8501, pick server = 127.0.0.1:8502.

## A. Launch
- [ ] A1 `Start Draft App.bat` starts both with no console windows; double-clicking again does nothing (ports already in use).
- [ ] A2 (optional) `python start.pyw --install` → app starts silently after a Windows restart; `--uninstall` removes it.

## B. Sidebar (app)
- [ ] B1 **Teams** dropdown + **My draft slot**: change them, refresh the page → values stick (autosaved to settings.json).
- [ ] B2 **League settings** (collapsed): rounds, roster slots, scoring, playoffs are editable; an impossible combo (e.g. 2 playoff weeks for 8 playoff teams) shows a red error.
- [ ] B3 **Sync league from ESPN**: resets them to the real league → 14 teams, 12 rounds, slot 9, 4 playoff teams.
- [ ] B4 **Strategy**: dropdown (greedy, durable, low_variance, playoffs, ir_early) + "run strategy simulation" (~2.5 min) → table appears, "best: greedy".
- [ ] B5 **Reset draft**: tick "Yes, clear every pick" → Clear picks → empty draft.

## C. Draft tab (app)
- [ ] C1 Header says whose turn it is ("Round 1, pick 1: Team 1 on the clock"; at pick 9 "YOUR PICK").
- [ ] C2 "Who was just picked?" autocomplete: type "jok" → Nikola Jokic → **Drafted** → he leaves the table, pick log grows.
- [ ] C3 "(player not listed)" option records an unknown pick.
- [ ] C4 **Undo last pick** removes the last one.
- [ ] C5 Top-10 table: Player (+ injury status), Pos, Pts/game, Games, Value, % still there. Values change as players go.
- [ ] C6 My roster + "Still needed: G, G, F, …" shrinks as you draft.
- [ ] C7 Picks entered by the extension show up here on the next click/refresh (same draft.json).

## D. Season tab (app) — fully meaningful only after the draft
- [ ] D1 Without `espn_auth.json` → friendly message (rename the file temporarily to check, then rename back).
- [ ] D2 "This week's matchup": you vs opponent, points so far, projected final, win chance (pre-draft: 0–0, 50 %).
- [ ] D3 Pickups (after draft): top free agents with games left this matchup, +pts this matchup, +pts rest of season, ★ for long-term keeps, and who to drop (never an IR player).
- [ ] D4 **Refresh data** (a few minutes): re-downloads stats + rebuilds projections (blends this season's games once they exist).

## E. Chrome extension (needs: Developer mode → Load unpacked → `extension/`)
- [ ] E1 Click the pinned icon → side panel opens.
- [ ] E2 App stopped → side panel shows red "App not running — double-click Start Draft App".
- [ ] E3 In an ESPN practice draft (mock lobby → Bronny's Boys → Practice Draft): every pick (yours and everyone else's) appears in the side panel within ~1 s.
- [ ] E4 Side panel shows whose turn / picks until yours, top 10, your roster, positions still needed; "Draft complete" at the end.
- [ ] E5 ESPN's player list: #1 recommendation green outline, #2–5 amber, tag "#k · NN% back". Highlights move after each pick.
- [ ] E6 Practice draft at a slot other than 9 → amber banner "Your draft slot on this page is N, settings say 9 — using N".
- [ ] E7 Stop the app mid-draft, start it again → side panel recovers, no picks lost, ESPN tab stays responsive.

## F. Command line (optional)
- [ ] F1 `python espn.py` → one-line sync summary.
- [ ] F2 `python sim.py --report --slot 9` → 14-team table: avg points, wins, playoff %, title % (you marked True).
- [ ] F3 After a practice draft: `python sim.py --from-draft` → the same table for the 14 real rosters.
- [ ] F4 `python -m pytest -q` (96 pass) and `node --test` (4 pass).
