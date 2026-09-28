# Spec v2 — live draft extension + in-season tab

v1 (docs/spec.md) stays authoritative for the model, draft engine and simulator unless changed here.
Decisions come from the 2026-09-28 grilling session with the user (Q1–Q23).

## Facts (read from ESPN 2026-09-28)

- League "Bronny's Boys", fba leagueId **178869420**, season 2027, private → cookies in git-ignored
  `espn_auth.json` (`{"espn_s2": "...", "SWID": "..."}`; user fills it; never print, log or commit it).
- User = team **"Raj"**, ESPN teamId **13**, draft slot **9**. Draft **Wed 2026-09-30 20:00 local**, snake,
  **12 rounds** (IR slots are not drafted: rounds = sum(slots) − IR).
- Scoring and lineup slots match settings.json. Lineup locks per game.
- Schedule: 19 regular-season matchups, **4 playoff teams, each playoff round = 2 matchup periods** (4 periods).
  Some matchup periods span >1 calendar week → use ESPN's matchup calendar, not Mon–Sun weeks.
- Adds: **2 per day** (user decision). Waivers: traditional, 24 h. Trade deadline 2027-02-19.
- ESPN's league API does **not** expose picks during a live draft (`playerId -1` while `inProgress`).
  Picks are only readable from the draft page's **Picks** panel (fixture: `tests/fixtures/espn_picks.html`):
  `li.pick-message__container` → `.playerinfo__playername`, `.playerinfo__playerteam`, `.playerinfo__playerpos`,
  `.pick-info` text `"R<round>, P<pick-in-round> - <fantasy team>"`; headshot `img` src holds the ESPN player id
  (`players/full/<id>.png`, else `nomug.png`). Page order is not guaranteed — sort by round/pick.
- Draft page URL: `https://fantasy.espn.com/basketball/draft?leagueId=<id>&seasonId=2027&teamId=<my team>...`.
  Practice drafts (mock lobby → "League Specific Practice Draft") use the same page and league settings.
- `recommend` + `survive` currently take ~0.01 s; compute is not the bottleneck.

## Decisions

| # | Decision |
|---|---|
| D1 | League settings come from ESPN (overwrite settings.json on sync). Sidebar stays editable as fallback. |
| D2 | Opponent model for "% back next turn": each opponent's pick depends on **their roster needs** and **ESPN's rank list** (what humans and autopick follow), not just value. |
| D3 | Daily lineup value respects **position slots** (G, G, F, SG/SF, PF/C, UTIL). |
| D4 | Extension reads every pick from the page, sends them to a local server, **highlights** ESPN's player list: #1 green outline, #2–5 amber, tag `#k · NN% back`. Nothing else on ESPN's page changes. |
| D5 | Chrome **side panel** = fast native panel (top 10, my roster, positions still needed) fed by a tiny local server; update < 1 s after a pick appears. Streamlit remains for setup + Season tab. |
| D6 | Launch: app starts silently at Windows login (Startup folder entry, removable). Installing that entry needs the user's explicit OK at install time. |
| D7 | Streamlit app gets two tabs: **Draft** (existing, now via `advise`) and **Season**. |
| D8 | Season tab v1: **Pickups** (points added this matchup + rest of season vs best drop; ★ when ROS gain > 0; sorted by this matchup since 2 adds/day makes streaming worthwhile), **Matchup** (score so far + projected rest, win %), **Refresh data** button. Trade check = v1.1 (not now). |
| D9 | Projection refresh blends this season's per-game stats: weight = n / (n + 20), n = games played. |
| D10 | Injuries this matchup: OUT / INJURY_RESERVE → 0 games left; DAY_TO_DAY → 50 %; others → scheduled. |
| D11 | Drop candidate = roster player whose removal costs the least lineup value; never a player in an IR slot. |
| D12 | Win % = Normal approx: (my points so far + projected rest) − (theirs), sd from per-game sd of expected starters. |
| D13 | Rosters/free agents/scores sync on app open (cached 10 min); stats + model only on the Refresh button. |
| D14 | Mock evaluation: `python sim.py --report --slot 9` — our strategy vs 13 optimal opponents, many drafts × seasons, per-team average points / wins / playoff % / title %. Plus a live ESPN practice draft run through the extension, whose 14 real rosters are then season-simulated the same way. |

## Components and owners

Worktree data: `data/` and `espn_auth.json` are git-ignored, so a worktree has neither. Read real data from
`C:\Users\adijo\OneDrive\Desktop\nba-fantasy-draft\data` (copy into the worktree's `data/` if needed; never write
to the original). Tests must not need the network or `espn_auth.json` (use small JSON fixtures).

### A — `espn.py` + fetch changes (agent A, wave 1)
- `auth(home)` → cookies dict from `home/espn_auth.json` or None. `league(views, league_id, season, cookies)` → JSON.
- `settings_from_league(js, settings)` → new settings dict: teams, rounds (= slots total − IR), slots (map ESPN
  slot ids via `fetch.SLOT_NAMES`), scoring (ESPN statIds 0,1,2,3,4,5,7,8,10,11,13,15,17,23,24,25,38,39,43 →
  our names), playoff_teams, `playoff_round_periods` (= playoffMatchupPeriodLength), `playoff_weeks` (= rounds of
  bracket × round periods), `league_id`, `espn_team_id` (owner SWID match), `my_team_name`, `my_slot` (index in
  draftSettings.pickOrder + 1), `adds_per_day` (keep existing value, default 2).
- `matchups(js, schedule)` → DataFrame `date, matchup, playoff` from `settings.scheduleSettings.matchupPeriods`
  (period → scoring-period ids; scoring period 1 = first NBA game date in schedule.csv). Saved as `data/matchups.csv`.
- `rosters(js)` → DataFrame `team_id, team_name, player_id, lineup_slot` ; `scores(js, matchup)` → my and
  opponent's points so far this matchup + opponent team_id.
- `python espn.py` = sync: writes settings.json (atomic) and data/matchups.csv; prints a one-line summary. No secrets printed.
- fetch.py: add `espn_rank` column to players.csv (`draftRanksByRankType.POINTS.rank`, fallback STANDARD, else 999).

### B — draft engine + simulator (agent B, wave 1; owns draft.py, sim.py, their tests; may fix call sites in app.py)
- D3: position-aware `team_value`. Players' value is slot-independent, so the matchable sets form a transversal
  matroid → per day, greedily add players by expected points if the set stays matchable to the starter slots.
  Budget: `recommend` ≤ 0.5 s; full `sim.py --slot` ≤ 2× current wall time.
- D2: `survive(...)` gains an opponent model: simulate the teams picking before my next turn in order; each picks
  from the top of ESPN's rank list (`espn_rank`, noisy) blended with value, down-weighting players who can't start
  for that team (roster needs via `matched`). Calibrate noise so the model is neither all-rank nor all-value.
- New `advise(b, ev, picks, settings, strategy="greedy")` — the single entry point for server and app.
  `picks` = list of `{"overall": int, "player": int|None}` (player = board index; None = unlisted). Derives each
  team from snake order + `my_slot`. Returns `{"on_clock": team, "my_next": overall|None, "top": DataFrame
  (player_id, name, pos, fp_pg, games, value, p_back; top 10; attrs["fallback"]), "roster": [...], "needs": [...]}`.
- sim.py: playoff rounds of `playoff_round_periods` matchup periods (score summed over the round); regular season
  and playoffs follow `data/matchups.csv` when present, else calendar weeks. ADP opponent scenario → ESPN-rank scenario.
- `--report --slot N [--drafts --seasons]`: D14 table, printed and saved to `data/report_slot{N}.json`.
  Also `report_from_rosters(rosters, ...)` so a real/practice draft's 14 rosters can be season-simulated.

### C — local server + Chrome extension + launcher (agent C, wave 2)
- `server.py` (stdlib `ThreadingHTTPServer`, 127.0.0.1:8502 only): `POST /picks` body
  `{"picks":[{"round","pick","overall","espn_id"|null,"name","nba_team","pos","fantasy_team"}], "url": "..."}`
  (full list every time; idempotent) → map to board index by espn_id, then normalized name (accents stripped) +
  NBA team, then name → persist `data/draft.json` in the app's format → return `advise` JSON.
  `GET /state` → last result. CORS for the extension. Loads board/ev once at startup; reload on settings change.
- `extension/` (Manifest V3): content script on `https://fantasy.espn.com/basketball/draft*`; `parse.js`
  (pure, no DOM globals beyond a passed root) parses the Picks panel; MutationObserver + 150 ms debounce → POST;
  applies D4 highlights to rows in the player list whose `.playerinfo__playername` matches (skip the Picks panel);
  re-apply after ESPN re-renders. Side panel (`sidepanel.html/js`) shows D5 content; clear "server not running" state.
  Tests: `node --test` on `parse.js` against `tests/fixtures/espn_picks.html`.
- `start.pyw`: starts server.py + `streamlit run app.py --server.headless true` without console windows; no-op if
  ports 8502/8501 are already in use. `start.pyw --install` / `--uninstall` adds/removes a Startup-folder entry
  (only run by the user or with the user's OK).

### D — Season tab + app tabs + projection blend (agent D, wave 2; owns app.py, new season.py, model.py blend)
- app.py: `st.tabs(["Draft", "Season"])`; Draft tab uses `advise`; sidebar "Sync league from ESPN" button.
- season.py: `pickups(...)`, `drop_candidate(...)`, `matchup_projection(...)` per D8, D10–D12, using the
  position-aware `team_value` over the remaining days of the current matchup and the rest of season.
- model.py: `blend(proj, current_season_games, k=20)` per D9; Refresh button runs fetch (current season) + model.

### E — review (wave 3, strongest model) then integration by the controller
Controller: merge, full `pytest` + `node --test`, run `sim.py --report --slot 9`, user loads the unpacked
extension (Developer mode is the user's toggle), rehearse in an ESPN practice draft, season-simulate its rosters.

## Out of scope (v2)
Trade check (v1.1), writing to ESPN (lineups, adds), punt strategies, auto lineup setting.
