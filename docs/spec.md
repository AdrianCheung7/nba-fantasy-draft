# NBA Fantasy Draft Assistant — Design Spec

Date: 2026-09-27 · Status: approved design, awaiting spec review · Season targeted: 2026-27 (ESPN `seasonId` 2027)

## 1. Intent

A local program that, during a live ESPN snake draft, recommends the best pick for me given my draft slot and every pick made so far — driven by a regression projection model on free live NBA data — plus a simulated draft strategy that differs from "everyone drafts optimally" while maximizing my chance to win and minimizing my chance to finish near the bottom.

### What the user specified
- ESPN points league, 14 teams (changeable via dropdown), snake draft, 14 rounds, weekly head-to-head points.
- Scoring (per event): FGM 0.5, FGMI −0.5, FTM 1, FTMI −1, 3PM 2, 3PMI −1, OREB 1.5, DREB 1, AST 1, STL 2, BLK 2, TO −1, EJ −100, FF −2, TF −3, TD 10, QD 100, PTS 1, TW 1. EJ −100 confirmed intentional.
- Roster: G ×2, F ×1, SG/SF ×1, PF/C ×1, UTIL ×1, Bench ×6, IR ×2. IR spots are drafted (14 rounds).
- Editable in the app: team count, my draft slot, roster slot counts, scoring weights.
- Picks entered manually with autocomplete. Draft only (no in-season tools in v1).
- Free data only, no paid keys. Runs locally on a Windows laptop (Python 3.12 present).
- Built by multiple subagents, each on the cheapest model adequate for its task. Code kept minimal (ponytail rules).

### Assumptions (user may correct)
- A1. Fantasy playoffs: last 3 weeks of the NBA regular season, 6 teams, single elimination, top-2 seeds get byes.
- A2. "Opponents play optimally" = opponents draft by the same projections and marginal-value logic as me, with small noise. A second scenario ("opponents follow ESPN ADP") is reported as a reality check.
- A3. ESPN daily lineups: each day the best available players fill the starting slots. The simulator approximates this as "top-N expected scorers who play that day" and ignores position eligibility for daily lineups (eligibility is enforced at draft time).
- A4. A roster may hold at most `total slots − IR slots` (=12) healthy players; the remaining picks must be IR-eligible (ESPN injury status OUT or INJURY_RESERVE at draft time).
- A5. Matchup weeks are Monday–Sunday calendar weeks from the first game date.

## 2. Data sources (verified reachable 2026-09-27, no key)

| Need | Endpoint | Notes |
|---|---|---|
| Per-game stat lines, 2021-22 → 2025-26 | `https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{2022..2026}/players?scoringPeriodId=0&view=kona_player_info` | One call per season returns all players (~80k game rows). Game rows = `stats[]` entries with `statSplitTypeId==5`, `statSourceId==0`. Includes EJ, FF, TF, TD, QD, TW. |
| Positions, ADP, injury status, ESPN projections for 2026-27 | same endpoint, `seasons/2027` | `eligibleSlots`, `ownership.averageDraftPosition`, `injuryStatus`; projections = `statSourceId==1, statSplitTypeId==0, seasonId==2027`. |
| 2026-27 schedule | `https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/2027?view=proTeamSchedules_wl` | `settings.proTeams[].proGamesByScoringPeriod` → game dates (ms epoch). |
| Player age | `https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id}` | `athlete.age` (current age). |

ESPN stat ids: PTS 0, BLK 1, STL 2, AST 3, OREB 4, DREB 5, EJ 7, FF 8, TF 10, TO 11, FGM 13, FGA 14, FTM 15, FTA 16, 3PM 17, 3PA 18, TD 38, QD 39, MIN 40, TW 43.
ESPN slot ids: 0 PG, 1 SG, 2 SF, 3 PF, 4 C, 5 G, 6 F, 7 SG/SF, 8 G/F, 9 PF/C, 10 F/C, 11 UTIL, 12 BE, 13 IR.

Rejected: `nba_api`/stats.nba.com (times out from this network), NBA CDN schedule (403), Basketball-Reference (scraping, rate limits; unnecessary).

## 3. Architecture

Five Python modules at the repo root, CSV files as the only storage, Streamlit as the UI.

```
fetch.py  → data/games_{season}.csv, data/players.csv, data/espn_proj.csv, data/ages.csv, data/schedule.csv
model.py  → data/projections.csv        (reads fetch outputs)
draft.py  → pure functions: snake order, board, feasibility, team value, recommend, strategies
sim.py    → strategy simulator; writes data/strategy_slot{N}.json (uses draft.py)
app.py    → Streamlit UI on http://localhost:8501 (uses model.points, draft.*, sim.run)
settings.json — league defaults (teams, rounds, slots, scoring, playoffs, seasons)
tests/    — one test file per logic module
```

### 3.1 Projection model (`model.py`)
- **Predict stats, not points.** Targets are per-game PTS, FGM, FGA, FTM, FTA, 3PM, 3PA, OREB, DREB, AST, STL, BLK, TO, MIN plus games played (GP). Fantasy points are computed at runtime by `points(df, scoring)`, so editing weights never needs retraining.
- **Regression:** `StandardScaler + Ridge(alpha=10)`, multi-output. Features: each stat's per-game value in the previous 3 seasons, GP in each of those seasons, and age in the target season. Missing older seasons copy the nearest available one; a missing season's GP is 0.
- Two fits: the stats model uses players who appeared in the target season, and the GP model uses every player from the prior 3 seasons (GP = 0 if absent, which captures injury and retirement risk).
- **Rare or team-driven events** (EJ, FF, TF, TD, QD, TW): a per-game rate weighted 3/2/1 over the last 3 seasons, shrunk toward the league mean with k = 40 weighted games.
- **Injury adjustment at draft time:** GP × {OUT 0.75, INJURY_RESERVE 0.5, SUSPENSION 0.9}. This is a calibration knob.
- **Players with no NBA history** (rookies) use ESPN's own 2026-27 projection.
- **Variance:** `cv` = std/mean of that player's fantasy points per game over the last 2 seasons (minimum 15 games, otherwise the league median cv).
- **Backtest:** train on targets ≤ 2025, predict 2025-26, and report the mean absolute error (MAE) of fantasy points per game against a naive "last season" baseline. Acceptance: the model's MAE is below the baseline's.

### 3.2 Draft engine (`draft.py`)
- **Snake order:** team index for every overall pick.
- **Board:** the projections plus `fp_pg` (current scoring), `avail = gp/82`, `sd = cv·fp_pg`, `season_fp = fp_pg·gp`.
- **Team value:** a season total of daily lineups. Each day, the top `n_start` values of `fp_pg·avail` among roster players whose team plays that day. Vectorized with numpy over candidates.
- **Feasibility:**
  - A candidate is legal only if (a) the healthy-player cap (A4) still holds, and (b) the remaining picks can still fill every starter slot. (b) is checked with a bipartite matching of players to starter slots.
- **Recommend:**
  - Takes the top 60 available players by `season_fp` that pass feasibility.
  - Computes `marginal` = team value with the player minus team value without, plus `marginal_po` (the same, counting playoff-week days only).
  - Scores them with the chosen strategy and returns a sorted table.
  - It also reports `survive`: the probability the player is still available at my next pick. This is estimated by 200 noisy simulations of the opponents' intervening picks.
- **Strategies** (score functions over the candidate table):

  | Strategy | Score |
  |---|---|
  | `greedy` | `marginal` |
  | `durable` | `marginal·sqrt(gp/82)` |
  | `low_variance` | `marginal − 2·sd·sqrt(gp)` |
  | `playoffs` | `marginal + 2·marginal_po` |
  | `ir_early` | `marginal × 1.3` if IR-eligible and an IR spot is open, else `marginal` |

### 3.3 Strategy simulator (`sim.py`)
- **Draft simulation:** a full snake draft. Opponents use `greedy`, with each candidate's score multiplied by lognormal(0, 0.1) noise, or they take the lowest ADP with the same noise. I use the strategy under test.
- **Season simulation:**
  - Each player-game is played with probability `avail`, and its fantasy points are drawn from Normal(`fp_pg`, `sd`).
  - The daily lineup is the top `n_start` by `fp_pg` among players who play that day. Points are summed by week.
  - The regular season is a round-robin via the circle method, with wins and ties at 0.5 and the tiebreak on points scored.
  - The playoffs are a seeded single-elimination bracket with byes, played over the last `playoff_weeks` weeks.
- **Metrics for my team**, per strategy × opponent model: P(champion), P(playoffs), P(bottom 3), and mean wins.
- **Objective:** `P(champion) − P(bottom 3)`, averaged over both opponent models. The best strategy for a slot is saved to `data/strategy_slot{N}.json` and pre-selected in the UI.

### 3.4 UI (`app.py`, Streamlit)
- **Sidebar:**
  - Team count (dropdown, 8–20), my slot (1..teams), rounds.
  - A number input per roster slot, and scoring weights in an expander.
  - A "Save settings" button that writes `settings.json`.
  - A strategy dropdown and a "Run strategy simulation" button.
- **Main area:**
  - Header showing whose pick it is (snake order).
  - A searchable player dropdown that includes a "player not listed" option, a "Drafted" button and an "Undo" button.
  - The recommendation table (top 10), my roster, and the pick log.
- **State:** the pick list is saved to `data/draft.json` after every change, so a refresh or crash loses nothing.

## 4. Error handling that must exist
- **Fetch:** retries with backoff. Writes each CSV to a temporary file first, then renames it, so a failed fetch leaves the previous data intact. Clear error messages name the failing URL.
- **Parsers:** tolerate missing `stats`, `ownership`, `eligibleSlots` or `injuryStatus` keys.
- **Picks:** a "player not listed" pick consumes the pick slot without breaking the draft order. Undo removes the last pick.
- **Settings:** the app rejects roster or playoff settings that are impossible, e.g. more playoff rounds than playoff weeks or more playoff teams than teams.

## 5. Agents

Each agent gets exactly one task document in `docs/agents/`.

| Doc | Agent | Model | Depends on |
|---|---|---|---|
| 01-data.md | Data fetcher | Sonnet | — |
| 02-model.md | Projection model | Sonnet | 01 |
| 03-draft.md | Draft engine | Sonnet | 02 (`model.points`) |
| 04-sim.md | Strategy simulator | Opus | 03 |
| 05-ui.md | Streamlit app | Haiku | 03 (04 optional at runtime) |
| 06-review.md | Integration + over-engineering review | Haiku | all |

Every agent invokes the `ponytail` skill before coding. The review agent runs `ponytail-review`.

## 6. Non-goals (v1)
In-season tools (start/sit, waivers, live box scores), auction drafts, keepers, automatic pick sync from ESPN, a database, user accounts, and deployment beyond localhost.

## 7. Success criteria
1. `python fetch.py && python model.py` completes from a clean checkout. The backtest MAE beats the last-season baseline.
2. `streamlit run app.py` shows recommendations within 2 s of each entered pick for a 14-team league.
3. `python sim.py --slot N` prints a strategy table and saves the best strategy.
4. `pytest` passes.
5. Changing team count, slots or scoring in the sidebar changes the recommendations without re-running `model.py`.
