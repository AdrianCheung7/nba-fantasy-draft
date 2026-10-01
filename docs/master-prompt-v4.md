# Master prompt v4: league changed to 16-team H2H categories (do not run until approved)

Paste everything below the line into a fresh Claude Code session opened in this repo.

---

You are updating an existing, working NBA fantasy draft assistant in this repo, branch `v2`. The ESPN league changed
its settings on draft day. The live draft is **today, 2026-09-30, 20:30 local**, with 60 s per pick.

Read these first:
- `docs/spec.md`, `docs/spec-v2.md`, `docs/research-v3.md`
- `espn.py`, `fetch.py`, `model.py`, `draft.py`, `sim.py`, `season.py`, `server.py`, `app.py`, `extension/`
- the tests

Rules:
- Ponytail style: smallest correct diff, no new dependencies, reuse existing helpers.
- TDD: a failing test first.
- Commit and push to `origin v2` after every step, with messages ending in the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never read, print or commit `espn_auth.json`.
- Use subagents on cost-appropriate models for independent parts: Sonnet for code, Opus for the final review.
- Before anything you are unsure about, ask the user.

## What ESPN reports now vs. what the app assumes (verified 2026-09-30 via the league API)

| Setting | App assumes | ESPN now |
|---|---|---|
| Scoring type | H2H **points** (FGM .5, FGMI −.5, …, EJ −100, QD 100, TW 1) | **H2H_CATEGORY** (each category is a win/loss; tie rule NONE) |
| Categories | n/a | 12, each weight 1: PTS(0) BLK(1) STL(2) AST(3) OREB(4) DREB(5) **PF(9, lower is better)** **TO(11, lower is better)** **FG%(19)** **FT%(20)** **3P%(21)** **TW(43, team wins)** |
| Teams | 14 | **16** (team ids 1–16) |
| My draft slot | 9 | **8** (team 13 "Raj"; pickOrder `[12,5,9,6,2,7,8,13,14,15,16,3,4,1,10,11]`) |
| Lineup slots | G2 F1 SG/SF1 PF/C1 UTIL1 BE6 IR2 | G2 F1 **SG/SF2** PF/C1 UTIL1 **BE4** IR2 (7 starters) |
| Draft rounds | 12 | **11** (= 13 slots − 2 IR) |
| Draft time | 20:00 | **20:30**, 60 s per pick |
| Adds | `adds_per_day` 2 | `matchupAcquisitionLimit` 0.5714 per scoring period ≈ **4 per matchup week**; waivers **48 h** (was 24) |
| Playoffs | 4 teams, 2-period rounds | unchanged; matchup calendar still ESPN's 1-day placeholder |
| ESPN rank list opponents follow | POINTS rank first | in a category league the room shows **STANDARD** (category) rank |

**Silent-corruption bug.** `espn.settings_from_league` reads `scoringItems[].points` as point weights. On this league it
would write PTS = BLK = STL = AST = OREB = DREB = TW = +1 and TO = **+1**. It ignores `isReverseItem` and the % stats, and
`fetch.py` does not collect PF (stat 9) at all. Today's "Sync league from ESPN" button would turn the app into a wrong
points league without warning.

## P0: must be done and tested before 20:30 (in this order)

1. **ESPN sync understands categories** (`espn.py`, `settings.json`).
   - Keep `scoring` for points leagues.
   - For `scoringType` in {H2H_CATEGORY, H2H_MOST_CATEGORIES, ROTO}, write `"format": "categories"` and
     `"categories": [{"stat": "PTS", "reverse": false}, …, {"stat": "FG%", "reverse": false}, {"stat": "PF", "reverse": true}, …]`.
   - Never write fake point weights.
   - Sync must also set: teams 16, rounds 11, slots, my_slot 8, `adds_per_week` 4 (derived as
     `round(matchupAcquisitionLimit*7)` when `matchupLimitPerScoringPeriod`), and `draft_time`.
   - Add a test with a category-league fixture (trimmed, no secrets).
   - Then run the sync for real and commit settings.json.
2. **Collect PF** (`fetch.py`): add `"PF": 9` to `STAT_IDS`, add PF to the per-game parse, the ESPN projection parse and
   `model.STATS`. Then run `python fetch.py --refresh` and `python model.py`.
3. **Category value instead of fantasy points** (`model.py` / `draft.py`). This is the core change. It must be ONE
   function so everything downstream keeps working.
   - Add `cat_value(df, settings)` that returns each player's per-game value as the sum of category z-scores, computed
     over a draftable pool (top teams × rounds by a first-pass value; iterate twice).
   - Counting cats: `(x − pool mean) / pool sd`.
   - Percentage cats are volume-weighted impact, e.g. `FGM − lg_FG% × FGA`, then z-scored. 3P% uses 3PM/3PA, FT% uses
     FTM/FTA.
   - Reverse cats (TO, PF) are negated.
   - TW is the player's per-game team-win rate.
   - Wherever `points(df, scoring)` feeds `fp_pg` (`draft.board`, `model` cv, `sim`, `season`), dispatch on
     `settings["format"]`: points leagues keep `points`, category leagues use `cat_value`.
   - `team_value`, `recommend`, `advise`, `survive`, the server and the extension then work unchanged (z-scores are
     additive, which is the standard approximation).
   - `cv` and `sd` must be computed on the same scale.
   - Label the UI column "Value" (not "Pts/game") in category mode.
4. **ESPN rank** (`fetch.py`): use `STANDARD` first when `format == "categories"`, POINTS first otherwise. Refetch.
5. **Re-run** `python sim.py --slot 8` and `python sim.py --report --slot 8`.
   - The strategy comparison can keep using the summed z-value as its season score for P0. Say in the README and the
     app caption that sim odds are approximate until P1 lands.
6. **Check end to end**:
   - `pytest`, and `node --test`.
   - Restart via `Start Draft App.bat`.
   - `/state` shows 16 teams, 11 rounds, slot 8.
   - A ~30-pick ESPN practice draft (mock lobby → "Bronny's Boys" → Practice Draft, which uses the new settings) syncs
     picks, highlights rows and shows "YOUR PICK" at overall pick 8.
   - Then reload the extension.

Fallback if P0 is not done by about 19:45: tell the user to draft from ESPN's own category ranking and use the side panel
only for "% back next turn". Do not ship a half-converted valuation.

## P1: after the draft (before tip-off)

7. **Category H2H sim** (`sim.py`). `sim_season` returns weekly per-category team totals (teams × weeks × cats).
   - Ratio cats are made/attempted sums.
   - The weekly result is category wins − losses (ties count as ties); standings follow H2H_RECORD.
   - Game-level noise: draw one multiplier per player-game ~ Normal(1, cv) and apply it to all counting stats (correlated
     line), keeping the vectorized style and the existing beta-binomial availability and back-to-back logic.
   - The objective stays p_champion − p_bottom3. Re-run strategies for slot 8.
8. **Punt strategies** (`draft.STRATEGIES`): add `punt_<cat>` variants that drop one category (or the TO+PF pair) from
   `cat_value`. Let the sim decide, with the same 2-SE gate vs. greedy. This is a real edge in category leagues, and
   the user asked about punting stocks earlier.
9. **Season tab** (`season.py`, `app.py`):
   - The matchup shows each category's projected total for me vs. the opponent, and P(win) per category via a
     Normal approximation.
   - Expected categories won replaces win %.
   - Pickups rank by the gain in expected categories won this matchup and by rest-of-season z-value.
   - Respect `adds_per_week` (4) instead of `adds_per_day`.
10. **Re-calibrate** the opponent model constants (`RANK_WEIGHT`, `PICK_NOISE`) against STANDARD ADP, using the same
    method as before.
11. Run a final Opus review of the whole diff, then fix its findings.

Report: what changed per step, test counts, the slot-8 strategy table, the new top-20 by value, and anything unverified.
