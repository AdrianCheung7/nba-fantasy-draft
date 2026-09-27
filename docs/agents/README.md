# Agent task documents

Spec: [../spec.md](../spec.md). Each document is a self-contained brief for one subagent. The orchestrator (main session) dispatches them in order, gives each one only its own document plus the spec, and reviews the result before starting the next.

| # | Doc | Agent | Model | Starts after | Produces |
|---|---|---|---|---|---|
| 1 | [01-data.md](01-data.md) | Data fetcher | `sonnet` | — | `requirements.txt`, `settings.json`, `.gitignore`, `fetch.py`, `tests/test_fetch.py`, `data/*.csv` |
| 2 | [02-model.md](02-model.md) | Projection model | `sonnet` | 1 | `model.py`, `tests/test_model.py`, `data/projections.csv` |
| 3 | [03-draft.md](03-draft.md) | Draft engine | `sonnet` | 2 | `draft.py`, `tests/test_draft.py` |
| 4 | [04-sim.md](04-sim.md) | Strategy simulator | `opus` | 3 | `sim.py`, `tests/test_sim.py` |
| 5 | [05-ui.md](05-ui.md) | Streamlit UI | `haiku` | 3 (can run parallel with 4) | `app.py` |
| 6 | [06-review.md](06-review.md) | Integration + ponytail review | `haiku` | 4 and 5 | fixes, `README.md` run section |

Why these models: the data and model tasks involve undocumented APIs and statistics (Sonnet). Draft feasibility and matching are fiddly (Sonnet). The simulator is the hardest reasoning: Monte Carlo drafts, season simulation and bracket logic (Opus). UI wiring and checklist review are mechanical (Haiku).

## Dispatch template (orchestrator)

```
Agent(subagent_type="general-purpose", model="<model from table>",
      description="<doc title>",
      prompt="You are implementing one part of a project. Read docs/spec.md and docs/agents/<NN-name>.md in the repo at <repo path>. "
             "Invoke the ponytail skill first. Follow the task document step by step (TDD, commit after each task). "
             "Do not edit files owned by other agents unless your document says so. Report: files changed, test output, anything that deviated from the document.")
```

## Shared contracts (all agents rely on these; do not rename)

**`data/games_{season}.csv`**: one row per player-game.
`season, player_id, period, PTS, BLK, STL, AST, OREB, DREB, EJ, FF, TF, TO, FGM, FGA, FTM, FTA, 3PM, 3PA, TD, QD, MIN, TW`

**`data/players.csv`**: active players for the 2026-27 season.
`player_id, name, team_id, positions, adp, injury_status, age`. `positions` is `|`-joined slot names, e.g. `SG|G|SG/SF|G/F|UTIL|BE|IR`.

**`data/espn_proj.csv`**: ESPN's 2026-27 per-game projections.
`player_id, gp, PTS, BLK, STL, AST, OREB, DREB, EJ, FF, TF, TO, FGM, FGA, FTM, FTA, 3PM, 3PA, TD, QD, MIN, TW`

**`data/schedule.csv`**: `team_id, date, week` (date is `YYYY-MM-DD`, week is 1-based Monday weeks).

**`data/projections.csv`**: `player_id, name, team_id, positions, adp, injury_status, age, gp, cv` plus per-game `PTS, FGM, FGA, FTM, FTA, 3PM, 3PA, OREB, DREB, AST, STL, BLK, TO, MIN, EJ, FF, TF, TD, QD, TW`.

**`model.points(df, scoring) -> pd.Series`**: fantasy points from stat columns using the `settings.json` scoring keys.

**`draft.py`** exports `snake, picks_before_next, starters, board, day_matrix, ev_matrix, team_value, matched, feasible, recommend, survive, STRATEGIES`. The exact signatures are in 03-draft.md.

**`sim.py`** exports `run(settings, slot, n_drafts, n_seasons, seed) -> pd.DataFrame`.
