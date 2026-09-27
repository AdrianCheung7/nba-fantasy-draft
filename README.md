# NBA Fantasy Draft Assistant

A local draft-day tool for an ESPN head-to-head points league. It projects every player's 2026-27 season with a regression model built on free ESPN data. It recommends the best pick given your draft slot and every pick made so far. It also simulates thousands of drafts and seasons to choose a draft strategy that beats "everyone drafts optimally".

**Status:** planning complete, implementation not started. See [docs/spec.md](docs/spec.md) and the agent task documents in [docs/agents/](docs/agents/README.md).

## League defaults (editable in the app)

- **Format:** 14 teams, snake draft, 14 rounds, weekly head-to-head points.
- **Roster:** G ×2, F, SG/SF, PF/C, UTIL, 6 bench, 2 IR.
- **Scoring:**

  | Stat | Points | Stat | Points |
  |---|---|---|---|
  | FGM | 0.5 | FGMI | −0.5 |
  | FTM | 1 | FTMI | −1 |
  | 3PM | 2 | 3PMI | −1 |
  | OREB | 1.5 | DREB | 1 |
  | AST | 1 | STL | 2 |
  | BLK | 2 | TO | −1 |
  | EJ | −100 | FF | −2 |
  | TF | −3 | TD | 10 |
  | QD | 100 | PTS | 1 |
  | TW | 1 | | |

## Run it (Windows, Python 3.12)

```bash
pip install -r requirements.txt
python fetch.py            # ~10 min the first time: free ESPN data into data/
python model.py --backtest # model accuracy vs. "repeat last season"
python model.py            # writes data/projections.csv
python sim.py --slot 7     # optional: best strategy for draft slot 7 (a few minutes)
streamlit run app.py       # opens http://localhost:8501
pytest                     # tests
```

On draft day: pick your team count and slot in the sidebar. After every pick in the real draft, type the player's name and click **Drafted**. The top table shows who to take and the chance each player is still there at your next turn.

## Layout

| File | Role |
|---|---|
| `fetch.py` | ESPN data → `data/*.csv` |
| `model.py` | Ridge regression projections → `data/projections.csv` |
| `draft.py` | Snake order, roster legality, marginal lineup value, strategies |
| `sim.py` | Monte Carlo drafts and head-to-head seasons, best strategy per slot |
| `app.py` | Streamlit UI |
| `settings.json` | League defaults |
