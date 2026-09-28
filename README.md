# NBA Fantasy Draft Assistant

A local draft-day tool for an ESPN head-to-head points league. It projects every player's 2026-27 season with a regression model built on free ESPN data. It recommends the best pick given your draft slot and every pick made so far. It also simulates thousands of drafts and seasons to choose a draft strategy that beats "everyone drafts optimally".

**Status:** fully implemented. See [docs/spec.md](docs/spec.md) for technical design.

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
python sim.py --slot 7   # optional: best strategy for slot 7 (~4 min)
python sim.py --report --slot 7  # optional: that strategy vs 13 near-optimal teams -> data/report_slot7.json
python sim.py --from-draft  # season-simulate the 14 real rosters in data/draft.json -> data/report_from_draft.json
streamlit run app.py       # opens http://localhost:8501
pytest                     # tests
```

On draft day: pick your team count and slot in the sidebar. After every pick in the real draft, type the player's name and click **Drafted**. The top table shows who to take and the chance each player is still there at your next turn. 

The **Strategy** panel in the sidebar has a "Run strategy simulation" button that tests all draft strategies for your slot with Monte Carlo drafts and seasons (~4 minutes). The results are cached per slot, so you can run it once during prep before the draft begins.

## Live draft (Chrome extension)

The extension reads picks from ESPN's draft room, sends them to a local server (`server.py`, 127.0.0.1:8502), highlights the top 5 in ESPN's player list (#1 green, #2-5 amber, with `#k · NN% back`), and shows the top 10, your roster and needed positions in Chrome's side panel.

1. Load it: `chrome://extensions` → turn on **Developer mode** → **Load unpacked** → select the `extension/` folder. Pin it.
2. Start the app: double-click `Start Draft App.bat` (starts the server and Streamlit with no windows), or run `python start.pyw --install` once to start it at every Windows login (`python start.pyw --uninstall` removes it).
3. Open the ESPN draft room (with the Picks panel visible) and click the extension icon to open the side panel. It turns red with "App not running" if the server is down.

Tests: `python -m pytest -q` and `node --test` (runs `extension/parse.test.js`).

## In-season (Season tab)

Put your ESPN cookies in `espn_auth.json` (git-ignored; `{"espn_s2": "...", "SWID": "{...}"}` from a logged-in fantasy.espn.com session). The sidebar's **Sync league from ESPN** button copies the league's settings (slots, scoring, playoffs, your team) into `settings.json` and its matchup calendar into `data/matchups.csv`.

The **Season** tab loads rosters and scores from ESPN when opened (cached 10 minutes) and shows:

- **This week's matchup**: your and your opponent's points so far, projected finals and your win chance. It plays each roster's best daily lineup over the remaining days (out players score nothing, day-to-day players half) and treats the rest as a Normal distribution.
- **Pickups**: the top 15 free agents by points added to your lineup this matchup (`+Pts this matchup`) and for the rest of the season, if you add them and drop the named player (the one costing your lineup least, never an IR player). ★ = worth keeping past this matchup. Sorted by this matchup because 2 adds a day makes streaming worthwhile.
- **Refresh data**: re-runs `fetch.py` and `model.py` (a few minutes). `fetch.py` also saves this season's game logs, and `model.py` blends them into the projections (weight n / (n + 20) after n games).

## Layout

| File | Role |
|---|---|
| `fetch.py` | ESPN data → `data/*.csv` |
| `model.py` | Ridge regression projections → `data/projections.csv` |
| `draft.py` | Snake order, roster legality, marginal lineup value, strategies |
| `sim.py` | Monte Carlo drafts and head-to-head seasons, best strategy per slot |
| `season.py` | In-season pickups, drop candidate and matchup win chance |
| `app.py` | Streamlit UI (Draft and Season tabs) |
| `server.py` | Local pick server for the extension (persists `data/draft.json`) |
| `extension/` | Chrome extension (Manifest V3): pick parser, highlights, side panel |
| `start.pyw` | Silent launcher for server + app; `--install` / `--uninstall` login entry |
| `settings.json` | League defaults |
