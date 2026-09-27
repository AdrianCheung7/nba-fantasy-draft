# 01 — Data Fetcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Invoke the `ponytail` skill before writing code.

**Agent model:** `sonnet`

**Goal:** Download free ESPN data (5 seasons of per-game stat lines, 2026-27 players/ADP/injuries/positions/projections, player ages, 2026-27 schedule) into CSV files.

**Architecture:** One module `fetch.py` with pure `parse_*` functions (unit-tested on inline JSON fixtures) and a thin `main()` that performs HTTP calls and atomic CSV writes. No database, no nba_api.

**Tech Stack:** Python 3.12, requests, pandas, pytest.

**Spec:** [../spec.md](../spec.md) — sections 2, 4.

## Global Constraints

- Python 3.12 on Windows; run commands from the repo root.
- Free endpoints only, no API keys. `nba_api`/stats.nba.com is NOT reachable — do not use it.
- Storage = CSV files in `data/` (gitignored). Column names exactly as in [README.md shared contracts](README.md#shared-contracts-all-agents-rely-on-these-do-not-rename).
- Scoring keys: `FGM FGMI FTM FTMI 3PM 3PMI OREB DREB AST STL BLK TO EJ FF TF TD QD PTS TW`.
- Ponytail: no classes, no config beyond `settings.json`, no new dependencies beyond `requirements.txt`.

## Review Focus

1. ESPN omits keys (`stats`, `ownership`, `eligibleSlots`, `injuryStatus`) for some players → parser must default, not crash (Task 1 tests).
2. DNP games appear as stat entries with empty `stats` → must be dropped, else GP is inflated (Task 1 test).
3. Game times near midnight UTC → date must be the US Eastern calendar day (Task 1 schedule test).
4. ESPN request fails mid-run → previous CSVs stay intact, error names the URL (Task 2 atomic save test).
5. `averageDraftPosition` of 0/missing for undrafted players → treated as 999 (Task 1 test).

---

### Task 1: Scaffolding and parsers

**Files:**
- Create: `requirements.txt`, `settings.json`, `.gitignore`, `fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fetch.STAT_IDS: dict[str,int]`, `fetch.SLOT_NAMES: dict[int,str]`, `parse_games(players: list[dict], season: int) -> DataFrame`, `parse_players(players: list[dict], ages: dict[int, float|None]) -> DataFrame`, `parse_proj(players: list[dict], season: int) -> DataFrame`, `parse_schedule(season_json: dict) -> DataFrame`, `save(df, path: Path) -> None`. `settings.json` (read by every other module).

- [ ] **Step 1: Create project files**

`requirements.txt`:
```
requests
pandas
numpy
scikit-learn
streamlit
pytest
```

`.gitignore`:
```
data/
__pycache__/
.pytest_cache/
*.tmp
```

`settings.json`:
```json
{
  "season": 2027,
  "history_seasons": [2022, 2023, 2024, 2025, 2026],
  "teams": 14,
  "my_slot": 1,
  "rounds": 14,
  "strategy": "greedy",
  "slots": {"G": 2, "F": 1, "SG/SF": 1, "PF/C": 1, "UTIL": 1, "BE": 6, "IR": 2},
  "scoring": {"FGM": 0.5, "FGMI": -0.5, "FTM": 1, "FTMI": -1, "3PM": 2, "3PMI": -1,
              "OREB": 1.5, "DREB": 1, "AST": 1, "STL": 2, "BLK": 2, "TO": -1,
              "EJ": -100, "FF": -2, "TF": -3, "TD": 10, "QD": 100, "PTS": 1, "TW": 1},
  "playoff_teams": 6,
  "playoff_weeks": 3
}
```

Run: `pip install -r requirements.txt`

- [ ] **Step 2: Write the failing tests** — `tests/test_fetch.py`

```python
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from fetch import parse_games, parse_players, parse_proj, parse_schedule, save

GAME = {"statSplitTypeId": 5, "statSourceId": 0, "seasonId": 2026, "scoringPeriodId": 7,
        "stats": {"0": 30.0, "4": 3.0, "7": 1.0, "13": 11.0, "14": 20.0, "40": 36.0, "43": 1.0}}
DNP = {**GAME, "scoringPeriodId": 8, "stats": {}}
PROJ = {"statSplitTypeId": 0, "statSourceId": 1, "seasonId": 2027, "stats": {"0": 1640.0, "42": 82.0}}
PLAYER = {"id": 1, "fullName": "Test Guy", "active": True, "proTeamId": 5,
          "eligibleSlots": [1, 5, 7, 11, 12, 13], "injuryStatus": "OUT",
          "ownership": {"averageDraftPosition": 12.5}, "stats": [GAME, DNP, PROJ]}


def test_parse_games_keeps_played_games_only():
    df = parse_games([PLAYER, {"id": 2}], 2026)
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.player_id, r.period, r.PTS, r.OREB, r.EJ, r.FGA, r.TW, r.BLK) == (1, 7, 30.0, 3.0, 1.0, 20.0, 1.0, 0.0)


def test_parse_players_positions_and_defaults():
    players = [PLAYER, {"id": 2, "active": True}, {"id": 3, "active": True, "eligibleSlots": [4, 11],
                                                   "ownership": {"averageDraftPosition": 0.0}}]
    df = parse_players(players, {1: 27})
    assert list(df.player_id) == [1, 3]
    a, b = df.iloc[0], df.iloc[1]
    assert a.positions == "SG|G|SG/SF|UTIL|BE|IR" and a.adp == 12.5 and a.injury_status == "OUT" and a.age == 27
    assert b.positions == "C|UTIL" and b.adp == 999.0 and b.injury_status == "ACTIVE" and b.team_id == 0


def test_parse_proj_per_game():
    r = parse_proj([PLAYER, {"id": 9}], 2027).iloc[0]
    assert (r.player_id, r.gp, r.PTS) == (1, 82.0, 20.0)


def test_parse_schedule_eastern_dates_and_weeks():
    t = lambda d: int(pd.Timestamp(d, tz="America/New_York").timestamp() * 1000)
    js = {"settings": {"proTeams": [
        {"id": 1, "proGamesByScoringPeriod": {"1": [{"date": t("2026-10-19 19:30")}],
                                               "7": [{"date": t("2026-10-25 19:30")}]}},
        {"id": 2, "proGamesByScoringPeriod": {"8": [{"date": t("2026-10-26 22:30")}]}},
        {"id": 0, "proGamesByScoringPeriod": {"1": [{"date": t("2026-10-19 19:30")}]}}]}}
    df = parse_schedule(js)
    assert list(df.columns) == ["team_id", "date", "week"]
    assert df.values.tolist() == [[1, "2026-10-19", 1], [1, "2026-10-25", 1], [2, "2026-10-26", 2]]


def test_save_is_atomic(tmp_path):
    p = tmp_path / "x.csv"
    save(pd.DataFrame({"a": [1]}), p)
    assert p.read_text().splitlines() == ["a", "1"]
    assert not (tmp_path / "x.tmp").exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fetch'`

- [ ] **Step 4: Implement `fetch.py`**

```python
"""Pull free ESPN data into data/*.csv.  Run: python fetch.py [--refresh]"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FANTASY = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}"
PLAYERS = FANTASY + "/players?scoringPeriodId=0&view=kona_player_info"
SCHEDULE = FANTASY + "?view=proTeamSchedules_wl"
ATHLETE = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}"

# ESPN fantasy stat ids, verified 2026-09-27 against a real 2025-26 stat line.
STAT_IDS = {"PTS": 0, "BLK": 1, "STL": 2, "AST": 3, "OREB": 4, "DREB": 5, "EJ": 7, "FF": 8,
            "TF": 10, "TO": 11, "FGM": 13, "FGA": 14, "FTM": 15, "FTA": 16, "3PM": 17,
            "3PA": 18, "TD": 38, "QD": 39, "MIN": 40, "TW": 43}
SLOT_NAMES = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "G", 6: "F", 7: "SG/SF",
              8: "G/F", 9: "PF/C", 10: "F/C", 11: "UTIL", 12: "BE", 13: "IR"}


def get(url, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if i == tries - 1:
                raise RuntimeError(f"ESPN request failed after {tries} tries: {url} ({e})") from e
            time.sleep(2 ** i)


def save(df, path):
    tmp = path.with_suffix(".tmp")  # write-then-rename: a failed run never clobbers good data
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def stat_row(stats):
    return {k: float(stats.get(str(v), 0.0)) for k, v in STAT_IDS.items()}


def entries(p, split, source, season):
    for s in p.get("stats") or []:
        if (s.get("statSplitTypeId"), s.get("statSourceId"), s.get("seasonId")) == (split, source, season):
            yield s, s.get("stats") or {}


def parse_games(players, season):
    rows = [{"season": season, "player_id": p["id"], "period": s.get("scoringPeriodId"), **stat_row(st)}
            for p in players for s, st in entries(p, 5, 0, season) if st.get("40", 0) > 0]
    return pd.DataFrame(rows, columns=["season", "player_id", "period", *STAT_IDS])


def parse_proj(players, season):
    rows = [{"player_id": p["id"], "gp": st["42"], **{k: v / st["42"] for k, v in stat_row(st).items()}}
            for p in players for _, st in entries(p, 0, 1, season) if st.get("42", 0) > 0]
    return pd.DataFrame(rows, columns=["player_id", "gp", *STAT_IDS])


def parse_players(players, ages):
    rows = [{"player_id": p["id"], "name": p.get("fullName", str(p["id"])), "team_id": p.get("proTeamId", 0),
             "positions": "|".join(SLOT_NAMES[i] for i in p["eligibleSlots"] if i in SLOT_NAMES),
             "adp": (p.get("ownership") or {}).get("averageDraftPosition") or 999.0,
             "injury_status": p.get("injuryStatus") or "ACTIVE",
             "age": ages.get(p["id"])}
            for p in players if p.get("active") and p.get("eligibleSlots")]
    return pd.DataFrame(rows, columns=["player_id", "name", "team_id", "positions", "adp", "injury_status", "age"])


def parse_schedule(season_json):
    rows = [{"team_id": t["id"], "date": g["date"]}
            for t in season_json["settings"]["proTeams"] if t["id"] != 0
            for games in (t.get("proGamesByScoringPeriod") or {}).values() for g in games]
    df = pd.DataFrame(rows)
    df["date"] = (pd.to_datetime(df.date, unit="ms", utc=True).dt.tz_convert("America/New_York")
                  .dt.tz_localize(None).dt.normalize())
    df = df.drop_duplicates()
    first_monday = df.date.min() - pd.Timedelta(days=df.date.min().weekday())
    df["week"] = (df.date - first_monday).dt.days // 7 + 1
    df["date"] = df.date.dt.strftime("%Y-%m-%d")
    return df.sort_values(["date", "team_id"]).reset_index(drop=True)[["team_id", "date", "week"]]


def fetch_ages(ids, cache):
    old = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["player_id", "age"])
    ages = dict(zip(old.player_id.astype(int), old.age))
    for pid in ids:
        if pid not in ages:
            try:
                ages[pid] = get(ATHLETE.format(pid=pid))["athlete"].get("age")
            except (RuntimeError, KeyError) as e:
                print(e)  # ponytail: missing age -> model fills the median
                ages[pid] = None
            time.sleep(0.2)
    save(pd.DataFrame(list(ages.items()), columns=["player_id", "age"]), cache)
    return ages


def main(refresh=False):
    s = json.loads((ROOT / "settings.json").read_text())
    DATA.mkdir(exist_ok=True)
    for season in s["history_seasons"]:
        out = DATA / f"games_{season}.csv"
        if refresh or not out.exists():
            print(f"fetching {season} game logs ...")
            save(parse_games(get(PLAYERS.format(season=season)), season), out)
    print("fetching 2026-27 players ...")
    cur = get(PLAYERS.format(season=s["season"]))
    save(parse_proj(cur, s["season"]), DATA / "espn_proj.csv")
    ids = [p["id"] for p in cur if p.get("active") and p.get("eligibleSlots") and p.get("proTeamId", 0) != 0]
    ages = fetch_ages(ids, DATA / "ages.csv")
    save(parse_players(cur, ages), DATA / "players.csv")
    save(parse_schedule(get(SCHEDULE.format(season=s["season"]))), DATA / "schedule.csv")
    print("done")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_fetch.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt settings.json .gitignore fetch.py tests/test_fetch.py
git commit -m "feat: ESPN data fetcher with tested parsers"
```

### Task 2: Live fetch and sanity check

**Files:**
- Modify: none expected (fix `fetch.py` only if the live data contradicts a parser assumption; add a test reproducing it first).

**Interfaces:**
- Produces: `data/games_2022.csv … data/games_2026.csv`, `data/players.csv`, `data/espn_proj.csv`, `data/ages.csv`, `data/schedule.csv`.

- [ ] **Step 1: Run the fetch** (first run takes ~5–10 min: 6 large player calls + ~500 age calls)

Run: `python fetch.py`
Expected: `fetching 2022 game logs ...` … `done`, no traceback.

- [ ] **Step 2: Sanity check**

Run:
```bash
python -c "import pandas as pd; [print(s, len(pd.read_csv(f'data/games_{s}.csv'))) for s in range(2022,2027)]; p=pd.read_csv('data/players.csv'); print('players', len(p), 'with adp', (p.adp<999).sum(), 'no age', p.age.isna().sum()); print('proj', len(pd.read_csv('data/espn_proj.csv'))); sc=pd.read_csv('data/schedule.csv'); print('teams', sc.team_id.nunique(), 'games/team', sc.groupby('team_id').size().agg(['min','max']).tolist(), 'weeks', sc.week.max())"
```
Expected: each season ≥ 20,000 game rows; players ≥ 400 with ≥ 150 having ADP; teams = 30, games/team = [82, 82] (± a couple if ESPN lists postponed games); weeks ≈ 24–26.
If `proj` is 0: ESPN has not published 2026-27 projections — record this in your report (model falls back to "no rookies"); do not invent data.
If `no age` > 50: report it (model fills the median).

- [ ] **Step 3: Spot-check one known player**

Run:
```bash
python -c "import pandas as pd; p=pd.read_csv('data/players.csv'); print(p[p.name.str.contains('Jokic|Draymond', regex=True)].to_string()); g=pd.read_csv('data/games_2026.csv'); print(g.groupby('player_id')[['EJ','TF','FF','TD']].sum().sort_values('TF').tail(5))"
```
Expected: Jokic has `C|…|UTIL` positions and a low ADP; the TF leaders are plausible technical-foul players (single-digit to ~15 per season).

- [ ] **Step 4: Commit** (data is gitignored; commit only if `fetch.py` changed)

```bash
git add fetch.py tests/test_fetch.py
git commit -m "fix: align fetch parsers with live ESPN data"
```

Report back: row counts from Step 2, output of Step 3, and any deviation.
