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
