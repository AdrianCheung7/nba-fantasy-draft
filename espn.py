"""Sync the private ESPN league into settings.json + data/matchups.csv.  Run: python espn.py"""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import fetch

ROOT = fetch.ROOT
DATA = fetch.DATA
LEAGUE = fetch.FANTASY + "/segments/0/leagues/{league_id}"
VIEWS = ["mSettings", "mTeam", "mRoster", "mStatus", "mMatchupScore", "mScoreboard"]
# ESPN scoring stat id -> settings.json scoring key
SCORING = {0: "PTS", 1: "BLK", 2: "STL", 3: "AST", 4: "OREB", 5: "DREB", 7: "EJ", 8: "FF", 10: "TF",
           11: "TO", 13: "FGM", 15: "FTM", 17: "3PM", 23: "FGMI", 24: "FTMI", 25: "3PMI", 38: "TD",
           39: "QD", 43: "TW"}
# category leagues: ESPN stat id -> category name (counting stats, % stats, PF, TW)
CATEGORY_TYPES = {"H2H_CATEGORY", "H2H_MOST_CATEGORIES", "ROTO"}
CATEGORIES = {0: "PTS", 1: "BLK", 2: "STL", 3: "AST", 4: "OREB", 5: "DREB", 6: "REB", 9: "PF", 11: "TO", 13: "FGM",
              15: "FTM", 17: "3PM", 19: "FG%", 20: "FT%", 21: "3P%", 37: "DD", 38: "TD", 43: "TW"}


def auth(home):
    """Cookies dict from home/espn_auth.json, or None when the file is missing."""
    p = Path(home) / "espn_auth.json"
    return json.loads(p.read_text()) if p.exists() else None


def league(views, league_id, season, cookies):
    """One league GET with a `view=` param per view."""
    url = LEAGUE.format(season=season, league_id=league_id) + "?" + "&".join(f"view={v}" for v in views)
    return fetch.get(url, cookies=cookies)


def settings_from_league(js, settings, swid=None):
    """New settings dict from ESPN. swid (cookie) picks my team; without it the old espn_team_id stays."""
    st = js["settings"]
    slots = {fetch.SLOT_NAMES[int(i)]: n for i, n in st["rosterSettings"]["lineupSlotCounts"].items()
             if n and int(i) in fetch.SLOT_NAMES}
    items = st["scoringSettings"]["scoringItems"]
    if st["scoringSettings"].get("scoringType") in CATEGORY_TYPES:
        if bad := [i["statId"] for i in items if i["statId"] not in CATEGORIES]:
            raise ValueError(f"unknown ESPN category stat id(s) {bad}: add them to espn.CATEGORIES")
        fmt = {"format": "categories",
               "categories": [{"stat": CATEGORIES[i["statId"]], "reverse": bool(i.get("isReverseItem"))} for i in items]}
    else:
        pts = {i["statId"]: float(i["points"]) for i in items}
        fmt = {"format": "points", "scoring": {k: pts.get(i, 0.0) for i, k in SCORING.items()}}
    acq, ds = st["acquisitionSettings"], st["draftSettings"]
    sch = st["scheduleSettings"]
    tid = settings.get("espn_team_id")
    if swid:
        me = swid.strip("{}").lower()
        tid = next((t["id"] for t in js["teams"] if me in [o.strip("{}").lower() for o in t["owners"]]), tid)
    order = ds["pickOrder"]
    team = next((t for t in js["teams"] if t["id"] == tid), {})
    stale = {"scoring", "categories", "adds_per_day"}  # the other format's keys, and the retired add limit
    return {**{k: v for k, v in settings.items() if k not in stale}, "league_id": js["id"], "teams": st["size"], "slots": slots,
            "rounds": sum(slots.values()) - slots.get("IR", 0),
            **fmt, "playoff_teams": sch["playoffTeamCount"], "playoff_round_periods": sch["playoffMatchupPeriodLength"],
            "playoff_weeks": math.ceil(math.log2(sch["playoffTeamCount"])) * sch["playoffMatchupPeriodLength"],
            "espn_team_id": tid, "my_team_name": team.get("name", settings.get("my_team_name")),
            "my_slot": order.index(tid) + 1 if tid in order else settings.get("my_slot", 1),
            "adds_per_week": round(acq["matchupAcquisitionLimit"] * (7 if acq["matchupLimitPerScoringPeriod"] else 1)),
            "pick_seconds": ds["timePerSelection"],
            "draft_time": datetime.fromtimestamp(ds["date"] / 1000).isoformat(timespec="minutes")
            if ds.get("date") else settings.get("draft_time")}


def matchups(js, schedule):
    """date, matchup, playoff per scoring day; scoring period 1 = first date in schedule."""
    sch = js["settings"]["scheduleSettings"]
    start = pd.Timestamp(schedule.date.min())
    rows = [{"date": (start + pd.Timedelta(days=sp - 1)).strftime("%Y-%m-%d"), "matchup": int(m),
             "playoff": int(m) > sch["matchupPeriodCount"]}
            for m, sps in sch["matchupPeriods"].items() for sp in sps]
    return pd.DataFrame(rows, columns=["date", "matchup", "playoff"]).sort_values("date").reset_index(drop=True)


def is_real_calendar(m, js):
    """False for ESPN's preseason placeholder (1 day per matchup)."""
    return (~m.playoff).sum() >= 7 * js["settings"]["scheduleSettings"]["matchupPeriodCount"]


def rosters(js):
    rows = [{"team_id": t["id"], "team_name": t["name"], "player_id": e["playerId"],
             "lineup_slot": fetch.SLOT_NAMES.get(e["lineupSlotId"], str(e["lineupSlotId"]))}
            for t in js["teams"] for e in t["roster"]["entries"]]
    return pd.DataFrame(rows, columns=["team_id", "team_name", "player_id", "lineup_slot"])


def scores(js, matchup, me):
    """My and my opponent's points so far in matchup period `matchup`; None if I have no game."""
    for g in js["schedule"]:
        sides = [s for s in (g.get("home"), g.get("away")) if s]
        if g["matchupPeriodId"] == matchup and any(s["teamId"] == me for s in sides) and len(sides) == 2:
            mine, opp = sorted(sides, key=lambda s: s["teamId"] != me)
            return {"me": mine["totalPoints"], "opp": opp["totalPoints"], "opp_id": opp["teamId"]}


def main():
    path = ROOT / "settings.json"
    old = json.loads(path.read_text())
    cookies = auth(ROOT)
    if not cookies or not old.get("league_id"):
        sys.exit("need espn_auth.json ({espn_s2, SWID}) and league_id in settings.json")
    js = league(VIEWS, old["league_id"], old["season"], cookies)
    new = settings_from_league(js, old, cookies.get("SWID"))
    tmp = path.with_suffix(".tmp")  # write-then-rename, like fetch.save
    tmp.write_text(json.dumps(new, indent=2))
    tmp.replace(path)
    m = matchups(js, pd.read_csv(DATA / "schedule.csv"))
    real = is_real_calendar(m, js)
    if real:
        fetch.save(m, DATA / "matchups.csv")
    else:  # ESPN's preseason placeholder is 1 day per matchup; sim falls back to calendar weeks
        (DATA / "matchups.csv").unlink(missing_ok=True)
    print(f"synced {new['format']} league, {new['my_team_name']} (team {new['espn_team_id']}, slot {new['my_slot']}/{new['teams']}): "
          f"{new['rounds']} rounds, {new['playoff_teams']} playoff teams, {new['playoff_weeks']} playoff periods, "
          f"{m.matchup.nunique()} matchup periods over {len(m)} days"
          + ("" if real else " (placeholder calendar, matchups.csv not written)"))


if __name__ == "__main__":
    main()
