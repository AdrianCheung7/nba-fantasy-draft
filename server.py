"""Local pick server for the Chrome extension.  Run: python server.py  (127.0.0.1:8502 only)"""
import json
import os
import re
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

import draft

PORT = 8502
# ESPN proTeamId -> abbreviation (matches the team_id column of players.csv)
TEAMS = dict(enumerate("ATL BOS NO CHI CLE DAL DEN DET GS HOU IND LAC LAL MIA MIL MIN BKN NY ORL PHI PHX POR SAC SA OKC "
                       "UTAH WSH TOR MEM CHA".split(), 1))
ALIAS = {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH", "PHO": "PHX", "BK": "BKN",
         "CHO": "CHA"}
CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Private-Network": "true"}


def norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", re.sub(r"[^a-z0-9 ]", "", s).strip())


def abbr(t):
    t = str(t).upper()
    return ALIAS.get(t, t)


class App:
    def __init__(self, home):
        self.settings_file = Path(home) / "settings.json"
        self.data = Path(home) / "data"
        self.lock = threading.RLock()
        self.mtime = None
        self.last = {"status": "no picks yet"}
        self.slot = None  # my slot as read off the Picks panel (my team's pick), else settings
        self.reset_at = None  # when the last new draft was detected (the side panel shows a notice for a few seconds)
        self.last_url = None  # draft page of the stored picks (set by the replay of last_picks.json)

    def load(self):
        """Board + expected-value matrix; rebuilt only when settings.json changes."""
        with self.lock:
            return self._load()

    def _load(self):
        m = self.settings_file.stat().st_mtime
        if m != self.mtime:
            s = json.loads(self.settings_file.read_text())
            b = draft.board(pd.read_csv(self.data / "projections.csv"), s)
            G, _ = draft.day_matrix(b, pd.read_csv(self.data / "schedule.csv"))
            self.s, self.b, self.ev, self.mtime = s, b, draft.ev_matrix(b, G), m
            self.by_id = dict(zip(b.player_id, b.index))
            self.by_name = {}
            for i, n in zip(b.index, b["name"]):
                self.by_name.setdefault(norm(n), []).append(i)
            if "top" not in self.last:
                self.last = {"status": "no picks yet", "teams": s["teams"]}
            try:  # settings changed or server restarted: redo the last result
                body = json.loads((self.data / "last_picks.json").read_text())
                self.slot = body.get("slot") if self.slot is None else self.slot
                self.compute(body)
            except (OSError, ValueError):
                pass
        return self.s

    def match(self, p):
        """Board index by ESPN id, else name + NBA team, else name only, else None."""
        if p.get("espn_id") in self.by_id:
            return self.by_id[p["espn_id"]]
        cand = self.by_name.get(norm(p.get("name", "")), [])
        same = [i for i in cand if TEAMS.get(int(self.b.team_id.iat[i])) == abbr(p.get("nba_team", ""))]
        return (same or cand or [None])[0]

    def write(self, name, obj):
        f = self.data / name
        tmp = f.with_suffix(".tmp")  # same write-then-rename as app.py
        tmp.write_text(json.dumps(obj))
        tmp.replace(f)

    def known(self):
        """Picks recorded so far (overall -> board index or None), read back from draft.json (app.py may edit it)."""
        try:
            ids = json.loads((self.data / "draft.json").read_text())
        except (OSError, ValueError):
            return {}
        return {o: self.by_id.get(x) for o, x in enumerate(ids, 1)}

    def post(self, body):
        with self.lock:
            self.load()
            return self.compute(body)

    def compute(self, body):
        s, teams = self.s, self.s["teams"]
        new, unmatched, mine = {}, [], None
        for p in body["picks"]:
            # a "pick" above the team count can only be the overall number; don't trust the client's team count
            o = p["pick"] if p["pick"] > teams else (p["round"] - 1) * teams + p["pick"]
            new[o] = self.match(p)
            if new[o] is None:
                unmatched.append(p["name"])
            if mine is None and p.get("fantasy_team") == s.get("my_team_name"):
                r, k = divmod(o - 1, teams)  # snake: odd rounds go 1..N, even rounds N..1
                mine = k + 1 if r % 2 == 0 else teams - k
        known = self.known()
        # a new draft if: another draft page (practice drafts each get their own leagueId), pick 1 changed,
        # or a known player shows up at a different pick (SGA/Jokic go 1-2 in every practice draft)
        league = re.search(r"leagueId=(\d+)", body.get("url") or "")
        old = re.search(r"leagueId=(\d+)", self.last_url or "")
        # an empty list never erases picks of the SAME league (panel just not rendered), a different league resets at once
        reset = bool((league and old and league[1] != old[1])
                     or (known.get(1) is not None and 1 in new and new[1] != known[1])
                     or any(v is not None and known.get(o) != v and v in known.values() for o, v in new.items()))
        if reset:
            known, self.slot, self.reset_at = {}, None, time.time()
        self.last_url = body.get("url") or self.last_url
        merged = {**known, **{o: v for o, v in new.items() if v is not None or o not in known}}  # never forget a pick
        self.slot = mine or self.slot
        slot = self.slot or s["my_slot"]
        n = max(merged, default=0)
        picks = [{"overall": o, "player": merged.get(o)} for o in range(1, n + 1)]
        self.write("draft.json", [None if x["player"] is None else int(self.b.player_id.iat[x["player"]]) for x in picks])
        self.write("last_picks.json", {**body, "slot": self.slot})
        r = draft.advise(self.b, self.ev, picks, {**s, "my_slot": slot}, s.get("strategy", "greedy"))
        top = r["top"]
        out = {**r, "top": json.loads(top.to_json(orient="records")), "fallback": bool(top.attrs.get("fallback")),
               "mapped": len(body["picks"]) - len(unmatched), "unmatched": unmatched, "n_picks": n,
               "teams": teams, "me": slot - 1, "slot": slot}
        if self.reset_at:
            out["reset_at"] = self.reset_at
        if slot != s["my_slot"]:
            out["warning"] = f"Your draft slot on this page is {slot}, settings say {s['my_slot']} \u2014 using {slot}"
        self.last = json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        return {**self.last, "reset": True} if reset else self.last


def make_server(home, port=PORT):
    app = App(home)
    try:
        app.load()  # redo the last result right away
    except (OSError, ValueError, KeyError):
        pass  # data files missing before fetch.py has run

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, obj=None):
            body = b"" if obj is None else json.dumps(obj).encode()
            self.send_response(code)
            for k, v in CORS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send(204)

        def foreign(self):
            # only the extension (or local tools that send no Origin) may read or post picks, not web pages
            o = self.headers.get("Origin")
            return o is not None and not o.startswith("chrome-extension://")

        def do_GET(self):
            if self.foreign():
                return self.send(403, {"error": "forbidden origin"})
            if self.path != "/state":
                return self.send(404, {"error": "not found"})
            try:
                app.load()  # picks up settings changes (teams) before any pick is posted
            except (OSError, ValueError, KeyError):
                pass
            self.send(200, app.last)

        def do_POST(self):
            if self.foreign():
                return self.send(403, {"error": "forbidden origin"})
            if self.path != "/picks":
                return self.send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n)) if 0 < n < 1 << 20 else None
                ok = isinstance(body, dict) and isinstance(body.get("picks"), list) and all(
                    isinstance(p, dict) and type(p.get("round")) is int and type(p.get("pick")) is int
                    and p["round"] >= 1 and p["pick"] >= 1 for p in body["picks"])
            except ValueError:
                ok = False
            if not ok:
                return self.send(400, {"error": "expected {picks: [{round, pick, ...}]}"})
            try:
                self.send(200, app.post(body))
            except Exception as e:  # e.g. data files missing before fetch.py has run
                self.send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    make_server(os.environ.get("DRAFT_HOME", Path(__file__).parent)).serve_forever()
