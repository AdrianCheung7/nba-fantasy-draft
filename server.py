"""Local pick server for the Chrome extension.  Run: python server.py  (127.0.0.1:8502 only)"""
import json
import os
import re
import threading
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
        return self.s

    def match(self, p):
        """Board index by ESPN id, else name + NBA team, else name only, else None."""
        if p.get("espn_id") in self.by_id:
            return self.by_id[p["espn_id"]]
        cand = self.by_name.get(norm(p.get("name", "")), [])
        same = [i for i in cand if TEAMS.get(int(self.b.team_id.iat[i])) == abbr(p.get("nba_team", ""))]
        return (same or cand or [None])[0]

    def post(self, body):
        with self.lock:
            s = self.load()
            teams = s["teams"]
            slots, unmatched = {}, []
            for p in body["picks"]:
                o = (p["round"] - 1) * teams + p["pick"]  # don't trust the client's team count
                slots[o] = self.match(p)
                if slots[o] is None:
                    unmatched.append(p["name"])
            n = max(slots, default=0)
            picks = [{"overall": o, "player": slots.get(o)} for o in range(1, n + 1)]
            ids = [None if x["player"] is None else int(self.b.player_id.iat[x["player"]]) for x in picks]
            draft_file = self.data / "draft.json"
            tmp = draft_file.with_suffix(".tmp")  # same write-then-rename as app.py
            tmp.write_text(json.dumps(ids))
            tmp.replace(draft_file)
            r = draft.advise(self.b, self.ev, picks, s, s.get("strategy", "greedy"))
            top = r["top"]
            out = {**r, "top": json.loads(top.to_json(orient="records")), "fallback": bool(top.attrs.get("fallback")),
                   "mapped": len(body["picks"]) - len(unmatched), "unmatched": unmatched, "n_picks": n,
                   "teams": teams, "me": s["my_slot"] - 1}
            self.last = json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
            return self.last


def make_server(home, port=PORT):
    app = App(home)

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
