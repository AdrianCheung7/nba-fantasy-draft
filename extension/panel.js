// Pure helpers for the side panel (node --test in panel.test.js).
const REAL = ["PG", "SG", "SF", "PF", "C"]; // the rest of ESPN's eligibility list is slots: G, F, G/F, F/C, UTIL, BE, IR

function shortPos(p) {
  const all = String(p ?? "").split(/\s*[|,]\s*/).filter(Boolean);
  const real = all.filter((x) => REAL.includes(x));
  return (real.length ? real : all.filter((x) => !["BE", "IR", "UTIL"].includes(x))).join(", ");
}

function turnText(s) {
  if (s.status) return { text: "Waiting for picks", mine: false };
  if (s.on_clock == null) return { text: "Draft complete", mine: false };
  if (s.on_clock === s.me) return { text: "YOUR PICK", mine: true };
  const pick = s.n_picks + 1;
  const tail = s.my_next == null ? "no picks left" : `yours in ${s.my_next - pick} (R${Math.ceil(s.my_next / s.teams)})`;
  return { text: `Pick ${pick} · ${tail}`, mine: false };
}

function ago(sec) {
  const n = Math.max(0, Math.round(sec));
  return n < 60 ? `${n}s ago` : `${Math.floor(n / 60)}m ago`;
}

if (typeof module !== "undefined") module.exports = { shortPos, turnText, ago };
