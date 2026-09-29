// Polls the local server; helpers (shortPos, turnText, ago) come from panel.js.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const rows = (list, from) => list.map((p, i) => `<tr class="${from + i ? "" : "best"}"><td>${from + i + 1}</td><td>${esc(p.name)}</td>` +
  `<td>${esc(shortPos(p.pos))}</td><td>${p.fp_pg.toFixed(1)}</td><td>${Math.round(p.value)}</td>` +
  `<td>${Math.round(p.p_back * 100)}%</td></tr>`).join("");
const last = (n) => String(n).split(" ").slice(1).join(" ") || n;

let shown = ""; // JSON of the state on screen: re-render only when it changes (no flicker, <details> stay open)
let state = null;

function render(s) {
  const t = turnText(s);
  $("turn").textContent = t.text;
  $("turn").classList.toggle("mine", t.mine);
  $("body").hidden = !!s.status;
  $("warn").textContent = s.warning || "";
  $("warn").hidden = !s.warning;
  if (s.status) return;
  $("top").innerHTML = rows(s.top.slice(0, 5), 0);
  $("more").innerHTML = rows(s.top.slice(5, 10), 5);
  $("morebox").hidden = s.top.length < 6;
  $("note").textContent = (s.fallback ? "No fully legal pick left - showing best available. " : "") +
    (s.unmatched.length ? `Not on board: ${s.unmatched.join(", ")}` : "");
  $("needs").innerHTML = s.needs.length ? s.needs.map((n) => `<span class="chip">${esc(n)}</span>`).join("") : '<span class="mute">None</span>';
  const names = s.roster.map((r) => (typeof r === "string" ? r : r.name));
  $("rosterSum").textContent = `${names.length}${s.rounds ? "/" + s.rounds : ""}` + (names.length ? " · " + names.map(last).join(", ") : "");
  $("roster").innerHTML = names.map((n, i) => `<div>${esc(n)} <span class="mute">${esc(shortPos(s.roster[i].pos))}</span></div>`).join("");
}

// time-dependent bits: refreshed every poll, they are not part of the changed-state comparison
function tick() {
  const now = Date.now() / 1000, s = state;
  $("reset").hidden = !(s && s.reset_at && now - s.reset_at < 5);
  $("foot").textContent = !s || s.status ? "" : `${s.n_picks}/${s.teams * s.rounds || "?"} picks` +
    (s.synced_at ? ` · synced ${ago(now - s.synced_at)}` : "");
}

async function poll() {
  try {
    const s = await (await fetch("http://127.0.0.1:8502/state")).json();
    $("down").hidden = true;
    $("ok").hidden = false;
    state = s;
    const { synced_at, ...rest } = s;
    const key = JSON.stringify(rest);
    if (key !== shown) render(s), (shown = key);
    tick();
  } catch {
    $("down").hidden = false;
    $("ok").hidden = true;
  }
}
poll();
setInterval(poll, 500);
