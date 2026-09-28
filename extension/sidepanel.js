const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pos = (p) => String(p ?? "").split("|").filter((x) => !["BE", "IR", "UTIL"].includes(x)).join(", ");

function render(s) {
  if (s.status) { // server up, nothing posted yet
    $("turn").textContent = "Waiting for picks";
    $("until").textContent = "Open the ESPN draft room and its Picks panel.";
    $("note").textContent = "";
    $("warn").hidden = true;
    $("top").innerHTML = $("roster").innerHTML = $("needs").innerHTML = "";
    return;
  }
  const mine = s.on_clock === s.me;
  $("warn").textContent = s.warning || "";
  $("warn").hidden = !s.warning;
  $("turn").textContent = s.on_clock == null ? "Draft complete" : mine ? "YOUR PICK" : `Team ${s.on_clock + 1} on the clock`;
  const next = s.my_next == null ? null : s.my_next - (s.n_picks + 1);
  $("until").textContent = s.on_clock == null || mine ? "" : next == null ? "No picks left for you." : `${next} pick${next === 1 ? "" : "s"} until yours`;
  $("note").textContent = (s.fallback ? "No fully legal pick left - showing best available. " : "") +
    (s.unmatched.length ? `Not on board: ${s.unmatched.join(", ")}` : "");
  $("top").innerHTML = s.top.map((p, i) => `<tr class="${i ? "" : "best"}"><td>${esc(p.name)}</td><td>${esc(pos(p.pos))}</td>` +
    `<td>${p.fp_pg.toFixed(1)}</td><td>${Math.round(p.games)}</td><td>${Math.round(p.value)}</td>` +
    `<td>${Math.round(p.p_back * 100)}%</td></tr>`).join("");
  $("roster").innerHTML = s.roster.length ? s.roster.map((r) => esc(typeof r === "string" ? r : r.name)).join("<br>") : '<span class="mute">Empty</span>';
  $("needs").textContent = s.needs.length ? s.needs.join(", ") : "None";
}

async function poll() {
  try {
    const s = await (await fetch("http://127.0.0.1:8502/state")).json();
    $("down").hidden = true;
    $("ok").hidden = false;
    render(s);
  } catch {
    $("down").hidden = false;
    $("ok").hidden = true;
  }
}
poll();
setInterval(poll, 500);
