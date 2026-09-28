// Reads the Picks panel, posts it to the local server, highlights the top 5 in ESPN's player list.
// parse.js is loaded first (shared content-script scope).
let teams = 14;
let sent = "";     // last pick list the server accepted
let best = [];     // top-5 from the last response
let timer;

const st = document.createElement("style");
st.textContent = `.fda-r1{outline:3px solid #16a34a!important;outline-offset:-3px}
.fda-r2{outline:3px solid #f59e0b!important;outline-offset:-3px}
.fda-tag{margin-left:6px;padding:0 4px;border-radius:3px;font:600 11px sans-serif;color:#fff;background:#16a34a}
.fda-tag.fda-r2{background:#b45309;outline:none!important}`;
document.head.append(st);

const norm = (s) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9 ]/g, "").trim();

function rowOf(el) {
  const r = el.closest("tr, [role=row]");
  if (r) return r;
  for (let a = el.parentElement; a; a = a.parentElement) if (a.children.length >= 3) return a;
  return el.parentElement;
}

function highlight() {
  document.querySelectorAll(".fda-tag").forEach((e) => e.remove());
  document.querySelectorAll(".fda-r1, .fda-r2").forEach((e) => e.classList.remove("fda-r1", "fda-r2"));
  best.forEach((p, i) => {
    for (const el of document.querySelectorAll(".playerinfo__playername")) {
      if (el.closest(".pick-message__container, .draft-column ul.pa3") || norm(el.textContent) !== norm(p.name)) continue;
      const c = i ? "fda-r2" : "fda-r1";
      rowOf(el).classList.add(c);
      const tag = document.createElement("span");
      tag.className = `fda-tag ${c}`;
      tag.textContent = `#${i + 1} · ${Math.round(p.p_back * 100)}% back`;
      el.after(tag);
    }
  });
}

function update() {
  obs.disconnect(); // our own DOM edits must not retrigger us
  highlight();
  obs.observe(document.body, { childList: true, subtree: true, characterData: true });
}

function run() {
  const picks = parsePicks(document, teams);
  const key = JSON.stringify(picks);
  if (key === sent || !picks.length) return update(); // never post an empty list: panel may just not be rendered
  chrome.runtime.sendMessage({ picks, url: location.href }, (r) => {
    if (chrome.runtime.lastError || !r || r.error) return void setTimeout(run, 2000); // server down: retry
    sent = key;
    teams = r.teams || teams;
    best = (r.top || []).slice(0, 5);
    update();
  });
}

const obs = new MutationObserver(() => {
  clearTimeout(timer);
  timer = setTimeout(run, 150);
});
obs.observe(document.body, { childList: true, subtree: true, characterData: true });
run();
