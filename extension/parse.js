// Parse ESPN's draft-room Picks panel. root: a DOM Element/Document, or an HTML string (Node tests).
// Both paths reduce to one HTML string per pick row, so there is a single extractor.
const ROW = /<li\b[^>]*pick-message__container[\s\S]*?<\/li>/g;
const UNESC = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " " };

function unescapeHtml(s) {
  return s.replace(/&(?:#(\d+)|#x([0-9a-f]+)|(\w+));/gi, (m, d, h, n) =>
    d ? String.fromCodePoint(+d) : h ? String.fromCodePoint(parseInt(h, 16)) : UNESC[n.toLowerCase()] ?? m);
}

function parsePicks(root, teams) {
  const rows = typeof root === "string" ? root.match(ROW) || []
    : [...root.querySelectorAll("li.pick-message__container")].map((e) => e.outerHTML);
  const cls = (row, c) => {
    const m = row.match(new RegExp(`class="[^"]*${c}[^"]*"[^>]*>([\\s\\S]*?)</span>`));
    return m ? unescapeHtml(m[1].replace(/<[^>]*>/g, "")).trim() : "";
  };
  const picks = [];
  for (const row of rows) {
    const info = row.match(/pick-info[^>]*>([\s\S]*?)<\/div>/);
    const m = info && unescapeHtml(info[1].replace(/<[^>]*>/g, "")).match(/R(\d+),\s*P(\d+)\s*-\s*(.*)/);
    if (!m) continue;
    const [round, pick] = [+m[1], +m[2]];
    const id = row.match(/players\/full\/(\d+)\.png/);
    picks.push({
      round, pick, overall: (round - 1) * teams + pick, espn_id: id ? +id[1] : null,
      name: cls(row, "playerinfo__playername"), nba_team: cls(row, "playerinfo__playerteam"),
      pos: cls(row, "playerinfo__playerpos"), fantasy_team: m[3].trim(),
    });
  }
  return picks.sort((a, b) => a.overall - b.overall);
}

if (typeof module !== "undefined") module.exports = { parsePicks };
