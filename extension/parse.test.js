const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { parsePicks } = require("./parse.js");

const html = fs.readFileSync(path.join(__dirname, "..", "tests", "fixtures", "espn_picks.html"), "utf8");

test("parses the Picks panel fixture, sorted by overall", () => {
  const p = parsePicks(html, 14);
  assert.deepStrictEqual(p.map((x) => x.overall), [1, 4, 9, 160]);
  assert.deepStrictEqual(p[0], {
    round: 1, pick: 1, overall: 1, espn_id: 3112335, name: "Nikola Jokic", nba_team: "DEN", pos: "C",
    fantasy_team: "Banana Chargers",
  });
  assert.strictEqual(p[1].pos, "PF, C");
  assert.strictEqual(p[1].fantasy_team, "Alex's Awesome Team");
  assert.strictEqual(p[3].espn_id, null); // nomug headshot
  assert.strictEqual(p[3].fantasy_team, "老北京泥根儿");
});

test("overall follows the team count it is given", () => {
  assert.deepStrictEqual(parsePicks(html, 12).map((x) => x.overall), [1, 4, 9, 138]);
});
