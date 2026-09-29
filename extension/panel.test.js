const test = require("node:test");
const assert = require("node:assert");
const { shortPos, turnText, ago } = require("./panel.js");

test("shortPos keeps the real positions only", () => {
  assert.strictEqual(shortPos("PG|SG|G|SG/SF|G/F|UTIL|BE"), "PG, SG");
  assert.strictEqual(shortPos("PF, C"), "PF, C");
  assert.strictEqual(shortPos("C|PF/C|F/C|F|UTIL|BE|IR"), "C");
  assert.strictEqual(shortPos("G|UTIL|BE"), "G"); // no real position listed: fall back to what is left
  assert.strictEqual(shortPos(null), "");
});

test("turnText", () => {
  const s = { on_clock: 2, me: 8, n_picks: 64, my_next: 68, teams: 14 };
  assert.deepStrictEqual(turnText(s), { text: "Pick 65 · yours in 3 (R5)", mine: false });
  assert.deepStrictEqual(turnText({ ...s, on_clock: 8, my_next: 79 }), { text: "YOUR PICK", mine: true });
  assert.strictEqual(turnText({ ...s, on_clock: null }).text, "Draft complete");
  assert.strictEqual(turnText({ ...s, my_next: null }).text, "Pick 65 · no picks left");
  assert.strictEqual(turnText({ ...s, n_picks: 0, my_next: 9 }).text, "Pick 1 · yours in 8 (R1)");
  assert.strictEqual(turnText({ status: "no picks yet" }).text, "Waiting for picks");
});

test("ago", () => {
  assert.strictEqual(ago(0.4), "0s ago");
  assert.strictEqual(ago(1.6), "2s ago");
  assert.strictEqual(ago(59), "59s ago");
  assert.strictEqual(ago(125), "2m ago");
  assert.strictEqual(ago(-3), "0s ago"); // clock skew
});
