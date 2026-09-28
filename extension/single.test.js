const test = require("node:test");
const assert = require("node:assert");
const { singleFlight } = require("./single.js");

// fake clock: setT records the (single) pending timer, clearT drops it
function clock() {
  const c = { fn: null, sets: 0 };
  c.setT = (fn) => { c.sets++; c.fn = fn; return c.sets; };
  c.clearT = () => { c.fn = null; };
  return c;
}

test("failed attempts under a firing observer keep one retry timer and one request in flight", () => {
  const c = clock();
  const dones = [];
  const run = singleFlight((done) => dones.push(done), 2000, c.setT, c.clearT);
  for (let i = 0; i < 5; i++) run(); // observer fires while the first request is pending
  assert.strictEqual(dones.length, 1);
  dones[0](false);
  assert.ok(c.fn);
  for (let i = 0; i < 5; i++) run(); // observer keeps firing, server still down
  assert.strictEqual(dones.length, 2); // the first one went out again, then one outstanding
  dones[1](false);
  assert.strictEqual(c.sets, 2); // one timer re-armed per failure, never several chains
});

test("a change during a request is sent right after it succeeds", () => {
  const c = clock();
  const dones = [];
  const run = singleFlight((done) => dones.push(done), 2000, c.setT, c.clearT);
  run(); run();
  dones[0](true);
  assert.strictEqual(dones.length, 2);
  dones[1](true);
  assert.strictEqual(dones.length, 2); // no loop when nothing changed
  assert.strictEqual(c.fn, null);
});
