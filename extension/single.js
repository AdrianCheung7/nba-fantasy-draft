// One request at a time and one shared retry timer. attempt(done) starts a request and calls done(ok) when it ends.
// A call made while a request is pending is remembered and re-run after that request succeeds.
function singleFlight(attempt, retryMs, setT = setTimeout, clearT = clearTimeout) {
  let busy = false, dirty = false, retry;
  function run() {
    if (busy) return void (dirty = true);
    busy = true;
    dirty = false;
    attempt((ok) => {
      busy = false;
      if (!ok) { clearT(retry); retry = setT(run, retryMs); } // failed: one retry timer, never a chain per call
      else if (dirty) { dirty = false; run(); }
    });
  }
  return run;
}

if (typeof module !== "undefined") module.exports = { singleFlight };
