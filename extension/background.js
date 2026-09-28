// Page-context fetch to localhost can be blocked (mixed content / Private Network Access); the worker isn't.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  fetch("http://127.0.0.1:8502/picks", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(msg),
  }).then((r) => r.json()).then(reply).catch((e) => reply({ error: String(e) }));
  return true; // reply asynchronously
});
