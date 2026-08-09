# Browser tests

Two pages, driven the way a person or an agent drives them.

`editor.test.mjs` covers the node graph, `workbench.test.mjs` the JSON workbench. Both are surfaces
the Rust test suite cannot see: it lives in a browser, it talks to the sampler across the wasm boundary, and its
failures are silent. A miswired port does not panic. A double free in the run path takes the page
down with a stack trace nobody reads. Both of those actually happened, and both are pinned here.

    npm install && npm test

The editor tests drive `window.ferrotherm`, the same scriptable API an agent would use, so a passing
run is also evidence that the automation surface works. The workbench tests paste request bodies
into its pane and check they run unchanged — the pane claims to take "the same JSON the API and the
MCP tools take", and a claim is worth testing.

`npm run live` drives the DEPLOYED editor rather than a local build. A page that loads is not a page
that is current.
