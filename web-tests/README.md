# Editor tests

The node editor is the surface most people meet first, and it is the one surface the Rust test
suite cannot see: it lives in a browser, it talks to the sampler across the wasm boundary, and its
failures are silent. A miswired port does not panic. A double free in the run path takes the page
down with a stack trace nobody reads. Both of those actually happened, and both are pinned here.

    npm install && npm test

The tests drive `window.ferrotherm`, the same scriptable API an agent would use, so a passing run
is also evidence that the automation surface works.
