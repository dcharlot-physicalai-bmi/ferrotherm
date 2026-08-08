# The browser surfaces

Three pages, each self-contained, each loading `ferrotherm.wasm` from beside it.

| page | what it is |
|---|---|
| `index.html` | the explainer: what a binary stochastic neuron is, why half a lattice updates at once, and the sampler running against Onsager's exact solution on your own GPU |
| `ide.html` | the workbench: build an Ising model as JSON, sample it, certify it, run it on WebGPU |
| `graph.html` | the node editor: a problem as variables, constraints and an objective, answered in your own names |

## Serving them

Over HTTP, not from the filesystem — `WebAssembly.instantiateStreaming` needs a real
`Content-Type: application/wasm`, and a `file://` page fails with a message that sounds like a
build error rather than a serving one.

```
python3 -m http.server -d docs 8000
```

## Where they are published

The Institute site serves these at
`https://energy.physicalai-bmi.org/assets/ferrotherm/`, from a copy under `v2/public/assets/`.
That copy is not automatic, so refresh it with:

```
scripts/publish-site-assets.sh          # rebuild, copy, verify
scripts/publish-site-assets.sh --check  # verify only; non-zero if the site is behind
```

GitHub Pages used to serve them too, from a workflow that never completed a deployment on
GitHub's side — the site kept quietly serving an older commit while every run reported failure.
Two publishing paths for one artefact is one more than the number that can be kept honest, so
there is now one.

## Driving the editor from outside

`graph.html` exposes `window.ferrotherm` for tests, scripts and agents:

```js
await window.ferrotherm.ready;         // the wasm is loaded
window.ferrotherm.types();             // the node vocabulary, self-describing
const v = window.ferrotherm.add('binary', 40, 40, { name: 'mon' });
window.ferrotherm.connect(v, someConstraint, 'a');
window.ferrotherm.run();               // returns the report as text
window.ferrotherm.check();             // problems visible without running
```

`web-tests/` drives exactly this surface, so a passing `npm test` is also evidence the automation
surface works. `npm run live` drives the deployed copy rather than a local build.
