# ferrotherm-serve

The agent-facing tier for [ferrotherm](https://crates.io/crates/ferrotherm): thermodynamic sampling
over Ising graphs, reachable over HTTP and over the Model Context Protocol.

One set of operations, two transports, no dependencies outside this workspace. The binary you run
against your own models is one you can read end to end.

```
cargo install ferrotherm-serve
```

## The operations

| Operation | What it does |
|---|---|
| `sample` | Draw a state from a Boltzmann distribution by chromatic block-Gibbs |
| `anneal` | Minimise an Ising energy down a geometric beta ladder |
| `energy` | Score a specific state under a specific graph |
| `verify` | Compare the sampler against exact enumeration (n ≤ 20) |
| `capabilities` | Describe the server: operations, limits, conventions |

States are −1/+1 and energy is `-Σ_ij J_ij s_i s_j - Σ_i h_i s_i`. Every run returns an energy
ledger priced at Z1-class device figures, and the same seed and thread count reproduce a run
exactly.

## HTTP

```
ferrotherm-serve 127.0.0.1:8479
```

```console
$ curl -s localhost:8479/v1/capabilities          # everything you need to use it
$ curl -s -X POST localhost:8479/v1/anneal \
    -d '{"graph":{"builtin":"lattice2d","l":10},"beta_min":0.05,"beta_max":4.0,"stages":40}'
{"best_energy":-200, ...}
```

A 10×10 periodic ferromagnetic lattice has 200 bonds, so −200 is the true ground state.

Caller mistakes come back as `400` with the fix in the message, not as a stack trace:

```console
$ curl -s -X POST localhost:8479/v1/sample -d '{"graph":{"n":4,"couplings":[[0,9,1]]}}'
{"error":"entry 0: j = 9 is out of range for a graph of 4 nodes","ok":false}
```

## Model Context Protocol

Point any MCP client at the `ferrotherm-mcp` binary:

```json
{
  "mcpServers": {
    "ferrotherm": { "command": "ferrotherm-mcp" }
  }
}
```

Five tools appear: `ferrotherm_sample`, `ferrotherm_anneal`, `ferrotherm_energy`,
`ferrotherm_verify`, `ferrotherm_capabilities`. Each carries a JSON Schema complete enough to call
from the schema alone — there is a test that lifts an example out of a tool description and runs it
verbatim, so that stays true.

Bad arguments return `isError: true` with a correctable message rather than failing the request, so
a model can fix its own call and retry.

## Verify before you scale

`verify` enumerates the exact Boltzmann distribution and reports the total variation distance from
the sampler's empirical one, **alongside the sampling noise floor for the number of draws you
asked for**. Compare against that floor, not against zero: a TV below it is agreement, not accuracy
you can quote.

If TV comes in above the floor at high beta, the usual cause is correlated draws rather than a
broken sampler — raise `thin`. On an 8-node ring at β = 1.6, back-to-back draws land at TV 0.051
against a 0.040 floor; thinning to 40 sweeps between draws brings it to well under. Both numbers
are asserted in the test suite.

## Limits

Advertised in `capabilities` so you can size a job rather than discover the wall by hitting it:
4,000,000 nodes and 20,000,000,000 node updates per request, and 20 nodes for exact verification
(enumeration is 2ⁿ). States for graphs over 4,096 nodes are omitted unless you pass
`"return_state": true`.

## On the name of the unit

The sampled unit is a **binary stochastic neuron** in machine learning, or an **Ising spin under
Glauber dynamics** in statistical physics. The 2016 coinage *p-bit* names the same object. The
[explainer](https://dcharlot-physicalai-bmi.github.io/ferrotherm/) has the full lineage and a table
of equivalent names.

## Licence

Apache-2.0. Institute for Physical AI @ BMI.
