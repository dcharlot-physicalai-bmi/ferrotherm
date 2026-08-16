# ferrotherm-cloud

Real fabricated Ising silicon, reached through the same `ferrotherm::fabric::Device` trait as a CPU.

Currently one fabric: **Hitachi's CMOS annealing machine**, through Annealing Cloud Web. A 384×384
King's graph with four-bit coefficients, and a free public API that essentially nobody has used —
two papers in all of OpenAlex mention the service. It is therefore the cheapest real fabric in the
world to support, and supporting it is what makes "runs on any fabric" checkable rather than
rhetorical.

## You bring your own credentials

**This crate ships no token, no account and no default identity.** It talks to an account you create
yourself, and until you do, it does nothing — there is exactly one network call in the crate and
reaching it takes a token you supplied, a program you laid out, and a call you made.

### Getting a token

1. Request one at <https://annealing-cloud.com/en/web-api/token-request.html>. The form asks for an
   **email address** and a **country**, and requires agreeing to two conditions: that you will not
   use the site or its output data for any purpose including the development of weapons of mass
   destruction (their Terms of Use, Section 8, Export Controls), and that you consent to the
   collection of personal information under those Terms. Intended use and user type are optional.
2. The administrator emails the token back to you. That page does not state how long issuance takes
   or what the usage limits are; the service homepage describes the Web API as free.
3. Put it in your environment, never in a file you commit:

```sh
export ACW_TOKEN=<the token they emailed you>
cargo run --release -p ferrotherm-cloud --example hitachi_run
```

`Hitachi::from_env` reads exactly that one variable and returns `Err` when it is unset. It does not
fall back to a bundled key, a config file or a credential store, because there are none.
`Hitachi::new` takes the token directly if you would rather source it another way.

Read the API you are calling first: <https://annealing-cloud.com/en/web-api/reference/v2.html>.

### Where it sends

The default endpoint is `ACW_ENDPOINT` — the public solve endpoint of the service this driver is
named for. It is a default, not a fixture:

```rust
let d = Hitachi::new(token, Machine::Asic).with_endpoint("http://127.0.0.1:8080/solve");
```

`endpoint()` tells you where an instance will post. A test in this crate pins the guarantee: with
`ACW_TOKEN` unset there is no device at all, and pointed at a port nothing can listen on, describing
the fabric and laying a program out still succeed — because those paths are local.

## What it does before it sends anything

Everything except the submission itself is local, and most of the value is there:

- **Sign conventions, measured rather than assumed.** Their energy is `Σ pᵢⱼ sᵢsⱼ`, *minimised*, so
  a positive coefficient is **antiferromagnetic** — the opposite of ferrotherm's `-Σ Jᵢⱼ sᵢsⱼ`. That
  was established empirically on the first call, not read off a document: four positive couplings on
  a 2×2 block came back as a checkerboard at energy −4. A sign error here produces entirely
  plausible output that is wrong on every problem.
- **Layout and refusal.** Sites are grid coordinates; neighbours are the eight surrounding cells,
  orthogonal *and* diagonal. A coupling between non-adjacent coordinates is an error, not a silently
  dropped term. `place()` embeds a model that does not already fit; `layout()` requires one that
  does.
- **Limits declared before submission.** The ASIC stores coefficients in four bits, `-7 ≤ p ≤ 7`,
  **integers** — so `J = 0.5` is refused here rather than quietly rounded on the way in. Knowing that
  before you submit is the difference between a refused job and a wrong answer.

```sh
cargo add ferrotherm-cloud
```

Apache-2.0. Part of [ferrotherm](https://github.com/dcharlot-physicalai-bmi/ferrotherm), from the
Institute for Physical AI @ BMI.
