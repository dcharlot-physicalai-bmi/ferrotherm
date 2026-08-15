//! Hitachi's CMOS annealing machine, through Annealing Cloud Web.
//!
//! This is real fabricated Ising silicon reachable from a free public API, and essentially nobody
//! has used it — two papers in all of OpenAlex mention the service. It is therefore the cheapest
//! real fabric in the world to support, and supporting it is what makes "universal" mean something
//! checkable rather than rhetorical.
//!
//! # The conventions, measured rather than assumed
//!
//! **The sign is inverted.** Their energy is `Σ pᵢⱼ sᵢsⱼ`, *minimised*, so a positive coefficient is
//! **antiferromagnetic**. ferrotherm's is `-Σ Jᵢⱼ sᵢsⱼ`, where a positive coupling is
//! ferromagnetic. Every weight negates crossing this boundary.
//!
//! That was established empirically on the first call, not read off a document: four positive
//! couplings on a 2×2 block came back as a checkerboard at energy −4. A sign error here produces
//! entirely plausible output that is wrong on every problem, so it is worth the one request.
//!
//! **The topology is a King's graph.** Sites are grid coordinates and neighbours are the eight
//! surrounding cells — orthogonal *and* diagonal. Coupling two non-adjacent coordinates is an error,
//! not a silently ignored term. A vertex's own field is expressed as a self-coupling, `x0 == x1`
//! and `y0 == y1`.
//!
//! **The ASIC stores coefficients in four bits.** `-7 ≤ p ≤ 7`, integers. That is the binding
//! constraint on the machine and it is exactly the class of limit [`ferrotherm::fabric`] exists to
//! declare: a model quantised into it still runs, it just answers a different question.

use ferrotherm::fabric::{Device, Fabric, Topology, Unsupported};
use ferrotherm::embed::Embedding;
use ferrotherm::ftp::Program;
use ferrotherm::ledger::{Ledger, Prices};
use ferrotherm::schedule::Schedule;

/// Which machine to run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Machine {
    /// The CMOS annealing ASIC. 384×384 sites, four-bit coefficients.
    Asic,
    /// GPU, 32-bit integer coefficients. 512×512 sites.
    GpuInt,
    /// GPU, 32-bit float coefficients. 512×512 sites.
    GpuFloat,
}

impl Machine {
    fn code(self) -> u32 {
        match self {
            Machine::Asic => 5,
            Machine::GpuInt => 3,
            Machine::GpuFloat => 4,
        }
    }
    /// Grid side. Sites are `side × side`.
    pub fn side(self) -> usize {
        match self {
            Machine::Asic => 384,
            _ => 512,
        }
    }
    /// How this machine stores a coefficient.
    ///
    /// The GPU float path is float32, not full `f64`, and saying `None` for it — as this did —
    /// claimed every `f64` arrives intact. It does not: a coefficient needing more than 24
    /// significand bits is rounded on the way in.
    fn precision(self) -> ferrotherm::fabric::Precision {
        use ferrotherm::fabric::Precision;
        match self {
            // -7..=7 is four bits with a sign
            Machine::Asic => Precision::Fixed { bits: 4 },
            Machine::GpuInt => Precision::Fixed { bits: 32 },
            Machine::GpuFloat => Precision::Float { mantissa: 24 },
        }
    }
    fn coefficient_limit(self) -> f64 {
        self.range().hi
    }

    /// What this machine can represent, in the shared vocabulary every fabric uses.
    ///
    /// The ASIC and the integer GPU take WHOLE NUMBERS; a bit count alone cannot say that, which is
    /// why this is separate from `coupling_bits`. A program with `J = 0.5` is representable on
    /// neither, and knowing that before submitting is the difference between a refused job and a
    /// wrong answer.
    fn range(self) -> ferrotherm::fabric::Range {
        use ferrotherm::fabric::Range;
        match self {
            Machine::Asic => Range::integers(-7.0, 7.0),
            Machine::GpuInt => Range::integers(-2_147_483_647.0, 2_147_483_647.0),
            Machine::GpuFloat => Range::continuous(-3.402_823e38, 3.402_823e38),
        }
    }
}

/// A ferrotherm spin index laid out on the machine's grid.
///
/// Spin `i` sits at `(i % side, i / side)`. Any model whose couplings are not between King-adjacent
/// sites under that layout is refused rather than embedded — embedding is a compiler pass, and
/// doing it silently inside a driver is how a caller ends up solving a different problem.
fn coord(i: usize, side: usize) -> (usize, usize) {
    (i % side, i / side)
}

fn king_adjacent(a: (usize, usize), b: (usize, usize)) -> bool {
    let dx = a.0.abs_diff(b.0);
    let dy = a.1.abs_diff(b.1);
    dx <= 1 && dy <= 1 && (dx | dy) != 0
}

/// Why a model could not be laid out on the grid.
#[derive(Clone, Debug, PartialEq)]
pub enum LayoutError {
    NotAdjacent { i: usize, j: usize, a: (usize, usize), b: (usize, usize) },
    OutOfGrid { i: usize },
    CoefficientRange { value: f64, limit: f64 },
}

impl core::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LayoutError::NotAdjacent { i, j, a, b } => write!(
                f,
                "spins {i} and {j} land at {a:?} and {b:?}, which are not King-adjacent on this \
                 machine's grid. Embed the model onto the grid first; a driver that placed it for \
                 you would be choosing an embedding you did not see"
            ),
            LayoutError::OutOfGrid { i } => write!(f, "spin {i} falls outside the grid"),
            LayoutError::CoefficientRange { value, limit } => write!(
                f,
                "coefficient {value} exceeds this machine's range of ±{limit}; requantize the \
                 program for this fabric before submitting"
            ),
        }
    }
}

/// The Hitachi annealer as a backend.
pub struct Hitachi {
    token: String,
    machine: Machine,
    model: Vec<[i64; 4]>,      // x0, y0, x1, y1
    coeff: Vec<f64>,
    spins: usize,
    ledger: Ledger,
    /// Last raw energies returned, in the machine's own sign convention.
    pub last_energies: Vec<f64>,
    /// Nanoseconds the machine reported for the last run.
    pub last_execution_ns: u64,
}

impl Hitachi {
    /// `token` comes from Annealing Cloud Web. Read it from the environment; do not commit it.
    pub fn new(token: impl Into<String>, machine: Machine) -> Hitachi {
        Hitachi {
            token: token.into(),
            machine,
            model: Vec::new(),
            coeff: Vec::new(),
            spins: 0,
            ledger: Ledger::default(),
            last_energies: Vec::new(),
            last_execution_ns: 0,
        }
    }

    /// From `ACW_TOKEN` in the environment.
    pub fn from_env(machine: Machine) -> Result<Hitachi, String> {
        std::env::var("ACW_TOKEN")
            .map(|t| Hitachi::new(t, machine))
            .map_err(|_| "set ACW_TOKEN to an Annealing Cloud Web token".to_string())
    }

    /// Place a program on the grid, embedding it if it does not already fit.
    ///
    /// [`Hitachi::layout`] requires a program whose couplings are already King-adjacent under the
    /// row-major layout — which is a real constraint on the caller and, until `ferrotherm::embed`
    /// existed, one this driver could only refuse. This tries that first, and when it fails uses
    /// minor embedding to find a placement.
    ///
    /// Returns the embedding, which is needed to read the answer back: a variable may now occupy
    /// several sites, and [`ferrotherm::embed::unembed`] turns those back into one value and says
    /// which chains broke.
    ///
    /// The embedded model is not the model that was written. Chains add couplings at
    /// `chain_strength`, and a program that fitted the grid already is placed unchanged, so the
    /// returned embedding is the identity and the distinction costs nothing.
    pub fn place(&mut self, p: &Program) -> Result<Embedding, String> {
        let side = self.machine.side();
        if self.layout(p).is_ok() {
            return Ok(Embedding {
                chains: (0..p.spins).map(|i| vec![i]).collect(),
                sites: side * side,
            });
        }

        let logical = p.to_graph().map_err(|e| e.to_string())?;
        let hardware = ferrotherm::embed::topology::king(side);
        let e = ferrotherm::embed::embed(&logical, &hardware, 0).ok_or_else(|| {
            format!(
                "no King-graph placement found for {} variables on this {side}x{side} machine. \
                 That is 'not found', not 'impossible' -- minor embedding is NP-hard and this is a \
                 heuristic. A different seed or a smaller model may succeed",
                p.spins
            )
        })?;

        let placed = ferrotherm::embed::apply(&logical, &hardware, &e);
        let program = Program::from_graph(&placed.graph, &Schedule::geometric(0.05, 6.0, 40, 20));
        self.layout(&program).map_err(|err| {
            format!("the embedded program still does not fit: {err}")
        })?;
        Ok(e)
    }

    /// Lay a program out on the grid, negating every weight for their sign convention.
    ///
    /// Requires every coupling to be King-adjacent already. [`Hitachi::place`] embeds when it is
    /// not, and is what a caller who has not laid their model out by hand wants.
    pub fn layout(&mut self, p: &Program) -> Result<(), LayoutError> {
        let side = self.machine.side();
        let lim = self.machine.coefficient_limit();
        self.model.clear();
        self.coeff.clear();
        self.spins = p.spins;

        let range = self.machine.range();
        let mut push = |a: (usize, usize), b: (usize, usize), w: f64| -> Result<(), LayoutError> {
            // Against the same Range the fabric declares, rather than a magnitude comparison of
            // its own. `|w| <= 7` admits 3.5, which a machine storing four-bit INTEGERS cannot
            // hold; a second, weaker copy of a limit is how the two drift apart.
            if !range.holds(w) {
                return Err(LayoutError::CoefficientRange { value: w, limit: lim });
            }
            Ok(())
        };

        for (i, h) in &p.bias {
            if *i >= side * side {
                return Err(LayoutError::OutOfGrid { i: *i });
            }
            let a = coord(*i, side);
            // their sign is inverted, so our -h·s becomes their +(-h)·s
            let w = -*h;
            push(a, a, w)?;
            self.model.push([a.0 as i64, a.1 as i64, a.0 as i64, a.1 as i64]);
            self.coeff.push(w);
        }

        for f in &p.factors {
            let vars: Vec<usize> = f.vars().collect();
            if vars.len() != 2 {
                continue; // arity is checked by the Fabric; this is the layout pass
            }
            let (i, j) = (vars[0], vars[1]);
            if i >= side * side || j >= side * side {
                return Err(LayoutError::OutOfGrid { i: i.max(j) });
            }
            let (a, b) = (coord(i, side), coord(j, side));
            if !king_adjacent(a, b) {
                return Err(LayoutError::NotAdjacent { i, j, a, b });
            }
            let w = -f.weight(); // sign inversion, measured
            push(a, b, w)?;
            self.model.push([a.0 as i64, a.1 as i64, b.0 as i64, b.1 as i64]);
            self.coeff.push(w);
        }
        Ok(())
    }

    fn request_json(&self, num_executions: usize, schedule: &Schedule) -> String {
        let mut model = String::from("[");
        for (k, m) in self.model.iter().enumerate() {
            if k > 0 {
                model.push(',');
            }
            let c = self.coeff[k];
            let c = if self.machine == Machine::GpuFloat {
                format!("{c}")
            } else {
                format!("{}", c.round() as i64)
            };
            model.push_str(&format!("[{},{},{},{},{}]", m[0], m[1], m[2], m[3], c));
        }
        model.push(']');

        // Their schedule is geometric in temperature; ours is geometric in beta. Convert at the
        // boundary rather than pretending the parameter names line up.
        let stages = schedule.stages();
        let (b0, b1) = match (stages.first(), stages.last()) {
            (Some(a), Some(z)) => (a.beta.max(1e-6), z.beta.max(1e-6)),
            _ => (0.1, 10.0),
        };
        let steps = stages.len().clamp(1, 100);
        let per = (schedule.total_sweeps() / steps.max(1) as u64).clamp(1, 1000);

        format!(
            "{{\"type\":{},\"num_executions\":{},\"model\":{},\
             \"parameters\":{{\"temperature_num_steps\":{},\"temperature_step_length\":{},\
             \"temperature_initial\":{},\"temperature_target\":{}}},\
             \"outputs\":{{\"energies\":true,\"spins\":true,\"execution_time\":true}}}}",
            self.machine.code(),
            num_executions.clamp(1, 10),
            model,
            steps,
            per,
            1.0 / b0,
            1.0 / b1,
        )
    }
}

impl Device for Hitachi {
    fn fabric(&self) -> Fabric {
        let side = self.machine.side();
        Fabric {
            name: match self.machine {
                Machine::Asic => "hitachi-cmos-asic",
                Machine::GpuInt => "hitachi-gpu-int32",
                Machine::GpuFloat => "hitachi-gpu-float32",
            },
            topology: Topology::Named("king-graph"),
            max_spins: Some(side * side),
            max_degree: Some(8), // King's graph: orthogonal and diagonal
            coupling_precision: self.machine.precision(),
            field_precision: self.machine.precision(),
            supports_field: true,
            max_arity: 2,
            // Spin i sits at (i % side, i / side) and couplings must already be King-adjacent;
            // the driver refuses anything else rather than embedding it, so placement is native by
            // construction and the caller does their own embedding beforehand.
            native_placement: true,
            unstated: &[],
            coupling_range: Some(self.machine.range()),
            field_range: Some(self.machine.range()),
            uniform_couplings: false,
            // NOT Z1_SPICE. This is Hitachi's CMOS annealing ASIC; Z1 is Extropic's, and it has
            // not been characterised. Declaring one vendor's pre-silicon SPICE estimates as
            // another vendor's measured cost produced a joules figure that looked exactly like a
            // real one -- which is the whole failure mode the ledger exists to prevent.
            //
            // This review did not locate published per-operation energy for Annealing Cloud Web's
            // hardware. `unstated` is what that fact looks like in the type system.
            prices: Prices::UNSTATED,
        }
    }

    fn program(&mut self, p: &Program) -> Vec<Unsupported> {
        let bad = self.fabric().check(p);
        if bad.is_empty() {
            // A successful load flashes every node onto the device: the write the ledger
            // is built to account for, and the one it was never charged.
            self.ledger.writes += p.spins as u64;
            if let Err(e) = self.layout(p) {
                // A layout failure is a capability failure, and it now says WHAT failed. Every one
                // of them used to come back as `TooHighDegree { degree: 0, limit: 8 }` -- which
                // reads as "degree 0 exceeds 8", is not true of anything, and told a caller with a
                // non-adjacent coupling nothing about their non-adjacent coupling.
                return vec![Unsupported::Unplaceable { detail: e.to_string() }];
            }
        }
        bad
    }

    fn run(&mut self, schedule: &Schedule, _seed: u64) -> Result<Vec<i8>, String> {
        if self.model.is_empty() {
            return Err("no program laid out".into());
        }
        let body = self.request_json(1, schedule);
        let resp = ureq::post("https://annealing-cloud.com/api/v2/solve")
            .set("Authorization", &format!("Bearer {}", self.token))
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(180))
            .send_string(&body)
            .map_err(|e| format!("annealing cloud: {e}"))?
            .into_string()
            .map_err(|e| format!("reading response: {e}"))?;

        let side = self.machine.side();
        let mut state = vec![-1i8; self.spins];

        // The response is small and regular; parsing it with the crate's own zero-dep reader would
        // mean a dependency cycle, so it is scanned directly.
        self.last_energies = scan_numbers(&resp, "\"energies\":[");
        self.last_execution_ns = scan_numbers(&resp, "\"execution_time\":")
            .first()
            .copied()
            .unwrap_or(0.0) as u64;

        let triples = scan_spins(&resp);
        for (x, y, s) in triples {
            let i = y * side + x;
            if i < state.len() {
                state[i] = s;
            }
        }
        self.ledger.samples += self.spins as u64;
        self.ledger.reads += self.spins as u64;
        Ok(state)
    }

    fn ledger(&self) -> Ledger {
        self.ledger
    }
}

fn scan_numbers(s: &str, after: &str) -> Vec<f64> {
    let Some(i) = s.find(after) else { return Vec::new() };
    let rest = &s[i + after.len()..];
    let end = rest.find(']').unwrap_or(rest.find(',').unwrap_or(rest.len()));
    rest[..end]
        .split(',')
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect()
}

/// Pull `[x,y,s]` triples out of the **first execution's** spins array.
///
/// The response nests one array per execution, so taking every triple in the document would mix
/// executions together and produce a state that is not any single run's answer.
fn scan_spins(s: &str) -> Vec<(usize, usize, i8)> {
    const KEY: &str = "\"spins\":[";
    let Some(i) = s.find(KEY) else { return Vec::new() };
    let rest = &s[i + KEY.len()..];

    // Bracket-match the first execution's block rather than guessing where it ends.
    let mut depth = 0i32;
    let mut end = rest.len();
    for (k, c) in rest.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    end = k + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    rest[..end]
        .split('[')
        .filter_map(|chunk| {
            let body = chunk.split(']').next()?;
            let parts: Vec<&str> =
                body.split(',').map(str::trim).filter(|p| !p.is_empty()).collect();
            if parts.len() != 3 {
                return None;
            }
            Some((
                parts[0].parse::<usize>().ok()?,
                parts[1].parse::<usize>().ok()?,
                if parts[2].parse::<i64>().ok()? > 0 { 1i8 } else { -1i8 },
            ))
        })
        .collect()
}

#[cfg(test)]
mod layout_reporting_tests {
    use super::*;
    use ferrotherm::embed::Embedding;
use ferrotherm::ftp::Program;

    fn asic() -> Hitachi {
        Hitachi::new(String::from("no-token-needed-for-a-capability-check"), Machine::Asic)
    }

    #[test]
    fn a_model_that_does_not_fit_the_grid_is_placed_rather_than_refused() {
        // This is what the driver could not do. Spins 0 and 500 are nowhere near each other on a
        // 384-wide grid, so `layout` refuses; `place` embeds and finds sites that ARE adjacent.
        let p = Program::from_ftp("ftp 1\nspins 600\nfactor 1 0 500\n").unwrap();
        let mut h = asic();
        assert!(h.layout(&p).is_err(), "it does not fit as written");

        let e = h.place(&p).expect("a King's graph has room for two coupled variables");
        assert_eq!(e.chains.len(), 600);
        assert!(e.chains.iter().all(|c| !c.is_empty()), "every variable got sites");

        // and the placement really is one
        let hardware = ferrotherm::embed::topology::king(384);
        e.verify(&p.to_graph().unwrap(), &hardware).expect("place must return a valid embedding");
    }

    #[test]
    fn a_model_already_on_the_grid_is_placed_unchanged() {
        // The common case, and the one the driver demanded of everybody: adjacent spins on the
        // row-major layout. It must cost nothing and change nothing.
        let p = Program::from_ftp("ftp 1\nspins 4\nfactor 1 0 1\nfactor 1 1 2\n").unwrap();
        let e = asic().place(&p).expect("already King-adjacent");
        assert!(
            e.chains.iter().enumerate().all(|(i, c)| c == &vec![i]),
            "an identity placement, not a rearrangement: {:?}",
            &e.chains[..4]
        );
    }

    #[test]
    fn a_layout_failure_says_what_failed() {
        // Every one of these used to come back as TooHighDegree { degree: 0, limit: 8 }, which
        // reads as "degree 0 exceeds 8" -- not true of anything, and silent about the actual cause.
        // Spins 0 and 500 are nowhere near each other on a 384-wide grid.
        let p = Program::from_ftp("ftp 1\nspins 600\nfactor 1 0 500\n").unwrap();
        let bad = asic().program(&p);
        assert_eq!(bad.len(), 1, "{bad:?}");
        let msg = bad[0].to_string();
        assert!(msg.contains("King-adjacent"), "it names the real problem: {msg}");
        assert!(!msg.contains("degree"), "and not a degree that was never the issue: {msg}");
    }

    #[test]
    fn a_fractional_coefficient_cannot_reach_a_machine_that_stores_integers() {
        // |3.5| <= 7, so a magnitude comparison admits it. The ASIC stores four-bit INTEGERS.
        //
        // `layout` is called DIRECTLY here, and deliberately. Going through `program` would prove
        // nothing about this: it runs `Fabric::check` first and only lays out when that comes back
        // clean, so the fabric's range check catches 3.5 and the layout is never reached. A first
        // version of this test did go through `program`, passed, and stayed passing when the
        // layout check was reverted to the magnitude comparison -- which is a test of the wrong
        // thing wearing the right name. The two checks are defence in depth and each must hold on
        // its own.
        let p = Program::from_ftp("ftp 1\nspins 2\nfactor 3.5 0 1\n").unwrap();
        let e = asic().layout(&p).expect_err("3.5 is not a four-bit integer");
        assert!(
            matches!(e, LayoutError::CoefficientRange { .. }),
            "and it is refused as a coefficient problem: {e}"
        );

        // a whole number in range lays out
        let ok = Program::from_ftp("ftp 1\nspins 2\nfactor 3 0 1\n").unwrap();
        assert!(asic().layout(&ok).is_ok());

        // and the fabric-level check refuses it too, independently
        let bad = asic().program(&p);
        assert!(
            bad.iter().any(|u| u.to_string().contains("integers -7..=7")),
            "the outer gate names what it can hold: {bad:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_king_graph_is_what_it_says() {
        assert!(king_adjacent((1, 1), (2, 2)), "diagonals count");
        assert!(king_adjacent((1, 1), (1, 2)));
        assert!(!king_adjacent((1, 1), (1, 1)), "a site is not its own neighbour");
        assert!(!king_adjacent((1, 1), (1, 3)));
    }

    #[test]
    fn the_asic_declares_its_four_bit_limit() {
        let d = Hitachi::new("x", Machine::Asic);
        let f = d.fabric();
        assert_eq!(f.coupling_precision, ferrotherm::fabric::Precision::Fixed { bits: 4 },
                   "-7..=7 is four bits with a sign");
        assert_eq!(f.max_spins, Some(384 * 384));
        assert_eq!(f.max_degree, Some(8));
    }

    #[test]
    fn a_non_adjacent_model_is_refused_and_says_to_embed_it() {
        // Refusing beats placing it silently: an embedding the caller did not choose is a different
        // problem than the one they posed.
        let mut d = Hitachi::new("x", Machine::Asic);
        let p = Program::from_ftp("ftp 1\nspins 20\nfactor 1 0 19\n").unwrap();
        let e = d.layout(&p).unwrap_err();
        assert!(matches!(e, LayoutError::NotAdjacent { .. }));
        assert!(e.to_string().contains("Embed the model"));
    }

    #[test]
    fn a_grid_neighbour_lays_out_and_the_sign_inverts() {
        let mut d = Hitachi::new("x", Machine::Asic);
        // spins 0 and 1 are (0,0) and (1,0) under the row-major layout: adjacent
        let p = Program::from_ftp("ftp 1\nspins 4\nfactor 1 0 1\n").unwrap();
        d.layout(&p).unwrap();
        assert_eq!(d.model.len(), 1);
        assert_eq!(d.coeff[0], -1.0, "our ferromagnetic +1 must cross as their -1");
    }

    #[test]
    fn an_out_of_range_coefficient_is_refused_before_submission() {
        let mut d = Hitachi::new("x", Machine::Asic);
        let p = Program::from_ftp("ftp 1\nspins 4\nfactor 40 0 1\n").unwrap();
        let e = d.layout(&p).unwrap_err();
        assert!(matches!(e, LayoutError::CoefficientRange { .. }));
        assert!(e.to_string().contains("requantize"));
    }

    #[test]
    fn the_request_is_shaped_the_way_the_api_documents() {
        let mut d = Hitachi::new("x", Machine::Asic);
        d.layout(&Program::from_ftp("ftp 1\nspins 4\nfactor 1 0 1\n").unwrap()).unwrap();
        let j = d.request_json(3, &Schedule::geometric(0.1, 10.0, 20, 50));
        assert!(j.contains("\"type\":5"));
        assert!(j.contains("\"num_executions\":3"));
        assert!(j.contains("[0,0,1,0,-1]"), "model triple with the inverted sign: {j}");
        assert!(j.contains("temperature_num_steps"));
    }

    #[test]
    fn responses_are_scanned_correctly() {
        // The exact shape the machine returned on the first real call.
        let r = r#"{"status":0,"result":{"energies":[-4.0,-4.0],"execution_time":693447567,
                   "spins":[[[0,0,1],[1,0,-1],[0,1,-1],[1,1,1]]]},"job_id":"x"}"#;
        assert_eq!(scan_numbers(r, "\"energies\":["), vec![-4.0, -4.0]);
        assert_eq!(scan_numbers(r, "\"execution_time\":")[0], 693447567.0);
        let s = scan_spins(r);
        assert!(s.contains(&(0, 0, 1)) && s.contains(&(1, 0, -1)));
    }
}
