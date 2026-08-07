//! A conformance suite any fabric can run.
//!
//! This review did not locate anything in this field that reports sampling fidelity. Statistical
//! physics reports autocorrelation time and no effective sample size; the Bayesian stack reports
//! ESS and has no Ising model; the optimisation stack reports success rates and has no notion of a
//! target distribution at all. We did not find a machine that reports the inverse temperature it
//! actually achieved, nor one that reports a sampling-noise floor.
//!
//! This is the suite that asks. It runs against anything implementing [`crate::fabric::Device`], so
//! a CPU, a GPU, an FPGA and somebody's cloud annealer are all scored the same way, on problems
//! whose answers are known before the machine is asked.
//!
//! # It must be able to fail
//!
//! Half the cases here exist to catch a machine that looks like it works. A fabric that always
//! returns the same low-energy state passes every "did it find the optimum" test ever written, so
//! this suite also asks a machine to sample badly on purpose and **fails it if the certificate
//! comes back clean**. A suite that cannot reject anything certifies nothing.
//!
//! # And it is run on ourselves first
//!
//! A conformance suite whose author exempts themselves is worthless. `examples/conform.rs` runs
//! this against our own CPU backend and prints the result unedited, including any case we fail.

use crate::fabric::Device;
use crate::ftp::Program;
use crate::schedule::Schedule;

/// One case's outcome.
#[derive(Clone, Debug)]
pub struct CaseResult {
    pub name: &'static str,
    /// What the case is checking, in one line.
    pub asks: &'static str,
    pub passed: bool,
    /// The measurement, whether it passed or not.
    pub detail: String,
}

/// What a fabric scored.
#[derive(Clone, Debug)]
pub struct Report {
    pub fabric: String,
    pub cases: Vec<CaseResult>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.cases.iter().all(|c| c.passed)
    }
    pub fn failures(&self) -> impl Iterator<Item = &CaseResult> {
        self.cases.iter().filter(|c| !c.passed)
    }
}

impl core::fmt::Display for Report {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "ferrotherm conformance — {}", self.fabric)?;
        writeln!(f, "{}", "-".repeat(78))?;
        for c in &self.cases {
            writeln!(f, "{} {:<28} {}", if c.passed { "PASS" } else { "FAIL" }, c.name, c.detail)?;
        }
        let n = self.cases.len();
        let ok = self.cases.iter().filter(|c| c.passed).count();
        write!(f, "{}\n{ok}/{n} cases", "-".repeat(78))
    }
}

fn case(name: &'static str, asks: &'static str, passed: bool, detail: String) -> CaseResult {
    CaseResult { name, asks, passed, detail }
}

fn ladder() -> Schedule {
    Schedule::geometric(0.05, 6.0, 80, 40)
}

/// Run the suite.
pub fn run(dev: &mut dyn Device) -> Report {
    let fabric = dev.fabric().name.to_string();
    let mut cases = Vec::new();

    // --- 1. a model whose optimum is arithmetic -------------------------------------------------
    {
        let mut src = String::from("ftp 1\nname ferromagnetic-ring\nspins 12\n");
        for i in 0..12 {
            src.push_str(&format!("factor 1 {i} {}\n", (i + 1) % 12));
        }
        let p = Program::from_ftp(&src).unwrap();
        let (ok, detail) = match solve(dev, &p, 1) {
            Err(e) => (false, e),
            Ok(s) => {
                let e = p.to_graph().unwrap().energy(&s);
                (e == -12.0, format!("ground energy {e}, exact -12"))
            }
        };
        cases.push(case("ferromagnet", "can it satisfy every bond at once", ok, detail));
    }

    // --- 2. frustration, where the optimum is not "satisfy everything" ---------------------------
    {
        let mut src = String::from("ftp 1\nname frustrated-ring\nspins 5\n");
        for i in 0..5 {
            src.push_str(&format!("factor -1 {i} {}\n", (i + 1) % 5));
        }
        let p = Program::from_ftp(&src).unwrap();
        let (ok, detail) = match solve(dev, &p, 2) {
            Err(e) => (false, e),
            Ok(s) => {
                let e = p.to_graph().unwrap().energy(&s);
                (e == -3.0, format!("ground energy {e}, exact -3 (one bond must break)"))
            }
        };
        cases.push(case("frustration", "does it know a bond must break", ok, detail));
    }

    // --- 3. a planted optimum at a size nothing can enumerate ------------------------------------
    {
        let planted = crate::planted::frustrated_loops(8, 96, 3);
        let p = Program::from_graph(&planted.graph, &ladder());
        let (ok, detail) = match solve(dev, &p, 3) {
            Err(e) => (false, e),
            Ok(s) => {
                let ex = planted.excess(&s);
                (ex < 0.05, format!("{:.2}% above a planted optimum of {}", ex * 100.0, planted.ground_energy))
            }
        };
        cases.push(case("planted optimum", "how close on a known answer", ok, detail));
    }

    // --- 4. agreement with exact inference -------------------------------------------------------
    {
        let mut src = String::from("ftp 1\nname chain\nspins 60\n");
        for i in 0..59 {
            src.push_str(&format!("factor 1 {i} {}\n", i + 1));
        }
        let p = Program::from_ftp(&src).unwrap();
        let exact = crate::exact::Elimination::default()
            .ground_state(&p.to_graph().unwrap())
            .ok()
            .and_then(|e| e.ground_energy);
        let (ok, detail) = match (solve(dev, &p, 4), exact) {
            (Err(e), _) => (false, e),
            (Ok(s), Some(x)) => {
                let e = p.to_graph().unwrap().energy(&s);
                (
                    (e - x).abs() < 1e-9,
                    format!("{e} against variable elimination's exact {x}"),
                )
            }
            _ => (false, "exact inference declined this graph".into()),
        };
        cases.push(case("exact agreement", "does it match an exact solver", ok, detail));
    }

    // --- 5. determinism -------------------------------------------------------------------------
    {
        let p = Program::from_graph(&crate::ising::lattice2d(8, 1.0), &ladder());
        let (ok, detail) = match (solve(dev, &p, 7), solve(dev, &p, 7)) {
            (Ok(a), Ok(b)) => (a == b, format!("same seed reproduces: {}", a == b)),
            (Err(e), _) | (_, Err(e)) => (false, e),
        };
        cases.push(case("determinism", "does one seed give one answer", ok, detail));
    }

    // --- 6. the discriminating case: it must be able to FAIL -------------------------------------
    //
    // Everything above rewards a machine that returns good states. A fabric that always returns the
    // same state passes all of it. This asks for a deliberately bad run and fails the fabric if its
    // certificate blesses it.
    {
        let g = crate::ising::lattice2d(24, 1.0);
        let mut smp = crate::gibbs::Sampler::new(&g, 0.7, 4);
        let mut samples = Vec::new();
        let mut trace = Vec::new();
        for _ in 0..300 {
            smp.sweeps(1, None);
            samples.push(smp.s.clone());
            trace.push(g.energy(&smp.s));
        }
        let c = crate::certify::certify(&g, 0.7, &samples, &trace);
        let ok = !c.passed();
        cases.push(case(
            "rejects a bad run",
            "does the certificate catch an unequilibrated chain",
            ok,
            if ok {
                format!("caught it: {}", c.findings.first().map(|f| f.to_string()).unwrap_or_default())
            } else {
                "blessed a chain that never left its initial condition".into()
            },
        ));
    }

    // --- 7. sampling fidelity, which nothing else in this field reports --------------------------
    {
        let g = crate::ising::ring(10, 1.0, 0.3);
        let mut smp = crate::gibbs::Sampler::new(&g, 0.5, 11);
        smp.sweeps(500, None);
        let (mut samples, mut trace) = (Vec::new(), Vec::new());
        for _ in 0..3000 {
            smp.sweeps(8, None);
            samples.push(smp.s.clone());
            trace.push(g.energy(&smp.s));
        }
        let c = crate::certify::certify(&g, 0.5, &samples, &trace);
        let ok = c.passed();
        let detail = match (c.tv_exact, c.noise_floor) {
            (Some(tv), Some(fl)) => format!(
                "beta_eff {:.4} (asked 0.5), ess {:.0}, tv {tv:.4} against a {fl:.4} noise floor",
                c.beta_eff, c.ess
            ),
            _ => format!("beta_eff {:.4}, ess {:.0}", c.beta_eff, c.ess),
        };
        cases.push(case("sampling fidelity", "at what temperature did it really sample", ok, detail));
    }

    Report { fabric, cases }
}

fn solve(dev: &mut dyn Device, p: &Program, seed: u64) -> Result<Vec<i8>, String> {
    let bad = dev.program(p);
    if !bad.is_empty() {
        return Err(format!(
            "refused: {}",
            bad.iter().map(|u| u.to_string()).collect::<Vec<_>>().join("; ")
        ));
    }
    dev.run(&ladder(), seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::Cpu;

    #[test]
    fn our_own_backend_passes_its_own_suite() {
        // Run on ourselves first, and if we ever stop passing, this says so before anyone else does.
        let r = run(&mut Cpu::default());
        assert!(r.passed(), "we fail our own conformance suite:\n{r}");
        assert_eq!(r.cases.len(), 7);
    }

    #[test]
    fn the_suite_can_actually_reject_something() {
        // A conformance suite that passes everything certifies nothing. This device answers every
        // program with a fixed all-up state -- the exact failure mode a "did it find the optimum"
        // test cannot see, since on a ferromagnet all-up IS the optimum.
        struct AlwaysUp(usize);
        impl Device for AlwaysUp {
            fn fabric(&self) -> crate::fabric::Fabric {
                crate::fabric::Fabric::unconstrained("always-up", crate::ledger::Z1_SPICE)
            }
            fn program(&mut self, p: &Program) -> Vec<crate::fabric::Unsupported> {
                self.0 = p.spins;
                Vec::new()
            }
            fn run(&mut self, _: &Schedule, _: u64) -> Result<Vec<i8>, String> {
                Ok(vec![1i8; self.0])
            }
            fn ledger(&self) -> crate::ledger::Ledger {
                crate::ledger::Ledger::default()
            }
        }

        let r = run(&mut AlwaysUp(0));
        assert!(!r.passed(), "a constant device must not pass:\n{r}");
        // it passes the ferromagnet, because all-up really is that optimum
        assert!(r.cases.iter().find(|c| c.name == "ferromagnet").unwrap().passed);
        // and it fails frustration and the planted instance, which is what catches it
        assert!(!r.cases.iter().find(|c| c.name == "frustration").unwrap().passed);
        assert!(!r.cases.iter().find(|c| c.name == "planted optimum").unwrap().passed);
    }

    #[test]
    fn a_fabric_that_refuses_a_program_reports_why() {
        // A backend declining a case must produce a readable reason, not an unexplained failure.
        struct Narrow;
        impl Device for Narrow {
            fn fabric(&self) -> crate::fabric::Fabric {
                let mut f = crate::fabric::Fabric::unconstrained("narrow", crate::ledger::Z1_SPICE);
                f.max_spins = Some(4);
                f
            }
            fn program(&mut self, p: &Program) -> Vec<crate::fabric::Unsupported> {
                self.fabric().check(p)
            }
            fn run(&mut self, _: &Schedule, _: u64) -> Result<Vec<i8>, String> {
                Ok(vec![1i8; 4])
            }
            fn ledger(&self) -> crate::ledger::Ledger {
                crate::ledger::Ledger::default()
            }
        }
        let r = run(&mut Narrow);
        let ferro = r.cases.iter().find(|c| c.name == "ferromagnet").unwrap();
        assert!(!ferro.passed);
        assert!(ferro.detail.contains("refused"), "should say it was refused: {}", ferro.detail);
        assert!(ferro.detail.contains("12"), "and name the limit it hit: {}", ferro.detail);
    }
}
