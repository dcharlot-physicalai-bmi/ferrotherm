//! ferrotherm-silicon — the IPAI-owned silicon layer for the ferrotherm deployment ladder.
//!
//! Pure Rust, no vendor tools, no third-party stacks. Selected algorithms ported with permission
//! from Open Interface Engineering's openie-fpga; this crate is fully independent of it.

pub mod bitstream;
pub mod frame;
pub mod framebuf;
pub mod json;
pub mod pips;
pub mod route;
pub mod segbits;
pub mod tilegrid;
pub mod lut;

#[cfg(feature = "flash")]
pub mod flash;

/// The Alchitry Pt V2 as a [`ferrotherm::fabric::Device`].
///
/// Porting this path onto the shared trait is what forced its real limits into the open, and they
/// are more restrictive than the LUT count suggests. A LUT6 has six inputs; the threshold table
/// built by [`lut::bsn_threshold_init`] spends one of them on the random bit and **counts** the
/// other five:
///
/// ```text
/// neighbours = popcount(I0..I4);  rand = I5;  fire if neighbours + rand >= threshold
/// ```
///
/// So the fabric as built supports **five neighbours** and **unweighted couplings**. The only
/// tunable per neuron is the integer threshold, which acts as its field. A spin glass cannot be
/// expressed on it, and no amount of LUT budget changes that — it is a property of the cell, not of
/// the placement.
///
/// Declaring this is the point. A fabric whose binding constraint is undocumented is how a third
/// party ends up discovering, from wrong answers, that a platform quantises to int8.
pub mod device {
    use ferrotherm::fabric::{Device, Fabric, Topology, Unsupported};
    use ferrotherm::ftp::Program;
    use ferrotherm::ledger::{Ledger, Z1_SPICE};
    use ferrotherm::schedule::Schedule;

    /// LUT6 inputs available to neighbours, after the random bit takes one.
    pub const NEIGHBOUR_INPUTS: usize = 5;
    /// LUT6 count on the XC7A100T.
    pub const LUT6_TOTAL: usize = 63_400;

    /// The Pt V2 board, as a backend.
    #[derive(Default)]
    pub struct PtV2 {
        program: Option<Program>,
        ledger: Ledger,
    }

    impl PtV2 {
        /// The declared capabilities, without needing a board attached.
        pub fn describe() -> Fabric {
            Fabric {
                name: "alchitry-pt-v2",
                topology: Topology::Degree(NEIGHBOUR_INPUTS),
                max_spins: Some(LUT6_TOTAL),
                max_degree: Some(NEIGHBOUR_INPUTS),
                // Not a precision limit but an absence of weighting; see `uniform_couplings`.
                coupling_precision: ferrotherm::fabric::Precision::Exact,
                // The threshold is an integer in 0..=6, so the field carries about three bits.
                field_precision: ferrotherm::fabric::Precision::Fixed { bits: 3 },
                supports_field: true,
                max_arity: 2,
                // One LUT per spin, addressed directly. Nothing is embedded.
                native_placement: true,
                unstated: &[],
                // The cell computes `popcount(I0..I4)` and fires when that plus a random bit
                // reaches the threshold. It COUNTS. A neighbour is present or absent, so the only
                // representable weights are 1 and 0 -- there is no negative coupling. Declaring
                // `integers(-1, 1)`, as this first did, claimed one the hardware cannot express
                // and would have accepted an antiferromagnet it silently cannot run.
                // `uniform_couplings` says a neighbouring part of the same fact; both are declared
                // because a caller checking one should not have to know to check the other.
                coupling_range: Some(ferrotherm::fabric::Range::integers(0.0, 1.0)),
                // The threshold is the field, an integer in 0..=6, and it is in THRESHOLD units
                // rather than the energy units the rest of the stack uses. A caller converts;
                // passing an Ising bias straight through would be a unit error the fabric cannot
                // see.
                field_range: Some(ferrotherm::fabric::Range::integers(0.0, 6.0)),
                uniform_couplings: true,
                prices: Z1_SPICE,
            }
        }

        /// Whether a board is attached and enumerating.
        ///
        /// False when the `flash` feature is off, which is the honest answer: without it this
        /// build cannot see a board even if one is plugged in.
        #[cfg(feature = "flash")]
        pub fn attached() -> bool {
            crate::flash::Ftdi::open("Alchitry").is_ok()
        }

        #[cfg(not(feature = "flash"))]
        pub fn attached() -> bool {
            false
        }
    }

    impl Device for PtV2 {
        fn fabric(&self) -> Fabric {
            Self::describe()
        }

        fn program(&mut self, p: &Program) -> Vec<Unsupported> {
            let bad = self.fabric().check(p);
            if bad.is_empty() {
                self.program = Some(p.clone());
            }
            bad
        }

        fn run(&mut self, _schedule: &Schedule, _seed: u64) -> Result<Vec<i8>, String> {
            if self.program.is_none() {
                return Err("no program loaded".into());
            }
            if !Self::attached() {
                return Err(
                    "no Pt V2 enumerating over USB. The bitstream path is complete and the fabric \
                     is emitted; running it needs the board present."
                        .into(),
                );
            }
            Err("the sequential fabric is not yet assembled: flip-flops and clock are pending, so \
                 there is nothing on the board to advance a sweep"
                .into())
        }

        fn ledger(&self) -> Ledger {
            self.ledger
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn prog(src: &str) -> Program {
            Program::from_ftp(src).unwrap()
        }

        #[test]
        fn the_declared_limits_match_the_cell_that_was_built() {
            let f = PtV2::describe();
            assert_eq!(f.max_degree, Some(5), "one LUT6 input is spent on the random bit");
            assert!(f.uniform_couplings, "the cell counts neighbours, it does not weight them");
            assert_eq!(f.max_arity, 2);
        }

        #[test]
        fn a_spin_glass_is_refused_with_the_reason() {
            // The limit that matters, and the one a LUT count would not reveal.
            let mut d = PtV2::default();
            let bad = d.program(&prog("ftp 1\nspins 3\nfactor 1 0 1\nfactor -1 1 2\n"));
            let u = bad
                .iter()
                .find(|u| matches!(u, Unsupported::NonUniformCouplings { .. }))
                .expect("mixed-sign couplings must be refused");
            assert!(u.to_string().contains("spin glass cannot be expressed"));
        }

        #[test]
        fn a_ferromagnet_within_the_degree_limit_is_accepted() {
            let mut d = PtV2::default();
            let src = "ftp 1\nspins 6\nfactor 1 0 1\nfactor 1 1 2\nfactor 1 2 3\nfactor 1 3 4\n\
                       factor 1 4 5\n";
            assert!(d.program(&prog(src)).is_empty(), "a uniform chain should fit");
        }

        #[test]
        fn degree_six_is_refused_because_the_rng_takes_an_input() {
            let mut d = PtV2::default();
            let mut src = String::from("ftp 1\nspins 8\n");
            for j in 1..=6 {
                src.push_str(&format!("factor 1 0 {j}\n"));
            }
            let bad = d.program(&prog(&src));
            assert!(bad.iter().any(|u| matches!(u, Unsupported::TooHighDegree { .. })),
                    "degree 6 must not fit in five neighbour inputs: {bad:?}");
        }

        #[test]
        fn running_without_a_board_says_which_thing_is_missing() {
            let mut d = PtV2::default();
            assert!(d.run(&Schedule::constant(1.0, 10), 1).unwrap_err().contains("no program"));
            d.program(&prog("ftp 1\nspins 3\nfactor 1 0 1\nfactor 1 1 2\n"));
            let e = d.run(&Schedule::constant(1.0, 10), 1).unwrap_err();
            assert!(e.contains("board") || e.contains("sequential fabric"), "{e}");
        }
    }
}


#[cfg(test)]
mod declared_capability_tests {
    use crate::device::PtV2;

    #[test]
    fn the_cell_cannot_express_a_negative_coupling() {
        // It computes popcount(I0..I4) and fires when that plus a random bit reaches the threshold.
        // It COUNTS: a neighbour is present or absent. Declaring integers(-1, 1), as this first
        // did, claimed a coupling the hardware has no way to represent — and would have accepted
        // an antiferromagnet it silently cannot run.
        let f = PtV2::describe();
        let r = f.coupling_range.expect("a counting fabric has a range");
        assert!(r.holds(1.0), "a neighbour can be present");
        assert!(r.holds(0.0), "or absent");
        assert!(!r.holds(-1.0), "and it cannot be negative");
        assert!(f.uniform_couplings, "which is the same fact said the other way");
    }

    #[test]
    fn an_antiferromagnet_is_refused_rather_than_accepted_and_mis_run() {
        use ferrotherm::ftp::Program;
        let anti = Program::from_ftp("ftp 1\nspins 3\nfactor -1 0 1\nfactor -1 1 2\n").unwrap();
        let bad = PtV2::describe().check(&anti);
        assert!(!bad.is_empty(), "a counting cell cannot run an antiferromagnet");
        assert!(
            bad.iter().any(|u| u.to_string().contains("integers 0..=1")),
            "and it says why: {bad:?}"
        );
    }
}
