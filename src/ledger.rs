//! The device energy ledger — first-class, because on this hardware class the story IS the I/O.
//!
//! Prices are per-node-operation costs of a device model. For the Z1-class costs (SPICE-derived,
//! pre-silicon; arXiv:2608.01615 Table IV) a WRITE costs ~21,700 Gibbs cycles and a READ ~239.
//! The vendor's own conclusion follows from these three numbers: the architecture wins where
//! "many local updates are performed between infrequent I/O operations" — and the ledger makes
//! that arithmetic executable instead of promotional.

/// Per-operation energy prices, in joules. These describe a DEVICE MODEL, not measured silicon,
/// unless the source says otherwise; keep the provenance in the name.
#[derive(Clone, Copy, Debug)]
pub struct Prices {
    /// One Gibbs update of one node.
    pub e_sample: f64,
    /// One node value read out to the chip edge.
    pub e_read: f64,
    /// One node's couplings/bias/clamp state flashed.
    pub e_write: f64,
    /// Maximum sustainable full-graph reflash rate, Hz (None = unstated).
    pub reflash_hz_cap: Option<f64>,
}

/// Z1-class prices from arXiv:2608.01615 Table IV (SPICE estimates for taped-out, uncharacterized
/// silicon; "measured" in the paper's prose is a misnomer the appendix itself contradicts).
pub const Z1_SPICE: Prices = Prices {
    e_sample: 7.09e-15,
    e_read: 1.692e-12,
    e_write: 153.6e-12,
    reflash_hz_cap: Some(1.0),
};

/// Operation counts accumulated by a run.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ledger {
    pub samples: u64,
    pub reads: u64,
    pub writes: u64,
}

impl Ledger {
    pub fn joules(&self, p: &Prices) -> f64 {
        self.samples as f64 * p.e_sample + self.reads as f64 * p.e_read + self.writes as f64 * p.e_write
    }

    /// Fractional breakdown (sample, read, write) of total energy under prices `p`.
    pub fn shares(&self, p: &Prices) -> (f64, f64, f64) {
        let s = self.samples as f64 * p.e_sample;
        let r = self.reads as f64 * p.e_read;
        let w = self.writes as f64 * p.e_write;
        let t = (s + r + w).max(1e-300);
        (s / t, r / t, w / t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_to_sample_ratio_is_the_finding() {
        // The structural number behind the robotics verdict: one write = how many samples?
        let ratio = Z1_SPICE.e_write / Z1_SPICE.e_sample;
        assert!((ratio - 21664.0).abs() < 100.0, "write/sample = {ratio}");
        let rr = Z1_SPICE.e_read / Z1_SPICE.e_sample;
        assert!((rr - 238.6).abs() < 5.0, "read/sample = {rr}");
    }
}
