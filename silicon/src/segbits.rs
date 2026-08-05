//! Project X-Ray segment-bit database parsing, and the address arithmetic that turns a feature
//! name into a physical position inside a configuration frame.
//!
//! The database is open data (prjxray-db, ISC licence) fetched from upstream — no third-party
//! software is involved, only a documented text format:
//!
//! ```text
//! CLBLL_L.SLICEL_X0.ALUT.INIT[00] 32_15
//! CLBLL_L.SLICEL_X0.AFFMUX.AX !30_00 30_01 !30_02 !30_03
//! ```
//!
//! Each line is a feature followed by the bits that define it. `FF_BB` names frame offset FF
//! within the tile's frame range and bit offset BB within the tile's words. A leading `!` means
//! the bit must be CLEAR for the feature to hold — dropping those inverted bits is a silent way
//! to emit a different feature than intended, so they are represented explicitly.

use std::collections::HashMap;

/// One bit of a feature: an offset pair plus the value it must take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegBit {
    /// Frame offset within the tile's frame range.
    pub frame: u16,
    /// Bit offset within the tile's words (bit 32k..32k+31 lives in word k).
    pub bit: u16,
    /// Whether the bit is set (`true`) or must be clear (`!` prefix -> `false`).
    pub set: bool,
}

impl SegBit {
    /// The word this bit lives in, and its position within that word.
    pub fn word_and_pos(&self) -> (u16, u16) {
        (self.bit / 32, self.bit % 32)
    }
}

/// A parsed segbits database: feature name -> the bits that define it.
#[derive(Debug, Clone, Default)]
pub struct SegBits {
    pub features: HashMap<String, Vec<SegBit>>,
}

/// Parse one `FF_BB` (optionally `!`-prefixed) coordinate.
pub fn parse_bit(tok: &str) -> Option<SegBit> {
    let (set, body) = match tok.strip_prefix('!') {
        Some(rest) => (false, rest),
        None => (true, tok),
    };
    let (f, b) = body.split_once('_')?;
    Some(SegBit { frame: f.parse().ok()?, bit: b.parse().ok()?, set })
}

impl SegBits {
    /// Parse a whole `.db` file. Lines that carry no coordinates (e.g. `<const0>` placeholders
    /// prjxray emits for always-on features) are kept with an empty bit list rather than
    /// dropped, so a lookup can tell "no bits needed" from "feature unknown".
    pub fn parse(text: &str) -> SegBits {
        let mut features = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let Some(name) = it.next() else { continue };
            let bits: Vec<SegBit> = it.filter_map(parse_bit).collect();
            features.insert(name.to_string(), bits);
        }
        SegBits { features }
    }

    pub fn get(&self, feature: &str) -> Option<&[SegBit]> {
        self.features.get(feature).map(|v| v.as_slice())
    }

    /// Collect a LUT's 64 INIT bits in index order, e.g. prefix
    /// `CLBLL_L.SLICEL_X0.ALUT` -> the bits for `INIT[00]` ..= `INIT[63]`.
    /// Returns `None` if any index is missing, because a partial map would silently write a
    /// truth table with holes in it.
    pub fn lut_init_bits(&self, lut_prefix: &str) -> Option<Vec<SegBit>> {
        let mut out = Vec::with_capacity(64);
        for i in 0..64 {
            let key = format!("{lut_prefix}.INIT[{i:02}]");
            let bits = self.features.get(&key)?;
            // an INIT bit is defined by exactly one set coordinate
            let b = bits.iter().find(|b| b.set)?;
            out.push(*b);
        }
        Some(out)
    }

    /// Every feature name sharing a prefix (for discovery/inspection).
    pub fn with_prefix<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.features.keys().filter(move |k| k.starts_with(prefix)).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real lines from the Artix-7 database (prjxray-db, ISC) — the format under test.
    const SAMPLE: &str = "\
CLBLL_L.SLICEL_X0.A5FF.ZINI 31_06
CLBLL_L.SLICEL_X0.A5FFMUX.IN_A 30_09
CLBLL_L.SLICEL_X0.AFFMUX.AX !30_00 30_01 !30_02 !30_03
CLBLL_L.SLICEL_X0.AFFMUX.CY 30_00 !30_01 30_02 !30_03
CLBLL_L.SLICEL_X0.ALUT.INIT[00] 32_15
CLBLL_L.SLICEL_X0.ALUT.INIT[01] 33_15
CLBLL_L.SLICEL_X0.ALUT.INIT[02] 32_14
CLBLL_L.SLICEL_X0.ALUT.INIT[03] 33_14
";

    #[test]
    fn parses_single_and_multi_bit_features() {
        let db = SegBits::parse(SAMPLE);
        assert_eq!(
            db.get("CLBLL_L.SLICEL_X0.A5FF.ZINI"),
            Some(&[SegBit { frame: 31, bit: 6, set: true }][..])
        );
        let ax = db.get("CLBLL_L.SLICEL_X0.AFFMUX.AX").unwrap();
        assert_eq!(ax.len(), 4);
        // the inverted bits must survive parsing — dropping them would emit a different mux
        assert_eq!(ax.iter().filter(|b| b.set).count(), 1);
        assert_eq!(ax[0], SegBit { frame: 30, bit: 0, set: false });
        assert_eq!(ax[1], SegBit { frame: 30, bit: 1, set: true });
    }

    /// Two mux settings that share coordinates must differ in polarity, not position — the
    /// property that makes the `!` prefix load-bearing.
    #[test]
    fn mux_settings_differ_only_in_polarity() {
        let db = SegBits::parse(SAMPLE);
        let ax = db.get("CLBLL_L.SLICEL_X0.AFFMUX.AX").unwrap();
        let cy = db.get("CLBLL_L.SLICEL_X0.AFFMUX.CY").unwrap();
        let pos = |v: &[SegBit]| v.iter().map(|b| (b.frame, b.bit)).collect::<Vec<_>>();
        assert_eq!(pos(ax), pos(cy));
        assert_ne!(
            ax.iter().map(|b| b.set).collect::<Vec<_>>(),
            cy.iter().map(|b| b.set).collect::<Vec<_>>()
        );
    }

    #[test]
    fn word_split_is_32_bits() {
        assert_eq!(SegBit { frame: 0, bit: 15, set: true }.word_and_pos(), (0, 15));
        assert_eq!(SegBit { frame: 0, bit: 32, set: true }.word_and_pos(), (1, 0));
        assert_eq!(SegBit { frame: 0, bit: 63, set: true }.word_and_pos(), (1, 31));
    }

    /// An incomplete INIT map must be refused rather than returned with holes.
    #[test]
    fn partial_lut_init_is_rejected() {
        let db = SegBits::parse(SAMPLE);
        assert!(db.lut_init_bits("CLBLL_L.SLICEL_X0.ALUT").is_none());
    }

    /// A complete 64-entry map is accepted and ordered by INIT index.
    #[test]
    fn complete_lut_init_is_ordered() {
        let mut text = String::new();
        for i in 0..64 {
            // mirror the real interleave: even indices in frame 32, odd in 33, bit walking down
            let frame = 32 + (i % 2);
            let bit = 15 - (i / 2).min(15);
            text.push_str(&format!("T.SLICEL_X0.ALUT.INIT[{i:02}] {frame}_{bit:02}\n"));
        }
        let db = SegBits::parse(&text);
        let bits = db.lut_init_bits("T.SLICEL_X0.ALUT").expect("complete map");
        assert_eq!(bits.len(), 64);
        assert_eq!(bits[0], SegBit { frame: 32, bit: 15, set: true });
        assert_eq!(bits[1], SegBit { frame: 33, bit: 15, set: true });
        assert_eq!(bits[2], SegBit { frame: 32, bit: 14, set: true });
    }
}
