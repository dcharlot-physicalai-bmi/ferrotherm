//! Vivado's logic-location file as an **outside witness** for the frame mapping.
//!
//! [`crate::tilegrid`] says of its own arithmetic that getting it wrong is the single most
//! dangerous failure in the whole path, because a well-formed bitstream written to the wrong
//! frames still passes CRC and still reads back correctly if the reader shares the bug. That is a
//! statement about what readback cannot settle: a reader built from the same table as the writer
//! will agree with it wherever it is wrong. Only a witness from outside the codebase can answer
//! the question, and Vivado publishes one.
//!
//! `write_bitstream -logic_location_file` emits a `.ll` listing every used configuration-memory
//! cell with the frame address Vivado itself placed it at:
//!
//! ```text
//!   Bit  88029176 0x0008cb0c 2072 SLR1 0 Block=SLICE_X104Y461 Latch=CQ Net=p_0_in5_in[6381]
//!        ^offset  ^frame     ^off  ^SLR  ^n ^the physical block
//! ```
//!
//! No popcount inference and no column walking: the frame address is stated. [`cross_check`]
//! takes such a listing and asks, for every slice Vivado names, whether this crate's tile grid
//! puts that slice in the same column — and reports every disagreement by name rather than a
//! count, because one slice in the wrong column is the whole defect.
//!
//! # What this does not do
//!
//! It checks the **column**, not the bit. A `.ll` locates individual cells, and matching those
//! exactly would require modelling every site's bit layout within a frame; the column is the part
//! [`crate::tilegrid`] derives and therefore the part that can be wrong. A design that agrees
//! here can still have its segbits wrong, and that is a separate witness.

use crate::tilegrid::{Far, TileGrid};

/// One configuration-memory cell, located by Vivado.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlBit {
    /// The frame address Vivado placed this cell in.
    pub frame_address: u32,
    /// The bit's offset within that frame.
    pub frame_offset: u32,
    /// Super-logic region, for parts that have more than one die.
    pub slr: String,
    /// The physical block, for example `SLICE_X104Y461`.
    pub block: String,
}

impl LlBit {
    /// The column this cell's frame belongs to: the same address with its minor field cleared.
    ///
    /// Built through [`Far`] rather than by masking a constant. The equivalent code for an
    /// `UltraScale+` part clears `0xFF`, because its minor field is eight bits wide; a 7-series
    /// minor is **seven** ([`Far::decode`] reads `far & 0x7F`), so carrying the wider mask across
    /// families would clear the low bit of the column field and quietly merge two adjacent
    /// columns into one base address. The cross-check would then agree in exactly the cases it
    /// exists to catch.
    #[must_use]
    pub fn column_base(&self) -> u32 {
        let mut far = Far::decode(self.frame_address);
        far.minor = 0;
        far.encode()
    }
}

/// One slice where Vivado and this crate's tile grid disagree about the column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disagreement {
    /// The slice Vivado named.
    pub block: String,
    /// The column base Vivado states.
    pub vivado: u32,
    /// The column base this crate's tile grid derives.
    pub ours: u32,
}

/// What the cross-check found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrossCheck {
    /// Slices where the two agree.
    pub agreed: usize,
    /// Slices where they do not, named.
    pub disagreed: Vec<Disagreement>,
    /// Slices Vivado named that this grid does not contain — a different part, or a site kind
    /// the grid was not loaded for. Counted, not treated as agreement.
    pub unknown: usize,
}

impl CrossCheck {
    /// Whether every slice the grid knows agreed with Vivado.
    ///
    /// An empty listing is not a pass: with nothing checked there is nothing to agree, and a
    /// cross-check that reports success on no evidence is the failure it was written to prevent.
    #[must_use]
    pub fn verified(&self) -> bool {
        self.agreed > 0 && self.disagreed.is_empty()
    }
}

/// Parse `.ll` text into the cells it locates, keeping only `Bit` lines that name a block.
///
/// Comment lines, `Info` lines and cells with no `Block=` are skipped: a cell Vivado did not
/// attribute to a physical block says nothing about placement.
#[must_use]
pub fn parse_ll(text: &str) -> Vec<LlBit> {
    let mut out = Vec::new();
    for line in text.lines() {
        if !line.starts_with("Bit") {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        // Bit <offset> <frame address> <frame offset> <SLR name> <SLR number> <key=value>...
        if cols.len() < 6 {
            continue;
        }
        let Some(frame_address) = cols[2]
            .strip_prefix("0x")
            .and_then(|h| u32::from_str_radix(h, 16).ok())
        else {
            continue;
        };
        let Ok(frame_offset) = cols[3].parse::<u32>() else {
            continue;
        };
        let mut block = None;
        for field in &cols[6..] {
            if let Some(name) = field.strip_prefix("Block=") {
                block = Some(name.to_string());
            }
        }
        let Some(block) = block else { continue };
        out.push(LlBit {
            frame_address,
            frame_offset,
            slr: cols[4].to_string(),
            block,
        });
    }
    out
}

/// Ask Vivado's listing whether this crate's tile grid places each named slice in the same column.
///
/// Every distinct slice is checked once; a `.ll` names the same slice on many lines, and counting
/// them all would report a single well-placed slice as hundreds of agreements.
#[must_use]
pub fn cross_check(grid: &TileGrid, bits: &[LlBit]) -> CrossCheck {
    let mut seen: Vec<&str> = Vec::new();
    let mut out = CrossCheck::default();
    for bit in bits {
        if seen.contains(&bit.block.as_str()) {
            continue;
        }
        seen.push(&bit.block);
        let ours = grid
            .tile_of_site(&bit.block)
            .and_then(crate::tilegrid::Tile::logic_block)
            .map(|b| b.baseaddr);
        let Some(ours) = ours else {
            out.unknown += 1;
            continue;
        };
        let vivado = bit.column_base();
        if ours == vivado {
            out.agreed += 1;
        } else {
            out.disagreed.push(Disagreement {
                block: bit.block.clone(),
                vivado,
                ours,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fragment in the documented shape, including the lines that must be skipped.
    const SAMPLE: &str = "\
; Created by Vivado 2025.2 SW Build 6299465
; Bit lines have the following form:
Bit   88029176 0x0008cb0c 2072 SLR1 0 Block=SLICE_X0Y0 Latch=CQ Net=p_0_in5_in[6381]
Bit   88029178 0x0008cb0c 2074 SLR1 0 Block=SLICE_X0Y0 Latch=DQ Net=p_0_in3_in[6380]
Bit   88029188 0x0008cb8c 2084 SLR1 0 Block=SLICE_X1Y0 Latch=CQ2
Bit   88029190 0x0008cb0c 2086 SLR1 0 Latch=DQ2 Net=unattributed
Info   something else entirely
";

    fn grid_json(base_a: &str, base_b: &str) -> String {
        format!(
            r#"{{
          "CLBLL_L_X0Y0": {{"type": "CLBLL_L", "grid_x": 0, "grid_y": 0,
            "bits": {{"CLB_IO_CLK": {{"baseaddr": "{base_a}", "frames": 36, "offset": 0, "words": 2}}}},
            "sites": {{"SLICE_X0Y0": "SLICEL"}}}},
          "CLBLL_L_X1Y0": {{"type": "CLBLL_L", "grid_x": 1, "grid_y": 0,
            "bits": {{"CLB_IO_CLK": {{"baseaddr": "{base_b}", "frames": 36, "offset": 0, "words": 2}}}},
            "sites": {{"SLICE_X1Y0": "SLICEL"}}}}}}"#
        )
    }

    #[test]
    fn parses_the_documented_line_shape_and_skips_the_rest() {
        let bits = parse_ll(SAMPLE);
        assert_eq!(bits.len(), 3, "comments, Info lines and unattributed cells are skipped");
        assert_eq!(bits[0].frame_address, 0x0008_cb0c);
        assert_eq!(bits[0].frame_offset, 2072);
        assert_eq!(bits[0].slr, "SLR1");
        assert_eq!(bits[0].block, "SLICE_X0Y0");
        assert_eq!(bits[2].block, "SLICE_X1Y0");
    }

    /// The column base clears the 7-series minor field and nothing else. `0x0008cb0c` has minor
    /// `0x0c`; clearing seven bits leaves `0x0008cb00`, and clearing eight -- the `UltraScale+`
    /// width -- would leave `0x0008cb00` too here but take the column's low bit whenever the
    /// address has bit 7 set, which `0x0008cb8c` does.
    #[test]
    fn the_column_base_clears_seven_bits_of_minor_not_eight() {
        let low = LlBit {
            frame_address: 0x0008_cb0c,
            frame_offset: 0,
            slr: "SLR1".into(),
            block: "SLICE_X0Y0".into(),
        };
        let high = LlBit {
            frame_address: 0x0008_cb8c,
            frame_offset: 0,
            slr: "SLR1".into(),
            block: "SLICE_X1Y0".into(),
        };
        assert_eq!(low.column_base(), 0x0008_cb00);
        assert_eq!(high.column_base(), 0x0008_cb80, "bit 7 is column, not minor");
        assert_ne!(
            high.column_base(),
            high.frame_address & !0xFF,
            "the UltraScale+ mask would merge this column with its neighbour"
        );
        assert_eq!(Far::decode(high.column_base()).minor, 0);
    }

    /// Agreement is reported per distinct slice, and a disagreement names the slice.
    #[test]
    fn the_cross_check_names_the_slice_that_disagrees() {
        let grid = TileGrid::parse(&grid_json("0x0008cb00", "0x0008cb80")).unwrap();
        let ok = cross_check(&grid, &parse_ll(SAMPLE));
        assert_eq!(ok.agreed, 2, "two distinct slices, though one is named twice");
        assert!(ok.disagreed.is_empty());
        assert_eq!(ok.unknown, 0);
        assert!(ok.verified());

        // Put the second column one frame address out, which is what a walker off by a column
        // produces, and the check must say which slice and both addresses.
        let grid = TileGrid::parse(&grid_json("0x0008cb00", "0x0008cc00")).unwrap();
        let bad = cross_check(&grid, &parse_ll(SAMPLE));
        assert_eq!(bad.agreed, 1);
        assert_eq!(bad.disagreed.len(), 1);
        assert_eq!(bad.disagreed[0].block, "SLICE_X1Y0");
        assert_eq!(bad.disagreed[0].vivado, 0x0008_cb80);
        assert_eq!(bad.disagreed[0].ours, 0x0008_cc00);
        assert!(!bad.verified());
    }

    /// An empty listing is not a pass: nothing was checked, so nothing is verified.
    #[test]
    fn an_empty_listing_verifies_nothing() {
        let grid = TileGrid::parse(&grid_json("0x0008cb00", "0x0008cb80")).unwrap();
        let none = cross_check(&grid, &[]);
        assert_eq!(none.agreed, 0);
        assert!(!none.verified(), "a check with no evidence must not report success");
        // A slice this grid has never heard of is counted as unknown, not as agreement.
        let elsewhere = parse_ll(
            "Bit   1 0x0008cb0c 2 SLR1 0 Block=SLICE_X99Y99 Latch=CQ\n",
        );
        let out = cross_check(&grid, &elsewhere);
        assert_eq!(out.unknown, 1);
        assert_eq!(out.agreed, 0);
        assert!(!out.verified());
    }
}
