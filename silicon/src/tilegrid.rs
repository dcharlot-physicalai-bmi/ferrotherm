//! The fabric map: which physical frame, word and bit a tile's configuration lives in.
//!
//! Source data is prjxray-db's `tilegrid.json` (open data, ISC licence, fetched from upstream).
//! Each tile carries one or more configuration blocks:
//!
//! ```json
//! "CLBLL_L_X2Y0": { "type": "CLBLL_L",
//!   "bits": { "CLB_IO_CLK": { "baseaddr": "0x00420100", "frames": 36, "offset": 0, "words": 2 } },
//!   "sites": { "SLICE_X0Y0": "SLICEL", "SLICE_X1Y0": "SLICEL" } }
//! ```
//!
//! With a segbit `FF_BB` from [`crate::segbits`], the physical position is
//!
//! ```text
//! frame address = baseaddr + FF          (FF indexes the column's frames, 0..frames)
//! word in frame = offset + BB / 32       (offset places the tile inside the frame's words)
//! bit in word   = BB % 32
//! ```
//!
//! Getting this wrong is the single most dangerous failure in the whole path, because a
//! well-formed bitstream written to the wrong frames still passes CRC and still "reads back
//! correctly" if the reader shares the bug. Physical placement is checked against an outside
//! witness, never against our own readback.

use crate::json::{parse, Json};
use crate::segbits::SegBit;
use std::collections::HashMap;

/// One configuration block of a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitsBlock {
    pub baseaddr: u32,
    pub frames: u16,
    pub offset: u16,
    pub words: u16,
}

#[derive(Debug, Clone)]
pub struct Tile {
    pub name: String,
    pub kind: String,
    pub grid_x: u32,
    pub grid_y: u32,
    /// block name (e.g. "CLB_IO_CLK", "BLOCK_RAM") -> address block
    pub bits: Vec<(String, BitsBlock)>,
    /// site name -> site type (e.g. SLICE_X0Y0 -> SLICEL)
    pub sites: Vec<(String, String)>,
}

impl Tile {
    /// The tile's primary configuration block (CLB/IO/CLK logic), if it has one.
    pub fn logic_block(&self) -> Option<BitsBlock> {
        self.bits
            .iter()
            .find(|(n, _)| n == "CLB_IO_CLK")
            .map(|(_, b)| *b)
    }
}

/// A physical configuration-bit position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitAddr {
    pub frame: u32,
    pub word: u16,
    pub bit: u16,
    pub set: bool,
}

impl BitsBlock {
    /// Resolve a tile-relative segbit to its physical position. Returns `None` when the segbit
    /// falls outside the block's declared extent — a mismatched database and part must fail
    /// loudly rather than write into a neighbouring tile.
    pub fn resolve(&self, sb: SegBit) -> Option<BitAddr> {
        if sb.frame >= self.frames {
            return None;
        }
        let (word_off, pos) = sb.word_and_pos();
        if word_off >= self.words {
            return None;
        }
        Some(BitAddr {
            frame: self.baseaddr + sb.frame as u32,
            word: self.offset + word_off,
            bit: pos,
            set: sb.set,
        })
    }
}

#[derive(Debug, Default)]
pub struct TileGrid {
    pub tiles: HashMap<String, Tile>,
}

impl TileGrid {
    pub fn parse(text: &str) -> Result<TileGrid, String> {
        let j = parse(text)?;
        let mut tiles = HashMap::new();
        for (name, v) in j.entries() {
            let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("").to_string();
            let grid_x = v.get("grid_x").and_then(|t| t.as_u64()).unwrap_or(0) as u32;
            let grid_y = v.get("grid_y").and_then(|t| t.as_u64()).unwrap_or(0) as u32;
            let mut bits = Vec::new();
            if let Some(Json::Obj(blocks)) = v.get("bits") {
                for (bname, b) in blocks {
                    let base = b
                        .get("baseaddr")
                        .and_then(|x| x.as_str())
                        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok());
                    if let Some(baseaddr) = base {
                        bits.push((
                            bname.to_string(),
                            BitsBlock {
                                baseaddr,
                                frames: b.get("frames").and_then(|x| x.as_u64()).unwrap_or(0) as u16,
                                offset: b.get("offset").and_then(|x| x.as_u64()).unwrap_or(0) as u16,
                                words: b.get("words").and_then(|x| x.as_u64()).unwrap_or(0) as u16,
                            },
                        ));
                    }
                }
            }
            let mut sites = Vec::new();
            if let Some(Json::Obj(ss)) = v.get("sites") {
                for (sname, sty) in ss {
                    sites.push((sname.to_string(), sty.as_str().unwrap_or("").to_string()));
                }
            }
            tiles.insert(
                name.to_string(),
                Tile { name: name.to_string(), kind, grid_x, grid_y, bits, sites },
            );
        }
        Ok(TileGrid { tiles })
    }

    /// Find the tile containing a given site (e.g. "SLICE_X0Y0").
    pub fn tile_of_site(&self, site: &str) -> Option<&Tile> {
        self.tiles.values().find(|t| t.sites.iter().any(|(n, _)| n == site))
    }

    pub fn of_kind<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Tile> + 'a {
        self.tiles.values().filter(move |t| t.kind == kind)
    }
}

/// Decode a 7-series frame address (UG470): block type, half, row, column, minor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Far {
    pub block_type: u8,
    pub bottom_half: bool,
    pub row: u8,
    pub column: u16,
    pub minor: u8,
}

impl Far {
    pub fn decode(far: u32) -> Far {
        Far {
            block_type: ((far >> 23) & 0x7) as u8,
            bottom_half: (far >> 22) & 1 == 1,
            row: ((far >> 17) & 0x1F) as u8,
            column: ((far >> 7) & 0x3FF) as u16,
            minor: (far & 0x7F) as u8,
        }
    }
    pub fn encode(&self) -> u32 {
        ((self.block_type as u32 & 0x7) << 23)
            | ((self.bottom_half as u32) << 22)
            | ((self.row as u32 & 0x1F) << 17)
            | ((self.column as u32 & 0x3FF) << 7)
            | (self.minor as u32 & 0x7F)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = r#"{"CLBLL_L_X2Y0": {"bits": {"CLB_IO_CLK": {"baseaddr": "0x00420100",
      "frames": 36, "offset": 0, "words": 2}}, "clock_region": "X0Y0", "grid_x": 10,
      "grid_y": 207, "sites": {"SLICE_X0Y0": "SLICEL", "SLICE_X1Y0": "SLICEL"},
      "type": "CLBLL_L"}}"#;

    #[test]
    fn loads_a_real_tile() {
        let g = TileGrid::parse(REAL).unwrap();
        let t = g.tiles.get("CLBLL_L_X2Y0").unwrap();
        assert_eq!(t.kind, "CLBLL_L");
        assert_eq!(t.sites.len(), 2);
        let b = t.logic_block().unwrap();
        assert_eq!(b, BitsBlock { baseaddr: 0x0042_0100, frames: 36, offset: 0, words: 2 });
        assert_eq!(g.tile_of_site("SLICE_X1Y0").unwrap().name, "CLBLL_L_X2Y0");
    }

    /// The address arithmetic, worked by hand: INIT[00] of SLICEL_X0's A-LUT sits at segbit
    /// 32_15, so in this tile it is frame baseaddr+32, word 0, bit 15.
    #[test]
    fn resolves_a_segbit_to_a_physical_position() {
        let g = TileGrid::parse(REAL).unwrap();
        let b = g.tiles["CLBLL_L_X2Y0"].logic_block().unwrap();
        let a = b.resolve(SegBit { frame: 32, bit: 15, set: true }).unwrap();
        assert_eq!(a, BitAddr { frame: 0x0042_0120, word: 0, bit: 15, set: true });
        // a bit in the tile's second word
        let a2 = b.resolve(SegBit { frame: 0, bit: 40, set: true }).unwrap();
        assert_eq!(a2, BitAddr { frame: 0x0042_0100, word: 1, bit: 8, set: true });
    }

    /// Out-of-extent segbits must be refused, not silently folded into a neighbouring tile.
    #[test]
    fn out_of_range_segbits_are_refused() {
        let g = TileGrid::parse(REAL).unwrap();
        let b = g.tiles["CLBLL_L_X2Y0"].logic_block().unwrap();
        assert!(b.resolve(SegBit { frame: 36, bit: 0, set: true }).is_none());
        assert!(b.resolve(SegBit { frame: 0, bit: 64, set: true }).is_none());
    }


    /// THE MULTI-WORD TRAP. prjxray's second coordinate is a bit index across the tile's whole
    /// word window, not a bit within one word. The naive reading (word = offset, bit = MM)
    /// happens to agree for 2-word tiles like a CLB, and silently writes every multi-word tile
    /// (BRAM words=10, IOB words=4, CFG_CENTER_MID words=101) into the wrong word. The correct
    /// arithmetic is offset*32 + MM, then split — which is what `resolve` computes.
    #[test]
    fn multi_word_tiles_split_correctly() {
        // a BRAM-shaped block: 10 words starting at word offset 20
        let b = BitsBlock { baseaddr: 0x0000_1000, frames: 28, offset: 20, words: 10 };
        // MM = 100 -> absolute bit 20*32 + 100 = 740 -> word 23, bit 4
        let a = b.resolve(SegBit { frame: 5, bit: 100, set: true }).unwrap();
        assert_eq!(a, BitAddr { frame: 0x0000_1005, word: 23, bit: 4, set: true });
        // the naive reading would have said word 20, bit 100 (which is not even a valid bit)
        assert_ne!(a.word, b.offset);

        // MM well beyond 63 must work too (cfg_center_mid reaches 2213)
        let wide = BitsBlock { baseaddr: 0x0000_2000, frames: 4, offset: 0, words: 101 };
        let a2 = wide.resolve(SegBit { frame: 0, bit: 2213, set: true }).unwrap();
        assert_eq!((a2.word, a2.bit), (2213 / 32, 2213 % 32));
        assert_eq!((a2.word, a2.bit), (69, 5));
    }

    /// INT and CLB tiles share a frame column (same baseaddr/offset/words), separated only by
    /// minor range. Set-bits must therefore OR together rather than overwrite: writing CLB
    /// control bits must not clobber routing already placed in minors 0-1.
    #[test]
    fn int_and_clb_share_a_column_without_clobbering() {
        let shared = BitsBlock { baseaddr: 0x0042_0600, frames: 36, offset: 0, words: 2 };
        let int_bit = shared.resolve(SegBit { frame: 1, bit: 39, set: true }).unwrap();
        let clb_bit = shared.resolve(SegBit { frame: 1, bit: 40, set: true }).unwrap();
        assert_eq!(int_bit.frame, clb_bit.frame, "same physical frame");
        assert_ne!((int_bit.word, int_bit.bit), (clb_bit.word, clb_bit.bit), "disjoint positions");
        let mut fb = crate::framebuf::FrameBuf::new();
        fb.apply(int_bit).unwrap();
        fb.apply(clb_bit).unwrap();
        let f = &fb.frames[&int_bit.frame];
        assert_eq!(f[int_bit.word as usize] >> int_bit.bit & 1, 1, "routing bit survived");
        assert_eq!(f[clb_bit.word as usize] >> clb_bit.bit & 1, 1, "logic bit landed");
    }

    /// FAR round-trip, and the decode of a real base address.
    #[test]
    fn far_round_trip() {
        let f = Far::decode(0x0042_0100);
        assert_eq!(f.block_type, 0);
        assert_eq!(f.bottom_half, true);
        assert_eq!(f.row, 1);
        assert_eq!(f.column, 2);
        assert_eq!(f.minor, 0);
        assert_eq!(f.encode(), 0x0042_0100);
        for far in [0u32, 0x0042_0120, 0x0080_1234, 0x00C0_0000] {
            assert_eq!(Far::decode(far).encode(), far, "round trip 0x{far:08X}");
        }
    }
}
