// Load a real prjxray tilegrid and report what the fabric looks like, then resolve a LUT's
// INIT bits to physical frame positions. Integration check for the parser + address arithmetic
// against the full-size database (11.7 MB, ~31k tiles for the XC7A100T).
//
// usage: cargo run -p ferrotherm-silicon --release --example gridinfo -- tilegrid.json [segbits.db]
use ferrotherm_silicon::segbits::SegBits;
use ferrotherm_silicon::tilegrid::{Far, TileGrid};
use std::collections::BTreeMap;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let grid_path = args.next().expect("usage: gridinfo <tilegrid.json> [segbits.db]");
    let text = std::fs::read_to_string(&grid_path).expect("read tilegrid");
    let t0 = Instant::now();
    let grid = TileGrid::parse(&text).expect("parse tilegrid");
    let dt = t0.elapsed();
    println!("parsed {:.1} MB, {} tiles in {:.2} s", text.len() as f64 / 1e6, grid.tiles.len(), dt.as_secs_f64());

    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    let mut with_bits = 0usize;
    let mut slices = 0usize;
    for t in grid.tiles.values() {
        *by_kind.entry(t.kind.as_str()).or_default() += 1;
        if t.logic_block().is_some() {
            with_bits += 1;
        }
        slices += t.sites.iter().filter(|(_, ty)| ty.starts_with("SLICE")).count();
    }
    let mut top: Vec<_> = by_kind.iter().collect();
    top.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    println!("tile kinds: {:?}", &top[..top.len().min(6)]);
    println!("tiles with a logic config block: {with_bits}; SLICE sites: {slices}");
    // OUTSIDE WITNESS: the XC7A100T datasheet says 63,400 LUT6, and a 7-series SLICE holds 4.
    // If the grid parse were wrong the slice count would not land on the published number.
    let datasheet_luts = 63_400usize;
    println!(
        "datasheet cross-check: {} slices x 4 = {} LUT6 vs datasheet {} -> {}",
        slices, slices * 4, datasheet_luts,
        if slices * 4 == datasheet_luts { "MATCH" } else { "MISMATCH - grid parse is wrong" }
    );

    // frame-address extent of the logic blocks
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    for t in grid.tiles.values() {
        if let Some(b) = t.logic_block() {
            lo = lo.min(b.baseaddr);
            hi = hi.max(b.baseaddr + b.frames as u32 - 1);
        }
    }
    println!("logic frame addresses: 0x{lo:08X} .. 0x{hi:08X}");
    println!("  low  = {:?}", Far::decode(lo));
    println!("  high = {:?}", Far::decode(hi));

    // resolve a real LUT's INIT bits if a segbits db was given
    if let Some(seg_path) = args.next() {
        let db = SegBits::parse(&std::fs::read_to_string(&seg_path).expect("read segbits"));
        let tile = grid
            .of_kind("CLBLL_L")
            .filter(|t| t.logic_block().is_some())
            .min_by_key(|t| (t.grid_y, t.grid_x))
            .expect("a CLBLL_L tile");
        let block = tile.logic_block().unwrap();
        let prefix = "CLBLL_L.SLICEL_X0.ALUT";
        match db.lut_init_bits(prefix) {
            Some(bits) => {
                let resolved: Vec<_> = bits.iter().filter_map(|b| block.resolve(*b)).collect();
                println!(
                    "\n{} in {} ({} sites): {}/64 INIT bits resolve",
                    prefix, tile.name, tile.sites.len(), resolved.len()
                );
                for (i, a) in resolved.iter().take(4).enumerate() {
                    println!("  INIT[{i:02}] -> frame 0x{:08X} word {} bit {}", a.frame, a.word, a.bit);
                }
                let frames: std::collections::BTreeSet<u32> = resolved.iter().map(|a| a.frame).collect();
                println!("  spans {} distinct frames, words {:?}", frames.len(),
                         resolved.iter().map(|a| a.word).collect::<std::collections::BTreeSet<_>>());
                println!("verdict: {}", if resolved.len() == 64 {
                    "all 64 truth-table bits have physical addresses — the map is complete"
                } else {
                    "INCOMPLETE map — refusing to trust it"
                });
            }
            None => println!("\n{prefix}: incomplete INIT map in this database"),
        }
    }
}
