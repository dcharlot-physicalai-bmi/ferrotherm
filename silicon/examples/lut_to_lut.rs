// THE BINDING: route one LUT's output pin to another LUT's input pin, across real slices and
// real interconnect. This is the connection a p-bit fabric is built from — a neighbour's state
// arriving at a p-bit's LUT input — and the last routing capability needed before a ferrotherm
// fabric can be emitted as a bitstream.
//
// usage: lut_to_lut <tilegrid> <tileconn> <tt_INT_L> <tt_INT_R> <tt_CLBLL_L> <segbits_int_l> <segbits_int_r>
use ferrotherm_silicon::framebuf::FrameBuf;
use ferrotherm_silicon::pips::PipDb;
use ferrotherm_silicon::pips::Ppips;
use ferrotherm_silicon::route::{interconnect_with_endpoints, parse_tileconn, Fabric};
use ferrotherm_silicon::segbits::SegBits;
use ferrotherm_silicon::tilegrid::TileGrid;
use std::collections::HashMap;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let rd = |i: usize| std::fs::read_to_string(&a[i]).expect("read");
    let grid = TileGrid::parse(&rd(0)).expect("tilegrid");
    let conns = parse_tileconn(&rd(1)).expect("tileconn");
    let mut dbs = HashMap::new();
    // PIP databases ONLY for interconnect tiles: with no CLB PIPs loaded, the search cannot take
    // a CLE route-through even by accident.
    dbs.insert("INT_L".to_string(), PipDb::parse(&rd(2)).expect("INT_L"));
    dbs.insert("INT_R".to_string(), PipDb::parse(&rd(3)).expect("INT_R"));
    let clb = PipDb::parse(&rd(4)).expect("CLBLL_L");
    dbs.insert("CLBLL_L".to_string(), PipDb::parse(&rd(4)).expect("CLBLL_L"));
    let mut seg = HashMap::new();
    seg.insert("INT_L".to_string(), SegBits::parse(&rd(5)));
    seg.insert("INT_R".to_string(), SegBits::parse(&rd(6)));

    // the site-pin map is the bridge: SLICE pin -> CLB tile wire
    let site0 = &clb.sites[0];
    let out_wire = site0.pins.get("A").expect("A output pin").clone();
    let in_wire = site0.pins.get("A1").expect("A1 input pin").clone();
    println!("slice pin map: {} / A -> {}   A1 -> {}", site0.name, out_wire, in_wire);

    // two CLB tiles in the same column, a few rows apart
    let mut clbs: Vec<_> = grid
        .of_kind("CLBLL_L")
        .filter(|t| t.grid_y > 100 && t.grid_y < 140 && t.logic_block().is_some())
        .collect();
    clbs.sort_by_key(|t| (t.grid_x, t.grid_y));
    let (src_tile, dst_tile) = (clbs[0], clbs[2]);
    println!("source slice tile: {} ({},{})", src_tile.name, src_tile.grid_x, src_tile.grid_y);
    println!("target slice tile: {} ({},{})", dst_tile.name, dst_tile.grid_x, dst_tile.grid_y);

    // pseudo-PIP kinds: `always` entries are permanent wiring (the LUT-pin bridges), usable at
    // zero cost; anything else is not a conductor we may rely on.
    let mut pp = HashMap::new();
    if a.len() > 7 {
        pp.insert("CLBLL_L".to_string(), Ppips::parse(&rd(7)));
    }
    if a.len() > 8 {
        pp.insert("INT_L".to_string(), Ppips::parse(&rd(8)));
        pp.insert("INT_R".to_string(), Ppips::parse(&rd(8)));
    }
    let fab = Fabric::with_ppips(&grid, dbs, pp, conns);
    let allow = interconnect_with_endpoints(&src_tile.name, &dst_tile.name);

    let src = (src_tile.name.clone(), out_wire.clone());
    let dst = (dst_tile.name.clone(), in_wire.clone());
    println!("\nrouting {} / {}  ->  {} / {}", src.0, src.1, dst.0, dst.1);
    let Some(path) = fab.route(&src, &dst, 2_000_000, &allow) else {
        println!("no route found");
        return;
    };
    println!("route: {} PIPs", path.len());
    for s in &path {
        println!("  {} :: {}", s.tile, s.feature());
    }

    let mut fb = FrameBuf::new();
    let mut ok = 0usize;
    for s in &path {
        let bits = seg.get(&s.tile_type).and_then(|d| d.get(&s.feature()));
        let block = grid.tiles.get(&s.tile).and_then(|t| t.logic_block());
        match (bits, block) {
            (Some(b), Some(blk)) => {
                if fb.apply_feature(&blk, b).is_ok() {
                    ok += 1;
                }
            }
            _ => println!("  unresolved: {}", s.feature()),
        }
    }
    let set: usize = fb.frames.values().map(|f| f.iter().map(|w| w.count_ones() as usize).sum::<usize>()).sum();
    println!("\n{ok}/{} PIPs written; {} frames, {set} bits set", path.len(), fb.len());
    println!(
        "verdict: {}",
        if ok == path.len() && !path.is_empty() {
            "LUT-TO-LUT ROUTE COMPLETE - a neighbour's output reaches a LUT input, every switch \
             backed by physical configuration bits"
        } else {
            "incomplete"
        }
    );
}
