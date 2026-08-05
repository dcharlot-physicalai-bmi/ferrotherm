// THE INTEGRATION: place a fabric of binary stochastic neurons into real slices, route its
// couplings through real interconnect, and emit a complete XC7A100T bitstream.
//
// One neuron is one LUT6 holding the stochastic-threshold truth table; its output is routed to
// the next neuron's LUT input, which is the coupling a sampler is built from. Everything is
// resolved to physical configuration bits and assembled into a loadable stream.
//
// usage: bsn_fabric <tilegrid> <tileconn> <tt_INT_L> <tt_INT_R> <tt_CLBLL_L>
//                   <segbits_int_l> <segbits_int_r> <segbits_clbll_l> <ppips_clbll_l>
//                   [ppips_int_l] [ppips_int_r] [n_neurons]
//
// The two interconnect ppips files are optional to pass and load-bearing to omit: without them
// every coupling fails, because permanent connections get routed through as if they were switches.
use ferrotherm_silicon::bitstream::{decode, find_sync, to_words, Packet};
use ferrotherm_silicon::frame::{assemble, words_to_bytes};
use ferrotherm_silicon::framebuf::FrameBuf;
use ferrotherm_silicon::lut::bsn_threshold_init;
use ferrotherm_silicon::pips::{PipDb, Ppips};
use ferrotherm_silicon::route::{parse_tileconn, Fabric};
use ferrotherm_silicon::segbits::SegBits;
use ferrotherm_silicon::tilegrid::TileGrid;
use std::collections::HashMap;

const IDCODE_XC7A100T: u32 = 0x0363_1093;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let rd = |i: usize| std::fs::read_to_string(&a[i]).expect("read");
    let n_neurons: usize = a.get(11).and_then(|s| s.parse().ok()).unwrap_or(8);

    let grid = TileGrid::parse(&rd(0)).expect("tilegrid");
    let conns = parse_tileconn(&rd(1)).expect("tileconn");
    let mut dbs = HashMap::new();
    dbs.insert("INT_L".to_string(), PipDb::parse(&rd(2)).expect("INT_L"));
    dbs.insert("INT_R".to_string(), PipDb::parse(&rd(3)).expect("INT_R"));
    dbs.insert("CLBLL_L".to_string(), PipDb::parse(&rd(4)).expect("CLBLL_L"));
    let mut seg = HashMap::new();
    seg.insert("INT_L".to_string(), SegBits::parse(&rd(5)));
    seg.insert("INT_R".to_string(), SegBits::parse(&rd(6)));
    let clb_seg = SegBits::parse(&rd(7));
    // the CLB's own segbits must be resolvable too: a route step landing in a slice tile has
    // bits there, and leaving them out makes every coupling "fail" for want of a lookup.
    seg.insert("CLBLL_L".to_string(), SegBits::parse(&rd(7)));
    let mut pp = HashMap::new();
    pp.insert("CLBLL_L".to_string(), Ppips::parse(&rd(8)));
    // INTERCONNECT ppips matter just as much: without them the router treats permanent
    // connections like INT_L.BYP_BOUNCE5.BYP_ALT5 as switches, then finds they have no bits.
    if a.len() > 9 {
        pp.insert("INT_L".to_string(), Ppips::parse(&rd(9)));
    }
    if a.len() > 10 {
        pp.insert("INT_R".to_string(), Ppips::parse(&rd(10)));
    }

    // ---- placement: N slices in one column ----
    let mut tiles: Vec<_> = grid
        .of_kind("CLBLL_L")
        .filter(|t| t.grid_y > 100 && t.logic_block().is_some() && t.sites.len() >= 2)
        .collect();
    tiles.sort_by_key(|t| (t.grid_x, t.grid_y));
    let placed: Vec<_> = tiles.into_iter().take(n_neurons).collect();
    if placed.len() < n_neurons {
        println!("only {} slices available", placed.len());
        return;
    }
    println!("placing {n_neurons} binary stochastic neurons, one LUT6 each:");
    for (i, t) in placed.iter().enumerate() {
        println!("  neuron {i}: {} site {}", t.name, t.sites[0].0);
    }

    // ---- write the truth tables ----
    let mut fb = FrameBuf::new();
    let threshold = 3u8;
    let init = bsn_threshold_init(threshold);
    println!("\nstochastic-threshold LUT (threshold {threshold}): INIT = 0x{init:016X}");
    let init_bits = clb_seg
        .lut_init_bits("CLBLL_L.SLICEL_X0.ALUT")
        .expect("complete 64-bit INIT map");
    let mut luts_written = 0usize;
    for t in &placed {
        let block = t.logic_block().unwrap();
        match fb.write_lut_init(&block, &init_bits, init) {
            Ok(()) => luts_written += 1,
            Err(e) => println!("  LUT write failed in {}: {e}", t.name),
        }
    }
    println!("{luts_written}/{n_neurons} truth tables written into frames");

    // ---- route the couplings: neuron i output -> neuron i+1 input ----
    let clb_pips = PipDb::parse(&rd(4)).expect("CLBLL_L");
    let out_wire = clb_pips.sites[0].pins.get("A").expect("A").clone();
    let in_wire = clb_pips.sites[0].pins.get("A1").expect("A1").clone();
    let fab = Fabric::with_ppips(&grid, dbs, pp, conns);

    let (mut routed, mut pips_total, mut failed) = (0usize, 0usize, 0usize);
    let n_links = n_neurons - 1;
    for i in 0..n_neurons {
        let src_t = &placed[i];
        if i + 1 >= n_neurons {
            continue; // a chain: the wrap-around would span the whole column
        }
        let dst_t = &placed[i + 1];
        let allow = ferrotherm_silicon::route::interconnect_with_endpoints(&src_t.name, &dst_t.name);
        let src = (src_t.name.clone(), out_wire.clone());
        let dst = (dst_t.name.clone(), in_wire.clone());
        match fab.route(&src, &dst, 3_000_000, &allow) {
            Some(path) => {
                let mut all_ok = true;
                for s in &path {
                    let bits = seg.get(&s.tile_type).and_then(|d| d.get(&s.feature()));
                    let blk = grid.tiles.get(&s.tile).and_then(|t| t.logic_block());
                    match (bits, blk) {
                        (Some(b), Some(bl)) => {
                            if let Err(e) = fb.apply_feature(&bl, b) {
                                if failed < 2 {
                                    println!("  link {i}: apply failed for {} in {} ({}): {e}",
                                             s.feature(), s.tile, s.tile_type);
                                    println!("    block = {:?}", bl);
                                }
                                all_ok = false;
                            }
                        }
                        (None, _) => {
                            if failed < 2 {
                                println!("  link {i}: NO BITS for {} (tile type {})",
                                         s.feature(), s.tile_type);
                            }
                            all_ok = false;
                        }
                        (_, None) => {
                            if failed < 2 {
                                println!("  link {i}: no config block for tile {}", s.tile);
                            }
                            all_ok = false;
                        }
                    }
                }
                if all_ok {
                    routed += 1;
                    pips_total += path.len();
                } else {
                    failed += 1;
                }
            }
            None => {
                failed += 1;
                println!("  no route: neuron {i} -> {}", i + 1);
            }
        }
    }
    println!("couplings routed: {routed} ({pips_total} PIPs total), failed: {failed}");

    // ---- emit the bitstream ----
    let words = assemble(IDCODE_XC7A100T, &fb);
    let bytes = words_to_bytes(&words);
    let set_bits: usize = fb.frames.values().map(|f| f.iter().map(|w| w.count_ones() as usize).sum::<usize>()).sum();
    println!("\nbitstream: {} bytes, {} frames touched, {set_bits} configuration bits set",
             bytes.len(), fb.len());

    // decode our own output as a check that it is well formed
    let sync = find_sync(&bytes).expect("sync");
    let packets = decode(&to_words(&bytes[sync..]));
    let unknown = packets.iter().filter(|p| matches!(p, Packet::Unknown(_))).count();
    let frame_words: usize = packets.iter().map(|p| match p {
        Packet::Continue { words } => *words,
        Packet::Write { reg, data } if *reg == ferrotherm_silicon::bitstream::reg::FDRI => data.len(),
        _ => 0,
    }).sum();
    println!("decoded: {} packets, {unknown} unknown, {frame_words} frame words ({} frames)",
             packets.len(), frame_words / 101);

    std::fs::write("bsn_fabric.bit", &bytes).expect("write");
    println!("wrote bsn_fabric.bit");
    println!(
        "\nverdict: {}",
        if luts_written == n_neurons && routed == n_links && failed == 0 && unknown == 0 {
            "FABRIC EMITTED - every neuron placed, every coupling routed, all bits resolved, \
             stream well formed"
        } else {
            "incomplete - see the counts above"
        }
    );
}
