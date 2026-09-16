// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface, and
// every public item in every library here carries a doc. An EXAMPLE has no API surface -- it is a
// program, and its helpers are private to it -- so the lint has nothing to guard and asks for doc
// comments on `fn main`'s scaffolding instead. Scoped off here rather than weakened there.
#![allow(missing_docs)]
// THE LAB: one command from a chip database to a bitstream a board will accept.
//
//   cargo run -p ferrotherm-silicon --example lab -- <prjxray-db> [neurons]
//
// `bsn_fabric` is the same path written out step by step, with every database file passed by
// hand, and it is the one to read when you want to see how a stage works. This one finds the
// files, runs the stages, and says at each point what it did and what would have gone wrong --
// so a first session ends with a configured board rather than with a path error.
use ferrotherm_silicon::bitstream::{decode, find_sync, to_words, write_bit, Packet};
use ferrotherm_silicon::frame::{assemble, words_to_bytes};
use ferrotherm_silicon::framebuf::FrameBuf;
use ferrotherm_silicon::lut::bsn_threshold_init;
use ferrotherm_silicon::pips::{PipDb, Ppips};
use ferrotherm_silicon::route::{contentions, interconnect_with_endpoints, parse_tileconn, Fabric, RouteStep};
use ferrotherm_silicon::segbits::SegBits;
use ferrotherm_silicon::tilegrid::TileGrid;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The part this lab targets, and the identity its bitstream declares.
const PART: &str = "xc7a100t";
const IDCODE: u32 = 0x0363_1093;
/// The chain's threshold: five neighbour inputs, one random bit, fire at three.
const THRESHOLD: u8 = 3;

/// Where a database file lives, and what it is for -- so a missing one names itself.
struct Needed {
    path: PathBuf,
    purpose: &'static str,
}

fn needed(db: &Path, rel: &str, purpose: &'static str) -> Needed {
    Needed { path: db.join(rel), purpose }
}

fn read(n: &Needed) -> Option<String> {
    match std::fs::read_to_string(&n.path) {
        Ok(s) => Some(s),
        Err(e) => {
            println!("  MISSING {}  ({})", n.path.display(), n.purpose);
            println!("          {e}");
            None
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(db_root) = args.first() else {
        println!("usage: lab <prjxray-db> [neurons]");
        println!();
        println!("  <prjxray-db>  a clone of https://github.com/f4pga/prjxray-db");
        println!("  [neurons]     how many stochastic neurons to place (default 64)");
        return;
    };
    let n_neurons: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let db = Path::new(db_root).join("artix7");

    // ---- stage 1: the map ----------------------------------------------------------------
    println!("== 1. the fabric map ==");
    let files = [
        needed(&db, &format!("{PART}/tilegrid.json"), "every tile and the frame its bits live in"),
        needed(&db, &format!("{PART}/tileconn.json"), "which wires continue across tile boundaries"),
        needed(&db, "tile_type_INT_L.json", "the left switchbox's switches"),
        needed(&db, "tile_type_INT_R.json", "the right switchbox's switches"),
        needed(&db, "tile_type_CLBLL_L.json", "the logic tile's site pins"),
        needed(&db, "segbits_int_l.db", "which bit turns each left switch on"),
        needed(&db, "segbits_int_r.db", "which bit turns each right switch on"),
        needed(&db, "segbits_clbll_l.db", "which bits hold a lookup table's truth table"),
        needed(&db, "ppips_clbll_l.db", "logic-tile connections that are permanent, not switches"),
        needed(&db, "ppips_int_l.db", "the same for the left switchbox"),
        needed(&db, "ppips_int_r.db", "and the right"),
    ];
    let mut text = Vec::new();
    for n in &files {
        let Some(s) = read(n) else {
            println!("\nThe database is open data, not a tool:");
            println!("  git clone https://github.com/f4pga/prjxray-db");
            return;
        };
        text.push(s);
    }
    let grid = TileGrid::parse(&text[0]).expect("tilegrid");
    let conns = parse_tileconn(&text[1]).expect("tileconn");
    println!("  {} tiles, {} wire-continuation rules", grid.tiles.len(), conns.len());

    let mut dbs = HashMap::new();
    for (i, k) in [(2, "INT_L"), (3, "INT_R"), (4, "CLBLL_L")] {
        dbs.insert(k.to_string(), PipDb::parse(&text[i]).expect(k));
    }
    let mut seg = HashMap::new();
    seg.insert("INT_L".to_string(), SegBits::parse(&text[5]));
    seg.insert("INT_R".to_string(), SegBits::parse(&text[6]));
    // The logic tile's own segbits belong here too: a route step that lands in a slice has bits
    // there, and leaving them out makes every coupling fail for want of a lookup.
    let clb_seg = SegBits::parse(&text[7]);
    seg.insert("CLBLL_L".to_string(), SegBits::parse(&text[7]));
    let mut pp = HashMap::new();
    pp.insert("CLBLL_L".to_string(), Ppips::parse(&text[8]));
    // Interconnect ppips matter as much: without them the router treats a permanent connection
    // like a switch, then finds it has no bits to set.
    pp.insert("INT_L".to_string(), Ppips::parse(&text[9]));
    pp.insert("INT_R".to_string(), Ppips::parse(&text[10]));

    // ---- stage 2: placement --------------------------------------------------------------
    println!("\n== 2. placement ==");
    let mut tiles: Vec<_> = grid
        .of_kind("CLBLL_L")
        .filter(|t| t.grid_y > 100 && t.logic_block().is_some() && t.sites.len() >= 2)
        .collect();
    tiles.sort_by_key(|t| (t.grid_x, t.grid_y));
    let placed: Vec<_> = tiles.into_iter().take(n_neurons).collect();
    if placed.len() < n_neurons {
        println!("  only {} slices match; ask for fewer neurons", placed.len());
        return;
    }
    println!(
        "  {n_neurons} neurons, one lookup table each, from {} to {}",
        placed[0].sites[0].0,
        placed[n_neurons - 1].sites[0].0
    );

    // ---- stage 3: the truth table --------------------------------------------------------
    println!("\n== 3. the neuron ==");
    let init = bsn_threshold_init(THRESHOLD);
    println!("  fire when popcount(5 neighbours) + coin >= {THRESHOLD}");
    println!("  that rule as a 64-entry table: 0x{init:016X}  ({} of 64 patterns fire)", init.count_ones());
    let init_bits = clb_seg
        .lut_init_bits("CLBLL_L.SLICEL_X0.ALUT")
        .expect("a complete 64-bit INIT map");
    let mut fb = FrameBuf::new();
    let mut written = 0usize;
    for t in &placed {
        if fb.write_lut_init(&t.logic_block().unwrap(), &init_bits, init).is_ok() {
            written += 1;
        }
    }
    println!("  {written}/{n_neurons} tables written into configuration frames");

    // ---- stage 4: routing ----------------------------------------------------------------
    println!("\n== 4. couplings ==");
    let clb_pips = PipDb::parse(&text[4]).expect("CLBLL_L");
    let out_wire = clb_pips.sites[0].pins.get("A").expect("output pin A").clone();
    let in_wire = clb_pips.sites[0].pins.get("A1").expect("input pin A1").clone();
    let fab = Fabric::with_ppips(&grid, dbs, pp, conns);
    let mut nets: Vec<Vec<RouteStep>> = Vec::new();
    let (mut routed, mut switches, mut failed) = (0usize, 0usize, 0usize);
    for i in 0..n_neurons.saturating_sub(1) {
        let (src_t, dst_t) = (&placed[i], &placed[i + 1]);
        let allow = interconnect_with_endpoints(&src_t.name, &dst_t.name);
        let src = (src_t.name.clone(), out_wire.clone());
        let dst = (dst_t.name.clone(), in_wire.clone());
        let Some(path) = fab.route(&src, &dst, 3_000_000, &allow) else {
            println!("  no route: neuron {i} -> {}", i + 1);
            failed += 1;
            continue;
        };
        let mut ok = true;
        for s in &path {
            let bits = seg.get(&s.tile_type).and_then(|d| d.get(&s.feature()));
            let blk = grid.tiles.get(&s.tile).and_then(ferrotherm_silicon::tilegrid::Tile::logic_block);
            match (bits, blk) {
                (Some(b), Some(bl)) if fb.apply_feature(&bl, b).is_ok() => {}
                _ => ok = false,
            }
        }
        if ok {
            routed += 1;
            switches += path.len();
            nets.push(path);
        } else {
            failed += 1;
        }
    }
    println!("  {routed} routed through {switches} switches, {failed} failed");

    // ---- stage 5: the check that has no downstream test ----------------------------------
    println!("\n== 5. contention ==");
    let clashes = contentions(&nets);
    for c in clashes.iter().take(5) {
        println!("  {} in {} driven by nets {:?}", c.wire, c.tile, c.nets);
    }
    println!(
        "  {} wires driven twice{}",
        clashes.len(),
        if clashes.is_empty() { " -- safe to write" } else { " -- DO NOT LOAD THIS" }
    );

    // ---- stage 6: the parity every vendor frame carries -----------------------------------
    println!("\n== 6. frame parity ==");
    let odd_before = fb.odd_frames();
    let sealed = fb.set_x7_parity();
    println!("  {odd_before} of {} frames had the parity no vendor frame has", fb.len());
    println!("  {sealed} parity bits written; {} frames still odd", fb.odd_frames());

    // ---- stage 7: the stream -------------------------------------------------------------
    println!("\n== 7. the bitstream ==");
    let frames = fb.len();
    let set_bits: usize = fb
        .frames
        .values()
        .map(|f| f.iter().map(|w| w.count_ones() as usize).sum::<usize>())
        .sum();
    let raw = words_to_bytes(&assemble(IDCODE, &fb));
    let sync = find_sync(&raw).expect("a sync word");
    let packets = decode(&to_words(&raw[sync..]));
    let unknown = packets.iter().filter(|p| matches!(p, Packet::Unknown(_))).count();
    println!(
        "  {} bytes, {frames} frames touched, {set_bits} bits set, {} packets ({unknown} unknown)",
        raw.len(),
        packets.len()
    );
    std::fs::write("lab_fabric.bin", &raw).expect("write");
    let wrapped = write_bit("lab_fabric;UserID=0XFFFFFFFF", "7a100tfgg484", "2026/01/01", "00:00:00", &raw);
    std::fs::write("lab_fabric.bit", &wrapped).expect("write");
    println!("  wrote lab_fabric.bin (raw) and lab_fabric.bit (with a header)");

    // ---- what to do next -----------------------------------------------------------------
    let even = fb.odd_frames() == 0;
    let complete = written == n_neurons && failed == 0 && unknown == 0 && clashes.is_empty() && even;
    println!("\n== {} ==", if complete { "READY" } else { "INCOMPLETE -- read the counts above" });
    if complete {
        println!("  openFPGALoader -c ft2232 --fpga-part xc7a100tfgg484 lab_fabric.bit");
        println!();
        println!("  The loader ends with `done 1`. That is the DONE pin: the chip took the");
        println!("  configuration and finished startup.");
    }
}
