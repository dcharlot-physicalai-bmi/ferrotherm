// Route a real signal across the real fabric: one LUT's output (LOGIC_OUTS) to another LUT's
// input (IMUX) in a different interconnect tile, then resolve every PIP on the path to physical
// configuration bits. This is the last capability between a ferrotherm fabric and silicon.
//
// usage: route_demo <tilegrid.json> <tileconn.json> <tt_INT_L.json> <tt_INT_R.json> <segbits_int_l.db> <segbits_int_r.db>
use ferrotherm_silicon::framebuf::FrameBuf;
use ferrotherm_silicon::pips::PipDb;
use ferrotherm_silicon::route::{interconnect_only, parse_tileconn, Fabric};
use ferrotherm_silicon::segbits::SegBits;
use ferrotherm_silicon::tilegrid::TileGrid;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let rd = |i: usize| std::fs::read_to_string(&a[i]).expect("read");
    let t0 = Instant::now();
    let grid = TileGrid::parse(&rd(0)).expect("tilegrid");
    let conns = parse_tileconn(&rd(1)).expect("tileconn");
    let mut dbs = HashMap::new();
    for (i, ty) in [(2usize, "INT_L"), (3, "INT_R")] {
        let db = PipDb::parse(&rd(i)).expect("tile type");
        dbs.insert(ty.to_string(), db);
    }
    let mut seg = HashMap::new();
    seg.insert("INT_L".to_string(), SegBits::parse(&rd(4)));
    seg.insert("INT_R".to_string(), SegBits::parse(&rd(5)));
    println!("fabric loaded in {:.2} s: {} tiles, {} connectivity rules",
             t0.elapsed().as_secs_f64(), grid.tiles.len(), conns.len());

    let fab = Fabric::new(&grid, dbs, conns);

    // a real interconnect tile near the middle of the die
    let start_tile = grid
        .of_kind("INT_L")
        .filter(|t| t.grid_y > 100 && t.grid_y < 150)
        .min_by_key(|t| (t.grid_x, t.grid_y))
        .expect("an INT_L tile")
        .name
        .clone();
    let src = (start_tile.clone(), "LOGIC_OUTS_L0".to_string());
    println!("\nsource: {} / {}  (a LUT output entering the interconnect)", src.0, src.1);

    // find IMUX sinks (LUT inputs) in OTHER tiles, reachable from here
    let t1 = Instant::now();
    let mut found: Vec<(String, String, usize)> = Vec::new();
    {
        use std::collections::{HashSet, VecDeque};
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut q = VecDeque::new();
        seen.insert(src.clone());
        q.push_back((src.clone(), 0usize));
        while let Some((n, d)) = q.pop_front() {
            if d > 4 || found.len() >= 3 {
                continue;
            }
            for (next, _) in fab.successors(&n) {
                if seen.contains(&next) {
                    continue;
                }
                if !grid.tiles.get(&next.0).map(|t| interconnect_only(&next.0, &t.kind)).unwrap_or(false) {
                    continue;
                }
                if next.1.starts_with("IMUX") && next.0 != src.0 && found.len() < 3 {
                    found.push((next.0.clone(), next.1.clone(), d + 1));
                }
                seen.insert(next.clone());
                q.push_back((next, d + 1));
            }
        }
    }
    println!("reachable LUT inputs in other tiles (searched {:.2} s):", t1.elapsed().as_secs_f64());
    for (t, w, d) in &found {
        println!("  {t} / {w}   (~{d} hops)");
    }
    let Some((dt, dw, _)) = found.first().cloned() else {
        println!("no cross-tile LUT input found");
        return;
    };

    println!("\nrouting {} / {}  ->  {} / {}", src.0, src.1, dt, dw);
    let path = match fab.route(&src, &(dt.clone(), dw.clone()), 400_000, &interconnect_only) {
        Some(p) => p,
        None => {
            println!("no route found");
            return;
        }
    };
    println!("route: {} PIPs", path.len());
    for s in &path {
        println!("  {} :: {}", s.tile, s.feature());
    }

    // resolve every PIP to physical configuration bits and write them
    let mut fb = FrameBuf::new();
    let (mut ok, mut miss) = (0usize, 0usize);
    for s in &path {
        let Some(db) = seg.get(&s.tile_type) else { miss += 1; continue };
        let Some(bits) = db.get(&s.feature()) else {
            println!("  NO BITS for {}", s.feature());
            miss += 1;
            continue;
        };
        let Some(block) = grid.tiles.get(&s.tile).and_then(|t| t.logic_block()) else {
            miss += 1;
            continue;
        };
        match fb.apply_feature(&block, bits) {
            Ok(()) => ok += 1,
            Err(e) => {
                println!("  apply failed for {}: {e}", s.feature());
                miss += 1;
            }
        }
    }
    let set_bits: usize = fb.frames.values().map(|f| f.iter().map(|w| w.count_ones() as usize).sum::<usize>()).sum();
    println!("\n{ok}/{} PIPs resolved to bits ({miss} missing); {} frames touched, {set_bits} bits set",
             path.len(), fb.len());
    println!(
        "verdict: {}",
        if miss == 0 && ok == path.len() && set_bits > 0 {
            "ROUTED - every switch on the path has physical configuration bits, written into frames"
        } else {
            "incomplete - some PIPs on the path have no bits"
        }
    );
}
