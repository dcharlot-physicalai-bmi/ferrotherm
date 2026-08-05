// Validate the PIP name-order mapping against the real databases: how many of the interconnect
// tile's PIPs resolve to configuration bits? If the segbits key were built source-first instead
// of destination-first, essentially none would — the router would emit routing that does nothing.
//
// usage: cargo run --release -p ferrotherm-silicon --example pipcheck -- tile_type_INT_L.json segbits_int_l.db [ppips_int_l.db]
use ferrotherm_silicon::pips::{PipDb, Ppips};
use ferrotherm_silicon::segbits::SegBits;

fn main() {
    let mut a = std::env::args().skip(1);
    let tt = std::fs::read_to_string(a.next().expect("tile_type json")).expect("read");
    let sb = std::fs::read_to_string(a.next().expect("segbits db")).expect("read");
    let pp = a.next().map(|p| Ppips::parse(&std::fs::read_to_string(p).expect("read")));

    let db = PipDb::parse(&tt).expect("parse tile type");
    let seg = SegBits::parse(&sb);
    println!("tile type {}: {} PIPs, {} wires; segbits features: {}",
             db.tile_type, db.pips.len(), db.wires.len(), seg.features.len());

    let (mut resolved, mut pseudo, mut missing) = (0usize, 0usize, 0usize);
    let mut missing_examples = Vec::new();
    let mut reversed_hits = 0usize;
    for p in &db.pips {
        let feat = p.feature(&db.tile_type);
        let is_pseudo = p.pseudo || pp.as_ref().is_some_and(|x| x.is_pseudo(&feat));
        if is_pseudo {
            pseudo += 1;
            continue;
        }
        if seg.get(&feat).is_some() {
            resolved += 1;
        } else {
            missing += 1;
            if missing_examples.len() < 5 {
                missing_examples.push(feat.clone());
            }
        }
        // how many would resolve if the key were built the other way round?
        if seg.get(&format!("{}.{}.{}", db.tile_type, p.src, p.dst)).is_some() {
            reversed_hits += 1;
        }
    }
    let real = resolved + missing;
    println!("\nnon-pseudo PIPs: {real}   resolved to bits: {resolved} ({:.1}%)   missing: {missing}",
             100.0 * resolved as f64 / real.max(1) as f64);
    println!("pseudo-PIPs skipped: {pseudo}");
    println!("would resolve with the key reversed (src-first): {reversed_hits}");
    if !missing_examples.is_empty() {
        println!("examples without bits: {missing_examples:?}");
    }
    println!(
        "\nverdict: {}",
        if resolved > reversed_hits * 10 && resolved * 100 / real.max(1) >= 80 {
            "destination-first is the correct key order and the mapping is near-complete"
        } else {
            "the key order is wrong or the databases disagree"
        }
    );
}
