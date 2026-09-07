//! What induced width does the elimination order actually achieve on structured graphs?
//!
//! `Elimination::order_for` builds two orders -- min-fill and nested dissection -- and keeps the
//! narrower, so this prints the width a caller will actually be charged, not the width of either
//! heuristic alone. Cost is `2^width`, and `Elimination::max_width` defaults to 24, so a row above
//! that is a family the default solver refuses.
use ferrotherm::exact::Elimination;
use ferrotherm::graph::GraphBuilder;

fn strip(w: usize, h: usize) -> ferrotherm::graph::Graph {
    let mut b = GraphBuilder::new(w * h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w { b.couple(i, y * w + x + 1, 1.0); }
            if y + 1 < h { b.couple(i, (y + 1) * w + x, 1.0); }
        }
    }
    b.build()
}

fn row(name: String, g: &ferrotherm::graph::Graph, treewidth: usize) {
    // What the solver will actually use, which is the narrower of the two heuristics.
    let kept = Elimination::order_for(g).1;
    println!("{:>12} {:>7} {:>8} {:>12}", name, g.n, kept, treewidth);
}

fn main() {
    println!("{:>12} {:>7} {:>8} {:>12}", "graph", "spins", "width", "treewidth");
    for (w, h) in [(3, 20), (4, 20), (5, 30), (6, 40), (8, 50), (10, 10)] {
        row(format!("{w}x{h} strip"), &strip(w, h), w.min(h));
    }
    // PERIODIC lattices, where the treewidth is 2L rather than L and min-fill drifts hardest. This
    // is the family `ising::lattice2d` actually builds, and the one every sampler here is verified
    // against, so leaving it out of the width table left the drift that matters undocumented.
    for l in [6usize, 8, 10, 12, 14] {
        row(format!("torus {l}x{l}"), &ferrotherm::ising::lattice2d(l, 1.0), 2 * l);
    }
    for n in [100, 500, 2000] {
        let mut b = GraphBuilder::new(n);
        for i in 0..n - 1 { b.couple(i, i + 1, 1.0); }
        row(format!("chain {n}"), &b.build(), 1);
    }
}
