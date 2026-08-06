//! What induced width does min-fill actually achieve on structured graphs?
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

fn main() {
    let el = Elimination::default();
    println!("{:>12} {:>7} {:>8} {:>12}", "graph", "spins", "width", "treewidth");
    for (w, h) in [(3, 20), (4, 20), (5, 30), (6, 40), (8, 50), (10, 10)] {
        let g = strip(w, h);
        println!("{:>12} {:>7} {:>8} {:>12}", format!("{w}x{h} strip"), g.n, el.width(&g), w.min(h));
    }
    for n in [100, 500, 2000] {
        let mut b = GraphBuilder::new(n);
        for i in 0..n - 1 { b.couple(i, i + 1, 1.0); }
        let g = b.build();
        println!("{:>12} {:>7} {:>8} {:>12}", format!("chain {n}"), g.n, el.width(&g), 1);
    }
}
