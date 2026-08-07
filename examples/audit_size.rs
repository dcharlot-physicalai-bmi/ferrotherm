use ferrotherm::model::Model;
use std::time::Instant;

fn main() {
    for hi in [500i64, 1000, 2000, 4000] {
        let mut m = Model::new();
        m.integer("t", 0, hi);
        let t0 = Instant::now();
        let c = m.compile().unwrap();
        println!("integer 0..={hi}: {} spins, {} edges, compile {:.2}s",
                 c.spins(), c.graph.n_edges, t0.elapsed().as_secs_f64());
    }
    // and the ffi/api accept it with no ceiling of any kind
    let mut m = Model::new();
    m.integer("t", 0, 3_000_000);
    println!("declared integer 0..=3000000 without complaint; compile would want {} spins \
              and {} penalty couplings",
             3_000_001u64, 3_000_001u64 * 3_000_000 / 2);
}
