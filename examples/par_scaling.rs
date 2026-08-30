// Does asking for threads make the sampler faster? It used to make it thirty-three times slower.
//
// `sweep_par` opened a `thread::scope` per COLOUR CLASS per SWEEP. Two thousand sweeps of a
// two-coloured graph on eighteen threads spawned 72,000 OS threads; spawning costs tens of
// microseconds and a colour class of five hundred nodes costs a few. The work never had a chance,
// and the shipped measurement of it was 0.03x serial at 1,024 spins -- reachable from the C ABI as
// `ft_sweep_par`, where nothing told a caller that the knob they were turning ran backwards.
//
// Two changes. The threads are spawned ONCE for the whole batch, with a `std::sync::Barrier` at
// every colour-class boundary; and a FLOOR caps the thread count so no thread is handed fewer than
// MIN_CHUNK nodes, because below that a thread's share finishes faster than the barrier it waits
// at. Below the floor the parallel entry points are literally the serial code, so they cannot lose.
//
// WHY THE ARMS ARE INTERLEAVED, and it is the reason to trust this table at all. The first
// calibration of the floor ran every serial repetition and then every parallel one. The machine got
// busier in between, and that alone made a floor of 256 look like a 1.45x win where interleaved it
// is a 0.48x loss -- it chose the wrong constant. Timing two things on a shared machine means
// timing them next to each other, or timing the machine instead.
//
// RATIOS, NOT RATES. This runs on whatever machine you have, next to whatever else is on it. The
// ratio of two arms measured back to back survives that; a flips-per-second figure does not.
//
// run: cargo run --release --example par_scaling

use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

fn glass(l: usize) -> Graph {
    let mut r = Pcg::new(7, 1);
    let mut b = GraphBuilder::new(l * l);
    for y in 0..l {
        for x in 0..l {
            let i = y * l + x;
            b.couple(i, y * l + (x + 1) % l, if r.f64() < 0.5 { 1.0 } else { -1.0 });
            b.couple(i, ((y + 1) % l) * l + x, if r.f64() < 0.5 { 1.0 } else { -1.0 });
        }
    }
    b.build()
}

fn main() {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    println!("Parallel sweeps against serial, on {threads} threads\n");
    println!(
        "Both arms are run back to back inside one loop, four times, best of four. The ratio is\n\
         what is measured; the rates are printed only so a reader can see the scale.\n"
    );
    println!(
        "{:>8} {:>10} {:>8} {:>12} {:>12} {:>10}",
        "spins", "per class", "threads", "serial M/s", "par M/s", "par/serial"
    );

    let mut worst = f64::INFINITY;
    let mut best: f64 = 0.0;
    for l in [32usize, 48, 64, 96, 128, 181] {
        let g = glass(l);
        let sweeps = if l >= 96 { 300 } else { 2000 };
        let smallest = g.classes.iter().map(|c| c.len()).min().unwrap_or(0);
        let mut s = Sampler::new(&g, 0.6, 3);
        s.sweeps(20, None);
        let (mut ser, mut par) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..4 {
            let t0 = std::time::Instant::now();
            s.sweeps(sweeps, None);
            ser = ser.min(t0.elapsed().as_secs_f64());
            let t1 = std::time::Instant::now();
            s.sweeps_par(sweeps, threads, None);
            par = par.min(t1.elapsed().as_secs_f64());
        }
        let rate = |sec: f64| (g.n * sweeps) as f64 / sec / 1e6;
        let ratio = ser / par;
        worst = worst.min(ratio);
        best = best.max(ratio);
        println!(
            "{:>8} {smallest:>10} {:>8} {:>12.1} {:>12.1} {ratio:>9.2}x",
            g.n,
            s.threads_used(),
            rate(ser),
            rate(par)
        );
    }

    println!(
        "\nTHE PROPERTY IS THE WORST CELL, NOT THE BEST ONE. Worst {worst:.2}x, best {best:.2}x.\n\
         A speedup that is sometimes a slowdown is not a speedup, it is a coin toss a caller cannot\n\
         call -- so what the floor buys is the left-hand end of this column, not the right."
    );
    println!(
        "\nWHERE THE THREAD COUNT COMES FROM. The `threads` column is what RAN, not what was asked\n\
         for: the floor caps it at one thread per {} nodes of the smallest colour class, so a small\n\
         graph reports 1 and takes the serial path. `ft_threads_used` says the same number over the\n\
         C ABI. A library that ran serially and reported eighteen would be lying about the only\n\
         thing a caller could use to work out why their run was slow.",
        1024
    );
    println!(
        "\nTHE LAST ROW IS ODD ON PURPOSE. 181 is odd, so a periodic lattice of it is NOT bipartite\n\
         -- it needs three colours, and the third is small. The floor sees a small class and refuses\n\
         to thread the whole graph, which is the correct answer and not an obvious one: a graph's\n\
         parallelism is set by its WORST colour class, not by its node count."
    );
}
