#![allow(missing_docs)]
// HOW FAR THE STRUCTURED CLIQUE EMBEDDINGS SIT FROM THE COUNTING BOUND ON CHAIN LENGTH, exactly.
//
// A clique of n logical variables needs every pair of chains adjacent, so the n chains must expose
// n(n-1) chain-to-chain coupler endpoints between them (each of the n(n-1)/2 logical edges needs one
// hardware edge, counted from both ends). A chain of l sites exposes at most the sum of its sites'
// degrees minus twice its internal edges (at least l - 1). So with S sites in n connected chains,
// the exposed endpoints are at most (sum of the S largest degrees in the hardware) - 2(S - n), and
// the least S with that at least n(n-1) is a lower bound on the total chain length -- S / n on the
// mean chain length -- that no embedding of K_n in that hardware can beat. It uses the actual
// degree sequence, so boundary sites with fewer couplers count for what they have.
//
// The crate's constructions (Boothby, King & Roy 2016; Boothby, Bunyk, Raymond & Roy 2020; Boothby,
// King & Raymond 2021) reach K_{4m} on Chimera C(m,m,4) at chain m+1, K_{12(m-1)} on Pegasus P_m
// at chain m+1 (the fragment), and K_{16m-8} on Zephyr Z_m at chain m+1. Each is verified against
// its hardware graph here, then measured against the bound: the exact factor by which its mean
// chain exceeds the least possible, and whether one more logical variable at the same chain length
// is excluded by counting alone.
//
// run: cargo run --release --example clique_chain_bound

use ferrotherm::device::{pegasus, zephyr};
use ferrotherm::embed::{chimera_clique, pegasus_clique_fragment, zephyr_clique, Embedding};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::ising::chimera;

fn complete_graph(n: usize) -> Graph {
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            b.couple(i, j, 1.0);
        }
    }
    b.build()
}

/// Least total chain sites S such that n connected chains on the S highest-degree sites could
/// expose n(n-1) coupler endpoints between them; `None` if the whole hardware cannot.
fn least_sites(degrees_desc: &[usize], n: usize) -> Option<usize> {
    let need = n * (n - 1);
    let mut prefix = 0usize;
    for (s, &d) in degrees_desc.iter().enumerate() {
        prefix += d;
        let sites = s + 1;
        if sites >= n && prefix >= 2 * (sites - n) + need {
            return Some(sites);
        }
    }
    None
}

fn degrees_desc(g: &Graph) -> Vec<usize> {
    let mut d: Vec<usize> = (0..g.n).map(|i| g.offset[i + 1] - g.offset[i]).collect();
    d.sort_unstable_by(|a, b| b.cmp(a));
    d
}

fn row(fabric: &str, m: usize, hw: &Graph, emb: Option<Embedding>) {
    let Some(emb) = emb else {
        println!("  {fabric:<8} {m:>3}   (no construction)");
        return;
    };
    let n = emb.chains.len();
    if let Err(e) = emb.verify(&complete_graph(n), hw) {
        println!("  {fabric:<8} {m:>3}   K_{n}: construction FAILED verification: {e}");
        return;
    }
    let longest = emb.longest_chain();
    let total: usize = emb.chains.iter().map(Vec::len).sum();
    let mean = total as f64 / n as f64;
    let degs = degrees_desc(hw);
    let bound = least_sites(&degs, n).map(|s| s as f64 / n as f64);
    let next = least_sites(&degs, n + 1).map(|s| s as f64 / (n + 1) as f64);
    let (b_str, ratio, next_str) = match (bound, next) {
        (Some(b), Some(nx)) => (
            format!("{b:.2}"),
            format!("{:.3}", mean / b),
            if nx > longest as f64 { format!("{nx:.2} > {longest}: excluded") } else { format!("{nx:.2} <= {longest}: not excluded") },
        ),
        _ => ("-".to_string(), "-".to_string(), "-".to_string()),
    };
    println!(
        "  {fabric:<8} {m:>3}   K_{n:<4} sites {:>5} of {:>6}   chain max {longest:>3} mean {mean:>6.2}   bound {b_str:>6}   ratio {ratio:>6}   K_{}: {next_str}",
        total,
        hw.n,
        n + 1
    );
}

fn main() {
    println!("STRUCTURED CLIQUE EMBEDDINGS AGAINST THE DEGREE-SEQUENCE BOUND ON MEAN CHAIN LENGTH\n");
    println!("  bound    least mean chain length at which n connected chains on the hardware's highest-degree sites could expose n(n-1) endpoints");
    println!("  ratio    the construction's mean chain over the bound: how far from the least possible");
    println!("  K_{{n+1}}  the bound at one more variable against the construction's LONGEST chain: 'excluded' is a proof by counting\n");
    for m in 2..=16 {
        let hw = chimera(m, m, 4, 1.0);
        row("chimera", m, &hw, chimera_clique(m, 4));
    }
    println!();
    for m in 3..=16 {
        let hw = pegasus(m, 1.0).graph;
        row("pegasus", m, &hw, pegasus_clique_fragment(m));
    }
    println!();
    for m in 1..=15 {
        let hw = zephyr(m, 4, 1.0).graph;
        row("zephyr", m, &hw, zephyr_clique(m, 4));
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  The bound is necessary, not sufficient: an embedding at the bound would have to place every chain");
    println!("  on the highest-degree sites with no wasted coupler, so the ratio is an upper limit on how much a");
    println!("  better construction could shorten chains. Where K_{{n+1}} is 'excluded', no embedding of one more");
    println!("  variable exists at the construction's chain length in that hardware, by counting alone; where it");
    println!("  is not, counting does not settle it and the question stays open for that size.");
}
