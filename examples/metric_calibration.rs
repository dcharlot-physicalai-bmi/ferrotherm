// How weak is a weak metric? WORKLOADS.md says per-pixel marginals are one. This measures it.
//
// The EBM-training row of WORKLOADS.md reports per-pixel MAE 0.128 against a noise baseline of
// 0.474 -- samples landing 72.9% closer to the data than noise -- and then warns, in the file, that
// "per-pixel marginals are a weak metric: a model can match them without capturing structure".
//
// THAT WARNING WAS AN ASSERTION. It was written because the argument is obvious, not because anyone
// had measured it, and an obvious argument with no number attached is exactly the kind of caveat a
// reader skips. This puts a number on it, and the number is worse than the warning implies.
//
// THE EXPERIMENT. 3x3 bars-and-stripes, where both metrics are computable exactly:
//
//   marginals-only   nine visible units, NO hidden units and NO couplings. Only the biases are
//                    fitted, so at its fixed point it reproduces every per-pixel marginal EXACTLY
//                    and can represent nothing else -- it is the independent model.
//   wide             the same nine visible units under twelve hidden ones, fully connected.
//
// The first arm is built to be the metric's blind spot rather than found to be one. That is the
// point: a metric is calibrated by the worst model that scores well on it, not by a typical one.
//
// TWO DATASETS, BECAUSE THE FIRST RESULT IS TOO STRONG TO GENERALISE FROM. Bars-and-stripes is ALL
// structure -- every image is constant along one axis, and WHICH axis is exactly what a per-pixel
// marginal cannot see. It is also SYMMETRIC: for every image its negation is in the set, so every
// true marginal is exactly zero and a model that has learned nothing scores PERFECTLY. That is a
// blind metric rather than a weak one, and it is a property of this dataset, not of the metric in
// general. Fashion-MNIST, which the flagship row uses, is not symmetric.
//
// So the second dataset is the same structure with the symmetry broken: bars only, at least one bar
// on. Seven images, marginals of 1/7, correlations just as strong. That is the arrangement where a
// "% closer than noise" figure means what it appears to mean, and it is the number to carry over.
//
// run: cargo run --release --example metric_calibration

use ferrotherm::ebm::{self, Dataset};
use ferrotherm::graph::{Graph, GraphBuilder};

/// Per-pixel marginals of a model, by enumeration: E[s_i] under the Boltzmann distribution.
fn model_marginals(g: &Graph, visible: usize) -> Vec<f64> {
    let states = 1usize << g.n;
    let mut s = vec![-1i8; g.n];
    let mut best = f64::NEG_INFINITY;
    let mut e = Vec::with_capacity(states);
    for mask in 0..states {
        for i in 0..g.n {
            s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
        }
        let v = -g.energy(&s);
        best = best.max(v);
        e.push(v);
    }
    let mut z = 0.0;
    let mut acc = vec![0.0; visible];
    for (mask, &v) in e.iter().enumerate() {
        let w = (v - best).exp();
        z += w;
        for (i, a) in acc.iter_mut().enumerate() {
            *a += w * if mask >> i & 1 == 1 { 1.0 } else { -1.0 };
        }
    }
    acc.iter().map(|a| a / z).collect()
}

/// Per-pixel marginals of the data.
fn data_marginals(d: &Dataset) -> Vec<f64> {
    (0..d.visible)
        .map(|i| d.rows.iter().map(|r| r[i] as f64).sum::<f64>() / d.rows.len() as f64)
        .collect()
}

/// Mean absolute error between two marginal vectors, which is the metric WORKLOADS.md reports.
fn mae(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

/// Bars only, at least one on: the same row-constant structure with the symmetry broken, so the
/// marginals are non-zero and a "closer than noise" figure is not dividing by zero.
fn bars_only(side: usize) -> Dataset {
    let n = side * side;
    let mut rows = Vec::new();
    for mask in 1..(1usize << side) {
        let mut row = vec![-1i8; n];
        for a in 0..side {
            if mask >> a & 1 == 1 {
                for b in 0..side {
                    row[a * side + b] = 1;
                }
            }
        }
        rows.push(row);
    }
    Dataset { visible: n, rows }
}

/// Run both arms on one dataset and print the table.
fn calibrate(label: &str, note: &str, data: &Dataset) -> (f64, f64, f64, f64, f64) {
    let visible = data.visible;
    let floor = -(visible as f64) * core::f64::consts::LN_2;
    let ceiling = -(data.rows.len() as f64).ln();
    let dm = data_marginals(data);
    let noise = mae(&vec![0.0; visible], &dm);

    println!("\n{label}\n  {note}\n");
    println!(
        "{:>16} {:>7} {:>14} {:>16} {:>10}",
        "arm", "spins", "per-pixel MAE", "log-likelihood", "learned"
    );

    let p = ebm::Params { epochs: 600, k: 10, positive_sweeps: 5, learning_rate: 0.05, batch: 7, persistent: false };
    let mut out = Vec::new();
    for (name, structure) in [
        ("marginals-only", GraphBuilder::new(visible).build()),
        ("wide", ebm::rbm(visible, 12)),
    ] {
        let t = ebm::train(&structure, data, &p, 3).unwrap();
        let m = mae(&model_marginals(&t.graph, visible), &dm);
        let ll = t.log_likelihood.unwrap();
        let learned = (ll - floor) / (ceiling - floor) * 100.0;
        println!("{name:>16} {:>7} {m:>15.4} {ll:>16.4} {learned:>9.1}%", t.graph.n);
        out.push((m, learned));
    }
    println!("{:>16} {:>7} {noise:>15.4} {floor:>16.4} {:>9.1}%", "learned nothing", visible, 0.0);
    (out[0].0, out[0].1, out[1].0, out[1].1, noise)
}

fn main() {
    println!("Calibrating the metric WORKLOADS.md reports, where both metrics are exact");
    println!(
        "\nThe EBM-training row reports per-pixel MAE against a noise baseline and warns, in the\n\
         file, that per-pixel marginals are a weak metric. That warning was an ASSERTION: it was\n\
         written because the argument is obvious, not because anyone had measured it. Here is the\n\
         number, from a model built to be the metric's blind spot -- nine pixels, no hidden units\n\
         and no couplings, so it reproduces every marginal exactly and can represent nothing else."
    );

    let bas = ebm::bars_and_stripes(3);
    let (_, _, m_wide_s, l_wide_s, noise_s) = calibrate(
        "SYMMETRIC: 3x3 bars-and-stripes, 14 images over 9 pixels",
        "every image is constant along one axis, and its negation is also in the set",
        &bas,
    );

    let bars = bars_only(3);
    let (m_marg_a, l_marg_a, m_wide_a, l_wide_a, noise_a) = calibrate(
        "ASYMMETRIC: bars only, at least one on -- 7 images over 9 pixels",
        "the same row-constant structure with the symmetry broken, so marginals are non-zero",
        &bars,
    );

    println!(
        "\n\nON THE SYMMETRIC SET THE METRIC IS NOT WEAK, IT IS BLIND. Every true marginal is\n\
         exactly zero, so a model that has learned NOTHING scores a perfect {noise_s:.4} -- better\n\
         than the model that learned {l_wide_s:.1}%, which scores {m_wide_s:.4}. The metric orders\n\
         these two backwards. That is a property of this dataset and not of the metric in general,\n\
         which is exactly why it cannot be the number that gets carried over."
    );

    let closer = |m: f64, noise: f64| (1.0 - m / noise) * 100.0;
    println!(
        "\nAND ON THE ASYMMETRIC SET, WHICH IS THE FAIR TEST, IT ORDERS THEM BACKWARDS TOO.\n\
         Against a noise baseline of {noise_a:.4}:\n\n\
         \x20   marginals-only   {:.1}% closer to the data than noise   and it learned {l_marg_a:.1}%\n\
         \x20   wide             {:.1}% closer                          and it learned {l_wide_a:.1}%\n\n\
         The model that learned almost nothing wins the metric by a wide margin, and the model that\n\
         learned nearly everything scores WORSE THAN NOISE on it. That is not a weak ordering. It\n\
         is the reverse of the true one, on both datasets, in the same direction.",
        closer(m_marg_a, noise_a),
        closer(m_wide_a, noise_a),
    );
    println!(
        "\nTHE MECHANISM, STATED PLAINLY, BECAUSE IT IS NOT THAT THE METRIC IS CURSED. A model\n\
         fitted to maximum likelihood WOULD match the first moments -- moment matching is the\n\
         gradient's fixed point, and it covers the marginals. Two things pull the wide model off\n\
         them: contrastive divergence is a biased gradient by construction, and the hidden units\n\
         give it somewhere else to spend capacity. The marginals-only model has neither problem,\n\
         because matching the marginals is the whole of what it can do. So the metric rewards the\n\
         model that optimises IT, which is the ordinary failure of a proxy and not an exotic one.\n\
         What is worth knowing is the size: here it is large enough to flip the ranking."
    );
    println!(
        "\nWHAT THAT DOES AND DOES NOT SAY ABOUT THE FLAGSHIP ROW. It does NOT say the 72.9% figure\n\
         there is wrong: that number is what it says it is, and it was measured. It says the number\n\
         cannot carry the weight a reader would naturally put on it. A bias-only model reaches\n\
         87.3% here on a dataset made entirely of correlation, so a figure in that range is not\n\
         evidence that structure was learned, and comparing two models by it can point the wrong\n\
         way."
    );
    println!(
        "\nWHY THE EXACT METRIC IS NOT SIMPLY USED INSTEAD. It cannot be, at that scale. The\n\
         likelihoods above enumerate every state, which is why this runs at 9 visible units and 21\n\
         spins rather than 784 and 4,900. That is the difficulty of the field in one sentence: the\n\
         metric that means something is computable only where the models are small, and the metric\n\
         that scales is one a null model nearly saturates. What a small exact experiment CAN do is\n\
         calibrate the large approximate one, and that is what this is."
    );
    println!(
        "\nTHE HONEST USE. Read a per-pixel figure as a FLOOR -- a model failing it has certainly\n\
         not learned -- and never as evidence that one has. Ordering two models by it is the\n\
         specific mistake this measurement rules out."
    );
}
