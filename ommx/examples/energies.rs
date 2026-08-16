//! Every state of a small lattice with ferrotherm's energy, so an independent implementation can
//! score the exported instance and be compared against it.
fn main() {
    let g = ferrotherm::ising::lattice2d(3, 1.0);
    for mask in 0..(1u32 << g.n) {
        let s: Vec<i8> = (0..g.n).map(|i| if (mask >> i) & 1 == 1 { 1 } else { -1 }).collect();
        println!("{mask} {}", g.energy(&s));
    }
}
