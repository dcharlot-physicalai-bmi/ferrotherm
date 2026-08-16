//! Write an OMMX instance for a small lattice, so the reference implementation can read it.
fn main() {
    let g = ferrotherm::ising::lattice2d(3, 1.0);
    let e = ferrotherm_ommx::export(&g);
    eprintln!("{} vars, constant {}, {} bytes", e.variables, e.constant, e.bytes.len());
    std::io::Write::write_all(&mut std::io::stdout(), &e.bytes).unwrap();
}
