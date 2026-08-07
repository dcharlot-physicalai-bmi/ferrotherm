use ferrotherm::ffi::*;

fn err(m: *const ModelHandle) -> String {
    let need = ft_model_error(m, core::ptr::null_mut(), 0) as usize;
    let mut b = vec![0u8; need];
    let got = ft_model_error(m, b.as_mut_ptr(), need as u32) as usize;
    String::from_utf8_lossy(&b[..got]).into_owned()
}

fn main() {
    // ---- A. count > 4 is silently truncated to 4 and reports SUCCESS ---------------------
    let m = ft_model_new();
    let v: Vec<u32> = (0..5).map(|_| ft_model_binary(m)).collect();
    // a caller with five variables: only four slots exist, so they pass count=5
    let rc = ft_model_cardinality(m, 5, 3, 1, v[0], v[1], v[2], v[3]);
    println!("A ft_model_cardinality(count=5, k=3) -> {rc}   error={:?}", err(m));
    let spins = ft_model_compile(m);
    ft_model_solve(m, 40);
    let on = v.iter().filter(|&&i| ft_model_value(m, i) == 1).count();
    println!("A   spins={spins} on={on}  (asked exactly 3 of FIVE; v4 is unconstrained)");
    ft_model_free(m);

    // ---- B. at_least k == count returns 1 but constrains nothing -------------------------
    let m = ft_model_new();
    let v: Vec<u32> = (0..4).map(|_| ft_model_binary(m)).collect();
    let rc = ft_model_at_least(m, 4, 4, 1, v[0], v[1], v[2], v[3]);
    for &i in &v { ft_model_objective_term(m, 1, 1.0, i, 0); }   // reward being OFF
    let spins = ft_model_compile(m);
    ft_model_solve(m, 40);
    let on = v.iter().filter(|&&i| ft_model_value(m, i) == 1).count();
    println!("B ft_model_at_least(count=4,k=4) -> {rc}  spins={spins} on={on} feasible={} (want 4 on)",
             ft_model_feasible(m));
    ft_model_free(m);

    // ---- C. at_most k == 0 returns 1 but constrains nothing ------------------------------
    let m = ft_model_new();
    let v: Vec<u32> = (0..3).map(|_| ft_model_binary(m)).collect();
    let rc = ft_model_at_most(m, 3, 0, 1, v[0], v[1], v[2], u32::MAX);
    for &i in &v { ft_model_objective_term(m, 1, 1.0, i, 1); }   // reward being ON
    ft_model_compile(m);
    ft_model_solve(m, 40);
    let on = v.iter().filter(|&&i| ft_model_value(m, i) == 1).count();
    println!("C ft_model_at_most(count=3,k=0) -> {rc}  on={on} feasible={} (want 0 on)",
             ft_model_feasible(m));
    ft_model_free(m);

    // ---- D. the graph.html shape: a refused constraint leaves a model that still compiles -
    let m = ft_model_new();
    let t = ft_model_integer(m, 10, 20);
    let rc = ft_model_fix(m, t, 0);                    // the node editor's DEFAULT value field
    println!("D ft_model_fix(t in 10..=20, value=0) -> {rc}   error={:?}", err(m));
    let spins = ft_model_compile(m);
    ft_model_solve(m, 8);
    println!("D   compile -> {spins} spins (no error shown), t = {}", ft_model_value(m, t));
    ft_model_free(m);
}
