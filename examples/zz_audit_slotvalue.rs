use ferrotherm::ffi::*;

fn main() {
    // Exactly the call sequence docs/graph.html makes for
    //   [Integer lo=10 hi=20] -> [Fix value = 0 (the node default)] -> [Solve]
    let m = ft_model_new();
    let t = ft_model_integer(m, 10, 20);
    let nm = b"temperature";
    ft_model_name(m, t, nm.as_ptr(), nm.len() as u32);
    let mut cons = 0u32;
    cons += ft_model_fix(m, t, 0);          // graph.html: `cons += W.ft_model_fix(...)`, never checked
    let spins = ft_model_compile(m);        // graph.html: only reads the error when this is 0
    ft_model_solve(m, 8);
    println!("cons accumulated = {cons}   (the Fix node was refused and dropped)");
    println!("compile returned = {spins} spins  -> no error surfaced");
    println!("feasible         = {}", ft_model_feasible(m));
    println!("temperature      = {}", ft_model_value(m, t));
    ft_model_free(m);
}
