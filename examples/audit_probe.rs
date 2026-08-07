use ferrotherm::model::{Constraint, Expr, Lit, Model, Sense, Var};

// Re-run each test's model with the *wrong* compilation to see whether the assertion notices.
fn main() {
    // (A) at_most_is_satisfied_by_taking_none: assertion is `on <= 3`.
    //     Compile at_most as an EXACT cardinality and see what the assertion says.
    let mut m = Model::new();
    let bits: Vec<Var> = (0..5).map(|i| m.binary(&format!("b{i}"))).collect();
    m.cardinality(bits.iter().map(|&v| Lit::Is(v, 1)).collect(), 3); // pretend at_most==exact
    let s = m.compile().unwrap().solve_best_of(20);
    let on = (0..5).filter(|i| s.value(&format!("b{i}")) == 1).count();
    println!("(A) at_most(...,3) compiled as EXACT -> on = {on}; test asserts on<=3 -> {}",
             if on <= 3 { "STILL PASSES" } else { "fails" });

    // (B) at_least_forces_a_minimum: assertion is `on == 4`, at_least 4 of 6, reward OFF.
    let mut m = Model::new();
    let bits: Vec<Var> = (0..6).map(|i| m.binary(&format!("b{i}"))).collect();
    m.cardinality(bits.iter().map(|&v| Lit::Is(v, 1)).collect(), 4); // pretend at_least==exact
    let mut e = Expr::zero();
    for &v in &bits { e = e.plus(Expr::lit(1.0, Lit::Is(v, 0))); }
    m.objective(Sense::Maximize, e);
    let s = m.compile().unwrap().solve_best_of(40);
    let on = (0..6).filter(|i| s.value(&format!("b{i}")) == 1).count();
    println!("(B) at_least(...,4) compiled as EXACT -> on = {on}; test asserts on==4 -> {}",
             if on == 4 { "STILL PASSES" } else { "fails" });

    // (C) exactly_one_holds: assertion is `on == 1`. Compile ExactlyOne as AtMostOne (drop reward).
    let mut m = Model::new();
    let a = m.binary("a"); let b = m.binary("b"); let cc = m.binary("c");
    m.constrain(Constraint::AtMostOne(vec![Lit::Is(a,1), Lit::Is(b,1), Lit::Is(cc,1)]));
    let s = m.compile().unwrap().solve_best_of(10);
    let on = ["a","b","c"].iter().filter(|n| s.value(n) == 1).count();
    println!("(C) ExactlyOne compiled as AtMostOne -> on = {on}; test asserts on==1 -> {}",
             if on == 1 { "STILL PASSES" } else { "fails" });

    // (D) an_objective_reads_like_arithmetic: is the `- 1.0 * x.is(0)` term load-bearing?
    let solve = |neg: bool| {
        let mut m = Model::new();
        let x = m.categorical("x", 4);
        let y = m.categorical("y", 4);
        let e = if neg { 5.0 * x.is(3) + 2.0 * y.is(1) - 1.0 * x.is(0) }
                else    { 5.0 * x.is(3) + 2.0 * y.is(1) + 1.0 * x.is(0) };
        m.objective(Sense::Maximize, e);
        let s = m.compile().unwrap().solve_best_of(20);
        (s.value("x"), s.value("y"))
    };
    println!("(D) with `-` = {:?}, with Sub broken into `+` = {:?} -> {}",
             solve(true), solve(false),
             if solve(true) == solve(false) { "INDISTINGUISHABLE" } else { "distinguishable" });

    // (E) a_variable_that_did_not_decode: does the `if !s.feasible()` body even run today?
    let mut m = Model::new();
    let x = m.categorical("x", 4);
    m.objective(Sense::Maximize, Expr::lit(50.0, Lit::Is(x, 3)));
    m.fixed_penalty(0.01);
    let s = m.compile().unwrap().solve_annealed(1);
    println!("(E) feasible = {} -> test body {}", s.feasible(),
             if s.feasible() { "SKIPPED ENTIRELY" } else { "runs" });
}
