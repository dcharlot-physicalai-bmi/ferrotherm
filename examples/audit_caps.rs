use ferrotherm::model::{Constraint, Expr, Lit, Model, Sense};

fn main() {
    // ---- 1. AtLeast with k == lits.len() ------------------------------------------------
    let mut m = Model::new();
    let bits: Vec<_> = (0..4).map(|i| m.binary(&format!("b{i}"))).collect();
    let lits: Vec<Lit> = bits.iter().map(|&v| Lit::Is(v, 1)).collect();
    m.at_least(lits.clone(), 4);           // "all four must be on"
    // push the other way so a real constraint would have to fight
    let mut e = Expr::zero();
    for &v in &bits { e = e.plus(Expr::lit(1.0, Lit::Is(v, 0))); }
    m.objective(Sense::Maximize, e);
    let c = m.compile().unwrap();
    let s = c.solve_best_of(40);
    let on = (0..4).filter(|i| s.value(&format!("b{i}")) == 1).count();
    println!("at_least(4 of 4): spins={} feasible={} on={}  (want 4)", c.spins(), s.feasible(), on);

    // ---- 2. AtMost with k == 0 ------------------------------------------------------------
    let mut m = Model::new();
    let bits: Vec<_> = (0..3).map(|i| m.binary(&format!("b{i}"))).collect();
    let lits: Vec<Lit> = bits.iter().map(|&v| Lit::Is(v, 1)).collect();
    m.at_most(lits, 0);                    // "none of them"
    let mut e = Expr::zero();
    for &v in &bits { e = e.plus(Expr::lit(1.0, Lit::Is(v, 1))); }
    m.objective(Sense::Maximize, e);
    let c = m.compile().unwrap();
    let s = c.solve_best_of(40);
    let on = (0..3).filter(|i| s.value(&format!("b{i}")) == 1).count();
    println!("at_most(0 of 3):  spins={} feasible={} on={}  (want 0)", c.spins(), s.feasible(), on);

    // ---- 3. baseline: at_most 1 of 3 does have a slack ------------------------------------
    let mut m = Model::new();
    let bits: Vec<_> = (0..3).map(|i| m.binary(&format!("b{i}"))).collect();
    let lits: Vec<Lit> = bits.iter().map(|&v| Lit::Is(v, 1)).collect();
    m.at_most(lits, 1);
    let mut e = Expr::zero();
    for &v in &bits { e = e.plus(Expr::lit(1.0, Lit::Is(v, 1))); }
    m.objective(Sense::Maximize, e);
    let c = m.compile().unwrap();
    let s = c.solve_best_of(40);
    let on = (0..3).filter(|i| s.value(&format!("b{i}")) == 1).count();
    println!("at_most(1 of 3):  spins={} feasible={} on={}  (want 1)", c.spins(), s.feasible(), on);

    // ---- 4. cardinality with k > lits.len() ------------------------------------------------
    let mut m = Model::new();
    let bits: Vec<_> = (0..3).map(|i| m.binary(&format!("b{i}"))).collect();
    let lits: Vec<Lit> = bits.iter().map(|&v| Lit::Is(v, 1)).collect();
    m.constrain(Constraint::Cardinality { lits, k: 9 });
    let c = m.compile().unwrap();
    let s = c.solve_best_of(20);
    println!("cardinality(9 of 3): feasible={} (impossible, yet reported feasible)", s.feasible());
}
