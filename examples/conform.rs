//! Run the conformance suite on ourselves, and print the result unedited.
//!
//! A conformance suite whose author exempts themselves is worthless.
fn main() {
    let r = ferrotherm::conform::run(&mut ferrotherm::fabric::Cpu::default());
    println!("{r}\n");
    for c in &r.cases {
        println!("  {:<22} asks: {}", c.name, c.asks);
    }
    if !r.passed() {
        println!("\nWE FAIL OUR OWN SUITE. That is published as-is.");
    }
}
