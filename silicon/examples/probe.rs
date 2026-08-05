// Hardware gate 1: talk to the FPGA's CONFIGURATION PORT over our own pure-Rust stack.
// Non-destructive: identifies the part, then reads the status register (does not alter config).
//
// run on the board host: cargo run -p ferrotherm-silicon --features flash --example probe
fn main() {
    let (ftdi, product) = match ferrotherm_silicon::flash::Ftdi::open("Alchitry") {
        Ok(v) => v,
        Err(e) => {
            println!("no board: {e}");
            return;
        }
    };
    println!("board: {product}");
    let mut tap = ferrotherm_silicon::flash::Tap::new(ftdi).expect("tap init");

    let id = tap.idcode().expect("idcode");
    let part = ferrotherm_silicon::flash::xc7_part(id).unwrap_or("unknown");
    println!("IDCODE: 0x{id:08X}  -> {part}");

    match tap.read_config_reg(7) {
        Ok(stat) => {
            let s = ferrotherm_silicon::flash::Stat(stat);
            println!("{}", s.describe());
            println!(
                "verdict: {}",
                if s.done() {
                    "a design is configured and running (DONE high)"
                } else {
                    "fabric is unconfigured (DONE low)"
                }
            );
        }
        Err(e) => println!("STAT read failed: {e}"),
    }
}
