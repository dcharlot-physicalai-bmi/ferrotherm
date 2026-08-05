// Hardware-in-the-loop gate 0: identify the FPGA on a connected Alchitry board over our own
// pure-Rust USB/MPSSE/JTAG stack. No vendor tools, no C libraries.
//
// run (on the machine with the board): cargo run -p ferrotherm-silicon --features flash --example idcode
fn main() {
    match ferrotherm_silicon::flash::Ftdi::open("Alchitry") {
        Ok((ftdi, product)) => {
            println!("found: {product}");
            let mut tap = ferrotherm_silicon::flash::Tap::new(ftdi).expect("tap init");
            match tap.idcode() {
                Ok(id) => {
                    let part = ferrotherm_silicon::flash::xc7_part(id).unwrap_or("unknown part");
                    println!("IDCODE = 0x{id:08X}  ->  {part}");
                }
                Err(e) => println!("IDCODE read failed: {e}"),
            }
        }
        Err(e) => println!("no board: {e}"),
    }
}
