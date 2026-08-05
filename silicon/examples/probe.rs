// Hardware gate 1: drive the FPGA's CONFIGURATION PORT over our own pure-Rust stack.
// Entirely non-destructive — identifies the part and reads registers; configures nothing.
//
// The self-check that matters: IDCODE is read TWICE by two independent mechanisms — the JTAG
// IDCODE data register, and a type-1 read of the IDCODE configuration register through
// CFG_IN/CFG_OUT. One known answer, two paths: if the packet machinery were wrong, they would
// disagree.
//
// run on the board host: cargo run -p ferrotherm-silicon --features flash --example probe
use ferrotherm_silicon::bitstream::reg;
use ferrotherm_silicon::flash::{xc7_part, Ftdi, Stat, Tap};

fn main() {
    let (ftdi, product) = match Ftdi::open("Alchitry") {
        Ok(v) => v,
        Err(e) => {
            println!("no board: {e}");
            return;
        }
    };
    println!("board: {product}");
    let mut tap = Tap::new(ftdi).expect("tap init");

    let jtag_id = tap.idcode().expect("jtag idcode");
    println!("IDCODE (JTAG DR):        0x{jtag_id:08X}  -> {}", xc7_part(jtag_id).unwrap_or("unknown"));

    let cfg_id = tap.read_config_reg(reg::IDCODE).expect("config idcode");
    println!("IDCODE (config port):    0x{cfg_id:08X}");
    println!(
        "cross-check: {}",
        if cfg_id == jtag_id {
            "PASS - two independent paths agree, so the packet machinery is sound"
        } else {
            "FAIL - paths disagree; the configuration packet encoding is wrong"
        }
    );

    let stat = Stat(tap.read_config_reg(reg::STAT).expect("stat"));
    println!("{}", stat.describe());
    if let Ok(boot) = tap.read_config_reg(reg::BOOTSTS) {
        println!("BOOTSTS=0x{boot:08X}  (valid_0={} fallback_0={})", boot & 1, boot >> 1 & 1);
    }
    println!(
        "verdict: {}",
        if stat.done() { "a design is configured and running (DONE high)" } else { "fabric unconfigured (DONE low)" }
    );
}
