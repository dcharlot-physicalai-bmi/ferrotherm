// Trace the configuration sequence step by step, sampling the status register at each stage, so
// the failing step names itself instead of being guessed at.
//
// usage: cargo run -p ferrotherm-silicon --features flash --release --example config_trace -- design.bit
use ferrotherm_silicon::bitstream::{parse_bit, reg};
use ferrotherm_silicon::flash::{Ftdi, Tap, IR_CFG_IN, IR_JPROGRAM, IR_JSTART, IR_BYPASS};

/// The value observed while the board's stored design was running — the target state.
const RUNNING: u32 = 0x5010_79FC;

fn stat(tap: &mut Tap, label: &str) -> u32 {
    match tap.read_config_reg(reg::STAT) {
        Ok(s) => {
            let d = s ^ RUNNING;
            println!("  {label:<22} STAT=0x{s:08X}   differs from running in bits {:?}",
                     (0..32).filter(|b| d >> b & 1 == 1).collect::<Vec<_>>());
            s
        }
        Err(e) => { println!("  {label:<22} read failed: {e}"); 0 }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: config_trace <file.bit>");
    let raw = std::fs::read(&path).expect("read");
    let bit = parse_bit(&raw);
    let (ftdi, product) = Ftdi::open("Alchitry").expect("no board");
    println!("board: {product}, payload {} bytes\n", bit.config.len());
    let mut tap = Tap::new(ftdi).expect("tap");

    stat(&mut tap, "at start");

    println!("\nJPROGRAM (clear configuration memory):");
    tap.reset().unwrap();
    tap.shift_ir(IR_JPROGRAM).unwrap();
    for i in 0..6 {
        tap.run_clocks(20_000).unwrap();
        stat(&mut tap, &format!("after {}k clocks", (i + 1) * 20));
    }

    println!("\nCFG_IN (stream the payload):");
    let payload: Vec<u8> = bit.config.iter().map(|b| ferrotherm_silicon::flash::reverse_byte(*b)).collect();
    tap.shift_ir(IR_CFG_IN).unwrap();
    match tap.shift_dr(&payload, false) {
        Ok(_) => println!("  payload shifted ({} bytes)", payload.len()),
        Err(e) => println!("  shift failed: {e}"),
    }
    stat(&mut tap, "after payload");

    println!("\nJSTART (release the startup sequencer):");
    tap.shift_ir(IR_JSTART).unwrap();
    for i in 0..4 {
        tap.run_clocks(4_000).unwrap();
        stat(&mut tap, &format!("after {}k clocks", (i + 1) * 4));
    }
    tap.shift_ir(IR_BYPASS).unwrap();
    tap.run_clocks(1_000).unwrap();
    let final_stat = stat(&mut tap, "final (BYPASS)");

    println!(
        "\nverdict: {}",
        if final_stat == RUNNING {
            "CONFIGURED - status matches the running-design reference exactly."
        } else {
            "NOT configured - status still differs from the running-design reference."
        }
    );
}
