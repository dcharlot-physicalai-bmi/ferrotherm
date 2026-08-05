// Configure the fabric over JTAG from a .bit file, using our own stack end to end.
//
// This is the path that makes a board usable without physical access: JPROGRAM clears
// configuration memory, CFG_IN streams the payload, JSTART releases the startup sequencer.
//
// usage (on the board host):
//   cargo run -p ferrotherm-silicon --features flash --release --example load_bit -- design.bit
use ferrotherm_silicon::bitstream::{find_sync, parse_bit, reg};
use ferrotherm_silicon::flash::{Consequence, Ftdi, Stat, Tap};

fn main() {
    let path = std::env::args().nth(1).expect("usage: load_bit <file.bit>");
    let raw = std::fs::read(&path).expect("read bitstream");
    let bit = parse_bit(&raw);
    let sync = find_sync(bit.config).expect("no sync word in this file");
    println!("bitstream: {} bytes, sync at {}", bit.config.len(), sync - 4);
    if !bit.part.is_empty() {
        println!("  container: part={:?} design={:?}", bit.part, bit.design);
    }

    let (ftdi, product) = Ftdi::open("Alchitry").expect("no board");
    println!("board: {product}");
    let mut tap = Tap::new(ftdi).expect("tap");
    let id = tap.idcode().expect("idcode");
    println!("IDCODE: 0x{id:08X}");
    if let Ok(s) = tap.read_config_reg(reg::STAT) {
        println!("before: {}", Stat(s).describe());
    }

    println!("\nconfiguring ({} bytes through CFG_IN)...", bit.config.len());
    match tap.configure(bit.config, Consequence::ClearsTheRunningDesign) {
        Ok(st) => {
            println!("after:  {}", st.describe());
            println!(
                "\nverdict: {}",
                if st.done() && !st.crc_error() {
                    "CONFIGURED - DONE is high and no CRC error. The fabric is running this design."
                } else if st.crc_error() {
                    "CRC ERROR - the stream was rejected; fabric left unconfigured."
                } else {
                    "DONE still low - configuration did not complete."
                }
            );
        }
        Err(e) => println!("configure failed: {e}"),
    }
}
