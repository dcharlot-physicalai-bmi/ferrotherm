// Establish the REMOTE UNDO before anything is ever written to the fabric.
//
// IPROG restarts configuration from the board's boot flash. On a board nobody can power-cycle,
// proving this works is the precondition for attempting a JTAG configuration at all: if a load
// fails and leaves the fabric blank, this is the only way back.
//
// It is also the safest possible WRITE test — it exercises our CFG_IN path by issuing a single
// command, and the device's response is to redo the boot it already completed successfully
// (BOOTSTS reported valid_0=1, fallback_0=0).
//
// run on the board host: cargo run -p ferrotherm-silicon --features flash --example undo_check
use ferrotherm_silicon::bitstream::reg;
use ferrotherm_silicon::flash::{Ftdi, Stat, Tap};
use std::{thread, time::Duration};

fn read_state(tap: &mut Tap, label: &str) -> Option<Stat> {
    match tap.read_config_reg(reg::STAT) {
        Ok(s) => {
            let st = Stat(s);
            println!("{label}: {}", st.describe());
            Some(st)
        }
        Err(e) => {
            println!("{label}: STAT read failed: {e}");
            None
        }
    }
}

fn main() {
    let (ftdi, product) = match Ftdi::open("Alchitry") {
        Ok(v) => v,
        Err(e) => {
            println!("no board: {e}");
            return;
        }
    };
    println!("board: {product}\n");
    let mut tap = Tap::new(ftdi).expect("tap");

    let before = read_state(&mut tap, "before");
    let boot_before = tap.read_config_reg(reg::BOOTSTS).ok();
    if let Some(b) = boot_before {
        println!("        BOOTSTS=0x{b:08X}");
    }
    if before.as_ref().map(|s| s.done()) != Some(true) {
        println!("\nfabric is not currently configured — nothing to restore, stopping.");
        return;
    }

    println!("\nissuing IPROG (device reloads its stored image from boot flash)...");
    if let Err(e) = tap.reboot_from_flash() {
        println!("IPROG failed to issue: {e}");
        return;
    }
    thread::sleep(Duration::from_millis(1500));

    // the TAP needs re-initialising after the device restarts its configuration
    let (ftdi2, _) = Ftdi::open("Alchitry").expect("reopen");
    let mut tap2 = Tap::new(ftdi2).expect("tap2");
    let after = read_state(&mut tap2, "after ");
    if let Ok(b) = tap2.read_config_reg(reg::BOOTSTS) {
        println!("        BOOTSTS=0x{b:08X}");
    }

    let ok = after.as_ref().map(|s| s.done()) == Some(true);
    println!(
        "\nverdict: {}",
        if ok {
            "UNDO PROVEN - the device reloaded from flash and DONE is high again. A failed \
             configuration is now recoverable without physical access."
        } else {
            "UNDO FAILED - the fabric did not come back. Do NOT attempt a configuration; it \
             would leave the board blank until someone can power-cycle it."
        }
    );
}
