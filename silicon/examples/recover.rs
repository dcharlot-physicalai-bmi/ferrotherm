// Diagnose and, if needed, recover the fabric: watch the status register settle, then re-issue
// IPROG so the device reloads its stored image from boot flash.
//
// run on the board host: cargo run -p ferrotherm-silicon --features flash --example recover
use ferrotherm_silicon::bitstream::reg;
use ferrotherm_silicon::flash::{Ftdi, Stat, Tap};
use std::{thread, time::Duration};

fn snapshot(label: &str) -> Option<u32> {
    let (ftdi, _) = Ftdi::open("Alchitry").ok()?;
    let mut tap = Tap::new(ftdi).ok()?;
    let s = tap.read_config_reg(reg::STAT).ok()?;
    println!("{label} {}", Stat(s).describe());
    Some(s)
}

fn main() {
    println!("watching the status register settle:");
    let mut last = None;
    for i in 0..5 {
        last = snapshot(&format!("  t+{}s ", i * 2));
        thread::sleep(Duration::from_millis(2000));
    }
    let target = 0x5010_79FCu32; // the value observed while the stored design was running

    if last == Some(target) {
        println!("\nfabric is back to its original state — nothing to do.");
        return;
    }
    println!("\nnot at the original 0x{target:08X}; re-issuing IPROG to reload from flash...");
    if let Ok((ftdi, _)) = Ftdi::open("Alchitry") {
        if let Ok(mut tap) = Tap::new(ftdi) {
            match tap.reboot_from_flash() {
                Ok(()) => println!("IPROG issued"),
                Err(e) => println!("IPROG failed: {e}"),
            }
        }
    }
    for i in 0..6 {
        thread::sleep(Duration::from_millis(2500));
        let s = snapshot(&format!("  after IPROG t+{}s ", (i + 1) * 2));
        if s == Some(target) {
            println!("\nRECOVERED: the device reloaded its stored image.");
            return;
        }
    }
    println!(
        "\nNOT recovered by IPROG. The board still answers JTAG (so remote access is intact), \
         but the fabric is not in its original state; a power cycle will reload it from flash."
    );
}
