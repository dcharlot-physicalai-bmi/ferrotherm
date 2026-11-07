#![allow(missing_docs)]

fn main() {
    // `dlopen` lives in libdl on some libcs and in libc on others. Linking it is a no-op
    // where the symbol is already in libc, and the difference is not something to discover
    // at runtime as an unresolved symbol.
    if std::env::var("CARGO_CFG_UNIX").is_ok() {
        println!("cargo:rustc-link-lib=dl");
    }
}
