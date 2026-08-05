//! HTTP sampling server. Usage: ferrotherm-serve [addr]  (default 127.0.0.1:8479)
fn main() -> std::io::Result<()> {
    let addr = std::env::args().nth(1).unwrap_or_else(|| "127.0.0.1:8479".to_string());
    ferrotherm_serve::http::serve(&addr)
}
