//! A small HTTP/1.1 server over `std::net`, enough to expose the API and nothing more.

use crate::api;
use crate::json::{write, Json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

/// Bodies larger than this are refused before they are read into memory.
pub const MAX_BODY: usize = 64 * 1024 * 1024;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: String,
}

pub struct Response {
    pub status: u16,
    pub body: String,
}

impl Response {
    fn json(status: u16, v: &Json) -> Response {
        Response { status, body: write(v) }
    }
    fn err(status: u16, msg: &str) -> Response {
        Response::json(
            status,
            &Json::obj(vec![("error", Json::s(msg)), ("ok", Json::Bool(false))]),
        )
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        _ => "Internal Server Error",
    }
}

/// Route one parsed request. Kept transport-agnostic so it is directly testable.
pub fn route(req: &Request) -> Response {
    if req.method == "OPTIONS" {
        return Response { status: 200, body: String::new() };
    }
    let path = req.path.split('?').next().unwrap_or("");
    match (req.method.as_str(), path) {
        ("GET", "/v1/health") => Response::json(
            200,
            &Json::obj(vec![("ok", Json::Bool(true)), ("service", Json::s("ferrotherm"))]),
        ),
        ("GET", "/v1/capabilities") | ("GET", "/") => Response::json(200, &api::capabilities()),
        ("POST", p) if p.starts_with("/v1/") => {
            let op = &p[4..];
            match api::parse_body(&req.body) {
                Err(e) => Response::err(400, &e),
                Ok(v) => match api::dispatch(op, &v) {
                    Ok(r) => Response::json(200, &r),
                    // A bad graph is the caller's mistake to fix, so it is a 400 with the reason,
                    // not a 500 that tells them nothing.
                    Err(e) if e.starts_with("unknown operation") => Response::err(404, &e),
                    Err(e) => Response::err(400, &e),
                },
            }
        }
        ("GET", _) => Response::err(404, "no such endpoint; GET /v1/capabilities lists them"),
        _ => Response::err(405, "method not allowed; the operations are POST /v1/<operation>"),
    }
}

fn handle(mut stream: TcpStream) -> std::io::Result<()> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let peer = stream.try_clone()?;
    let mut rd = BufReader::new(peer);

    let mut line = String::new();
    if rd.read_line(&mut line)? == 0 {
        return Ok(());
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if rd.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
    }

    let resp = if len > MAX_BODY {
        Response::err(413, "request body too large")
    } else {
        let mut buf = vec![0u8; len];
        rd.read_exact(&mut buf)?;
        let body = String::from_utf8_lossy(&buf).into_owned();
        route(&Request { method, path, body })
    };

    let head = format!(
        "HTTP/1.1 {} {}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: content-type\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Connection: close\r\n\r\n",
        resp.status,
        reason(resp.status),
        resp.body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(resp.body.as_bytes())?;
    stream.flush()
}

/// Listen until killed, one thread per connection.
pub fn serve(addr: &str) -> std::io::Result<()> {
    let l = TcpListener::bind(addr)?;
    eprintln!("ferrotherm serving on http://{addr}");
    eprintln!("  GET  /v1/capabilities   what this server can do");
    eprintln!("  POST /v1/sample         draw from a Boltzmann distribution");
    eprintln!("  POST /v1/anneal         minimise an Ising energy");
    eprintln!("  POST /v1/verify         check the sampler against exact enumeration");
    for s in l.incoming() {
        match s {
            Ok(s) => {
                std::thread::spawn(move || {
                    let _ = handle(s);
                });
            }
            Err(e) => eprintln!("accept failed: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    fn get(path: &str) -> Response {
        route(&Request { method: "GET".into(), path: path.into(), body: String::new() })
    }
    fn post(path: &str, body: &str) -> Response {
        route(&Request { method: "POST".into(), path: path.into(), body: body.into() })
    }

    #[test]
    fn health_and_capabilities() {
        assert_eq!(get("/v1/health").status, 200);
        let c = get("/v1/capabilities");
        assert_eq!(c.status, 200);
        let v = parse(&c.body).unwrap();
        assert!(v.get("operations").unwrap().as_arr().unwrap().len() >= 4);
    }

    #[test]
    fn sample_over_http() {
        let r = post("/v1/sample", r#"{"graph":{"builtin":"ring","n":16},"sweeps":10}"#);
        assert_eq!(r.status, 200);
        assert_eq!(parse(&r.body).unwrap().get("nodes").unwrap().as_f64(), Some(16.0));
    }

    #[test]
    fn query_strings_do_not_break_routing() {
        assert_eq!(get("/v1/health?verbose=1").status, 200);
    }

    #[test]
    fn caller_errors_are_400_with_a_reason() {
        let r = post("/v1/sample", "not json");
        assert_eq!(r.status, 400);
        assert!(parse(&r.body).unwrap().get("error").unwrap().as_str().unwrap().contains("JSON"));

        let r = post("/v1/sample", r#"{"graph":{"n":4,"couplings":[[0,7,1.0]]}}"#);
        assert_eq!(r.status, 400);
    }

    #[test]
    fn unknown_operation_is_404() {
        assert_eq!(post("/v1/teleport", "{}").status, 404);
        assert_eq!(get("/nope").status, 404);
    }

    #[test]
    fn wrong_method_is_405() {
        assert_eq!(
            route(&Request { method: "DELETE".into(), path: "/v1/sample".into(), body: String::new() })
                .status,
            405
        );
    }

    #[test]
    fn preflight_succeeds() {
        assert_eq!(
            route(&Request { method: "OPTIONS".into(), path: "/v1/sample".into(), body: String::new() })
                .status,
            200
        );
    }

    #[test]
    fn empty_body_is_a_readable_error_not_a_panic() {
        let r = post("/v1/sample", "");
        assert_eq!(r.status, 400);
        assert!(parse(&r.body).unwrap().get("error").unwrap().as_str().unwrap().contains("graph"));
    }
}
