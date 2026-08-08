//! A JSON value, parser and writer in std alone.
//!
//! The core crate has no dependencies and this tier keeps that property, which matters more here
//! than elsewhere: an MCP server is something people run on their own machine against their own
//! models, and a dependency-free binary is one they can audit in an afternoon.

use std::collections::BTreeMap;
use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as usize),
            _ => None,
        }
    }
    /// A signed integer, which is what a modeller's value is: an integer variable's range may
    /// start below zero, and `as_usize` would refuse the whole lower half of it.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Num(n) if n.fract() == 0.0 && n.abs() < 9.007_199_254_740_992e15 => Some(*n as i64),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            // 1 and 0 as well. Plenty of clients -- shell scripts, older serialisers, agents
            // writing JSON by hand -- send the integer form, and a flag that silently reads as
            // FALSE inverts the caller's whole objective without a word. Anything that is neither
            // a boolean nor a number is still None, so a typo is still refused.
            Json::Num(n) if n.fract() == 0.0 => Some(*n != 0.0),
            _ => None,
        }
    }
    /// An object's key-value pairs, for a caller enumerating a schema it did not write.
    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    /// Convenience for building objects in handler code.
    pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    pub fn s(v: &str) -> Json {
        Json::Str(v.to_string())
    }
    pub fn n(v: f64) -> Json {
        Json::Num(v)
    }
}

// ---- writing ----------------------------------------------------------------------------------

pub fn write(v: &Json) -> String {
    let mut out = String::new();
    wr(v, &mut out);
    out
}

fn wr(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            // JSON has no NaN or Infinity. Emitting them produces a document that parsers reject,
            // so a non-finite number becomes null rather than a syntax error at the far end.
            if n.is_finite() {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{}", n);
                }
            } else {
                out.push_str("null");
            }
        }
        Json::Str(s) => wr_str(s, out),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                wr(x, out);
            }
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                wr_str(k, out);
                out.push(':');
                wr(x, out);
            }
            out.push('}');
        }
    }
}

fn wr_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---- parsing ----------------------------------------------------------------------------------

pub fn parse(text: &str) -> Result<Json, String> {
    let b: Vec<char> = text.chars().collect();
    let mut i = 0;
    let v = p_value(&b, &mut i)?;
    skip_ws(&b, &mut i);
    if i != b.len() {
        return Err(format!("trailing input at char {i}"));
    }
    Ok(v)
}

fn skip_ws(b: &[char], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], ' ' | '\t' | '\n' | '\r') {
        *i += 1;
    }
}

fn p_value(b: &[char], i: &mut usize) -> Result<Json, String> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Err("unexpected end of input".into());
    }
    match b[*i] {
        '{' => p_obj(b, i),
        '[' => p_arr(b, i),
        '"' => Ok(Json::Str(p_str(b, i)?)),
        't' => p_lit(b, i, "true", Json::Bool(true)),
        'f' => p_lit(b, i, "false", Json::Bool(false)),
        'n' => p_lit(b, i, "null", Json::Null),
        _ => p_num(b, i),
    }
}

fn p_lit(b: &[char], i: &mut usize, word: &str, v: Json) -> Result<Json, String> {
    if b[*i..].len() >= word.len() && b[*i..*i + word.len()].iter().collect::<String>() == word {
        *i += word.len();
        Ok(v)
    } else {
        Err(format!("bad literal at char {i}", i = *i))
    }
}

fn p_num(b: &[char], i: &mut usize) -> Result<Json, String> {
    let start = *i;
    if *i < b.len() && (b[*i] == '-' || b[*i] == '+') {
        *i += 1;
    }
    while *i < b.len() && (b[*i].is_ascii_digit() || matches!(b[*i], '.' | 'e' | 'E' | '+' | '-')) {
        // an exponent sign only continues the number directly after e/E
        if matches!(b[*i], '+' | '-') && !matches!(b[*i - 1], 'e' | 'E') {
            break;
        }
        *i += 1;
    }
    let s: String = b[start..*i].iter().collect();
    s.parse::<f64>().map(Json::Num).map_err(|_| format!("bad number {s:?}"))
}

fn p_str(b: &[char], i: &mut usize) -> Result<String, String> {
    *i += 1; // opening quote
    let mut out = String::new();
    while *i < b.len() {
        match b[*i] {
            '"' => {
                *i += 1;
                return Ok(out);
            }
            '\\' => {
                *i += 1;
                if *i >= b.len() {
                    break;
                }
                match b[*i] {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    '/' => out.push('/'),
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    'u' => {
                        let hex: String = b.get(*i + 1..*i + 5).ok_or("short \\u")?.iter().collect();
                        let cp = u32::from_str_radix(&hex, 16).map_err(|_| "bad \\u")?;
                        *i += 4;
                        // A high surrogate is only half a character; pair it with the low one that
                        // must follow, or the text silently loses every non-BMP codepoint.
                        if (0xD800..0xDC00).contains(&cp) && b.get(*i + 1) == Some(&'\\') {
                            let hex2: String =
                                b.get(*i + 3..*i + 7).ok_or("short surrogate")?.iter().collect();
                            let lo = u32::from_str_radix(&hex2, 16).map_err(|_| "bad \\u")?;
                            if (0xDC00..0xE000).contains(&lo) {
                                *i += 6;
                                let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                out.push(char::from_u32(c).ok_or("bad surrogate pair")?);
                            } else {
                                out.push('\u{fffd}');
                            }
                        } else {
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                    }
                    c => out.push(c),
                }
                *i += 1;
            }
            c => {
                out.push(c);
                *i += 1;
            }
        }
    }
    Err("unterminated string".into())
}

fn p_arr(b: &[char], i: &mut usize) -> Result<Json, String> {
    *i += 1;
    let mut out = Vec::new();
    skip_ws(b, i);
    if b.get(*i) == Some(&']') {
        *i += 1;
        return Ok(Json::Arr(out));
    }
    loop {
        out.push(p_value(b, i)?);
        skip_ws(b, i);
        match b.get(*i) {
            Some(',') => *i += 1,
            Some(']') => {
                *i += 1;
                return Ok(Json::Arr(out));
            }
            _ => return Err(format!("expected , or ] at char {i}", i = *i)),
        }
    }
}

fn p_obj(b: &[char], i: &mut usize) -> Result<Json, String> {
    *i += 1;
    let mut out = Vec::new();
    let mut seen = BTreeMap::new();
    skip_ws(b, i);
    if b.get(*i) == Some(&'}') {
        *i += 1;
        return Ok(Json::Obj(out));
    }
    loop {
        skip_ws(b, i);
        if b.get(*i) != Some(&'"') {
            return Err(format!("expected key at char {i}", i = *i));
        }
        let k = p_str(b, i)?;
        skip_ws(b, i);
        if b.get(*i) != Some(&':') {
            return Err(format!("expected : at char {i}", i = *i));
        }
        *i += 1;
        let v = p_value(b, i)?;
        // last key wins, matching what every mainstream parser does with duplicates
        if let Some(&idx) = seen.get(&k) {
            out[idx] = (k, v);
        } else {
            seen.insert(k.clone(), out.len());
            out.push((k, v));
        }
        skip_ws(b, i);
        match b.get(*i) {
            Some(',') => *i += 1,
            Some('}') => {
                *i += 1;
                return Ok(Json::Obj(out));
            }
            _ => return Err(format!("expected , or }} at char {i}", i = *i)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        for src in [
            r#"{"a":1,"b":[1,2,3],"c":"x","d":null,"e":true}"#,
            r#"[]"#,
            r#"{}"#,
            r#"[-1.5,2e3,0.25]"#,
        ] {
            let v = parse(src).expect(src);
            let again = parse(&write(&v)).expect("rewrite");
            assert_eq!(v, again, "{src}");
        }
    }

    #[test]
    fn escapes_survive() {
        let v = Json::s("quote \" back \\ newline \n tab \t ctrl \u{1}");
        let out = write(&v);
        assert_eq!(parse(&out).unwrap(), v);
    }

    #[test]
    fn surrogate_pairs_decode() {
        // a naive \u handler turns this into two replacement chars
        let v = parse(r#""😀""#).unwrap();
        assert_eq!(v.as_str().unwrap(), "\u{1f600}");
    }

    #[test]
    fn non_finite_becomes_null_not_invalid_json() {
        let v = Json::Arr(vec![Json::Num(f64::NAN), Json::Num(f64::INFINITY), Json::Num(1.0)]);
        let out = write(&v);
        assert_eq!(out, "[null,null,1]");
        assert!(parse(&out).is_ok(), "must stay parseable");
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(parse("{} {}").is_err());
        assert!(parse("[1,2]x").is_err());
    }

    #[test]
    fn duplicate_keys_last_wins() {
        let v = parse(r#"{"a":1,"a":2}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_f64(), Some(2.0));
    }
}
