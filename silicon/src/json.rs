//! A small JSON reader — enough for the fabric databases, and no dependencies.
//!
//! The databases are tens of megabytes, so this parses into a compact tree with borrowed string
//! slices rather than a general-purpose document model.

#[derive(Debug, Clone, PartialEq)]
pub enum Json<'a> {
    Null,
    Bool(bool),
    Num(f64),
    Str(&'a str),
    Arr(Vec<Json<'a>>),
    Obj(Vec<(&'a str, Json<'a>)>),
}

impl<'a> Json<'a> {
    pub fn get(&self, key: &str) -> Option<&Json<'a>> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| *k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&'a str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) => Some(*n as u64),
            _ => None,
        }
    }
    pub fn entries(&self) -> &[(&'a str, Json<'a>)] {
        match self {
            Json::Obj(m) => m,
            _ => &[],
        }
    }
}

pub fn parse(src: &str) -> Result<Json<'_>, String> {
    let b = src.as_bytes();
    let mut i = 0usize;
    let v = parse_value(src, b, &mut i)?;
    Ok(v)
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') {
        *i += 1;
    }
}

fn parse_value<'a>(src: &'a str, b: &[u8], i: &mut usize) -> Result<Json<'a>, String> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Err("unexpected end of input".into());
    }
    match b[*i] {
        b'{' => {
            *i += 1;
            let mut out = Vec::new();
            loop {
                skip_ws(b, i);
                if *i < b.len() && b[*i] == b'}' {
                    *i += 1;
                    break;
                }
                let key = parse_string(src, b, i)?;
                skip_ws(b, i);
                if *i >= b.len() || b[*i] != b':' {
                    return Err(format!("expected ':' at {i}"));
                }
                *i += 1;
                let val = parse_value(src, b, i)?;
                out.push((key, val));
                skip_ws(b, i);
                if *i < b.len() && b[*i] == b',' {
                    *i += 1;
                }
            }
            Ok(Json::Obj(out))
        }
        b'[' => {
            *i += 1;
            let mut out = Vec::new();
            loop {
                skip_ws(b, i);
                if *i < b.len() && b[*i] == b']' {
                    *i += 1;
                    break;
                }
                out.push(parse_value(src, b, i)?);
                skip_ws(b, i);
                if *i < b.len() && b[*i] == b',' {
                    *i += 1;
                }
            }
            Ok(Json::Arr(out))
        }
        b'"' => Ok(Json::Str(parse_string(src, b, i)?)),
        b't' => {
            *i += 4;
            Ok(Json::Bool(true))
        }
        b'f' => {
            *i += 5;
            Ok(Json::Bool(false))
        }
        b'n' => {
            *i += 4;
            Ok(Json::Null)
        }
        _ => {
            let start = *i;
            while *i < b.len()
                && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
            {
                *i += 1;
            }
            src[start..*i].parse::<f64>().map(Json::Num).map_err(|e| e.to_string())
        }
    }
}

fn parse_string<'a>(src: &'a str, b: &[u8], i: &mut usize) -> Result<&'a str, String> {
    skip_ws(b, i);
    if *i >= b.len() || b[*i] != b'"' {
        return Err(format!("expected string at {i}"));
    }
    *i += 1;
    let start = *i;
    while *i < b.len() && b[*i] != b'"' {
        // the fabric databases contain no escapes; step over one if a future file has any
        if b[*i] == b'\\' {
            *i += 1;
        }
        *i += 1;
    }
    let s = &src[start..*i];
    *i += 1;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_tilegrid_entry() {
        // a real entry, verbatim
        let src = r#"{"CLBLL_L_X2Y0": {"bits": {"CLB_IO_CLK": {"baseaddr": "0x00420100",
          "frames": 36, "offset": 0, "words": 2}}, "clock_region": "X0Y0", "grid_x": 10,
          "grid_y": 207, "sites": {"SLICE_X0Y0": "SLICEL", "SLICE_X1Y0": "SLICEL"},
          "type": "CLBLL_L"}}"#;
        let j = parse(src).expect("parse");
        let tile = j.get("CLBLL_L_X2Y0").expect("tile");
        assert_eq!(tile.get("type").unwrap().as_str(), Some("CLBLL_L"));
        assert_eq!(tile.get("grid_x").unwrap().as_u64(), Some(10));
        let blk = tile.get("bits").unwrap().get("CLB_IO_CLK").unwrap();
        assert_eq!(blk.get("baseaddr").unwrap().as_str(), Some("0x00420100"));
        assert_eq!(blk.get("frames").unwrap().as_u64(), Some(36));
        assert_eq!(blk.get("words").unwrap().as_u64(), Some(2));
        let sites = tile.get("sites").unwrap().entries();
        assert_eq!(sites.len(), 2);
        assert_eq!(sites[0], ("SLICE_X0Y0", Json::Str("SLICEL")));
    }

    #[test]
    fn handles_arrays_and_empties() {
        let j = parse(r#"{"a": [], "b": [1, 2.5, -3], "c": {}, "d": null, "e": true}"#).unwrap();
        assert_eq!(j.get("a"), Some(&Json::Arr(vec![])));
        match j.get("b").unwrap() {
            Json::Arr(v) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[2], Json::Num(-3.0));
            }
            _ => panic!("expected array"),
        }
        assert_eq!(j.get("d"), Some(&Json::Null));
        assert_eq!(j.get("e"), Some(&Json::Bool(true)));
    }
}
