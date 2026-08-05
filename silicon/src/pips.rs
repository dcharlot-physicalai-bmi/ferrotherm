//! Programmable interconnect points — the switches that make routing.
//!
//! Two databases describe them and they use DIFFERENT NAME ORDERS, which is the first trap:
//!
//! ```text
//! tile_type_INT_L.json   key "INT_L.BYP_ALT0->>BYP_BOUNCE0"   = TILE.SRC ->> DST
//! segbits_int_l.db       key "INT_L.BYP_BOUNCE0.BYP_ALT0"     = TILE.DST . SRC
//! ```
//!
//! A router that builds the segbits key in source-first order silently finds no bits for every
//! PIP and emits routing that does nothing — the failure mode that looks like "the database is
//! incomplete" rather than "my key is backwards".
//!
//! Pseudo-PIPs (`ppips_*.db`) carry no bits at all: `always` connections exist unconditionally.
//! They must be skipped when emitting, not treated as missing fuses.

use crate::json::{parse, Json};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pip {
    pub src: String,
    pub dst: String,
    pub directional: bool,
    pub pseudo: bool,
}

impl Pip {
    /// The segbits feature name for this PIP in a given tile type: `TILE.DST.SRC`.
    pub fn feature(&self, tile_type: &str) -> String {
        format!("{tile_type}.{}.{}", self.dst, self.src)
    }
}

/// A site inside a tile (e.g. SLICE_X0Y0) and the tile wire each of its pins lands on.
#[derive(Debug, Clone, Default)]
pub struct Site {
    /// Full site name, e.g. "SLICE_X0Y0" (the JSON stores prefix and name separately).
    pub name: String,
    /// pin name (A, A1..A6, AQ, AMUX, ...) -> tile wire
    pub pins: HashMap<String, String>,
}

#[derive(Debug, Default)]
pub struct PipDb {
    pub tile_type: String,
    pub pips: Vec<Pip>,
    /// wire name -> indices of PIPs driving FROM it
    pub by_src: HashMap<String, Vec<u32>>,
    pub wires: HashSet<String>,
    /// sites in this tile type, in declaration order (X0Y0, X1Y0, ...)
    pub sites: Vec<Site>,
}

impl PipDb {
    /// Parse a `tile_type_*.json`.
    pub fn parse(text: &str) -> Result<PipDb, String> {
        let j = parse(text)?;
        let tile_type = j.get("tile_type").and_then(|t| t.as_str()).unwrap_or("").to_string();
        let mut pips = Vec::new();
        if let Some(Json::Obj(list)) = j.get("pips") {
            for (_key, v) in list {
                let src = v.get("src_wire").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let dst = v.get("dst_wire").and_then(|x| x.as_str()).unwrap_or("").to_string();
                if src.is_empty() || dst.is_empty() {
                    continue;
                }
                pips.push(Pip {
                    src,
                    dst,
                    directional: v.get("is_directional").and_then(|x| x.as_str()) == Some("1"),
                    pseudo: v.get("is_pseudo").and_then(|x| x.as_str()) == Some("1"),
                });
            }
        }
        let mut wires = HashSet::new();
        if let Some(Json::Obj(ws)) = j.get("wires") {
            for (w, _) in ws {
                wires.insert(w.to_string());
            }
        }
        let mut by_src: HashMap<String, Vec<u32>> = HashMap::new();
        for (i, p) in pips.iter().enumerate() {
            by_src.entry(p.src.clone()).or_default().push(i as u32);
        }
        // sites: pins carry the tile wire they land on — the bridge from logic to interconnect
        let mut sites = Vec::new();
        if let Some(Json::Arr(list)) = j.get("sites") {
            for s in list {
                let prefix = s.get("prefix").and_then(|x| x.as_str()).unwrap_or("");
                let sname = s.get("name").and_then(|x| x.as_str()).unwrap_or("");
                let mut pins = HashMap::new();
                if let Some(Json::Obj(ps)) = s.get("site_pins") {
                    for (pin, v) in ps {
                        if let Some(w) = v.get("wire").and_then(|x| x.as_str()) {
                            pins.insert(pin.to_string(), w.to_string());
                        }
                    }
                }
                sites.push(Site { name: format!("{prefix}_{sname}"), pins });
            }
        }
        Ok(PipDb { tile_type, pips, by_src, wires, sites })
    }

    /// PIPs driven from `wire` (the routing fan-out).
    pub fn from(&self, wire: &str) -> impl Iterator<Item = &Pip> + '_ {
        self.by_src.get(wire).into_iter().flatten().map(move |&i| &self.pips[i as usize])
    }
}

/// Pseudo-PIP set from a `ppips_*.db` (feature -> kind: always / default / hint).
#[derive(Debug, Default)]
pub struct Ppips {
    pub kinds: HashMap<String, String>,
}

impl Ppips {
    pub fn parse(text: &str) -> Ppips {
        let mut kinds = HashMap::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if let (Some(name), Some(kind)) = (it.next(), it.next()) {
                kinds.insert(name.to_string(), kind.to_string());
            }
        }
        Ppips { kinds }
    }
    /// Pseudo-PIPs carry no configuration bits and must be skipped when emitting.
    pub fn is_pseudo(&self, feature: &str) -> bool {
        self.kinds.contains_key(feature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // real entries, trimmed of timing data
    const TT: &str = r#"{"pips": {
      "INT_L.BYP_ALT0->>BYP_BOUNCE0": {"dst_wire": "BYP_BOUNCE0", "is_directional": "1",
        "is_pseudo": "0", "src_wire": "BYP_ALT0"},
      "INT_L.BYP_ALT0->>BYP_L0": {"dst_wire": "BYP_L0", "is_directional": "1",
        "is_pseudo": "0", "src_wire": "BYP_ALT0"},
      "INT_L.GFAN0->>IMUX_L0": {"dst_wire": "IMUX_L0", "is_directional": "1",
        "is_pseudo": "1", "src_wire": "GFAN0"}},
      "sites": [], "tile_type": "INT_L", "wires": {"BYP_ALT0": null, "BYP_BOUNCE0": null,
        "BYP_L0": null, "GFAN0": null, "IMUX_L0": null}}"#;

    #[test]
    fn parses_pips_and_fanout() {
        let db = PipDb::parse(TT).unwrap();
        assert_eq!(db.tile_type, "INT_L");
        assert_eq!(db.pips.len(), 3);
        assert_eq!(db.wires.len(), 5);
        let fan: Vec<_> = db.from("BYP_ALT0").map(|p| p.dst.as_str()).collect();
        assert_eq!(fan.len(), 2);
        assert!(fan.contains(&"BYP_BOUNCE0") && fan.contains(&"BYP_L0"));
        assert!(db.from("IMUX_L0").next().is_none(), "nothing is driven from a sink-only wire");
    }

    /// THE NAME-ORDER TRAP: the tile database keys PIPs source-first, the fuse database keys them
    /// destination-first. Building the segbits key in the wrong order finds no bits for any PIP.
    #[test]
    fn segbits_feature_is_destination_first() {
        let db = PipDb::parse(TT).unwrap();
        let p = db.from("BYP_ALT0").find(|p| p.dst == "BYP_BOUNCE0").unwrap();
        assert_eq!(p.feature("INT_L"), "INT_L.BYP_BOUNCE0.BYP_ALT0");
        // the source-first form is what the tile database uses, and it is NOT a segbits key
        assert_ne!(p.feature("INT_L"), "INT_L.BYP_ALT0.BYP_BOUNCE0");
    }

    /// Site pins name the tile wire they land on — this is how a LUT output reaches the
    /// interconnect, and it is the piece that makes a logic-to-logic route possible.
    #[test]
    fn site_pins_map_to_tile_wires() {
        let src = r#"{"tile_type": "CLBLL_L", "pips": {}, "wires": {},
          "sites": [{"name": "X0Y0", "prefix": "SLICE", "site_pins": {
            "A":  {"wire": "CLBLL_LL_A"},
            "A1": {"wire": "CLBLL_LL_A1"},
            "A6": {"wire": "CLBLL_LL_A6"}}}]}"#;
        let db = PipDb::parse(src).unwrap();
        assert_eq!(db.sites.len(), 1);
        assert_eq!(db.sites[0].name, "SLICE_X0Y0");
        assert_eq!(db.sites[0].pins.get("A").map(|s| s.as_str()), Some("CLBLL_LL_A"));
        assert_eq!(db.sites[0].pins.get("A6").map(|s| s.as_str()), Some("CLBLL_LL_A6"));
    }

    #[test]
    fn pseudo_pips_are_flagged_and_skipped() {
        let db = PipDb::parse(TT).unwrap();
        let p = db.from("GFAN0").next().unwrap();
        assert!(p.pseudo, "is_pseudo=1 must be carried through");
        let pp = Ppips::parse("INT_L.IMUX_L0.GFAN0 always\nINT_L.FOO.BAR default\n");
        assert!(pp.is_pseudo(&p.feature("INT_L")));
        assert!(!pp.is_pseudo("INT_L.BYP_BOUNCE0.BYP_ALT0"));
    }
}
