//! The router: a wire graph over the fabric, and path search across it.
//!
//! Nodes are (tile, wire). Two kinds of edge:
//!  * a PIP inside a tile — a real switch, and the only thing that costs configuration bits;
//!  * a wire continuation across a tile boundary, from `tileconn.json`: tile type A at (x,y)
//!    meets type B at (x+dx, y+dy) and their named wires are the same physical conductor.
//!
//! Search is constrained to interconnect tiles by default. CLE-tile PIPs are LUT route-throughs:
//! they borrow an unrelated slice's LUT as a buffer and only conduct if that LUT happens to be
//! programmed, so a path through one is a path that may silently not exist.

use crate::json::{parse, Json};
use crate::pips::{PipDb, Pip};
use crate::tilegrid::TileGrid;
use std::collections::{HashMap, HashSet, VecDeque};

/// One inter-tile connectivity rule.
#[derive(Debug)]
pub struct Conn {
    pub from_type: String,
    pub to_type: String,
    pub dx: i32,
    pub dy: i32,
    /// wire in `from_type` -> wire in `to_type`
    pub pairs: HashMap<String, String>,
}

pub fn parse_tileconn(text: &str) -> Result<Vec<Conn>, String> {
    let j = parse(text)?;
    let Json::Arr(items) = j else { return Err("tileconn.json must be an array".into()) };
    let mut out = Vec::new();
    for it in items {
        let (Some(Json::Arr(types)), Some(Json::Arr(deltas)), Some(Json::Arr(pairs))) =
            (it.get("tile_types"), it.get("grid_deltas"), it.get("wire_pairs"))
        else {
            continue;
        };
        if types.len() != 2 || deltas.len() != 2 {
            continue;
        }
        let mut map = HashMap::new();
        for p in pairs {
            if let Json::Arr(pair) = p {
                if let (Some(a), Some(b)) = (pair.first().and_then(|x| x.as_str()), pair.get(1).and_then(|x| x.as_str())) {
                    map.insert(a.to_string(), b.to_string());
                }
            }
        }
        out.push(Conn {
            from_type: types[0].as_str().unwrap_or("").to_string(),
            to_type: types[1].as_str().unwrap_or("").to_string(),
            dx: match deltas[0] { Json::Num(n) => n as i32, _ => 0 },
            dy: match deltas[1] { Json::Num(n) => n as i32, _ => 0 },
            pairs: map,
        });
    }
    Ok(out)
}

/// A node in the wire graph.
pub type Node = (String, String); // (tile name, wire name)

/// One step of a route: the PIP that was taken, in the tile it belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteStep {
    pub tile: String,
    pub tile_type: String,
    pub pip: Pip,
}

impl RouteStep {
    /// The segbits feature that turns this PIP on.
    pub fn feature(&self) -> String {
        self.pip.feature(&self.tile_type)
    }
}

pub struct Fabric<'g> {
    pub grid: &'g TileGrid,
    /// tile type -> PIPs
    pub pipdbs: HashMap<String, PipDb>,
    /// tile type -> pseudo-PIP kinds. `always` connections are PERMANENT wiring: traversable at
    /// zero cost and emitting no bits. `default` and `hint` are router metadata, not conductors.
    pub ppips: HashMap<String, crate::pips::Ppips>,
    pub conns: Vec<Conn>,
    /// (grid_x, grid_y) -> tile name
    at: HashMap<(u32, u32), String>,
    /// tile type -> indices of conns that start there
    from_type: HashMap<String, Vec<usize>>,
    to_type: HashMap<String, Vec<usize>>,
}

impl<'g> Fabric<'g> {
    pub fn new(grid: &'g TileGrid, pipdbs: HashMap<String, PipDb>, conns: Vec<Conn>) -> Fabric<'g> {
        Self::with_ppips(grid, pipdbs, HashMap::new(), conns)
    }

    pub fn with_ppips(
        grid: &'g TileGrid,
        pipdbs: HashMap<String, PipDb>,
        ppips: HashMap<String, crate::pips::Ppips>,
        conns: Vec<Conn>,
    ) -> Fabric<'g> {
        let mut at = HashMap::new();
        for t in grid.tiles.values() {
            at.insert((t.grid_x, t.grid_y), t.name.clone());
        }
        let mut from_type: HashMap<String, Vec<usize>> = HashMap::new();
        let mut to_type: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, c) in conns.iter().enumerate() {
            from_type.entry(c.from_type.clone()).or_default().push(i);
            to_type.entry(c.to_type.clone()).or_default().push(i);
        }
        Fabric { grid, pipdbs, ppips, conns, at, from_type, to_type }
    }

    fn tile_type(&self, tile: &str) -> Option<&str> {
        self.grid.tiles.get(tile).map(|t| t.kind.as_str())
    }

    /// Successors of a node: PIP hops inside the tile, plus wire continuations into neighbours.
    pub fn successors(&self, node: &Node) -> Vec<(Node, Option<RouteStep>)> {
        let mut out = Vec::new();
        let Some(tile) = self.grid.tiles.get(&node.0) else { return out };
        // PIPs within this tile
        if let Some(db) = self.pipdbs.get(&tile.kind) {
            for p in db.from(&node.1) {
                let feat = p.feature(&tile.kind);
                let kind = self.ppips.get(&tile.kind).and_then(|pp| pp.kinds.get(&feat));
                let step = match kind.map(|s| s.as_str()) {
                    // permanent wiring: usable, but there is nothing to configure
                    Some("always") => None,
                    // fallback drivers and router hints are not conductors we may rely on
                    Some(_) => continue,
                    None => {
                        if p.pseudo {
                            continue; // flagged pseudo with no ppips entry: cannot be turned on
                        }
                        Some(RouteStep {
                            tile: tile.name.clone(),
                            tile_type: tile.kind.clone(),
                            pip: p.clone(),
                        })
                    }
                };
                out.push(((tile.name.clone(), p.dst.clone()), step));
            }
        }
        // wire continuations outward
        for &ci in self.from_type.get(&tile.kind).into_iter().flatten() {
            let c = &self.conns[ci];
            if let Some(w2) = c.pairs.get(&node.1) {
                let (nx, ny) = (tile.grid_x as i32 + c.dx, tile.grid_y as i32 + c.dy);
                if nx >= 0 && ny >= 0 {
                    if let Some(name) = self.at.get(&(nx as u32, ny as u32)) {
                        if self.tile_type(name) == Some(c.to_type.as_str()) {
                            out.push(((name.clone(), w2.clone()), None));
                        }
                    }
                }
            }
        }
        // and inward (the same physical wire, described from the other side)
        for &ci in self.to_type.get(&tile.kind).into_iter().flatten() {
            let c = &self.conns[ci];
            for (a, b) in &c.pairs {
                if b == &node.1 {
                    let (nx, ny) = (tile.grid_x as i32 - c.dx, tile.grid_y as i32 - c.dy);
                    if nx >= 0 && ny >= 0 {
                        if let Some(name) = self.at.get(&(nx as u32, ny as u32)) {
                            if self.tile_type(name) == Some(c.from_type.as_str()) {
                                out.push(((name.clone(), a.clone()), None));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Breadth-first route from `src` to `dst`, expanding only through tiles whose type passes
    /// `allow` (default: interconnect only). Returns the PIPs to enable, in order.
    pub fn route(
        &self,
        src: &Node,
        dst: &Node,
        max_nodes: usize,
        allow: &dyn Fn(&str, &str) -> bool,
    ) -> Option<Vec<RouteStep>> {
        let mut seen: HashSet<Node> = HashSet::new();
        let mut prev: HashMap<Node, (Node, Option<RouteStep>)> = HashMap::new();
        let mut q = VecDeque::new();
        seen.insert(src.clone());
        q.push_back(src.clone());
        let mut visited = 0usize;
        while let Some(n) = q.pop_front() {
            if &n == dst {
                // walk back
                let mut steps = Vec::new();
                let mut cur = n;
                while let Some((p, step)) = prev.get(&cur) {
                    if let Some(s) = step {
                        steps.push(s.clone());
                    }
                    cur = p.clone();
                }
                steps.reverse();
                return Some(steps);
            }
            visited += 1;
            if visited > max_nodes {
                return None;
            }
            for (next, step) in self.successors(&n) {
                if seen.contains(&next) {
                    continue;
                }
                if let Some(k) = self.tile_type(&next.0) {
                    if !allow(&next.0, k) {
                        continue;
                    }
                }
                seen.insert(next.clone());
                prev.insert(next.clone(), (n.clone(), step));
                q.push_back(next);
            }
        }
        None
    }
}

/// The default expansion rule: interconnect tiles only.
pub fn interconnect_only(_name: &str, kind: &str) -> bool {
    kind.starts_with("INT_")
}

/// Interconnect everywhere, plus two named logic tiles as ENDPOINTS. Logic tiles are never
/// transited: a path that enters one and leaves again would be a LUT route-through, which only
/// conducts if that unrelated LUT happens to be programmed.
pub fn interconnect_with_endpoints<'a>(
    src_tile: &'a str,
    dst_tile: &'a str,
) -> impl Fn(&str, &str) -> bool + 'a {
    move |name: &str, kind: &str| kind.starts_with("INT_") || name == src_tile || name == dst_tile
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pips::PipDb;

    fn tiny_fabric_json() -> (&'static str, &'static str, &'static str) {
        // two interconnect tiles side by side
        let grid = r#"{
          "INT_L_X0Y0": {"type": "INT_L", "grid_x": 0, "grid_y": 0, "bits": {}, "sites": {}},
          "INT_R_X1Y0": {"type": "INT_R", "grid_x": 1, "grid_y": 0, "bits": {}, "sites": {}},
          "CLBLL_L_X2Y0": {"type": "CLBLL_L", "grid_x": 2, "grid_y": 0, "bits": {}, "sites": {}}}"#;
        let int_l = r#"{"tile_type": "INT_L", "sites": [], "wires": {},
          "pips": {"a": {"src_wire": "SRC", "dst_wire": "EE2A0", "is_directional": "1", "is_pseudo": "0"}}}"#;
        let int_r = r#"{"tile_type": "INT_R", "sites": [], "wires": {},
          "pips": {"b": {"src_wire": "EE2END0", "dst_wire": "SINK", "is_directional": "1", "is_pseudo": "0"},
                   "c": {"src_wire": "EE2END0", "dst_wire": "PSEUDO", "is_directional": "1", "is_pseudo": "1"}}}"#;
        (grid, int_l, int_r)
    }

    fn conns() -> Vec<Conn> {
        let mut pairs = HashMap::new();
        pairs.insert("EE2A0".to_string(), "EE2END0".to_string());
        vec![Conn { from_type: "INT_L".into(), to_type: "INT_R".into(), dx: 1, dy: 0, pairs }]
    }

    #[test]
    fn routes_across_a_tile_boundary() {
        let (g, l, r) = tiny_fabric_json();
        let grid = TileGrid::parse(g).unwrap();
        let mut dbs = HashMap::new();
        dbs.insert("INT_L".to_string(), PipDb::parse(l).unwrap());
        dbs.insert("INT_R".to_string(), PipDb::parse(r).unwrap());
        let fab = Fabric::new(&grid, dbs, conns());

        let path = fab
            .route(
                &("INT_L_X0Y0".into(), "SRC".into()),
                &("INT_R_X1Y0".into(), "SINK".into()),
                10_000,
                &interconnect_only,
            )
            .expect("a route exists");
        assert_eq!(path.len(), 2, "one PIP per tile, the boundary hop costs none");
        assert_eq!(path[0].tile, "INT_L_X0Y0");
        assert_eq!(path[0].feature(), "INT_L.EE2A0.SRC");
        assert_eq!(path[1].tile, "INT_R_X1Y0");
        assert_eq!(path[1].feature(), "INT_R.SINK.EE2END0");
    }

    /// Pseudo-PIPs must never appear in a route: they carry no bits, so a path through one
    /// cannot be turned on.
    #[test]
    fn pseudo_pips_are_not_routable() {
        let (g, l, r) = tiny_fabric_json();
        let grid = TileGrid::parse(g).unwrap();
        let mut dbs = HashMap::new();
        dbs.insert("INT_L".to_string(), PipDb::parse(l).unwrap());
        dbs.insert("INT_R".to_string(), PipDb::parse(r).unwrap());
        let fab = Fabric::new(&grid, dbs, conns());
        assert!(fab
            .route(
                &("INT_L_X0Y0".into(), "SRC".into()),
                &("INT_R_X1Y0".into(), "PSEUDO".into()),
                10_000,
                &interconnect_only,
            )
            .is_none());
    }

    /// The search must not wander into CLE tiles, whose PIPs are LUT route-throughs.
    #[test]
    fn expansion_is_confined_to_interconnect() {
        assert!(interconnect_only("INT_L_X0Y0", "INT_L") && interconnect_only("t", "INT_R"));
        assert!(!interconnect_only("CLBLL_L_X2Y0", "CLBLL_L"));
        assert!(!interconnect_only("t", "CLBLM_R"));
        // endpoints are reachable but never transited
        let rule = interconnect_with_endpoints("CLBLL_L_X2Y0", "CLBLL_L_X2Y9");
        assert!(rule("CLBLL_L_X2Y0", "CLBLL_L"), "source endpoint allowed");
        assert!(rule("CLBLL_L_X2Y9", "CLBLL_L"), "target endpoint allowed");
        assert!(!rule("CLBLL_L_X2Y5", "CLBLL_L"), "any other logic tile is refused");
    }

    #[test]
    fn parses_real_tileconn_shape() {
        let src = r#"[{"grid_deltas": [1, 0], "tile_types": ["INT_L", "INT_R"],
          "wire_pairs": [["EE2A0", "EE2END0"], ["EE2A1", "EE2END1"]]}]"#;
        let c = parse_tileconn(src).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].dx, c[0].dy), (1, 0));
        assert_eq!(c[0].pairs.get("EE2A1").map(|s| s.as_str()), Some("EE2END1"));
    }
}
