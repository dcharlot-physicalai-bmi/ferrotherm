use ferrotherm_serve::{api, json};

fn go(body: &str) {
    match api::dispatch("solve", &json::parse(body).unwrap()) {
        Ok(r) => {
            let v = r.get("values").unwrap();
            let on: Vec<&str> = ["a","b","c","d"].iter()
                .filter(|n| v.get(n).map(|x| x.as_f64()) == Some(Some(1.0))).copied().collect();
            println!("  feasible={:?} on={:?} spins={:?}",
                     r.get("feasible").unwrap().as_bool(), on,
                     r.get("spins").unwrap().as_f64());
        }
        Err(e) => println!("  ERR {e}"),
    }
}

fn main() {
    let vars = r#"{"name":"a","values":2},{"name":"b","values":2},{"name":"c","values":2},{"name":"d","values":2}"#;
    let want_on  = r#""objective":{"maximize":true,"terms":[{"var":"a","value":1},{"var":"b","value":1},{"var":"c","value":1},{"var":"d","value":1}]}"#;
    let want_off = r#""objective":{"maximize":true,"terms":[{"var":"a","value":0},{"var":"b","value":0},{"var":"c","value":0},{"var":"d","value":0}]}"#;
    let of4 = r#"[{"var":"a","value":1},{"var":"b","value":1},{"var":"c","value":1},{"var":"d","value":1}]"#;

    println!("at_least k=4 of 4 (want on=4, reward is OFF):");
    go(&format!(r#"{{"variables":[{vars}],"constraints":[{{"type":"at_least","k":4,"of":{of4}}}],{want_off},"tries":60}}"#));

    println!("at_most k=0 of 4 (want on=0, reward is ON):");
    go(&format!(r#"{{"variables":[{vars}],"constraints":[{{"type":"at_most","k":0,"of":{of4}}}],{want_on},"tries":60}}"#));

    println!("cardinality k=9 of 4 (impossible):");
    go(&format!(r#"{{"variables":[{vars}],"constraints":[{{"type":"cardinality","k":9,"of":{of4}}}],"tries":60}}"#));

    println!("at_least k=9 of 4 (impossible):");
    go(&format!(r#"{{"variables":[{vars}],"constraints":[{{"type":"at_least","k":9,"of":{of4}}}],{want_off},"tries":60}}"#));

    println!("cardinality of FIVE (the API has no 4-variable cap):");
    go(&format!(r#"{{"variables":[{vars},{{"name":"e","values":2}}],"constraints":[{{"type":"cardinality","k":3,"of":[{{"var":"a","value":1}},{{"var":"b","value":1}},{{"var":"c","value":1}},{{"var":"d","value":1}},{{"var":"e","value":1}}]}}],"tries":60}}"#));
}
