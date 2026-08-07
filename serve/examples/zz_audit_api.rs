fn main() {
    for body in [
        // "fix" with a missing "value" silently defaults to 0
        r#"{"variables":[{"name":"t","lo":-5,"hi":5}],"constraints":[{"type":"fix","var":"t"}],"tries":8}"#,
        // and a cardinality item with a missing "value" silently defaults to 1
        r#"{"variables":[{"name":"a","lo":10,"hi":14},{"name":"b","lo":10,"hi":14}],
            "constraints":[{"type":"cardinality","k":1,"of":[{"var":"a"},{"var":"b"}]}],"tries":8}"#,
    ] {
        let j = ferrotherm_serve::json::parse(body).unwrap();
        println!("--- {body}");
        match ferrotherm_serve::api::dispatch("solve", &j) {
            Ok(r) => println!("ok: {}", ferrotherm_serve::json::write(&r)),
            Err(e) => println!("err: {e}"),
        }
    }
}
