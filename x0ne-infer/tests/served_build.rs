//! The served binary carries no candle: candle is the CPU reference of the
//! tests (`candle-reference`), and no feature a build passes reaches it.
//! The build itself is checked in the GPU VM (`cargo tree`, `nm`, cuobjdump);
//! this is the part a CPU-only `cargo test` can prove, from the manifest.

use std::collections::BTreeSet;
use std::path::Path;

fn manifest() -> toml::Table {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    std::fs::read_to_string(path)
        .expect("Cargo.toml")
        .parse()
        .expect("toml")
}

const CANDLE: [&str; 3] = ["candle-core", "candle-nn", "candle-transformers"];

/// Every feature and `dep:` a list of features turns on, recursively.
fn closure(features: &toml::Table, roots: &[&str]) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut todo: Vec<String> = roots.iter().map(|r| (*r).to_string()).collect();
    while let Some(f) = todo.pop() {
        if !seen.insert(f.clone()) {
            continue;
        }
        if let Some(list) = features.get(&f).and_then(|v| v.as_array()) {
            todo.extend(list.iter().filter_map(|v| v.as_str()).map(String::from));
        }
    }
    seen
}

fn names_candle(item: &str) -> bool {
    item == "candle-reference" || CANDLE.iter().any(|c| item.contains(c))
}

/// A failure means a release build (`cuda`, `nvml`, the default) links
/// candle again, or candle stopped being optional.
#[test]
fn no_feature_of_the_served_build_reaches_candle() {
    let m = manifest();
    let deps = m["dependencies"].as_table().expect("[dependencies]");
    for c in CANDLE {
        assert_eq!(
            deps[c].get("optional").and_then(toml::Value::as_bool),
            Some(true),
            "{c} must be optional"
        );
    }
    let features = m["features"].as_table().expect("[features]");
    let served = closure(features, &["default", "cuda", "nvml"]);
    let reached: Vec<&String> = served.iter().filter(|f| names_candle(f)).collect();
    assert!(reached.is_empty(), "the served build reaches {reached:?}");
    for (name, list) in features {
        let list: Vec<&str> = list
            .as_array()
            .expect("feature list")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        if list.iter().any(|i| i.starts_with("dep:candle")) {
            assert_eq!(name, "candle-reference", "{name} links candle");
        }
    }
}

/// The tests keep candle: the dev-dependency on this crate turns on
/// `candle-reference`, which `cargo build` never sees.
#[test]
fn the_tests_have_the_candle_reference() {
    let m = manifest();
    let me = &m["dev-dependencies"]["x0ne-infer"];
    let on: Vec<&str> = me["features"]
        .as_array()
        .expect("features")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(on.contains(&"candle-reference"), "{on:?}");
    assert!(cfg!(feature = "candle-reference"));
}

/// The attested server's start-up names no candle: it builds its backend
/// from the registry, whose default is x0ne Core.
#[test]
fn the_server_builds_its_backend_by_the_registry() {
    let main = include_str!("../src/main.rs");
    for word in ["candle_backend", "CandleBackend", "candle_core"] {
        assert!(!main.contains(word), "main.rs names {word}");
    }
    assert_eq!(x0ne_infer::backends::DEFAULT, "core");
}
