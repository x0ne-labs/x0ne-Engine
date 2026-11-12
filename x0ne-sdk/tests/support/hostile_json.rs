//! Structure-aware mutations of a JSON document, for the `hostile_bytes`
//! suites (`hostile.rs`): a value replaced where a parser expects another
//! type, a number at the edge of its field, a key removed, renamed or
//! repeated, nesting as deep as `serde_json` reads.
//!
//! Apart from `hostile.rs` because x0ne-ohttp's tests include that file and
//! have no `serde_json`.

#![allow(dead_code)]

use crate::hostile::{Rng, AWKWARD_TEXT};
use serde_json::{json, Value};

/// `serde_json` refuses a document nested deeper than 128: the deepest value
/// a parser of a `Value` is handed by the wire.
pub const DEEPEST: usize = 127;

#[derive(Clone, Debug)]
enum Step {
    Key(String),
    Index(usize),
}

fn paths(v: &Value, here: &mut Vec<Step>, out: &mut Vec<Vec<Step>>) {
    out.push(here.clone());
    match v {
        Value::Object(m) => {
            for (k, child) in m {
                here.push(Step::Key(k.clone()));
                paths(child, here, out);
                here.pop();
            }
        }
        Value::Array(a) => {
            for (i, child) in a.iter().enumerate() {
                here.push(Step::Index(i));
                paths(child, here, out);
                here.pop();
            }
        }
        _ => {}
    }
}

fn at<'a>(v: &'a mut Value, path: &[Step]) -> &'a mut Value {
    let mut cur = v;
    for step in path {
        cur = match step {
            Step::Key(k) => &mut cur[k.as_str()],
            Step::Index(i) => &mut cur[*i],
        };
    }
    cur
}

fn shown(path: &[Step]) -> String {
    let mut s = String::from("$");
    for step in path {
        match step {
            Step::Key(k) => s.push_str(&format!(".{k}")),
            Step::Index(i) => s.push_str(&format!("[{i}]")),
        }
    }
    s
}

/// `depth` arrays, one inside the other.
pub fn nested(depth: usize) -> Value {
    let mut v = json!(0);
    for _ in 0..depth {
        v = json!([v]);
    }
    v
}

fn edge_number(rng: &mut Rng) -> Value {
    match rng.below(10) {
        0 => json!(0),
        1 => json!(-1),
        2 => json!(1),
        3 => json!(u64::MAX),
        4 => json!(i64::MIN),
        5 => json!(1e308),
        6 => json!(0.5),
        7 => json!(u32::MAX),
        8 => json!(u64::from(u32::MAX) + 1),
        _ => json!(-0.0),
    }
}

/// One mutation of `v`, in place, and what it was.
pub fn mutate_json_once(rng: &mut Rng, v: &mut Value) -> String {
    let mut all = Vec::new();
    paths(v, &mut Vec::new(), &mut all);
    let path = rng.pick(&all).clone();
    let target = at(v, &path);
    let what = match rng.below(14) {
        0 => {
            *target = Value::Null;
            "null".to_string()
        }
        1 => {
            *target = json!(rng.below(2) == 0);
            format!("{target}")
        }
        2 => {
            *target = edge_number(rng);
            format!("{target}")
        }
        3 => {
            *target = json!("");
            "an empty string".to_string()
        }
        4 => {
            let n = *rng.pick(&[255usize, 256, 4096, 65_536, 1 << 20]);
            *target = json!("A".repeat(n));
            format!("a string of {n} bytes")
        }
        5 => {
            let t = *rng.pick(&AWKWARD_TEXT);
            *target = match target.as_str() {
                Some(s) => json!(format!("{s}{t}")),
                None => json!(t),
            };
            format!("the text {t:?} added")
        }
        6 => {
            *target = json!([]);
            "an empty array".to_string()
        }
        7 => {
            *target = json!({});
            "an empty object".to_string()
        }
        8 => {
            // Under the path's own depth, so the document still parses.
            let depth = DEEPEST.saturating_sub(path.len() + 1);
            *target = nested(depth);
            format!("{depth} nested arrays")
        }
        9 => match target {
            Value::String(s) => {
                let cut = (0..=s.len() / 2)
                    .rev()
                    .find(|&i| s.is_char_boundary(i))
                    .unwrap_or(0);
                s.truncate(cut);
                format!("its string cut to {cut} bytes")
            }
            Value::Number(n) => {
                let new = match (n.as_u64(), n.as_i64()) {
                    (Some(u), _) => json!(u.wrapping_add(1)),
                    (_, Some(i)) => json!(i.wrapping_sub(1)),
                    _ => json!(n.as_f64().unwrap_or(0.0) * 2.0),
                };
                *target = new;
                format!("its number moved to {target}")
            }
            Value::Bool(b) => {
                *b = !*b;
                "its boolean inverted".to_string()
            }
            _ => {
                *target = json!("x");
                "the string x".to_string()
            }
        },
        10 => match target {
            Value::Object(m) if !m.is_empty() => {
                let keys: Vec<String> = m.keys().cloned().collect();
                let k = rng.pick(&keys).clone();
                match rng.below(3) {
                    0 => {
                        m.remove(&k);
                        format!("key {k:?} removed")
                    }
                    1 => {
                        let child = m.remove(&k).unwrap_or(Value::Null);
                        let renamed = format!("{k}{}", rng.pick(&["", " ", "_", "\u{e9}", "2"]));
                        m.insert(renamed.clone(), child);
                        format!("key {k:?} renamed {renamed:?}")
                    }
                    _ => {
                        let child = m.get(&k).cloned().unwrap_or(Value::Null);
                        m.insert(format!("{k}\u{0}"), child);
                        format!("key {k:?} repeated under another name")
                    }
                }
            }
            Value::Array(a) if !a.is_empty() => {
                let i = rng.below(a.len());
                match rng.below(4) {
                    0 => {
                        a.remove(i);
                        format!("element {i} removed")
                    }
                    1 => {
                        let e = a[i].clone();
                        a.insert(i, e);
                        format!("element {i} repeated")
                    }
                    2 => {
                        let e = a[i].clone();
                        let n = *rng.pick(&[15usize, 16, 17, 64, 65, 1000]);
                        a.extend(std::iter::repeat_n(e, n));
                        format!("element {i} appended {n} times")
                    }
                    _ => {
                        let j = rng.below(a.len());
                        a.swap(i, j);
                        format!("elements {i} and {j} swapped")
                    }
                }
            }
            _ => {
                *target = json!({ "a": target.clone() });
                "wrapped in an object".to_string()
            }
        },
        11 => {
            // The other type that reads the same.
            let new = match &*target {
                Value::String(s) => match s.parse::<u64>() {
                    Ok(n) => json!(n),
                    Err(_) => json!([s]),
                },
                Value::Number(n) => json!(n.to_string()),
                Value::Bool(b) => json!(b.to_string()),
                other => json!(other.to_string()),
            };
            *target = new;
            "its value as another type".to_string()
        }
        12 => {
            *target = json!([target.clone()]);
            "wrapped in an array".to_string()
        }
        _ => {
            let n = *rng.pick(&[-1i64, 0, 1, 2, 16, 17, 255, 256, 65_535, 65_536]);
            *target = json!(n);
            format!("{n}")
        }
    };
    format!("{}: {what}", shown(&path))
}

/// `seed` after one to three mutations, and what they were.
pub fn json_mutant(rng: &mut Rng, seed: &Value) -> (Value, String) {
    let mut v = seed.clone();
    let n = 1 + rng.below(3);
    let what: Vec<String> = (0..n).map(|_| mutate_json_once(rng, &mut v)).collect();
    (v, what.join("; "))
}
