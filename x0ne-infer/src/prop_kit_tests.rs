//! Property testing by hand, for `generation_prop_tests` and
//! `service_prop_tests`: a seeded generator, a runner that names the failing
//! seed and retries it at smaller sizes, and what the properties share (an
//! independent JSON reader and schema validator, schema and text generators,
//! byte-level tokenizers, scripted sessions and a scripted model).
//!
//! No crate is added for it: the CI builds offline from vendored crates.
//!
//! A failure prints `X0NE_PROP_REPLAY='<property>:<seed>:<size>'`; set it to
//! run that one case again. `X0NE_PROP_DEEP=<n>` runs `n` times the cases.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use serde_json::{json, Value};
use tokenizers::Tokenizer;

use crate::backend::{Completion, GenerationParams, Isolation, LoadedModel, NoSink, TokenSink};
use crate::constrain::TokenTable;
use crate::prompt::Prompt;
use crate::stepped::{LogitsSession, TableCache};

// ── The generator and the runner ────────────────────────────────────────────

/// splitmix64: one `u64` of state, the same draws on every machine.
#[derive(Clone, Debug)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// In `0..n`; 0 when `n` is 0.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next() % n as u64) as usize
    }

    /// In `lo..=hi`.
    pub(crate) fn between(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi.saturating_sub(lo) + 1)
    }

    /// True `num` times out of `den`.
    pub(crate) fn chance(&mut self, num: usize, den: usize) -> bool {
        self.below(den) < num
    }

    /// In `[0, 1)`, 24 bits.
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub(crate) fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }

    /// A generator of its own, drawn from this one.
    pub(crate) fn fork(&mut self) -> Self {
        Self(self.next())
    }
}

fn fnv1a(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "a panic that is not text".into())
}

/// `X0NE_PROP_REPLAY='<name>:<seed>:<size>'` for this property.
fn replay(name: &str) -> Option<(u64, usize)> {
    let asked = std::env::var("X0NE_PROP_REPLAY").ok()?;
    let mut parts = asked.rsplitn(3, ':');
    let size = parts.next()?.parse().ok()?;
    let seed = parts.next()?;
    let seed = u64::from_str_radix(seed.trim_start_matches("0x"), 16).ok()?;
    (parts.next()? == name).then_some((seed, size))
}

/// How many times the cases of every property run: `X0NE_PROP_DEEP`, 1 by default.
pub(crate) fn deep() -> usize {
    std::env::var("X0NE_PROP_DEEP")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(1)
}

/// Runs `property` on `cases` seeds drawn from `name`, at sizes `1..=max_size`
/// in turn. A case that panics is run again at every smaller size: the
/// smallest that still fails is reported with its seed, then the panic goes on.
pub(crate) fn check(name: &str, cases: usize, max_size: usize, property: impl Fn(&mut Rng, usize)) {
    let run = |seed: u64, size: usize| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            property(&mut Rng::new(seed), size);
        }))
    };
    if let Some((seed, size)) = replay(name) {
        property(&mut Rng::new(seed), size);
        return;
    }
    let mut seeds = Rng::new(fnv1a(name));
    for case in 0..cases * deep() {
        let seed = seeds.next();
        let size = 1 + case % max_size.max(1);
        let Err(first) = run(seed, size) else {
            continue;
        };
        let (smallest, payload) = (1..size)
            .find_map(|s| run(seed, s).err().map(|p| (s, p)))
            .unwrap_or((size, first));
        // Printed here: a panic hook that withholds payloads may be installed.
        eprintln!(
            "property `{name}` failed at case {case}, seed {seed:#x}, size {size}; \
             smallest failing size {smallest}: {}\n\
             replay: X0NE_PROP_REPLAY='{name}:{seed:#x}:{smallest}'",
            panic_text(payload.as_ref())
        );
        std::panic::resume_unwind(payload);
    }
}

/// The seeds and sizes [`check`] would run, for a property that is a future
/// and runs its cases itself, each under a [`Named`] guard.
pub(crate) fn cases(name: &str, cases: usize, max_size: usize) -> Vec<(u64, usize)> {
    if let Some(one) = replay(name) {
        return vec![one];
    }
    let mut seeds = Rng::new(fnv1a(name));
    (0..cases * deep())
        .map(|case| (seeds.next(), 1 + case % max_size.max(1)))
        .collect()
}

/// Names the case it guards if a panic unwinds through it.
pub(crate) struct Named<'a>(pub(crate) &'a str, pub(crate) u64, pub(crate) usize);

impl Drop for Named<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let Self(name, seed, size) = self;
            eprintln!(
                "property `{name}` failed at seed {seed:#x}, size {size}\n\
                 replay: X0NE_PROP_REPLAY='{name}:{seed:#x}:{size}'"
            );
        }
    }
}

// ── JSON, read apart from serde_json and from the automaton ─────────────────

/// A JSON document as it was written: numbers keep their text, objects keep
/// their keys in order and repeated.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum J {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

struct Reader<'a> {
    s: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.at).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn eat(&mut self, word: &str) -> Result<(), String> {
        if self.s[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(format!("expected {word} at byte {}", self.at))
        }
    }

    fn value(&mut self) -> Result<J, String> {
        match self.peek() {
            Some(b'n') => self.eat("null").map(|()| J::Null),
            Some(b't') => self.eat("true").map(|()| J::Bool(true)),
            Some(b'f') => self.eat("false").map(|()| J::Bool(false)),
            Some(b'"') => self.string().map(J::Str),
            Some(b'-' | b'0'..=b'9') => self.number().map(J::Num),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.at += 1;
                    return Ok(J::Arr(items));
                }
                loop {
                    self.ws();
                    items.push(self.value()?);
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(J::Arr(items));
                        }
                        _ => return Err(format!("expected , or ] at byte {}", self.at)),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut members = Vec::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.at += 1;
                    return Ok(J::Obj(members));
                }
                loop {
                    self.ws();
                    let key = self.string()?;
                    self.ws();
                    self.eat(":")?;
                    self.ws();
                    members.push((key, self.value()?));
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(J::Obj(members));
                        }
                        _ => return Err(format!("expected , or }} at byte {}", self.at)),
                    }
                }
            }
            _ => Err(format!("no value at byte {}", self.at)),
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .s
            .get(self.at..self.at + 4)
            .and_then(|d| std::str::from_utf8(d).ok())
            .ok_or("a cut \\u escape")?;
        self.at += 4;
        u32::from_str_radix(digits, 16).map_err(|_| "a \\u escape that is not hex".to_string())
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat("\"")?;
        let mut out = Vec::new();
        loop {
            let b = self.peek().ok_or("an open string")?;
            self.at += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let e = self.peek().ok_or("a cut escape")?;
                    self.at += 1;
                    let c = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                self.eat("\\u")?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("a lone leading surrogate".into());
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            char::from_u32(code).ok_or("a lone surrogate")?
                        }
                        _ => return Err(format!("\\{} is no escape", char::from(e))),
                    };
                    out.extend_from_slice(c.to_string().as_bytes());
                }
                0..=0x1F => return Err("a control character in a string".into()),
                _ => out.push(b),
            }
        }
        String::from_utf8(out).map_err(|_| "a string that is not UTF-8".to_string())
    }

    fn digits(&mut self) -> usize {
        let from = self.at;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.at += 1;
        }
        self.at - from
    }

    fn number(&mut self) -> Result<String, String> {
        let from = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        let leading_zero = self.peek() == Some(b'0');
        let int = self.digits();
        if int == 0 || (leading_zero && int > 1) {
            return Err(format!("a malformed number at byte {from}"));
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            if self.digits() == 0 {
                return Err("a fraction without digits".into());
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if self.digits() == 0 {
                return Err("an exponent without digits".into());
            }
        }
        Ok(String::from_utf8_lossy(&self.s[from..self.at]).into_owned())
    }
}

/// RFC 8259, whole text, nothing before or after the value.
pub(crate) fn parse_json(text: &str) -> Result<J, String> {
    let mut r = Reader {
        s: text.as_bytes(),
        at: 0,
    };
    let v = r.value()?;
    if r.at == text.len() {
        Ok(v)
    } else {
        Err(format!("{} bytes after the value", text.len() - r.at))
    }
}

/// A `serde_json` value as the one text the engine writes it in under `enum`
/// and `const`: compact, keys sorted by byte.
pub(crate) fn canonical_text(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let members: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}:{}", Value::String(k.clone()), canonical_text(&m[k])))
                .collect();
            format!("{{{}}}", members.join(","))
        }
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(canonical_text).collect();
            format!("[{}]", items.join(","))
        }
        scalar => scalar.to_string(),
    }
}

fn is_integer_text(n: &str) -> bool {
    let digits = n.strip_prefix('-').unwrap_or(n);
    !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && (digits == "0" || !digits.starts_with('0'))
}

/// The meaning of the supported subset, written apart from the compiler: what
/// a document must be for `schema`, as the engine promises it (declared
/// properties only, each once; a literal in the one form the engine writes).
pub(crate) fn validates(schema: &Value, doc: &J) -> Result<(), String> {
    let m = schema.as_object().ok_or("a schema that is no object")?;
    let literals: Option<Vec<&Value>> = match (m.get("enum"), m.get("const")) {
        (Some(e), _) => Some(e.as_array().ok_or("enum")?.iter().collect()),
        (None, Some(c)) => Some(vec![c]),
        (None, None) => None,
    };
    if let Some(literals) = literals {
        return literals
            .iter()
            .any(|v| parse_json(&canonical_text(v)).as_ref() == Ok(doc))
            .then_some(())
            .ok_or_else(|| format!("{doc:?} is none of the literals"));
    }
    let types: Vec<&str> = match m.get("type") {
        Some(Value::String(t)) => vec![t.as_str()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
        _ => return Err("a schema without type".into()),
    };
    let mut why = format!("{doc:?} is none of {types:?}");
    for t in types {
        let held = match (t, doc) {
            ("integer", J::Num(n)) if is_integer_text(n) => Ok(()),
            ("null", J::Null)
            | ("boolean", J::Bool(_))
            | ("string", J::Str(_))
            | ("number", J::Num(_)) => Ok(()),
            ("array", J::Arr(items)) => {
                let n = items.len() as u64;
                let min = m.get("minItems").and_then(Value::as_u64).unwrap_or(0);
                let max = m.get("maxItems").and_then(Value::as_u64);
                if n < min || max.is_some_and(|x| n > x) {
                    Err(format!("{n} items outside {min}..={max:?}"))
                } else {
                    items.iter().try_for_each(|x| validates(&m["items"], x))
                }
            }
            ("object", J::Obj(members)) => {
                let empty = serde_json::Map::new();
                let props = m
                    .get("properties")
                    .and_then(Value::as_object)
                    .unwrap_or(&empty);
                let mut seen: Vec<&str> = Vec::new();
                let mut held = Ok(());
                for (k, v) in members {
                    if seen.contains(&k.as_str()) {
                        held = Err(format!("the key {k:?} twice"));
                    } else if let Some(sub) = props.get(k) {
                        held = validates(sub, v);
                    } else {
                        held = Err(format!("the key {k:?} is not declared"));
                    }
                    if held.is_err() {
                        break;
                    }
                    seen.push(k);
                }
                let required = m.get("required").and_then(Value::as_array);
                let missing = required
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .find(|r| !seen.contains(r));
                match (held, missing) {
                    (Ok(()), Some(r)) => Err(format!("the required key {r:?} is missing")),
                    (held, _) => held,
                }
            }
            _ => continue,
        };
        match held {
            Ok(()) => return Ok(()),
            Err(e) => why = e,
        }
    }
    Err(why)
}

// ── Generators: strings, schemas, documents ─────────────────────────────────

/// Strings a key, a literal or a string value may be: empty, structure,
/// quotes and backslashes, controls, multi-byte characters, keys that start
/// alike.
const STRINGS: &[&str] = &[
    "",
    "a",
    "ab",
    "abc",
    "b",
    "name",
    "a\"b",
    "back\\slash",
    "n\u{e9}",
    "\u{6771}\u{4eac}",
    "\u{1f5fc}",
    "line\nbreak",
    "tab\t",
    "\u{1}",
    " ",
    "a b",
    "{",
    "}",
    "[",
    "]",
    ",",
    ":",
    "\"",
    "\\",
    "null",
    "true",
    "1",
    "-",
    "/",
    "\u{7f}",
    "\u{80}",
    "\u{fffd}",
    "e\u{301}",
    "\u{2028}",
    "\u{0}",
];

pub(crate) fn gen_string(rng: &mut Rng) -> String {
    let mut s = (*rng.pick(STRINGS)).to_string();
    if rng.chance(1, 4) {
        s.push_str(rng.pick(STRINGS));
    }
    s
}

fn gen_literal(rng: &mut Rng, depth: usize) -> Value {
    match rng.below(if depth == 0 { 12 } else { 10 }) {
        0 => Value::Null,
        1 => json!(rng.chance(1, 2)),
        // 1 and 10, 1 and 1.5: one literal is the start of another.
        2 => json!(*rng.pick(&[0i64, 1, 10, 100, -1, -10, 7, i64::MAX, i64::MIN])),
        3 => json!(*rng.pick(&[1.5f64, -0.5, 1e20, 1e-7, 0.1, 123.456, -0.0])),
        4 => json!(u64::MAX),
        5..=9 => json!(gen_string(rng)),
        10 => Value::Array(
            (0..rng.below(3))
                .map(|_| gen_literal(rng, depth + 1))
                .collect(),
        ),
        _ => Value::Object(
            (0..rng.below(3))
                .map(|_| (gen_string(rng), gen_literal(rng, depth + 1)))
                .collect(),
        ),
    }
}

/// A schema of the supported subset (`json_schema.rs`): `type` and lists of
/// types, `enum`, `const`, `properties`, `required`, `additionalProperties:
/// false`, `items`, `minItems`, `maxItems`, annotations. `size` bounds its
/// width; its depth stays under 4.
pub(crate) fn gen_schema(rng: &mut Rng, size: usize, depth: usize) -> Value {
    let width = size.clamp(1, 4);
    let mut s = serde_json::Map::new();
    if rng.chance(1, 6) {
        let n = rng.between(1, width + 1);
        let values: Vec<Value> = (0..n).map(|_| gen_literal(rng, 0)).collect();
        s.insert("enum".into(), Value::Array(values));
    } else if rng.chance(1, 12) {
        s.insert("const".into(), gen_literal(rng, 0));
    } else {
        let leaf = depth >= 3;
        let all = [
            "string", "integer", "number", "boolean", "null", "object", "array",
        ];
        let pool = if leaf { &all[..5] } else { &all[..] };
        let mut types: Vec<&str> = Vec::new();
        // Containers more often than their share: they hold the other rules.
        let first = if !leaf && rng.chance(1, 2) {
            *rng.pick(&["object", "array"])
        } else {
            *rng.pick(pool)
        };
        types.push(first);
        while types.len() < 3 && rng.chance(1, 4) {
            let t = *rng.pick(pool);
            if !types.contains(&t) {
                types.push(t);
            }
        }
        if types.contains(&"object") {
            let mut props = serde_json::Map::new();
            for _ in 0..rng.below(width + 1) {
                props.insert(gen_string(rng), gen_schema(rng, size, depth + 1));
            }
            let required: Vec<Value> = props
                .keys()
                .filter(|_| rng.chance(1, 2))
                .map(|k| json!(k))
                .collect();
            if !props.is_empty() || rng.chance(1, 2) {
                s.insert("properties".into(), Value::Object(props));
            }
            if !required.is_empty() || rng.chance(1, 4) {
                s.insert("required".into(), Value::Array(required));
            }
            if rng.chance(1, 2) {
                s.insert("additionalProperties".into(), json!(false));
            }
        }
        if types.contains(&"array") {
            s.insert("items".into(), gen_schema(rng, size, depth + 1));
            let min = rng.below(3);
            if min > 0 || rng.chance(1, 4) {
                s.insert("minItems".into(), json!(min));
            }
            if rng.chance(1, 2) {
                s.insert("maxItems".into(), json!(min + rng.below(3)));
            }
        }
        let ty = if types.len() == 1 && rng.chance(3, 4) {
            json!(types[0])
        } else {
            json!(types)
        };
        s.insert("type".into(), ty);
    }
    if rng.chance(1, 8) {
        s.insert("description".into(), json!(gen_string(rng)));
    }
    if rng.chance(1, 16) {
        s.insert("title".into(), json!("t"));
    }
    Value::Object(s)
}

/// One optional space, where the engine takes one between two tokens.
fn space(rng: &mut Rng, out: &mut String) {
    if rng.chance(1, 5) {
        out.push(' ');
    }
}

/// A string value in one of the forms RFC 8259 gives it: raw, short escapes,
/// `\u` escapes of the Basic Multilingual Plane.
fn write_string(rng: &mut Rng, s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        let code = u32::from(c);
        let short = match c {
            '"' => Some("\\\""),
            '\\' => Some("\\\\"),
            '\n' => Some("\\n"),
            '\t' => Some("\\t"),
            '\r' => Some("\\r"),
            '\u{8}' => Some("\\b"),
            '\u{c}' => Some("\\f"),
            _ => None,
        };
        let must_escape = code < 0x20 || c == '"' || c == '\\';
        match short {
            Some(e) if rng.chance(3, 4) => out.push_str(e),
            _ if code < 0x10000 && (must_escape || rng.chance(1, 8)) => {
                let upper = rng.chance(1, 2);
                let hex = if upper {
                    format!("\\u{code:04X}")
                } else {
                    format!("\\u{code:04x}")
                };
                out.push_str(&hex);
            }
            _ if c == '/' && rng.chance(1, 2) => out.push_str("\\/"),
            _ => out.push(c),
        }
    }
    out.push('"');
}

fn write_number(rng: &mut Rng, integer: bool, out: &mut String) {
    if rng.chance(1, 3) {
        out.push('-');
    }
    if rng.chance(1, 5) {
        out.push('0');
    } else {
        out.push(char::from(b'1' + rng.below(9) as u8));
        // 18 digits before the point at most (`MAX_INT_DIGITS`).
        for _ in 0..*rng.pick(&[0usize, 0, 1, 2, 5, 17]) {
            out.push(char::from(b'0' + rng.below(10) as u8));
        }
    }
    if integer {
        return;
    }
    if rng.chance(1, 2) {
        out.push('.');
        for _ in 0..rng.between(1, 6) {
            out.push(char::from(b'0' + rng.below(10) as u8));
        }
    }
    if rng.chance(1, 3) {
        out.push(*rng.pick(&['e', 'E']));
        match rng.below(3) {
            0 => out.push('+'),
            1 => out.push('-'),
            _ => {}
        }
        // Two digits of exponent at most (`MAX_EXP_DIGITS`).
        for _ in 0..rng.between(1, 2) {
            out.push(char::from(b'0' + rng.below(10) as u8));
        }
    }
}

/// A document `schema` validates, in one of the texts the engine accepts:
/// any key order, one optional space between tokens, any escape of a string
/// value. `schema` is one of [`gen_schema`].
pub(crate) fn write_instance(rng: &mut Rng, schema: &Value, out: &mut String) {
    let m = schema.as_object().expect("a schema");
    if let Some(values) = m.get("enum").and_then(Value::as_array) {
        out.push_str(&canonical_text(rng.pick(values)));
        return;
    }
    if let Some(c) = m.get("const") {
        out.push_str(&canonical_text(c));
        return;
    }
    let types: Vec<&str> = match &m["type"] {
        Value::String(t) => vec![t.as_str()],
        Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
        _ => unreachable!("gen_schema writes a type"),
    };
    match *rng.pick(&types) {
        "null" => out.push_str("null"),
        "boolean" => out.push_str(if rng.chance(1, 2) { "true" } else { "false" }),
        "string" => {
            let text = gen_string(rng);
            write_string(rng, &text, out);
        }
        "integer" => write_number(rng, true, out),
        "number" => write_number(rng, false, out),
        "array" => {
            let min = m.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
            let max = m
                .get("maxItems")
                .and_then(Value::as_u64)
                .map_or(min + 3, |x| x as usize);
            out.push('[');
            let n = rng.between(min, max);
            for i in 0..n {
                if i > 0 {
                    out.push(',');
                }
                space(rng, out);
                write_instance(rng, &m["items"], out);
                space(rng, out);
            }
            if n == 0 {
                space(rng, out);
            }
            out.push(']');
        }
        "object" => {
            let empty = serde_json::Map::new();
            let props = m
                .get("properties")
                .and_then(Value::as_object)
                .unwrap_or(&empty);
            let required: Vec<&str> = m
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let mut keys: Vec<&String> = props
                .keys()
                .filter(|k| required.contains(&k.as_str()) || rng.chance(1, 2))
                .collect();
            for i in (1..keys.len()).rev() {
                keys.swap(i, rng.below(i + 1));
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                space(rng, out);
                // A key is a literal: the one text serde_json gives it.
                out.push_str(&Value::String((*k).clone()).to_string());
                space(rng, out);
                out.push(':');
                space(rng, out);
                write_instance(rng, &props[k.as_str()], out);
                space(rng, out);
            }
            if keys.is_empty() {
                space(rng, out);
            }
            out.push('}');
        }
        other => unreachable!("{other}"),
    }
}

/// Text a message may hold to forge a turn: the markers of every template,
/// cut and whole, look-alikes, role lines, controls, what a tokenizer's
/// normalizer folds. Characters a compiler or a reviewer would not see are
/// written as escapes.
pub(crate) fn nasty_text(rng: &mut Rng, size: usize, markers: &[&str]) -> String {
    const FRAGMENTS: &[&str] = &[
        "hello",
        " ",
        "\n",
        "\n\n",
        "system\n",
        "assistant\n",
        "user\n",
        "\n<|system|>\n",
        "<|assistant|>\n",
        "<|user|>\n",
        "<think>",
        "</think>",
        "\u{0}",
        "\u{1}",
        "\u{7f}",
        "\u{feff}",
        "\u{200b}",
        "\u{200d}",
        "\u{202e}",
        "\u{fffd}",
        "e\u{301}",
        "\u{e9}",
        "\u{1f600}",
        "\u{10ffff}",
        // Fullwidth and small forms NFKC folds to ASCII.
        "\u{ff1c}\u{ff5c}im_end\u{ff5c}\u{ff1e}",
        "\u{ff1c}\u{ff5c}im_start\u{ff5c}\u{ff1e}system\n",
        "\u{fe64}s\u{fe65}",
        "\u{ff3b}INST\u{ff3d}",
        "\u{ff3b}/INST\u{ff3d}",
        // Cyrillic and Greek letters that look Latin.
        "<|\u{456}m_end|>",
        "[\u{399}NST]",
        "<",
        "|",
        ">",
        "[",
        "]",
        "/",
        "<|",
        "|>",
        "<|im_",
        "end|>",
        "start|>",
        "[/",
        "INST]",
        "</",
        "s>",
        "<s",
    ];
    let mut out = String::new();
    for _ in 0..rng.between(0, 2 + size) {
        match rng.below(10) {
            0..=2 if !markers.is_empty() => out.push_str(rng.pick(markers)),
            3 | 4 if !markers.is_empty() => {
                // A marker cut on a character, or with one character inside.
                let m = *rng.pick(markers);
                let cut = rng.between(1, m.len() - 1);
                match rng.below(4) {
                    0 | 1 => out.push_str(&m[..cut]),
                    2 => out.push_str(&m[cut..]),
                    _ => {
                        out.push_str(&m[..cut]);
                        out.push_str(rng.pick(&["\u{200b}", "\u{0}", " ", "\u{301}"]));
                        out.push_str(&m[cut..]);
                    }
                }
            }
            _ => out.push_str(rng.pick(FRAGMENTS)),
        }
    }
    out
}

// ── Tokenizers ──────────────────────────────────────────────────────────────

/// GPT-2's printable stand-in for byte `b` (`constrain.rs`).
pub(crate) fn gpt2_char(b: u8) -> char {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    if printable(b) {
        return char::from(b);
    }
    let shifted = (0..=b).filter(|&x| !printable(x)).count() as u32;
    char::from_u32(255 + shifted).expect("a character")
}

fn added_token(id: usize, content: &str, special: bool) -> Value {
    json!({"id": id, "content": content, "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": special})
}

/// A GPT-2 byte-level BPE: one token per byte (id = byte), then `pieces`
/// (longer than a byte, distinct) as tokens of their own, then `added`
/// (content, special) at the ids that follow. No merge: text is encoded
/// byte by byte, and any token decodes to its bytes.
pub(crate) fn byte_bpe(pieces: &[Vec<u8>], added: &[(&str, bool)], normalizer: Value) -> Tokenizer {
    let mut vocab = serde_json::Map::new();
    for b in 0..=255u8 {
        vocab.insert(gpt2_char(b).to_string(), u32::from(b).into());
    }
    for p in pieces.iter().filter(|p| p.len() > 1) {
        let word: String = p.iter().map(|&b| gpt2_char(b)).collect();
        let id = vocab.len();
        vocab.entry(word).or_insert_with(|| id.into());
    }
    let n = vocab.len();
    let added: Vec<Value> = added
        .iter()
        .enumerate()
        .map(|(i, (content, special))| added_token(n + i, content, *special))
        .collect();
    let byte_level = json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true,
    });
    let j = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": normalizer, "pre_tokenizer": byte_level.clone(),
        "post_processor": null, "decoder": byte_level,
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    });
    Tokenizer::from_bytes(serde_json::to_vec(&j).expect("json")).expect("a tokenizer")
}

/// The end of an answer in [`json_tokenizer`].
pub(crate) const EOS: &str = "<|im_end|>";

/// [`byte_bpe`] over the pieces of `constrain::tests::varied_pieces` (structure,
/// bridges, escapes, partial characters, words), ChatML's markers as special
/// tokens, a special token that ends nothing and a plain added one: what the
/// constraint, the stream and the service properties generate with.
pub(crate) fn json_tokenizer() -> Tokenizer {
    byte_bpe(
        &crate::constrain::tests::varied_pieces(),
        &[
            ("<|im_start|>", true),
            (EOS, true),
            ("<|pad|>", true),
            ("<think>", false),
        ],
        Value::Null,
    )
}

pub(crate) fn id_of(tokenizer: &Tokenizer, token: &str) -> u32 {
    tokenizer.token_to_id(token).expect("a token")
}

/// The bytes each id of a [`byte_bpe`] tokenizer decodes to, read from the
/// tokenizer and not from the engine's table: a special token has none.
pub(crate) fn bytes_by_id(tokenizer: &Tokenizer) -> Vec<Option<Vec<u8>>> {
    let added = tokenizer.get_added_tokens_decoder();
    let byte_of: std::collections::HashMap<char, u8> =
        (0..=255u8).map(|b| (gpt2_char(b), b)).collect();
    (0..tokenizer.get_vocab_size(true) as u32)
        .map(|id| match added.get(&id) {
            Some(a) if a.special => None,
            Some(a) => Some(a.content.clone().into_bytes()),
            None => tokenizer
                .id_to_token(id)
                .map(|piece| piece.chars().map(|c| byte_of[&c]).collect()),
        })
        .collect()
}

/// Every id of `table` that is text, with its bytes.
pub(crate) fn text_tokens(table: &TokenTable) -> Vec<(u32, Vec<u8>)> {
    (0..table.len() as u32)
        .filter_map(|id| table.bytes(id).map(|b| (id, b.to_vec())))
        .collect()
}

/// `bytes` as tokens of `tokens`, cut at random: any token that starts the
/// rest may come next, so a character or an escape straddles two of them.
pub(crate) fn tokenize_at_random(
    rng: &mut Rng,
    tokens: &[(u32, Vec<u8>)],
    bytes: &[u8],
) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let fits: Vec<&(u32, Vec<u8>)> = tokens
            .iter()
            .filter(|(_, b)| bytes[at..].starts_with(b))
            .collect();
        let (id, b) = *rng.pick(&fits);
        ids.push(*id);
        at += b.len();
    }
    ids
}

// ── Scripted sessions and a scripted model ──────────────────────────────────

/// What a [`RowSession`] was asked.
#[derive(Default, Debug)]
pub(crate) struct SessionLog {
    pub(crate) fed: Vec<u32>,
    pub(crate) finished: usize,
    pub(crate) rows_read: usize,
}

/// A session whose row after the tokens fed so far is `next_row`'s.
pub(crate) struct RowSession<F: FnMut(&[u32]) -> Vec<f32>> {
    row: Vec<f32>,
    fed: Vec<u32>,
    next_row: F,
    log: Rc<RefCell<SessionLog>>,
}

impl<F: FnMut(&[u32]) -> Vec<f32>> RowSession<F> {
    pub(crate) fn new(mut next_row: F, log: Rc<RefCell<SessionLog>>) -> Self {
        Self {
            row: next_row(&[]),
            fed: Vec::new(),
            next_row,
            log,
        }
    }
}

impl<F: FnMut(&[u32]) -> Vec<f32>> LogitsSession for RowSession<F> {
    fn logits(&self) -> &[f32] {
        self.log.borrow_mut().rows_read += 1;
        &self.row
    }
    fn step(&mut self, token: u32) -> anyhow::Result<()> {
        self.fed.push(token);
        self.log.borrow_mut().fed.push(token);
        self.row = (self.next_row)(&self.fed);
        Ok(())
    }
    fn finish(self) -> anyhow::Result<()> {
        self.log.borrow_mut().finished += 1;
        Ok(())
    }
}

/// A row that picks `token` greedily.
pub(crate) fn peaked(vocab: usize, token: u32) -> Vec<f32> {
    let mut row = vec![0.0f32; vocab];
    row[token as usize] = 10.0;
    row
}

/// A model for the service properties: the real loop (`stepped`) over
/// [`json_tokenizer`], with rows drawn from the prompt's ids and the tokens
/// fed. The same request gives the same answer on any instance, so one
/// instance is the reference of what another served.
pub(crate) struct ScriptedModel {
    tokenizer: Tokenizer,
    eos: Vec<u32>,
    table: TableCache,
    context: usize,
    /// Decode steps run, all requests together.
    pub(crate) steps: Arc<AtomicUsize>,
    /// Slept at every step: a generation a test can leave half-way.
    pub(crate) step_delay: std::time::Duration,
}

impl ScriptedModel {
    pub(crate) fn new() -> Self {
        let tokenizer = crate::prompt::text_only(json_tokenizer());
        let eos = vec![id_of(&tokenizer, EOS)];
        Self {
            tokenizer,
            eos,
            table: TableCache::default(),
            context: 4096,
            steps: Arc::new(AtomicUsize::new(0)),
            step_delay: std::time::Duration::ZERO,
        }
    }
}

impl LoadedModel for ScriptedModel {
    fn generate(&self, prompt: &Prompt, max_tokens: usize) -> anyhow::Result<(String, (u32, u32))> {
        let c = self.generate_with(prompt, &GenerationParams::greedy(max_tokens))?;
        Ok((c.text, (c.prompt_tokens, c.completion_tokens)))
    }
    fn generate_with(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
    ) -> anyhow::Result<Completion> {
        self.generate_streaming(prompt, params, &mut NoSink)
    }
    fn generate_whole(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        self.generate_streaming(prompt, params, sink)
    }
    fn generate_streaming(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        let req =
            crate::stepped::prepare(&self.tokenizer, prompt, params.max_tokens, self.context)?;
        let table = self.table.get(&self.tokenizer, &self.eos, params)?;
        let vocab = self.tokenizer.get_vocab_size(true);
        let eos = self.eos[0];
        let seed = req.prompt_ids.iter().fold(0x5eed_u64, |h, &id| {
            (h ^ u64::from(id)).wrapping_mul(0x0100_0000_01b3)
        });
        let (steps, delay) = (self.steps.clone(), self.step_delay);
        let next_row = move |fed: &[u32]| {
            if !fed.is_empty() {
                steps.fetch_add(1, SeqCst);
                if !delay.is_zero() {
                    std::thread::sleep(delay);
                }
            }
            let mut rng = Rng::new(seed ^ (fed.len() as u64).wrapping_mul(0x9E37_79B9));
            let mut row: Vec<f32> = (0..vocab).map(|_| rng.unit()).collect();
            // Mostly printable ASCII, sometimes any token, an end one time in 24.
            let favourite = if rng.chance(3, 4) {
                rng.between(0x20, 0x7E)
            } else {
                rng.below(vocab)
            };
            row[favourite] += 4.0;
            if rng.chance(1, 24) {
                row[eos as usize] += 8.0;
            }
            row
        };
        let log = Rc::new(RefCell::new(SessionLog::default()));
        crate::stepped::run_with_table(
            &self.tokenizer,
            &self.eos,
            table,
            &req,
            params,
            sink,
            |_| Ok(RowSession::new(next_row, log)),
        )
    }
    fn model_digest_hex(&self) -> String {
        "5c21b7ed".into()
    }
    fn compute_device(&self) -> String {
        "cpu".into()
    }
    fn isolation(&self) -> Isolation {
        Isolation::NONE_OF_IT
    }
}

// ── The kit holds itself to its own rules ───────────────────────────────────

/// The reader agrees with `serde_json` on what is JSON, on documents it wrote
/// and on those texts cut or doubled; and it keeps what `serde_json` folds: a
/// key written twice.
#[test]
fn the_independent_reader_agrees_with_serde_json_on_what_is_json() {
    check("kit_reader", 400, 6, |rng, size| {
        let schema = gen_schema(rng, size, 0);
        let mut text = String::new();
        write_instance(rng, &schema, &mut text);
        let ours = parse_json(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
        assert!(serde_json::from_str::<Value>(&text).is_ok(), "{text}");
        validates(&schema, &ours).unwrap_or_else(|e| panic!("{e}: {text} for {schema}"));
        let mut cut = rng.below(text.len() + 1);
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        for broken in [text[..cut].to_string(), format!("{text}{text}")] {
            assert_eq!(
                parse_json(&broken).is_ok(),
                serde_json::from_str::<Value>(&broken).is_ok(),
                "{broken}"
            );
        }
    });
    let twice = parse_json(r#"{"a":1,"a":2}"#).unwrap();
    assert_eq!(
        twice,
        J::Obj(vec![
            ("a".into(), J::Num("1".into())),
            ("a".into(), J::Num("2".into()))
        ])
    );
    let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
    assert!(validates(&schema, &twice).is_err());
    assert!(parse_json(r#""\ud800""#).is_err(), "a lone surrogate");
    assert_eq!(parse_json(r#""😀""#), Ok(J::Str("\u{1f600}".into())));
}

/// A failing case names its seed and the smallest size that still fails, and
/// the panic goes on to the test harness.
#[test]
fn a_failing_property_reports_its_seed_and_its_smallest_size() {
    let sizes = RefCell::new(Vec::new());
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check("kit_fails_from_size_3", 20, 8, |_, size| {
            sizes.borrow_mut().push(size);
            assert!(size < 3, "size {size}");
        });
    }));
    assert_eq!(panic_text(caught.unwrap_err().as_ref()), "size 3");
    // Sizes 1 and 2 pass, 3 fails, then 1 and 2 again: nothing smaller fails.
    assert_eq!(*sizes.borrow(), [1, 2, 3, 1, 2]);
}
