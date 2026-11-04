// SPDX-License-Identifier: Apache-2.0
//! `response_format: json_schema`: a JSON Schema subset compiled to a byte
//! automaton whose every prefix can be completed; the rest is refused by name.
//! Compiled into `x0ne-sdk` too (`#[path]`): the client checks with it.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// A schema's compact serialization, at most.
pub const MAX_SCHEMA_BYTES: usize = 16 * 1024;
/// Nesting levels of subschemas below the root (OpenAI's limit, 2026-10).
pub const MAX_DEPTH: usize = 10;
/// Properties of one object: their set fits a `u64`.
pub const MAX_PROPERTIES: usize = 64;
/// Values of one `enum`.
pub const MAX_ENUM: usize = 256;
/// Compiled subschemas in one schema.
pub const MAX_NODES: usize = 512;
/// Digits before a number's point: what i64 and u64 always hold.
pub const MAX_INT_DIGITS: u8 = 18;
/// Digits of an exponent. Measured 2026-10-30: `504e+657853570652060901548415`
/// is RFC 8259 JSON that serde_json refuses ("number out of range"); with
/// these two bounds every number stays under 1e119.
pub const MAX_EXP_DIGITS: u8 = 2;

/// Spaces in a row between two tokens of a container (Outlines' `[ ]?`).
/// 2026-10-30, Qwen3-0.6B greedy, 4 prompts: with none, 2 answers degenerate
/// (`"name":", "`); with one, none.
pub const MAX_WHITESPACE: u8 = 1;
/// The whitespace taken between tokens.
const WHITESPACE: &[u8] = b" ";

/// A container frame per level and one value frame: the stack never grows past it.
const MAX_FRAMES: usize = MAX_DEPTH + 4;

const ANNOTATIONS: &[&str] = &[
    "$schema",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
];
const KEYWORDS: &[&str] = &[
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
];

type NodeId = u16;

#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// One of these canonical serializations, sorted, distinct.
    Literals(Vec<Box<[u8]>>),
    String,
    Number {
        integer: bool,
    },
    /// Keys serialized with their quotes, sorted; each key's value node;
    /// the required keys as bits of their sorted index.
    Object {
        keys: Vec<Box<[u8]>>,
        values: Vec<NodeId>,
        required: u64,
    },
    Array {
        items: NodeId,
        min: u32,
        max: Option<u32>,
    },
    /// The alternatives of a `type` list: each starts with its own bytes.
    Union(Vec<NodeId>),
}

/// A compiled schema and the digest a receipt signs.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonSchema {
    nodes: Vec<Node>,
    root: NodeId,
    sha256_hex: String,
}

/// What a text is to a schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accepted {
    /// A whole document the schema validates.
    Complete,
    /// The start of one: what an answer cut by `max_tokens` is.
    Prefix,
    No,
}

/// The form the digest is taken over: compact, object keys sorted by byte
/// at every level, numbers and strings as `serde_json` writes them.
pub fn canonical(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_canonical(v, &mut out);
    out
}

fn write_canonical(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push(b'{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend(serde_json::to_vec(k).unwrap_or_default());
                out.push(b':');
                write_canonical(&m[k.as_str()], out);
            }
            out.push(b'}');
        }
        Value::Array(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(x, out);
            }
            out.push(b']');
        }
        scalar => out.extend(serde_json::to_vec(scalar).unwrap_or_default()),
    }
}

/// SHA-256 of a schema's canonical form, lowercase hex.
pub fn schema_sha256_hex(schema: &Value) -> String {
    Sha256::digest(canonical(schema))
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `s` cut to 64 bytes on a character boundary, for an error message.
pub fn quoted(s: &str) -> String {
    if s.len() <= 64 {
        return s.to_string();
    }
    let mut end = 64;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// The end of a schema path, for an error message.
fn short(path: &str) -> String {
    if path.len() <= 96 {
        return path.to_string();
    }
    let mut start = path.len() - 96;
    while !path.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &path[start..])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    Null,
    Boolean,
    String,
    Integer,
    Number,
    Object,
    Array,
}

impl Ty {
    fn holds(self, v: &Value) -> bool {
        match self {
            Ty::Null => v.is_null(),
            Ty::Boolean => v.is_boolean(),
            Ty::String => v.is_string(),
            Ty::Integer => v.is_i64() || v.is_u64() || v.as_f64().is_some_and(|f| f.fract() == 0.0),
            Ty::Number => v.is_number(),
            Ty::Object => v.is_object(),
            Ty::Array => v.is_array(),
        }
    }
}

fn types(t: Option<&Value>) -> Result<Option<Vec<Ty>>, String> {
    let names: Vec<&Value> = match t {
        None => return Ok(None),
        Some(Value::Array(a)) if !a.is_empty() => a.iter().collect(),
        Some(v @ Value::String(_)) => vec![v],
        Some(_) => return Err("type: a type name or a non-empty list of them".into()),
    };
    let mut out = Vec::new();
    for n in names {
        let ty = match n.as_str() {
            Some("null") => Ty::Null,
            Some("boolean") => Ty::Boolean,
            Some("string") => Ty::String,
            Some("integer") => Ty::Integer,
            Some("number") => Ty::Number,
            Some("object") => Ty::Object,
            Some("array") => Ty::Array,
            Some(other) => {
                return Err(format!(
                    "type: `{}` is not a JSON Schema type",
                    quoted(other)
                ))
            }
            None => return Err("type: a type name is a string".into()),
        };
        if !out.contains(&ty) {
            out.push(ty);
        }
    }
    Ok(Some(out))
}

struct Compiler {
    nodes: Vec<Node>,
}

impl Compiler {
    fn push(&mut self, n: Node) -> Result<NodeId, String> {
        if self.nodes.len() >= MAX_NODES {
            return Err(format!("schema: more than {MAX_NODES} subschemas"));
        }
        self.nodes.push(n);
        Ok((self.nodes.len() - 1) as NodeId)
    }

    fn node(&mut self, s: &Value, path: &str, depth: usize) -> Result<NodeId, String> {
        let at = |why: String| format!("{}: {why}", short(path));
        if depth > MAX_DEPTH {
            return Err(format!("schema: nested more than {MAX_DEPTH} levels"));
        }
        let Value::Object(m) = s else {
            return Err(at("a schema is an object".into()));
        };
        if let Some(k) = m
            .keys()
            .find(|k| !KEYWORDS.contains(&k.as_str()) && !ANNOTATIONS.contains(&k.as_str()))
        {
            return Err(at(format!("`{}` is not supported", quoted(k))));
        }
        let types = types(m.get("type")).map_err(at)?;
        let literals = match (m.get("enum"), m.get("const")) {
            (Some(_), Some(_)) => return Err(at("enum and const together".into())),
            (Some(Value::Array(v)), None) => Some((v.as_slice(), "enum")),
            (Some(_), None) => return Err(at("enum: an array of values".into())),
            (None, Some(c)) => Some((std::slice::from_ref(c), "const")),
            (None, None) => None,
        };
        if let Some((values, word)) = literals {
            for k in [
                "properties",
                "required",
                "additionalProperties",
                "items",
                "minItems",
                "maxItems",
            ] {
                if m.contains_key(k) {
                    return Err(at(format!("{k} beside {word} is not supported")));
                }
            }
            if values.is_empty() {
                return Err(at(format!("{word}: no value")));
            }
            if values.len() > MAX_ENUM {
                return Err(at(format!(
                    "{word}: {} values, more than {MAX_ENUM} values",
                    values.len()
                )));
            }
            if let Some(types) = &types {
                if values.iter().any(|v| !types.iter().any(|t| t.holds(v))) {
                    return Err(at(format!("{word}: a value is not of the declared type")));
                }
            }
            let mut lits: Vec<Box<[u8]>> = values
                .iter()
                .map(|v| canonical(v).into_boxed_slice())
                .collect();
            lits.sort();
            lits.dedup();
            return self.push(Node::Literals(lits));
        }
        let Some(types) = types else {
            return Err(at("type: a schema needs `type`, `enum` or `const`".into()));
        };
        for (k, ty, what) in [
            ("properties", Ty::Object, "objects"),
            ("required", Ty::Object, "objects"),
            ("additionalProperties", Ty::Object, "objects"),
            ("items", Ty::Array, "arrays"),
            ("minItems", Ty::Array, "arrays"),
            ("maxItems", Ty::Array, "arrays"),
        ] {
            if m.contains_key(k) && !types.contains(&ty) {
                return Err(at(format!("{k} applies to {what} only")));
            }
        }
        let mut alts = Vec::new();
        let mut words: Vec<Box<[u8]>> = Vec::new();
        for t in &types {
            match t {
                Ty::Null => words.push(Box::from(&b"null"[..])),
                Ty::Boolean => {
                    words.push(Box::from(&b"false"[..]));
                    words.push(Box::from(&b"true"[..]));
                }
                Ty::String => alts.push(self.push(Node::String)?),
                Ty::Integer if types.contains(&Ty::Number) => {}
                Ty::Integer => alts.push(self.push(Node::Number { integer: true })?),
                Ty::Number => alts.push(self.push(Node::Number { integer: false })?),
                Ty::Object => alts.push(self.object(m, path, depth)?),
                Ty::Array => alts.push(self.array(m, path, depth)?),
            }
        }
        if !words.is_empty() {
            words.sort();
            alts.push(self.push(Node::Literals(words))?);
        }
        match alts.as_slice() {
            [one] => Ok(*one),
            _ => self.push(Node::Union(alts)),
        }
    }

    fn object(
        &mut self,
        m: &Map<String, Value>,
        path: &str,
        depth: usize,
    ) -> Result<NodeId, String> {
        let at = |why: String| format!("{}: {why}", short(path));
        match m.get("additionalProperties") {
            None | Some(Value::Bool(false)) => {}
            Some(_) => {
                return Err(at(
                    "additionalProperties: only false is enforced; the engine \
                     writes declared properties only"
                        .into(),
                ))
            }
        }
        let empty = Map::new();
        let props = match m.get("properties") {
            None => &empty,
            Some(Value::Object(p)) => p,
            Some(_) => return Err(at("properties: an object of schemas".into())),
        };
        if props.len() > MAX_PROPERTIES {
            return Err(at(format!(
                "properties: {} properties, more than {MAX_PROPERTIES} properties",
                props.len()
            )));
        }
        let mut entries: Vec<(Box<[u8]>, &String, &Value)> = props
            .iter()
            .map(|(k, v)| {
                (
                    serde_json::to_vec(k).unwrap_or_default().into_boxed_slice(),
                    k,
                    v,
                )
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut values = Vec::with_capacity(entries.len());
        for (_, name, sub) in &entries {
            values.push(self.node(sub, &format!("{path}.{name}"), depth + 1)?);
        }
        let mut required = 0u64;
        match m.get("required") {
            None => {}
            Some(Value::Array(names)) => {
                for name in names {
                    let Some(name) = name.as_str() else {
                        return Err(at("required: property names, as strings".into()));
                    };
                    let Some(i) = entries.iter().position(|e| e.1 == name) else {
                        return Err(at(format!(
                            "required: `{}` is not in properties",
                            quoted(name)
                        )));
                    };
                    required |= 1u64 << i;
                }
            }
            Some(_) => return Err(at("required: an array of property names".into())),
        }
        let keys = entries.into_iter().map(|e| e.0).collect();
        self.push(Node::Object {
            keys,
            values,
            required,
        })
    }

    fn array(
        &mut self,
        m: &Map<String, Value>,
        path: &str,
        depth: usize,
    ) -> Result<NodeId, String> {
        let at = |why: String| format!("{}: {why}", short(path));
        let Some(items) = m.get("items") else {
            return Err(at(
                "items: an array needs `items`; arrays of any value are not supported".into(),
            ));
        };
        let items = self.node(items, &format!("{path}[]"), depth + 1)?;
        let count = |k: &str| -> Result<Option<u32>, String> {
            match m.get(k) {
                None => Ok(None),
                Some(v) => v
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .map(Some)
                    .ok_or_else(|| at(format!("{k}: a non-negative integer"))),
            }
        };
        let min = count("minItems")?.unwrap_or(0);
        let max = count("maxItems")?;
        if max.is_some_and(|x| x < min) {
            return Err(at("maxItems: less than minItems".into()));
        }
        self.push(Node::Array { items, min, max })
    }
}

impl JsonSchema {
    /// Compiles `schema` or names, in under 256 bytes, why it is refused.
    pub fn compile(schema: &Value) -> Result<Self, String> {
        let compact = canonical(schema);
        if compact.len() > MAX_SCHEMA_BYTES {
            return Err(format!(
                "schema: {} bytes compact, more than {MAX_SCHEMA_BYTES} bytes",
                compact.len()
            ));
        }
        let mut c = Compiler { nodes: Vec::new() };
        let root = c.node(schema, "schema", 0)?;
        Ok(Self {
            nodes: c.nodes,
            root,
            sha256_hex: Sha256::digest(&compact)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        })
    }

    /// SHA-256 of the schema's canonical form ([`canonical`]).
    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }

    pub fn matcher(&self) -> Matcher<'_> {
        Matcher {
            schema: self,
            stack: [Frame::Value(self.root); MAX_FRAMES],
            len: 1,
            failed: false,
            ws: 0,
        }
    }

    pub fn accepts(&self, text: &str) -> Accepted {
        let mut m = self.matcher();
        if !m.feed_bytes(text.as_bytes()) {
            Accepted::No
        } else if m.complete() {
            Accepted::Complete
        } else {
            Accepted::Prefix
        }
    }

    /// Whether a value of `n` may start with `b`.
    fn starts(&self, n: NodeId, b: u8) -> bool {
        match &self.nodes[n as usize] {
            Node::Literals(l) => l.iter().any(|x| x.first() == Some(&b)),
            Node::String => b == b'"',
            Node::Number { .. } => b == b'-' || b.is_ascii_digit(),
            Node::Object { .. } => b == b'{',
            Node::Array { .. } => b == b'[',
            Node::Union(_) => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Str {
    Plain,
    Escape,
    Hex { n: u8, v: u16 },
    Utf8 { need: u8, lo: u8, hi: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Num {
    Start,
    Minus,
    Zero,
    Int(u8),
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Obj {
    Open,
    Key { lo: u16, hi: u16, pos: u16 },
    Colon(u16),
    Next,
    Comma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arr {
    Open,
    Next,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frame {
    /// A value of this node, not started.
    Value(NodeId),
    /// The literals `lo..hi` of the node share the `pos` bytes read.
    Lit {
        node: NodeId,
        lo: u16,
        hi: u16,
        pos: u16,
    },
    Str(Str),
    Num {
        integer: bool,
        at: Num,
    },
    Obj {
        node: NodeId,
        seen: u64,
        at: Obj,
    },
    Arr {
        node: NodeId,
        count: u32,
        at: Arr,
    },
}

enum StrStep {
    Go(Str),
    Close,
    No,
}

fn string_step(s: Str, b: u8) -> StrStep {
    use StrStep::{Close, Go, No};
    let utf8 = |need, lo, hi| Go(Str::Utf8 { need, lo, hi });
    match s {
        Str::Plain => match b {
            b'"' => Close,
            b'\\' => Go(Str::Escape),
            0x20..=0x7F => Go(Str::Plain),
            // RFC 3629: no overlong form, no surrogate, nothing past U+10FFFF.
            0xC2..=0xDF => utf8(1, 0x80, 0xBF),
            0xE0 => utf8(2, 0xA0, 0xBF),
            0xE1..=0xEC | 0xEE..=0xEF => utf8(2, 0x80, 0xBF),
            0xED => utf8(2, 0x80, 0x9F),
            0xF0 => utf8(3, 0x90, 0xBF),
            0xF1..=0xF3 => utf8(3, 0x80, 0xBF),
            0xF4 => utf8(3, 0x80, 0x8F),
            // Control bytes (RFC 8259), a continuation byte alone, the rest.
            _ => No,
        },
        Str::Utf8 { need, lo, hi } => {
            if !(lo..=hi).contains(&b) {
                No
            } else if need == 1 {
                Go(Str::Plain)
            } else {
                utf8(need - 1, 0x80, 0xBF)
            }
        }
        Str::Escape => match b {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Go(Str::Plain),
            b'u' => Go(Str::Hex { n: 0, v: 0 }),
            _ => No,
        },
        Str::Hex { n, v } => {
            let Some(d) = char::from(b).to_digit(16) else {
                return No;
            };
            let v = (v << 4) | d as u16;
            // A surrogate escape is refused at its second digit (\uD8 to \uDF),
            // so no accepted prefix is a dead end.
            if n == 1 && (v & 0xF8) == 0xD8 {
                No
            } else if n == 3 {
                Go(Str::Plain)
            } else {
                Go(Str::Hex { n: n + 1, v })
            }
        }
    }
}

/// RFC 8259's number within [`MAX_INT_DIGITS`] and [`MAX_EXP_DIGITS`]; an
/// `integer` has neither fraction nor exponent.
fn num_step(integer: bool, at: Num, b: u8) -> Option<Num> {
    let digit = b.is_ascii_digit();
    Some(match (at, b) {
        (Num::Start, b'-') => Num::Minus,
        (Num::Start | Num::Minus, b'0') => Num::Zero,
        (Num::Start | Num::Minus, _) if digit => Num::Int(1),
        (Num::Int(n), _) if digit && n < MAX_INT_DIGITS => Num::Int(n + 1),
        (Num::Zero | Num::Int(_), b'.') if !integer => Num::Dot,
        (Num::Zero | Num::Int(_) | Num::Frac, b'e' | b'E') if !integer => Num::Exp,
        (Num::Dot | Num::Frac, _) if digit => Num::Frac,
        (Num::Exp, b'+' | b'-') => Num::ExpSign,
        (Num::Exp | Num::ExpSign, _) if digit => Num::ExpDigits(1),
        (Num::ExpDigits(n), _) if digit && n < MAX_EXP_DIGITS => Num::ExpDigits(n + 1),
        _ => return None,
    })
}

fn num_ends(at: Num) -> bool {
    matches!(at, Num::Zero | Num::Int(_) | Num::Frac | Num::ExpDigits(_))
}

/// The sub-range of `list[lo..hi]` (sorted, sharing their first `pos` bytes)
/// whose byte at `pos` is `b`.
fn narrow(list: &[Box<[u8]>], lo: u16, hi: u16, pos: u16, b: u8) -> (u16, u16) {
    let r = &list[lo as usize..hi as usize];
    let key = |x: &[u8]| x.get(pos as usize).map_or(-1i16, |&c| i16::from(c));
    let a = r.partition_point(|x| key(x) < i16::from(b));
    let z = r.partition_point(|x| key(x) <= i16::from(b));
    (lo + a as u16, lo + z as u16)
}

/// The automaton's state after the bytes it was fed: a copy is a fork.
#[derive(Debug, Clone, Copy)]
pub struct Matcher<'a> {
    schema: &'a JsonSchema,
    stack: [Frame; MAX_FRAMES],
    len: u8,
    failed: bool,
    /// Whitespace bytes taken since the last token.
    ws: u8,
}

impl Matcher<'_> {
    /// Takes `b`, or refuses it and every byte after.
    pub fn feed(&mut self, b: u8) -> bool {
        if self.failed || !self.step(b) {
            self.failed = true;
            return false;
        }
        true
    }

    pub fn feed_bytes(&mut self, s: &[u8]) -> bool {
        s.iter().all(|&b| self.feed(b))
    }

    /// A whole document: what was fed may end here.
    pub fn complete(&self) -> bool {
        if self.failed {
            return false;
        }
        let mut len = self.len as usize;
        while len > 0 {
            match self.stack[len - 1] {
                Frame::Lit { node, lo, pos, .. } => match &self.schema.nodes[node as usize] {
                    Node::Literals(l) if l[lo as usize].len() == pos as usize => len -= 1,
                    _ => return false,
                },
                Frame::Num { at, .. } if num_ends(at) => len -= 1,
                _ => return false,
            }
        }
        true
    }

    /// Complete, and no byte may follow.
    pub fn done(&self) -> bool {
        self.complete()
            && (0..=255u8).all(|b| {
                let mut m = *self;
                !m.feed(b)
            })
    }

    /// Inside a string value, between whole characters: a quote, a backslash
    /// or a control byte changes the state, any other whole character keeps it.
    pub fn in_plain_string(&self) -> bool {
        !self.failed && self.len > 0 && self.stack[self.len as usize - 1] == Frame::Str(Str::Plain)
    }

    fn push(&mut self, f: Frame) -> bool {
        if self.len as usize >= MAX_FRAMES {
            return false;
        }
        self.stack[self.len as usize] = f;
        self.len += 1;
        true
    }

    /// `b` is whitespace: taken between two tokens of a container, after
    /// closing the number or literal it ends; `None` inside a string or a
    /// literal, where the byte is read as any other.
    fn space(&mut self) -> Option<bool> {
        let mut m = *self;
        loop {
            if m.len == 0 {
                return Some(false);
            }
            let between = match m.stack[m.len as usize - 1] {
                Frame::Str(_)
                | Frame::Obj {
                    at: Obj::Key { .. },
                    ..
                } => return None,
                Frame::Lit { node, lo, pos, .. } => {
                    let Node::Literals(l) = &m.schema.nodes[node as usize] else {
                        return Some(false);
                    };
                    // A literal's end; a string literal goes on reading.
                    if pos == 0 || l[lo as usize].len() != pos as usize {
                        return None;
                    }
                    m.len -= 1;
                    continue;
                }
                Frame::Num { at, .. } => {
                    if !num_ends(at) {
                        return Some(false);
                    }
                    m.len -= 1;
                    continue;
                }
                // A value of a container, not started; the root's takes none.
                Frame::Value(_) => m.len > 1,
                Frame::Obj { .. } | Frame::Arr { .. } => true,
            };
            if !between || m.ws >= MAX_WHITESPACE {
                return Some(false);
            }
            m.ws += 1;
            *self = m;
            return Some(true);
        }
    }

    fn step(&mut self, b: u8) -> bool {
        let schema = self.schema;
        if WHITESPACE.contains(&b) {
            if let Some(taken) = self.space() {
                return taken;
            }
        }
        self.ws = 0;
        loop {
            if self.len == 0 {
                return false; // the document is closed: nothing after it
            }
            let top = self.len as usize - 1;
            match self.stack[top] {
                Frame::Value(n) => {
                    let next = match &schema.nodes[n as usize] {
                        Node::Literals(l) => Frame::Lit {
                            node: n,
                            lo: 0,
                            hi: l.len() as u16,
                            pos: 0,
                        },
                        Node::Number { integer } => Frame::Num {
                            integer: *integer,
                            at: Num::Start,
                        },
                        Node::String => {
                            self.stack[top] = Frame::Str(Str::Plain);
                            return b == b'"';
                        }
                        Node::Object { .. } => {
                            self.stack[top] = Frame::Obj {
                                node: n,
                                seen: 0,
                                at: Obj::Open,
                            };
                            return b == b'{';
                        }
                        Node::Array { .. } => {
                            self.stack[top] = Frame::Arr {
                                node: n,
                                count: 0,
                                at: Arr::Open,
                            };
                            return b == b'[';
                        }
                        Node::Union(alts) => match alts.iter().find(|&&a| schema.starts(a, b)) {
                            Some(&a) => Frame::Value(a),
                            None => return false,
                        },
                    };
                    self.stack[top] = next;
                }
                Frame::Lit { node, lo, hi, pos } => {
                    let Node::Literals(l) = &schema.nodes[node as usize] else {
                        return false;
                    };
                    let (a, z) = narrow(l, lo, hi, pos, b);
                    if a < z {
                        self.stack[top] = Frame::Lit {
                            node,
                            lo: a,
                            hi: z,
                            pos: pos + 1,
                        };
                        return true;
                    }
                    // Only a number ends where another value goes on, and
                    // nothing that continues a number closes a container.
                    if l[lo as usize].len() != pos as usize {
                        return false;
                    }
                    self.len -= 1;
                }
                Frame::Str(s) => {
                    return match string_step(s, b) {
                        StrStep::Go(next) => {
                            self.stack[top] = Frame::Str(next);
                            true
                        }
                        StrStep::Close => {
                            self.len -= 1;
                            true
                        }
                        StrStep::No => false,
                    }
                }
                Frame::Num { integer, at } => match num_step(integer, at, b) {
                    Some(next) => {
                        self.stack[top] = Frame::Num { integer, at: next };
                        return true;
                    }
                    None if num_ends(at) => self.len -= 1,
                    None => return false,
                },
                Frame::Obj { node, seen, at } => {
                    let Node::Object {
                        keys,
                        values,
                        required,
                    } = &schema.nodes[node as usize]
                    else {
                        return false;
                    };
                    match at {
                        Obj::Open if b == b'}' => {
                            self.len -= 1;
                            return *required == 0;
                        }
                        Obj::Open | Obj::Comma => {
                            self.stack[top] = Frame::Obj {
                                node,
                                seen,
                                at: Obj::Key {
                                    lo: 0,
                                    hi: keys.len() as u16,
                                    pos: 0,
                                },
                            };
                        }
                        Obj::Key { lo, hi, pos } => {
                            let (a, z) = narrow(keys, lo, hi, pos, b);
                            if !(a..z).any(|i| seen & (1u64 << i) == 0) {
                                return false;
                            }
                            // JSON strings are prefix-free: a key that ends is alone.
                            self.stack[top] = if keys[a as usize].len() == pos as usize + 1 {
                                Frame::Obj {
                                    node,
                                    seen: seen | (1u64 << a),
                                    at: Obj::Colon(a),
                                }
                            } else {
                                Frame::Obj {
                                    node,
                                    seen,
                                    at: Obj::Key {
                                        lo: a,
                                        hi: z,
                                        pos: pos + 1,
                                    },
                                }
                            };
                            return true;
                        }
                        Obj::Colon(i) => {
                            if b != b':' {
                                return false;
                            }
                            self.stack[top] = Frame::Obj {
                                node,
                                seen,
                                at: Obj::Next,
                            };
                            return self.push(Frame::Value(values[i as usize]));
                        }
                        Obj::Next => {
                            if b == b',' && (seen.count_ones() as usize) < keys.len() {
                                self.stack[top] = Frame::Obj {
                                    node,
                                    seen,
                                    at: Obj::Comma,
                                };
                                return true;
                            }
                            if b == b'}' && required & !seen == 0 {
                                self.len -= 1;
                                return true;
                            }
                            return false;
                        }
                    }
                }
                Frame::Arr { node, count, at } => {
                    let Node::Array { items, min, max } = &schema.nodes[node as usize] else {
                        return false;
                    };
                    match at {
                        Arr::Open => {
                            if b == b']' && *min == 0 {
                                self.len -= 1;
                                return true;
                            }
                            if *max == Some(0) {
                                return false;
                            }
                            self.stack[top] = Frame::Arr {
                                node,
                                count: 1,
                                at: Arr::Next,
                            };
                            if !self.push(Frame::Value(*items)) {
                                return false;
                            }
                        }
                        Arr::Next => {
                            if b == b',' && max.is_none_or(|m| count < m) {
                                self.stack[top] = Frame::Arr {
                                    node,
                                    count: count.saturating_add(1),
                                    at: Arr::Next,
                                };
                                return self.push(Frame::Value(*items));
                            }
                            if b == b']' && count >= *min {
                                self.len -= 1;
                                return true;
                            }
                            return false;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema(v: Value) -> JsonSchema {
        JsonSchema::compile(&v).unwrap()
    }

    fn prefix_ok(s: &JsonSchema, text: &[u8]) -> bool {
        s.matcher().feed_bytes(text)
    }

    fn whole_ok(s: &JsonSchema, text: &[u8]) -> bool {
        let mut m = s.matcher();
        m.feed_bytes(text) && m.complete()
    }

    fn person() -> JsonSchema {
        schema(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2},
                "score": {"type": ["number", "null"]},
                "ok": {"type": "boolean"}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        }))
    }

    /// A failure means the automaton accepts a document the schema forbids,
    /// or refuses one it allows: the mask would steer the model wrong.
    #[test]
    fn the_matcher_accepts_exactly_the_valid_documents() {
        let s = person();
        for good in [
            r#"{"name":"Ada","age":36}"#,
            r#"{"age":-1,"name":"\"é\n\u00e9","tags":["a","b"],"score":null,"ok":true}"#,
            r#"{"name":"","age":0,"score":-0.5e+3,"tags":[]}"#,
            r#"{"name":"東京 🗼","age":10}"#,
        ] {
            assert!(whole_ok(&s, good.as_bytes()), "{good}");
            assert_eq!(s.accepts(good), Accepted::Complete, "{good}");
        }
        for bad in [
            r#"{"name":"Ada"}"#,                    // age required
            r#"{"name":"Ada","age":1.5}"#,          // integer
            r#"{"name":"Ada","age":01}"#,           // leading zero
            r#"{"name":"Ada","age":1,"x":1}"#,      // undeclared property
            r#"{"name":"Ada","name":"B","age":1}"#, // repeated key
            r#"{"name":"Ada","age":1,"tags":["c"]}"#,
            r#"{"name":"Ada","age":1,"tags":["a","a","a"]}"#, // maxItems
            r#"{"name":"Ada","age":1,}"#,                     // trailing comma
            r#"{  "name":"Ada","age":1}"#,                    // two spaces in a row
            "{\"name\":\"a\u{1}\",\"age\":1}",                // control character in a string
            r#"{"name":"Ada","age":1}x"#,                     // anything after the end
            r#"{"name":"\ud83d\ude00","age":1}"#,             // a surrogate escape
            r#"{"name":"\q","age":1}"#,                       // not an escape
        ] {
            assert!(!whole_ok(&s, bad.as_bytes()), "{bad}");
            assert_ne!(s.accepts(bad), Accepted::Complete, "{bad}");
        }
        assert!(prefix_ok(&s, br#"{"na"#));
        assert!(!prefix_ok(&s, br#"{"nb"#));
        assert!(!whole_ok(&s, br#"{"name":"Ada","age":1"#), "open object");
        assert_eq!(s.accepts(r#"{"name":"Ada","age":1"#), Accepted::Prefix);
        assert_eq!(s.accepts(r#"{"nb"#), Accepted::No);
        let n = schema(json!({"type": "integer"}));
        assert!(
            whole_ok(&n, b"42") && prefix_ok(&n, b"42"),
            "a top-level number may end"
        );
        // Whichever order the values come in, each one ends where it ends.
        for values in [json!([1, 10, "x"]), json!([10, 1, "x"])] {
            let e = schema(json!({ "enum": values }));
            assert!(whole_ok(&e, b"1") && whole_ok(&e, b"10"), "{values}");
            assert!(!whole_ok(&e, b"100") && !whole_ok(&e, b"0"), "{values}");
            assert!(whole_ok(&e, br#""x""#), "{values}");
        }
        let in_list = schema(json!({"type": "array", "items": {"enum": [10, 1]}}));
        assert!(whole_ok(&in_list, b"[1,10,1]"));
        // Numbers every JSON reader holds: 18 digits before the point, 2 of exponent.
        let x = schema(json!({"type": "number"}));
        assert!(whole_ok(&x, b"-123456789012345678.5e-99"));
        assert!(!prefix_ok(&x, b"1234567890123456789"));
        assert!(!prefix_ok(&x, b"1e100"));
    }

    /// One space at most between two tokens of a container, none around the
    /// document or inside a number; in a string a space is a character. A
    /// failure: the space a model writes after a colon refused, or a run.
    #[test]
    fn one_space_between_tokens_and_no_more() {
        let s = person();
        for good in [
            r#"{ "name": "Ada", "age": 36 }"#,
            r#"{"name" : "Ada" , "age" :36}"#,
            r#"{"name":"A d a","age":1,"tags": [ "a" , "b" ], "score": null, "ok": true }"#,
        ] {
            assert_eq!(s.accepts(good), Accepted::Complete, "{good}");
        }
        for bad in [
            r#"{  "name":"Ada","age":1}"#,
            "{\n\"name\":\"Ada\",\"age\":1}",
            "{\t\"name\":\"Ada\",\"age\":1}",
            r#" {"name":"Ada","age":1}"#,
            r#"{"name":"Ada","age":1} "#,
            r#"{"name":"Ada","age":1 2}"#,
            r#"{"name":"Ada","age":- 1}"#,
            r#"{"name":"Ada","age":1,"ok":tr ue}"#,
            r#"{"name":"Ada","age":1,"tags":["a" "b"]}"#,
        ] {
            assert_eq!(s.accepts(bad), Accepted::No, "{bad:?}");
        }
        assert_eq!(s.accepts(r#"{ "name": "Ada", "age": 1 "#), Accepted::Prefix);
        let n = schema(json!({"type": "integer"}));
        assert_eq!(
            n.accepts("1 "),
            Accepted::No,
            "nothing after a top-level number"
        );
        let e = schema(json!({"enum": ["a b", 1, 10]}));
        assert_eq!(e.accepts(r#""a b""#), Accepted::Complete);
        let list = schema(json!({"type": "array", "items": {"enum": [1, 10]}}));
        assert_eq!(list.accepts("[1 , 10 ]"), Accepted::Complete);
        assert_eq!(list.accepts("[1 0]"), Accepted::No);
    }

    /// Bytes, not characters: a byte-level vocabulary splits `é` and `🗼`
    /// across tokens, so a string holds a partial character between two of
    /// them; what never becomes UTF-8 is refused at its first wrong byte.
    #[test]
    fn a_string_takes_utf8_byte_by_byte_and_refuses_what_is_not() {
        let s = schema(json!({"type": "string"}));
        let tower = "\"🗼\"".as_bytes();
        let mut m = s.matcher();
        for (i, b) in tower.iter().enumerate() {
            assert!(m.feed(*b), "byte {i}");
            assert_eq!(
                m.in_plain_string(),
                i == 0 || i == tower.len() - 2,
                "byte {i}"
            );
        }
        assert!(m.complete());
        for bad in [
            &b"\"\xC3\x28"[..], // a lead byte, then no continuation
            b"\"\xC0\xAF",      // an overlong encoding
            b"\"\xED\xA0\x80",  // a surrogate, encoded
            b"\"\xF5",          // past U+10FFFF
            b"\"\x80",          // a continuation byte alone
        ] {
            assert!(!s.matcher().feed_bytes(bad), "{bad:?}");
        }
        assert!(s.matcher().feed_bytes(b"\"\xC3"), "a character still open");
        assert!(
            !s.matcher().feed_bytes(b"\"\xC3\""),
            "closed before its end"
        );
    }

    /// A failure means a constraint the engine cannot enforce is accepted
    /// and silently ignored: output the caller's schema forbids gets served.
    #[test]
    fn unsupported_keywords_are_refused_by_name() {
        for (s, word) in [
            (json!({"type": "string", "pattern": "^a"}), "pattern"),
            (json!({"type": "string", "minLength": 2}), "minLength"),
            (json!({"anyOf": [{"type": "string"}]}), "anyOf"),
            (json!({"$ref": "#/defs/x"}), "$ref"),
            (
                json!({"type": "object", "items": {"type": "string"}}),
                "items",
            ),
            (json!({"type": "array"}), "items"),
            (
                json!({"type": "object", "properties": {}, "required": ["a"]}),
                "required",
            ),
            (json!({"type": "string", "enum": [1]}), "declared type"),
            (json!({"type": "date"}), "date"),
            (json!({}), "type"),
            (json!(true), "object"),
            (
                json!({"type": "array", "items": {"type": "null"}, "minItems": 3, "maxItems": 2}),
                "maxItems",
            ),
        ] {
            let e = JsonSchema::compile(&s).unwrap_err();
            assert!(e.contains(word), "{s} -> {e}");
        }
    }

    /// Every bound is judged at compile time, before a token is generated,
    /// and its refusal names it. A failure means a schema whose cost has no
    /// ceiling reaches the generation loop.
    #[test]
    fn a_schema_past_a_bound_is_refused_by_its_bound() {
        let nested = (0..=MAX_DEPTH).fold(
            json!({"type": "null"}),
            |inner, _| json!({"type": "array", "items": inner}),
        );
        let deep_enough = (0..MAX_DEPTH).fold(
            json!({"type": "null"}),
            |inner, _| json!({"type": "array", "items": inner}),
        );
        assert!(JsonSchema::compile(&deep_enough).is_ok());
        let props = |n: usize| -> Value {
            let p: serde_json::Map<String, Value> = (0..n)
                .map(|i| (format!("p{i}"), json!({"type": "null"})))
                .collect();
            json!({"type": "object", "properties": p})
        };
        assert!(JsonSchema::compile(&props(MAX_PROPERTIES)).is_ok());
        let values = |n: usize| json!({ "enum": (0..n).collect::<Vec<_>>() });
        assert!(JsonSchema::compile(&values(MAX_ENUM)).is_ok());
        let big = json!({"type": "string", "description": "x".repeat(MAX_SCHEMA_BYTES)});
        // 26 objects of 20 properties each: 548 subschemas, under 16 KiB.
        let wide = json!({"type": "array", "items": {"type": "object", "properties":
            (0..26).map(|i| (format!("o{i}"), props(20))).collect::<serde_json::Map<_, _>>()}});
        assert!(serde_json::to_vec(&wide).unwrap().len() < MAX_SCHEMA_BYTES);
        for (s, bound) in [
            (nested, "levels"),
            (props(MAX_PROPERTIES + 1), "properties"),
            (values(MAX_ENUM + 1), "values"),
            (big, "bytes"),
            (wide, "subschemas"),
        ] {
            let e = JsonSchema::compile(&s).unwrap_err();
            assert!(e.contains(bound), "{bound}: {e}");
            assert!(e.len() < 256, "a refusal quotes no schema: {e}");
        }
        // A keyword name is quoted to 64 bytes at most.
        let long = "k".repeat(10_000);
        let e = JsonSchema::compile(&json!({"type": "null", long.clone(): 1})).unwrap_err();
        assert!(e.len() < 256, "{e}");
    }

    /// The subset's meaning, written apart from the automaton.
    fn validates(schema: &Value, v: &Value) -> bool {
        let m = schema.as_object().unwrap();
        if let Some(e) = m.get("enum") {
            return e.as_array().unwrap().contains(v);
        }
        if let Some(c) = m.get("const") {
            return c == v;
        }
        let types: Vec<&str> = match &m["type"] {
            Value::String(t) => vec![t.as_str()],
            Value::Array(a) => a.iter().map(|t| t.as_str().unwrap()).collect(),
            _ => unreachable!(),
        };
        let typed = types.iter().any(|t| match *t {
            "null" => v.is_null(),
            "boolean" => v.is_boolean(),
            "string" => v.is_string(),
            // Past 64 bits serde_json reads a float, still without a fraction.
            "integer" => v.is_i64() || v.is_u64() || v.as_f64().is_some_and(|f| f.fract() == 0.0),
            "number" => v.is_number(),
            "object" => v.is_object(),
            "array" => v.is_array(),
            _ => false,
        });
        if !typed {
            return false;
        }
        if let Value::Object(o) = v {
            let props = m.get("properties").and_then(Value::as_object);
            let declared = |k: &str| props.is_some_and(|p| p.contains_key(k));
            let required = m.get("required").and_then(Value::as_array);
            return o.keys().all(|k| declared(k))
                && required.is_none_or(|r| r.iter().all(|k| o.contains_key(k.as_str().unwrap())))
                && o.iter().all(|(k, x)| validates(&props.unwrap()[k], x));
        }
        if let Value::Array(a) = v {
            let n = a.len() as u64;
            return m.get("minItems").is_none_or(|x| n >= x.as_u64().unwrap())
                && m.get("maxItems").is_none_or(|x| n <= x.as_u64().unwrap())
                && a.iter().all(|x| validates(&m["items"], x));
        }
        true
    }

    /// SECURITY: random walks over the bytes the automaton takes write JSON
    /// that serde_json parses and the schema validates, and never dead-end.
    /// A failure: an answer that breaks its schema, or nothing left to pick.
    #[test]
    fn random_walks_write_valid_documents_and_never_dead_end() {
        let schemas = [
            json!({"type": "object", "properties": {
                "name": {"type": "string"}, "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"enum": ["a", "b", 10, 1, null]}, "minItems": 1, "maxItems": 3},
                "score": {"type": ["number", "null", "boolean"]},
                "nested": {"type": "object", "properties": {"x": {"type": ["string", "array"],
                    "items": {"type": "integer"}}}, "required": ["x"], "additionalProperties": false}},
                "required": ["name", "tags"]}),
            json!({"type": "array", "items": {"type": "array", "items": {"const": {"k": [1, "é"]}}, "maxItems": 2}}),
            json!({"type": ["integer", "string"]}),
        ];
        let mut seed = 0x2026_1030_u64;
        let mut rand = |n: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as usize) % n
        };
        for raw in &schemas {
            let s = schema(raw.clone());
            let mut finished = 0;
            for _ in 0..1500 {
                let mut m = s.matcher();
                let mut text = Vec::new();
                while text.len() < 300 {
                    let takes: Vec<u8> = (0..=255u8).filter(|&b| m.clone().feed(b)).collect();
                    assert!(
                        !takes.is_empty() || m.complete(),
                        "dead end after {:?}",
                        String::from_utf8_lossy(&text)
                    );
                    // End as soon as allowed half the time, so walks finish.
                    if takes.is_empty() || (m.complete() && rand(2) == 0) {
                        let v: Value = serde_json::from_slice(&text).unwrap_or_else(|e| {
                            panic!("{e}: {:?}", String::from_utf8_lossy(&text))
                        });
                        assert!(validates(raw, &v), "{v}");
                        assert_eq!(
                            s.accepts(std::str::from_utf8(&text).unwrap()),
                            Accepted::Complete
                        );
                        finished += 1;
                        break;
                    }
                    let b = takes[rand(takes.len())];
                    assert!(m.feed(b));
                    text.push(b);
                }
            }
            assert!(finished > 150, "{finished} walks finished on {raw}");
        }
    }

    /// The signed digest does not depend on the order the caller wrote keys
    /// in, and is SHA-256 of the compact serialization with keys sorted.
    #[test]
    fn the_digest_is_over_the_sorted_compact_schema() {
        use sha2::{Digest, Sha256};
        let a: Value = serde_json::from_str(
            r#"{"type":"object","properties":{"b":{"type":"null"},"a":{"type":"null"}}}"#,
        )
        .unwrap();
        let b: Value = serde_json::from_str(
            r#"{ "properties": {"a":{"type":"null"},"b":{"type":"null"}}, "type": "object" }"#,
        )
        .unwrap();
        assert_eq!(schema(a).sha256_hex(), schema(b.clone()).sha256_hex());
        let sorted = br#"{"properties":{"a":{"type":"null"},"b":{"type":"null"}},"type":"object"}"#;
        let want: String = Sha256::digest(sorted)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect();
        assert_eq!(schema(b).sha256_hex(), want);
    }
}
