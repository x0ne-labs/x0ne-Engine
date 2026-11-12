//! Properties of generation, each played on seeded random cases
//! (`prop_kit_tests`): the `json_schema` mask against models that prefer what
//! breaks a document, the refusal of what the subset does not hold, a message
//! that never becomes a control token, stop sequences and the stream, and the
//! draws.
//!
//! A failure names its seed: `X0NE_PROP_REPLAY` runs it again alone.

use std::cell::RefCell;
use std::rc::Rc;

use serde_json::{json, Value};
use tokenizers::{NormalizedString, Normalizer, Tokenizer};

use crate::arch::Architecture;
use crate::backend::{Completion, GenerationParams, NoSink, Sampling, TokenSink};
use crate::constrain::{Constraint, TokenTable};
use crate::json_schema::{Accepted, JsonSchema};
use crate::prop_kit_tests::{
    byte_bpe, bytes_by_id, check, gen_schema, gen_string, id_of, json_tokenizer, nasty_text,
    parse_json, peaked, text_tokens, tokenize_at_random, validates, write_instance, Rng,
    RowSession, SessionLog, EOS,
};
use crate::sampler::Sampler;
use crate::server::ChatMessage;
use crate::stepped::tests::Recorder;
use crate::stepped::{argmax, pick, run, run_with_table, Prepared, TableCache};

// ── The bench: one tokenizer, its table, a scripted session ─────────────────

struct Bench {
    tokenizer: Tokenizer,
    eos: Vec<u32>,
    table: TokenTable,
    vocab: usize,
}

fn bench() -> Bench {
    let tokenizer = crate::prompt::text_only(json_tokenizer());
    let eos = vec![id_of(&tokenizer, EOS)];
    let table = TokenTable::from_tokenizer(&tokenizer, &eos).expect("a byte-level table");
    let vocab = tokenizer.get_vocab_size(true);
    assert_eq!(table.len(), vocab);
    Bench {
        tokenizer,
        eos,
        table,
        vocab,
    }
}

/// One answer through the engine's loop, its rows from `next_row`.
fn generate(
    b: &Bench,
    params: &GenerationParams,
    sink: &mut dyn TokenSink,
    next_row: impl FnMut(&[u32]) -> Vec<f32>,
) -> (anyhow::Result<Completion>, SessionLog) {
    let req = Prepared {
        prompt_ids: vec![0x41],
        max_tokens: params.max_tokens,
    };
    let log = Rc::new(RefCell::new(SessionLog::default()));
    let table = params.json_schema.is_some().then_some(&b.table);
    let out = run_with_table(&b.tokenizer, &b.eos, table, &req, params, sink, |_| {
        Ok(RowSession::new(next_row, log.clone()))
    });
    let log = Rc::try_unwrap(log)
        .expect("the session is finished")
        .into_inner();
    (out, log)
}

fn constrained(
    schema: &JsonSchema,
    max_tokens: usize,
    sampling: Option<Sampling>,
) -> GenerationParams {
    GenerationParams {
        max_tokens,
        sampling,
        stop: Vec::new(),
        json_schema: Some(schema.clone()),
    }
}

fn compiled(schema: &Value) -> JsonSchema {
    JsonSchema::compile(schema).unwrap_or_else(|e| panic!("{e}: {schema}"))
}

/// What the server asks of a constrained answer before it signs it
/// (`server::Decoding::holds`), and more: a whole document is one the schema
/// validates, read by a parser that is not the automaton; an answer said cut
/// spent its budget on a document that was still open.
fn holds(source: &Value, schema: &JsonSchema, c: &Completion, budget: usize) {
    let reason = c.finish_reason(budget);
    let text = &c.text;
    match schema.accepts(text) {
        Accepted::No => panic!("{text:?} is not the start of a document of {source}"),
        Accepted::Prefix => assert_eq!(reason, "length", "{text:?} is open, for {source}"),
        Accepted::Complete => {
            let doc = parse_json(text).unwrap_or_else(|e| panic!("{e}: {text:?}"));
            assert!(serde_json::from_str::<Value>(text).is_ok(), "{text:?}");
            validates(source, &doc).unwrap_or_else(|e| panic!("{e}: {text:?} for {source}"));
        }
    }
    assert!(c.completion_tokens as usize <= budget, "past the budget");
    if reason == "length" {
        assert_eq!(
            c.completion_tokens as usize, budget,
            "cut before the budget"
        );
        let mut m = schema.matcher();
        m.feed_bytes(text.as_bytes());
        assert!(!m.done(), "{text:?} is closed, and said cut, for {source}");
    }
}

// ── 1. The mask: sound whatever the model prefers ───────────────────────────

/// What a scripted model prefers.
#[derive(Debug, Clone, Copy)]
enum Taste {
    /// Nothing: uniform rows.
    Any,
    /// Structure out of place, cut characters, controls, the end and the
    /// special tokens, at this height.
    Breaker(f32),
    /// Never the end: digits, letters, openings.
    Rambler,
    /// The end, at once.
    Hasty,
    /// Nothing at all: every logit is minus infinity.
    Dead,
}

fn breakers(b: &Bench) -> Vec<u32> {
    let mut ids: Vec<u32> = b"\"\\}],:{[\0\n \x7f\x80\xbf\xc3\xed\xf4\xff"
        .iter()
        .map(|&x| u32::from(x))
        .collect();
    ids.extend(b.eos.iter().copied());
    for added in ["<|im_start|>", "<|pad|>", "<think>"] {
        ids.push(id_of(&b.tokenizer, added));
    }
    ids
}

fn row_of(rng: &mut Rng, b: &Bench, taste: Taste, breakers: &[u32]) -> Vec<f32> {
    let mut row: Vec<f32> = (0..b.vocab).map(|_| rng.unit() * 8.0).collect();
    match taste {
        Taste::Any => {}
        Taste::Breaker(height) => {
            for &id in breakers {
                if rng.chance(1, 2) {
                    row[id as usize] = height;
                }
            }
        }
        Taste::Rambler => {
            for id in (b'0'..=b'9').chain(b'a'..=b'z').chain(*b"[{ ") {
                row[id as usize] += 3.0;
            }
            row[b.eos[0] as usize] = f32::NEG_INFINITY;
        }
        Taste::Hasty => row[b.eos[0] as usize] = 50.0,
        Taste::Dead => row.fill(f32::NEG_INFINITY),
    }
    row
}

/// Item 1, soundness. A random schema of the subset, a model that prefers
/// what breaks a document, a random budget, greedy or sampled: the answer is
/// a document the schema validates, or the start of one when the budget cut
/// it, never anything else; the stream is the answer; the session is finished
/// once. A model with no finite logit gets an error, not a document.
#[test]
fn a_constrained_answer_is_a_document_of_its_schema_whatever_the_model_prefers() {
    let b = bench();
    let breakers = breakers(&b);
    check("constraint_soundness", 260, 6, |rng, size| {
        let source = gen_schema(rng, size, 0);
        let schema = compiled(&source);
        let taste = match rng.below(10) {
            0 | 1 => Taste::Any,
            2 | 3 => Taste::Breaker(50.0),
            4 => Taste::Breaker(f32::MAX),
            5 => Taste::Breaker(f32::INFINITY),
            6 | 7 => Taste::Rambler,
            8 => Taste::Hasty,
            _ => Taste::Dead,
        };
        let finite = match taste {
            Taste::Breaker(height) => height <= 50.0,
            Taste::Dead => false,
            Taste::Any | Taste::Rambler | Taste::Hasty => true,
        };
        let sampling = (finite && rng.chance(1, 3)).then(|| Sampling {
            temperature: *rng.pick(&[0.0, 0.7, 1.0, 2.0]),
            top_p: *rng.pick(&[1.0, 0.9, 0.5, 0.0]),
            seed: rng.next(),
        });
        let budget = if rng.chance(1, 8) {
            rng.between(64, 160)
        } else {
            rng.between(1, 8 * size)
        };
        let params = constrained(&schema, budget, sampling);
        let mut rows = rng.fork();
        let mut sink = Recorder::default();
        let (out, log) = generate(&b, &params, &mut sink, |_| {
            row_of(&mut rows, &b, taste, &breakers)
        });
        assert_eq!(log.finished, 1, "the session is finished once");
        assert!(log.fed.len() < budget, "{} tokens fed", log.fed.len());
        let c = match (out, taste) {
            (Err(_), Taste::Dead) => return,
            (Err(e), _) => panic!("{taste:?} on {source}: {e:#}"),
            (Ok(c), _) => c,
        };
        holds(&source, &schema, &c, budget);
        assert_eq!(sink.deltas.concat(), c.text, "the stream is the answer");
        assert!(sink.deltas.iter().all(|d| !d.is_empty()));
    });
}

/// Item 1, completeness. Any document a schema validates, in any text the
/// engine admits for it (key order, one space, escapes), cut into tokens at
/// random: the mask lets every token through, then the end; and a model that
/// writes those tokens is answered with that document.
#[test]
fn every_document_of_a_schema_passes_the_mask_token_by_token() {
    let b = bench();
    let tokens = text_tokens(&b.table);
    check("constraint_completeness", 260, 6, |rng, size| {
        let source = gen_schema(rng, size, 0);
        let schema = compiled(&source);
        let mut text = String::new();
        write_instance(rng, &source, &mut text);
        let doc = parse_json(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
        validates(&source, &doc).unwrap_or_else(|e| panic!("{e}: {text} for {source}"));
        assert_eq!(
            schema.accepts(&text),
            Accepted::Complete,
            "{text} for {source}"
        );
        let ids = tokenize_at_random(rng, &tokens, text.as_bytes());
        let mut constraint = Constraint::new(&schema, &b.table, &b.eos);
        for (i, &id) in ids.iter().enumerate() {
            assert!(
                constraint.allowed(b.vocab)[id as usize],
                "token {i} ({id}) of {text} is masked, for {source}"
            );
            constraint.advance(id).expect("an allowed token is taken");
        }
        assert!(constraint.complete());
        assert!(
            constraint.allowed(b.vocab)[b.eos[0] as usize],
            "the end is masked"
        );
        // Every cut of it is a start, and no start is a document unless a
        // number or a literal may end there.
        let mut cut = rng.below(text.len());
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        assert_ne!(
            schema.accepts(&text[..cut]),
            Accepted::No,
            "{}",
            &text[..cut]
        );

        let script = ids.clone();
        let eos = b.eos[0];
        let params = constrained(&schema, ids.len() + 1, None);
        let (out, _) = generate(&b, &params, &mut NoSink, |fed| {
            peaked(b.vocab, script.get(fed.len()).copied().unwrap_or(eos))
        });
        let c = out.unwrap_or_else(|e| panic!("{text} for {source}: {e:#}"));
        assert_eq!(c.text, text);
        assert_eq!(c.finish_reason(ids.len() + 1), "stop");
    });
}

/// What the subset does not hold, and the name its refusal carries.
fn unsupported(rng: &mut Rng) -> (Value, &'static str) {
    const KEYWORDS: &[&str] = &[
        "pattern",
        "format",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "multipleOf",
        "$ref",
        "$defs",
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        "if",
        "patternProperties",
        "propertyNames",
        "prefixItems",
        "uniqueItems",
        "contains",
        "minProperties",
        "nullable",
    ];
    match rng.below(20) {
        0..=5 => {
            let k = *rng.pick(KEYWORDS);
            let mut s = json!({"type": "string"});
            s[k] = json!(1);
            (s, k)
        }
        6 => (
            json!({"type": "object", "additionalProperties": true}),
            "additionalProperties",
        ),
        7 => (
            json!({"type": "object", "additionalProperties": {"type": "string"}}),
            "additionalProperties",
        ),
        8 => (json!({"type": "float"}), "type"),
        9 => (json!({"type": []}), "type"),
        10 => (json!({"type": "array"}), "items"),
        11 => (
            json!({"type": "object", "properties": {"a": {"type": "null"}}, "required": ["b"]}),
            "required",
        ),
        12 => (json!({"enum": []}), "enum"),
        13 => (json!({"enum": [1], "const": 1}), "enum and const"),
        14 => (
            json!({"type": "array", "items": {"type": "null"}, "minItems": 3, "maxItems": 2}),
            "maxItems",
        ),
        15 => (
            json!({"type": "array", "items": {"type": "null"}, "minItems": -1}),
            "minItems",
        ),
        16 => (json!(true), "a schema is an object"),
        17 => (json!({"description": "anything"}), "type"),
        18 => (json!({"enum": ["a"], "properties": {}}), "properties"),
        _ => (
            json!({"type": "string", "items": {"type": "null"}}),
            "items",
        ),
    }
}

/// Item 1, refusals. A schema with one thing the subset does not hold, at
/// the root or under properties and items of a schema that compiles: refused
/// with the name of what is not held, in under 256 bytes, by the compiler and
/// by the request parser alike. Nothing is compiled from the rest of it.
#[test]
fn a_schema_outside_the_subset_is_refused_by_name_and_never_half_applied() {
    check("unsupported_schemas", 1500, 6, |rng, size| {
        let (mut schema, name) = unsupported(rng);
        for _ in 0..rng.below(4) {
            schema = if rng.chance(1, 2) {
                let mut outer = json!({"type": "object", "properties": {}});
                for _ in 0..rng.below(size) {
                    outer["properties"][gen_string(rng)] = gen_schema(rng, size, 2);
                }
                outer["properties"]["k"] = schema;
                outer
            } else {
                json!({"type": ["array", "null"], "items": schema})
            };
        }
        let refused = JsonSchema::compile(&schema).expect_err("compiled");
        assert!(refused.contains(name), "{refused:?} does not name {name}");
        assert!(refused.len() < 256, "{} bytes", refused.len());
        let format = json!({"type": "json_schema", "json_schema": {"name": "n", "schema": schema}});
        let at_the_door = crate::decoding::response_format(Some(&format)).expect_err("accepted");
        assert!(at_the_door.contains(name), "{at_the_door:?}");
        let request = json!({
            "messages": [{"role": "user", "content": "x"}],
            "response_format": format,
        });
        let (status, said) = crate::server::parse_chat_request(request).expect_err("parsed");
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            said.contains("response_format") && said.contains(name),
            "{said:?}"
        );
    });
}

/// Item 1, the bounds: a schema past one of them is refused by the bound's
/// name, and one at the bound compiles.
#[test]
fn a_schema_past_a_bound_is_refused_by_the_bound() {
    use crate::json_schema::{MAX_DEPTH, MAX_ENUM, MAX_NODES, MAX_PROPERTIES, MAX_SCHEMA_BYTES};
    let nested = |levels: usize| {
        (0..levels).fold(
            json!({"type": "null"}),
            |inner, _| json!({"type": "array", "items": inner}),
        )
    };
    let wide = |n: usize| {
        let props: serde_json::Map<String, Value> = (0..n)
            .map(|i| (format!("k{i}"), json!({"type": "null"})))
            .collect();
        json!({"type": "object", "properties": props})
    };
    let listed = |n: usize| json!({"enum": (0..n).collect::<Vec<_>>()});
    // 60 properties of 9 subschemas each: 541 with the root.
    let many = json!({"type": "object", "properties": (0..60).map(|i| {
        (format!("k{i}"), nested(8))
    }).collect::<serde_json::Map<String, Value>>()});
    let long = json!({"type": "string", "description": "d".repeat(MAX_SCHEMA_BYTES)});
    for (at, past, name) in [
        (nested(MAX_DEPTH), nested(MAX_DEPTH + 1), "levels"),
        (wide(MAX_PROPERTIES), wide(MAX_PROPERTIES + 1), "properties"),
        (listed(MAX_ENUM), listed(MAX_ENUM + 1), "values"),
        (wide(MAX_PROPERTIES), many, "subschemas"),
        (json!({"type": "string"}), long, "bytes"),
    ] {
        assert!(JsonSchema::compile(&at).is_ok(), "at the bound of {name}");
        let refused = JsonSchema::compile(&past).expect_err("compiled");
        assert!(refused.contains(name), "{refused:?} does not name {name}");
        assert!(refused.len() < 256);
    }
    assert!(MAX_NODES < 541);
}

/// A budget that ends inside a character of a property name or of a literal:
/// the answer is the start of the document up to its last whole character,
/// said cut. Until 2026-11-10 the half character was decoded as U+FFFD, which
/// no key holds: the server judged the answer "not a document" (500) for a
/// budget one token short.
#[test]
fn a_budget_that_cuts_a_character_of_a_key_gives_the_start_of_the_document() {
    let b = bench();
    for (source, script, whole) in [
        (
            json!({"type": "object", "properties": {"n\u{e9}": {"type": "null"}}, "required": ["n\u{e9}"]}),
            &b"{\"n\xc3"[..],
            "{\"n",
        ),
        (json!({"enum": ["\u{1f5fc}"]}), &b"\"\xf0\x9f\x97"[..], "\""),
        (json!({"type": "string"}), &b"\"a\xe6\x9d"[..], "\"a"),
    ] {
        let schema = compiled(&source);
        let params = constrained(&schema, script.len(), None);
        let mut sink = Recorder::default();
        let (out, _) = generate(&b, &params, &mut sink, |fed| {
            peaked(b.vocab, u32::from(script[fed.len().min(script.len() - 1)]))
        });
        let c = out.expect("an answer");
        assert_eq!(c.text, whole, "for {source}");
        assert_eq!(c.completion_tokens as usize, script.len());
        assert_eq!(sink.deltas.concat(), whole);
        holds(&source, &schema, &c, script.len());
    }
}

/// A document that closes on the last token of its budget was not cut: its
/// `finish_reason` is `stop`, as for the same document under a larger budget.
#[test]
fn a_document_closed_on_its_last_budgeted_token_is_not_said_cut() {
    let b = bench();
    let source = json!({"type": "boolean"});
    let schema = compiled(&source);
    for budget in [4, 5, 64] {
        let params = constrained(&schema, budget, None);
        let (out, _) = generate(&b, &params, &mut NoSink, |fed| {
            peaked(b.vocab, u32::from(b"true"[fed.len().min(3)]))
        });
        let c = out.expect("an answer");
        assert_eq!((c.text.as_str(), c.completion_tokens), ("true", 4));
        assert_eq!(c.finish_reason(budget), "stop", "budget {budget}");
        holds(&source, &schema, &c, budget);
    }
}

// ── 2. A message never becomes a control token ──────────────────────────────

const ARCHITECTURES: [Architecture; 4] = [
    Architecture::Qwen3,
    Architecture::Olmo2,
    Architecture::Olmo3,
    Architecture::Ministral3,
];

/// What a message may type to forge a turn: every template's markers, OLMo
/// 2's role lines, and a plain added token.
const TYPED: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "<|user|>\n",
    "<|assistant|>\n",
    "<|system|>\n",
    "<think>",
];

/// A piece of a prompt read back from its ids: a control token by its
/// string, or text.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Control(String),
    Text(String),
}

fn control(marker: &str) -> Piece {
    Piece::Control(marker.into())
}

/// Adjacent texts joined, empty ones dropped: what ids can say.
fn joined(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    for p in pieces {
        match (out.last_mut(), p) {
            (_, Piece::Text(t)) if t.is_empty() => {}
            (Some(Piece::Text(last)), Piece::Text(t)) => last.push_str(&t),
            (_, p) => out.push(p),
        }
    }
    out
}

fn role_of(m: &ChatMessage) -> &'static str {
    match m.role.as_str() {
        "system" => "system",
        "assistant" => "assistant",
        _ => "user",
    }
}

/// The template of `arch` written a second time, from its published form:
/// the pieces a conversation must be read back as.
fn expected_pieces(arch: Architecture, messages: &[ChatMessage]) -> Vec<Piece> {
    let text = |s: &str| Piece::Text(s.into());
    let mut out = Vec::new();
    let last = messages.len() - 1;
    match arch {
        Architecture::Qwen3 => {
            for m in messages {
                out.push(control("<|im_start|>"));
                out.push(Piece::Text(format!("{}\n{}", role_of(m), m.content)));
                out.push(control("<|im_end|>"));
                out.push(text("\n"));
            }
            out.push(control("<|im_start|>"));
            out.push(text("assistant\n<think>\n\n</think>\n\n"));
        }
        Architecture::Olmo2 => {
            out.push(control("<|endoftext|>"));
            for (i, m) in messages.iter().enumerate() {
                out.push(Piece::Text(format!("<|{}|>\n{}", role_of(m), m.content)));
                if role_of(m) == "assistant" {
                    out.push(control("<|endoftext|>"));
                    if i != last {
                        out.push(text("\n"));
                    }
                } else {
                    out.push(text("\n"));
                }
            }
            out.push(text("<|assistant|>\n"));
        }
        Architecture::Olmo3 => {
            if messages.iter().all(|m| role_of(m) != "system") {
                out.push(control("<|im_start|>"));
                out.push(text(
                    "system\nYou are a helpful function-calling AI assistant. You do not \
                     currently have access to any functions. <functions></functions>",
                ));
                out.push(control("<|im_end|>"));
                out.push(text("\n"));
            }
            for (i, m) in messages.iter().enumerate() {
                out.push(control("<|im_start|>"));
                out.push(Piece::Text(format!("{}\n{}", role_of(m), m.content)));
                if role_of(m) == "assistant" && i == last {
                    out.push(control("<|endoftext|>"));
                } else {
                    out.push(control("<|im_end|>"));
                    out.push(text("\n"));
                }
            }
            out.push(control("<|im_start|>"));
            out.push(text("assistant\n"));
        }
        Architecture::Ministral3 => {
            out.push(control("<s>"));
            let mut i = 0;
            while i < messages.len() {
                let role = role_of(&messages[i]);
                let mut turn = messages[i].content.clone();
                i += 1;
                while role != "system" && i < messages.len() && role_of(&messages[i]) == role {
                    turn.push_str("\n\n");
                    turn.push_str(&messages[i].content);
                    i += 1;
                }
                match role {
                    "system" => {
                        out.push(control("[SYSTEM_PROMPT]"));
                        out.push(Piece::Text(turn));
                        out.push(control("[/SYSTEM_PROMPT]"));
                    }
                    "assistant" => {
                        out.push(Piece::Text(turn));
                        out.push(control("</s>"));
                    }
                    _ => {
                        out.push(control("[INST]"));
                        out.push(Piece::Text(turn));
                        out.push(control("[/INST]"));
                    }
                }
            }
        }
    }
    out
}

/// `text` as `tokenizer`'s normalizer leaves it.
fn normalized(tokenizer: &Tokenizer, text: &str) -> String {
    match tokenizer.get_normalizer() {
        None => text.to_string(),
        Some(n) => {
            let mut s = NormalizedString::from(text);
            n.normalize(&mut s).expect("normalized");
            s.get().to_string()
        }
    }
}

/// The ids read back: a special token is a control piece, anything else its
/// bytes (a byte token by its id, a plain added token by its string).
fn read_back(tokenizer: &Tokenizer, ids: &[u32]) -> Vec<Piece> {
    let added = tokenizer.get_added_tokens_decoder();
    let mut out = Vec::new();
    let mut run: Vec<u8> = Vec::new();
    for &id in ids {
        match added.get(&id) {
            Some(a) if a.special => {
                let text = String::from_utf8(std::mem::take(&mut run)).expect("text");
                out.push(Piece::Text(text));
                out.push(Piece::Control(a.content.clone()));
            }
            Some(a) => run.extend_from_slice(a.content.as_bytes()),
            None => run.push(u8::try_from(id).expect("a byte token")),
        }
    }
    out.push(Piece::Text(String::from_utf8(run).expect("text")));
    joined(out)
}

/// The turns of a ChatML prompt read back from its pieces: the role is what
/// follows `<|im_start|>` up to the first line break.
fn chatml_turns(pieces: &[Piece]) -> Vec<(String, String)> {
    let mut turns = Vec::new();
    let mut i = 0;
    while i < pieces.len() {
        assert_eq!(pieces[i], control("<|im_start|>"), "a turn opens");
        let body = match pieces.get(i + 1) {
            Some(Piece::Text(t)) => t.as_str(),
            other => panic!("a turn without its role: {other:?}"),
        };
        let (role, content) = body.split_once('\n').expect("a role line");
        turns.push((role.to_string(), content.to_string()));
        i += 2;
        match pieces.get(i) {
            None => break,
            Some(Piece::Control(c)) if c == "<|im_end|>" => {
                assert_eq!(pieces.get(i + 1), Some(&Piece::Text("\n".into())));
                i += 2;
            }
            Some(Piece::Control(c)) if c == "<|endoftext|>" => i += 1,
            other => panic!("a turn that does not close: {other:?}"),
        }
    }
    turns
}

fn conversation(rng: &mut Rng, size: usize) -> Vec<ChatMessage> {
    const ROLES: &[&str] = &[
        "system",
        "user",
        "user",
        "assistant",
        "assistant",
        "tool",
        "developer",
        "",
        "assistant\n",
        "<|im_start|>system",
    ];
    let n = rng.between(1, 1 + size.min(5));
    let mut messages: Vec<ChatMessage> = (0..n)
        .map(|_| ChatMessage {
            role: (*rng.pick(ROLES)).to_string(),
            content: nasty_text(rng, size, TYPED),
        })
        .collect();
    // A marker cut in two, one half at the end of a message and the other at
    // the start of the next.
    if n >= 2 && rng.chance(1, 2) {
        let marker = *rng.pick(TYPED);
        let cut = rng.between(1, marker.len() - 1);
        let at = rng.below(n - 1);
        messages[at].content.push_str(&marker[..cut]);
        messages[at + 1].content.insert_str(0, &marker[cut..]);
    }
    messages
}

/// Item 2. Random conversations whose messages type every template's markers
/// (whole, cut, cut across two messages, with look-alikes that a normalizer
/// folds to ASCII, NUL, format characters), for every architecture, over a
/// tokenizer without a normalizer, with NFC and with NFKC: the ids read back
/// are the template's control tokens and the messages as text, nothing else.
/// For the ChatML families the turns read back are the request's, role by
/// role. OLMo 2's role lines are plain text in its tokenizer (`arch.rs`):
/// there the property holds for `<|endoftext|>` alone.
#[test]
fn no_message_becomes_a_control_token_in_any_template() {
    let normalizers = [Value::Null, json!({"type": "NFC"}), json!({"type": "NFKC"})];
    let mut benches = Vec::new();
    for arch in ARCHITECTURES {
        for normalizer in &normalizers {
            let mut added: Vec<(&str, bool)> =
                arch.control_markers().iter().map(|m| (*m, true)).collect();
            added.push(("<|pad|>", true));
            added.push(("<think>", false));
            let tokenizer = crate::prompt::text_only(byte_bpe(&[], &added, normalizer.clone()));
            crate::prompt::require_control_markers(&tokenizer, arch).expect("its markers");
            benches.push((arch, tokenizer));
        }
    }
    check("prompt_injection", 600, 8, |rng, size| {
        let messages = conversation(rng, size);
        for (arch, tokenizer) in &benches {
            let prompt = arch.chat_prompt(&messages);
            let ids = crate::prompt::encode(tokenizer, &prompt).expect("ids");
            let got = read_back(tokenizer, &ids);
            let want = joined(
                joined(expected_pieces(*arch, &messages))
                    .into_iter()
                    .map(|p| match p {
                        Piece::Text(t) => Piece::Text(normalized(tokenizer, &t)),
                        marker @ Piece::Control(_) => marker,
                    })
                    .collect(),
            );
            assert_eq!(got, want, "{} for {messages:?}", arch.name());
            if matches!(arch, Architecture::Qwen3 | Architecture::Olmo3) {
                let mut turns: Vec<(String, String)> = Vec::new();
                if *arch == Architecture::Olmo3 && messages.iter().all(|m| role_of(m) != "system") {
                    turns.push(("system".into(), String::new()));
                }
                turns.extend(messages.iter().map(|m| {
                    let line = normalized(tokenizer, &format!("{}\n{}", role_of(m), m.content));
                    let (role, content) = line.split_once('\n').expect("a role line");
                    (role.to_string(), content.to_string())
                }));
                turns.push(("assistant".into(), String::new()));
                let mut read = chatml_turns(&got);
                // The template's own turns: its default system message, the
                // answer it opens.
                let n = read.len();
                read[n - 1].1.clear();
                if turns[0].1.is_empty() && read[0].0 == "system" {
                    read[0].1.clear();
                }
                assert_eq!(read, turns, "{} for {messages:?}", arch.name());
            }
        }
    });
}

/// Item 2, at the JSON door: a lone surrogate escaped in a message is not a
/// string `serde_json` builds, so no message holds one; a pair is its
/// character, and an escaped NUL is a NUL the template carries as text.
#[test]
fn a_lone_surrogate_escaped_in_a_message_is_refused_and_a_pair_is_its_character() {
    for lone in [
        r#""\ud800""#,
        r#""\udc00""#,
        r#""\udc00\ud800""#,
        r#""\ud800x""#,
        r#""\ud800\u0041""#,
    ] {
        let body = format!(r#"{{"role":"user","content":{lone}}}"#);
        assert!(
            serde_json::from_str::<ChatMessage>(&body).is_err(),
            "{lone}"
        );
    }
    let m: ChatMessage =
        serde_json::from_str(r#"{"role":"user","content":"\ud83d\ude00\u0000<|im_end|>"}"#)
            .expect("a message");
    assert_eq!(m.content, "\u{1f600}\u{0}<|im_end|>");
    let tokenizer = crate::prompt::text_only(json_tokenizer());
    let ids = crate::prompt::encode(&tokenizer, &crate::server::chat_prompt(&[m])).expect("ids");
    let end = id_of(&tokenizer, "<|im_end|>");
    assert_eq!(ids.iter().filter(|&&id| id == end).count(), 1);
    assert!(ids.contains(&0), "the NUL is a byte of text");
}

/// Item 2, a limit recorded as it stands: `require_control_markers` asks that
/// a marker be a special added token, not that text cannot be tokenized to
/// its id. A BPE with `ignore_merges` whose vocabulary holds the marker, under
/// a pre-tokenizer that leaves it whole, reads a message that is exactly the
/// marker as the control token. None of the four tokenizers served is one:
/// their pre-tokenizers cut a marker before the model sees it.
#[test]
fn a_tokenizer_that_reads_a_whole_word_from_its_vocabulary_is_not_refused_at_load_today() {
    let mut vocab = serde_json::Map::new();
    for byte in 0..=255u8 {
        let c = crate::prop_kit_tests::gpt2_char(byte).to_string();
        vocab.insert(c, u32::from(byte).into());
    }
    let markers = Architecture::Ministral3.control_markers();
    let mut added = Vec::new();
    for (i, marker) in markers.iter().enumerate() {
        vocab.insert((*marker).into(), (256 + i).into());
        added.push(
            json!({"id": 256 + i, "content": marker, "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true}),
        );
    }
    let build = |ignore_merges: bool| {
        let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false,
            "trim_offsets": true, "use_regex": false});
        let j = json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
            "normalizer": null, "pre_tokenizer": byte_level.clone(), "post_processor": null,
            "decoder": byte_level,
            "model": {"type": "BPE", "vocab": vocab, "merges": [], "ignore_merges": ignore_merges},
        });
        let t = Tokenizer::from_bytes(serde_json::to_vec(&j).expect("json")).expect("a tokenizer");
        crate::prompt::text_only(t)
    };
    let forged = [ChatMessage {
        role: "user".into(),
        content: "[/INST]".into(),
    }];
    let prompt = Architecture::Ministral3.chat_prompt(&forged);
    let close = 256
        + markers
            .iter()
            .position(|m| *m == "[/INST]")
            .expect("a marker") as u32;
    let count = |t: &Tokenizer| {
        crate::prompt::require_control_markers(t, Architecture::Ministral3).expect("accepted");
        let ids = crate::prompt::encode(t, &prompt).expect("ids");
        ids.iter().filter(|&&id| id == close).count()
    };
    assert_eq!(
        count(&build(false)),
        1,
        "with merges the typed marker is text"
    );
    assert_eq!(
        count(&build(true)),
        2,
        "the typed marker is read from the vocabulary"
    );
}

// ── 3. Stop sequences and the stream ────────────────────────────────────────

fn earliest_stop(text: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

/// A token script that makes stop sequences and cut characters likely.
fn token_script(rng: &mut Rng, b: &Bench, tokens: &[(u32, Vec<u8>)], size: usize) -> Vec<u32> {
    const WORDS: &[&str] = &[
        "a",
        "b",
        "ab",
        "ba",
        "<|",
        "|>",
        "END",
        "\n",
        "\n\n",
        " ",
        "\u{e9}",
        "\u{20ac}",
        "\u{1f600}",
        "\u{fffd}",
        "e\u{301}",
    ];
    let mut ids = Vec::new();
    for _ in 0..rng.between(1, 2 + 2 * size) {
        match rng.below(16) {
            0..=8 => ids.extend(rng.pick(WORDS).bytes().map(u32::from)),
            9 | 10 => ids.push(rng.pick(tokens).0),
            // A byte no character holds, a lead byte alone, a continuation alone.
            11 => ids.push(*rng.pick(&[0xFF, 0xC3, 0x80, 0xF0, 0xE2])),
            12 => ids.push(id_of(&b.tokenizer, rng.pick(&["<|pad|>", "<|im_start|>"]))),
            13 => ids.push(id_of(&b.tokenizer, "<think>")),
            14 => ids.push(rng.below(b.vocab) as u32),
            _ => {
                if rng.chance(1, 4) {
                    ids.push(b.eos[0]);
                }
            }
        }
    }
    ids
}

fn stops_for(rng: &mut Rng, full: &str) -> Vec<String> {
    const FIXED: &[&str] = &[
        "ab",
        "b",
        "\u{e9}",
        "\u{fffd}",
        "|>",
        "<|im_end|>",
        "zz",
        "\n\n",
        "\u{1f600}",
        "",
        "aba",
        "a\u{20ac}",
    ];
    let starts: Vec<usize> = full.char_indices().map(|(i, _)| i).collect();
    (0..rng.below(5))
        .map(|_| {
            if starts.is_empty() || rng.chance(1, 3) {
                return (*rng.pick(FIXED)).to_string();
            }
            // A run of whole characters of the answer: it straddles tokens
            // wherever a token ends inside it.
            let from = rng.below(starts.len());
            let to = (from + rng.between(1, 6)).min(starts.len());
            let end = starts.get(to).copied().unwrap_or(full.len());
            full[starts[from]..end].to_string()
        })
        .collect()
}

/// Item 3. A random token stream (words, characters cut over several tokens,
/// bytes no character holds, special and added tokens, an early end), random
/// stop sequences taken from the answer itself, a random budget. The answer
/// is the decoded stream up to the first stop sequence; the deltas, joined,
/// are the answer, none empty, none retracted; a stopped answer took no token
/// more than the one that completed the sequence; the counts and the finish
/// are those of the same request without a reader.
#[test]
fn the_stream_is_the_answer_and_stops_where_the_whole_answer_stops() {
    let b = bench();
    let tokens = text_tokens(&b.table);
    let bytes = bytes_by_id(&b.tokenizer);
    let decoded = |ids: &[u32]| -> String {
        let all: Vec<u8> = ids
            .iter()
            .filter_map(|&id| bytes[id as usize].as_deref())
            .flatten()
            .copied()
            .collect();
        String::from_utf8_lossy(&all).into_owned()
    };
    check("stream_and_stops", 1500, 12, |rng, size| {
        let script = token_script(rng, &b, &tokens, size);
        let budget = rng.between(1, script.len() + 2);
        let taken: Vec<u32> = script
            .iter()
            .copied()
            .take(budget)
            .take_while(|id| !b.eos.contains(id))
            .collect();
        let stop = stops_for(rng, &decoded(&taken));

        let params = GenerationParams {
            max_tokens: budget,
            sampling: None,
            stop: stop.clone(),
            json_schema: None,
        };
        let eos = b.eos[0];
        let rows = |fed: &[u32]| peaked(b.vocab, script.get(fed.len()).copied().unwrap_or(eos));
        let mut sink = Recorder::default();
        let (out, log) = generate(&b, &params, &mut sink, rows);
        let c = out.unwrap_or_else(|e| panic!("{script:?} stop {stop:?}: {e:#}"));
        let said = format!("{script:?} budget {budget} stop {stop:?}");
        // The first stop sequence completed ends the answer: the text is
        // what the tokens taken decode to, up to it.
        let n = c.completion_tokens as usize;
        assert!(n <= taken.len(), "{said}");
        let seen = decoded(&taken[..n]);
        let cut = earliest_stop(&seen, &stop);
        assert_eq!(c.text, seen[..cut.unwrap_or(seen.len())], "{said}");
        assert_eq!(c.stopped, cut.is_some(), "{said}");
        assert_eq!(sink.deltas.concat(), c.text, "{said}");
        assert!(sink.deltas.iter().all(|d| !d.is_empty()), "{said}");
        assert_eq!((sink.started, log.finished), (1, 1));

        assert_eq!(taken[..n], script[..n]);
        assert_eq!(c.length, n >= budget, "{said}");
        if c.stopped {
            assert!(earliest_stop(&c.text, &stop).is_none(), "{said}");
            // One token less left the sequence unfinished, or a character open.
            if let Some(before) = n.checked_sub(1) {
                let earlier = decoded(&taken[..before]);
                assert!(
                    earlier.ends_with('\u{fffd}') || earliest_stop(&earlier, &stop).is_none(),
                    "{said}: {n} tokens taken"
                );
            }
            assert_eq!(c.finish_reason(budget), "stop");
        } else {
            assert_eq!(n, taken.len(), "{said}");
            let reason = if n >= budget { "length" } else { "stop" };
            assert_eq!(c.finish_reason(budget), reason, "{said}");
        }
        let (whole, _) = generate(&b, &params, &mut NoSink, rows);
        assert_eq!(whole.expect("an answer"), c, "{said}");

        // A reader that leaves is an error, never a shorter answer.
        if !sink.deltas.is_empty() {
            let mut leaves = Recorder {
                break_at: Some(rng.below(sink.deltas.len())),
                ..Recorder::default()
            };
            let (left, log) = generate(&b, &params, &mut leaves, rows);
            let e = left.expect_err("an answer to a reader that left");
            assert!(
                e.downcast_ref::<crate::backend::Cancelled>().is_some(),
                "{e:#}"
            );
            assert_eq!(log.finished, 1);
        }
        // A reader gone without a word is noticed at the next step: no
        // token is computed past the one in hand.
        if n > 0 {
            let gone_at = rng.below(n);
            let mut gone = Recorder {
                dead_after: Some(gone_at),
                ..Recorder::default()
            };
            let (left, log) = generate(&b, &params, &mut gone, rows);
            let e = left.expect_err("an answer to a reader that was gone");
            assert!(
                e.downcast_ref::<crate::backend::Cancelled>().is_some(),
                "{e:#}"
            );
            assert!(log.fed.len() <= gone_at, "{said}: {} steps", log.fed.len());
            assert_eq!(log.finished, 1);
        }
    });
}

// ── 6. The draws ────────────────────────────────────────────────────────────

/// A row of logits: wide or narrow, with ties, with masked entries.
fn logits(rng: &mut Rng, size: usize) -> Vec<f32> {
    let n = rng.between(1, 8 * size);
    let scale = *rng.pick(&[0.01f32, 1.0, 8.0, 40.0]);
    let ties = rng.chance(1, 3);
    let masked = rng.chance(1, 3);
    let mut row: Vec<f32> = (0..n)
        .map(|_| {
            let v = (rng.unit() - 0.5) * scale;
            if ties {
                (v * 2.0).round() / 2.0
            } else {
                v
            }
        })
        .collect();
    if masked {
        let keep = rng.below(n);
        for (i, v) in row.iter_mut().enumerate() {
            if i != keep && rng.chance(1, 2) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    row
}

/// Item 6. A temperature of 0, or a nucleus of mass 0, draws the greedy
/// token of every row, whatever the seed: the first of the largest values.
#[test]
fn a_temperature_of_zero_draws_the_greedy_token() {
    check("sampling_greedy", 3000, 16, |rng, size| {
        let mut row = logits(rng, size);
        if rng.chance(1, 8) {
            let at = rng.below(row.len());
            row[at] = f32::INFINITY;
        }
        let greedy = row
            .iter()
            .enumerate()
            .fold(0, |best, (i, v)| if *v > row[best] { i } else { best });
        assert_eq!(argmax(&row), Some(greedy as u32));
        assert_eq!(pick(&row, None).expect("a token"), greedy as u32);
        let (temperature, top_p) = *rng.pick(&[(0.0, 1.0), (5e-8, 0.3), (1.0, 0.0), (2.0, -1.0)]);
        let mut sampler = Sampler::new(Sampling {
            temperature,
            top_p,
            seed: rng.next(),
        });
        assert!(sampler.is_greedy());
        assert_eq!(sampler.sample(&row).expect("a token"), greedy as u32);
    });
}

fn drawn(rng: &mut Rng) -> Sampling {
    Sampling {
        temperature: *rng.pick(&[0.2, 0.7, 1.0, 1.5, 2.0]),
        top_p: *rng.pick(&[1.0, 0.99, 0.9, 0.5, 0.1, 0.01]),
        seed: rng.next(),
    }
}

/// Item 6. The same seed over the same rows draws the same tokens; and a
/// draw is never a masked token nor one outside the nucleus: the tokens more
/// probable than the one drawn weigh less than `top_p`.
#[test]
fn a_seed_repeats_its_draws_and_no_draw_leaves_the_nucleus() {
    check("sampling_nucleus", 1500, 16, |rng, size| {
        let sampling = drawn(rng);
        let rows: Vec<Vec<f32>> = (0..rng.between(1, 6)).map(|_| logits(rng, size)).collect();
        let draw = || -> Vec<u32> {
            let mut sampler = Sampler::new(sampling);
            rows.iter()
                .map(|r| sampler.sample(r).expect("a token"))
                .collect()
        };
        let tokens = draw();
        assert_eq!(draw(), tokens, "{sampling:?}");
        for (row, &t) in rows.iter().zip(&tokens) {
            let t = t as usize;
            assert!(
                row[t] > f32::NEG_INFINITY,
                "a masked token drawn: {sampling:?}"
            );
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weight = |v: f32| (f64::from(v - max) / sampling.temperature).exp();
            let total: f64 = row.iter().map(|&v| weight(v)).sum();
            // Clearly above: two logits a rounding apart are one probability
            // in f32, ordered by index.
            let above: f64 = row
                .iter()
                .map(|&v| weight(v))
                .filter(|&w| w > weight(row[t]) * 1.0001)
                .sum();
            assert!(
                above / total < sampling.top_p + 1e-3,
                "{:.6} of the mass is above the token drawn, top_p {}",
                above / total,
                sampling.top_p
            );
        }
    });
}

/// Item 6. A row no token can be drawn from is an error, never a panic and
/// never a token out of the row: a NaN anywhere, for every way of drawing;
/// an infinite logit, a row of minus infinities, the largest floats, an
/// empty row, under a temperature that overflows the scaled logits.
#[test]
fn a_row_that_cannot_be_drawn_from_is_refused_without_a_panic() {
    check("sampling_refusals", 3000, 12, |rng, size| {
        let mut row = logits(rng, size);
        let poison = rng.below(7);
        let at = rng.below(row.len());
        match poison {
            0 => row[at] = f32::NAN,
            1 => row[at] = f32::INFINITY,
            2 => row.fill(f32::NEG_INFINITY),
            3 => row[at] = f32::MAX,
            4 => row[at] = f32::MIN,
            5 => row.clear(),
            _ => row.fill(f32::INFINITY),
        }
        let mut sampling = drawn(rng);
        if rng.chance(1, 4) {
            sampling.temperature = *rng.pick(&[0.0, 1e-6, 1e-30, 1e30]);
        }
        let mut sampler = Sampler::new(sampling);
        for drew in [sampler.sample(&row), pick(&row, None)] {
            match drew {
                Err(_) => {}
                Ok(_) if poison == 0 || poison == 5 => panic!("a token from {row:?}"),
                Ok(t) => assert!((t as usize) < row.len(), "token {t} of {}", row.len()),
            }
        }
    });
}

/// Item 6. Under a schema the draw keeps to the mask at every temperature:
/// a sampled answer is a document, and the same seed gives it again.
#[test]
fn a_sampled_constrained_answer_repeats_by_seed() {
    let b = bench();
    check("sampling_constrained", 60, 4, |rng, size| {
        let source = gen_schema(rng, size, 0);
        let schema = compiled(&source);
        let budget = rng.between(1, 40);
        let params = constrained(&schema, budget, Some(drawn(rng)));
        let seed = rng.next();
        let run = || {
            let mut rows = Rng::new(seed);
            let (out, _) = generate(&b, &params, &mut NoSink, |_| {
                row_of(&mut rows, &b, Taste::Any, &[])
            });
            out.unwrap_or_else(|e| panic!("{source}: {e:#}"))
        };
        let c = run();
        holds(&source, &schema, &c, budget);
        assert_eq!(run(), c);
    });
}

/// A model's kept table against a table read for the request. One
/// [`TableCache`] serves every case, as a loaded model serves its requests:
/// random schemas one after the other, each model's taste, greedy or sampled.
/// Each answer is, token for token, the one of the same request over a table
/// read for it alone ([`run`]): the same ids fed, the same stream, the same
/// completion or the same error. The kept table is read from the tokenizer
/// and the end tokens, by a call that takes nothing of a request.
#[test]
fn a_kept_table_answers_token_for_token_as_a_table_read_for_the_request() {
    let b = bench();
    let breakers = breakers(&b);
    let kept = TableCache::default();
    check("constraint_kept_table", 200, 6, |rng, size| {
        let source = gen_schema(rng, size, 0);
        let schema = compiled(&source);
        let taste = match rng.below(8) {
            0 | 1 => Taste::Any,
            2 | 3 => Taste::Breaker(50.0),
            4 | 5 => Taste::Rambler,
            6 => Taste::Hasty,
            _ => Taste::Dead,
        };
        let sampling = rng.chance(1, 3).then(|| drawn(rng));
        let budget = rng.between(1, 8 * size);
        let params = constrained(&schema, budget, sampling);
        let seed = rng.next();
        let req = Prepared {
            prompt_ids: vec![0x41],
            max_tokens: budget,
        };
        let answer = |table: Option<&TokenTable>| {
            let mut rows = Rng::new(seed);
            let mut sink = Recorder::default();
            let log = Rc::new(RefCell::new(SessionLog::default()));
            let rows = |_: &[u32]| row_of(&mut rows, &b, taste, &breakers);
            let (tk, eos) = (&b.tokenizer, &b.eos);
            let out = match table {
                Some(_) => run_with_table(tk, eos, table, &req, &params, &mut sink, |_| {
                    Ok(RowSession::new(rows, log.clone()))
                }),
                None => run(tk, eos, &req, &params, &mut sink, |_| {
                    Ok(RowSession::new(rows, log.clone()))
                }),
            };
            let fed = log.borrow().fed.clone();
            (out.map_err(|e| format!("{e:#}")), fed, sink.deltas)
        };
        let table = kept.read(&b.tokenizer, &b.eos).expect("a byte-level table");
        let from_kept = answer(Some(table));
        let from_fresh = answer(None);
        assert_eq!(from_kept, from_fresh, "{taste:?} on {source}");
        if !matches!(taste, Taste::Dead) {
            assert!(
                from_kept.0.is_ok(),
                "{taste:?} on {source}: {:?}",
                from_kept.0
            );
        }
    });
    // After every request, the kept table is still the tokenizer's.
    let fresh = TokenTable::from_tokenizer(&b.tokenizer, &b.eos).unwrap();
    let table = kept.read(&b.tokenizer, &b.eos).unwrap();
    assert_eq!(table.len(), fresh.len());
    for id in 0..fresh.len() as u32 {
        assert_eq!(table.bytes(id), fresh.bytes(id), "id {id}");
    }
}

/// Item 6. `top_k` is not a parameter the engine draws with: a request that
/// sends it is refused by its name, not answered without it.
#[test]
fn top_k_is_refused_by_name_rather_than_ignored() {
    let request = json!({"messages": [{"role": "user", "content": "x"}], "top_k": 40});
    let (status, said) = crate::server::parse_chat_request(request).expect_err("parsed");
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert!(said.contains("top_k"), "{said}");
}
