//! The token mask of a `json_schema` answer: the tokens whose bytes keep a
//! valid prefix ([`crate::json_schema`]), bridge tokens included, and EOS once
//! complete. Its shortcuts equal feeding every token (tested, 2026-10-30).

use std::collections::HashMap;

use crate::json_schema::{JsonSchema, Matcher};

/// Each token id's bytes, as the tokenizer's decoder writes them.
pub struct TokenTable {
    /// `None`: never part of a document (special, end-of-sequence, empty,
    /// or decoded to U+FFFD).
    bytes: Vec<Option<Box<[u8]>>>,
    /// Ids that keep an open string open, offered there without a check.
    neutral: Vec<u32>,
    /// Every other offered id, by its first two bytes.
    by_first: Vec<Bucket>,
    /// Every offered id, by its first two bytes.
    all_by_first: Vec<Bucket>,
}

/// The ids of one first byte, one-byte ids first, the rest by second byte.
/// 2026-10-30, Qwen3: after `{`, feeding every id that starts with a space
/// cost 3 ms a mask; split by second byte, 48 µs.
#[derive(Clone, Default)]
struct Bucket {
    ids: Vec<u32>,
    /// `ids[at[k]..at[k + 1]]`: one byte long for `k` 0, second byte `k - 1`.
    at: Vec<u32>,
}

impl Bucket {
    fn of(mut ids: Vec<u32>, bytes: &[Option<Box<[u8]>>]) -> Self {
        let key = |id: &u32| {
            bytes[*id as usize]
                .as_deref()
                .and_then(|b| b.get(1))
                .map_or(0, |&s| s as usize + 1)
        };
        ids.sort_by_key(key);
        let mut at = vec![0u32; 258];
        for id in &ids {
            at[key(id) + 1] += 1;
        }
        for k in 1..258 {
            at[k] += at[k - 1];
        }
        Self { ids, at }
    }

    fn range(&self, k: usize) -> &[u32] {
        match self.at.get(k..k + 2) {
            Some(&[a, z]) => &self.ids[a as usize..z as usize],
            _ => &[],
        }
    }
}

/// GPT-2's byte-level alphabet: the character each byte is written as.
fn gpt2_chars() -> HashMap<char, u8> {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut shifted = 0u32;
    (0..=255u8)
        .map(|b| {
            let c = if printable(b) {
                char::from(b)
            } else {
                shifted += 1;
                char::from_u32(255 + shifted).unwrap_or('\0')
            };
            (c, b)
        })
        .collect()
}

impl TokenTable {
    pub fn new(bytes: Vec<Option<Vec<u8>>>) -> Self {
        let bytes: Vec<Option<Box<[u8]>>> = bytes
            .into_iter()
            .map(|b| b.filter(|b| !b.is_empty()).map(Vec::into_boxed_slice))
            .collect();
        let mut neutral = Vec::new();
        let mut by_first = vec![Vec::new(); 256];
        let mut all_by_first = vec![Vec::new(); 256];
        for (id, b) in bytes.iter().enumerate() {
            let Some(b) = b else { continue };
            let id = id as u32;
            all_by_first[b[0] as usize].push(id);
            let plain = std::str::from_utf8(b).is_ok()
                && !b.iter().any(|&c| c == b'"' || c == b'\\' || c < 0x20);
            if plain {
                neutral.push(id);
            } else {
                by_first[b[0] as usize].push(id);
            }
        }
        let buckets = |v: Vec<Vec<u32>>| -> Vec<Bucket> {
            v.into_iter().map(|ids| Bucket::of(ids, &bytes)).collect()
        };
        let by_first = buckets(by_first);
        let all_by_first = buckets(all_by_first);
        Self {
            bytes,
            neutral,
            by_first,
            all_by_first,
        }
    }

    /// The table of `tok`'s vocabulary. Only decoders that write a sequence as
    /// the concatenation of its tokens are read (`ByteLevel`, `Fuse`); another
    /// is refused by name, since a token's text then depends on its neighbours.
    pub fn from_tokenizer(tok: &tokenizers::Tokenizer, eos: &[u32]) -> Result<Self, String> {
        use tokenizers::decoders::DecoderWrapper;
        let byte_level = match tok.get_decoder() {
            Some(DecoderWrapper::ByteLevel(_)) => true,
            Some(DecoderWrapper::Fuse(_)) => false,
            Some(other) => {
                let name = serde_json::to_value(other)
                    .ok()
                    .and_then(|v| v["type"].as_str().map(str::to_string))
                    .unwrap_or_else(|| "unknown".into());
                return Err(format!(
                    "the model's tokenizer decodes tokens in context ({name} decoder); \
                     json_schema needs a ByteLevel or Fuse decoder"
                ));
            }
            None => {
                return Err("the model's tokenizer has no decoder; json_schema needs a \
                     ByteLevel or Fuse decoder"
                    .into())
            }
        };
        let n = tok
            .get_vocab(true)
            .values()
            .max()
            .map_or(0, |m| *m as usize + 1);
        let added = tok.get_added_tokens_decoder();
        let chars = gpt2_chars();
        let bytes = (0..n as u32)
            .map(|id| {
                if eos.contains(&id) || added.get(&id).is_some_and(|a| a.special) {
                    return None;
                }
                let piece = tok.id_to_token(id)?;
                // As tokenizers' ByteLevel decoder: a token all in the alphabet
                // is its bytes, any other is its own UTF-8.
                let b = if byte_level {
                    piece
                        .chars()
                        .map(|c| chars.get(&c).copied())
                        .collect::<Option<Vec<u8>>>()
                        .unwrap_or_else(|| piece.as_bytes().to_vec())
                } else {
                    piece.into_bytes()
                };
                let lossy = "\u{FFFD}".as_bytes();
                (!b.windows(3).any(|w| w == lossy)).then_some(b)
            })
            .collect();
        Ok(Self::new(bytes))
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn bytes(&self, id: u32) -> Option<&[u8]> {
        self.bytes.get(id as usize).and_then(|b| b.as_deref())
    }
}

/// One answer's constraint: the automaton's state and the bytes fed so far.
pub struct Constraint<'a> {
    m: Matcher<'a>,
    table: &'a TokenTable,
    eos: &'a [u32],
    fed: Vec<u8>,
}

impl<'a> Constraint<'a> {
    pub fn new(schema: &'a JsonSchema, table: &'a TokenTable, eos: &'a [u32]) -> Self {
        Self {
            m: schema.matcher(),
            table,
            eos,
            fed: Vec::new(),
        }
    }

    /// The ids below `vocab` that may come next.
    pub fn allowed(&self, vocab: usize) -> Vec<bool> {
        let mut out = vec![false; vocab];
        let plain = self.m.in_plain_string();
        if plain {
            for &id in &self.table.neutral {
                if let Some(o) = out.get_mut(id as usize) {
                    *o = true;
                }
            }
        }
        let buckets = if plain {
            &self.table.by_first
        } else {
            &self.table.all_by_first
        };
        for first in 0..=255u8 {
            let mut after = self.m;
            if !after.feed(first) {
                continue;
            }
            let bucket = &buckets[first as usize];
            for &id in bucket.range(0) {
                if let Some(o) = out.get_mut(id as usize) {
                    *o = true;
                }
            }
            for second in 0..=255u8 {
                let ids = bucket.range(second as usize + 1);
                if ids.is_empty() {
                    continue;
                }
                let mut after2 = after;
                if !after2.feed(second) {
                    continue;
                }
                for &id in ids {
                    let (Some(o), Some(b)) = (out.get_mut(id as usize), self.table.bytes(id))
                    else {
                        continue;
                    };
                    let mut m = after2;
                    *o = m.feed_bytes(&b[2..]);
                }
            }
        }
        if self.m.complete() {
            for &e in self.eos {
                if let Some(o) = out.get_mut(e as usize) {
                    *o = true;
                }
            }
        }
        out
    }

    /// Sets every id the schema forbids to minus infinity in `row`; an error
    /// when none is left.
    pub fn mask_logits(&self, row: &mut [f32]) -> anyhow::Result<()> {
        let allowed = self.allowed(row.len());
        if !allowed.iter().any(|a| *a) {
            anyhow::bail!("json_schema: no token of this vocabulary continues the document");
        }
        for (x, ok) in row.iter_mut().zip(allowed) {
            if !ok {
                *x = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    /// Takes the picked id; a refused one leaves the state as it was.
    pub fn advance(&mut self, id: u32) -> anyhow::Result<()> {
        if self.eos.contains(&id) {
            anyhow::ensure!(
                self.m.complete(),
                "json_schema: end of sequence in an open document"
            );
            return Ok(());
        }
        let Some(b) = self.table.bytes(id) else {
            anyhow::bail!("json_schema: token {id} is not text a document holds");
        };
        let mut m = self.m;
        anyhow::ensure!(
            m.feed_bytes(b),
            "json_schema: token {id} breaks the document"
        );
        self.m = m;
        self.fed.extend_from_slice(b);
        Ok(())
    }

    /// No byte may follow: the answer ends here.
    pub fn done(&self) -> bool {
        self.m.done()
    }

    pub fn complete(&self) -> bool {
        self.m.complete()
    }

    /// Whether `text`, the tokenizer's decode of the answer, is the bytes the
    /// mask let through (U+FFFD for a character cut at the end).
    pub fn joins_to(&self, text: &str) -> bool {
        String::from_utf8_lossy(&self.fed) == text
    }

    #[cfg(test)]
    pub(crate) fn set_matcher_for_test(&mut self, m: &Matcher<'a>) {
        self.m = *m;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    fn table(pieces: &[&[u8]]) -> TokenTable {
        TokenTable::new(
            pieces
                .iter()
                .map(|p| (!p.is_empty()).then(|| p.to_vec()))
                .collect(),
        )
    }

    fn ids(allowed: &[bool]) -> Vec<usize> {
        allowed
            .iter()
            .enumerate()
            .filter(|x| *x.1)
            .map(|x| x.0)
            .collect()
    }

    /// The mask lets through exactly the pieces that keep the prefix valid,
    /// and end-of-sequence only once the document may end.
    #[test]
    fn the_mask_follows_the_prefix_and_gates_the_end() {
        let s = JsonSchema::compile(
            &json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}),
        )
        .unwrap();
        let t = table(&[
            b"{",
            b"}",
            b"\"ok\"",
            b":",
            b"true",
            b"tr",
            b"ue}",
            b" ",
            b"",
            b"{\"ok\":",
        ]);
        let eos = [8u32];
        let mut c = Constraint::new(&s, &t, &eos);
        assert_eq!(ids(&c.allowed(12)), [0, 9], "{{ and the bridge {{\"ok\":");
        for id in [0, 2, 3] {
            c.advance(id).unwrap();
        }
        assert_eq!(ids(&c.allowed(12)), [4, 5, 7], "true, tr and one space");
        assert!(c.advance(1).is_err(), "a token the mask refused");
        c.advance(5).unwrap();
        assert_eq!(ids(&c.allowed(12)), [6]);
        c.advance(6).unwrap();
        assert!(c.done() && c.complete());
        assert_eq!(ids(&c.allowed(12)), [8], "only the end once closed");
        assert!(c.joins_to("{\"ok\":true}"));
        assert!(!c.joins_to("{\"ok\": true}"));
        let mut row = vec![0.5f32; 12];
        c.mask_logits(&mut row).unwrap();
        assert!(row
            .iter()
            .enumerate()
            .all(|(i, v)| (i == 8) == v.is_finite()));
    }

    /// A row in which no token continues the document is an error, never a
    /// token the schema forbids.
    #[test]
    fn a_row_with_nothing_left_to_pick_is_an_error() {
        let s = JsonSchema::compile(&json!({"type": "null"})).unwrap();
        let t = table(&[b"x", b"nul"]);
        let c = Constraint::new(&s, &t, &[]);
        let mut row = vec![0.0f32; 2];
        c.mask_logits(&mut row).unwrap();
        assert_eq!(row[1], 0.0);
        let mut m = Constraint::new(&s, &t, &[]);
        m.advance(1).unwrap();
        let mut row = vec![0.0f32; 2];
        assert!(
            m.mask_logits(&mut row).is_err(),
            "no `l` in this vocabulary"
        );
    }

    /// Pieces a byte-level vocabulary holds: structure, the bridges between
    /// structure and text, partial characters, escapes, controls, words.
    pub(crate) fn varied_pieces() -> Vec<Vec<u8>> {
        let mut v: Vec<Vec<u8>> = [
            "{", "}", "[", "]", ":", ",", "\"", "{\"", "\":", "\",", "\"}", "]}", "\"]", "},",
            "\":\"", "\":[", "\":{", "true", "false", "null", "tr", "ue", "nu", "ll", "0", "1",
            "2", "10", "-", ".", "e", "E", "+", "5e", "a", "b", "ab", "name", "age", "tags",
            "score", "ok", "n", "na", "me", "\\", "\\n", "\\\"", "\\u", "00", "e9", "d8", "\n",
            " ", "\t", " \"", "é", "東京", "🗼", "A\"", "x\\", "\"a\"", "\"b\"",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
        // Partial characters: the lead and continuation bytes of é and 🗼.
        v.extend([
            vec![0xC3],
            vec![0xA9],
            vec![0xF0, 0x9F],
            vec![0x97, 0xBC],
            vec![0xBC],
            vec![0xC3, 0xA9, b'"'],
            vec![0x1F],
        ]);
        v
    }

    /// SECURITY: the shortcuts (plain string tokens, two-byte buckets) give the
    /// mask of feeding every token, at every byte of several documents. A
    /// failure: a forbidden token can be picked, or an allowed one never is.
    #[test]
    fn the_fast_mask_is_the_mask_of_feeding_every_token() {
        let pieces = varied_pieces();
        let t = TokenTable::new(pieces.iter().cloned().map(Some).collect());
        let eos = [pieces.len() as u32];
        let schemas = [
            (
                json!({"type": "object", "properties": {
                    "name": {"type": "string"}, "age": {"type": "integer"},
                    "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2},
                    "score": {"type": ["number", "null"]}, "ok": {"type": "boolean"}},
                    "required": ["name", "age"]}),
                r#"{"name":"\"é\n\u00e9 東京🗼","age":-10,"tags":["a"],"score":5e+2,"ok":false}"#,
            ),
            (
                json!({"type": "array", "items": {"type": "string"}}),
                r#"["ab","","\\x"]"#,
            ),
            (json!({"enum": [10, 1, "na\"me", null]}), r#""na\"me""#),
        ];
        for (s, doc) in schemas {
            let s = JsonSchema::compile(&s).unwrap();
            let doc = doc.as_bytes();
            for cut in 0..=doc.len() {
                let mut m = s.matcher();
                assert!(m.feed_bytes(&doc[..cut]), "{cut}");
                let mut c = Constraint::new(&s, &t, &eos);
                c.set_matcher_for_test(&m);
                let fast = c.allowed(pieces.len() + 1);
                let slow: Vec<bool> = pieces
                    .iter()
                    .map(|p| m.clone().feed_bytes(p))
                    .chain(std::iter::once(m.complete()))
                    .collect();
                assert_eq!(
                    ids(&fast),
                    ids(&slow),
                    "after {:?}",
                    String::from_utf8_lossy(&doc[..cut])
                );
            }
        }
    }

    /// A GPT-2 byte-level BPE: the 256 byte tokens, merges for `é`, `{"` and
    /// `":`, a special end-of-turn token and a plain added one.
    pub(crate) fn byte_level_bpe() -> tokenizers::Tokenizer {
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            vocab.insert(gpt2_char(b).to_string(), u32::from(b).into());
        }
        let merged = [("Ã", "©"), ("{", "\""), ("\"", ":")];
        for (i, (a, b)) in merged.iter().enumerate() {
            vocab.insert(format!("{a}{b}"), (256 + i).into());
        }
        let merges: Vec<String> = merged.iter().map(|(a, b)| format!("{a} {b}")).collect();
        let byte_level = json!({
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true,
        });
        let added = |id: usize, content: &str, special: bool| {
            json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": special})
        };
        let j = json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(259, "<|im_end|>", true), added(260, "<think>", false)],
            "normalizer": null, "pre_tokenizer": byte_level.clone(),
            "post_processor": null, "decoder": byte_level,
            "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        });
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).unwrap()
    }

    /// GPT-2's printable stand-in for byte `b`.
    fn gpt2_char(b: u8) -> char {
        let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        if printable(b) {
            return char::from(b);
        }
        let shifted = (0..=b).filter(|&x| !printable(x)).count() as u32;
        char::from_u32(255 + shifted).unwrap()
    }

    /// Byte-level tokens are their bytes: a lead byte alone is offered inside
    /// a string as itself, never as the U+FFFD a lone decode makes of it.
    /// Special tokens and end-of-sequence are never pieces of a document.
    #[test]
    fn a_byte_level_vocabulary_is_read_as_bytes() {
        let tok = byte_level_bpe();
        let t = TokenTable::from_tokenizer(&tok, &[259]).unwrap();
        assert_eq!(t.len(), 261);
        assert_eq!(t.bytes(0xC3), Some(&[0xC3u8][..]));
        assert_eq!(t.bytes(u32::from(b' ')), Some(&b" "[..]));
        assert_eq!(t.bytes(256), Some("é".as_bytes()));
        assert_eq!(t.bytes(257), Some(&b"{\""[..]));
        assert_eq!(t.bytes(259), None, "end of sequence");
        assert_eq!(
            t.bytes(260),
            Some(&b"<think>"[..]),
            "a plain added token is text"
        );
        for id in 0..259u32 {
            let one = tok.decode(&[id], true).unwrap();
            if !one.contains('\u{FFFD}') {
                assert_eq!(t.bytes(id), Some(one.as_bytes()), "{id}");
            }
        }
        let s = JsonSchema::compile(&json!({"type": "string"})).unwrap();
        let mut c = Constraint::new(&s, &t, &[259]);
        c.advance(u32::from(b'"')).unwrap();
        assert!(c.allowed(261)[0xC3], "a lead byte opens a character");
        c.advance(0xC3).unwrap();
        let after = c.allowed(261);
        assert!(after[0xA9] && !after[usize::from(b'"')] && !after[256]);
    }

    /// A vocabulary that is not byte-level: each id's own decode, and an id
    /// that decodes to nothing or to U+FFFD is never offered.
    #[test]
    fn another_vocabulary_is_read_by_its_decoder() {
        let j = json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [{"id": 3, "content": "<s>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}],
            "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null, "decoder": {"type": "Fuse"},
            "model": {"type": "WordLevel", "vocab": {"{": 0, "}": 1, "\u{FFFD}": 2, "<s>": 3},
                "unk_token": "{"},
        });
        let tok = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).unwrap();
        let t = TokenTable::from_tokenizer(&tok, &[]).unwrap();
        assert_eq!(t.bytes(0), Some(&b"{"[..]));
        assert_eq!(t.bytes(2), None);
        assert_eq!(t.bytes(3), None);
    }

    /// A decoder that writes a token by its neighbours (no decoder: words
    /// joined by spaces; Metaspace: a leading space dropped) is refused by
    /// name: the bytes offered would not be the text served.
    #[test]
    fn a_decoder_that_reads_tokens_in_context_is_refused() {
        let words = crate::candle_backend::tests::word_tokenizer(4);
        let tok = tokenizers::Tokenizer::from_bytes(words).unwrap();
        let e = TokenTable::from_tokenizer(&tok, &[]).err().unwrap();
        assert!(e.contains("no decoder"), "{e}");
        let mut j: serde_json::Value =
            serde_json::from_slice(&crate::candle_backend::tests::word_tokenizer(4)).unwrap();
        j["decoder"] = json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true});
        let tok = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&j).unwrap()).unwrap();
        let e = TokenTable::from_tokenizer(&tok, &[]).err().unwrap();
        assert!(e.contains("Metaspace"), "{e}");
    }

    /// Measured, not asserted: the mask's cost on a real vocabulary. Run with
    /// `X0NE_TOKENIZER_JSON=<tokenizer.json> cargo test -p x0ne-infer --release
    /// constrain::tests::mask_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs a tokenizer.json on disk"]
    fn mask_cost_on_a_real_vocabulary() {
        let path = std::env::var("X0NE_TOKENIZER_JSON").expect("X0NE_TOKENIZER_JSON");
        let tok = tokenizers::Tokenizer::from_file(path).unwrap();
        let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tok.token_to_id(t))
            .collect();
        let t0 = std::time::Instant::now();
        let t = TokenTable::from_tokenizer(&tok, &eos).unwrap();
        let built = t0.elapsed();
        let offered = (0..t.len() as u32)
            .filter(|&i| t.bytes(i).is_some())
            .count();
        println!(
            "table: {} ids, {offered} offered, built in {built:?}",
            t.len()
        );
        let s = JsonSchema::compile(&json!({"type": "object", "properties": {
            "name": {"type": "string"}, "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "kind": {"enum": ["cat", "dog", "bird"]}}, "required": ["name", "age"]}))
        .unwrap();
        let doc =
            r#"{"name":"Ada Lovelace, née Byron","age":36,"tags":["math","poetry"],"kind":"bird"}"#;
        let mut worst = std::time::Duration::ZERO;
        let mut total = std::time::Duration::ZERO;
        for cut in 0..doc.len() {
            let mut m = s.matcher();
            if !m.feed_bytes(&doc.as_bytes()[..cut]) {
                continue;
            }
            let mut c = Constraint::new(&s, &t, &eos);
            c.set_matcher_for_test(&m);
            let t0 = std::time::Instant::now();
            let n = c.allowed(t.len()).iter().filter(|a| **a).count();
            let took = t0.elapsed();
            total += took;
            if took > worst {
                worst = took;
                println!("after {:?}: {n} allowed in {took:?}", &doc[..cut]);
            }
        }
        println!(
            "{} masks: mean {:?}, worst {worst:?}",
            doc.len(),
            total / doc.len() as u32
        );
    }
}
