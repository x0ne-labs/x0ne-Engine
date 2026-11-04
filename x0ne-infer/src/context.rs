// SPDX-License-Identifier: Apache-2.0
//! The context a turn of a conversation reused, as its receipt signs it, and
//! the check the relay (`local_api.rs`) and `x0ne-sdk` (`#[path]`) run on it.
//!
//! A request may name a conversation: `"x0ne": {"conversation": "new"}`, then
//! the handle the previous answer carried. The TD keeps that conversation's
//! keys and values between its turns (`crate::conversation`) and reuses the
//! prefix the next prompt shares with them. Such a turn's receipt signs
//! `kv_cache_shared_between_requests: true` and a `SignedContext`: how many
//! prompt tokens were reused, a commitment to them, and one to the handle
//! sent and the handle handed back. A receipt that hides a reuse, or claims
//! one the request did not ask for, is refused here.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// `x0ne.conversation` that opens a conversation.
pub const NEW: &str = "new";
/// The scope every signed context names: keys and values an earlier turn of
/// this conversation computed, in this TD, for this route.
pub const SCOPE: &str = "session";
/// The label of the commitment to the reused token ids.
pub const REUSED_PREFIX_LABEL: &[u8] = b"x0ne-infer/commit/reused-prefix/v1\0";
/// The label of the commitment to the handle sent and the handle issued.
pub const CONVERSATION_LABEL: &[u8] = b"x0ne-infer/commit/conversation/v1\0";

/// The TD's 32 random bytes that name one turn's successor: a bearer
/// secret. Compared in constant time, never printed by `Debug`, zeroed when
/// dropped.
#[derive(Clone)]
pub struct Handle([u8; 32]);

impl Handle {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// 64 lowercase hex digits, or nothing: the reason never quotes `s`.
    pub fn parse(s: &str) -> Option<Self> {
        if s.len() != 64 || !s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(Self(out))
    }

    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl PartialEq for Handle {
    /// Every byte is compared, wherever the first difference falls.
    fn eq(&self, other: &Self) -> bool {
        let diff = self
            .0
            .iter()
            .zip(other.0.iter())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b));
        std::hint::black_box(diff) == 0
    }
}

impl Eq for Handle {}

impl std::fmt::Debug for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Handle(..)")
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.0);
    }
}

impl Serialize for Handle {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.hex())
    }
}

impl<'de> Deserialize<'de> for Handle {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::parse(&s)
            .ok_or_else(|| serde::de::Error::custom("a handle is 64 lowercase hex digits"))
    }
}

/// What a request asks of a conversation. The default asks nothing: nothing
/// is kept after it, nothing reused, and its receipt signs no context.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AskedConversation {
    #[default]
    None,
    /// `"new"`. `end`: keep nothing after this turn.
    New { end: bool },
    /// The handle the previous answer carried.
    Continue { handle: Handle, end: bool },
}

impl AskedConversation {
    /// From the request's `x0ne` object: `conversation` (`"new"` or a handle)
    /// and `end_conversation` (a boolean, only beside `conversation`). Other
    /// keys are the server's to refuse. A refusal is the field and why, never
    /// its value.
    pub fn from_x0ne(options: Option<&Value>) -> Result<Self, (&'static str, String)> {
        let Some(Value::Object(o)) = options else {
            return Ok(Self::None);
        };
        let end = match o.get("end_conversation") {
            None | Some(Value::Null) => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(_) => return Err(("end_conversation", "true or false".into())),
        };
        let end_flag = end.unwrap_or(false);
        match o.get("conversation") {
            None | Some(Value::Null) if end.is_some() => {
                Err(("end_conversation", "only beside conversation".into()))
            }
            None | Some(Value::Null) => Ok(Self::None),
            Some(Value::String(s)) if s == NEW => Ok(Self::New { end: end_flag }),
            Some(Value::String(s)) => Handle::parse(s)
                .map(|handle| Self::Continue {
                    handle,
                    end: end_flag,
                })
                .ok_or_else(|| {
                    (
                        "conversation",
                        "\"new\" or the 64 lowercase hex digits of the handle the previous \
                         answer carried"
                            .into(),
                    )
                }),
            Some(_) => Err(("conversation", "a string".into())),
        }
    }

    /// The handle sent, `None` for `"new"` and without a conversation.
    pub fn presented(&self) -> Option<&Handle> {
        match self {
            Self::Continue { handle, .. } => Some(handle),
            _ => None,
        }
    }

    /// Whether the receipt `signed` (its signed payload) says what this ask
    /// implies, and the handle in `x0ne` (the answer's `x0ne` block, with its
    /// opening) is the one it commits to. Without a conversation: no context,
    /// `kv_cache_shared_between_requests` false, no handle. With one: a
    /// context in scope `session`, the cache flag true, no prefix shared
    /// between sessions, at most the prompt less one token reused, 0 on turn
    /// 1, and the commitment opens to the handle sent and the one received.
    pub fn check_signed(&self, signed: &Value, x0ne: &Value) -> Result<(), String> {
        // Absent, a flag claims nothing; a turn of a conversation must sign it true.
        let flag = |name: &str| match signed.get("isolation").and_then(|i| i.get(name)) {
            None => Ok(false),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => Err(format!("isolation.{name}: not a boolean")),
        };
        if flag("prefix_cache_shared_between_sessions")? {
            return Err(
                "isolation: the receipt signs a prefix cache shared between \
                        sessions"
                    .into(),
            );
        }
        let shared = flag("kv_cache_shared_between_requests")?;
        let context = match signed.get("context") {
            None => None,
            Some(v) => Some(
                serde_json::from_value::<SignedContext>(v.clone())
                    .map_err(|e| format!("context: {e}"))?,
            ),
        };
        if matches!(self, Self::None) {
            if shared {
                return Err("isolation: the receipt signs a KV cache shared between \
                            requests; the request named no conversation"
                    .into());
            }
            if context.is_some() {
                return Err(
                    "context: the receipt signs reused context; the request named no \
                            conversation"
                        .into(),
                );
            }
            if x0ne.get("conversation").is_some() {
                return Err("conversation: a handle the request did not ask for".into());
            }
            return Ok(());
        }
        let Some(context) = context else {
            return Err(
                "context: the request named a conversation; the receipt signs no \
                        context"
                    .into(),
            );
        };
        if !shared {
            return Err("isolation: a turn of a conversation signs \
                        kv_cache_shared_between_requests false"
                .into());
        }
        if context.scope != SCOPE {
            return Err("context: a scope other than session".into());
        }
        let prompt_tokens = signed
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .ok_or("prompt_tokens: not signed")?;
        if context.turn == 0 {
            return Err("context: turn 0".into());
        }
        if context.reused_tokens > 0 && u64::from(context.reused_tokens) >= prompt_tokens {
            return Err("context: more tokens reused than the prompt less its last".into());
        }
        if context.turn == 1 && context.reused_tokens > 0 {
            return Err("context: tokens reused on the turn that opens a conversation".into());
        }
        if matches!(self, Self::New { .. }) && context.turn != 1 {
            return Err("context: \"new\" answered as a later turn".into());
        }
        let issued: Handle = x0ne
            .get("conversation")
            .and_then(|c| c.get("handle"))
            .cloned()
            .ok_or("conversation: no handle in the answer")
            .and_then(|v| {
                serde_json::from_value(v).map_err(|_| "conversation: a malformed handle")
            })?;
        let salt = x0ne
            .get("opening")
            .and_then(|o| o.get("salt_hex"))
            .and_then(Value::as_str)
            .and_then(unhex32)
            .ok_or("opening: no salt")?;
        if conversation_commitment(&salt, self.presented(), &issued)
            != context.conversation_commitment_hex
        {
            return Err(
                "context: the commitment does not open to the handle sent and the \
                        handle received"
                    .into(),
            );
        }
        Ok(())
    }
}

/// What a turn of a conversation signs about the context it reused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedContext {
    /// Always [`SCOPE`].
    pub scope: String,
    /// 1 on a turn computed from token 0 (`"new"`, or a handle the TD did not
    /// hold: evicted, expired, spent, another route, another boot), then one
    /// more per continued turn.
    pub turn: u32,
    /// Prompt tokens whose keys and values an earlier turn computed.
    pub reused_tokens: u32,
    /// SHA-256([`REUSED_PREFIX_LABEL`] ‖ salt ‖ the reused ids, u32 LE).
    pub reused_prefix_commitment_hex: String,
    /// SHA-256([`CONVERSATION_LABEL`] ‖ salt ‖ 0x01 and the handle sent, or
    /// 0x00 and 32 zero bytes for `"new"` ‖ the handle issued).
    pub conversation_commitment_hex: String,
}

impl SignedContext {
    /// Under `salt`, the receipt's salt.
    pub fn commit(
        salt: &[u8; 32],
        turn: u32,
        reused: &[u32],
        presented: Option<&Handle>,
        issued: &Handle,
    ) -> Self {
        Self {
            scope: SCOPE.into(),
            turn,
            reused_tokens: u32::try_from(reused.len()).unwrap_or(u32::MAX),
            reused_prefix_commitment_hex: reused_prefix_commitment(salt, reused),
            conversation_commitment_hex: conversation_commitment(salt, presented, issued),
        }
    }
}

/// The answer's `x0ne.conversation`: the handle of the next turn. Not
/// signed; the receipt's commitment binds it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Issued {
    pub handle: Handle,
}

pub fn reused_prefix_commitment(salt: &[u8; 32], reused: &[u32]) -> String {
    let mut h = Sha256::new();
    h.update(REUSED_PREFIX_LABEL);
    h.update(salt);
    for id in reused {
        h.update(id.to_le_bytes());
    }
    hex(&h.finalize())
}

pub fn conversation_commitment(
    salt: &[u8; 32],
    presented: Option<&Handle>,
    issued: &Handle,
) -> String {
    let mut h = Sha256::new();
    h.update(CONVERSATION_LABEL);
    h.update(salt);
    h.update([u8::from(presented.is_some())]);
    h.update(presented.map_or(&[0u8; 32], Handle::as_bytes));
    h.update(issued.as_bytes());
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::*;
    use serde_json::json;

    const SALT: [u8; 32] = [7; 32];

    fn handle(b: u8) -> Handle {
        Handle::from_bytes([b; 32])
    }

    /// The signed fields the check reads, and the answer's `x0ne` block.
    fn turn(
        kv_shared: bool,
        context: Option<&SignedContext>,
        prompt_tokens: u32,
        issued: Option<&Handle>,
    ) -> (Value, Value) {
        let mut signed = json!({"prompt_tokens": prompt_tokens, "isolation": {
            "kv_cache_shared_between_requests": kv_shared,
            "prefix_cache_shared_between_sessions": false,
            "requests_batched_across_clients": false, "prompt_text_logged": false}});
        if let Some(c) = context {
            signed["context"] = serde_json::to_value(c).unwrap();
        }
        let mut x0ne = json!({"opening": {"salt_hex": hex(&SALT)}});
        if let Some(h) = issued {
            x0ne["conversation"] = json!({"handle": h.hex()});
        }
        (signed, x0ne)
    }

    /// `conversation` and `end_conversation` are read or refused by name; a
    /// refusal never quotes the value, and `Debug` never prints a handle. A
    /// failure: a bearer handle copied into an error or a log line.
    #[test]
    fn a_conversation_is_read_by_name_and_never_quoted() {
        let read = |v: Value| AskedConversation::from_x0ne(Some(&v));
        assert_eq!(read(json!({})).unwrap(), AskedConversation::None);
        assert_eq!(
            AskedConversation::from_x0ne(None).unwrap(),
            AskedConversation::None
        );
        assert_eq!(
            read(json!({"conversation": "new"})).unwrap(),
            AskedConversation::New { end: false }
        );
        let h = handle(0xab);
        assert_eq!(
            read(json!({"conversation": h.hex(), "end_conversation": true})).unwrap(),
            AskedConversation::Continue {
                handle: h.clone(),
                end: true
            }
        );
        for (v, field) in [
            (json!({"conversation": "AB".repeat(32)}), "conversation"),
            (json!({"conversation": "ab".repeat(31)}), "conversation"),
            (json!({"conversation": "ab".repeat(33)}), "conversation"),
            (json!({"conversation": 7}), "conversation"),
            (json!({"end_conversation": true}), "end_conversation"),
            (
                json!({"conversation": "new", "end_conversation": "yes"}),
                "end_conversation",
            ),
        ] {
            let (f, why) = read(v.clone()).unwrap_err();
            assert_eq!(f, field, "{v}");
            assert!(!why.contains("AB") && !why.contains("abab"), "{why}");
        }
        assert_eq!(format!("{h:?}"), "Handle(..)");
        let asked = read(json!({"conversation": h.hex()})).unwrap();
        assert!(!format!("{asked:?}").contains("abab"), "{asked:?}");
    }

    /// SECURITY: a receipt that hides a reuse, or signs one the request did
    /// not ask for, or a handle it does not commit to, is refused. A failure:
    /// a caller told its context was computed afresh when it was reused, or
    /// the reverse.
    #[test]
    fn a_receipt_that_hides_or_invents_reused_context_is_refused() {
        let (sent, issued) = (handle(1), handle(2));
        let cont = AskedConversation::Continue {
            handle: sent.clone(),
            end: false,
        };
        let new = AskedConversation::New { end: false };
        let none = AskedConversation::None;
        let honest = SignedContext::commit(&SALT, 2, &[1, 2, 3], Some(&sent), &issued);
        let (s, x) = turn(true, Some(&honest), 10, Some(&issued));
        cont.check_signed(&s, &x).unwrap();
        let opened = SignedContext::commit(&SALT, 1, &[], None, &issued);
        let (s, x) = turn(true, Some(&opened), 10, Some(&issued));
        new.check_signed(&s, &x).unwrap();
        // A handle the TD did not hold: turn 1, nothing reused.
        let cold = SignedContext::commit(&SALT, 1, &[], Some(&sent), &issued);
        let (s, x) = turn(true, Some(&cold), 10, Some(&issued));
        cont.check_signed(&s, &x).unwrap();
        let (s, x) = turn(false, None, 10, None);
        none.check_signed(&s, &x).unwrap();

        let with = |f: &dyn Fn(&mut SignedContext)| {
            let mut c = honest.clone();
            f(&mut c);
            c
        };
        let cases = [
            (
                &cont,
                turn(true, None, 10, Some(&issued)),
                "signs no context",
            ),
            (
                &cont,
                turn(false, Some(&honest), 10, Some(&issued)),
                "false",
            ),
            (
                &cont,
                turn(true, Some(&honest), 3, Some(&issued)),
                "more tokens",
            ),
            (
                &cont,
                turn(
                    true,
                    Some(&with(&|c| c.scope = "global".into())),
                    10,
                    Some(&issued),
                ),
                "scope",
            ),
            (
                &cont,
                turn(true, Some(&with(&|c| c.turn = 1)), 10, Some(&issued)),
                "opens",
            ),
            (
                &cont,
                turn(true, Some(&with(&|c| c.turn = 0)), 10, Some(&issued)),
                "turn 0",
            ),
            (
                &cont,
                turn(true, Some(&honest), 10, Some(&handle(3))),
                "commitment",
            ),
            (&cont, turn(true, Some(&honest), 10, None), "no handle"),
            (
                &new,
                turn(true, Some(&honest), 10, Some(&issued)),
                "later turn",
            ),
            (&none, turn(true, None, 10, None), "named no conversation"),
            (
                &none,
                turn(false, Some(&honest), 10, None),
                "named no conversation",
            ),
            (&none, turn(false, None, 10, Some(&issued)), "did not ask"),
        ];
        for (asked, (s, x), word) in cases {
            let e = asked.check_signed(&s, &x).unwrap_err();
            assert!(e.contains(word), "{word}: {e}");
        }
        for (asked, (mut s, x)) in [
            (&cont, turn(true, Some(&honest), 10, Some(&issued))),
            (&none, turn(false, None, 10, None)),
        ] {
            s["isolation"]["prefix_cache_shared_between_sessions"] = json!(true);
            let e = asked.check_signed(&s, &x).unwrap_err();
            assert!(e.contains("between sessions"), "{e}");
        }
        let (mut s, x) = turn(true, Some(&honest), 10, Some(&issued));
        s["context"]["shared_with"] = json!("everyone");
        assert!(cont.check_signed(&s, &x).unwrap_err().contains("context"));
    }

    /// The commitments are the hashes the docs give: label, salt, ids in u32
    /// little-endian; `"new"` apart from any handle sent. A failure: a
    /// commitment a caller cannot recompute from docs/verification.md.
    #[test]
    fn the_commitments_are_the_documented_hashes() {
        let digest = |parts: &[&[u8]]| {
            let mut h = Sha256::new();
            for p in parts {
                h.update(p);
            }
            hex(&h.finalize())
        };
        assert_eq!(
            reused_prefix_commitment(&SALT, &[1, 256]),
            digest(&[REUSED_PREFIX_LABEL, &SALT, &[1, 0, 0, 0, 0, 1, 0, 0]])
        );
        assert_eq!(
            conversation_commitment(&SALT, None, &handle(2)),
            digest(&[CONVERSATION_LABEL, &SALT, &[0], &[0; 32], &[2; 32]])
        );
        assert_eq!(
            conversation_commitment(&SALT, Some(&handle(0)), &handle(2)),
            digest(&[CONVERSATION_LABEL, &SALT, &[1], &[0; 32], &[2; 32]])
        );
    }
}
