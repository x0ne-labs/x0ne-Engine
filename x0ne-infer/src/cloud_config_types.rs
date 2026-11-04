// SPDX-License-Identifier: Apache-2.0
//! The session config's shape, `x0ne-engine/cloud-config/v1`: what the relay
//! serializes and the TD hashes into every quote (`cloud_config.rs` parses,
//! validates and hashes it). Compiled into x0ne-sdk too (`#[path]`), which
//! reads it from a key document: serde only, no crate paths here.

use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "x0ne-engine/cloud-config/v1";
/// A config is a few hundred bytes; anything bigger is not one.
pub const MAX_BYTES: u64 = 64 * 1024;
/// `cpu` is parsed so x0ne-infer refuses it by name (exit 90, `cpu_serving` in
/// main.rs), not as an unknown value; only a candle-reference test build serves it.
pub const ATTEST_VALUES: &[&str] = &["cpu+gpu", "cpu"];

/// The session the relay asks of one TD; the TD serves under it and binds
/// the SHA-256 of its exact bytes into every quote. Nothing in it is secret.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudConfig {
    pub schema: String,
    /// A catalog name (`qwen3:0.6b`): the TD pulls it, the client checks it.
    pub model: String,
    /// SHA-256 of the client's TLS SubjectPublicKeyInfo, hex: the one client
    /// certificate the TD accepts, on every route.
    pub client_spki_sha256: String,
    /// 32 random bytes, hex. Freshness: the quote is made after this exists.
    pub client_nonce: String,
    pub expires_at_unix: u64,
    /// `cpu+gpu` or `cpu`.
    pub attest: String,
    pub listen_port: u16,
    pub max_completion_tokens: usize,
    /// Absent: the catalog model, public weights. Present: the model owner's
    /// sealed weights for that base model (`model_key.rs`); the TD loads
    /// nothing until the owner's relay releases the key to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sealed: Option<SealedWeights>,
    /// The cadence of a caller-sealed stream (docs/sdk-e2e.md, step 6).
    /// Absent: [`E2eStream::DEFAULT`]. Bound into every quote with the rest:
    /// the caller reads it before sealing, the host knows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e2e_stream: Option<E2eStream>,
}

/// What every caller-sealed stream of the session looks like on the wire,
/// whatever its text: a chunk of `chunk_bytes` bytes of plaintext every
/// `tick_ms`, the first `tick_ms` after the request, then one last chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct E2eStream {
    /// Plaintext bytes of every chunk but the last (L).
    pub chunk_bytes: u32,
    /// Milliseconds from one chunk to the next (T).
    pub tick_ms: u32,
}

impl E2eStream {
    /// L = 256: one chunk holds an SSE event of about 100 bytes of text, at
    /// 2.5 kB/s of stream for T = 100 ms; T = 100 ms: the tick of `/v1`.
    pub const DEFAULT: Self = Self {
        chunk_bytes: 256,
        tick_ms: 100,
    };
    /// Under 128 a chunk cannot hold the response's head; over 4096 the
    /// queue and the receipt no longer fit the last chunk together.
    pub const CHUNK_BYTES: std::ops::RangeInclusive<u32> = 128..=4096;
    pub const TICK_MS: std::ops::RangeInclusive<u32> = 20..=1000;

    /// Bounds a parser cannot judge.
    pub fn check(&self) -> Result<(), String> {
        if !Self::CHUNK_BYTES.contains(&self.chunk_bytes) {
            return Err(format!(
                "e2e_stream.chunk_bytes {} outside {:?}",
                self.chunk_bytes,
                Self::CHUNK_BYTES
            ));
        }
        if !Self::TICK_MS.contains(&self.tick_ms) {
            return Err(format!(
                "e2e_stream.tick_ms {} outside {:?}",
                self.tick_ms,
                Self::TICK_MS
            ));
        }
        Ok(())
    }
}

/// The sealed file the TD holds and the digest it must serve once unsealed.
/// Neither is secret; both are hashed into every quote with the config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedWeights {
    /// SHA-256 of the sealed file (`x0ne-infer seal` prints it).
    pub sealed_sha256: String,
    /// The served-model digest (v2) of the base model's config.json and
    /// tokenizer.json with the unsealed weights: what every receipt carries.
    pub digest_hex: String,
    /// Where the TD fetches the sealed file when it is not given one
    /// (`--sealed-weights`). Absent: the file must be on the VM already.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SealedSource>,
}

/// The sealed file's URL and size, for the TD to fetch it after its
/// attestation, as it pulls the catalog model: https to a host that is no
/// literal non-public address nor metadata name (DNS is not checked), or
/// plain http towards the loopback (a local bench); see
/// `registry::check_source_url`. The bytes are ciphertext; the host sees the
/// URL and the size, as it sees the file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedSource {
    pub url: String,
    pub size: u64,
}
