//! The client's session config for a cloud TD.
//!
//! The relay writes it, the lifecycle puts the exact bytes in the VM
//! metadata (`x0ne-config`), the image writes the exact bytes to a file,
//! `x0ne-infer --cloud-config` reads them. The quote binds
//! `SHA-256(bytes)`: no canonicalisation anywhere, the bytes are the
//! contract. Nothing in it is secret.

use sha2::{Digest, Sha256};
use std::path::Path;

// Moved to hex.rs (2026-10-24): x0ne-sdk compiles that file, not this one.
pub use crate::hex::{hex_array, hex_bytes};

// The types moved to cloud_config_types.rs (2026-10-30): x0ne-sdk compiles that file.
pub use crate::cloud_config_types::{
    CloudConfig, E2eStream, SealedSource, SealedWeights, ATTEST_VALUES, MAX_BYTES, SCHEMA,
};

/// Per-completion ceiling a config may ask for (the TD clamps requests to it).
pub const MAX_COMPLETION_TOKENS_CEILING: usize = 65_536;

/// The config as bytes, parsed, and hashed — the three forms every side uses.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub bytes: Vec<u8>,
    pub config: CloudConfig,
    pub sha256: [u8; 32],
}

impl CloudConfig {
    pub fn new(
        model: &str,
        client_spki_sha256: [u8; 32],
        client_nonce: [u8; 32],
        expires_at_unix: u64,
        attest: &str,
        listen_port: u16,
        max_completion_tokens: usize,
    ) -> Self {
        Self {
            schema: SCHEMA.into(),
            model: model.into(),
            client_spki_sha256: hex(&client_spki_sha256),
            client_nonce: hex(&client_nonce),
            expires_at_unix,
            attest: attest.into(),
            listen_port,
            max_completion_tokens,
            sealed: None,
            e2e_stream: None,
        }
    }

    /// The same session, for the owner's sealed weights.
    #[must_use]
    pub fn with_sealed(mut self, sealed: SealedWeights) -> Self {
        self.sealed = Some(sealed);
        self
    }

    pub fn is_sealed(&self) -> bool {
        self.sealed.is_some()
    }

    /// Everything a parser cannot judge: schema, hex shapes, enumerations,
    /// and that the model is one the embedded catalog knows.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!("schema {:?}, expected {SCHEMA:?}", self.schema));
        }
        crate::registry::Catalog::embedded()
            .get(&self.model)
            .map_err(|e| format!("model: {e}"))?;
        unhex32(&self.client_spki_sha256).map_err(|e| format!("client_spki_sha256: {e}"))?;
        unhex32(&self.client_nonce).map_err(|e| format!("client_nonce: {e}"))?;
        if self.expires_at_unix == 0 {
            return Err("expires_at_unix is 0".into());
        }
        if !ATTEST_VALUES.contains(&self.attest.as_str()) {
            return Err(format!(
                "attest {:?}, expected one of {ATTEST_VALUES:?}",
                self.attest
            ));
        }
        if self.listen_port == 0 {
            return Err("listen_port is 0".into());
        }
        if self.max_completion_tokens == 0
            || self.max_completion_tokens > MAX_COMPLETION_TOKENS_CEILING
        {
            return Err(format!(
                "max_completion_tokens {} outside 1..={MAX_COMPLETION_TOKENS_CEILING}",
                self.max_completion_tokens
            ));
        }
        if let Some(stream) = &self.e2e_stream {
            stream.check()?;
        }
        if let Some(s) = &self.sealed {
            unhex32(&s.sealed_sha256).map_err(|e| format!("sealed.sealed_sha256: {e}"))?;
            unhex32(&s.digest_hex).map_err(|e| format!("sealed.digest_hex: {e}"))?;
            if let Some(src) = &s.source {
                crate::registry::check_source_url(&src.url)
                    .map_err(|e| format!("sealed.source.url: {e}"))?;
                if src.size == 0 || src.size > crate::fsread::WEIGHTS_CAP {
                    return Err(format!(
                        "sealed.source.size {} outside 1..={}",
                        src.size,
                        crate::fsread::WEIGHTS_CAP
                    ));
                }
            }
        }
        Ok(())
    }

    /// The bytes that travel. Serialized once by the relay; every other side
    /// only ever hashes and compares what it received.
    pub fn to_loaded(&self) -> Result<Loaded, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        let sha256 = sha256(&bytes);
        Ok(Loaded {
            bytes,
            config: self.clone(),
            sha256,
        })
    }

    pub fn gpu_required(&self) -> bool {
        self.attest != "cpu"
    }

    /// Validated by [`CloudConfig::validate`]; panics only on an unvalidated value.
    pub fn client_spki(&self) -> [u8; 32] {
        unhex32(&self.client_spki_sha256).expect("validated")
    }
    pub fn nonce(&self) -> [u8; 32] {
        unhex32(&self.client_nonce).expect("validated")
    }
}

/// Parse and validate the exact bytes a peer sent.
pub fn parse(bytes: &[u8]) -> Result<Loaded, String> {
    if bytes.len() as u64 > MAX_BYTES {
        return Err(format!("config is {} bytes, cap {MAX_BYTES}", bytes.len()));
    }
    let config: CloudConfig =
        serde_json::from_slice(bytes).map_err(|e| format!("config is not a cloud config: {e}"))?;
    config.validate()?;
    Ok(Loaded {
        bytes: bytes.to_vec(),
        sha256: sha256(bytes),
        config,
    })
}

/// `--cloud-config FILE`: a regular file, capped, bytes kept verbatim.
pub fn load(path: &Path) -> Result<Loaded, String> {
    let bytes = crate::fsread::read_capped(path, MAX_BYTES)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// 64 lowercase hex characters, or a refusal naming what was wrong.
pub fn unhex32(s: &str) -> Result<[u8; 32], String> {
    unhex::<32>(s)
}

pub fn unhex<const N: usize>(s: &str) -> Result<[u8; N], String> {
    if s.len() != N * 2 {
        return Err(format!(
            "expected {} hex characters, got {}",
            N * 2,
            s.len()
        ));
    }
    if !s
        .bytes()
        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err("not lowercase hex".into());
    }
    let mut out = [0u8; N];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CloudConfig {
        CloudConfig::new(
            "qwen3:0.6b",
            [0xab; 32],
            [0xcd; 32],
            1_760_303_100,
            "cpu+gpu",
            443,
            4096,
        )
    }

    /// The serialized shape, byte for byte in field order: the quote binds
    /// the hash of these exact bytes.
    #[test]
    fn serializes_in_the_documented_field_order() {
        let l = sample().to_loaded().unwrap();
        let text = String::from_utf8(l.bytes.clone()).unwrap();
        assert!(text.starts_with(
            r#"{"schema":"x0ne-engine/cloud-config/v1","model":"qwen3:0.6b","client_spki_sha256":""#
        ));
        assert!(
            text.ends_with(r#""attest":"cpu+gpu","listen_port":443,"max_completion_tokens":4096}"#)
        );
        assert_eq!(l.sha256, sha256(&l.bytes));
        let back = parse(&l.bytes).unwrap();
        assert_eq!(back.config, l.config);
        assert_eq!(back.sha256, l.sha256);
    }

    /// SECURITY: the hash is over the bytes as sent. A re-serialization
    /// that changed one byte would change the hash, and must.
    #[test]
    fn the_hash_follows_the_bytes_not_the_meaning() {
        let l = sample().to_loaded().unwrap();
        let spaced = String::from_utf8(l.bytes.clone())
            .unwrap()
            .replace(',', ", ");
        let other = parse(spaced.as_bytes()).unwrap();
        assert_eq!(other.config, l.config, "same meaning");
        assert_ne!(other.sha256, l.sha256, "different bytes");
    }

    #[test]
    fn every_invalid_field_is_named() {
        let cases: Vec<(&str, Box<dyn Fn(&mut CloudConfig)>)> = vec![
            (
                "schema",
                Box::new(|c| c.schema = "x0ne-engine/cloud-config/v0".into()),
            ),
            ("model", Box::new(|c| c.model = "nope:1b".into())),
            (
                "client_spki_sha256",
                Box::new(|c| c.client_spki_sha256.truncate(10)),
            ),
            (
                "client_nonce",
                Box::new(|c| c.client_nonce = c.client_nonce.to_uppercase()),
            ),
            ("expires_at_unix", Box::new(|c| c.expires_at_unix = 0)),
            ("attest", Box::new(|c| c.attest = "gpu".into())),
            ("listen_port", Box::new(|c| c.listen_port = 0)),
            (
                "max_completion_tokens",
                Box::new(|c| c.max_completion_tokens = 0),
            ),
        ];
        for (field, mutate) in cases {
            let mut c = sample();
            mutate(&mut c);
            let err = c.validate().expect_err(field);
            assert!(err.contains(field), "{field}: {err}");
        }
        let extra = br#"{"schema":"x0ne-engine/cloud-config/v1","model":"qwen3:0.6b","client_spki_sha256":"00","client_nonce":"00","expires_at_unix":1,"attest":"cpu","listen_port":443,"max_completion_tokens":1,"debug":true}"#;
        assert!(parse(extra).unwrap_err().contains("unknown field"));
    }

    fn sealed() -> SealedWeights {
        SealedWeights {
            sealed_sha256: "33".repeat(32),
            digest_hex: "44".repeat(32),
            source: None,
        }
    }

    fn sourced(url: &str, size: u64) -> SealedWeights {
        SealedWeights {
            source: Some(SealedSource {
                url: url.into(),
                size,
            }),
            ..sealed()
        }
    }

    /// A sealed source travels with the config (so it is in the quote), as
    /// an https URL, or plain http towards the loopback only; its size is
    /// bounded like a weight file. A config without one is unchanged.
    #[test]
    fn a_sealed_source_is_carried_and_judged() {
        let l = sample()
            .with_sealed(sourced("https://w.example/m/model.sealed", 5))
            .to_loaded()
            .unwrap();
        let text = String::from_utf8(l.bytes.clone()).unwrap();
        assert!(
            text.ends_with(r#","source":{"url":"https://w.example/m/model.sealed","size":5}}}"#),
            "{text}"
        );
        assert_eq!(
            parse(&l.bytes).unwrap().config.sealed.unwrap().source,
            Some(SealedSource {
                url: "https://w.example/m/model.sealed".into(),
                size: 5
            })
        );
        assert!(sample()
            .with_sealed(sourced("http://127.0.0.1:1/model.sealed", 5))
            .validate()
            .is_ok());
        for (url, size, why) in [
            ("http://w.example/model.sealed", 5, "https"),
            ("http://127.0.0.1:1@w.example/model.sealed", 5, "userinfo"),
            ("https://u:p@w.example/model.sealed", 5, "userinfo"),
            ("https://169.254.169.254/model.sealed", 5, "link-local"),
            (
                "https://metadata.google.internal/model.sealed",
                5,
                "metadata",
            ),
            ("https://172.16.0.1/model.sealed", 5, "RFC 1918"),
            ("ftp://w.example/model.sealed", 5, "https"),
            ("not a url", 5, "https"),
            ("https://w.example/model.sealed", 0, "size"),
            (
                "https://w.example/model.sealed",
                crate::fsread::WEIGHTS_CAP + 1,
                "size",
            ),
        ] {
            let err = sample()
                .with_sealed(sourced(url, size))
                .validate()
                .expect_err(url);
            assert!(err.contains("sealed.source"), "{url}: {err}");
            assert!(err.contains(why), "{url}: {err}");
            assert!(!err.contains("u:p@"), "{url}: {err}");
        }
        let text = String::from_utf8(
            sample()
                .with_sealed(sourced("https://w.example/model.sealed", 5))
                .to_loaded()
                .unwrap()
                .bytes,
        )
        .unwrap()
        .replace(r#""size":5"#, r#""size":5,"token":"x""#);
        assert!(parse(text.as_bytes())
            .unwrap_err()
            .contains("unknown field"));
    }

    /// A config without sealed weights is byte for byte what it was before
    /// the field existed: every running session's hash stays the same.
    #[test]
    fn a_public_config_has_no_sealed_field_at_all() {
        let l = sample().to_loaded().unwrap();
        assert!(!String::from_utf8(l.bytes).unwrap().contains("sealed"));
    }

    /// A sealed config names the sealed file and the digest the TD must
    /// serve; both are hashed into the quote with the rest.
    #[test]
    fn a_sealed_config_round_trips_and_is_bound() {
        let public = sample().to_loaded().unwrap();
        let l = sample().with_sealed(sealed()).to_loaded().unwrap();
        let text = String::from_utf8(l.bytes.clone()).unwrap();
        assert!(text.ends_with(&format!(
            r#""max_completion_tokens":4096,"sealed":{{"sealed_sha256":"{}","digest_hex":"{}"}}}}"#,
            "33".repeat(32),
            "44".repeat(32)
        )));
        assert_ne!(l.sha256, public.sha256);
        assert_eq!(parse(&l.bytes).unwrap().config.sealed, Some(sealed()));
        assert!(l.config.is_sealed() && !public.config.is_sealed());
    }

    #[test]
    fn a_sealed_fields_bad_hex_or_unknown_field_is_named() {
        let mut c = sample().with_sealed(sealed());
        c.sealed.as_mut().unwrap().sealed_sha256.truncate(10);
        assert!(c.validate().unwrap_err().contains("sealed_sha256"));
        let mut c = sample().with_sealed(sealed());
        c.sealed.as_mut().unwrap().digest_hex = "ab".repeat(32).to_uppercase();
        assert!(c.validate().unwrap_err().contains("digest_hex"));
        let text = String::from_utf8(sample().with_sealed(sealed()).to_loaded().unwrap().bytes)
            .unwrap()
            .replace(r#""digest_hex""#, r#""key":"00","digest_hex""#);
        assert!(parse(text.as_bytes())
            .unwrap_err()
            .contains("unknown field"));
    }

    #[test]
    fn a_file_is_read_verbatim_and_capped() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        let l = sample().to_loaded().unwrap();
        std::fs::write(&p, &l.bytes).unwrap();
        assert_eq!(load(&p).unwrap().sha256, l.sha256);
        std::fs::write(&p, vec![b' '; MAX_BYTES as usize + 1]).unwrap();
        assert!(load(&p).is_err());
    }
    /// The cadence of caller-sealed streams travels in the config when the
    /// relay sets one, so the quote binds it, and is judged; absent, the
    /// bytes are those of a config written before it existed.
    #[test]
    fn an_e2e_stream_cadence_is_carried_and_judged() {
        let plain = sample().to_loaded().unwrap();
        assert!(!String::from_utf8_lossy(&plain.bytes).contains("e2e_stream"));
        let mut c = sample();
        c.e2e_stream = Some(E2eStream {
            chunk_bytes: 512,
            tick_ms: 50,
        });
        let l = c.to_loaded().unwrap();
        let text = String::from_utf8(l.bytes.clone()).unwrap();
        assert!(
            text.ends_with(r#""e2e_stream":{"chunk_bytes":512,"tick_ms":50}}"#),
            "{text}"
        );
        assert_eq!(parse(&l.bytes).unwrap().config, c);
        assert_ne!(l.sha256, plain.sha256);
        for (chunk_bytes, tick_ms) in [(127, 100), (4097, 100), (256, 19), (256, 1001), (0, 0)] {
            c.e2e_stream = Some(E2eStream {
                chunk_bytes,
                tick_ms,
            });
            assert!(c.validate().is_err(), "{chunk_bytes} {tick_ms}");
        }
        c.e2e_stream = Some(E2eStream::DEFAULT);
        assert!(c.validate().is_ok());
    }
}
