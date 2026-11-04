//! What the engine knows about a model's openness, looked up by a digest of
//! its files — never by a name, which anyone can write in a config. A
//! generative base is looked up by its served-model digest (v2,
//! `model_digest.rs`: config, tokenizer, every weight file); one weight
//! file by its sha256.
//!
//! Levels, from the Open Source Initiative's Open Source AI Definition 1.0
//! (2024-10-28, <https://opensource.org/ai/open-source-ai-definition>):
//!
//! - `fully-open`: weights, the complete training code, and the training data
//!   (or, per OSAID, enough information to rebuild an equivalent set) are
//!   published under open licences. This table only says `fully-open` when
//!   the DATA itself is published, which is stricter than OSAID and matches
//!   Class I of the Linux Foundation's Model Openness Framework.
//! - `open-weights`: the weights are under an open licence; data or training
//!   code are not published.
//! - `unknown`: not in this table. Not a judgement: the engine has not checked.
//!
//! Each entry was measured on the date written (download, sha256, model card).

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Level {
    FullyOpen,
    OpenWeights,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct Openness {
    pub level: Level,
    /// The key of a generative base: its served-model digest (v2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub served_digest: Option<String>,
    /// The weight file's sha256: the file a known entry names.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_license: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<&'static str>,
}

struct Known {
    sha256: &'static str,
    /// The served-model digest (v2) of the catalog entry with this weight
    /// file; None for a model the catalog does not serve (the encoder).
    served_digest: Option<&'static str>,
    level: Level,
    name: &'static str,
    weights_license: &'static str,
    data: &'static str,
    code: &'static str,
    source: &'static str,
}

/// `model.safetensors` as published. Measured 2026-09-25/26.
const KNOWN: &[Known] = &[
    Known {
        sha256: "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f",
        served_digest: Some("084ebfa73b958301de45a991f3f037bc5f50ac8e272d9813279b5c6641508d76"),
        level: Level::FullyOpen,
        name: "allenai/OLMo-2-0425-1B-Instruct",
        weights_license: "Apache-2.0",
        data: "published by Ai2: Dolma (pretraining), Tülu 3 SFT mix, \
               olmo-2-0425-1b-preference-mix (DPO), RLVR-MATH (RL)",
        code: "published by Ai2: OLMo-core (training), open-instruct (post-training), olmes (evaluation)",
        source: "https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct/tree/48d788eca847d4d7548f375ad03d3c9312f6139e",
    },
    Known {
        sha256: "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        served_digest: Some("b430888f3024072a1e7dd46e501bca046ed5fdd8e74397e3068a9a66d575d2a4"),
        level: Level::OpenWeights,
        name: "Qwen/Qwen3-0.6B",
        weights_license: "Apache-2.0",
        data: "not published",
        code: "inference code published; training code not published",
        source: "https://huggingface.co/Qwen/Qwen3-0.6B",
    },
];

impl Level {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullyOpen => "fully-open",
            Self::OpenWeights => "open-weights",
            Self::Unknown => "unknown",
        }
    }
}

fn found(
    k: Option<&Known>,
    served_digest: Option<String>,
    weights_sha256: Option<String>,
) -> Openness {
    match k {
        Some(k) => Openness {
            level: k.level,
            served_digest,
            weights_sha256: weights_sha256.or_else(|| Some(k.sha256.into())),
            name: Some(k.name),
            weights_license: Some(k.weights_license),
            data: Some(k.data),
            code: Some(k.code),
            source: Some(k.source),
        },
        None => Openness {
            level: Level::Unknown,
            served_digest,
            weights_sha256,
            name: None,
            weights_license: None,
            data: None,
            code: None,
            source: None,
        },
    }
}

/// By the sha256 of one weight file.
pub fn lookup(weights_sha256: &str) -> Openness {
    let k = KNOWN
        .iter()
        .find(|k| k.sha256.eq_ignore_ascii_case(weights_sha256));
    found(k, None, Some(weights_sha256.to_ascii_lowercase()))
}

/// By the served-model digest (v2): a generative base, one file or shards.
pub fn lookup_served(digest: &str) -> Openness {
    let k = KNOWN.iter().find(|k| {
        k.served_digest
            .is_some_and(|d| d.eq_ignore_ascii_case(digest))
    });
    found(k, Some(digest.to_ascii_lowercase()), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failure means a model is declared open because of its NAME, or an
    /// unlisted model is declared anything but unknown.
    #[test]
    fn openness_is_keyed_on_the_weights_digest_only() {
        let o = lookup("36D044C73655BB904F822915E6294BA3DAE8E6E1AF5E703E9D452F2D6A3A294F");
        assert_eq!(o.level, Level::FullyOpen);
        assert_eq!(o.name, Some("allenai/OLMo-2-0425-1B-Instruct"));
        let o = lookup(&"0".repeat(64));
        assert_eq!(o.level, Level::Unknown);
        assert!(o.name.is_none() && o.data.is_none());
        assert_eq!(
            serde_json::to_value(lookup("x")).unwrap()["level"],
            "unknown"
        );
        for k in KNOWN {
            assert_eq!(k.sha256.len(), 64);
            assert!(k.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }

    /// A generative entry's served digest is the catalog's for the entry
    /// whose model.safetensors is this sha256, and follows from the
    /// catalog's own file hashes. The weights sha256 alone finds nothing.
    #[test]
    fn a_generative_base_is_found_by_its_served_digest_only() {
        let catalog = crate::registry::Catalog::embedded();
        let mut served = 0;
        for k in KNOWN {
            let Some(d) = k.served_digest else { continue };
            let Some(entry) = catalog
                .models
                .iter()
                .find(|e| e.weight_files().any(|f| f.sha256 == k.sha256))
            else {
                // Out of the served catalog since 2026-11-04: Qwen3 alone is served.
                assert_eq!(k.name, "allenai/OLMo-2-0425-1B-Instruct");
                continue;
            };
            served += 1;
            assert_eq!(entry.digest_sha256.as_deref(), Some(d), "{}", entry.name);
            assert_eq!(entry.catalog_digest().unwrap(), d, "{}", entry.name);
            let o = lookup_served(d);
            assert_eq!((o.level, o.name), (k.level, Some(k.name)));
            assert_eq!(o.served_digest.as_deref(), Some(d));
            assert_eq!(lookup_served(k.sha256).level, Level::Unknown);
        }
        assert_eq!(served, 1, "Qwen3-0.6B");
        let o = lookup_served(&"0".repeat(64));
        assert_eq!(o.level, Level::Unknown);
        assert!(o.weights_sha256.is_none() && o.name.is_none());
    }
}
