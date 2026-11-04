//! Frozen reference logits (refs/README.md). The manifest and the per-position
//! stats live in the tree; full logits live outside it. Every file is checked
//! against the sha256 recorded in the manifest before use.

use crate::engine::PosStats;
use crate::error::{format_err, Error, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub struct RefSet {
    small: PathBuf,
    big: PathBuf,
    manifest: Value,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub name: String,
    pub ids: Vec<u32>,
    pub stats: Vec<PosStats>,
    /// Positions whose full logits are stored.
    pub full_positions: Vec<usize>,
    pub vocab: usize,
    pub greedy_prompt_len: usize,
    pub greedy: Vec<u32>,
    full_file: PathBuf,
    full_sha256: String,
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

fn read_checked(path: &Path, want: &str) -> Result<Vec<u8>> {
    let bytes =
        std::fs::read(path).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
    let got = format!("{:x}", Sha256::digest(&bytes));
    if got != want {
        return format_err(format!(
            "{}: sha256 {got}, manifest says {want}",
            path.display()
        ));
    }
    Ok(bytes)
}

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| Error::Format(format!("manifest: missing {key}")))
}

fn str_of<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    field(v, key)?
        .as_str()
        .ok_or_else(|| Error::Format(format!("manifest: {key} is not a string")))
}

fn u64s(v: &Value, key: &str) -> Result<Vec<u64>> {
    field(v, key)?
        .as_array()
        .ok_or_else(|| Error::Format(format!("manifest: {key} is not an array")))?
        .iter()
        .map(|x| {
            x.as_u64()
                .ok_or_else(|| Error::Format(format!("manifest: {key} holds a non-integer")))
        })
        .collect()
}

/// Bytes per position in a stats file: u32 argmax, f32 max logit, f32 logsumexp, f32 next logprob.
pub const STATS_RECORD: usize = 16;

pub fn parse_stats(bytes: &[u8]) -> Result<Vec<PosStats>> {
    if !bytes.len().is_multiple_of(STATS_RECORD) {
        return format_err("stats file is not a whole number of records");
    }
    let f = |b: &[u8]| f32::from_le_bytes(b.try_into().unwrap());
    Ok(bytes
        .as_chunks::<STATS_RECORD>()
        .0
        .iter()
        .map(|r| PosStats {
            argmax: u32::from_le_bytes(r[0..4].try_into().unwrap()),
            logsumexp: f(&r[8..12]),
            next_logprob: f(&r[12..16]),
        })
        .collect())
}

impl RefSet {
    /// `small`: the tracked directory holding manifest.json and the stats files.
    /// `big`: the untracked directory holding the full logits.
    pub fn open(small: &Path, big: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(small.join("manifest.json")).map_err(|e| {
            Error::Format(format!("{}: {e}", small.join("manifest.json").display()))
        })?;
        let manifest =
            serde_json::from_str(&text).map_err(|e| Error::Format(format!("manifest: {e}")))?;
        Ok(RefSet {
            small: small.to_owned(),
            big: big.to_owned(),
            manifest,
        })
    }

    pub fn manifest(&self) -> &Value {
        &self.manifest
    }

    pub fn sequences(&self) -> Result<Vec<Sequence>> {
        let seqs = field(&self.manifest, "sequences")?
            .as_array()
            .ok_or_else(|| Error::Format("manifest: sequences is not an array".into()))?;
        let vocab = field(&self.manifest, "vocab")?
            .as_u64()
            .ok_or_else(|| Error::Format("manifest: vocab".into()))?;
        let mut out = Vec::with_capacity(seqs.len());
        for s in seqs {
            let name = str_of(s, "name")?.to_owned();
            let ids: Vec<u32> = u64s(s, "ids")?.into_iter().map(|x| x as u32).collect();
            let st = field(s, "stats")?;
            let stats = parse_stats(&read_checked(
                &self.small.join(str_of(st, "file")?),
                str_of(st, "sha256")?,
            )?)?;
            if stats.len() != ids.len() {
                return format_err(format!(
                    "{name}: {} stats records for {} ids",
                    stats.len(),
                    ids.len()
                ));
            }
            let full = field(s, "full_logits")?;
            let greedy = field(s, "greedy")?;
            out.push(Sequence {
                name,
                ids,
                stats,
                full_positions: u64s(full, "positions")?
                    .into_iter()
                    .map(|x| x as usize)
                    .collect(),
                vocab: vocab as usize,
                greedy_prompt_len: field(greedy, "prompt_len")?.as_u64().unwrap_or(0) as usize,
                greedy: u64s(greedy, "ids")?.into_iter().map(|x| x as u32).collect(),
                full_file: self.big.join(str_of(full, "file")?),
                full_sha256: str_of(full, "sha256")?.to_owned(),
            });
        }
        Ok(out)
    }
}

impl Sequence {
    /// The stored full logits, one row per `full_positions` entry, f32 little-endian.
    pub fn full_logits(&self) -> Result<Vec<Vec<f32>>> {
        let bytes = read_checked(&self.full_file, &self.full_sha256)?;
        let row = self.vocab * 4;
        if bytes.len() != row * self.full_positions.len() {
            return format_err(format!(
                "{}: {} bytes, expected {}",
                self.full_file.display(),
                bytes.len(),
                row * self.full_positions.len()
            ));
        }
        Ok(bytes
            .chunks_exact(row)
            .map(|r| {
                r.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| f32::from_le_bytes(b))
                    .collect()
            })
            .collect())
    }

    pub fn full_file_sha256(&self) -> Result<String> {
        sha256_file(&self.full_file)
    }
}
