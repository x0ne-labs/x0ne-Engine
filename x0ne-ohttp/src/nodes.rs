//! The node set: the key configurations of every TD serving an epoch, one list
//! for all clients. Its SHA-256 is what gets published (in the design, an entry
//! of a transparency log that clients check for inclusion), so a list made for
//! one user does not verify.
//!
//! The client draws its TD itself, uniformly, from its own generator. Each TD
//! holds its own key, so a request opens only on the TD the client drew: the
//! operator in front can forward it or drop it, not send it elsewhere.

use sha2::{Digest, Sha256};

use crate::config::{join_keys_media, split_keys_media, ConfigError, KeyPin, PinnedConfig};

const LABEL: &[u8] = b"x0ne-ohttp-node-set-v1\0";

/// SHA-256 of an encoded node set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SetDigest([u8; 32]);

impl SetDigest {
    pub fn of(encoded: &[u8]) -> Self {
        Self(Sha256::digest(encoded).into())
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// What a client requires of a set before drawing from it.
#[derive(Clone, Copy, Debug)]
pub struct NodePolicy {
    /// Below this, a compromised TD reads too large a share of the requests.
    pub min_nodes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NodeSetError {
    #[error("malformed node set")]
    Malformed,
    #[error("node set does not match the published digest")]
    Mismatch,
    #[error("node set too small: {got} nodes, {min} required")]
    TooFew { got: usize, min: usize },
    #[error("key id {0} listed twice")]
    DuplicateKeyId(u8),
    #[error("one public key listed twice")]
    DuplicateKey,
    #[error("node configuration: {0}")]
    Config(ConfigError),
    #[error("the system random generator failed")]
    Rng,
    #[error("node set of epoch {got}; the current epoch is {current}")]
    Epoch { got: u64, current: u64 },
}

#[derive(Clone, Debug)]
pub struct NodeSet {
    epoch: u64,
    nodes: Vec<PinnedConfig>,
}

impl NodeSet {
    /// `label || epoch (u64, big endian) || application/ohttp-keys`.
    pub fn encode<C: AsRef<[u8]>>(epoch: u64, configs: &[C]) -> Vec<u8> {
        let mut out = LABEL.to_vec();
        out.extend_from_slice(&epoch.to_be_bytes());
        // A configuration too long for the format leaves the set malformed, so
        // `verify` refuses it.
        match join_keys_media(configs) {
            Ok(keys) => out.extend_from_slice(&keys),
            Err(_) => out.truncate(LABEL.len()),
        }
        out
    }

    /// Accepts `encoded` only if it is the published set (`digest`), of the current
    /// epoch or the next, well formed, without a node listed twice, and large enough
    /// for `policy`. A digest proves a set was published once, not that it is still
    /// served: an old set handed to one user would single them out. Where
    /// `current_epoch` comes from (the log's latest entry, a clock tied to the epoch
    /// length) is the caller's to choose.
    pub fn verify(
        encoded: &[u8],
        digest: &SetDigest,
        current_epoch: u64,
        policy: &NodePolicy,
    ) -> Result<Self, NodeSetError> {
        if SetDigest::of(encoded) != *digest {
            return Err(NodeSetError::Mismatch);
        }
        let rest = encoded.strip_prefix(LABEL).ok_or(NodeSetError::Malformed)?;
        if rest.len() < 8 {
            return Err(NodeSetError::Malformed);
        }
        let (epoch, keys) = rest.split_at(8);
        let mut e = [0u8; 8];
        e.copy_from_slice(epoch);
        let epoch = u64::from_be_bytes(e);
        if epoch != current_epoch && current_epoch.checked_add(1) != Some(epoch) {
            return Err(NodeSetError::Epoch {
                got: epoch,
                current: current_epoch,
            });
        }
        let configs = split_keys_media(keys).map_err(|_| NodeSetError::Malformed)?;

        let mut nodes: Vec<PinnedConfig> = Vec::with_capacity(configs.len());
        for c in configs {
            let node =
                PinnedConfig::from_config(c, &KeyPin::of(c)).map_err(NodeSetError::Config)?;
            if nodes
                .iter()
                .any(|n| n.kem_id() == node.kem_id() && n.public_key() == node.public_key())
            {
                return Err(NodeSetError::DuplicateKey);
            }
            if nodes.iter().any(|n| n.key_id() == node.key_id()) {
                return Err(NodeSetError::DuplicateKeyId(node.key_id()));
            }
            nodes.push(node);
        }
        let min = policy.min_nodes.max(1);
        if nodes.len() < min {
            return Err(NodeSetError::TooFew {
                got: nodes.len(),
                min,
            });
        }
        Ok(Self { epoch, nodes })
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[PinnedConfig] {
        &self.nodes
    }

    /// One node, uniformly, from the operating system's generator. A caller keeps
    /// it for the whole session and does not draw again after a failure: a
    /// redraw lets whoever drops requests choose the node.
    pub fn choose(&self) -> Result<&PinnedConfig, NodeSetError> {
        let i = uniform_index(self.nodes.len(), getrandom::u32).map_err(|_| NodeSetError::Rng)?;
        self.nodes
            .get(i)
            .ok_or(NodeSetError::TooFew { got: 0, min: 1 })
    }
}

/// A uniform index below `n` from 32-bit draws, by rejection: draws in the top
/// `2^k mod n` values are discarded, so no index is favoured (k = 32, or 64 from
/// two draws when `n` exceeds 2^32). `n = 0` returns 0, which `NodeSet::choose`
/// then refuses.
pub fn uniform_index<E>(
    n: usize,
    mut next_u32: impl FnMut() -> Result<u32, E>,
) -> Result<usize, E> {
    let n = n as u128;
    if n <= 1 {
        return Ok(0);
    }
    let wide = n > 1 << 32;
    let range: u128 = if wide { 1 << 64 } else { 1 << 32 };
    let limit = range - range % n;
    loop {
        let mut v = u128::from(next_u32()?);
        if wide {
            v = (v << 32) | u128::from(next_u32()?);
        }
        if v < limit {
            return Ok((v % n) as usize);
        }
    }
}
