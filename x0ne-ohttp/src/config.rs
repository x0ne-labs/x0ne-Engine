//! Key configurations a client may encapsulate to.
//!
//! A client never trusts a key configuration because a server handed it over: it
//! takes the one whose SHA-256 it was given beforehand (a [`KeyPin`]), out of a
//! source every client shares (a transparency log, a node set, a binary). Handing
//! one user a key of their own is how an operator would single them out
//! (RFC 9458 section 7; Trail of Bits, TOB-WAPI-3, on Meta's Private Processing).

use sha2::{Digest, Sha256};

/// KEMs this client encapsulates to: DHKEM(X25519, HKDF-SHA256) and X-Wing
/// (ML-KEM-768 + X25519). Fewer accepted configurations, fewer ways to split users
/// into groups.
const KEM_X25519: u16 = 0x0020;
const KEM_XWING: u16 = 0x647a;
/// The one symmetric suite, HKDF-SHA256 with AES-128-GCM: it is written in the
/// clear in front of every request, so a second one would split users in two.
const KDF_HKDF_SHA256: u16 = 0x0001;
const AEAD_AES128GCM: u16 = 0x0001;

/// SHA-256 of one encoded key configuration (RFC 9458 section 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeyPin([u8; 32]);

impl KeyPin {
    pub fn of(encoded_config: &[u8]) -> Self {
        Self(Sha256::digest(encoded_config).into())
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// 64 hexadecimal digits, either case.
    pub fn from_hex(s: &str) -> Result<Self, ConfigError> {
        let s = s.as_bytes();
        if s.len() != 64 {
            return Err(ConfigError::Malformed);
        }
        let mut out = [0u8; 32];
        for (i, pair) in s.as_chunks::<2>().0.iter().enumerate() {
            out[i] = (nibble(pair[0])? << 4) | nibble(pair[1])?;
        }
        Ok(Self(out))
    }

    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

fn nibble(c: u8) -> Result<u8, ConfigError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(ConfigError::Malformed),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// Bad encoding: the whole collection is discarded (RFC 9458 section 3.2).
    #[error("malformed key configuration")]
    Malformed,
    /// A client takes one configuration at a time, never a choice among several.
    #[error("expected exactly one key configuration, got {0}")]
    NotExactlyOne(usize),
    /// Not the configuration this client was told to expect.
    #[error("key configuration does not match the pin")]
    Mismatch,
    /// Well formed, but no KEM or suite this client uses.
    #[error("unsupported key configuration")]
    Unsupported,
}

/// A key configuration that matched its pin and that this client can use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinnedConfig {
    encoded: Vec<u8>,
    key_id: u8,
    kem: u16,
    public_key: Vec<u8>,
}

impl PinnedConfig {
    /// `body` is an `application/ohttp-keys` collection. It must hold exactly one
    /// configuration, and that configuration must match `pin`.
    pub fn from_keys_media(body: &[u8], pin: &KeyPin) -> Result<Self, ConfigError> {
        let configs = split_keys_media(body)?;
        if configs.len() != 1 {
            return Err(ConfigError::NotExactlyOne(configs.len()));
        }
        Self::from_config(configs[0], pin)
    }

    /// `encoded` is one key configuration; it must match `pin`.
    pub fn from_config(encoded: &[u8], pin: &KeyPin) -> Result<Self, ConfigError> {
        if KeyPin::of(encoded) != *pin {
            return Err(ConfigError::Mismatch);
        }
        let (key_id, kem, public_key) = parse(encoded)?;
        // The crate's own decoder checks the public key itself.
        match ohttp::KeyConfig::decode(encoded) {
            Ok(_) => {}
            Err(ohttp::Error::Unsupported) => return Err(ConfigError::Unsupported),
            Err(_) => return Err(ConfigError::Malformed),
        }
        Ok(Self {
            encoded: encoded.to_vec(),
            key_id,
            kem,
            public_key: public_key.to_vec(),
        })
    }

    pub fn key_id(&self) -> u8 {
        self.key_id
    }

    pub fn kem_id(&self) -> u16 {
        self.kem
    }

    pub fn public_key(&self) -> &[u8] {
        &self.public_key
    }

    pub fn encoded(&self) -> &[u8] {
        &self.encoded
    }

    pub fn pin(&self) -> KeyPin {
        KeyPin::of(&self.encoded)
    }

    /// The configuration the client encapsulates to: this key with HKDF-SHA256 and
    /// AES-128-GCM alone. The `ohttp` crate takes the first suite listed, so the
    /// published order would otherwise choose it.
    pub(crate) fn encapsulation_config(&self) -> Vec<u8> {
        let mut out = vec![self.key_id];
        out.extend_from_slice(&self.kem.to_be_bytes());
        out.extend_from_slice(&self.public_key);
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(&KDF_HKDF_SHA256.to_be_bytes());
        out.extend_from_slice(&AEAD_AES128GCM.to_be_bytes());
        out
    }
}

/// Splits an `application/ohttp-keys` body into its configurations. Any
/// truncation or trailing byte discards the whole collection.
pub fn split_keys_media(body: &[u8]) -> Result<Vec<&[u8]>, ConfigError> {
    let mut out = Vec::new();
    let mut rest = body;
    while !rest.is_empty() {
        if rest.len() < 2 {
            return Err(ConfigError::Malformed);
        }
        let len = usize::from(u16::from_be_bytes([rest[0], rest[1]]));
        rest = &rest[2..];
        if len == 0 || rest.len() < len {
            return Err(ConfigError::Malformed);
        }
        out.push(&rest[..len]);
        rest = &rest[len..];
    }
    Ok(out)
}

/// Joins configurations into an `application/ohttp-keys` body.
pub fn join_keys_media<C: AsRef<[u8]>>(configs: &[C]) -> Result<Vec<u8>, ConfigError> {
    let mut out = Vec::new();
    for c in configs {
        let c = c.as_ref();
        let len = u16::try_from(c.len()).map_err(|_| ConfigError::Malformed)?;
        if len == 0 {
            return Err(ConfigError::Malformed);
        }
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(c);
    }
    Ok(out)
}

/// Strict parse of one key configuration: (key id, KEM id, public key).
fn parse(c: &[u8]) -> Result<(u8, u16, &[u8]), ConfigError> {
    if c.len() < 3 {
        return Err(ConfigError::Malformed);
    }
    let key_id = c[0];
    let kem = u16::from_be_bytes([c[1], c[2]]);
    let npk = match kem {
        KEM_X25519 => 32,
        KEM_XWING => 1216,
        _ => return Err(ConfigError::Unsupported),
    };
    let rest = &c[3..];
    if rest.len() < npk + 2 {
        return Err(ConfigError::Malformed);
    }
    let (pk, rest) = rest.split_at(npk);
    let sym_len = usize::from(u16::from_be_bytes([rest[0], rest[1]]));
    let suites = &rest[2..];
    if sym_len == 0 || sym_len % 4 != 0 || suites.len() != sym_len {
        return Err(ConfigError::Malformed);
    }
    let usable = suites.as_chunks::<4>().0.iter().any(|s| {
        let kdf = u16::from_be_bytes([s[0], s[1]]);
        let aead = u16::from_be_bytes([s[2], s[3]]);
        kdf == KDF_HKDF_SHA256 && aead == AEAD_AES128GCM
    });
    if !usable {
        return Err(ConfigError::Unsupported);
    }
    Ok((key_id, kem, pk))
}
