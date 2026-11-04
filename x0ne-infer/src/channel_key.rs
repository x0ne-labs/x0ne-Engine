//! In cloud mode the model key arrives over the attested channel,
//! after the client's verdict, not through an environment variable.
//!
//! A [`KeyBroker`](crate::keys::KeyBroker) over 32 bytes the client POSTed to `/v2/key`. `keys.rs`
//! is untouched: the release decision, the lease and the material handling
//! are the existing ones; only where the bytes come from changes.

use crate::harden::SecretBytes;
use crate::keys::KeyBroker;
use serde::Deserialize;
use zeroize::Zeroizing;

pub const KEY_ID: &str = "client-session-key";
pub const SCHEMA: &str = "x0ne-infer/key/v1";
pub const KEY_BYTES: usize = 32;

pub struct ClientKeyBroker {
    material: SecretBytes,
    id: &'static str,
}

impl ClientKeyBroker {
    /// Moves the key into locked memory (`harden::SecretBytes`); the
    /// caller's copy is zeroed on return.
    pub fn new(material: Zeroizing<Vec<u8>>) -> Result<Self, String> {
        Self::for_id(material, KEY_ID)
    }

    /// The same broker for another key id: the model owner's key
    /// (`model_key::KEY_ID`) goes through the same release decision.
    pub fn for_id(material: Zeroizing<Vec<u8>>, id: &'static str) -> Result<Self, String> {
        if material.len() != KEY_BYTES {
            return Err(format!(
                "session key is {} bytes, expected {KEY_BYTES}",
                material.len()
            ));
        }
        let material =
            SecretBytes::from_slice(&material).map_err(|e| format!("session key memory: {e}"))?;
        Ok(Self { material, id })
    }
}

impl KeyBroker for ClientKeyBroker {
    fn unwrap_key(&self, key_id: &str) -> anyhow::Result<Zeroizing<Vec<u8>>> {
        if key_id != self.id {
            anyhow::bail!("the channel broker holds {:?}, not {key_id:?}", self.id);
        }
        Ok(Zeroizing::new(self.material.expose().to_vec()))
    }
}

/// The body of `POST /v2/key`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyRequest {
    pub schema: String,
    pub key_hex: String,
}

impl KeyRequest {
    /// The client side: the JSON for 32 bytes.
    pub fn to_json(key: &[u8]) -> String {
        format!(
            "{{\"schema\":\"{SCHEMA}\",\"key_hex\":\"{}\"}}",
            crate::cloud_config::hex(key)
        )
    }

    /// The TD side: the 32 bytes, or why not.
    pub fn parse(body: &[u8]) -> Result<Zeroizing<Vec<u8>>, String> {
        if body.len() > 4096 {
            return Err("key request too large".into());
        }
        let req: KeyRequest =
            serde_json::from_slice(body).map_err(|e| format!("key request: {e}"))?;
        if req.schema != SCHEMA {
            return Err(format!(
                "key request schema {:?}, expected {SCHEMA:?}",
                req.schema
            ));
        }
        let bytes =
            crate::cloud_config::unhex32(&req.key_hex).map_err(|e| format!("key_hex: {e}"))?;
        let out = Zeroizing::new(bytes.to_vec());
        drop(Zeroizing::new(req.key_hex));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_request_round_trips_and_the_broker_answers_its_one_id() {
        let key = [0x42u8; 32];
        let body = KeyRequest::to_json(&key);
        let parsed = KeyRequest::parse(body.as_bytes()).unwrap();
        assert_eq!(&parsed[..], &key[..]);
        let broker = ClientKeyBroker::new(parsed).unwrap();
        #[cfg(target_os = "linux")]
        assert_ne!(broker.material.backing(), crate::harden::Backing::Heap);
        assert_eq!(&broker.unwrap_key(KEY_ID).unwrap()[..], &key[..]);
        assert!(broker.unwrap_key("model-key-0").is_err(), "another id");
    }

    #[test]
    fn malformed_requests_are_refused_by_name() {
        assert!(
            KeyRequest::parse(br#"{"schema":"x0ne-infer/key/v0","key_hex":""}"#)
                .unwrap_err()
                .contains("schema")
        );
        assert!(
            KeyRequest::parse(br#"{"schema":"x0ne-infer/key/v1","key_hex":"abc"}"#)
                .unwrap_err()
                .contains("key_hex")
        );
        assert!(
            KeyRequest::parse(br#"{"schema":"x0ne-infer/key/v1","key_hex":"","extra":1}"#).is_err()
        );
        assert!(ClientKeyBroker::new(Zeroizing::new(vec![0; 16])).is_err());
    }
}
