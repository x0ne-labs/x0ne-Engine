//! Keys after proof — and never in the clear outside the enclave.
//!
//! Sealed weights are AES-256-GCM (`seal.rs`). The data key is released
//! through [`x0ne_confidential::model::KeyReleaseDecision::grant`] — the same
//! rule as the gate: fresh evidence, cryptographic verdict. The plaintext data
//! key lives in a [`Zeroizing`] buffer and is wiped on drop; there is no
//! accessor that returns it by value.
//!
//! No KEK wrap is implemented: the only production [`KeyBroker`]
//! ([`HeldKeyBroker`]) holds the plaintext key taken from
//! `X0NE_INFER_KEY_HEX` at start-up.

use x0ne_confidential::lease::CapabilityLease;
use x0ne_confidential::model::{
    AttestationEvidence, KeyReleaseDecision, KeyReleaseRequest, WorkloadIdentity,
};
use zeroize::Zeroizing;

use crate::gate::Attested;

/// A released model key: the plaintext data key, in memory that is zeroized
/// on drop, and the lease under which it was released (a release is an
/// authority over time, not a fact).
// No `Debug`, on purpose: the material must not be printable by accident.
pub struct ReleasedKey {
    key_id: String,
    to: WorkloadIdentity,
    lease: CapabilityLease,
    /// Monotonic instant of the release: the lease's anchor.
    released_at: std::time::Instant,
    material: Zeroizing<Vec<u8>>,
}

impl ReleasedKey {
    pub fn key_id(&self) -> &str {
        &self.key_id
    }
    pub fn released_to(&self) -> &WorkloadIdentity {
        &self.to
    }
    pub fn lease(&self) -> &CapabilityLease {
        &self.lease
    }
    pub fn released_at(&self) -> std::time::Instant {
        self.released_at
    }
    /// Borrow the material for a decrypt; it never leaves by value.
    pub fn with_material<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(&self.material)
    }
}

/// The v0 key variable. Taken out of the environment at start-up.
pub const KEY_ENV: &str = "X0NE_INFER_KEY_HEX";

/// Take `name` out of the environment: its value in zeroized memory, the
/// variable removed (no child, no later `getenv`) and zeroed in the exec-time
/// block (`/proc/<pid>/environ`, Linux). The launcher's environment keeps it.
///
/// # Safety
///
/// The contract of [`std::env::remove_var`], unsafe in edition 2024: no
/// other thread may read or write the environment meanwhile. `main` calls
/// this before any thread exists.
///
/// ```
/// // SAFETY: a doctest's main thread, before any other thread exists.
/// assert!(unsafe { x0ne_infer::keys::take_env_key("X0NE_DOC_UNSET_KEY") }.is_none());
/// ```
///
/// The same call without `unsafe` does not compile:
///
/// ```compile_fail
/// assert!(x0ne_infer::keys::take_env_key("X0NE_DOC_UNSET_KEY").is_none());
/// ```
pub unsafe fn take_env_key(name: &str) -> Option<Zeroizing<String>> {
    let raw = std::env::var_os(name)?;
    // SAFETY: the caller upholds the contract above.
    unsafe { wipe_exec_value(name) };
    // SAFETY: the caller upholds the contract above.
    unsafe { std::env::remove_var(name) };
    let raw = match raw.into_string() {
        Ok(s) => s,
        Err(os) => {
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStringExt;
                drop(Zeroizing::new(os.into_vec()));
            }
            #[cfg(not(unix))]
            drop(os);
            return Some(Zeroizing::new(String::new()));
        }
    };
    Some(Zeroizing::new(raw))
}

/// Zeroes each `name=` value `environ` points to: at start-up, the exec-time
/// block shown as `/proc/<pid>/environ`, where `remove_var` left the key (2026-10-22).
/// Safety: as [`take_env_key`], no other thread touches the environment.
#[cfg(target_os = "linux")]
unsafe fn wipe_exec_value(name: &str) {
    extern "C" {
        static mut environ: *const *mut libc::c_char;
    }
    let name = name.as_bytes();
    // SAFETY: a copy of the pointer, no reference to the static; nothing
    // changes it meanwhile (caller's contract).
    let mut entry = unsafe { environ };
    if entry.is_null() {
        return;
    }
    loop {
        // SAFETY: `environ` is an array of C strings ended by a null pointer.
        let s = unsafe { *entry };
        if s.is_null() {
            return;
        }
        // SAFETY: each entry is a NUL-terminated string.
        let len = unsafe { libc::strlen(s) };
        let is_name = {
            // SAFETY: `len` readable bytes; the slice ends before any write.
            let b = unsafe { std::slice::from_raw_parts(s.cast::<u8>(), len) };
            b.len() > name.len() && b.starts_with(name) && b[name.len()] == b'='
        };
        if is_name {
            for i in name.len() + 1..len {
                // SAFETY: inside the string; the exec-time block is writable
                // (the initial stack), as is a string a setenv allocated.
                unsafe { std::ptr::write_volatile(s.add(i), 0) };
            }
        }
        // SAFETY: not past the terminating null pointer, checked above.
        entry = unsafe { entry.add(1) };
    }
}

/// Elsewhere there is no `/proc/<pid>/environ` to clear.
#[cfg(not(target_os = "linux"))]
unsafe fn wipe_exec_value(_name: &str) {}

/// 64 hex characters into 32 bytes; nothing else is a key.
pub fn parse_hex_key(hex: &str) -> anyhow::Result<Zeroizing<Vec<u8>>> {
    if hex.len() != 64 || !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
        anyhow::bail!("{KEY_ENV} must be 64 hex chars");
    }
    let mut out = Zeroizing::new(vec![0u8; 32]);
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)?;
    }
    Ok(out)
}

/// v0 broker: the hex key taken from the environment at start-up
/// ([`take_env_key`]). The release decision still runs the real path.
pub struct HeldKeyBroker(pub Option<Zeroizing<String>>);

impl KeyBroker for HeldKeyBroker {
    fn unwrap_key(&self, key_id: &str) -> anyhow::Result<Zeroizing<Vec<u8>>> {
        match &self.0 {
            Some(hex) => parse_hex_key(hex),
            None => anyhow::bail!("{KEY_ENV} not set (key {key_id})"),
        }
    }
}

/// Who unwraps the key once the decision says `Released`. Production: the
/// client's key-broker (remote, verifies the same evidence on its side);
/// tests: an in-memory broker.
pub trait KeyBroker {
    /// Hand back the plaintext data key for `key_id`, or refuse.
    fn unwrap_key(&self, key_id: &str) -> anyhow::Result<Zeroizing<Vec<u8>>>;
}

/// Request the model key. Takes the [`Attested`] proof: without it this
/// function cannot be called, which is non-negotiable 3 of the spec ("one
/// rule, two gates"). Inside, the decision is still re-taken by
/// [`KeyReleaseDecision::grant`] on the SAME evidence and verdict — belt and
/// braces, and the lease it mints is what bounds the key in time.
pub fn release(
    proof: &Attested,
    evidence: &AttestationEvidence,
    key_id: &str,
    broker: &dyn KeyBroker,
    now_unix: u64,
) -> anyhow::Result<ReleasedKey> {
    let request = KeyReleaseRequest {
        key_id: key_id.to_string(),
        evidence: evidence.clone(),
    };
    match KeyReleaseDecision::grant(&request, proof.verdict(), now_unix) {
        KeyReleaseDecision::Released { key_id, to, lease } => {
            let released_at = std::time::Instant::now();
            let material = broker.unwrap_key(&key_id)?;
            Ok(ReleasedKey {
                key_id,
                to,
                lease,
                released_at,
                material,
            })
        }
        KeyReleaseDecision::Refused { key_id, reason } => {
            anyhow::bail!("key {key_id} refused: {reason}")
        }
    }
}

/// A released key for OTHER modules' tests (candle_backend's unseal path):
/// takes the whole production path — attest, then release — with a scripted
/// broker holding `material`. Test-only; production keys exist only through
/// `release` with a real proof.
#[cfg(test)]
pub(crate) fn test_released_key(material: Vec<u8>) -> ReleasedKey {
    use x0ne_confidential::model::{
        EvidenceFormat, ExecutionEnvironment, GpuCcMode, TeeTechnology, Verdict, VerifierStrength,
        WorkloadIdentity,
    };
    struct OneKey(Vec<u8>);
    impl KeyBroker for OneKey {
        fn unwrap_key(&self, _key_id: &str) -> anyhow::Result<Zeroizing<Vec<u8>>> {
            Ok(Zeroizing::new(self.0.clone()))
        }
    }
    let evidence = AttestationEvidence {
        format: EvidenceFormat::StructuredClaims,
        payload: b"{}".to_vec(),
        collected_at_unix: 1_000,
    };
    let verdict = Verdict::Verified {
        identity: WorkloadIdentity {
            name: "x0ne-infer".into(),
            environment: ExecutionEnvironment::ConfidentialVmWithGpu {
                cpu: TeeTechnology::IntelTdx,
                gpu_mode: GpuCcMode::On,
            },
            measurements: vec![],
        },
        verifier: "test".into(),
        strength: VerifierStrength::Cryptographic,
    };
    let proof = crate::gate::Gate::attest(
        ExecutionEnvironment::ConfidentialVmWithGpu {
            cpu: TeeTechnology::IntelTdx,
            gpu_mode: GpuCcMode::On,
        },
        &evidence,
        verdict,
        1_000,
    )
    .expect("test proof");
    release(&proof, &evidence, "test-key", &OneKey(material), 1_000).expect("test key")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::Gate;
    use x0ne_confidential::model::{
        EvidenceFormat, ExecutionEnvironment, GpuCcMode, TeeTechnology, Verdict, VerifierStrength,
    };

    struct MemBroker(Vec<u8>);
    impl KeyBroker for MemBroker {
        fn unwrap_key(&self, _key_id: &str) -> anyhow::Result<Zeroizing<Vec<u8>>> {
            Ok(Zeroizing::new(self.0.clone()))
        }
    }

    fn env() -> ExecutionEnvironment {
        ExecutionEnvironment::ConfidentialVmWithGpu {
            cpu: TeeTechnology::IntelTdx,
            gpu_mode: GpuCcMode::On,
        }
    }
    fn evidence(at: u64) -> AttestationEvidence {
        AttestationEvidence {
            format: EvidenceFormat::StructuredClaims,
            payload: b"{}".to_vec(),
            collected_at_unix: at,
        }
    }
    fn verified() -> Verdict {
        Verdict::Verified {
            identity: WorkloadIdentity {
                name: "x0ne-infer".into(),
                environment: env(),
                measurements: vec![],
            },
            verifier: "test".into(),
            strength: VerifierStrength::Cryptographic,
        }
    }

    /// With the proof, the key is released under a lease and the material is
    /// reachable only by borrow; the lease binds it to the measured identity.
    #[test]
    fn a_key_is_released_under_the_proof_and_only_by_borrow() {
        let proof = Gate::attest(env(), &evidence(1_000), verified(), 1_000).unwrap();
        let key = release(
            &proof,
            &evidence(1_000),
            "model-key-1",
            &MemBroker(vec![7u8; 32]),
            1_000,
        )
        .expect("released");
        assert_eq!(key.key_id(), "model-key-1");
        assert_eq!(key.released_to().name, "x0ne-infer");
        assert_eq!(key.with_material(|m| m.len()), 32);
        assert!(key.lease().is_live(1_000));
    }

    /// The same proof, stale evidence: refused — the decision is re-taken on
    /// the evidence, the proof alone does not carry the key.
    #[test]
    fn stale_evidence_refuses_the_key_even_with_a_proof() {
        let proof = Gate::attest(env(), &evidence(1_000), verified(), 1_000).unwrap();
        let err = match release(
            &proof,
            &evidence(1_000),
            "model-key-1",
            &MemBroker(vec![7u8; 32]),
            9_000,
        ) {
            Ok(_) => panic!("stale evidence must refuse the key"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("freshness"));
    }

    /// The header must not call AES-GCM unimplemented while `seal.rs`
    /// implements it.
    #[test]
    fn the_header_does_not_deny_the_cipher_seal_implements() {
        assert!(include_str!("seal.rs").contains("Aes256Gcm::new"));
        let header = include_str!("keys.rs").split("\nuse ").next().unwrap_or("");
        assert!(
            !header.contains("not implemented"),
            "keys.rs header is stale"
        );
    }

    #[test]
    fn a_hex_key_parses_to_32_bytes_and_nothing_else_does() {
        assert_eq!(*parse_hex_key(&"ab".repeat(32)).unwrap(), vec![0xab; 32]);
        for bad in [
            "",
            "ab",
            &"ab".repeat(33),
            &format!("{}zz", "ab".repeat(31)),
        ] {
            assert!(parse_hex_key(bad).is_err(), "{bad:?} accepted");
        }
        let none = HeldKeyBroker(None).unwrap_key("k").err().unwrap();
        assert!(none.to_string().contains(KEY_ENV), "{none}");
    }

    /// The key stayed in the environment for the life of the
    /// process. Child half of the test below; does nothing unless re-run.
    #[test]
    fn take_env_key_child_probe() {
        if std::env::var_os("X0NE_TAKE_KEY_PROBE").is_none() {
            return;
        }
        // SAFETY: this child runs this one test; its only other thread,
        // libtest's, is waiting for the result.
        let key = unsafe { take_env_key("X0NE_TAKE_KEY_PROBE_KEY") }.expect("key present at exec");
        let environ = std::fs::read("/proc/self/environ").unwrap_or_default();
        let child = std::process::Command::new("sh")
            .args(["-c", "printenv X0NE_TAKE_KEY_PROBE_KEY || true"])
            .output()
            .unwrap();
        println!(
            "PROBE taken={} var_gone={} child_sees={} proc_environ_has={} others_intact={}",
            key.as_str() == "cd".repeat(32),
            std::env::var_os("X0NE_TAKE_KEY_PROBE_KEY").is_none(),
            !child.stdout.is_empty(),
            environ.windows(64).any(|w| w == "cd".repeat(32).as_bytes()),
            environ
                .split(|b| *b == 0)
                .any(|e| e == b"X0NE_TAKE_KEY_PROBE=1")
                && std::env::var_os("X0NE_TAKE_KEY_PROBE").is_some(),
        );
    }

    /// SECURITY: a fresh process, so the key is in the exec-time block. Until
    /// 2026-10-22 `proc_environ_has` was true. A failure: the key is readable
    /// in `/proc/<pid>/environ` again, or a neighbouring variable was damaged.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_key_leaves_the_environment_children_and_proc_environ() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "keys::tests::take_env_key_child_probe",
                "--nocapture",
            ])
            .env("X0NE_TAKE_KEY_PROBE", "1")
            .env("X0NE_TAKE_KEY_PROBE_KEY", "cd".repeat(32))
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .find(|l| l.contains("PROBE "))
            .unwrap_or_else(|| panic!("no probe line: {stdout}"));
        assert!(
            line.ends_with(
                "taken=true var_gone=true child_sees=false proc_environ_has=false others_intact=true"
            ),
            "{line}"
        );
    }

    /// `main` must take the key while it is the only thread: a plain
    /// `fn`, the take before the tokio runtime and its workers.
    #[test]
    fn main_takes_the_key_before_it_builds_the_runtime() {
        let src = include_str!("main.rs");
        let start = src.find("\nfn main() {").expect("a plain fn main");
        let main = &src[start..];
        let main = &main[..main.find("\n}\n").expect("end of main")];
        let take = main.find("take_env_key(").expect("main takes the key");
        let built = main.find("Runtime::new()").expect("a runtime");
        assert!(take < built, "the key is taken after the runtime starts");
        assert!(!main[..take].contains("spawn("), "a thread starts first");
    }

    /// `_verdict_of` was a public, hidden alias of `Attested::verdict`
    /// whose announced test never existed. keys.rs exposes no verdict.
    #[test]
    fn keys_exposes_no_verdict_accessor() {
        let src = include_str!("keys.rs");
        let body = &src[..src.find("#[cfg(test)]").unwrap_or(src.len())];
        assert!(!body.contains("_verdict_of"), "keys.rs exposes _verdict_of");
        assert!(!body.contains("-> &Verdict"), "keys.rs returns a verdict");
    }
}
