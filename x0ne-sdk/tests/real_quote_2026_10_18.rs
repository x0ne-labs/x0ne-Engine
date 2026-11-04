//! `attest` on a real TD Quote v4: the GCE a3-highgpu-1g boot of 2026-10-18
//! (Intel TDX, one H100 in CC mode, x0ne-cvm 0.2.3, relay verdict VERIFIED,
//! `docs/hardware-runs.md`), with Intel PCS production collateral vendored in
//! `x0ne-infer/tests/pcs_live/` (`VENDORED.md`). Offline, at fixed clocks.
//!
//! The quote answers the relay's channel (binding v2, recomputed below from
//! the run's attestation document), not an OHTTP key: no key document carries
//! it past the `binding` step of `verify_key`. The steps after it run through
//! `verify_quote`, which `verify_key` calls, with that binding expected.
//!
//! Not kept from the run: Intel's TCB info and QE identity of 2026-10-18. The
//! `tcb` step runs on Intel's answers of 2026-10-22, at their own clock:
//! under `strict` (the relay's policy on the run's day) on `early` alone,
//! under `transparent` (its default since 2026-10-28) on both.
//!
//! The run's reference is admitted with its signature by the TEST archive
//! key (`x0ne-cvm-0.2.3.json.sig`) and a TEST log that shows it.

use base64::Engine as _;
use p256::ecdsa::signature::Signer;
use p256::ecdsa::{Signature, SigningKey};
use serde_json::json;
use sha2::{Digest, Sha256, Sha384, Sha512};
use x0ne_sdk::abi::{self, OP_VERIFY_KEY};
use x0ne_sdk::attest::{
    self, Collateral, KeyDocument, Policy, Refusal, SignedReference, KEY_SCHEMA,
};
use x0ne_sdk::e2e::Gateway;
use x0ne_sdk::TrustAnchor;

const QUOTE: &[u8] =
    include_bytes!("../../x0ne-infer/tests/run_fixtures/gce-a3-tdx-2026-10-18.quote");
/// `document.tdx.ccel_b64` of the same document, cut 256 bytes into its 0xFF
/// padding (`x0ne-infer/src/run_fixtures.rs`).
const CCEL: &[u8] =
    include_bytes!("../../x0ne-infer/tests/run_fixtures/gce-a3-tdx-2026-10-18.ccel");
/// The image reference of the run, x0ne-cvm 0.2.3, as the relay read it.
const REFERENCE: &[u8] = include_bytes!("../../x0ne-infer/tests/run_fixtures/x0ne-cvm-0.2.3.json");
/// The same reference with RTMR0 pinned (`rtmr0.policy` `gce-endorsed`) to
/// the run's own boot under Google's binary 53c558fe… (`docs/pinning.md`,
/// the procedure; `run_fixtures.rs` checks it differs in `rtmr0` alone),
/// unsigned: signed here by the TEST key. Since 2026-10-31 the SDK admits no
/// reference that reports RTMR0, the run's included.
const REFERENCE_PINNED: &[u8] =
    include_bytes!("../../x0ne-infer/tests/run_fixtures/x0ne-cvm-0.2.3-rtmr0-pinned.json");
/// SHA-384 of the firmware binary Google's endorsement of MRTD c1ee9c16…
/// names, and of its configuration volume, measured on the binary
/// (`x0ne-infer/tests/gce_firmware_live.rs`, 2026-10-22).
const FIRMWARE_SHA384: &str = "53c558fe06714203d7428484c46ee7c0b9df748a9789e173421c28d097115b30efb8476304617532488ab40c3122db88";
const CFV_SHA384: &str = "4e1221cf6f7815beec39a994bff99142ed305210e2735b1134fc8ef363923cbdcca91fa71e2e25ee8e689fb7d2c075bf";
/// Its detached signature by the TEST archive key, made 2026-10-18 12:33 UTC.
const REFERENCE_SIG: &[u8] =
    include_bytes!("../../x0ne-infer/tests/run_fixtures/x0ne-cvm-0.2.3.json.sig");
const MACHINE: &str = "a3-highgpu-1g";
/// Google's launch endorsement of the run's MRTD (`tests/VENDORED.md`).
const ENDORSEMENT: &[u8] =
    include_bytes!("../../x0ne-infer/tests/gce_endorsement_tdx_c1ee9c16.binarypb");
/// Google's launch endorsement of another firmware (MRTD 85aefc5d…).
const OTHER_ENDORSEMENT: &[u8] =
    include_bytes!("../../x0ne-infer/tests/gce_endorsement_tdx_85aefc5d.binarypb");
/// Intel's SGX root CA certificate: a genuine root, not Google's.
const INTEL_ROOT_DER: &[u8] = include_bytes!("../../x0ne-infer/tests/intel_sgx_root_ca.der");
/// `document.tdx.quote_b64` of the run's attestation document (SHA-256
/// 8b8f6c6c…, kept outside this repository), byte for byte.
const QUOTE_SHA256: &str = "bb95fe6130fc4ee51ed01329959d5e62a1fd76630fb65a66c58292691c321a5a";

// Intel PCS, 2026-10-22, FMSPC 00806f050000 (this platform), `update=early`.
const TCB: &str =
    include_str!("../../x0ne-infer/tests/pcs_live/2026-10-22/tcb_info_00806f050000_early.json");
const TCB_STANDARD: &str =
    include_str!("../../x0ne-infer/tests/pcs_live/2026-10-22/tcb_info_00806f050000_standard.json");
const QE: &str = include_str!("../../x0ne-infer/tests/pcs_live/2026-10-22/qe_identity_early.json");
const QE_STANDARD: &str =
    include_str!("../../x0ne-infer/tests/pcs_live/2026-10-22/qe_identity_standard.json");
const CRL: &[u8] =
    include_bytes!("../../x0ne-infer/tests/pcs_live/2026-10-22/pck_crl_platform.der");
const TCB_CHAIN: &str = include_str!("../../x0ne-infer/tests/pcs_live/tcb_issuer_chain.pem");
const QE_CHAIN: &str = include_str!("../../x0ne-infer/tests/pcs_live/qe_issuer_chain.pem");
// Another platform (FMSPC 90c06f000000): on 2026-10-22, and on 2026-08-20.
const TCB_OTHER_PLATFORM: &str =
    include_str!("../../x0ne-infer/tests/pcs_live/2026-10-22/tcb_info_90c06f000000_early.json");
const TCB_2026_08_20: &str = include_str!("../../x0ne-infer/tests/pcs_live/tcb_info.json");
const QE_2026_08_20: &str = include_str!("../../x0ne-infer/tests/pcs_live/qe_identity.json");
/// The PCK Platform CA's CRL of the run's day (19:08:58 UTC).
const CRL_2026_10_18: &[u8] =
    include_bytes!("../../x0ne-infer/tests/pcs_live/pck_crl_platform_2026-10-18.der");
/// Intel's root CA CRL: a genuine CRL, of another issuer than the PCK's.
const ROOT_CA_CRL: &[u8] =
    include_bytes!("../../x0ne-infer/tests/pcs_live/2026-10-23/root_ca_crl.der");

/// 2026-10-18T17:54:50Z, `document.server.attested_at_unix`.
const ATTESTED: u64 = 1_792_346_090;
/// 2026-10-22T10:55:40Z, the `Date` of the last of the six PCS answers.
const NOW: u64 = 1_792_666_540;
const DAY: u64 = 86_400;

// The run's binding v2 (`document.binding`, `document.config_b64`):
// SHA-512("x0ne-infer/v2" ‖ session nonce ‖ session key ‖ SHA-256(TD's TLS
// SPKI) ‖ SHA-256(config)), x0ne-infer/src/binding.rs.
const SESSION_NONCE: &str = "0e4309d4f93f2f8b74bdc1bd4485ef5b37d83778aabad6e08e6d2caa56d0ddd5";
const SESSION_PUBKEY: &str = "96dbaeb08eca9927f7816c66ceaa0c0047fdbeced5b87b7098273f85f8fcf58b";
const TLS_SPKI_SHA256: &str = "10220d5ab17f39e4a3782aa2c44c986799886aec5750c0c310e6a831fdfe4538";
const CONFIG: &str = r#"{"schema":"x0ne-engine/cloud-config/v1","model":"qwen3:32b","client_spki_sha256":"55cd175f360c4ffc1a784d79cf991b95c6f03b637c9ee08ff4102a7116e0c9a6","client_nonce":"04025646154239a270373a4a6cfac613fd918844793a0824cfab93326a4f077e","expires_at_unix":1792348209,"attest":"cpu+gpu","listen_port":443,"max_completion_tokens":4096}"#;
const CONFIG_SHA256: &str = "2f4cd9f2d14c4098b27bd4d31633524b0c48e862ed28bb0cbd0cd01e2020f191";

/// Google's launch endorsement of this firmware (`gce_endorsement_tdx_c1ee9c16.binarypb`).
const MRTD: &str = "c1ee9c16e3afc506cfe042c5b846a368528f3b37618eafb27469bc114cf914e9222c91618470e7f2b28ac360968270a5";
/// The image build's prediction (`measurements/0.2.0-beta.7.json` of 2026-10-18).
const RTMR1: &str = "758a634e6bf5d06ac59414cb2285616757e64a6ef4bc1b636bd0e8854ed33a80c8721e223089898234e412584c62d5c7";
const RTMR2: &str = "8b1d6ce2e48f55ac320bac37b1376a79bbe8b5d76950d968b7bd94a03d2e3e3a2854f158847cca4501fe40b660ddc886";
/// Reported, not pinned; the relay's verdict of the run names the same.
const RTMR0: &str = "10944cd1f86a98bb39718a85718e6cf84855b38afce35cc84edff0c660998d0500dfedf998af8484e423949c23c40011";
/// The relay's verdict of the run, its `image`.
const IMAGE: &str = "x0ne-cvm 0.2.3";
/// The relay's verdict of the run, register `pck_sha256`.
const PCK_SHA256: &str = "a044fd633817e2fd29dafc64f55c4326e22e03b654084c5055071ac3014e5545";

// TD Quote v4: header 48, TD report body 584 (signed), then the signature section.
const SIGNED: usize = 632;
const TD_ATTRIBUTES: usize = 48 + 120;
const XFAM: usize = 48 + 128;
const MR_TD: usize = 48 + 136;
const MR_CONFIG_ID: usize = 48 + 184;
const RTMR_AT: usize = 48 + 328;
const REPORT_DATA: usize = 48 + 520;
const QUOTE_SIGNATURE: usize = SIGNED + 4;
const ATTESTATION_KEY: usize = QUOTE_SIGNATURE + 64;
const QE_REPORT: usize = ATTESTATION_KEY + 64 + 6;
const QE_REPORT_SIGNATURE: usize = QE_REPORT + 384;
const QE_AUTH_DATA: usize = QE_REPORT_SIGNATURE + 64 + 2;

fn unhex<const N: usize>(s: &str) -> [u8; N] {
    let v: Vec<u8> = (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect();
    v.try_into().unwrap()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn b64(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

fn run_report_data() -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(b"x0ne-infer/v2");
    h.update(unhex::<32>(SESSION_NONCE));
    h.update(unhex::<32>(SESSION_PUBKEY));
    h.update(unhex::<32>(TLS_SPKI_SHA256));
    h.update(Sha256::digest(CONFIG.as_bytes()));
    h.finalize().into()
}

fn collateral() -> Collateral {
    Collateral {
        tcb_info: TCB.into(),
        tcb_info_chain_pem: TCB_CHAIN.into(),
        qe_identity: QE.into(),
        qe_identity_chain_pem: QE_CHAIN.into(),
        pck_crl_der_b64: b64(CRL),
        root_ca_crl_der_b64: b64(ROOT_CA_CRL),
        gce_endorsement_b64: b64(ENDORSEMENT),
        ..Collateral::default()
    }
}

/// [`collateral`] and Intel's `standard` answers of the same day: what
/// `transparent` accepts on.
fn collateral_both() -> Collateral {
    Collateral {
        tcb_info_standard: TCB_STANDARD.into(),
        tcb_info_standard_chain_pem: TCB_CHAIN.into(),
        qe_identity_standard: QE_STANDARD.into(),
        qe_identity_standard_chain_pem: QE_CHAIN.into(),
        ..collateral()
    }
}

/// A TEST log that shows the run's reference, cosigned at [`NOW`].
fn tlog_proof() -> String {
    attest::sign_reference_under_test_keys(REFERENCE, NOW).1
}

/// The run's reference with RTMR0 pinned ([`REFERENCE_PINNED`]), signed by
/// the TEST archive key and logged, both at [`NOW`]. The run's own, which
/// reports RTMR0, is refused (104): `the_runs_reference_…` below.
fn reference() -> attest::AdmittedReference {
    admitted(REFERENCE_PINNED)
}

/// `json` signed by the TEST archive key and logged at [`NOW`], admitted.
fn admitted(json: &[u8]) -> attest::AdmittedReference {
    let (signature, tlog_proof) = attest::sign_reference_under_test_keys(json, NOW);
    attest::admit_reference_under_test_keys(
        &SignedReference {
            json,
            signature: &signature,
            tlog_proof: &tlog_proof,
        },
        &mut attest::LogMemory::new(),
        None,
        NOW,
    )
    .expect("signed and logged")
}

/// Google's binary as `tdvf::verify_endorsed` measured it, for the replay
/// without the 2 MiB binary (not vendored): `test-anchor`'s
/// [`Policy::firmware_measured_elsewhere`]. The binary itself:
/// `the_real_quote_passes_on_googles_binary_relayed_and_measured_here`.
fn measured_firmware() -> attest::FirmwareMeasuredElsewhere {
    attest::FirmwareMeasuredElsewhere {
        sha384_hex: FIRMWARE_SHA384.into(),
        cfv_sha384: unhex(CFV_SHA384),
    }
}

/// The run's reference under `strict`, the registers the run measured
/// pinned on top.
fn policy() -> Policy {
    production().pinned(unhex(MRTD), [unhex(RTMR1), unhex(RTMR2), [0; 48]])
}

fn verify(
    quote: &[u8],
    col: &Collateral,
    p: &Policy,
    now: u64,
) -> Result<attest::VerifiedQuote, Refusal> {
    attest::verify_quote(quote, CCEL, &run_report_data(), col, p, now)
}

/// A key document as a TD would answer it, around the run's real quote: an
/// OHTTP key configuration and, as receipt key, the run's session key; the
/// run's session config and boot binding, which its quote binds under the
/// relay's context, not the e2e one.
fn key_document(quote: &[u8]) -> KeyDocument {
    KeyDocument {
        schema: KEY_SCHEMA.into(),
        key_config_b64: b64(&Gateway::from_seed(&[7; 32]).key_config()),
        session_pubkey_hex: SESSION_PUBKEY.into(),
        session_config_b64: b64(CONFIG.as_bytes()),
        boot_nonce_hex: SESSION_NONCE.into(),
        tls_spki_sha256_hex: TLS_SPKI_SHA256.into(),
        quote_b64: b64(quote),
        ccel_b64: b64(CCEL),
        gpu_evidence_list: None,
        nras_answer: None,
        nras_jwks: None,
        nras_ocsp: None,
    }
}

#[track_caller]
fn refused<T: std::fmt::Debug>(r: Result<T, Refusal>, step: &str, says: &str) {
    let e = r.expect_err("accepted");
    assert_eq!(e.step, step, "{e}");
    assert!(e.reason.contains(says), "{e}");
}

fn flipped(quote: &[u8], at: usize, bit: u8) -> Vec<u8> {
    let mut q = quote.to_vec();
    q[at] ^= bit;
    q
}

/// The signature section of a TD Quote v4, as `pcs.rs` reads it.
#[derive(Clone)]
struct SigSection {
    quote_signature: Vec<u8>,
    attestation_key: Vec<u8>,
    qe_report: Vec<u8>,
    qe_report_signature: Vec<u8>,
    qe_auth_data: Vec<u8>,
    pck_chain_pem: Vec<u8>,
}

fn le(b: &[u8]) -> usize {
    b.iter().rev().fold(0, |n, &x| (n << 8) | x as usize)
}

fn split(quote: &[u8]) -> SigSection {
    let mut at = QUOTE_SIGNATURE;
    let mut take = |n: usize| {
        at += n;
        quote[at - n..at].to_vec()
    };
    let quote_signature = take(64);
    let attestation_key = take(64);
    assert_eq!(le(&take(2)), 6, "QE report certification data");
    take(4);
    let qe_report = take(384);
    let qe_report_signature = take(64);
    let auth = le(&take(2));
    let qe_auth_data = take(auth);
    assert_eq!(le(&take(2)), 5, "PCK chain PEM");
    let pem = le(&take(4));
    SigSection {
        quote_signature,
        attestation_key,
        qe_report,
        qe_report_signature,
        qe_auth_data,
        pck_chain_pem: take(pem),
    }
}

fn join(signed: &[u8], s: &SigSection) -> Vec<u8> {
    let cert_data = 384 + 64 + 2 + s.qe_auth_data.len() + 2 + 4 + s.pck_chain_pem.len();
    let mut q = signed[..SIGNED].to_vec();
    q.extend((64 + 64 + 2 + 4 + cert_data as u32).to_le_bytes());
    q.extend(&s.quote_signature);
    q.extend(&s.attestation_key);
    q.extend(6u16.to_le_bytes());
    q.extend((cert_data as u32).to_le_bytes());
    q.extend(&s.qe_report);
    q.extend(&s.qe_report_signature);
    q.extend((s.qe_auth_data.len() as u16).to_le_bytes());
    q.extend(&s.qe_auth_data);
    q.extend(5u16.to_le_bytes());
    q.extend((s.pck_chain_pem.len() as u32).to_le_bytes());
    q.extend(&s.pck_chain_pem);
    q
}

/// The PEM blocks of a chain, leaf first.
fn certificates(pem: &[u8]) -> Vec<String> {
    std::str::from_utf8(pem)
        .unwrap()
        .split_inclusive("-----END CERTIFICATE-----")
        .filter(|b| b.contains("-----BEGIN CERTIFICATE-----"))
        .map(|b| format!("{}\n", b.trim_start()))
        .collect()
}

fn with_chain(quote: &[u8], certs: &[&str]) -> Vec<u8> {
    let mut s = split(quote);
    s.pck_chain_pem = certs.concat().into_bytes();
    join(quote, &s)
}

// A chain of our own, root, CA and PCK as Intel's (`pcs.rs` takes no other
// shape), P-256, DER written by hand: the real TD report re-signed under it
// reaches the checks Intel's signature shields.

fn der(tag: u8, content: &[u8]) -> Vec<u8> {
    let n = content.len();
    let mut out = vec![tag];
    match n {
        0..=0x7f => out.push(n as u8),
        0x80..=0xff => out.extend([0x81, n as u8]),
        _ => out.extend([0x82, (n >> 8) as u8, n as u8]),
    }
    out.extend_from_slice(content);
    out
}

fn seq(parts: &[&[u8]]) -> Vec<u8> {
    der(0x30, &parts.concat())
}

fn uint(be: &[u8]) -> Vec<u8> {
    let first = be.iter().position(|&b| b != 0).unwrap_or(be.len() - 1);
    let mut v = be[first..].to_vec();
    if v[0] & 0x80 != 0 {
        v.insert(0, 0);
    }
    der(0x02, &v)
}

const ECDSA_WITH_SHA256: &[u8] = &[0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02];
const EC_PUBLIC_KEY: &[u8] = &[0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
const PRIME256V1: &[u8] = &[0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
const COMMON_NAME: &[u8] = &[0x06, 0x03, 0x55, 0x04, 0x03];
const BASIC_CONSTRAINTS: &[u8] = &[0x06, 0x03, 0x55, 0x1d, 0x13];
const TRUE: &[u8] = &[0x01, 0x01, 0xff];

fn name(cn: &str) -> Vec<u8> {
    seq(&[&der(0x31, &seq(&[COMMON_NAME, &der(0x0c, cn.as_bytes())]))])
}

fn xy(key: &SigningKey) -> Vec<u8> {
    key.verifying_key().to_sec1_point(false).as_bytes()[1..].to_vec()
}

/// `ca`: a critical basicConstraints cA TRUE; otherwise no extension.
fn certificate(
    subject: &str,
    key: &SigningKey,
    issuer: &str,
    by: &SigningKey,
    serial: u8,
    ca: bool,
) -> Vec<u8> {
    let alg = seq(&[ECDSA_WITH_SHA256]);
    let mut point = vec![0, 4];
    point.extend(xy(key));
    let spki = seq(&[&seq(&[EC_PUBLIC_KEY, PRIME256V1]), &der(0x03, &point)]);
    let validity = seq(&[&der(0x17, b"200101000000Z"), &der(0x17, b"491231235959Z")]);
    let mut fields = vec![
        der(0xa0, &der(0x02, &[2])),
        der(0x02, &[serial]),
        alg.clone(),
        name(issuer),
        validity,
        name(subject),
        spki,
    ];
    if ca {
        let extension = seq(&[BASIC_CONSTRAINTS, TRUE, &der(0x04, &seq(&[TRUE]))]);
        fields.push(der(0xa3, &seq(&[&extension])));
    }
    let tbs = der(0x30, &fields.concat());
    let sig: Signature = by.sign(&tbs);
    let (r, s) = sig.split_bytes();
    let mut bits = vec![0];
    bits.extend(seq(&[&uint(&r), &uint(&s)]));
    seq(&[&tbs, &alg, &der(0x03, &bits)])
}

fn pem(der: &[u8]) -> String {
    let body = b64(der);
    let lines: Vec<&str> = body
        .as_bytes()
        .chunks(64)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect();
    format!(
        "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
        lines.join("\n")
    )
}

struct TestRoot {
    root_der: Vec<u8>,
    chain_pem: String,
    pck: SigningKey,
}

fn test_root() -> TestRoot {
    let root = SigningKey::from_slice(&[0x11; 32]).unwrap();
    let ca = SigningKey::from_slice(&[0x44; 32]).unwrap();
    let pck = SigningKey::from_slice(&[0x22; 32]).unwrap();
    let root_der = certificate("x0ne test root", &root, "x0ne test root", &root, 1, true);
    let ca_der = certificate("x0ne test CA", &ca, "x0ne test root", &root, 3, true);
    let leaf = certificate("x0ne test PCK", &pck, "x0ne test CA", &ca, 2, false);
    TestRoot {
        chain_pem: pem(&leaf) + &pem(&ca_der) + &pem(&root_der),
        root_der,
        pck,
    }
}

/// `signed` (header and TD report) signed by a fresh attestation key, which
/// the real QE report, re-signed by the test PCK, binds.
fn resigned(signed: &[u8], t: &TestRoot) -> Vec<u8> {
    let real = split(QUOTE);
    let att = SigningKey::from_slice(&[0x33; 32]).unwrap();
    let attestation_key = xy(&att);
    let mut qe_report = real.qe_report.clone();
    let mut h = Sha256::new();
    h.update(&attestation_key);
    h.update(&real.qe_auth_data);
    qe_report[320..352].copy_from_slice(&h.finalize());
    let qe_sig: Signature = t.pck.sign(&qe_report);
    let quote_sig: Signature = att.sign(&signed[..SIGNED]);
    join(
        signed,
        &SigSection {
            quote_signature: quote_sig.to_bytes().to_vec(),
            attestation_key,
            qe_report_signature: qe_sig.to_bytes().to_vec(),
            qe_report,
            qe_auth_data: real.qe_auth_data,
            pck_chain_pem: t.chain_pem.clone().into_bytes(),
        },
    )
}

fn under(t: &TestRoot) -> Policy {
    Policy {
        anchor: TrustAnchor::RootDer(t.root_der.clone()),
        ..policy()
    }
}

// ── What the run recorded ────────────────────────────────────────────────

#[test]
fn the_quote_is_the_runs_and_binds_the_relays_channel_v2_not_an_ohttp_key() {
    assert_eq!(hex(&Sha256::digest(QUOTE)), QUOTE_SHA256);
    assert_eq!(hex(&Sha256::digest(CONFIG.as_bytes())), CONFIG_SHA256);
    let claims = x0ne_confidential::tdx_quote::parse(QUOTE).unwrap();
    assert_eq!(claims.report_data, run_report_data());
    assert_eq!(hex(&claims.mr_td), MRTD);
    assert_eq!(hex(&claims.rtmr[0]), RTMR0);
    assert_eq!(hex(&claims.rtmr[1]), RTMR1);
    assert_eq!(hex(&claims.rtmr[2]), RTMR2);
    assert_eq!(claims.rtmr[3], [0; 48]);
    // SEPT_VE_DISABLE alone; DEBUG clear.
    assert_eq!(u64::from_le_bytes(claims.td_attributes), 1 << 28);
    // The rebuild below gives the quote back, but for Google's zero padding.
    let end = join(QUOTE, &split(QUOTE)).len();
    assert_eq!(end, 4935);
    assert_eq!(join(QUOTE, &split(QUOTE)), QUOTE[..end]);
    assert!(
        QUOTE[end..].iter().all(|&b| b == 0),
        "{} bytes",
        QUOTE.len() - end
    );
    assert_eq!(
        certificates(&split(QUOTE).pck_chain_pem)
            .concat()
            .as_bytes(),
        split(QUOTE).pck_chain_pem
    );
}

// ── The caller's function, then the steps it calls ───────────────────────

/// `verify_key` takes the real quote through `schema`, `key` and `quote`
/// (Intel's root, at the run's clock and four days on) and stops at `binding`:
/// it binds the relay's channel, not this key configuration and nonce.
#[test]
fn verify_key_passes_the_real_quote_to_binding_and_refuses_it_there() {
    let doc = key_document(QUOTE);
    for now in [ATTESTED, NOW] {
        refused(
            attest::verify_key(&doc, &collateral(), &policy(), &[9; 32], now),
            "binding",
            "report_data is not SHA-512 over this key configuration",
        );
    }
    // Control: one bit of Intel's signature, and it stops one step earlier.
    let doc = key_document(&flipped(QUOTE, QUOTE_SIGNATURE + 31, 1));
    refused(
        attest::verify_key(&doc, &collateral(), &policy(), &[9; 32], NOW),
        "quote",
        "quote signature does not verify",
    );
}

/// The caller-sealed path is TDX only: a genuine AMD SEV-SNP report (Azure,
/// Genoa, Apache-2.0, x0ne-infer/tests/VENDORED.md) in the quote's place is
/// refused at the quote, and a key document carrying SEV-SNP evidence beside
/// the quote is refused before any check, so neither falls back to TDX.
#[test]
fn sev_snp_evidence_is_refused_by_the_caller_path() {
    const SNP_REPORT: &[u8] =
        include_bytes!("../../x0ne-infer/tests/snp/trustee_azure_genoa_report.bin");
    let doc = key_document(SNP_REPORT);
    let e = attest::verify_key(&doc, &collateral(), &policy(), &[9; 32], NOW)
        .expect_err("an SEV-SNP report accepted as a TDX quote");
    assert_eq!(e.step, "quote", "{e}");
    let mut json = serde_json::to_value(key_document(QUOTE)).unwrap();
    json["snp"] = json!({ "evidence_b64": b64(SNP_REPORT) });
    let e = serde_json::from_value::<KeyDocument>(json).expect_err("snp field accepted");
    assert!(e.to_string().contains("unknown field `snp`"), "{e}");
}

/// The browser's entry (the C ABI, Intel's root, no test anchor given) says the same.
#[test]
fn the_c_abi_refuses_the_real_quote_at_binding() {
    let (sig, proof) = attest::sign_reference_under_test_keys(REFERENCE_PINNED, NOW);
    let input = json!({
        "document": key_document(QUOTE),
        "collateral": collateral(),
        "policy": {
            "reference_json": std::str::from_utf8(REFERENCE_PINNED).unwrap(),
            "reference_sig_b64": b64(&sig),
            "reference_tlog_proof": proof,
            "reference_keys": "test",
            "tcb": "strict",
            "machine_type": MACHINE,
            "mrtd_hex": MRTD, "rtmr1_hex": RTMR1, "rtmr2_hex": RTMR2, "rtmr3_hex": "00".repeat(48),
            // Required; refused at binding, before any answer names a model.
            "model_digest_hex": "ab".repeat(32),
            "attest": "cpu+gpu",
        },
        "client_nonce_hex": "09".repeat(32),
        "now": NOW,
    });
    let out = abi::call(OP_VERIFY_KEY, &serde_json::to_vec(&input).unwrap());
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .starts_with("refused at binding:"),
        "{out}"
    );
}

/// Every step but the key binding, on the real quote: Intel's chain to the
/// pinned root, the QE, the TD attributes, the run's binding, MRTD and
/// RTMR1..3, Intel's TCB evaluation 22 for this platform and QE, the PCK CRL.
#[test]
fn every_step_after_binding_accepts_the_real_quote_on_intels_collateral_of_2026_10_22() {
    let v = verify(QUOTE, &collateral(), &policy(), NOW).expect("the real quote");
    assert_eq!(
        (v.tcb_platform.as_str(), v.tcb_qe.as_str()),
        ("UpToDate", "UpToDate")
    );
    assert_eq!(v.mrtd_hex, MRTD);
    assert_eq!(v.rtmr0_hex, RTMR0);
    assert_eq!(v.image, IMAGE);
    assert_eq!(
        v.pck_sha256_hex, PCK_SHA256,
        "the PCK the relay named on 2026-10-18"
    );
    // The collateral is at most a day old: the same answers 23 hours on.
    verify(QUOTE, &collateral(), &policy(), NOW + DAY - 3_600).expect("a day");
}

/// At the run's own clock the chain holds (`binding` reached above), but no
/// TCB info of that day was kept: Intel's of 2026-10-22 is not yet issued.
#[test]
fn at_the_runs_clock_the_tcb_info_kept_is_not_yet_valid() {
    refused(
        verify(QUOTE, &collateral(), &policy(), ATTESTED),
        "tcb",
        "validity window",
    );
}

// ── Refusals on Intel's signed bytes ─────────────────────────────────────

#[test]
fn a_bit_of_any_signature_or_of_what_the_qe_signs_is_refused_at_quote() {
    for (at, says) in [
        (QUOTE_SIGNATURE + 31, "quote signature does not verify"),
        (QUOTE_SIGNATURE + 63, "quote signature does not verify"),
        (
            ATTESTATION_KEY,
            "attestation key is not bound by the QE report",
        ),
        (QE_REPORT + 64, "QE report signature does not verify"),
        (QE_REPORT + 320, "QE report signature does not verify"),
        (
            QE_REPORT_SIGNATURE + 63,
            "QE report signature does not verify",
        ),
        (
            QE_AUTH_DATA,
            "attestation key is not bound by the QE report",
        ),
    ] {
        refused(
            verify(&flipped(QUOTE, at, 1), &collateral(), &policy(), NOW),
            "quote",
            says,
        );
    }
}

/// MRTD, each RTMR, the TD attributes (DEBUG, MIGRATABLE), XFAM, MRCONFIGID
/// and report_data are inside what Intel's attestation key signed.
#[test]
fn a_bit_of_the_signed_td_report_is_refused_at_quote() {
    let mut fields = vec![
        ("DEBUG", TD_ATTRIBUTES, 1),
        ("MIGRATABLE", TD_ATTRIBUTES + 3, 1 << 5),
        ("XFAM", XFAM, 1),
        ("MRTD", MR_TD, 1),
        ("MRCONFIGID", MR_CONFIG_ID, 1),
        ("report_data", REPORT_DATA + 63, 1),
    ];
    for i in 0..4 {
        fields.push(("RTMR", RTMR_AT + 48 * i, 0x80));
    }
    for (what, at, bit) in fields {
        let q = flipped(QUOTE, at, bit);
        let e = verify(&q, &collateral(), &policy(), NOW).unwrap_err();
        assert_eq!(e.step, "quote", "{what}: {e}");
        assert!(
            e.reason.contains("quote signature does not verify"),
            "{what}: {e}"
        );
    }
    // The bit flipped is the one the verifier reads as DEBUG.
    assert!(
        x0ne_confidential::tdx_quote::parse(&flipped(QUOTE, TD_ATTRIBUTES, 1))
            .unwrap()
            .is_debug()
    );
}

#[test]
fn pins_other_than_the_quotes_registers_are_refused_at_measurement() {
    let mut p = policy();
    p.mrtd_pin.as_mut().unwrap()[47] ^= 1;
    refused(
        verify(QUOTE, &collateral(), &p, NOW),
        "measurement",
        "MRTD c1ee9c16",
    );
    for i in 0..3 {
        let mut p = policy();
        p.rtmr_pins[i].as_mut().unwrap()[0] ^= 0x80;
        refused(
            verify(QUOTE, &collateral(), &p, NOW),
            "measurement",
            &format!("RTMR{} ", i + 1),
        );
    }
    let mut rtmr0 = unhex::<48>(RTMR0);
    rtmr0[47] ^= 1;
    refused(
        verify(QUOTE, &collateral(), &policy().pinned_rtmr0(rtmr0), NOW),
        "measurement",
        &format!("RTMR0 {RTMR0} is not the pinned"),
    );
    verify(
        QUOTE,
        &collateral(),
        &policy().pinned_rtmr0(unhex(RTMR0)),
        NOW,
    )
    .expect("the control: the quote's RTMR0");
}

#[test]
fn a_report_data_other_than_the_one_expected_is_refused_at_binding() {
    let mut other = run_report_data();
    other[0] ^= 1;
    refused(
        attest::verify_quote(QUOTE, CCEL, &other, &collateral(), &policy(), NOW),
        "binding",
        "report_data is not the one this caller expects",
    );
}

// ── The checks Intel's signature shields, on the real TD report re-signed ─

/// The real TD report under a root of our own reaches the checks behind the
/// signature: unchanged, it passes `quote` and `binding` and stops at `tcb`
/// (Intel's collateral is not under that root); Intel's root refuses it.
/// Its registers changed stop at `tcb` too: as in the relay, they are judged
/// only on a platform Intel's collateral accepted.
#[test]
fn the_real_td_report_resigned_under_a_test_root_reaches_each_check() {
    let t = test_root();
    let same = resigned(QUOTE, &t);
    refused(
        verify(&same, &collateral(), &under(&t), NOW),
        "tcb",
        "SGX extension absent from PCK leaf",
    );
    refused(
        verify(&same, &collateral(), &policy(), NOW),
        "quote",
        "root does not match the pinned trust anchor",
    );
    for (what, at, bit, step, says) in [
        (
            "DEBUG",
            TD_ATTRIBUTES,
            1,
            "quote",
            "TDATTRIBUTES.DEBUG is set",
        ),
        (
            "MIGRATABLE",
            TD_ATTRIBUTES + 3,
            1 << 5,
            "quote",
            "TDATTRIBUTES carries MIGRATABLE",
        ),
        (
            "MRTD",
            MR_TD,
            1,
            "tcb",
            "SGX extension absent from PCK leaf",
        ),
        (
            "RTMR0",
            RTMR_AT,
            1,
            "tcb",
            "SGX extension absent from PCK leaf",
        ),
        (
            "RTMR1",
            RTMR_AT + 48,
            1,
            "tcb",
            "SGX extension absent from PCK leaf",
        ),
        (
            "RTMR2",
            RTMR_AT + 96,
            1,
            "tcb",
            "SGX extension absent from PCK leaf",
        ),
        (
            "RTMR3",
            RTMR_AT + 144,
            1,
            "tcb",
            "SGX extension absent from PCK leaf",
        ),
        ("report_data", REPORT_DATA, 1, "binding", "report_data"),
    ] {
        let q = resigned(&flipped(QUOTE, at, bit), &t);
        let e = verify(&q, &collateral(), &under(&t), NOW).unwrap_err();
        assert_eq!(e.step, step, "{what}: {e}");
        assert!(e.reason.contains(says), "{what}: {e}");
    }
}

// ── The PCK chain ────────────────────────────────────────────────────────

#[test]
fn a_cut_pck_chain_is_refused_at_quote() {
    let certs = certificates(&split(QUOTE).pck_chain_pem);
    assert_eq!(certs.len(), 3, "PCK, PCK Platform CA, Intel SGX Root CA");
    let (leaf, ca, root) = (certs[0].as_str(), certs[1].as_str(), certs[2].as_str());
    // Control: the same three, rebuilt, still reach `binding` past `quote`.
    let whole = with_chain(QUOTE, &[leaf, ca, root]);
    verify(&whole, &collateral(), &policy(), NOW).expect("rebuilt whole");
    for (chain, says) in [
        (
            vec![leaf, root, root],
            "chain link 0: signature does not verify",
        ),
        (vec![leaf, ca], "chain has 2 certificate(s)"),
        // Intel's own CA in the PCK's place: every signature holds.
        (vec![ca, root, root], "the PCK is a CA"),
        // One certificate more than Intel's chain.
        (vec![leaf, ca, root, root], "chain has 4 certificate(s)"),
        (vec![leaf], "chain has 1 certificate(s)"),
        (vec![], "chain has 0 certificate(s)"),
        (
            vec![leaf, ca, ca],
            "root does not match the pinned trust anchor",
        ),
    ] {
        refused(
            verify(&with_chain(QUOTE, &chain), &collateral(), &policy(), NOW),
            "quote",
            says,
        );
    }
    let cut = &certs.concat()[..certs.concat().len() - 200];
    refused(
        verify(&with_chain(QUOTE, &[cut]), &collateral(), &policy(), NOW),
        "quote",
        "PEM: incomplete PEM",
    );
}

/// The PCK certificate was issued 2026-08-13: before, the chain is out of its window.
#[test]
fn the_pck_chain_is_refused_outside_its_validity() {
    refused(
        verify(QUOTE, &collateral(), &policy(), 1_785_542_400),
        "quote",
        "outside its validity window",
    );
}

// ── Intel's collateral: stale, another platform's, cut ───────────────────

fn with(edit: impl FnOnce(&mut Collateral)) -> Collateral {
    let mut c = collateral();
    edit(&mut c);
    c
}

#[test]
fn collateral_stale_of_another_platform_or_cut_is_refused_at_tcb() {
    let cases: Vec<(&str, Collateral, u64, &str)> = vec![
        (
            "another platform",
            with(|c| c.tcb_info = TCB_OTHER_PLATFORM.into()),
            NOW,
            "tcbInfo is for FMSPC 90c06f000000, the platform is 00806f050000",
        ),
        (
            "expired, another platform",
            with(|c| c.tcb_info = TCB_2026_08_20.into()),
            NOW,
            "validity window",
        ),
        (
            "expired QE identity",
            with(|c| c.qe_identity = QE_2026_08_20.into()),
            NOW,
            "validity window",
        ),
        (
            "Intel's default evaluation",
            with(|c| c.tcb_info = TCB_STANDARD.into()),
            NOW,
            "evaluation 20",
        ),
        (
            "two days old",
            collateral(),
            NOW + 2 * DAY,
            "this verifier accepts at most 86400 s",
        ),
        (
            "edited",
            with(|c| c.tcb_info = c.tcb_info.replacen("UpToDate", "OutOfDate", 1)),
            NOW,
            "signature does not verify over the collateral bytes",
        ),
        (
            "TCB signing chain cut to its leaf",
            with(|c| c.tcb_info_chain_pem = certificates(TCB_CHAIN.as_bytes())[0].clone()),
            NOW,
            "issuer chain has 1 certificate(s)",
        ),
        (
            "TCB signing chain withheld",
            with(|c| c.tcb_info_chain_pem = String::new()),
            NOW,
            "update=early collateral relayed without tcb_info_chain_pem",
        ),
        (
            "TCB signing chain of no certificate",
            with(|c| c.tcb_info_chain_pem = "\n".into()),
            NOW,
            "issuer chain has 0 certificate(s)",
        ),
        (
            "withheld",
            Collateral {
                pck_crl_der_b64: b64(CRL),
                root_ca_crl_der_b64: b64(ROOT_CA_CRL),
                gce_endorsement_b64: b64(ENDORSEMENT),
                ..Collateral::default()
            },
            NOW,
            "accepts on Intel's update=early collateral: none relayed",
        ),
    ];
    for (what, col, now, says) in cases {
        let e = verify(QUOTE, &col, &policy(), now).unwrap_err();
        assert_eq!(e.step, "tcb", "{what}: {e}");
        assert!(e.reason.contains(says), "{what}: {e}");
    }
}

#[test]
fn a_pck_crl_stale_of_another_issuer_or_tampered_is_refused_at_revocation() {
    let mut tampered = CRL.to_vec();
    let n = tampered.len();
    tampered[n - 3] ^= 1;
    for (what, crl, says) in [
        (
            "the run's day, four days on",
            CRL_2026_10_18.to_vec(),
            "PCK CRL was issued",
        ),
        ("Intel's root CA CRL", ROOT_CA_CRL.to_vec(), "issued by"),
        ("one bit", tampered, "signature does not verify"),
        ("withheld", vec![], "not a DER CRL"),
    ] {
        let col = with(|c| c.pck_crl_der_b64 = b64(&crl));
        let e = verify(QUOTE, &col, &policy(), NOW).unwrap_err();
        assert_eq!(e.step, "revocation", "{what}: {e}");
        assert!(e.reason.contains(says), "{what}: {e}");
    }
}

// ── The image, as the relay judges it: no pin from the caller ────────────

/// The relay's checks of the run, made by the caller: Google's launch
/// endorsement of the MRTD under Google's root, the TD's log replayed into
/// RTMR0..3 and judged against the reference x0ne-cvm 0.2.3 with RTMR0
/// pinned: its 16 RTMR0 events, the configuration volume of Google's
/// binary second. No pin from the caller, no warning left.
#[test]
fn the_real_quote_passes_on_googles_endorsement_and_its_own_log_without_pins() {
    let p = production();
    let v = verify(QUOTE, &collateral(), &p, NOW).expect("the run, as the relay judges it");
    assert_eq!(v.image, IMAGE);
    assert_eq!((v.mrtd_hex.as_str(), v.rtmr0_hex.as_str()), (MRTD, RTMR0));
    assert_eq!(v.firmware_sha384_hex, FIRMWARE_SHA384);
    assert_eq!(
        (v.tcb_platform.as_str(), v.tcb_qe.as_str()),
        ("UpToDate", "UpToDate")
    );
    assert!(v.warnings.is_empty(), "{:?}", v.warnings);
}

/// The relay's policy on the run's day: `strict`, no pin; Google's binary
/// as measured elsewhere ([`measured_firmware`]).
fn production() -> Policy {
    Policy::production(reference(), MACHINE)
        .with_strict_tcb()
        .with_firmware_measured_elsewhere(measured_firmware())
}

/// [`production`] on the pinned reference edited by `f`, then signed by the
/// TEST archive key and logged at [`NOW`]: a reference x0ne could publish.
fn production_on(f: impl FnOnce(&mut serde_json::Value)) -> Policy {
    let mut v: serde_json::Value = serde_json::from_slice(REFERENCE_PINNED).unwrap();
    f(&mut v);
    let json = serde_json::to_vec(&v).unwrap();
    Policy::production(admitted(&json), MACHINE)
        .with_strict_tcb()
        .with_firmware_measured_elsewhere(measured_firmware())
}

/// The relay's default since 2026-10-28 on the run's quote: `transparent`
/// accepts on Intel's `standard` answer of 2026-10-22 (evaluation 20) and
/// names its `early` one (evaluation 22), both UpToDate, in the relay's
/// disclosure. `early` withheld: accepted, its absence named. `standard`
/// withheld: refused, as the relay accepts on what it fetched.
#[test]
fn transparent_accepts_the_real_quote_on_standard_and_names_early() {
    let p = Policy::production(reference(), MACHINE)
        .with_firmware_measured_elsewhere(measured_firmware());
    assert_eq!(p.tcb_mode, x0ne_sdk::TcbMode::Transparent);
    let v = verify(QUOTE, &collateral_both(), &p, NOW).expect("the run, under transparent");
    assert_eq!(v.tcb.schema, x0ne_sdk::TCB_DISCLOSURE_SCHEMA);
    assert_eq!(v.tcb.policy, "transparent");
    let intel = v.tcb.intel.clone().expect("Intel's verdict");
    assert_eq!(
        (intel.accepted_under.as_str(), intel.accepted_evaluation),
        ("standard", 20)
    );
    assert_eq!(intel.platform, "UpToDate");
    let latest = intel.latest.expect("Intel's early view, named");
    assert_eq!(
        (latest.platform.as_str(), latest.tcb_evaluation),
        ("UpToDate", 22)
    );
    let no_early = Collateral {
        tcb_info: String::new(),
        tcb_info_chain_pem: String::new(),
        qe_identity: String::new(),
        qe_identity_chain_pem: String::new(),
        ..collateral_both()
    };
    let v = verify(QUOTE, &no_early, &p, NOW).expect("early withheld");
    let intel = v.tcb.intel.expect("Intel's verdict");
    assert!(intel.latest.is_none());
    assert!(intel.latest_unavailable.is_some());
    refused(
        verify(QUOTE, &collateral(), &p, NOW),
        "tcb",
        "update=standard",
    );
}

/// The run's own reference under its own signature by the TEST archive key
/// (2026-10-18): the signature and the log pass, then it is refused for
/// reporting RTMR0 instead of pinning it, with the relay's 104, named
/// (since 2026-10-31; until then it was admitted). One byte of it changed:
/// refused with 99, before that. The pinned one is admitted and named TEST
/// in the provenance every verified key carries.
#[test]
fn the_runs_reference_is_refused_with_104_and_the_pinned_one_admitted_as_test() {
    let e = attest::admit_reference_under_test_keys(
        &SignedReference {
            json: REFERENCE,
            signature: REFERENCE_SIG,
            tlog_proof: &tlog_proof(),
        },
        &mut attest::LogMemory::new(),
        None,
        NOW,
    )
    .unwrap_err();
    assert_eq!((e.step, e.code), ("reference", 104), "{e}");
    assert!(
        e.reason.contains("RTMR0")
            && e.reason.contains("\"report\"")
            && e.reason.contains("gce-endorsed"),
        "{e}"
    );
    let r = reference();
    assert_eq!(r.provenance()["signed_by"]["test_key"], true);
    let mut edited = REFERENCE.to_vec();
    let at = edited.iter().position(|b| *b == b'3').unwrap();
    edited[at] = b'4';
    let e = attest::admit_reference_under_test_keys(
        &SignedReference {
            json: &edited,
            signature: REFERENCE_SIG,
            tlog_proof: &tlog_proof(),
        },
        &mut attest::LogMemory::new(),
        None,
        NOW,
    )
    .unwrap_err();
    assert_eq!((e.step, e.code), ("reference", 99), "{e}");
}

/// SECURITY: the MRTD holds only under Google's endorsement of exactly it,
/// signed under the root the reference names: each case at `measurement`.
#[test]
fn an_endorsement_withheld_altered_of_another_firmware_or_root_is_refused() {
    let mut signature = ENDORSEMENT.to_vec();
    let n = signature.len();
    signature[n - 1] ^= 1;
    let mut signed = ENDORSEMENT.to_vec();
    signed[3] ^= 1;
    let endorsed = |blob: &[u8]| with(|c| c.gce_endorsement_b64 = b64(blob));
    let edit = |f: &dyn Fn(&mut Policy)| {
        let mut p = production();
        f(&mut p);
        p
    };
    let cases: Vec<(&str, Collateral, Policy, String)> = vec![
        (
            "another firmware's",
            endorsed(OTHER_ENDORSEMENT),
            production(),
            format!("MRTD: MRTD {MRTD} is not in the Google endorsement"),
        ),
        (
            "its signature, one bit",
            endorsed(&signature),
            production(),
            "Google endorsement: endorsement signature".into(),
        ),
        (
            "what Google signed, one bit",
            endorsed(&signed),
            production(),
            "Google endorsement: ".into(),
        ),
        (
            "withheld",
            endorsed(&[]),
            production(),
            format!("MRTD {MRTD} has no Google launch endorsement"),
        ),
        (
            "Intel's root in Google's place",
            collateral(),
            edit(&|p| p.gce_root_der = INTEL_ROOT_DER.to_vec()),
            "names Google root 910dcbe4".into(),
        ),
        (
            "a reference naming another Google root",
            collateral(),
            production_on(|v| v["mrtd"]["root_spki_sha256"] = json!("ab".repeat(32))),
            format!("names Google root {}", "ab".repeat(32)),
        ),
        (
            "a machine type the reference does not name",
            collateral(),
            edit(&|p| p.machine_type = "c3-standard-4".into()),
            "MRTD: reference has no mrtd.ram_gib entry".into(),
        ),
        (
            "an SVN floor above the endorsed firmware's",
            collateral(),
            production_on(|v| v["mrtd"]["min_svn"] = json!(1_000)),
            "below the reference's minimum 1000".into(),
        ),
    ];
    for (what, col, p, says) in cases {
        let e = verify(QUOTE, &col, &p, NOW).unwrap_err();
        assert_eq!(e.step, "measurement", "{what}: {e}");
        assert!(e.reason.contains(&says), "{what}: {e}");
    }
}

// The run's log, by offset: each event is MR index (4), type (4), digest
// count (4), algorithm (2), SHA-384 (48), data size (4), data.
const TYPE: usize = 4;
const DIGEST: usize = 14;
const DATA: usize = 66;
/// RTMR1, EV_EFI_ACTION "Calling EFI Application from Boot Option".
const EV_CALLING: usize = 24_566;
/// RTMR1, EV_EFI_GPT_EVENT: the reference's GPT.
const EV_GPT: usize = 24_742;
/// RTMR2, EV_IPL ".linux": the kernel's digest.
const EV_LINUX: usize = 25_598;
/// RTMR1, EV_EFI_ACTION "Exit Boot Services Returned with Success": the last.
const EV_LAST: usize = 26_890;
/// After the last event, the log's 0xFF padding.
const LOG_END: usize = 26_996;

fn ccel_with(edit: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut log = CCEL.to_vec();
    edit(&mut log);
    log
}

/// SECURITY: RTMR1 and RTMR2 hold only as the replay of the TD's log, each
/// event on the reference's closed list: one event changed, cut or added,
/// and the log withheld, are refused at `measurement`.
#[test]
fn a_log_changed_cut_extended_or_withheld_is_refused_at_measurement() {
    assert_eq!(&CCEL[EV_CALLING + DATA..][..7], b"Calling");
    assert_eq!(&CCEL[EV_GPT + DATA..][..8], b"EFI PART");
    assert_eq!(&CCEL[EV_LINUX + DATA..][..4], b".\0l\0");
    assert_eq!(&CCEL[EV_LAST + DATA..][..9], b"Exit Boot");
    assert_eq!(&CCEL[LOG_END..][..4], [0xff; 4]);
    let separator_on_rtmr2 = {
        let mut ev = 3u32.to_le_bytes().to_vec(); // MR index 3: RTMR2
        ev.extend(4u32.to_le_bytes()); // EV_SEPARATOR
        ev.extend(1u32.to_le_bytes());
        ev.extend(0x000cu16.to_le_bytes()); // TPM_ALG_SHA384
        ev.extend(Sha384::digest([0u8; 4]));
        ev.extend(4u32.to_le_bytes());
        ev.extend([0u8; 4]);
        ev
    };
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        (
            "an EFI action's text, its digest kept",
            ccel_with(|l| l[EV_CALLING + DATA] ^= 1),
            "image: RTMR1 event",
        ),
        (
            "the GPT's event type",
            ccel_with(|l| l[EV_GPT + TYPE..][..4].copy_from_slice(&0x8000_0007u32.to_le_bytes())),
            "the reference GPT digest as a",
        ),
        (
            "a byte of the GPT, its digest kept",
            ccel_with(|l| l[EV_GPT + DATA + 20] ^= 1),
            "whose data does not hash to it",
        ),
        (
            "the kernel's digest",
            ccel_with(|l| l[EV_LINUX + DIGEST] ^= 1),
            "image: CCEL replay of RTMR2",
        ),
        (
            "the last event cut",
            ccel_with(|l| l[EV_LAST..LOG_END].fill(0xff)),
            "image: CCEL replay of RTMR1",
        ),
        (
            "a separator appended on RTMR2",
            ccel_with(|l| {
                l[LOG_END..][..separator_on_rtmr2.len()].copy_from_slice(&separator_on_rtmr2)
            }),
            "image: CCEL replay of RTMR2",
        ),
        ("withheld", vec![], "CCEL"),
    ];
    for (what, log, says) in cases {
        let e = attest::verify_quote(
            QUOTE,
            &log,
            &run_report_data(),
            &collateral(),
            &production(),
            NOW,
        )
        .unwrap_err();
        assert_eq!(e.step, "measurement", "{what}: {e}");
        assert!(e.reason.contains(says), "{what}: {e}");
    }
}

/// SECURITY: RTMR0 is pinned to the binary Google's endorsement names,
/// relayed in the collateral and measured here. A carrier that withholds it
/// is refused at `measurement`, as the relay refuses a bucket without it;
/// a `test-anchor` measurement made elsewhere stands in for the binary only
/// if the endorsement names that binary.
#[test]
fn the_endorsed_binary_withheld_or_measured_as_another_is_refused() {
    let no_hook = Policy::production(reference(), MACHINE).with_strict_tcb();
    refused(
        verify(QUOTE, &collateral(), &no_hook, NOW),
        "measurement",
        &format!(
            "firmware: the binary Google's endorsement names ({FIRMWARE_SHA384}) was not relayed"
        ),
    );
    let other = Policy::production(reference(), MACHINE)
        .with_strict_tcb()
        .with_firmware_measured_elsewhere(attest::FirmwareMeasuredElsewhere {
            sha384_hex: "00".repeat(48),
            ..measured_firmware()
        });
    refused(
        verify(QUOTE, &collateral(), &other, NOW),
        "measurement",
        &format!("firmware: the endorsement names binary {FIRMWARE_SHA384}"),
    );
    // Another configuration volume measured: RTMR0's second event is not it.
    let mut cfv = measured_firmware();
    cfv.cfv_sha384[0] ^= 1;
    let p = Policy::production(reference(), MACHINE)
        .with_strict_tcb()
        .with_firmware_measured_elsewhere(cfv);
    refused(
        verify(QUOTE, &collateral(), &p, NOW),
        "measurement",
        "the configuration volume measured (4e1221cf",
    );
}

/// SECURITY: an RTMR0 pinned by the reference to another boot's events is
/// refused, at the event that differs, before any pin of the caller's.
#[test]
fn an_rtmr0_event_other_than_the_references_is_refused_at_measurement() {
    let p = production_on(|v| {
        v["rtmr0"]["events_sha384"][MACHINE][2] = json!("ab".repeat(48));
    });
    // The log's event 4 is the third that extends RTMR0.
    refused(
        verify(QUOTE, &collateral(), &p, NOW),
        "measurement",
        "RTMR0 event 4: cfa4e2c6",
    );
    let p = production_on(|v| {
        v["rtmr0"]["events_sha384"][MACHINE]
            .as_array_mut()
            .unwrap()
            .pop();
    });
    refused(
        verify(QUOTE, &collateral(), &p, NOW),
        "measurement",
        "one event more than the 15 the reference pins",
    );
}

/// Google's binary itself, relayed as a carrier relays it and measured
/// here, no measurement made elsewhere. `X0NE_GCE_FIRMWARE` names the 2 MiB
/// binary 53c558fe… from Google's bucket (`x0ne-infer/tests/gce_firmware_live.rs`
/// fetches it); one byte of it changed is not the binary Google signed.
#[test]
#[ignore = "needs Google's firmware binary: X0NE_GCE_FIRMWARE=<path to 53c558fe….fd>"]
fn the_real_quote_passes_on_googles_binary_relayed_and_measured_here() {
    let path = std::env::var("X0NE_GCE_FIRMWARE").expect("X0NE_GCE_FIRMWARE");
    let binary = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let p = Policy::production(reference(), MACHINE).with_strict_tcb();
    let v = verify(QUOTE, &with(|c| c.gce_firmware_b64 = b64(&binary)), &p, NOW)
        .expect("the run on Google's binary");
    assert_eq!(v.firmware_sha384_hex, FIRMWARE_SHA384);
    assert_eq!(v.rtmr0_hex, RTMR0);
    assert!(v.warnings.is_empty(), "{:?}", v.warnings);
    let mut other = binary;
    other[0x10] ^= 1;
    refused(
        verify(QUOTE, &with(|c| c.gce_firmware_b64 = b64(&other)), &p, NOW),
        "measurement",
        "is not the firmware Google endorsed",
    );
}

/// What a carrier fetches after the endorsement: the binary it names, in
/// Google's bucket, read from the blob without believing it.
#[test]
fn the_firmware_url_is_the_binary_the_endorsement_names_in_googles_bucket() {
    assert_eq!(
        attest::gce_firmware_url(ENDORSEMENT).unwrap(),
        format!(
            "https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/{FIRMWARE_SHA384}.fd"
        )
    );
    assert!(attest::gce_firmware_url(b"not an endorsement").is_err());
}

// ── Intel's root CA CRL ──────────────────────────────────────────────────

/// SECURITY: the root CA CRL, relayed like the rest, is read as the relay
/// reads it (`tcb::verify_root_crl`): withheld, another issuer's or altered,
/// refused at `revocation` (the relay's 76).
#[test]
fn a_root_ca_crl_withheld_of_another_issuer_or_altered_is_refused_at_revocation() {
    let mut altered = ROOT_CA_CRL.to_vec();
    let n = altered.len();
    altered[n - 3] ^= 1;
    for (what, crl, says) in [
        ("withheld", vec![], "root CA CRL: not a DER CRL"),
        ("the PCK CA's CRL", CRL.to_vec(), "root CA CRL: issued by"),
        ("one bit", altered, "root CA CRL: signature does not verify"),
    ] {
        let col = with(|c| c.root_ca_crl_der_b64 = b64(&crl));
        let e = verify(QUOTE, &col, &policy(), NOW).unwrap_err();
        assert_eq!(e.step, "revocation", "{what}: {e}");
        assert!(e.reason.contains(says), "{what}: {e}");
    }
}

// ── The relay's order ────────────────────────────────────────────────────

/// The relay judges the platform (TCB, then both CRLs) before the image: the
/// same evidence gets the same first refusal from both verifiers.
#[test]
fn the_platform_is_judged_before_the_image_as_in_the_relay() {
    let mut pinned_wrong = policy();
    pinned_wrong.mrtd_pin.as_mut().unwrap()[0] ^= 1;
    let no_endorsement = |c: &mut Collateral| c.gce_endorsement_b64 = String::new();
    for (what, col, p, log, step) in [
        (
            "Intel's default evaluation, a wrong pin",
            with(|c| c.tcb_info = TCB_STANDARD.into()),
            pinned_wrong,
            CCEL.to_vec(),
            "tcb",
        ),
        (
            "Intel's default evaluation, no log",
            with(|c| c.tcb_info = TCB_STANDARD.into()),
            production(),
            vec![],
            "tcb",
        ),
        (
            "no PCK CRL, no endorsement",
            with(|c| {
                c.pck_crl_der_b64 = String::new();
                no_endorsement(c)
            }),
            production(),
            CCEL.to_vec(),
            "revocation",
        ),
        (
            "no root CA CRL, no endorsement, no log",
            with(|c| {
                c.root_ca_crl_der_b64 = String::new();
                no_endorsement(c)
            }),
            production(),
            vec![],
            "revocation",
        ),
        (
            "no endorsement, no log",
            with(no_endorsement),
            production(),
            vec![],
            "measurement",
        ),
    ] {
        let e = attest::verify_quote(QUOTE, &log, &run_report_data(), &col, &p, NOW).unwrap_err();
        assert_eq!(e.step, step, "{what}: {e}");
    }
}

// ── What a carrier is told to fetch ──────────────────────────────────────

/// Intel's default answer is its `standard` evaluation, refused above under
/// the floor of 22: the GETs `Collateral` names ask for `update=early`, as the
/// relay's own (`tcb::pcs_tcb_url`, `tcb::pcs_qe_identity_url`) do.
#[test]
fn the_gets_named_on_collateral_are_the_ones_the_relay_makes() {
    let relay = include_str!("../../x0ne-infer/src/tcb.rs");
    assert!(relay.contains("\"{base}/tdx/certification/v4/tcb?fmspc={fmspc_hex}&update=early\""));
    assert!(relay.contains("\"{base}/tdx/certification/v4/qe/identity?update=early\""));
    let sdk = include_str!("../src/attest.rs");
    assert!(sdk.contains("/// `GET /tdx/certification/v4/tcb?fmspc=<fmspc>&update=early`"));
    assert!(sdk.contains("/// `GET /tdx/certification/v4/qe/identity?update=early`"));
}

/// The root CA CRL and Google's endorsement come from the URLs the relay
/// reads: the same constants, compiled from its files.
#[test]
fn the_root_crl_and_the_endorsement_are_fetched_where_the_relay_fetches_them() {
    assert_eq!(
        attest::ROOT_CA_CRL_URL,
        "https://certificates.trustedservices.intel.com/IntelSGXRootCA.der"
    );
    assert_eq!(
        attest::gce_endorsement_url(QUOTE).unwrap(),
        format!(
            "https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/tdx/{MRTD}.binarypb"
        )
    );
}
