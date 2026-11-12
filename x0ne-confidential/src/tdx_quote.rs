//! Structural verification of a raw Intel TDX quote.
//!
//! Scope, stated plainly: this checks FORMAT (TD Quote v4), TEE TYPE (TDX),
//! and FRESHNESS (our nonce echoed in report_data). It does NOT verify the
//! Intel PCS signature chain — `x0ne-infer` does (`pcs::verify_tdx_chain`),
//! and claiming it here would be the green badge lying. `VerifierStrength::Structural`
//! carries exactly this meaning, and the cc report prints the limitation.
//!
//! Layout (Intel TD Quote v4, the only version parsed): 48-byte header
//! (version u16 LE at 0, att_key_type u16 LE at 2, 2 = ECDSA with P-256,
//! tee_type u32 LE at 4, TDX = 0x81, two reserved u16 at 8, qe_vendor_id at
//! 12..28, user data at 28..48), then the 584-byte TD report body. Body offsets:
//! td_attributes [120..128], mr_td [136..184], rtmr0..3 [328..520],
//! report_data [520..584]. [280..328] is MROWNERCONFIG, not RTMR0.

use crate::model::{
    ExecutionEnvironment, TeeTechnology, Verdict, VerifierStrength, WorkloadIdentity,
};

const HEADER: usize = 48;
const BODY: usize = 584;
const TD_ATTRIBUTES: usize = 120;
const MR_TD: usize = 136;
const RTMR0: usize = 328;
const REPORT_DATA: usize = 520;
const TEE_TDX: u32 = 0x81;
const ATT_KEY_TYPE: usize = 2;
const QE_VENDOR_ID: usize = 12;

/// `att_key_type` of a quote signed with ECDSA on P-256, the one type
/// `x0ne-infer` verifies. 3 is ECDSA on P-384.
pub const ATT_KEY_ECDSA_P256: u16 = 2;

/// `qe_vendor_id` of Intel's quoting enclave. Intel's verifier requires it
/// (SGX-TDX-DCAP-QuoteVerificationLibrary, `Quote::validate`, read
/// 2026-11-10); the two real quotes of the tree carry it.
pub const INTEL_QE_VENDOR_ID: [u8; 16] = [
    0x93, 0x9a, 0x72, 0x33, 0xf7, 0x9c, 0x4c, 0xa9, 0x94, 0x0a, 0x0d, 0xb3, 0x95, 0x7f, 0x06, 0x07,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdxQuoteClaims {
    /// TDATTRIBUTES, little-endian; bit 0 is DEBUG.
    pub td_attributes: [u8; 8],
    pub mr_td: [u8; 48],
    pub rtmr: [[u8; 48]; 4],
    pub report_data: [u8; 64],
    pub tee_type: u32,
}

/// TDATTRIBUTES a production TD may carry, by the groups of Intel's TDX Module
/// ABI reference (348551-008, May 2026, table 3.23). Refused: TUD (3:0) and TUP
/// (15:4), the host debugs or profiles the TD; MIGRATABLE; every reserved bit
/// Intel says must be 0 (RESERVED_N 26:23, 55:32, 61:56). Allowed: the SEC bits
/// marked positive (ICSSD, SERVTD_EXT, RESERVED_P 22:18, which Intel lets a
/// verifier accept, LASS, PKS, reserved bit 31), SEPT_VE_DISABLE, which a Linux
/// guest requires, and the two defined OTHER bits, TPA and PERFMON.
pub const TD_ATTRIBUTES_ALLOWED: u64 =
    (0x7f << 16) | (1 << 27) | (1 << 28) | (1 << 30) | (1 << 31) | (1 << 62) | (1 << 63);

/// Names from Linux `arch/x86/include/asm/shared/tdx.h` (v6.19), plus SERVTD_EXT
/// from the ABI reference. Bit 31, `KL` in Linux, is reserved in the ABI (008).
const TD_ATTRIBUTE_NAMES: [(u32, &str); 12] = [
    (0, "DEBUG"),
    (4, "HGS_PLUS_PROF"),
    (5, "PERF_PROF"),
    (6, "PMT_PROF"),
    (16, "ICSSD"),
    (17, "SERVTD_EXT"),
    (27, "LASS"),
    (28, "SEPT_VE_DISABLE"),
    (29, "MIGRATABLE"),
    (30, "PKS"),
    (62, "TPA"),
    (63, "PERFMON"),
];

/// `DEBUG, PERF_PROF, bit 12`: each set bit of `bits`, by name when it has one.
pub fn td_attribute_names(bits: u64) -> String {
    (0..64u32)
        .filter(|b| bits & (1 << b) != 0)
        .map(|b| match TD_ATTRIBUTE_NAMES.iter().find(|(n, _)| *n == b) {
            Some((_, name)) => (*name).to_string(),
            None => format!("bit {b}"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

impl TdxQuoteClaims {
    /// TDATTRIBUTES.DEBUG: the host VMM can read and write this TD's state.
    pub fn is_debug(&self) -> bool {
        self.td_attributes[0] & 1 != 0
    }

    /// The set bits outside [`TD_ATTRIBUTES_ALLOWED`]; 0 for a production TD.
    pub fn refused_attributes(&self) -> u64 {
        u64::from_le_bytes(self.td_attributes) & !TD_ATTRIBUTES_ALLOWED
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuoteRejected {
    TooShort { len: usize, need: usize },
    NotTdx { tee_type: u32 },
    UnsupportedVersion { version: u16 },
    UnsupportedAttestationKeyType { att_key_type: u16 },
    NotIntelQe { qe_vendor_id: [u8; 16] },
}

impl std::fmt::Display for QuoteRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { len, need } => write!(f, "{len} bytes, a TD quote has {need}"),
            Self::NotTdx { tee_type } => {
                write!(f, "tee_type {tee_type:#x} is not TDX (expected {TEE_TDX:#x})")
            }
            Self::UnsupportedVersion { version } => {
                write!(f, "version {version} unsupported (expected 4)")
            }
            Self::UnsupportedAttestationKeyType { att_key_type } => write!(
                f,
                "att_key_type {att_key_type} unsupported (expected {ATT_KEY_ECDSA_P256}: ECDSA with P-256)"
            ),
            Self::NotIntelQe { qe_vendor_id } => write!(
                f,
                "qe_vendor_id {} is not Intel's ({})",
                hex(qe_vendor_id),
                hex(&INTEL_QE_VENDOR_ID)
            ),
        }
    }
}

/// The fields of the 48-byte header whose value the format fixes: version 4,
/// an ECDSA P-256 attestation key, TDX, Intel's QE. Each is refused by name.
/// The header is under the quote's signature; this reads it before any
/// signature is checked, and says nothing of who wrote the quote.
pub fn check_header(quote: &[u8]) -> Result<(), QuoteRejected> {
    if quote.len() < HEADER {
        return Err(QuoteRejected::TooShort {
            len: quote.len(),
            need: HEADER + BODY,
        });
    }
    let version = u16::from_le_bytes([quote[0], quote[1]]);
    // v4 ONLY (2026-08-20). v5 used to be accepted with the v4 offsets:
    // fail-closed (a wrong offset makes measurements mismatch, never match by
    // luck on 48-byte digests), but dishonest — a parser that names a version
    // it does not implement. v5 support means implementing its layout, with a
    // real v5 quote in hand, not waving it through.
    if version != 4 {
        return Err(QuoteRejected::UnsupportedVersion { version });
    }
    let tee_type = u32::from_le_bytes([quote[4], quote[5], quote[6], quote[7]]);
    if tee_type != TEE_TDX {
        return Err(QuoteRejected::NotTdx { tee_type });
    }
    // SECURITY: the signature section is read as a P-256 signature and key
    // whatever this field says. Another type is another layout and another
    // curve: refused, not read as this one.
    let att_key_type = u16::from_le_bytes([quote[ATT_KEY_TYPE], quote[ATT_KEY_TYPE + 1]]);
    if att_key_type != ATT_KEY_ECDSA_P256 {
        return Err(QuoteRejected::UnsupportedAttestationKeyType { att_key_type });
    }
    let mut qe_vendor_id = [0u8; 16];
    qe_vendor_id.copy_from_slice(&quote[QE_VENDOR_ID..QE_VENDOR_ID + 16]);
    if qe_vendor_id != INTEL_QE_VENDOR_ID {
        return Err(QuoteRejected::NotIntelQe { qe_vendor_id });
    }
    Ok(())
}

pub fn parse(quote: &[u8]) -> Result<TdxQuoteClaims, QuoteRejected> {
    let need = HEADER + BODY;
    if quote.len() < need {
        return Err(QuoteRejected::TooShort {
            len: quote.len(),
            need,
        });
    }
    check_header(quote)?;
    let body = &quote[HEADER..HEADER + BODY];
    let mut claims = TdxQuoteClaims {
        td_attributes: [0; 8],
        mr_td: [0; 48],
        rtmr: [[0; 48]; 4],
        report_data: [0; 64],
        tee_type: TEE_TDX,
    };
    claims
        .td_attributes
        .copy_from_slice(&body[TD_ATTRIBUTES..TD_ATTRIBUTES + 8]);
    claims.mr_td.copy_from_slice(&body[MR_TD..MR_TD + 48]);
    for i in 0..4 {
        claims.rtmr[i].copy_from_slice(&body[RTMR0 + i * 48..RTMR0 + (i + 1) * 48]);
    }
    claims
        .report_data
        .copy_from_slice(&body[REPORT_DATA..REPORT_DATA + 64]);
    Ok(claims)
}

/// SECURITY: the nonce comparison is the anti-replay. A structurally perfect
/// quote with someone else's report_data is an OLD quote; accepting it would
/// let a compromised host replay a proof from before it was compromised.
pub fn verify(quote: &[u8], expected_nonce: &[u8; 64]) -> (Option<TdxQuoteClaims>, Verdict) {
    let claims = match parse(quote) {
        Ok(c) => c,
        Err(e) => {
            return (
                None,
                Verdict::Rejected {
                    reason: format!("quote unparseable: {e:?}"),
                    verifier: "tdx-structural".into(),
                },
            )
        }
    };
    if claims.is_debug() {
        return (
            Some(claims),
            Verdict::Rejected {
                reason: "TDATTRIBUTES.DEBUG is set: the host can read this TD's memory".into(),
                verifier: "tdx-structural".into(),
            },
        );
    }
    let refused = claims.refused_attributes();
    if refused != 0 {
        return (
            Some(claims),
            Verdict::Rejected {
                reason: format!(
                    "TDATTRIBUTES carries {}: not a production TD",
                    td_attribute_names(refused)
                ),
                verifier: "tdx-structural".into(),
            },
        );
    }
    if &claims.report_data != expected_nonce {
        return (
            Some(claims),
            Verdict::Rejected {
                reason: "report_data does not echo the fresh nonce (stale or replayed quote)"
                    .into(),
                verifier: "tdx-structural".into(),
            },
        );
    }
    let verdict = Verdict::Verified {
        identity: WorkloadIdentity {
            name: "x0ne-cc-prover".into(),
            environment: ExecutionEnvironment::ConfidentialVm {
                cpu: TeeTechnology::IntelTdx,
            },
            measurements: Vec::new(), // claims travel beside the verdict, not inside
        },
        verifier: "tdx-structural".into(),
        strength: VerifierStrength::Structural,
    };
    (Some(claims), verdict)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD Quote v4 offsets (Intel SGX/TDX Quote format): 48-byte header,
    /// then 584-byte TD report body; within the body, mr_td at [136..184],
    /// MROWNERCONFIG at [280..328], rtmr0..3 at [328..520], report_data at
    /// [520..584]. MROWNERCONFIG gets a sentinel so a shifted read shows.
    fn synthetic_quote(nonce: &[u8; 64]) -> Vec<u8> {
        let mut q = vec![0u8; 48 + 584 + 8]; // + qe sig len placeholder
        q[0] = 4; // version = 4 (little-endian u16)
        q[2] = 2; // att_key_type: ECDSA with P-256
        q[12..28].copy_from_slice(&INTEL_QE_VENDOR_ID);
        q[4..8].copy_from_slice(&0x81u32.to_le_bytes()); // tee_type TDX
        q[48 + 136..48 + 184].copy_from_slice(&[0xAB; 48]); // mr_td
        q[48 + 280..48 + 328].copy_from_slice(&[0xEE; 48]); // mr_owner_config
        for i in 0..4 {
            let at = 48 + 328 + i * 48;
            q[at..at + 48].copy_from_slice(&[0xC0 + i as u8; 48]); // rtmr i
        }
        q[48 + 520..48 + 584].copy_from_slice(nonce); // report_data
        q
    }

    /// v5 is REFUSED, never again read with v4 offsets. A failure
    /// would mean a parser claiming to understand a version whose layout it
    /// does not implement.
    #[test]
    fn a_v5_quote_is_refused_not_read_with_v4_offsets() {
        let mut q = synthetic_quote(&[7u8; 64]);
        q[0] = 5;
        assert!(matches!(
            parse(&q),
            Err(QuoteRejected::UnsupportedVersion { version: 5 })
        ));
    }

    /// The header says which key signs the quote, and the verifier reads a
    /// P-256 key and signature whatever it says: any other type is refused,
    /// and the refusal names the field.
    #[test]
    fn an_attestation_key_type_other_than_ecdsa_p256_is_refused() {
        for other in [0u16, 1, 3, 0x0200, 0x0102, u16::MAX] {
            let mut q = synthetic_quote(&[7u8; 64]);
            q[2..4].copy_from_slice(&other.to_le_bytes());
            let e = parse(&q).expect_err("another key type parses");
            assert_eq!(
                format!("{e:?}"),
                format!("UnsupportedAttestationKeyType {{ att_key_type: {other} }}")
            );
        }
    }

    /// The QE vendor of the header is Intel's, as in Intel's own verifier.
    #[test]
    fn a_qe_vendor_other_than_intel_is_refused() {
        for at in 12..28 {
            let mut q = synthetic_quote(&[7u8; 64]);
            q[at] ^= 1;
            let e = parse(&q).expect_err("another QE vendor parses");
            assert!(
                format!("{e:?}").starts_with("NotIntelQe { qe_vendor_id"),
                "{e:?}"
            );
        }
    }

    #[test]
    fn a_wellformed_quote_yields_its_claims() {
        let nonce = [7u8; 64];
        let claims = parse(&synthetic_quote(&nonce)).unwrap();
        assert_eq!(claims.tee_type, 0x81);
        assert_eq!(claims.mr_td, [0xAB; 48]);
        for i in 0..4 {
            assert_eq!(claims.rtmr[i], [0xC0 + i as u8; 48], "rtmr{i}");
        }
        assert_eq!(claims.report_data, nonce);
    }

    #[test]
    fn a_short_or_wrong_tee_quote_is_rejected_with_the_reason() {
        assert!(matches!(
            parse(&[0u8; 100]),
            Err(QuoteRejected::TooShort { .. })
        ));
        let mut q = synthetic_quote(&[0u8; 64]);
        q[4..8].copy_from_slice(&0u32.to_le_bytes()); // SGX, not TDX
        assert!(matches!(parse(&q), Err(QuoteRejected::NotTdx { .. })));
    }

    /// Anti-replay. A structurally perfect quote whose report_data does NOT
    /// carry our nonce is an old quote: Rejected. A failure of this test
    /// would mean a replayed proof is accepted.
    #[test]
    fn a_stale_nonce_is_rejected_however_wellformed_the_quote() {
        let quote = synthetic_quote(&[1u8; 64]);
        let (claims, verdict) = verify(&quote, &[2u8; 64]);
        assert!(!verdict.is_verified());
        assert!(
            claims.is_some(),
            "claims still reported for the failure report"
        );
    }

    /// SECURITY: TDATTRIBUTES bit 0 (DEBUG) set means the host can
    /// read the TD's memory. A fresh, well-formed quote from one is refused.
    #[test]
    fn a_debug_td_is_refused_however_fresh_the_quote() {
        let nonce = [9u8; 64];
        let mut q = synthetic_quote(&nonce);
        q[48 + 120] |= 1;
        let (_, verdict) = verify(&q, &nonce);
        assert!(!verdict.is_verified(), "debug TD verified: {verdict:?}");
    }

    fn with_attributes(attributes: u64, nonce: &[u8; 64]) -> Vec<u8> {
        let mut q = synthetic_quote(nonce);
        q[48 + 120..48 + 128].copy_from_slice(&attributes.to_le_bytes());
        q
    }

    fn reason(verdict: &Verdict) -> &str {
        match verdict {
            Verdict::Rejected { reason, .. } => reason,
            Verdict::Verified { .. } => "",
        }
    }

    /// SECURITY: Intel's TDX Module ABI (348551-008, table 3.23) calls a TD
    /// with any TUD bit (3:0) untrusted, and one with any TUP bit (15:4)
    /// profiled by the host. Each of the 16 bits alone refuses, by name.
    #[test]
    fn every_debug_or_profiling_bit_refuses() {
        let nonce = [9u8; 64];
        for bit in 0..16 {
            let name = match bit {
                0 => "DEBUG".to_string(),
                4 => "HGS_PLUS_PROF".into(),
                5 => "PERF_PROF".into(),
                6 => "PMT_PROF".into(),
                b => format!("bit {b}"),
            };
            let (_, verdict) = verify(&with_attributes(1 << bit, &nonce), &nonce);
            assert!(!verdict.is_verified(), "bit {bit} verified: {verdict:?}");
            assert!(reason(&verdict).contains(&name), "bit {bit}: {verdict:?}");
        }
    }

    /// SECURITY: a migratable TD may be moved to a platform this quote does
    /// not attest. Bits Intel reserves with a negative impact (26:23), and the
    /// reserved 55:32 and 61:56, which must be 0, are bits this verifier
    /// cannot judge. Each refuses.
    #[test]
    fn a_migratable_td_and_the_reserved_negative_bits_refuse() {
        let nonce = [9u8; 64];
        for (bit, name) in [
            (29, "MIGRATABLE"),
            (23, "bit 23"),
            (26, "bit 26"),
            (32, "bit 32"),
            (55, "bit 55"),
            (56, "bit 56"),
            (61, "bit 61"),
        ] {
            let attributes = (1 << 28) | (1u64 << bit);
            let (_, verdict) = verify(&with_attributes(attributes, &nonce), &nonce);
            assert!(!verdict.is_verified(), "bit {bit} verified: {verdict:?}");
            assert!(reason(&verdict).contains(name), "bit {bit}: {verdict:?}");
        }
    }

    /// The control: GCE's SEPT_VE_DISABLE alone (2026-10-17 and 10-18), the
    /// vendored public quote (plus PERFMON), each SEC bit Intel marks positive,
    /// TPA and PERFMON verify. The policy refuses no more than it says.
    #[test]
    fn the_attributes_of_real_tds_and_the_positive_bits_verify() {
        let nonce = [9u8; 64];
        let bits = [16, 17, 18, 19, 20, 21, 22, 27, 30, 31, 62, 63];
        let mut cases = vec![0, 1 << 28, (1 << 28) | (1 << 63)];
        cases.extend(bits.iter().map(|b| (1 << 28) | (1u64 << b)));
        for attributes in cases {
            let (_, verdict) = verify(&with_attributes(attributes, &nonce), &nonce);
            assert!(verdict.is_verified(), "{attributes:#x}: {verdict:?}");
        }
    }

    #[test]
    fn a_fresh_nonce_verifies_structurally() {
        let nonce = [9u8; 64];
        let (claims, verdict) = verify(&synthetic_quote(&nonce), &nonce);
        assert!(verdict.is_verified());
        assert_eq!(claims.unwrap().report_data, nonce);
    }
}
