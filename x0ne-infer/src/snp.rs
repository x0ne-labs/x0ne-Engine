//! Cryptographic verification of an AMD SEV-SNP attestation report.
//!
//! The chains, as AMD's Key Distribution Service defines them (VCEK
//! specification 57230, and the ABI 56860 rev. 1.58 for the report):
//!
//! ```text
//! ARK-<product>  (AMD root, RSA-4096, self-signed; pinned here by SPKI)
//!   ├─ signs → ASK, "SEV-<product>"  (RSA-4096, RSASSA-PSS SHA-384, salt 48; pinned)
//!   │    └─ signs → VCEK  (P-384, one per chip and TCB; its extensions name
//!   │                      the product, the chip and the TCB it was derived for)
//!   ├─ signs → ASVK, "SEV-VLEK-<product>"  (RSA-4096, same scheme; pinned)
//!   │    └─ signs → VLEK  (P-384, one per cloud provider and TCB; names the
//!   │                      provider, `cspId`, and no chip)
//!   └─ signs → the product's CRL  (lists a revoked ASK or ASVK)
//! VCEK or VLEK ── signs → ATTESTATION_REPORT[0x000..0x2A0]  (ECDSA P-384, SHA-384)
//! ```
//!
//! Every arrow is checked here, in order, fail-closed, and every certificate
//! against its own validity window. Then the report must describe the TCB its
//! key was issued for (`REPORTED_TCB` = its SPL extensions) and, for a VCEK,
//! the chip (`CHIP_ID` = its hwID); and the guest must be one this crate
//! serves on: VMPL 0, no DEBUG, no migration agent, SMT as the written policy
//! says ([`SnpSmt`], forbidden by default), no policy, key or platform bit
//! this crate does not know, and every TCB the report carries at or above a
//! minimum ([`SnpPolicy`]: per CPU line from AMD's security bulletins,
//! [`default_minimum`], raised component by component by the written floor
//! of the image reference). A VLEK is accepted only from a cloud provider
//! the policy names; none by default.
//!
//! Milan is refused unless the caller says otherwise (`allow_milan`): a 2026
//! preprint (Shen and Qin, arXiv 2605.12990) describes forging Milan
//! attestation reports. Genoa and Turin are accepted.
//!
//! Revocation: [`verify_snp_report`] also verifies AMD's CRL for the
//! report's product (signed by the pinned ARK, inside its thisUpdate to
//! nextUpdate window) and refuses a report whose ASK or ASVK it lists. The
//! relay and the guest fetch that CRL from AMD KDS at each verdict. A VCEK
//! or a VLEK has no revocation list of its own.
//!
//! The evidence payload is the report (0x4A0 bytes) followed by the VCEK's or
//! VLEK's DER ([`evidence_payload`]). The certificate comes from wherever the
//! guest got it (the host's extended report, AMD KDS): its signature by a
//! pinned ASK or ASVK is what makes it AMD's, never its source.
//!
//! What this module does NOT do, said in every verdict ([`VerifiedSnp`]'s
//! `limitations`): judge the boot loader, TEE and FMC levels beyond the
//! written floor, the mitigation vectors of a v5 report or the guest policy
//! bits other than DEBUG, MIGRATE_MA and SMT, or attest a GPU.
//!
//! [`SnpPolicy`]: crate::snp::SnpPolicy
//! [`SnpSmt`]: crate::snp::SnpSmt
//! [`default_minimum`]: crate::snp::default_minimum
//! [`evidence_payload`]: crate::snp::evidence_payload
//! [`verify_snp_report`]: crate::snp::verify_snp_report
//! [`VerifiedSnp`]: crate::snp::VerifiedSnp

use rsa::pkcs8::DecodePublicKey as _;
use rsa::signature::Verifier as _;
use rsa::RsaPublicKey;
use sha2::{Digest, Sha256};
use x0ne_confidential::model::Measurement;
use x0ne_confidential::policy::ExpectedMeasurement;
use x0ne_confidential::snp_report::{
    self, Cpuid, SnpReport, TcbLayout, TcbParts, REPORT_DATA_AT, REPORT_LEN, SIGNED_LEN,
};
use x509_parser::oid_registry::{OID_NIST_HASH_SHA384, OID_PKCS1_RSASSAPSS};
use x509_parser::prelude::{
    AlgorithmIdentifier, CertificateRevocationList, FromDer, X509Certificate, X509Version,
};
use x509_parser::signature_algorithm::SignatureAlgorithm;

// ── The pinned AMD roots ─────────────────────────────────────────────────────

/// SHA-256 of the DER SubjectPublicKeyInfo of each ARK and ASK.
///
/// MEASURED, not recalled: fetched from AMD KDS
/// `https://kdsintf.amd.com/vcek/v1/{Milan,Genoa,Turin}/cert_chain` on
/// 2026-10-22 (ASK first, ARK second in each file), byte-identical to the
/// `ark.pem`/`ask.pem` virtee/sev ships (commit a966d06), and digested with
/// `openssl x509 -pubkey -noout | openssl pkey -pubin -outform der | sha256sum`.
/// The certificates themselves are embedded below (tests/snp/amd_*.der) and
/// checked against these pins at every verification
/// (`tests::the_production_pins_match_amds_kds_certificates`).
pub const AMD_ARK_MILAN_SPKI_SHA256_HEX: &str =
    "9f056bee44377e29308cb5ffa895bdfb62d18881fa6bed8d6f075b0204089cb9";
pub const AMD_ASK_MILAN_SPKI_SHA256_HEX: &str =
    "42491669ccb851d627ea9846cd24eccbeb7de7c2a62740120fff88001d38ebb6";
pub const AMD_ARK_GENOA_SPKI_SHA256_HEX: &str =
    "429a69c9422aa258ee4d8db5fcda9c6470ef15f8cd5a9cebd6cbc7d90b863831";
pub const AMD_ASK_GENOA_SPKI_SHA256_HEX: &str =
    "0324d55bcd48ba877a8fd87c0e1b4934dec8448e3807a24384a34dc6a6c9ebb8";
pub const AMD_ARK_TURIN_SPKI_SHA256_HEX: &str =
    "4f125410563a2ab9a50356f9243f6fe0b6f73de98603f53f90339c70e9d7ad08";
pub const AMD_ASK_TURIN_SPKI_SHA256_HEX: &str =
    "0000ead352025f208bf55297c4192c34cd1b878002107366d8f2065a9ae7d3a9";

/// SHA-256 of the DER SubjectPublicKeyInfo of each ASVK, the key that signs
/// cloud providers' VLEKs.
///
/// MEASURED: fetched from AMD KDS
/// `https://kdsintf.amd.com/vlek/v1/{Milan,Genoa,Turin}/cert_chain` on
/// 2026-10-28 (ASVK first, then the ARK, byte-identical to the ARK above),
/// digested as above.
pub const AMD_ASVK_MILAN_SPKI_SHA256_HEX: &str =
    "66450a123ea5876206df67f2514a4fea224c0f63b18313423078b0edc4535e33";
pub const AMD_ASVK_GENOA_SPKI_SHA256_HEX: &str =
    "7d1633252ebbadaa75f96e9b4e6f7d43a90ceb3f843937482630f8da2459264e";
pub const AMD_ASVK_TURIN_SPKI_SHA256_HEX: &str =
    "bfc441408b2f806c5dbf180bd8920e6e60dd404d82141780d85dbf962954d386";

pub const AMD_ARK_MILAN_DER: &[u8] = include_bytes!("../tests/snp/amd_milan_ark.der");
pub const AMD_ASK_MILAN_DER: &[u8] = include_bytes!("../tests/snp/amd_milan_ask.der");
pub const AMD_ASVK_MILAN_DER: &[u8] = include_bytes!("../tests/snp/amd_milan_asvk.der");
pub const AMD_ARK_GENOA_DER: &[u8] = include_bytes!("../tests/snp/amd_genoa_ark.der");
pub const AMD_ASK_GENOA_DER: &[u8] = include_bytes!("../tests/snp/amd_genoa_ask.der");
pub const AMD_ASVK_GENOA_DER: &[u8] = include_bytes!("../tests/snp/amd_genoa_asvk.der");
pub const AMD_ARK_TURIN_DER: &[u8] = include_bytes!("../tests/snp/amd_turin_ark.der");
pub const AMD_ASK_TURIN_DER: &[u8] = include_bytes!("../tests/snp/amd_turin_ask.der");
pub const AMD_ASVK_TURIN_DER: &[u8] = include_bytes!("../tests/snp/amd_turin_asvk.der");

/// AMD's Key Distribution Service: the CRLs, by product and key kind
/// ([`kds_crl_url`]).
pub const KDS_BASE_URL: &str = "https://kdsintf.amd.com";

/// RSASSA-PSS salt of every AMD SEV certificate signature (SHA-384 → 48).
const PSS_SALT_LEN: usize = 48;
/// AMD's arc for the VCEK and VLEK extensions (57230).
const AMD_OID: &str = "1.3.6.1.4.1.3704.1.";

/// Guest policy bits 0..=25 are defined by 56860 rev. 1.58 (Table 9); a
/// higher bit is one this crate cannot judge.
const POLICY_KNOWN: u64 = (1 << 26) - 1;
/// `KEY_INFO`: AUTHOR_KEY_EN, MASK_CHIP_KEY, SIGNING_KEY (bits 4:0).
const KEY_INFO_KNOWN: u32 = 0b1_1111;
/// `PLATFORM_INFO`: SMT_EN, TSME_EN, ECC_EN, RAPL_DIS, CIPHERTEXT_HIDING_EN,
/// ALIAS_CHECK_COMPLETE (bits 5:0).
const PLATFORM_INFO_KNOWN: u64 = 0b11_1111;
/// `PLATFORM_INFO.SMT_EN`: SMT is enabled on the platform.
const PLATFORM_INFO_SMT_EN: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SnpProduct {
    /// 3rd generation EPYC (family 19h, models 00h-0Fh).
    Milan,
    /// 4th generation EPYC: Genoa, Genoa-X, Bergamo, Siena (family 19h,
    /// models 10h-1Fh and A0h-AFh).
    Genoa,
    /// 5th generation EPYC (family 1Ah, models 00h-11h).
    Turin,
}

impl SnpProduct {
    pub fn name(self) -> &'static str {
        match self {
            Self::Milan => "Milan",
            Self::Genoa => "Genoa",
            Self::Turin => "Turin",
        }
    }

    /// `Milan`, `Genoa`, `Turin`, as a written policy names them.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Milan" => Some(Self::Milan),
            "Genoa" => Some(Self::Genoa),
            "Turin" => Some(Self::Turin),
            _ => None,
        }
    }

    /// The certificate's productName (`1.3.6.1.4.1.3704.1.2`): "Milan-B0",
    /// "Genoa", "Turin" in the VCEKs in hand, "Milan" in the VLEK; a
    /// stepping may follow a dash.
    fn from_vcek_product_name(name: &str) -> Option<Self> {
        match name.split('-').next() {
            Some(n) => Self::from_name(n),
            None => None,
        }
    }

    pub fn tcb_layout(self) -> TcbLayout {
        match self {
            Self::Milan | Self::Genoa => TcbLayout::MilanGenoa,
            Self::Turin => TcbLayout::Turin,
        }
    }

    /// virtee/sev (a966d06, `src/lib.rs`) maps CPUID to product the same way.
    fn cpuid_matches(self, c: Cpuid) -> bool {
        match self {
            Self::Milan => c.family == 0x19 && c.model <= 0x0F,
            Self::Genoa => {
                c.family == 0x19
                    && ((0x10..=0x1F).contains(&c.model) || (0xA0..=0xAF).contains(&c.model))
            }
            Self::Turin => c.family == 0x1A && c.model <= 0x11,
        }
    }
}

/// The key that signed a report (`KEY_INFO.SIGNING_KEY`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SnpSigningKey {
    /// Versioned Chip Endorsement Key: one per chip and TCB, under the ASK.
    Vcek,
    /// Versioned Loaded Endorsement Key: one per cloud provider and TCB,
    /// under the ASVK.
    Vlek,
}

impl SnpSigningKey {
    pub fn name(self) -> &'static str {
        match self {
            Self::Vcek => "VCEK",
            Self::Vlek => "VLEK",
        }
    }

    /// The ARK-signed certificate between the root and the key.
    pub fn intermediate(self) -> &'static str {
        match self {
            Self::Vcek => "ASK",
            Self::Vlek => "ASVK",
        }
    }

    fn kds_path(self) -> &'static str {
        match self {
            Self::Vcek => "vcek",
            Self::Vlek => "vlek",
        }
    }
}

/// AMD KDS's CRL for one product. The `vcek` and `vlek` paths served the
/// same file for each product on 2026-10-28 (tests/VENDORED.md).
pub fn kds_crl_url(base: &str, product: SnpProduct, key: SnpSigningKey) -> String {
    format!(
        "{}/{}/v1/{}/crl",
        base.trim_end_matches('/'),
        key.kds_path(),
        product.name()
    )
}

/// One product's pinned chains: the certificates, and the SPKI digests they
/// must carry.
#[derive(Debug, Clone)]
pub struct SnpRoot {
    pub product: SnpProduct,
    pub ark_der: &'static [u8],
    pub ask_der: &'static [u8],
    pub asvk_der: &'static [u8],
    pub ark_spki_sha256_hex: String,
    pub ask_spki_sha256_hex: String,
    pub asvk_spki_sha256_hex: String,
}

/// The AMD roots a verdict may end on, and where their CRLs are published.
/// Production values only through [`SnpAnchors::production`].
#[derive(Debug, Clone)]
pub struct SnpAnchors {
    pub roots: Vec<SnpRoot>,
    pub kds_base_url: String,
}

impl SnpAnchors {
    /// Milan, Genoa and Turin, as AMD KDS served them on 2026-10-22 (ARK,
    /// ASK) and 2026-10-28 (ASVK). Milan is pinned like the others and
    /// refused by [`SnpPolicy`] unless allowed: the chain is AMD's either
    /// way, what is in doubt is the chip.
    pub fn production() -> Self {
        let root = |product, ark_der, ask_der, asvk_der, pins: [&str; 3]| SnpRoot {
            product,
            ark_der,
            ask_der,
            asvk_der,
            ark_spki_sha256_hex: pins[0].into(),
            ask_spki_sha256_hex: pins[1].into(),
            asvk_spki_sha256_hex: pins[2].into(),
        };
        Self {
            roots: vec![
                root(
                    SnpProduct::Milan,
                    AMD_ARK_MILAN_DER,
                    AMD_ASK_MILAN_DER,
                    AMD_ASVK_MILAN_DER,
                    [
                        AMD_ARK_MILAN_SPKI_SHA256_HEX,
                        AMD_ASK_MILAN_SPKI_SHA256_HEX,
                        AMD_ASVK_MILAN_SPKI_SHA256_HEX,
                    ],
                ),
                root(
                    SnpProduct::Genoa,
                    AMD_ARK_GENOA_DER,
                    AMD_ASK_GENOA_DER,
                    AMD_ASVK_GENOA_DER,
                    [
                        AMD_ARK_GENOA_SPKI_SHA256_HEX,
                        AMD_ASK_GENOA_SPKI_SHA256_HEX,
                        AMD_ASVK_GENOA_SPKI_SHA256_HEX,
                    ],
                ),
                root(
                    SnpProduct::Turin,
                    AMD_ARK_TURIN_DER,
                    AMD_ASK_TURIN_DER,
                    AMD_ASVK_TURIN_DER,
                    [
                        AMD_ARK_TURIN_SPKI_SHA256_HEX,
                        AMD_ASK_TURIN_SPKI_SHA256_HEX,
                        AMD_ASVK_TURIN_SPKI_SHA256_HEX,
                    ],
                ),
            ],
            kds_base_url: KDS_BASE_URL.into(),
        }
    }

    fn root(&self, product: SnpProduct) -> Result<&SnpRoot, SnpRejected> {
        self.roots
            .iter()
            .find(|r| r.product == product)
            .ok_or_else(|| SnpRejected::NoRoot(product.name().into()))
    }
}

// ── Policy ───────────────────────────────────────────────────────────────────

/// The lowest TCB a report may carry, component by component, in every TCB
/// field it has (`REPORTED_TCB`, `CURRENT_TCB`, `COMMITTED_TCB`,
/// `LAUNCH_TCB`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SnpTcbMinimum {
    /// Turin only (Milan and Genoa have no FMC component; theirs reads 0).
    pub fmc: u8,
    pub boot_loader: u8,
    pub tee: u8,
    pub snp: u8,
    pub microcode: u8,
    /// `PLATFORM_INFO.ALIAS_CHECK_COMPLETE` required: the firmware looked
    /// for aliased DRAM addresses since reset and found none (BadRAM,
    /// AMD-SB-3015, Milan and Genoa).
    pub require_alias_check: bool,
}

impl SnpTcbMinimum {
    /// Each component at least `floor`'s: a written floor raises, never lowers.
    #[must_use]
    pub fn raised_to(self, floor: SnpTcbMinimum) -> Self {
        Self {
            fmc: self.fmc.max(floor.fmc),
            boot_loader: self.boot_loader.max(floor.boot_loader),
            tee: self.tee.max(floor.tee),
            snp: self.snp.max(floor.snp),
            microcode: self.microcode.max(floor.microcode),
            require_alias_check: self.require_alias_check || floor.require_alias_check,
        }
    }
}

/// SMT, as the written policy says.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SnpSmt {
    /// `POLICY.SMT` clear (the firmware refuses to launch the guest while SMT
    /// is on) and `PLATFORM_INFO.SMT_EN` clear.
    #[default]
    Forbidden,
    /// Either way; every verdict says a sibling thread may run host code.
    Allowed,
}

/// What a caller accepts beyond the fixed rules (VMPL 0, no DEBUG, no
/// migration agent, no bit this crate does not know).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SnpPolicy {
    /// Accept a Milan chip. Off by default: arXiv 2605.12990 (Shen and
    /// Qin, 2026) describes forging Milan attestation reports.
    pub allow_milan: bool,
    /// `Some`: only these products (the written policy's list). `None`:
    /// any pinned product, Milan still under `allow_milan`.
    pub products: Option<Vec<SnpProduct>>,
    /// `None`: the default of the report's CPU line ([`default_minimum`]).
    /// `Some`: this minimum, for every line — an explicit judgement.
    pub minimum: Option<SnpTcbMinimum>,
    /// The written floor (the image reference's `tcb_minimum`): it raises
    /// the minimum above component by component, never lowers it.
    pub floor: Option<SnpTcbMinimum>,
    pub smt: SnpSmt,
    /// The cloud providers (the `cspId` their VLEK names) whose VLEK-signed
    /// reports are accepted. Empty, the default: VCEK only.
    pub vlek_csp_ids: Vec<String>,
    /// Which bulletins the default minimum imposes ([`mode_minimum`]):
    /// `Strict` (the default of a policy that names no mode) every one,
    /// `Transparent` those older than [`SNP_BULLETIN_GRACE_SECS`]. Ignored
    /// under an explicit `minimum`.
    pub tcb_mode: crate::tcb::TcbMode,
}

/// The default minimum TCB of one CPU line, and where it comes from.
///
/// Sources, AMD product security bulletins as published on 2026-10-22:
/// SB-3020 (CVE-2025-0033, RMPocalypse, rev. 2025-10-13) gives the SNP
/// firmware SPL (Milan and Genoa 0x1B, Turin 0x04) and a microcode level;
/// SB-3027 (CVE-2025-29943, StackWarp, rev. 2026-02-23) gives
/// "TCB\[MICROCODE\] >=" per line, the higher of the two (Milan DE, Milan-X
/// 47, Genoa 56, Genoa-X 51, Bergamo 1B, Turin C1 51, Turin Dense 4E).
/// SB-3015 (BadRAM, CVE-2024-21944) names Milan and Genoa, whence the alias
/// check; Turin is not in its list. No bulletin in hand gives a boot loader,
/// TEE or FMC level: those default to 0, which judges nothing, and every
/// verdict says so.
///
/// The CPU line comes from the report's CPUID (v3+), as the bulletins'
/// CPUID table (SB-3019) names them: Milan 01h stepping 1, Milan-X 01h/2,
/// Genoa 11h/1, Genoa-X 11h/2, Bergamo and Siena A0h/2, Turin 02h/1; Turin
/// Dense 11h/0 is read from its microcode patch ID (0x0B10104E). A v2 report
/// carries no CPUID: the product's highest level applies, which may refuse
/// a genuine Bergamo — fail-closed. A CPUID outside these lines has no
/// default and is refused until the caller passes a minimum.
pub fn default_minimum(
    product: SnpProduct,
    cpuid: Option<Cpuid>,
) -> Result<(SnpTcbMinimum, &'static str), SnpRejected> {
    // (model, stepping, microcode, source); the first row is the highest
    // microcode level of the product, the one a v2 report is held to.
    type Line = (u8, u8, u8, &'static str);
    let (snp, require_alias_check, lines, v2): (u8, bool, &[Line], &'static str) = match product {
        SnpProduct::Milan => (
            0x1B,
            true,
            &[
                (
                    0x01,
                    1,
                    0xDE,
                    "default, Milan (AMD-SB-3020, AMD-SB-3027, AMD-SB-3015)",
                ),
                (
                    0x01,
                    2,
                    0x47,
                    "default, Milan-X (AMD-SB-3020, AMD-SB-3027, AMD-SB-3015)",
                ),
            ],
            "default, Milan, CPU line unknown in a v2 report: the product's highest level",
        ),
        SnpProduct::Genoa => (
            0x1B,
            true,
            &[
                (
                    0x11,
                    1,
                    0x56,
                    "default, Genoa (AMD-SB-3020, AMD-SB-3027, AMD-SB-3015)",
                ),
                (
                    0x11,
                    2,
                    0x51,
                    "default, Genoa-X (AMD-SB-3020, AMD-SB-3027, AMD-SB-3015)",
                ),
                (
                    0xA0,
                    2,
                    0x1B,
                    "default, Bergamo or Siena (AMD-SB-3020, AMD-SB-3027, AMD-SB-3015)",
                ),
            ],
            "default, Genoa, CPU line unknown in a v2 report: the product's highest level",
        ),
        SnpProduct::Turin => (
            0x04,
            false,
            &[
                (0x02, 1, 0x51, "default, Turin (AMD-SB-3020, AMD-SB-3027)"),
                (
                    0x11,
                    0,
                    0x4E,
                    "default, Turin Dense (AMD-SB-3020, AMD-SB-3027)",
                ),
            ],
            "default, Turin, CPU line unknown in a v2 report: the product's highest level",
        ),
    };
    let (microcode, source) = match cpuid {
        None => (lines[0].2, v2),
        Some(c) => lines
            .iter()
            .find(|(model, stepping, _, _)| *model == c.model && *stepping == c.stepping)
            .map(|(_, _, microcode, source)| (*microcode, *source))
            .ok_or_else(|| {
                SnpRejected::NoDefaultMinimum(format!(
                    "{} family {:#04x} model {:#04x} stepping {}",
                    product.name(),
                    c.family,
                    c.model,
                    c.stepping
                ))
            })?,
    };
    Ok((
        SnpTcbMinimum {
            fmc: 0,
            boot_loader: 0,
            tee: 0,
            snp,
            microcode,
            require_alias_check,
        },
        source,
    ))
}

/// One AMD product security bulletin the default minimum is built from,
/// dated by its first publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnpBulletin {
    pub id: &'static str,
    pub cve: &'static str,
    pub name: &'static str,
    /// `YYYY-MM-DD`, and that day at 00:00 UTC.
    pub published: &'static str,
    pub published_unix: u64,
}

/// BadRAM, disclosed in December 2024 (badram.eu, IEEE S&P 2025). The day
/// was not re-read on AMD's page for this constant; the bulletin is past
/// [`SNP_BULLETIN_GRACE_SECS`] by ten months either way.
pub const AMD_SB_3015: SnpBulletin = SnpBulletin {
    id: "AMD-SB-3015",
    cve: "CVE-2024-21944",
    name: "BadRAM",
    published: "2024-12-10",
    published_unix: 1_733_788_800,
};
/// RMPocalypse: AMD's bulletin of 2025-10-13 (read 2026-10-22).
pub const AMD_SB_3020: SnpBulletin = SnpBulletin {
    id: "AMD-SB-3020",
    cve: "CVE-2025-0033",
    name: "RMPocalypse",
    published: "2025-10-13",
    published_unix: 1_760_313_600,
};
/// StackWarp: published 2026-01-15, revised 2026-02-23 (read 2026-10-22).
pub const AMD_SB_3027: SnpBulletin = SnpBulletin {
    id: "AMD-SB-3027",
    cve: "CVE-2025-29943",
    name: "StackWarp",
    published: "2026-01-15",
    published_unix: 1_768_435_200,
};

/// How long a bulletin is named before `Transparent` imposes it: a year,
/// as Intel moves its `standard` TCB about 12 months after `early`
/// (decision of 2026-10-28).
pub const SNP_BULLETIN_GRACE_SECS: u64 = 365 * 86_400;

/// What one bulletin asks of one CPU line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BulletinFloor {
    pub bulletin: SnpBulletin,
    pub minimum: SnpTcbMinimum,
    /// Set when a part of `minimum` is another bulletin's level, because
    /// this build does not hold this bulletin's own for the line.
    pub note: Option<&'static str>,
}

/// AMD-SB-3020 also gives a microcode level per line; only Bergamo and
/// Siena's (0x1B, the same as AMD-SB-3027's) is in hand. Elsewhere the
/// higher AMD-SB-3027 level stands in: fail-closed.
const SB_3020_MICROCODE_FROM_SB_3027: &str =
    "microcode: AMD-SB-3027's level; AMD-SB-3020's own for this CPU line is not in this build";

/// One CPU line: (model, stepping, name, AMD-SB-3020 microcode if in hand,
/// AMD-SB-3027 microcode). The first row of a product is its highest level.
type BulletinLine = (u8, u8, &'static str, Option<u8>, u8);

/// The CPU line's name and the floor each bulletin sets on it: the same
/// lines and sources as [`default_minimum`], which is their component-wise
/// maximum. AMD-SB-3015 sets the alias check (Milan and Genoa families),
/// AMD-SB-3020 the SNP firmware SPL, AMD-SB-3027 the microcode.
pub fn bulletin_floors(
    product: SnpProduct,
    cpuid: Option<Cpuid>,
) -> Result<(String, Vec<BulletinFloor>), SnpRejected> {
    let (snp, alias, lines): (u8, bool, &[BulletinLine]) = match product {
        SnpProduct::Milan => (
            0x1B,
            true,
            &[
                (0x01, 1, "Milan", None, 0xDE),
                (0x01, 2, "Milan-X", None, 0x47),
            ],
        ),
        SnpProduct::Genoa => (
            0x1B,
            true,
            &[
                (0x11, 1, "Genoa", None, 0x56),
                (0x11, 2, "Genoa-X", None, 0x51),
                (0xA0, 2, "Bergamo or Siena", Some(0x1B), 0x1B),
            ],
        ),
        SnpProduct::Turin => (
            0x04,
            false,
            &[
                (0x02, 1, "Turin", None, 0x51),
                (0x11, 0, "Turin Dense", None, 0x4E),
            ],
        ),
    };
    let (name, sb3020_microcode, sb3027_microcode) = match cpuid {
        None => (
            format!(
                "{}, CPU line unknown in a v2 report: the product's highest level",
                product.name()
            ),
            lines[0].3,
            lines[0].4,
        ),
        Some(c) => lines
            .iter()
            .find(|(model, stepping, ..)| *model == c.model && *stepping == c.stepping)
            .map(|(_, _, name, own, sb3027)| (name.to_string(), *own, *sb3027))
            .ok_or_else(|| {
                SnpRejected::NoDefaultMinimum(format!(
                    "{} family {:#04x} model {:#04x} stepping {}",
                    product.name(),
                    c.family,
                    c.model,
                    c.stepping
                ))
            })?,
    };
    let zero = SnpTcbMinimum {
        fmc: 0,
        boot_loader: 0,
        tee: 0,
        snp: 0,
        microcode: 0,
        require_alias_check: false,
    };
    let mut floors = Vec::new();
    if alias {
        floors.push(BulletinFloor {
            bulletin: AMD_SB_3015,
            minimum: SnpTcbMinimum {
                require_alias_check: true,
                ..zero
            },
            note: None,
        });
    }
    floors.push(BulletinFloor {
        bulletin: AMD_SB_3020,
        minimum: SnpTcbMinimum {
            snp,
            microcode: sb3020_microcode.unwrap_or(sb3027_microcode),
            ..zero
        },
        note: sb3020_microcode
            .is_none()
            .then_some(SB_3020_MICROCODE_FROM_SB_3027),
    });
    floors.push(BulletinFloor {
        bulletin: AMD_SB_3027,
        minimum: SnpTcbMinimum {
            microcode: sb3027_microcode,
            ..zero
        },
        note: None,
    });
    Ok((name, floors))
}

/// True when `mode` imposes `bulletin` at `now_unix`: `Strict` always,
/// `Transparent` once it is [`SNP_BULLETIN_GRACE_SECS`] old.
pub fn bulletin_imposed(bulletin: &SnpBulletin, mode: crate::tcb::TcbMode, now_unix: u64) -> bool {
    match mode {
        crate::tcb::TcbMode::Strict => true,
        crate::tcb::TcbMode::Transparent => {
            now_unix
                >= bulletin
                    .published_unix
                    .saturating_add(SNP_BULLETIN_GRACE_SECS)
        }
    }
}

/// The default minimum of one CPU line under `mode` at `now_unix`, its
/// source, the line's name and each bulletin's floor with whether it is
/// imposed. `Strict` is [`default_minimum`], same values, same source.
#[allow(clippy::type_complexity)]
pub fn mode_minimum(
    product: SnpProduct,
    cpuid: Option<Cpuid>,
    mode: crate::tcb::TcbMode,
    now_unix: u64,
) -> Result<(SnpTcbMinimum, String, String, Vec<(BulletinFloor, bool)>), SnpRejected> {
    let (line, floors) = bulletin_floors(product, cpuid)?;
    let floors: Vec<(BulletinFloor, bool)> = floors
        .into_iter()
        .map(|f| {
            let imposed = bulletin_imposed(&f.bulletin, mode, now_unix);
            (f, imposed)
        })
        .collect();
    if mode == crate::tcb::TcbMode::Strict {
        let (minimum, source) = default_minimum(product, cpuid)?;
        return Ok((minimum, source.to_string(), line, floors));
    }
    let mut minimum = SnpTcbMinimum {
        fmc: 0,
        boot_loader: 0,
        tee: 0,
        snp: 0,
        microcode: 0,
        require_alias_check: false,
    };
    for (f, imposed) in &floors {
        if *imposed {
            minimum = minimum.raised_to(f.minimum);
        }
    }
    let ids = |imposed: bool| -> Vec<&str> {
        floors
            .iter()
            .filter(|(_, i)| *i == imposed)
            .map(|(f, _)| f.bulletin.id)
            .collect()
    };
    let (on, named) = (ids(true), ids(false));
    let mut source = format!("default, {line}, transparent TCB policy: ");
    source.push_str(&if on.is_empty() {
        "no bulletin imposed".to_string()
    } else {
        format!("{} imposed", on.join(", "))
    });
    if !named.is_empty() {
        source.push_str(&format!(
            "; {} named, not imposed (under a year old)",
            named.join(", ")
        ));
    }
    Ok((minimum, source, line, floors))
}

/// One bulletin against one verified report: imposed or only named, and
/// what the lowest of the report's four TCBs misses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BulletinStanding {
    pub bulletin: SnpBulletin,
    pub imposed: bool,
    pub minimum: SnpTcbMinimum,
    pub note: Option<&'static str>,
    /// `"microcode 0x15 < 0x1b"`, `"alias check not complete"`: empty when met.
    pub not_met: Vec<String>,
}

/// The component-wise lowest of `REPORTED_TCB`, `CURRENT_TCB`,
/// `COMMITTED_TCB` and `LAUNCH_TCB`: the level a minimum is held to.
pub fn lowest_tcb(report: &SnpReport, product: SnpProduct) -> TcbParts {
    let layout = product.tcb_layout();
    let parts = [
        report.reported_tcb,
        report.current_tcb,
        report.committed_tcb,
        report.launch_tcb,
    ]
    .map(|raw| TcbParts::decode(raw, layout));
    TcbParts {
        fmc: parts.iter().map(|t| t.fmc).min().unwrap_or(0),
        boot_loader: parts.iter().map(|t| t.boot_loader).min().unwrap_or(0),
        tee: parts.iter().map(|t| t.tee).min().unwrap_or(0),
        snp: parts.iter().map(|t| t.snp).min().unwrap_or(0),
        microcode: parts.iter().map(|t| t.microcode).min().unwrap_or(0),
    }
}

/// What `lowest` (and the alias check) misses of `minimum`, part by part.
pub fn tcb_gaps(
    minimum: &SnpTcbMinimum,
    lowest: &TcbParts,
    alias_check_complete: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    for (part, got, min) in [
        ("FMC", lowest.fmc, minimum.fmc),
        ("boot loader", lowest.boot_loader, minimum.boot_loader),
        ("TEE", lowest.tee, minimum.tee),
        ("SNP", lowest.snp, minimum.snp),
        ("microcode", lowest.microcode, minimum.microcode),
    ] {
        if got < min {
            out.push(format!("{part} {got:#04x} < {min:#04x}"));
        }
    }
    if minimum.require_alias_check && !alias_check_complete {
        out.push("alias check not complete (PLATFORM_INFO.ALIAS_CHECK_COMPLETE clear)".into());
    }
    out
}

/// Each bulletin of the report's CPU line under `mode` at `now_unix`,
/// judged on the report as it is, whether or not a verdict would accept
/// it: what a refused report would have to reach.
pub fn bulletin_standings(
    report: &SnpReport,
    product: SnpProduct,
    mode: crate::tcb::TcbMode,
    now_unix: u64,
) -> Result<(String, Vec<BulletinStanding>), SnpRejected> {
    let (_, _, line, floors) = mode_minimum(product, report.cpuid, mode, now_unix)?;
    let lowest = lowest_tcb(report, product);
    let alias = report.alias_check_complete();
    Ok((
        line,
        floors
            .into_iter()
            .map(|(f, imposed)| BulletinStanding {
                bulletin: f.bulletin,
                imposed,
                minimum: f.minimum,
                note: f.note,
                not_met: tcb_gaps(&f.minimum, &lowest, alias),
            })
            .collect(),
    ))
}

// ── Errors and results ───────────────────────────────────────────────────────

/// Why a report was refused. One variant per kind of link, so a refusal
/// names what broke.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SnpRejected {
    #[error("SNP evidence: {0}")]
    Evidence(String),
    #[error("SNP report: {0}")]
    Report(String),
    #[error("SNP signing key: {0}")]
    SigningKey(String),
    #[error("VCEK: {0}")]
    Vcek(String),
    #[error("VLEK: {0}")]
    Vlek(String),
    #[error("no pinned AMD root for product {0}")]
    NoRoot(String),
    #[error("{0} root does not match its pinned SPKI digest")]
    RootNotPinned(String),
    #[error("AMD chain: {0}")]
    Chain(String),
    #[error("AMD CRL: {0}")]
    Crl(String),
    #[error("revoked by AMD's CRL: {0}")]
    Revoked(String),
    #[error("the report signature does not verify under its VCEK or VLEK")]
    ReportSignature,
    #[error("the signing key was not issued for this report: {0}")]
    KeyMismatch(String),
    #[error("CPUID: {0}")]
    Cpuid(String),
    #[error("Milan refused: arXiv 2605.12990 describes forging Milan attestation reports; allow it explicitly to accept one")]
    MilanRefused,
    #[error("product {0} is not one the policy names")]
    ProductRefused(String),
    #[error("{0}")]
    Vmpl(String),
    #[error(
        "POLICY bit 17 is clear: the firmware sets it on every guest, this report is malformed"
    )]
    PolicyReserved,
    #[error("POLICY.DEBUG is set: the host can read and write this guest's memory")]
    Debug,
    #[error("migration agent: {0}")]
    MigrationAgent(String),
    #[error("SMT: {0}")]
    Smt(String),
    #[error("a bit this verifier does not know: {0}")]
    Unknown(String),
    #[error("no default TCB minimum for {0}: pass one explicitly")]
    NoDefaultMinimum(String),
    #[error("TCB below the minimum: {0}")]
    TcbTooLow(String),
    #[error("PLATFORM_INFO.ALIAS_CHECK_COMPLETE is clear: no proof the firmware ruled out aliased DRAM (BadRAM, AMD-SB-3015)")]
    AliasCheck,
}

/// What a verified report established, with its stated limitations.
#[derive(Debug, Clone)]
pub struct VerifiedSnp {
    pub product: SnpProduct,
    pub signing_key: SnpSigningKey,
    /// The cloud provider a VLEK names (`cspId`); `None` for a VCEK.
    pub csp_id: Option<String>,
    /// Lowercase-hex SHA-256 of the VCEK or VLEK DER: names the key and its
    /// TCB without shipping the certificate.
    pub key_sha256_hex: String,
    /// The report, every field covered by the signature just verified.
    pub report: SnpReport,
    pub reported_tcb: TcbParts,
    /// The minimum the TCBs were held to, and where it came from.
    pub minimum: SnpTcbMinimum,
    pub minimum_source: String,
    /// The policy's [`crate::tcb::TcbMode`], the report's CPU line, and each
    /// bulletin of that line, imposed or only named (empty under an explicit
    /// minimum: the caller judged instead of the bulletins).
    pub tcb_mode: crate::tcb::TcbMode,
    pub cpu_line: Option<String>,
    pub bulletins: Vec<BulletinStanding>,
    /// thisUpdate and nextUpdate (Unix seconds) of the AMD CRL read.
    pub crl_window: (i64, i64),
    pub limitations: Vec<String>,
}

impl VerifiedSnp {
    /// `REPORT_DATA`, as the hardware signed it. Returned rather than
    /// judged: the caller holds the session it must equal.
    pub fn report_data(&self) -> &[u8; 64] {
        &self.report.report_data
    }
}

/// What every SNP verdict says it did not establish, whatever the report.
pub const LIMITATIONS: &[&str] = &[
    "revocation: AMD's CRL for the product (signed by the pinned ARK, read inside its window) \
     lists ASKs and ASVKs; a VCEK or VLEK is not revoked one by one, its TCB version is held \
     to the minimum instead",
    "launch measurement: MEASUREMENT covers what the host loaded into the guest before launch \
     (firmware, and with measured direct boot the kernel, initrd and command line hashes); \
     what the guest runs afterwards is not in the report",
    "the mitigation vectors of a v5 report are reported, not judged",
    "guest policy: the ABI version, single socket, CXL, AES-256-XTS, RAPL, ciphertext hiding \
     and page swap bits are read, not judged",
    "physical attacks on DRAM (interposers, TEE.fail class) and side channels are out of scope",
    CPU_ONLY,
];

/// The last line of [`LIMITATIONS`]; a verdict that attests GPUs beside the
/// report replaces it.
pub const CPU_ONLY: &str = "CPU only: no GPU is attested with an SNP report";

fn limitations_of(
    signing_key: SnpSigningKey,
    csp_id: Option<&str>,
    minimum: SnpTcbMinimum,
    source: &str,
    smt: SnpSmt,
    platform_smt: bool,
) -> Vec<String> {
    let mut out: Vec<String> = LIMITATIONS.iter().map(|s| s.to_string()).collect();
    let zero: Vec<&str> = [
        ("boot loader", minimum.boot_loader),
        ("TEE", minimum.tee),
        ("FMC", minimum.fmc),
    ]
    .iter()
    .filter(|(_, v)| *v == 0)
    .map(|(n, _)| *n)
    .collect();
    let mut tcb = format!(
        "TCB: held to boot loader {:#04x}, TEE {:#04x}, SNP {:#04x}, microcode {:#04x}, \
         FMC {:#04x} ({source})",
        minimum.boot_loader, minimum.tee, minimum.snp, minimum.microcode, minimum.fmc
    );
    if !zero.is_empty() {
        tcb.push_str(&format!("; {} at 0 judge nothing", zero.join(", ")));
    }
    out.insert(1, tcb);
    if smt == SnpSmt::Allowed {
        out.push(format!(
            "SMT allowed by the written policy (PLATFORM_INFO.SMT_EN {}): with SMT on, a \
             sibling thread on the same core may run host code",
            u8::from(platform_smt)
        ));
    }
    if signing_key == SnpSigningKey::Vlek {
        out.push(format!(
            "signed by a VLEK, the key AMD issued to {}: the report is bound to that \
             provider's key and TCB, not to one chip (CHIP_ID is not checked)",
            csp_id.unwrap_or("?")
        ));
    }
    out
}

// ── Evidence ─────────────────────────────────────────────────────────────────

/// The payload of an `EvidenceFormat::SevSnpReport`: the report, then the
/// VCEK's or VLEK's DER.
pub fn evidence_payload(report: &[u8], key_der: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(report.len() + key_der.len());
    out.extend_from_slice(report);
    out.extend_from_slice(key_der);
    out
}

/// `(report, key_der)`.
pub fn split_evidence(payload: &[u8]) -> Result<(&[u8], &[u8]), SnpRejected> {
    if payload.len() <= REPORT_LEN {
        return Err(SnpRejected::Evidence(format!(
            "{} bytes: a {REPORT_LEN}-byte report followed by a VCEK or VLEK is expected",
            payload.len()
        )));
    }
    Ok(payload.split_at(REPORT_LEN))
}

/// The product and key kind a payload CLAIMS, read without verifying
/// anything: only to choose which AMD CRL to fetch. The CRL is then
/// verified under the pinned ARK of the product the chain proves.
pub fn evidence_product(payload: &[u8]) -> Result<(SnpProduct, SnpSigningKey), SnpRejected> {
    let (report_bytes, key_der) = split_evidence(payload)?;
    let report = snp_report::parse(report_bytes).map_err(|e| SnpRejected::Report(e.to_string()))?;
    let kind = signing_key_kind(&report)?;
    let (_, cert) =
        X509Certificate::from_der(key_der).map_err(|e| key_error(kind, format!("DER: {e}")))?;
    let ext = AmdExtensions::of(&cert).map_err(|e| key_error(kind, e))?;
    let name = ext
        .ia5("2", "productName")
        .map_err(|e| key_error(kind, e))?;
    let product = SnpProduct::from_vcek_product_name(&name)
        .ok_or_else(|| key_error(kind, format!("unknown productName {name:?}")))?;
    Ok((product, kind))
}

/// The signed bytes of a report minus `REPORT_DATA`: what two reports of the
/// same guest on the same chip and TCB share, whatever they bind.
pub fn signed_body_without_report_data(report: &[u8]) -> Option<Vec<u8>> {
    if report.len() != REPORT_LEN {
        return None;
    }
    let mut out = report[..REPORT_DATA_AT].to_vec();
    out.extend_from_slice(&report[REPORT_DATA_AT + 64..SIGNED_LEN]);
    Some(out)
}

// ── Verification ─────────────────────────────────────────────────────────────

fn key_error(kind: SnpSigningKey, m: String) -> SnpRejected {
    match kind {
        SnpSigningKey::Vcek => SnpRejected::Vcek(m),
        SnpSigningKey::Vlek => SnpRejected::Vlek(m),
    }
}

/// `KEY_INFO.SIGNING_KEY`, refusing an unsigned report and reserved values.
fn signing_key_kind(report: &SnpReport) -> Result<SnpSigningKey, SnpRejected> {
    match report.signing_key() {
        snp_report::SIGNING_KEY_VCEK => Ok(SnpSigningKey::Vcek),
        snp_report::SIGNING_KEY_VLEK => Ok(SnpSigningKey::Vlek),
        snp_report::SIGNING_KEY_NONE => {
            Err(SnpRejected::SigningKey("the report is not signed".into()))
        }
        other => Err(SnpRejected::SigningKey(format!(
            "SIGNING_KEY {other} is reserved"
        ))),
    }
}

/// What a VCEK or VLEK verified to a pinned root says.
#[derive(Debug)]
struct KeyFacts {
    product: SnpProduct,
    key: p384::ecdsa::VerifyingKey,
    tcb_raw: [u8; 8],
    /// VCEK: 64 bytes on Milan and Genoa, 8 on Turin. VLEK: none.
    hwid: Option<Vec<u8>>,
    /// VLEK: the cloud provider. VCEK: none.
    csp_id: Option<String>,
    sha256_hex: String,
}

/// One product's pinned ARK, verified: the root a CRL and an intermediate
/// must chain to.
struct VerifiedArk {
    ark: X509Certificate<'static>,
    key: RsaPublicKey,
}

/// The pinned ARK: pin, self-issued, self-signed, window.
fn verify_ark(root: &SnpRoot, now_unix: u64) -> Result<VerifiedArk, SnpRejected> {
    let product = root.product.name();
    let chain = |m: String| SnpRejected::Chain(format!("{product}: {m}"));
    let ark = parse_cert(root.ark_der).map_err(|e| chain(format!("ARK: {e}")))?;
    // The pin first: a certificate that is not the pinned one is not
    // verified at all, however well it chains.
    if spki_sha256_hex(&ark) != root.ark_spki_sha256_hex {
        return Err(SnpRejected::RootNotPinned(product.into()));
    }
    if ark.issuer().as_raw() != ark.subject().as_raw() {
        return Err(chain("the ARK is not self-issued".into()));
    }
    let key = rsa_key(&ark).map_err(|e| chain(format!("ARK key: {e}")))?;
    signed_by(&ark, &key, "ARK", now_unix).map_err(chain)?;
    Ok(VerifiedArk { ark, key })
}

/// The pinned intermediate of `kind` (ASK for a VCEK, ASVK for a VLEK):
/// pin, issued and signed by the verified ARK, a CA, window.
fn verify_intermediate(
    root: &SnpRoot,
    ark: &VerifiedArk,
    kind: SnpSigningKey,
    now_unix: u64,
) -> Result<(X509Certificate<'static>, RsaPublicKey), SnpRejected> {
    let product = root.product.name();
    let what = kind.intermediate();
    let chain = |m: String| SnpRejected::Chain(format!("{product}: {m}"));
    let (der, pin) = match kind {
        SnpSigningKey::Vcek => (root.ask_der, &root.ask_spki_sha256_hex),
        SnpSigningKey::Vlek => (root.asvk_der, &root.asvk_spki_sha256_hex),
    };
    let cert = parse_cert(der).map_err(|e| chain(format!("{what}: {e}")))?;
    if spki_sha256_hex(&cert) != *pin {
        return Err(SnpRejected::RootNotPinned(format!("{product} {what}")));
    }
    if cert.issuer().as_raw() != ark.ark.subject().as_raw() {
        return Err(chain(format!(
            "{what} {} is not issued by ARK {}",
            cert.subject(),
            ark.ark.subject()
        )));
    }
    signed_by(&cert, &ark.key, what, now_unix).map_err(chain)?;
    match cert.basic_constraints() {
        Ok(Some(bc)) if bc.value.ca => {}
        _ => return Err(chain(format!("the {what} is not a CA"))),
    }
    let key = rsa_key(&cert).map_err(|e| chain(format!("{what} key: {e}")))?;
    Ok((cert, key))
}

/// The pinned ARK and ASK of one product (the VCEK chain).
#[cfg(test)]
fn verify_root(root: &SnpRoot, now_unix: u64) -> Result<(), SnpRejected> {
    let ark = verify_ark(root, now_unix)?;
    verify_intermediate(root, &ark, SnpSigningKey::Vcek, now_unix).map(|_| ())
}

/// A VCEK or VLEK, to the pinned root of the product it names. Returns the
/// facts, the verified ARK (for the CRL) and the intermediate's serial.
fn verify_signing_cert(
    der: &[u8],
    kind: SnpSigningKey,
    anchors: &SnpAnchors,
    now_unix: u64,
) -> Result<(KeyFacts, VerifiedArk, Vec<u8>), SnpRejected> {
    let err = |m: String| key_error(kind, m);
    let (rest, cert) = X509Certificate::from_der(der).map_err(|e| err(format!("DER: {e}")))?;
    if !rest.is_empty() {
        return Err(err(format!("{} bytes after the certificate", rest.len())));
    }
    let ext = AmdExtensions::of(&cert).map_err(err)?;
    let name = ext.ia5("2", "productName").map_err(err)?;
    let product = SnpProduct::from_vcek_product_name(&name)
        .ok_or_else(|| err(format!("unknown productName {name:?}")))?;
    let root = anchors.root(product)?;
    let ark = verify_ark(root, now_unix)?;
    let (inter, inter_key) = verify_intermediate(root, &ark, kind, now_unix)?;
    let chain = |m: String| SnpRejected::Chain(format!("{}: {m}", product.name()));
    if cert.issuer().as_raw() != inter.subject().as_raw() {
        return Err(chain(format!(
            "{} issuer {} is not the pinned {} {}",
            kind.name(),
            cert.issuer(),
            kind.intermediate(),
            inter.subject()
        )));
    }
    // SECURITY: the issuer NAME proves nothing (anyone can write
    // "SEV-Genoa"); this signature is the link.
    signed_by(&cert, &inter_key, kind.name(), now_unix).map_err(chain)?;
    if let Ok(Some(bc)) = cert.basic_constraints() {
        if bc.value.ca {
            return Err(err("a CA, not an end entity".into()));
        }
    }
    let key = <p384::ecdsa::VerifyingKey as p384::pkcs8::DecodePublicKey>::from_public_key_der(
        cert.public_key().raw,
    )
    .map_err(|e| err(format!("not a P-384 key: {e}")))?;
    let spl = |oid: &str| ext.spl(oid).map_err(err);
    let tcb_raw = match product.tcb_layout() {
        TcbLayout::MilanGenoa => [
            spl("3.1")?,
            spl("3.2")?,
            spl("3.4")?,
            spl("3.5")?,
            spl("3.6")?,
            spl("3.7")?,
            spl("3.3")?,
            spl("3.8")?,
        ],
        TcbLayout::Turin => [
            spl("3.9")?,
            spl("3.1")?,
            spl("3.2")?,
            spl("3.3")?,
            spl("3.5")?,
            spl("3.6")?,
            spl("3.7")?,
            spl("3.8")?,
        ],
    };
    // A VCEK names a chip and no provider; a VLEK a provider and no chip.
    let (hwid, csp_id) = match kind {
        SnpSigningKey::Vcek => {
            if ext.has("5") {
                return Err(err("carries a cspId: a VLEK's extension".into()));
            }
            let hwid = ext.raw("4", "hwID").map_err(err)?.to_vec();
            let want = match product {
                SnpProduct::Turin => 8,
                SnpProduct::Milan | SnpProduct::Genoa => 64,
            };
            if hwid.len() != want {
                return Err(err(format!(
                    "hwID is {} bytes, a {} VCEK carries {want}",
                    hwid.len(),
                    product.name()
                )));
            }
            (Some(hwid), None)
        }
        SnpSigningKey::Vlek => {
            if ext.has("4") {
                return Err(err("carries a hwID: a VCEK's extension".into()));
            }
            let csp = ext.ia5("5", "cspId").map_err(err)?;
            if csp.is_empty() {
                return Err(err("empty cspId".into()));
            }
            (None, Some(csp))
        }
    };
    Ok((
        KeyFacts {
            product,
            key,
            tcb_raw,
            hwid,
            csp_id,
            sha256_hex: hex(&Sha256::digest(der)),
        },
        ark,
        inter.raw_serial().to_vec(),
    ))
}

/// A VCEK to its pinned root, for a caller that holds no report.
#[cfg(test)]
fn verify_vcek(der: &[u8], anchors: &SnpAnchors, now_unix: u64) -> Result<KeyFacts, SnpRejected> {
    verify_signing_cert(der, SnpSigningKey::Vcek, anchors, now_unix).map(|(f, _, _)| f)
}

/// A serial without the sign byte DER puts before a high bit.
fn serial_key(raw: &[u8]) -> &[u8] {
    let mut s = raw;
    while let [0, rest @ ..] = s {
        s = rest;
    }
    s
}

/// AMD's CRL for a product: issued and signed (RSASSA-PSS SHA-384, salt 48)
/// by the verified ARK, v2, inside its thisUpdate..nextUpdate window at
/// `now_unix`, no critical extension; `intermediate_serial` not listed.
/// Returns the window.
fn verify_crl(
    crl_der: &[u8],
    product: SnpProduct,
    ark: &VerifiedArk,
    intermediate: &str,
    intermediate_serial: &[u8],
    now_unix: u64,
) -> Result<(i64, i64), SnpRejected> {
    let bad = |m: String| SnpRejected::Crl(format!("{}: {m}", product.name()));
    let crl = match CertificateRevocationList::from_der(crl_der) {
        Ok(([], crl)) => crl,
        Ok((rest, _)) => return Err(bad(format!("{} bytes after the CRL", rest.len()))),
        Err(e) => return Err(bad(format!("DER: {e}"))),
    };
    let tbs = &crl.tbs_cert_list;
    if tbs.version != Some(X509Version::V2) {
        return Err(bad("not a v2 CRL".into()));
    }
    if tbs.issuer.as_raw() != ark.ark.subject().as_raw() {
        return Err(bad(format!(
            "issued by {}, not the pinned ARK {}",
            tbs.issuer,
            ark.ark.subject()
        )));
    }
    if !crate::pcs::signature_algorithm_repeated(crl.as_raw()) {
        return Err(bad("the signed and the outer algorithms differ".into()));
    }
    // SECURITY: the issuer NAME proves nothing; the ARK's signature is the
    // link (a rogue ARK of the same name signs a CRL that lists nothing).
    pss_verify(
        &crl.signature_algorithm,
        &ark.key,
        tbs.as_ref(),
        crl.signature_value.data.as_ref(),
        "CRL",
    )
    .map_err(bad)?;
    for e in tbs.extensions() {
        if e.critical {
            return Err(bad(format!(
                "critical extension {} is not read here",
                e.oid
            )));
        }
    }
    let this_update = tbs.this_update.timestamp();
    let next_update = tbs
        .next_update
        .ok_or_else(|| bad("no nextUpdate: no window to judge it in".into()))?
        .timestamp();
    let now = i64::try_from(now_unix).unwrap_or(i64::MAX);
    if now < this_update || now > next_update {
        return Err(bad(format!(
            "outside its window [{this_update}, {next_update}] at {now}"
        )));
    }
    for r in &tbs.revoked_certificates {
        if r.extensions().iter().any(|e| e.critical) {
            return Err(bad(format!(
                "entry {} carries a critical extension",
                r.raw_serial_as_string()
            )));
        }
        if serial_key(r.raw_serial()) == serial_key(intermediate_serial) {
            return Err(SnpRejected::Revoked(format!(
                "the {} {intermediate} (serial {}) is listed",
                product.name(),
                r.raw_serial_as_string()
            )));
        }
    }
    Ok((this_update, next_update))
}

fn parse_cert(der: &'static [u8]) -> Result<X509Certificate<'static>, String> {
    match X509Certificate::from_der(der) {
        Ok(([], cert)) => Ok(cert),
        Ok((rest, _)) => Err(format!("{} bytes after the certificate", rest.len())),
        Err(e) => Err(format!("DER: {e}")),
    }
}

fn spki_sha256_hex(cert: &X509Certificate<'_>) -> String {
    hex(&Sha256::digest(cert.public_key().raw))
}

fn rsa_key(cert: &X509Certificate<'_>) -> Result<RsaPublicKey, String> {
    RsaPublicKey::from_public_key_der(cert.public_key().raw).map_err(|e| e.to_string())
}

/// `signed` signed under `alg` = RSASSA-PSS SHA-384 / MGF1-SHA-384 / salt
/// 48, by `issuer`. Every AMD SEV certificate and CRL uses exactly these.
fn pss_verify(
    alg: &AlgorithmIdentifier<'_>,
    issuer: &RsaPublicKey,
    signed: &[u8],
    signature: &[u8],
    what: &str,
) -> Result<(), String> {
    if alg.algorithm != OID_PKCS1_RSASSAPSS {
        return Err(format!(
            "{what}: signature algorithm {} is not RSASSA-PSS",
            alg.algorithm
        ));
    }
    let parsed =
        SignatureAlgorithm::try_from(alg).map_err(|e| format!("{what}: PSS parameters: {e}"))?;
    let SignatureAlgorithm::RSASSA_PSS(p) = parsed else {
        return Err(format!("{what}: parameters are not PSS"));
    };
    let mgf = p
        .mask_gen_algorithm()
        .map_err(|e| format!("{what}: MGF: {e}"))?;
    if *p.hash_algorithm_oid() != OID_NIST_HASH_SHA384
        || mgf.hash != OID_NIST_HASH_SHA384
        || p.salt_length() as usize != PSS_SALT_LEN
    {
        return Err(format!(
            "{what}: PSS parameters are not SHA-384/MGF1-SHA-384/salt 48"
        ));
    }
    let vk = rsa::pss::VerifyingKey::<rsa::sha2::Sha384>::new_with_salt_len(
        issuer.clone(),
        PSS_SALT_LEN,
    );
    let sig =
        rsa::pss::Signature::try_from(signature).map_err(|e| format!("{what}: signature: {e}"))?;
    vk.verify(signed, &sig)
        .map_err(|_| format!("{what}: signature does not verify under its issuer's key"))
}

/// `cert` is inside its validity window at `now_unix` and signed by `issuer`
/// ([`pss_verify`]), under the algorithm its signed part names too.
fn signed_by(
    cert: &X509Certificate<'_>,
    issuer: &RsaPublicKey,
    what: &str,
    now_unix: u64,
) -> Result<(), String> {
    let (not_before, not_after) = (
        cert.validity().not_before.timestamp(),
        cert.validity().not_after.timestamp(),
    );
    let now = i64::try_from(now_unix).unwrap_or(i64::MAX);
    if now < not_before || now > not_after {
        return Err(format!(
            "{what}: outside its validity window [{not_before}, {not_after}] at {now}"
        ));
    }
    if !crate::pcs::signature_algorithm_repeated(cert.as_raw()) {
        return Err(format!(
            "{what}: the signed and the outer algorithms differ"
        ));
    }
    pss_verify(
        &cert.signature_algorithm,
        issuer,
        cert.tbs_certificate.as_ref(),
        cert.signature_value.data.as_ref(),
        what,
    )
}

/// The AMD extensions (57230) of a VCEK or VLEK, each present once and
/// none critical.
struct AmdExtensions<'a> {
    by_oid: Vec<(String, &'a [u8])>,
}

impl<'a> AmdExtensions<'a> {
    fn of(cert: &'a X509Certificate<'_>) -> Result<Self, String> {
        let mut by_oid: Vec<(String, &'a [u8])> = Vec::new();
        for e in cert.extensions() {
            // A critical extension this verifier does not read would be a
            // constraint it silently ignores; AMD marks none critical.
            if e.critical {
                return Err(format!("critical extension {} is not read here", e.oid));
            }
            let oid = e.oid.to_id_string();
            if let Some(rest) = oid.strip_prefix(AMD_OID) {
                if by_oid.iter().any(|(o, _)| o == rest) {
                    return Err(format!("extension {oid} twice"));
                }
                by_oid.push((rest.to_string(), e.value));
            }
        }
        Ok(Self { by_oid })
    }

    fn has(&self, oid: &str) -> bool {
        self.by_oid.iter().any(|(o, _)| o == oid)
    }

    fn raw(&self, oid: &str, name: &str) -> Result<&'a [u8], String> {
        self.by_oid
            .iter()
            .find(|(o, _)| o == oid)
            .map(|(_, v)| *v)
            .ok_or_else(|| format!("no {name} extension ({AMD_OID}{oid})"))
    }

    /// An SPL: a DER INTEGER from 0 to 255.
    fn spl(&self, oid: &str) -> Result<u8, String> {
        let v = self.raw(oid, "SPL")?;
        match v {
            [0x02, 0x01, b] if *b < 0x80 => Ok(*b),
            [0x02, 0x02, 0x00, b] if *b >= 0x80 => Ok(*b),
            _ => Err(format!("{AMD_OID}{oid} is not an INTEGER 0..=255")),
        }
    }

    fn ia5(&self, oid: &str, name: &str) -> Result<String, String> {
        match self.raw(oid, name)? {
            [0x16, len, rest @ ..] if usize::from(*len) == rest.len() && rest.is_ascii() => {
                Ok(String::from_utf8_lossy(rest).into_owned())
            }
            _ => Err(format!("{name} is not a short IA5String")),
        }
    }
}

/// R and S of the report, little-endian 72-byte fields, as one P-384
/// signature. Bytes 48..72 of each must be zero: a wider value is not a
/// P-384 scalar, and truncating it would accept two encodings of one
/// signature.
fn report_signature(report: &SnpReport) -> Result<p384::ecdsa::Signature, SnpRejected> {
    let mut rs = [0u8; 96];
    for (i, part) in [&report.signature_r, &report.signature_s]
        .iter()
        .enumerate()
    {
        if part[48..].iter().any(|b| *b != 0) {
            return Err(SnpRejected::ReportSignature);
        }
        let mut be = part[..48].to_vec();
        be.reverse();
        rs[i * 48..(i + 1) * 48].copy_from_slice(&be);
    }
    p384::ecdsa::Signature::from_slice(&rs).map_err(|_| SnpRejected::ReportSignature)
}

/// Bits of `POLICY`, `KEY_INFO` and `PLATFORM_INFO` this verifier does not
/// know: refused, never ignored.
fn check_known_bits(report: &SnpReport) -> Result<(), SnpRejected> {
    if report.policy & !POLICY_KNOWN != 0 {
        return Err(SnpRejected::Unknown(format!(
            "POLICY {:#018x} sets bits above 25",
            report.policy
        )));
    }
    if report.key_info & !KEY_INFO_KNOWN != 0 {
        return Err(SnpRejected::Unknown(format!(
            "KEY_INFO {:#010x} sets bits above 4",
            report.key_info
        )));
    }
    if report.platform_info & !PLATFORM_INFO_KNOWN != 0 {
        return Err(SnpRejected::Unknown(format!(
            "PLATFORM_INFO {:#018x} sets bits above 5",
            report.platform_info
        )));
    }
    Ok(())
}

/// SMT under [`SnpSmt::Forbidden`]: the guest's policy refuses it and the
/// platform reports it off.
fn check_smt(report: &SnpReport, smt: SnpSmt) -> Result<(), SnpRejected> {
    if smt == SnpSmt::Allowed {
        return Ok(());
    }
    if report.policy & snp_report::POLICY_SMT != 0 {
        return Err(SnpRejected::Smt(
            "POLICY.SMT is set: the guest accepts a host with SMT on, the written policy forbids it"
                .into(),
        ));
    }
    if report.platform_info & PLATFORM_INFO_SMT_EN != 0 {
        return Err(SnpRejected::Smt(
            "PLATFORM_INFO.SMT_EN is set: SMT is on, the written policy forbids it".into(),
        ));
    }
    Ok(())
}

/// The minimum the TCBs are held to: the explicit one or the CPU line's
/// default under the policy's mode at `now_unix`, raised by the written
/// floor; with the line's name and bulletins when the default applies.
#[allow(clippy::type_complexity)]
fn minimum_for(
    policy: &SnpPolicy,
    product: SnpProduct,
    cpuid: Option<Cpuid>,
    now_unix: u64,
) -> Result<
    (
        SnpTcbMinimum,
        String,
        Option<String>,
        Vec<(BulletinFloor, bool)>,
    ),
    SnpRejected,
> {
    let (base, source, line, floors) = match policy.minimum {
        Some(m) => (m, "explicit".to_string(), None, Vec::new()),
        None => {
            let (m, s, line, floors) = mode_minimum(product, cpuid, policy.tcb_mode, now_unix)?;
            (m, s, Some(line), floors)
        }
    };
    Ok(match policy.floor {
        Some(floor) => (
            base.raised_to(floor),
            format!("{source}, raised by the written floor"),
            line,
            floors,
        ),
        None => (base, source, line, floors),
    })
}

/// Verify one SNP evidence payload to a pinned AMD root, with AMD's CRL for
/// its product, under `policy`, at `now_unix` (every certificate's window
/// and the CRL's are tested against it). Fail-closed on every link.
pub fn verify_snp_report(
    payload: &[u8],
    crl_der: &[u8],
    anchors: &SnpAnchors,
    policy: &SnpPolicy,
    now_unix: u64,
) -> Result<VerifiedSnp, SnpRejected> {
    use p384::ecdsa::signature::Verifier as _;

    let (report_bytes, key_der) = split_evidence(payload)?;
    let report = snp_report::parse(report_bytes).map_err(|e| SnpRejected::Report(e.to_string()))?;
    if report.signature_algo != snp_report::SIGNATURE_ALGO_ECDSA_P384_SHA384 {
        return Err(SnpRejected::Report(format!(
            "SIGNATURE_ALGO {}: only 1 (ECDSA P-384 with SHA-384) is defined",
            report.signature_algo
        )));
    }
    check_known_bits(&report)?;
    let kind = signing_key_kind(&report)?;
    if report.mask_chip_key() {
        return Err(SnpRejected::SigningKey(
            "MASK_CHIP_KEY is set: CHIP_ID is zeroed, nothing ties the report to its key".into(),
        ));
    }

    // 1. The VCEK or VLEK, to a pinned root. 2. The report, by that key.
    let (facts, ark, intermediate_serial) = verify_signing_cert(key_der, kind, anchors, now_unix)?;
    let signature = report_signature(&report)?;
    facts
        .key
        .verify(&report_bytes[..SIGNED_LEN], &signature)
        .map_err(|_| SnpRejected::ReportSignature)?;
    let product = facts.product;

    // 3. The key was issued for THIS TCB and, a VCEK, THIS chip.
    if let Some(hwid) = &facts.hwid {
        let chip = match product {
            SnpProduct::Turin => &report.chip_id[..8],
            SnpProduct::Milan | SnpProduct::Genoa => &report.chip_id[..],
        };
        if chip != hwid.as_slice() {
            return Err(SnpRejected::KeyMismatch(format!(
                "CHIP_ID {} is not the VCEK's hwID {}",
                hex(chip),
                hex(hwid)
            )));
        }
    }
    if report.reported_tcb != facts.tcb_raw {
        return Err(SnpRejected::KeyMismatch(format!(
            "REPORTED_TCB {} is not the {}'s TCB {}",
            hex(&report.reported_tcb),
            kind.name(),
            hex(&facts.tcb_raw)
        )));
    }
    if let Some(c) = report.cpuid {
        if !product.cpuid_matches(c) {
            return Err(SnpRejected::Cpuid(format!(
                "family {:#04x} model {:#04x} is not a {} CPU",
                c.family,
                c.model,
                product.name()
            )));
        }
    }

    // 4. The chip, the key's owner, the guest, the firmware.
    if let Some(products) = &policy.products {
        if !products.contains(&product) {
            return Err(SnpRejected::ProductRefused(product.name().into()));
        }
    }
    if product == SnpProduct::Milan && !policy.allow_milan {
        return Err(SnpRejected::MilanRefused);
    }
    if let Some(csp) = &facts.csp_id {
        if !policy.vlek_csp_ids.iter().any(|c| c == csp) {
            return Err(SnpRejected::SigningKey(format!(
                "signed by a VLEK of {csp:?}, a provider the policy does not name \
                 (VLEK accepted only from a named provider)"
            )));
        }
    }
    match report.vmpl {
        0 => {}
        snp_report::VMPL_HOST_REQUESTED => {
            return Err(SnpRejected::Vmpl(
                "the host requested this report (VMPL 0xFFFFFFFF): its REPORT_DATA is the host's"
                    .into(),
            ))
        }
        v => {
            return Err(SnpRejected::Vmpl(format!(
                "requested at VMPL {v}: only VMPL 0, the guest's most privileged level, is accepted"
            )))
        }
    }
    if report.policy & snp_report::POLICY_RESERVED_MUST_BE_ONE == 0 {
        return Err(SnpRejected::PolicyReserved);
    }
    if report.is_debug() {
        return Err(SnpRejected::Debug);
    }
    if report.allows_migration_agent() {
        return Err(SnpRejected::MigrationAgent(
            "POLICY.MIGRATE_MA is set: a migration agent may be associated".into(),
        ));
    }
    if report.has_migration_agent() {
        return Err(SnpRejected::MigrationAgent(
            "REPORT_ID_MA names a migration agent".into(),
        ));
    }
    check_smt(&report, policy.smt)?;
    let (minimum, minimum_source, cpu_line, floors) =
        minimum_for(policy, product, report.cpuid, now_unix)?;
    let layout = product.tcb_layout();
    for (field, raw) in [
        ("REPORTED_TCB", report.reported_tcb),
        ("CURRENT_TCB", report.current_tcb),
        ("COMMITTED_TCB", report.committed_tcb),
        ("LAUNCH_TCB", report.launch_tcb),
    ] {
        let t = TcbParts::decode(raw, layout);
        for (part, got, min) in [
            ("FMC", t.fmc, minimum.fmc),
            ("boot loader", t.boot_loader, minimum.boot_loader),
            ("TEE", t.tee, minimum.tee),
            ("SNP", t.snp, minimum.snp),
            ("microcode", t.microcode, minimum.microcode),
        ] {
            if got < min {
                return Err(SnpRejected::TcbTooLow(format!(
                    "{field}: {part} {got:#04x} < {min:#04x} ({minimum_source})"
                )));
            }
        }
    }
    if minimum.require_alias_check && !report.alias_check_complete() {
        return Err(SnpRejected::AliasCheck);
    }

    // 5. Revocation, last: AMD's CRL for the product, under the same ARK.
    let crl_window = verify_crl(
        crl_der,
        product,
        &ark,
        kind.intermediate(),
        &intermediate_serial,
        now_unix,
    )?;
    let lowest = lowest_tcb(&report, product);
    let bulletins: Vec<BulletinStanding> = floors
        .into_iter()
        .map(|(f, imposed)| BulletinStanding {
            bulletin: f.bulletin,
            imposed,
            minimum: f.minimum,
            note: f.note,
            not_met: tcb_gaps(&f.minimum, &lowest, report.alias_check_complete()),
        })
        .collect();
    let mut limitations = limitations_of(
        kind,
        facts.csp_id.as_deref(),
        minimum,
        &minimum_source,
        policy.smt,
        report.platform_info & PLATFORM_INFO_SMT_EN != 0,
    );
    for b in bulletins.iter().filter(|b| !b.imposed) {
        limitations.push(format!(
            "TCB: {} ({}, {}) is named, not imposed: under a year old, transparent TCB policy; {}",
            b.bulletin.id,
            b.bulletin.name,
            b.bulletin.published,
            if b.not_met.is_empty() {
                "the reported TCB meets it".to_string()
            } else {
                format!(
                    "the reported TCB does not meet it: {}",
                    b.not_met.join(", ")
                )
            }
        ));
    }
    Ok(VerifiedSnp {
        product,
        signing_key: kind,
        csp_id: facts.csp_id,
        key_sha256_hex: facts.sha256_hex,
        reported_tcb: TcbParts::decode(report.reported_tcb, layout),
        report,
        minimum,
        minimum_source,
        tcb_mode: policy.tcb_mode,
        cpu_line,
        bulletins,
        crl_window,
        limitations,
    })
}

// ── Measurements ─────────────────────────────────────────────────────────────

/// The fields of a report a client may pin, under the names a verdict
/// gives them: the launch digest (`MEASUREMENT`), what the host passed at
/// launch (`HOST_DATA`), and the digests of the ID and author keys that
/// signed the launch's ID block, if any.
pub const SNP_REGISTERS: [&str; 4] = [
    "snp_measurement",
    "snp_host_data",
    "snp_id_key_digest",
    "snp_author_key_digest",
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `vcek_sha256` (the chip and TCB) or `vlek_sha256` (the provider's key
/// and TCB), then [`SNP_REGISTERS`], from a report whose signature has been
/// verified.
pub fn snp_measurements(v: &VerifiedSnp) -> Vec<Measurement> {
    let m = |register: &str, algorithm: &str, value_hex: String| Measurement {
        register: register.into(),
        algorithm: algorithm.into(),
        value_hex,
    };
    let key_register = match v.signing_key {
        SnpSigningKey::Vcek => "vcek_sha256",
        SnpSigningKey::Vlek => "vlek_sha256",
    };
    vec![
        m(key_register, "sha256", v.key_sha256_hex.clone()),
        m("snp_measurement", "sha384", hex(&v.report.measurement)),
        m("snp_host_data", "raw", hex(&v.report.host_data)),
        m("snp_id_key_digest", "sha384", hex(&v.report.id_key_digest)),
        m(
            "snp_author_key_digest",
            "sha384",
            hex(&v.report.author_key_digest),
        ),
    ]
}

/// `snp_measurement=<96 hex>` → an expectation. Refuses what it could never
/// enforce: an unknown register, or a value of the wrong length.
pub fn parse_expected_snp_measurement(spec: &str) -> Result<ExpectedMeasurement, String> {
    let (register, value) = spec
        .split_once('=')
        .ok_or_else(|| "expected <register>=<hex>; the value given is not printed".to_string())?;
    let (algorithm, len) = match register {
        "snp_measurement" | "snp_id_key_digest" | "snp_author_key_digest" => ("sha384", 96),
        "snp_host_data" => ("raw", 64),
        _ => {
            return Err(format!(
                "not an SEV-SNP field before `=`: use one of {SNP_REGISTERS:?}; the value given is not printed"
            ))
        }
    };
    if value.len() != len || !value.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "{register}: {len} hex characters expected, got {}",
            value.len()
        ));
    }
    Ok(ExpectedMeasurement {
        register: register.into(),
        algorithm: algorithm.into(),
        value_hex: value.to_lowercase(),
    })
}

/// Exact match on every expectation; one naming a field the verdict does not
/// carry refuses.
pub fn unmet_snp(measured: &[Measurement], expected: &[ExpectedMeasurement]) -> Option<String> {
    expected.iter().find_map(|want| {
        let got = measured
            .iter()
            .find(|m| m.register == want.register && m.algorithm == want.algorithm);
        match got {
            None => Some(format!(
                "expected measurement '{}' ({}) is not in the SNP report",
                want.register, want.algorithm
            )),
            Some(m) if !m.value_hex.eq_ignore_ascii_case(&want.value_hex) => Some(format!(
                "measurement '{}' mismatch: pinned {}, the hardware reports {} — \
                 genuine SEV-SNP, but not the image this policy names",
                want.register, want.value_hex, m.value_hex
            )),
            Some(_) => None,
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use p384::ecdsa::signature::Signer as _;

    /// 2026-02-02: inside every TEST certificate's window and the TEST
    /// CRLs' (2026-01-27 to 2026-03-10).
    pub(crate) const NOW: u64 = 1_770_000_000;
    /// 2026-10-28 00:00 UTC: inside AMD's CRLs of 2026-10-27 (next update
    /// 2026-12-09) and the real VCEKs' windows.
    pub(crate) const NOW_REAL: u64 = 1_793_145_600;
    /// The TEST keys' private scalars (scripts/make-snp-fixtures.py).
    pub(crate) const CURRENT: u8 = 0x5A;
    pub(crate) const OLD: u8 = 0x3C;
    pub(crate) const TURIN: u8 = 0x6B;
    pub(crate) const VLEK: u8 = 0x7C;
    /// The cloud provider the TEST VLEK names.
    pub(crate) const TEST_CSP: &str = "CN=test-csp.x0ne.invalid";
    pub(crate) const GENOA_TCB_CURRENT: [u8; 8] = [0x0A, 0, 0, 0, 0, 0, 0x1B, 0x58];
    pub(crate) const GENOA_TCB_OLD: [u8; 8] = [0x0A, 0, 0, 0, 0, 0, 0x17, 0x54];
    pub(crate) const TURIN_TCB: [u8; 8] = [1, 1, 1, 0x04, 0, 0, 0, 0x52];
    pub(crate) const TURIN_HWID: [u8; 8] = [0x7a, 0x1e, 0x5c, 0x26, 0x6c, 0x01, 0x08, 0xdb];

    pub(crate) const TEST_GENOA_ARK: &[u8] = include_bytes!("../tests/snp/test_genoa_ark.der");
    pub(crate) const TEST_GENOA_ASK: &[u8] = include_bytes!("../tests/snp/test_genoa_ask.der");
    pub(crate) const TEST_GENOA_ASVK: &[u8] = include_bytes!("../tests/snp/test_genoa_asvk.der");
    pub(crate) const TEST_TURIN_ARK: &[u8] = include_bytes!("../tests/snp/test_turin_ark.der");
    pub(crate) const TEST_TURIN_ASK: &[u8] = include_bytes!("../tests/snp/test_turin_ask.der");
    pub(crate) const VCEK_CURRENT: &[u8] =
        include_bytes!("../tests/snp/test_genoa_vcek_current.der");
    pub(crate) const VCEK_OLD: &[u8] = include_bytes!("../tests/snp/test_genoa_vcek_old.der");
    pub(crate) const VCEK_ROGUE: &[u8] = include_bytes!("../tests/snp/test_genoa_vcek_rogue.der");
    pub(crate) const VCEK_TURIN: &[u8] = include_bytes!("../tests/snp/test_turin_vcek.der");
    pub(crate) const VLEK_TEST: &[u8] = include_bytes!("../tests/snp/test_genoa_vlek.der");

    /// TEST CRLs, signed by the TEST ARKs: the good ones list serial 0x0F
    /// only (Genoa) or nothing (Turin).
    pub(crate) const TEST_GENOA_CRL: &[u8] = include_bytes!("../tests/snp/test_genoa_crl.der");
    pub(crate) const TEST_TURIN_CRL: &[u8] = include_bytes!("../tests/snp/test_turin_crl.der");
    const CRL_ASK_REVOKED: &[u8] = include_bytes!("../tests/snp/test_genoa_crl_ask_revoked.der");
    const CRL_ASVK_REVOKED: &[u8] = include_bytes!("../tests/snp/test_genoa_crl_asvk_revoked.der");
    const CRL_EXPIRED: &[u8] = include_bytes!("../tests/snp/test_genoa_crl_expired.der");
    const CRL_ROGUE: &[u8] = include_bytes!("../tests/snp/test_genoa_crl_rogue.der");
    const CRL_CRITICAL: &[u8] = include_bytes!("../tests/snp/test_genoa_crl_critical.der");

    // AMD's CRLs, fetched from KDS on 2026-10-28 (tests/VENDORED.md).
    const AMD_MILAN_CRL: &[u8] = include_bytes!("../tests/snp/amd_milan_crl_2026-10-27.der");
    pub(crate) const AMD_GENOA_CRL: &[u8] =
        include_bytes!("../tests/snp/amd_genoa_crl_2026-10-27.der");
    const AMD_TURIN_CRL: &[u8] = include_bytes!("../tests/snp/amd_turin_crl_2026-10-27.der");

    // Real reports and certificates, Apache-2.0 (tests/VENDORED.md).
    pub(crate) const AZURE_GENOA_REPORT: &[u8] =
        include_bytes!("../tests/snp/trustee_azure_genoa_report.bin");
    pub(crate) const AZURE_GENOA_VCEK: &[u8] =
        include_bytes!("../tests/snp/trustee_azure_genoa_vcek.der");
    const VIRTEE_MILAN_REPORT: &[u8] = include_bytes!("../tests/snp/virtee_milan_report.bin");
    const VIRTEE_MILAN_VCEK: &[u8] = include_bytes!("../tests/snp/virtee_milan_vcek.der");
    const VIRTEE_TURIN_VCEK: &[u8] = include_bytes!("../tests/snp/virtee_turin_vcek.der");
    const AWS_VLEK_REPORT: &[u8] = include_bytes!("../tests/snp/trustee_aws_vlek_report.bin");
    const AWS_VLEK: &[u8] = include_bytes!("../tests/snp/trustee_aws_vlek.der");

    /// The guest report of the Scaleway Elastic Metal run of 2026-10-28 (AMD
    /// EPYC, CPUID model 0xA0 stepping 2: Bergamo or Siena; x0ne-snp run
    /// 1028-110059) and its VCEK as AMD KDS served it (tests/VENDORED.md).
    const SCALEWAY_SIENA_REPORT: &[u8] =
        include_bytes!("../tests/snp/scaleway_siena_report_2026-10-28.bin");
    const SCALEWAY_SIENA_VCEK: &[u8] =
        include_bytes!("../tests/snp/scaleway_siena_vcek_2026-10-28.der");

    /// The Azure Genoa report's own level: AMD-SB-3019's (SNP 0x17,
    /// microcode 0x54), below SB-3020 and SB-3027.
    pub(crate) const SB3019_GENOA: SnpTcbMinimum = SnpTcbMinimum {
        fmc: 0,
        boot_loader: 0,
        tee: 0,
        snp: 0x17,
        microcode: 0x54,
        require_alias_check: true,
    };

    pub(crate) fn genoa_chip_id() -> [u8; 64] {
        std::array::from_fn(|i| 0xC0u8.wrapping_add(i as u8))
    }

    fn spki_hex_of(der: &[u8]) -> String {
        let (_, c) = X509Certificate::from_der(der).unwrap();
        hex(&Sha256::digest(c.public_key().raw))
    }

    /// The TEST chains, pinned the way production pins AMD's. Turin has no
    /// TEST ASVK: a Turin VLEK finds no pinned one.
    pub(crate) fn test_anchors() -> SnpAnchors {
        let root = |product, ark: &'static [u8], ask: &'static [u8], asvk: &'static [u8]| SnpRoot {
            product,
            ark_der: ark,
            ask_der: ask,
            asvk_der: asvk,
            ark_spki_sha256_hex: spki_hex_of(ark),
            ask_spki_sha256_hex: spki_hex_of(ask),
            asvk_spki_sha256_hex: if asvk.is_empty() {
                String::new()
            } else {
                spki_hex_of(asvk)
            },
        };
        SnpAnchors {
            roots: vec![
                root(
                    SnpProduct::Genoa,
                    TEST_GENOA_ARK,
                    TEST_GENOA_ASK,
                    TEST_GENOA_ASVK,
                ),
                root(SnpProduct::Turin, TEST_TURIN_ARK, TEST_TURIN_ASK, &[]),
            ],
            kds_base_url: "http://127.0.0.1:9".into(),
        }
    }

    /// The TEST CRL of the product a payload claims.
    pub(crate) fn test_crl(payload: &[u8]) -> &'static [u8] {
        match evidence_product(payload) {
            Ok((SnpProduct::Turin, _)) => TEST_TURIN_CRL,
            _ => TEST_GENOA_CRL,
        }
    }

    fn amd_crl(payload: &[u8]) -> &'static [u8] {
        match evidence_product(payload) {
            Ok((SnpProduct::Milan, _)) => AMD_MILAN_CRL,
            Ok((SnpProduct::Turin, _)) => AMD_TURIN_CRL,
            _ => AMD_GENOA_CRL,
        }
    }

    /// The fields a test sets; [`Spec::bytes`] lays them out per Table 23.
    #[derive(Clone)]
    pub(crate) struct Spec {
        pub(crate) version: u32,
        pub(crate) policy: u64,
        pub(crate) vmpl: u32,
        pub(crate) signature_algo: u32,
        pub(crate) platform_info: u64,
        pub(crate) key_info: u32,
        pub(crate) report_data: [u8; 64],
        pub(crate) measurement: [u8; 48],
        pub(crate) host_data: [u8; 32],
        pub(crate) report_id: [u8; 32],
        pub(crate) report_id_ma: [u8; 32],
        pub(crate) reported_tcb: [u8; 8],
        pub(crate) current_tcb: [u8; 8],
        pub(crate) committed_tcb: [u8; 8],
        pub(crate) launch_tcb: [u8; 8],
        pub(crate) cpuid: [u8; 3],
        pub(crate) chip_id: [u8; 64],
    }

    impl Spec {
        /// A Genoa guest as this crate wants one: v3, VMPL 0, SMT forbidden
        /// and off, no debug, no migration agent, alias check done, TCB
        /// current.
        pub(crate) fn genoa() -> Self {
            Self {
                version: 3,
                policy: snp_report::POLICY_RESERVED_MUST_BE_ONE,
                vmpl: 0,
                signature_algo: 1,
                platform_info: 0x24,
                key_info: 0,
                report_data: [0x64; 64],
                measurement: [0x4D; 48],
                host_data: [0; 32],
                report_id: [0x1D; 32],
                report_id_ma: [0xFF; 32],
                reported_tcb: GENOA_TCB_CURRENT,
                current_tcb: GENOA_TCB_CURRENT,
                committed_tcb: GENOA_TCB_CURRENT,
                launch_tcb: GENOA_TCB_CURRENT,
                cpuid: [0x19, 0x11, 0x01],
                chip_id: genoa_chip_id(),
            }
        }

        /// A Turin classic (C1) guest, Turin TCB layout, 8-byte chip ID.
        pub(crate) fn turin() -> Self {
            let mut chip_id = [0u8; 64];
            chip_id[..8].copy_from_slice(&TURIN_HWID);
            Self {
                platform_info: 0x04,
                reported_tcb: TURIN_TCB,
                current_tcb: TURIN_TCB,
                committed_tcb: TURIN_TCB,
                launch_tcb: TURIN_TCB,
                cpuid: [0x1A, 0x02, 0x01],
                chip_id,
                ..Self::genoa()
            }
        }

        pub(crate) fn bytes(&self) -> Vec<u8> {
            let mut r = vec![0u8; REPORT_LEN];
            r[0x00..0x04].copy_from_slice(&self.version.to_le_bytes());
            r[0x08..0x10].copy_from_slice(&self.policy.to_le_bytes());
            r[0x30..0x34].copy_from_slice(&self.vmpl.to_le_bytes());
            r[0x34..0x38].copy_from_slice(&self.signature_algo.to_le_bytes());
            r[0x38..0x40].copy_from_slice(&self.current_tcb);
            r[0x40..0x48].copy_from_slice(&self.platform_info.to_le_bytes());
            r[0x48..0x4C].copy_from_slice(&self.key_info.to_le_bytes());
            r[0x50..0x90].copy_from_slice(&self.report_data);
            r[0x90..0xC0].copy_from_slice(&self.measurement);
            r[0xC0..0xE0].copy_from_slice(&self.host_data);
            r[0x140..0x160].copy_from_slice(&self.report_id);
            r[0x160..0x180].copy_from_slice(&self.report_id_ma);
            r[0x180..0x188].copy_from_slice(&self.reported_tcb);
            if self.version >= 3 {
                r[0x188..0x18B].copy_from_slice(&self.cpuid);
            }
            r[0x1A0..0x1E0].copy_from_slice(&self.chip_id);
            r[0x1E0..0x1E8].copy_from_slice(&self.committed_tcb);
            r[0x1E8..0x1EB].copy_from_slice(&[0x31, 0x37, 0x01]);
            r[0x1EC..0x1EF].copy_from_slice(&[0x31, 0x37, 0x01]);
            r[0x1F0..0x1F8].copy_from_slice(&self.launch_tcb);
            r
        }
    }

    /// Sign `report[..0x2A0]` with a TEST key; R and S little-endian in
    /// their 72-byte fields (Table 141).
    pub(crate) fn sign(mut report: Vec<u8>, scalar: u8) -> Vec<u8> {
        let key = p384::ecdsa::SigningKey::from_slice(&[scalar; 48]).unwrap();
        let sig: p384::ecdsa::Signature = key.sign(&report[..SIGNED_LEN]);
        let rs = sig.to_bytes();
        let mut r_le = rs[..48].to_vec();
        r_le.reverse();
        let mut s_le = rs[48..].to_vec();
        s_le.reverse();
        report[0x2A0..0x2E8].fill(0);
        report[0x2E8..0x330].fill(0);
        report[0x2A0..0x2A0 + 48].copy_from_slice(&r_le);
        report[0x2E8..0x2E8 + 48].copy_from_slice(&s_le);
        report
    }

    /// A signed Genoa evidence payload, `edit` applied before signing.
    pub(crate) fn genoa_evidence(edit: impl FnOnce(&mut Spec)) -> Vec<u8> {
        let mut spec = Spec::genoa();
        edit(&mut spec);
        evidence_payload(&sign(spec.bytes(), CURRENT), VCEK_CURRENT)
    }

    /// A Genoa report signed by the TEST VLEK (`KEY_INFO.SIGNING_KEY` 1).
    pub(crate) fn vlek_evidence(edit: impl FnOnce(&mut Spec)) -> Vec<u8> {
        let mut spec = Spec::genoa();
        spec.key_info = snp_report::SIGNING_KEY_VLEK << 2;
        edit(&mut spec);
        evidence_payload(&sign(spec.bytes(), VLEK), VLEK_TEST)
    }

    /// A policy that accepts the TEST provider's VLEK.
    pub(crate) fn test_csp_policy() -> SnpPolicy {
        SnpPolicy {
            vlek_csp_ids: vec![TEST_CSP.into()],
            ..SnpPolicy::default()
        }
    }

    fn verify(payload: &[u8]) -> Result<VerifiedSnp, SnpRejected> {
        verify_with(payload, &SnpPolicy::default())
    }

    fn verify_with(payload: &[u8], policy: &SnpPolicy) -> Result<VerifiedSnp, SnpRejected> {
        verify_snp_report(payload, test_crl(payload), &test_anchors(), policy, NOW)
    }

    fn verify_crl_of(payload: &[u8], crl: &[u8]) -> Result<VerifiedSnp, SnpRejected> {
        verify_snp_report(payload, crl, &test_anchors(), &test_csp_policy(), NOW)
    }

    fn production(payload: &[u8], policy: &SnpPolicy) -> Result<VerifiedSnp, SnpRejected> {
        verify_snp_report(
            payload,
            amd_crl(payload),
            &SnpAnchors::production(),
            policy,
            NOW_REAL,
        )
    }

    /// The real reports' guests accept SMT (POLICY bit 16): judged here
    /// under a policy that allows it, so another check is what decides.
    fn smt_allowed() -> SnpPolicy {
        SnpPolicy {
            smt: SnpSmt::Allowed,
            ..SnpPolicy::default()
        }
    }

    // ── The chain and the signature ──────────────────────────────────────

    #[test]
    fn a_report_signed_by_a_vcek_of_the_pinned_chain_verifies() {
        let v = verify(&genoa_evidence(|_| {})).expect("verifies");
        assert_eq!(v.product, SnpProduct::Genoa);
        assert_eq!(v.signing_key, SnpSigningKey::Vcek);
        assert_eq!(v.csp_id, None);
        assert_eq!(v.report.measurement, [0x4D; 48]);
        assert_eq!(v.report_data(), &[0x64; 64]);
        assert_eq!(
            v.reported_tcb,
            TcbParts {
                fmc: 0,
                boot_loader: 0x0A,
                tee: 0,
                snp: 0x1B,
                microcode: 0x58
            }
        );
        assert_eq!(v.key_sha256_hex, hex(&Sha256::digest(VCEK_CURRENT)));
        assert_eq!(v.minimum.snp, 0x1B);
        assert_eq!(v.minimum.microcode, 0x56);
        assert!(v.minimum_source.contains("Genoa"));
        // The TEST CRL's window, read.
        assert!(v.crl_window.0 <= NOW as i64 && NOW as i64 <= v.crl_window.1);
        // What it did not establish is said, the floor's zeros by name.
        let all = v.limitations.join("\n");
        assert!(
            all.contains("boot loader, TEE, FMC at 0 judge nothing"),
            "{all}"
        );
        assert!(all.contains("CPU only"), "{all}");
        assert!(!all.contains("SMT allowed"), "{all}");
    }

    /// SECURITY: the signature covers every byte before 0x2A0. One flipped
    /// bit anywhere there, or in R or S, and the report is not AMD's.
    #[test]
    fn a_report_altered_after_signing_is_refused() {
        for at in [0x00, 0x08, 0x50, 0x90, 0x180, 0x1A0, 0x29F, 0x2A0, 0x2E8] {
            let mut p = genoa_evidence(|_| {});
            p[at] ^= 0x01;
            let e = verify(&p).unwrap_err();
            assert!(
                matches!(e, SnpRejected::ReportSignature | SnpRejected::Report(_)),
                "byte {at:#x}: {e}"
            );
        }
        // The bytes the signature covers decide: a flipped measurement is a signature failure.
        let mut p = genoa_evidence(|_| {});
        p[0x90] ^= 0x80;
        assert_eq!(verify(&p).unwrap_err(), SnpRejected::ReportSignature);
    }

    /// R and S are 72-byte fields holding 48-byte values: anything above is
    /// not a P-384 scalar, and is refused rather than truncated.
    #[test]
    fn a_signature_component_wider_than_p384_is_refused() {
        let mut p = genoa_evidence(|_| {});
        p[0x2A0 + 60] = 1;
        assert_eq!(verify(&p).unwrap_err(), SnpRejected::ReportSignature);
        let mut p = genoa_evidence(|_| {});
        p[0x2E8 + 71] = 1;
        assert_eq!(verify(&p).unwrap_err(), SnpRejected::ReportSignature);
    }

    /// SECURITY: an ASK with the pinned ASK's name, under another ARK, signs
    /// a VCEK with the genuine key and extensions. The names match; the
    /// signature does not.
    #[test]
    fn a_vcek_from_an_unpinned_ask_of_the_same_name_is_refused() {
        let report = sign(Spec::genoa().bytes(), CURRENT);
        let e = verify(&evidence_payload(&report, VCEK_ROGUE)).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Chain(m) if m.contains("VCEK") && m.contains("signature")),
            "{e}"
        );
    }

    /// Required refusal: the wrong root. A real AMD report under the TEST
    /// anchors, and a TEST report under AMD's: each chain is genuine, and
    /// neither is the pinned one.
    #[test]
    fn a_chain_to_another_root_is_refused_both_ways() {
        let real = evidence_payload(AZURE_GENOA_REPORT, AZURE_GENOA_VCEK);
        let e = verify_with(&real, &smt_allowed()).unwrap_err();
        assert!(matches!(e, SnpRejected::Chain(_)), "{e}");
        let test = genoa_evidence(|_| {});
        let e = production(&test, &SnpPolicy::default()).unwrap_err();
        assert!(matches!(e, SnpRejected::Chain(_)), "{e}");
        // The TEST VLEK under AMD's ASVK: same refusal.
        let e = production(&vlek_evidence(|_| {}), &test_csp_policy()).unwrap_err();
        assert!(matches!(e, SnpRejected::Chain(_)), "{e}");
    }

    /// The SPKI pin is checked, not assumed: a root whose certificate does
    /// not carry the pinned key is refused before anything chains to it.
    #[test]
    fn a_root_that_does_not_match_its_pin_is_refused() {
        let p = genoa_evidence(|_| {});
        let run = |anchors: &SnpAnchors| {
            verify_snp_report(&p, TEST_GENOA_CRL, anchors, &SnpPolicy::default(), NOW).unwrap_err()
        };
        let mut anchors = test_anchors();
        anchors.roots[0].ark_spki_sha256_hex = AMD_ARK_GENOA_SPKI_SHA256_HEX.into();
        assert_eq!(run(&anchors), SnpRejected::RootNotPinned("Genoa".into()));
        let mut anchors = test_anchors();
        anchors.roots[0].ask_der = TEST_TURIN_ASK;
        assert_eq!(
            run(&anchors),
            SnpRejected::RootNotPinned("Genoa ASK".into())
        );
        // The ASVK pin, on a VLEK report.
        let mut anchors = test_anchors();
        anchors.roots[0].asvk_spki_sha256_hex = AMD_ASVK_GENOA_SPKI_SHA256_HEX.into();
        let e = verify_snp_report(
            &vlek_evidence(|_| {}),
            TEST_GENOA_CRL,
            &anchors,
            &test_csp_policy(),
            NOW,
        )
        .unwrap_err();
        assert_eq!(e, SnpRejected::RootNotPinned("Genoa ASVK".into()));
    }

    #[test]
    fn a_product_without_a_pinned_root_is_refused() {
        let mut anchors = test_anchors();
        anchors.roots.retain(|r| r.product != SnpProduct::Genoa);
        let e = verify_snp_report(
            &genoa_evidence(|_| {}),
            TEST_GENOA_CRL,
            &anchors,
            &SnpPolicy::default(),
            NOW,
        )
        .unwrap_err();
        assert_eq!(e, SnpRejected::NoRoot("Genoa".into()));
    }

    #[test]
    fn a_vcek_outside_its_validity_window_is_refused() {
        // The TEST VCEKs are valid 2025-01-01 to 2032-01-01, their ASK to 2045.
        let p = genoa_evidence(|_| {});
        for now in [1_700_000_000, 1_960_000_000] {
            let e = verify_snp_report(
                &p,
                TEST_GENOA_CRL,
                &test_anchors(),
                &SnpPolicy::default(),
                now,
            )
            .unwrap_err();
            assert!(
                matches!(&e, SnpRejected::Chain(m) if m.contains("validity window")),
                "{now}: {e}"
            );
        }
    }

    // ── The key describes the report ────────────────────────────────────

    /// SECURITY: a VCEK is derived per chip and per TCB. A report claiming
    /// another TCB than the VCEK's was not signed by the key that TCB
    /// would have derived.
    #[test]
    fn a_reported_tcb_that_is_not_the_vceks_is_refused() {
        let p = genoa_evidence(|s| s.reported_tcb[7] = 0x59);
        let e = verify(&p).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::KeyMismatch(m) if m.contains("REPORTED_TCB")),
            "{e}"
        );
    }

    #[test]
    fn a_chip_id_that_is_not_the_vceks_is_refused() {
        let p = genoa_evidence(|s| s.chip_id[63] ^= 1);
        let e = verify(&p).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::KeyMismatch(m) if m.contains("CHIP_ID")),
            "{e}"
        );
    }

    /// `MASK_CHIP_KEY` zeroes `CHIP_ID`: a report that names no chip cannot
    /// be tied to its VCEK.
    #[test]
    fn a_report_with_its_chip_id_masked_is_refused() {
        let p = genoa_evidence(|s| {
            s.key_info = 0b10;
            s.chip_id = [0; 64];
        });
        let e = verify(&p).unwrap_err();
        assert!(matches!(e, SnpRejected::SigningKey(_)), "{e}");
    }

    #[test]
    fn a_cpuid_of_another_product_is_refused() {
        for cpuid in [[0x1A, 0x11, 0x01], [0x19, 0x01, 0x01], [0x19, 0x30, 0x00]] {
            let p = genoa_evidence(|s| s.cpuid = cpuid);
            let e = verify(&p).unwrap_err();
            assert!(matches!(e, SnpRejected::Cpuid(_)), "{cpuid:02x?}: {e}");
        }
    }

    #[test]
    fn a_signature_algorithm_other_than_ecdsa_p384_is_refused() {
        let p = genoa_evidence(|s| s.signature_algo = 2);
        let e = verify(&p).unwrap_err();
        assert!(matches!(e, SnpRejected::Report(_)), "{e}");
    }

    // ── VLEK ────────────────────────────────────────────────────────────

    /// A VLEK is the key AMD issues to one cloud provider: its report
    /// verifies to the pinned ASVK, and only for a provider the policy names.
    #[test]
    fn a_vlek_report_verifies_to_the_pinned_asvk_only_for_a_named_provider() {
        let p = vlek_evidence(|_| {});
        let e = verify(&p).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::SigningKey(m) if m.contains(TEST_CSP)),
            "{e}"
        );
        let other = SnpPolicy {
            vlek_csp_ids: vec!["CN=another.invalid".into()],
            ..SnpPolicy::default()
        };
        assert!(matches!(
            verify_with(&p, &other).unwrap_err(),
            SnpRejected::SigningKey(_)
        ));
        let v = verify_with(&p, &test_csp_policy()).expect("named provider");
        assert_eq!(v.signing_key, SnpSigningKey::Vlek);
        assert_eq!(v.csp_id.as_deref(), Some(TEST_CSP));
        assert_eq!(v.key_sha256_hex, hex(&Sha256::digest(VLEK_TEST)));
        let all = v.limitations.join("\n");
        assert!(all.contains("VLEK") && all.contains(TEST_CSP), "{all}");
        assert_eq!(snp_measurements(&v)[0].register, "vlek_sha256");
        // A VLEK names no chip: CHIP_ID is not bound (a stated limitation).
        verify_with(
            &vlek_evidence(|s| s.chip_id = [0x33; 64]),
            &test_csp_policy(),
        )
        .expect("CHIP_ID is not a VLEK's");
        // Its TCB is, as a VCEK's.
        let e = verify_with(
            &vlek_evidence(|s| s.reported_tcb[6] = 0x1C),
            &test_csp_policy(),
        )
        .unwrap_err();
        assert!(matches!(e, SnpRejected::KeyMismatch(_)), "{e}");
    }

    /// `KEY_INFO.SIGNING_KEY` and the certificate must agree: a VLEK under
    /// the ASK's path, or a VCEK under the ASVK's, chains to nothing.
    #[test]
    fn a_key_presented_as_the_other_kind_is_refused() {
        let mut spec = Spec::genoa();
        let vcek_report = sign(spec.bytes(), VLEK);
        let e =
            verify_crl_of(&evidence_payload(&vcek_report, VLEK_TEST), TEST_GENOA_CRL).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Vcek(_) | SnpRejected::Chain(_)),
            "{e}"
        );
        spec.key_info = snp_report::SIGNING_KEY_VLEK << 2;
        let vlek_report = sign(spec.bytes(), CURRENT);
        let e = verify_crl_of(
            &evidence_payload(&vlek_report, VCEK_CURRENT),
            TEST_GENOA_CRL,
        )
        .unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Vlek(_) | SnpRejected::Chain(_)),
            "{e}"
        );
        // A product with no pinned ASVK refuses a VLEK by its chain.
        let mut t = Spec::turin();
        t.key_info = snp_report::SIGNING_KEY_VLEK << 2;
        let e = verify_with(
            &evidence_payload(&sign(t.bytes(), TURIN), VCEK_TURIN),
            &test_csp_policy(),
        )
        .unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Vlek(_) | SnpRejected::Chain(_)),
            "{e}"
        );
    }

    /// RFC 5280, sections 4.1.1.2 and 5.1.1.2, byte for byte, on AMD's real
    /// Genoa ARK and CRL: one bit of the algorithm after the signed part
    /// changed, the two tags x509-parser does not read included, and neither
    /// is taken.
    #[test]
    fn amds_ark_and_crl_with_one_bit_of_the_outer_algorithm_changed_are_refused() {
        /// Where the outer `AlgorithmIdentifier` is: after the signed part.
        fn outer(der: &[u8], tbs: &[u8]) -> std::ops::Range<usize> {
            let at = tbs.as_ptr() as usize - der.as_ptr() as usize + tbs.len();
            assert_eq!(der[at], 0x30);
            assert!(der[at + 1] < 0x80, "a short length");
            at..at + 2 + usize::from(der[at + 1])
        }
        let anchors = SnpAnchors::production();
        let root = anchors.root(SnpProduct::Genoa).unwrap();
        let ark = verify_ark(root, NOW_REAL).expect("AMD's Genoa ARK, intact");
        let (inter, _) = verify_intermediate(root, &ark, SnpSigningKey::Vcek, NOW_REAL).unwrap();
        let crl = |der: &[u8]| {
            verify_crl(
                der,
                SnpProduct::Genoa,
                &ark,
                "ASK",
                inter.raw_serial(),
                NOW_REAL,
            )
        };
        crl(AMD_GENOA_CRL).expect("AMD's Genoa CRL, intact");

        let mut parsed = 0;
        let at = outer(root.ark_der, ark.ark.tbs_certificate.as_ref());
        for (byte, bit) in at.flat_map(|byte| (0..8).map(move |bit| (byte, bit))) {
            let mut m = root.ark_der.to_vec();
            m[byte] ^= 1 << bit;
            let Ok((_, c)) = X509Certificate::from_der(&m) else {
                continue;
            };
            parsed += 1;
            let e = signed_by(&c, &ark.key, "ARK", NOW_REAL)
                .expect_err("an ARK with one bit of its outer algorithm changed verifies");
            assert!(
                e.contains("algorithms differ"),
                "byte {byte} bit {bit}: {e}"
            );
        }
        assert!(parsed >= 2, "x509-parser reads {parsed} of them");

        let (_, intact) = CertificateRevocationList::from_der(AMD_GENOA_CRL).unwrap();
        let at = outer(AMD_GENOA_CRL, intact.tbs_cert_list.as_ref());
        for (byte, bit) in at.flat_map(|byte| (0..8).map(move |bit| (byte, bit))) {
            let mut m = AMD_GENOA_CRL.to_vec();
            m[byte] ^= 1 << bit;
            let e = crl(&m).expect_err("a CRL with one bit of its outer algorithm changed passes");
            assert!(
                matches!(e, SnpRejected::Crl(_)),
                "byte {byte} bit {bit}: {e}"
            );
        }
    }

    // ── Revocation: AMD's CRL ───────────────────────────────────────────

    /// AMD's CRLs as KDS served them: each signed by its pinned ARK, and
    /// Genoa's lists the first Genoa ASK (0x020001), not the pinned one
    /// (0x020002).
    #[test]
    fn amds_crls_verify_under_the_pinned_arks_and_name_the_revoked_genoa_ask() {
        let anchors = SnpAnchors::production();
        for (product, crl) in [
            (SnpProduct::Milan, AMD_MILAN_CRL),
            (SnpProduct::Genoa, AMD_GENOA_CRL),
            (SnpProduct::Turin, AMD_TURIN_CRL),
        ] {
            let root = anchors.root(product).unwrap();
            let ark = verify_ark(root, NOW_REAL).unwrap();
            for kind in [SnpSigningKey::Vcek, SnpSigningKey::Vlek] {
                let (inter, _) = verify_intermediate(root, &ark, kind, NOW_REAL).unwrap();
                let w = verify_crl(
                    crl,
                    product,
                    &ark,
                    kind.intermediate(),
                    inter.raw_serial(),
                    NOW_REAL,
                )
                .unwrap_or_else(|e| panic!("{product:?} {kind:?}: {e}"));
                assert!(w.0 < NOW_REAL as i64 && NOW_REAL as i64 <= w.1);
            }
        }
        let genoa = anchors.root(SnpProduct::Genoa).unwrap();
        let ark = verify_ark(genoa, NOW_REAL).unwrap();
        let e = verify_crl(
            AMD_GENOA_CRL,
            SnpProduct::Genoa,
            &ark,
            "ASK",
            &[2, 0, 1],
            NOW_REAL,
        )
        .unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Revoked(m) if m.contains("ASK")),
            "{e}"
        );
        // Another product's CRL is not issued by this ARK.
        let e = verify_crl(
            AMD_MILAN_CRL,
            SnpProduct::Genoa,
            &ark,
            "ASK",
            &[2, 0, 2],
            NOW_REAL,
        )
        .unwrap_err();
        assert!(matches!(e, SnpRejected::Crl(_)), "{e}");
        // Past its nextUpdate (2026-12-09), the same CRL is refused.
        let e = verify_crl(
            AMD_GENOA_CRL,
            SnpProduct::Genoa,
            &ark,
            "ASK",
            &[2, 0, 2],
            1_797_000_000,
        )
        .unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Crl(m) if m.contains("window")),
            "{e}"
        );
        assert_eq!(
            kds_crl_url(KDS_BASE_URL, SnpProduct::Genoa, SnpSigningKey::Vcek),
            "https://kdsintf.amd.com/vcek/v1/Genoa/crl"
        );
        assert_eq!(
            kds_crl_url(
                "https://kds.example/",
                SnpProduct::Turin,
                SnpSigningKey::Vlek
            ),
            "https://kds.example/vlek/v1/Turin/crl"
        );
    }

    /// SECURITY: a report whose ASK AMD has revoked is refused, though its
    /// chain, its signature and its every field verify.
    #[test]
    fn a_report_under_a_revoked_ask_is_refused() {
        let e = verify_crl_of(&genoa_evidence(|_| {}), CRL_ASK_REVOKED).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Revoked(m) if m.contains("ASK") && m.contains("02")),
            "{e}"
        );
        // The same CRL leaves the ASVK, and so a VLEK report, standing.
        verify_crl_of(&vlek_evidence(|_| {}), CRL_ASK_REVOKED).expect("ASVK not revoked");
    }

    #[test]
    fn a_vlek_report_under_a_revoked_asvk_is_refused() {
        let e = verify_crl_of(&vlek_evidence(|_| {}), CRL_ASVK_REVOKED).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Revoked(m) if m.contains("ASVK")),
            "{e}"
        );
        verify_crl_of(&genoa_evidence(|_| {}), CRL_ASVK_REVOKED).expect("ASK not revoked");
    }

    /// Freshness: a CRL is read only inside its thisUpdate..nextUpdate
    /// window, and one without a nextUpdate has no window.
    #[test]
    fn a_crl_outside_its_window_is_refused() {
        let e = verify_crl_of(&genoa_evidence(|_| {}), CRL_EXPIRED).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Crl(m) if m.contains("window")),
            "{e}"
        );
        // Before its thisUpdate (2026-01-27), the good one too.
        let e = verify_snp_report(
            &genoa_evidence(|_| {}),
            TEST_GENOA_CRL,
            &test_anchors(),
            &SnpPolicy::default(),
            1_768_000_000,
        )
        .unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Crl(m) if m.contains("window")),
            "{e}"
        );
    }

    /// SECURITY: a CRL with the pinned ARK's name, signed by another key,
    /// listing nothing that matters: the name matches, the signature does not.
    #[test]
    fn a_crl_signed_by_a_rogue_ark_of_the_same_name_is_refused() {
        let e = verify_crl_of(&genoa_evidence(|_| {}), CRL_ROGUE).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Crl(m) if m.contains("signature")),
            "{e}"
        );
    }

    #[test]
    fn a_crl_with_a_critical_extension_this_verifier_does_not_read_is_refused() {
        let e = verify_crl_of(&genoa_evidence(|_| {}), CRL_CRITICAL).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Crl(m) if m.contains("critical")),
            "{e}"
        );
    }

    #[test]
    fn a_crl_of_another_product_or_malformed_is_refused() {
        let p = genoa_evidence(|_| {});
        for crl in [TEST_TURIN_CRL, &[][..], &[0x30, 0x03, 0x02, 0x01, 0x01][..]] {
            let e = verify_crl_of(&p, crl).unwrap_err();
            assert!(matches!(e, SnpRejected::Crl(_)), "{e}");
        }
        let mut trailing = TEST_GENOA_CRL.to_vec();
        trailing.push(0);
        let e = verify_crl_of(&p, &trailing).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Crl(m) if m.contains("after")),
            "{e}"
        );
        // One bit of the signed list, and the ARK's signature no longer holds.
        let mut flipped = TEST_GENOA_CRL.to_vec();
        let at = flipped.len() / 3;
        flipped[at] ^= 1;
        assert!(verify_crl_of(&p, &flipped).is_err());
    }

    /// The CRL URL is chosen from what the payload claims; the claim is
    /// then proved by the chain (the CRL must be the proved product's ARK's).
    #[test]
    fn the_claimed_product_names_the_crl_to_fetch() {
        assert_eq!(
            evidence_product(&genoa_evidence(|_| {})),
            Ok((SnpProduct::Genoa, SnpSigningKey::Vcek))
        );
        assert_eq!(
            evidence_product(&vlek_evidence(|_| {})),
            Ok((SnpProduct::Genoa, SnpSigningKey::Vlek))
        );
        assert_eq!(
            evidence_product(&evidence_payload(AWS_VLEK_REPORT, AWS_VLEK)),
            Ok((SnpProduct::Milan, SnpSigningKey::Vlek))
        );
        assert!(evidence_product(&[0u8; 10]).is_err());
    }

    // ── The guest policy ────────────────────────────────────────────────

    /// Required refusal: debug. The host can read and write a debug
    /// guest's memory, whatever the chain says.
    #[test]
    fn a_debug_guest_is_refused_though_its_chain_and_signature_verify() {
        let p = genoa_evidence(|s| s.policy |= snp_report::POLICY_DEBUG);
        assert_eq!(verify(&p).unwrap_err(), SnpRejected::Debug);
        // Under every policy this crate takes.
        let lenient = SnpPolicy {
            smt: SnpSmt::Allowed,
            minimum: Some(SB3019_GENOA),
            ..test_csp_policy()
        };
        assert_eq!(verify_with(&p, &lenient).unwrap_err(), SnpRejected::Debug);
        let p = vlek_evidence(|s| s.policy |= snp_report::POLICY_DEBUG);
        assert_eq!(verify_with(&p, &lenient).unwrap_err(), SnpRejected::Debug);
    }

    /// Migration refused, both forms: the policy allowing an agent, and an
    /// agent bound.
    #[test]
    fn a_migration_agent_is_refused() {
        let p = genoa_evidence(|s| s.policy |= snp_report::POLICY_MIGRATE_MA);
        assert!(matches!(
            verify(&p).unwrap_err(),
            SnpRejected::MigrationAgent(_)
        ));
        let p = genoa_evidence(|s| s.report_id_ma = [0x11; 32]);
        assert!(matches!(
            verify(&p).unwrap_err(),
            SnpRejected::MigrationAgent(_)
        ));
    }

    #[test]
    fn policy_bit_17_clear_is_refused() {
        let p = genoa_evidence(|s| s.policy &= !snp_report::POLICY_RESERVED_MUST_BE_ONE);
        assert_eq!(verify(&p).unwrap_err(), SnpRejected::PolicyReserved);
    }

    /// SMT as the written policy says: forbidden by default, the guest's
    /// policy and the platform both; allowed, it is said in the verdict.
    #[test]
    fn smt_is_refused_by_default_and_accepted_only_when_the_policy_allows_it() {
        let in_policy = genoa_evidence(|s| s.policy |= snp_report::POLICY_SMT);
        let on_platform = genoa_evidence(|s| s.platform_info |= 1);
        for p in [&in_policy, &on_platform] {
            let e = verify(p).unwrap_err();
            assert!(matches!(e, SnpRejected::Smt(_)), "{e}");
        }
        let both = genoa_evidence(|s| {
            s.policy |= snp_report::POLICY_SMT;
            s.platform_info |= 1;
        });
        let v = verify_with(&both, &smt_allowed()).expect("SMT allowed");
        let all = v.limitations.join("\n");
        assert!(
            all.contains("SMT allowed") && all.contains("SMT_EN 1"),
            "{all}"
        );
    }

    /// SECURITY: a bit this verifier does not know is a meaning it would
    /// silently ignore: refused, in POLICY, KEY_INFO and PLATFORM_INFO.
    #[test]
    fn a_bit_this_verifier_does_not_know_is_refused() {
        for edit in [
            (|s: &mut Spec| s.policy |= 1 << 26) as fn(&mut Spec),
            |s| s.policy |= 1 << 63,
            |s| s.key_info |= 1 << 5,
            |s| s.key_info |= 1 << 31,
            |s| s.platform_info |= 1 << 6,
            |s| s.platform_info |= 1 << 63,
        ] {
            let e = verify(&genoa_evidence(edit)).unwrap_err();
            assert!(matches!(e, SnpRejected::Unknown(_)), "{e}");
        }
    }

    /// A reserved byte set is a report this parser does not understand.
    #[test]
    fn a_reserved_byte_set_is_refused() {
        for at in [
            0x4C, 0x4F, 0x18B, 0x19F, 0x1EB, 0x1EF, 0x1F8, 0x29F, 0x330, 0x49F,
        ] {
            let mut bytes = Spec::genoa().bytes();
            bytes[at] = 1;
            let p = evidence_payload(&sign(bytes, CURRENT), VCEK_CURRENT);
            let e = verify(&p).unwrap_err();
            assert!(
                matches!(&e, SnpRejected::Report(m) if m.contains("reserved")),
                "{at:#x}: {e}"
            );
        }
    }

    /// SECURITY: a report requested from VMPL 1-3 carries a REPORT_DATA a
    /// less privileged layer chose; one the HOST requested (0xFFFFFFFF)
    /// carries the host's.
    #[test]
    fn a_report_not_requested_at_vmpl0_is_refused() {
        for vmpl in [1, 3, 0xFFFF_FFFF] {
            let e = verify(&genoa_evidence(|s| s.vmpl = vmpl)).unwrap_err();
            assert!(matches!(e, SnpRejected::Vmpl(_)), "{vmpl}: {e}");
        }
        let e = verify(&genoa_evidence(|s| s.vmpl = 0xFFFF_FFFF)).unwrap_err();
        assert!(e.to_string().contains("host"), "{e}");
    }

    #[test]
    fn a_product_the_policy_does_not_name_is_refused() {
        let only_turin = SnpPolicy {
            products: Some(vec![SnpProduct::Turin]),
            ..SnpPolicy::default()
        };
        assert_eq!(
            verify_with(&genoa_evidence(|_| {}), &only_turin).unwrap_err(),
            SnpRejected::ProductRefused("Genoa".into())
        );
        let genoa = SnpPolicy {
            products: Some(vec![SnpProduct::Genoa]),
            ..SnpPolicy::default()
        };
        verify_with(&genoa_evidence(|_| {}), &genoa).expect("named");
    }

    // ── The TCB ─────────────────────────────────────────────────────────

    /// SECURITY: a genuine report from a chip below the default minimum —
    /// signed by the VCEK AMD derived for that old TCB.
    #[test]
    fn a_tcb_below_the_default_minimum_is_refused() {
        let p = old_genoa();
        let e = verify(&p).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::TcbTooLow(m) if m.contains("REPORTED_TCB") && m.contains("SNP")),
            "{e}"
        );
        // The same report under an explicit minimum at its own level: it is the
        // minimum that refused, nothing else.
        let lenient = SnpPolicy {
            minimum: Some(SB3019_GENOA),
            ..SnpPolicy::default()
        };
        verify_with(&p, &lenient).expect("explicit minimum");
    }

    fn old_genoa() -> Vec<u8> {
        let mut spec = Spec::genoa();
        spec.reported_tcb = GENOA_TCB_OLD;
        spec.current_tcb = GENOA_TCB_OLD;
        spec.committed_tcb = GENOA_TCB_OLD;
        spec.launch_tcb = GENOA_TCB_OLD;
        evidence_payload(&sign(spec.bytes(), OLD), VCEK_OLD)
    }

    /// Required refusal: a TCB under the written floor. The floor judges
    /// what the bulletins leave at 0 (boot loader, TEE), and raises,
    /// never lowers, what they set.
    #[test]
    fn a_tcb_under_the_written_floor_is_refused_and_the_floor_never_lowers() {
        let p = genoa_evidence(|_| {});
        let floor = |f: SnpTcbMinimum| SnpPolicy {
            floor: Some(f),
            ..SnpPolicy::default()
        };
        let zero = SnpTcbMinimum {
            fmc: 0,
            boot_loader: 0,
            tee: 0,
            snp: 0,
            microcode: 0,
            require_alias_check: false,
        };
        let e = verify_with(
            &p,
            &floor(SnpTcbMinimum {
                boot_loader: 0x0B,
                ..zero
            }),
        )
        .unwrap_err();
        assert!(
            matches!(&e, SnpRejected::TcbTooLow(m) if m.contains("boot loader") && m.contains("floor")),
            "{e}"
        );
        let e = verify_with(&p, &floor(SnpTcbMinimum { tee: 1, ..zero })).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::TcbTooLow(m) if m.contains("TEE")),
            "{e}"
        );
        let e = verify_with(
            &p,
            &floor(SnpTcbMinimum {
                microcode: 0x59,
                ..zero
            }),
        )
        .unwrap_err();
        assert!(
            matches!(&e, SnpRejected::TcbTooLow(m) if m.contains("microcode")),
            "{e}"
        );
        // At the report's level: verifies, the floor named in the verdict.
        let v = verify_with(
            &p,
            &floor(SnpTcbMinimum {
                boot_loader: 0x0A,
                snp: 0x1B,
                microcode: 0x58,
                ..zero
            }),
        )
        .expect("at the floor");
        assert_eq!((v.minimum.boot_loader, v.minimum.microcode), (0x0A, 0x58));
        assert!(v.minimum_source.contains("floor"), "{}", v.minimum_source);
        assert!(v.limitations.join("\n").contains("TEE, FMC at 0"));
        // A floor below the default does not lower it: the old chip stays refused.
        let e = verify_with(&old_genoa(), &floor(SB3019_GENOA)).unwrap_err();
        assert!(matches!(e, SnpRejected::TcbTooLow(_)), "{e}");
    }

    /// A guest launched under old firmware, the firmware since updated: its
    /// VCEK is current, its LAUNCH_TCB is not. Its memory was exposed then.
    #[test]
    fn a_launch_or_committed_tcb_below_the_minimum_is_refused() {
        for field in ["LAUNCH_TCB", "COMMITTED_TCB", "CURRENT_TCB"] {
            let p = genoa_evidence(|s| match field {
                "LAUNCH_TCB" => s.launch_tcb = GENOA_TCB_OLD,
                "COMMITTED_TCB" => s.committed_tcb = GENOA_TCB_OLD,
                _ => s.current_tcb = GENOA_TCB_OLD,
            });
            let e = verify(&p).unwrap_err();
            assert!(
                matches!(&e, SnpRejected::TcbTooLow(m) if m.contains(field)),
                "{field}: {e}"
            );
        }
    }

    #[test]
    fn a_genoa_without_the_badram_alias_check_is_refused() {
        let p = genoa_evidence(|s| s.platform_info = 0x04);
        assert_eq!(verify(&p).unwrap_err(), SnpRejected::AliasCheck);
    }

    #[test]
    fn a_cpu_line_without_a_default_minimum_is_refused_until_one_is_given() {
        let p = genoa_evidence(|s| s.cpuid = [0x19, 0x18, 0x00]);
        let e = verify(&p).unwrap_err();
        assert!(matches!(e, SnpRejected::NoDefaultMinimum(_)), "{e}");
        let explicit = SnpPolicy {
            minimum: Some(SnpTcbMinimum {
                snp: 0x1B,
                microcode: 0x58,
                ..SB3019_GENOA
            }),
            ..SnpPolicy::default()
        };
        verify_with(&p, &explicit).expect("explicit minimum");
    }

    /// The default table, line by line, and the v2 fallback to the
    /// product's highest level.
    #[test]
    fn the_default_minimum_is_the_bulletins_per_line() {
        let line = |product, cpuid: Option<[u8; 3]>| {
            default_minimum(
                product,
                cpuid.map(|[family, model, stepping]| Cpuid {
                    family,
                    model,
                    stepping,
                }),
            )
            .map(|(m, _)| (m.snp, m.microcode, m.require_alias_check))
        };
        use SnpProduct::*;
        assert_eq!(line(Genoa, Some([0x19, 0x11, 1])), Ok((0x1B, 0x56, true)));
        assert_eq!(line(Genoa, Some([0x19, 0x11, 2])), Ok((0x1B, 0x51, true)));
        assert_eq!(line(Genoa, Some([0x19, 0xA0, 2])), Ok((0x1B, 0x1B, true)));
        assert_eq!(line(Genoa, None), Ok((0x1B, 0x56, true)));
        assert_eq!(line(Turin, Some([0x1A, 0x02, 1])), Ok((0x04, 0x51, false)));
        assert_eq!(line(Turin, Some([0x1A, 0x11, 0])), Ok((0x04, 0x4E, false)));
        assert_eq!(line(Turin, None), Ok((0x04, 0x51, false)));
        assert_eq!(line(Milan, Some([0x19, 0x01, 1])), Ok((0x1B, 0xDE, true)));
        assert_eq!(line(Milan, Some([0x19, 0x01, 2])), Ok((0x1B, 0x47, true)));
        assert_eq!(line(Milan, None), Ok((0x1B, 0xDE, true)));
        assert!(line(Genoa, Some([0x19, 0x18, 0])).is_err());
        assert!(line(Turin, Some([0x1A, 0x10, 0])).is_err());
    }

    // ── Turin, v5 ───────────────────────────────────────────────────────

    /// Turin's TCB layout (FMC first, SNP at byte 3) and its 8-byte hwID.
    #[test]
    fn a_turin_report_verifies_with_turins_layout() {
        let p = evidence_payload(&sign(Spec::turin().bytes(), TURIN), VCEK_TURIN);
        let v = verify(&p).expect("verifies");
        assert_eq!(v.product, SnpProduct::Turin);
        assert_eq!(
            v.reported_tcb,
            TcbParts {
                fmc: 1,
                boot_loader: 1,
                tee: 1,
                snp: 0x04,
                microcode: 0x52
            }
        );
        // Turin is not on the BadRAM list: no alias check by default.
        assert!(!v.minimum.require_alias_check);
        // The 8 bytes of CHIP_ID the hwID names are bound to it.
        let mut spec = Spec::turin();
        spec.chip_id[0] ^= 1;
        let e = verify(&evidence_payload(&sign(spec.bytes(), TURIN), VCEK_TURIN)).unwrap_err();
        assert!(matches!(e, SnpRejected::KeyMismatch(_)), "{e}");
        // The FMC component is judged on Turin by the written floor.
        let floor = SnpPolicy {
            floor: Some(SnpTcbMinimum {
                fmc: 2,
                boot_loader: 0,
                tee: 0,
                snp: 0,
                microcode: 0,
                require_alias_check: false,
            }),
            ..SnpPolicy::default()
        };
        let e = verify_with(&p, &floor).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::TcbTooLow(m) if m.contains("FMC")),
            "{e}"
        );
    }

    /// v5 is read from Table 23 of ABI rev. 1.58: its mitigation vectors
    /// are reported, not judged (a stated limitation).
    #[test]
    fn a_v5_report_verifies_and_carries_its_mitigation_vectors() {
        let mut spec = Spec::genoa();
        spec.version = 5;
        let mut bytes = spec.bytes();
        bytes[0x1F8..0x200].copy_from_slice(&0x36u64.to_le_bytes());
        bytes[0x200..0x208].copy_from_slice(&0x16u64.to_le_bytes());
        let v = verify(&evidence_payload(&sign(bytes, CURRENT), VCEK_CURRENT)).expect("v5");
        assert_eq!(v.report.launch_mit_vector, Some(0x36));
        assert_eq!(v.report.current_mit_vector, Some(0x16));
    }

    #[test]
    fn a_version_4_report_is_refused() {
        let mut spec = Spec::genoa();
        spec.version = 4;
        let p = evidence_payload(&sign(spec.bytes(), CURRENT), VCEK_CURRENT);
        let e = verify(&p).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Report(m) if m.contains("version 4")),
            "{e}"
        );
    }

    #[test]
    fn a_payload_too_short_for_a_report_and_a_key_is_refused() {
        for len in [0, REPORT_LEN - 1, REPORT_LEN] {
            let e = verify(&genoa_evidence(|_| {})[..len]).unwrap_err();
            assert!(matches!(e, SnpRejected::Evidence(_)), "{len}: {e}");
        }
        let mut p = genoa_evidence(|_| {});
        p.push(0);
        let e = verify(&p).unwrap_err();
        assert!(matches!(e, SnpRejected::Vcek(_)), "trailing byte: {e}");
    }

    // ── Real AMD reports ────────────────────────────────────────────────

    /// The pins and the embedded certificates agree, and each chain (ASK
    /// and ASVK) verifies to its ARK: a failure means a file or a constant
    /// changed, both review events.
    #[test]
    fn the_production_pins_match_amds_kds_certificates() {
        let anchors = SnpAnchors::production();
        let names: Vec<_> = anchors.roots.iter().map(|r| r.product.name()).collect();
        assert_eq!(names, ["Milan", "Genoa", "Turin"]);
        assert_eq!(anchors.kds_base_url, "https://kdsintf.amd.com");
        for root in &anchors.roots {
            assert_eq!(spki_hex_of(root.ark_der), root.ark_spki_sha256_hex);
            assert_eq!(spki_hex_of(root.ask_der), root.ask_spki_sha256_hex);
            assert_eq!(spki_hex_of(root.asvk_der), root.asvk_spki_sha256_hex);
            let ark = verify_ark(root, NOW_REAL)
                .unwrap_or_else(|e| panic!("{}: {e}", root.product.name()));
            for kind in [SnpSigningKey::Vcek, SnpSigningKey::Vlek] {
                verify_intermediate(root, &ark, kind, NOW_REAL)
                    .unwrap_or_else(|e| panic!("{} {kind:?}: {e}", root.product.name()));
            }
            verify_root(root, NOW).unwrap();
        }
        // DER digests as fetched (tests/VENDORED.md).
        for (der, sha) in [
            (AMD_ARK_MILAN_DER, "69d063b4"),
            (AMD_ARK_GENOA_DER, "4c6598d1"),
            (AMD_ARK_TURIN_DER, "1f084161"),
            (AMD_ASVK_MILAN_DER, "c5e081f5"),
            (AMD_ASVK_GENOA_DER, "197e6107"),
            (AMD_ASVK_TURIN_DER, "104e10a8"),
            (AMD_MILAN_CRL, "c06e8684"),
            (AMD_GENOA_CRL, "d803150f"),
            (AMD_TURIN_CRL, "949c93ca"),
        ] {
            assert!(hex(&Sha256::digest(der)).starts_with(sha), "{sha}");
        }
    }

    /// A real Genoa report (Azure, trustee test data): AMD's chain, AMD's
    /// signature, AMD's CRL of 2026-10-27, and a TCB at AMD-SB-3019's level
    /// — below the default. Its guest accepts SMT: refused by default.
    #[test]
    fn a_real_genoa_report_verifies_under_amds_roots_and_crl_and_is_held_to_the_default_minimum() {
        let p = evidence_payload(AZURE_GENOA_REPORT, AZURE_GENOA_VCEK);
        let e = production(&p, &SnpPolicy::default()).unwrap_err();
        assert!(matches!(e, SnpRejected::Smt(_)), "{e}");
        let e = production(&p, &smt_allowed()).unwrap_err();
        assert!(
            matches!(&e, SnpRejected::TcbTooLow(m) if m.contains("SNP 0x17")),
            "{e}"
        );
        let at_its_level = SnpPolicy {
            minimum: Some(SB3019_GENOA),
            ..smt_allowed()
        };
        let v = production(&p, &at_its_level).expect("AMD's chain, signature and CRL");
        assert_eq!(v.product, SnpProduct::Genoa);
        assert_eq!(v.report.version, 3);
        assert_eq!(
            v.report.cpuid,
            Some(Cpuid {
                family: 0x19,
                model: 0x11,
                stepping: 1
            })
        );
        assert_eq!(v.reported_tcb.snp, 0x17);
        assert_eq!(v.reported_tcb.microcode, 0x54);
        assert!(hex(&v.report.measurement).starts_with("f57dc09a507c6ecd"));
        // One bit of the real report, and it is no longer AMD's.
        let mut tampered = p.clone();
        tampered[0x90] ^= 1;
        assert_eq!(
            production(&tampered, &at_its_level).unwrap_err(),
            SnpRejected::ReportSignature
        );
        // One bit of the real VCEK's signature, and the chain breaks.
        let mut vcek = AZURE_GENOA_VCEK.to_vec();
        let last = vcek.len() - 1;
        vcek[last] ^= 1;
        let e =
            production(&evidence_payload(AZURE_GENOA_REPORT, &vcek), &at_its_level).unwrap_err();
        assert!(matches!(e, SnpRejected::Chain(_)), "{e}");
        // AMD's Genoa CRL, past its nextUpdate: refused, as any stale CRL.
        let e = verify_snp_report(
            &p,
            AMD_GENOA_CRL,
            &SnpAnchors::production(),
            &at_its_level,
            1_797_000_000,
        )
        .unwrap_err();
        assert!(
            matches!(&e, SnpRejected::Crl(m) if m.contains("window")),
            "{e}"
        );
    }

    /// 2026-10-28: the Scaleway Elastic Metal host of 2026-10-28, as this verifier
    /// judges it. Genuine (AMD's chain, signature and CRL at its own level).
    /// Refused by default under both policies: its guest's POLICY.SMT is set.
    /// With SMT allowed, refused under both: AMD-SB-3020 is a year old on
    /// 2026-10-13 and imposed by both, SNP 0x17 < 0x1b. AMD-SB-3027 is named
    /// under transparent, imposed under strict; AMD-SB-3015's alias check is
    /// not complete (PLATFORM_INFO 0x4).
    #[test]
    fn the_scaleway_host_of_2026_10_28_under_both_tcb_policies() {
        use crate::tcb::TcbMode::{Strict, Transparent};
        let p = evidence_payload(SCALEWAY_SIENA_REPORT, SCALEWAY_SIENA_VCEK);
        let its_level = SnpPolicy {
            minimum: Some(SnpTcbMinimum {
                fmc: 0,
                boot_loader: 0x09,
                tee: 0,
                snp: 0x17,
                microcode: 0x15,
                require_alias_check: false,
            }),
            ..smt_allowed()
        };
        let v = production(&p, &its_level).expect("AMD's chain, signature and CRL");
        assert_eq!(v.product, SnpProduct::Genoa);
        assert_eq!(
            v.report.cpuid,
            Some(Cpuid {
                family: 0x19,
                model: 0xA0,
                stepping: 2
            })
        );
        assert_eq!((v.report.policy, v.report.platform_info), (0x30000, 0x4));
        assert!(!v.report.alias_check_complete());
        for mode in [Transparent, Strict] {
            let e = production(
                &p,
                &SnpPolicy {
                    tcb_mode: mode,
                    ..SnpPolicy::default()
                },
            )
            .unwrap_err();
            assert!(matches!(e, SnpRejected::Smt(_)), "{mode:?}: {e}");
        }
        let refused = |mode| {
            production(
                &p,
                &SnpPolicy {
                    tcb_mode: mode,
                    ..smt_allowed()
                },
            )
            .unwrap_err()
            .to_string()
        };
        assert_eq!(
            refused(Transparent),
            "TCB below the minimum: REPORTED_TCB: SNP 0x17 < 0x1b (default, Bergamo or Siena, \
             transparent TCB policy: AMD-SB-3015, AMD-SB-3020 imposed; AMD-SB-3027 named, not \
             imposed (under a year old))"
        );
        assert_eq!(
            refused(Strict),
            "TCB below the minimum: REPORTED_TCB: SNP 0x17 < 0x1b (default, Bergamo or Siena \
             (AMD-SB-3020, AMD-SB-3027, AMD-SB-3015))"
        );
        let (bytes, _) = split_evidence(&p).unwrap();
        let report = snp_report::parse(bytes).unwrap();
        let standings = |mode, at| {
            let (line, b) = bulletin_standings(&report, SnpProduct::Genoa, mode, at).unwrap();
            assert_eq!(line, "Bergamo or Siena");
            b.into_iter()
                .map(|s| (s.bulletin.id, s.imposed, s.not_met))
                .collect::<Vec<_>>()
        };
        let alias = || {
            vec!["alias check not complete (PLATFORM_INFO.ALIAS_CHECK_COMPLETE clear)".to_string()]
        };
        let sb3020 = || {
            vec![
                "SNP 0x17 < 0x1b".to_string(),
                "microcode 0x15 < 0x1b".to_string(),
            ]
        };
        let sb3027 = || vec!["microcode 0x15 < 0x1b".to_string()];
        assert_eq!(
            standings(Transparent, NOW_REAL),
            [
                ("AMD-SB-3015", true, alias()),
                ("AMD-SB-3020", true, sb3020()),
                ("AMD-SB-3027", false, sb3027()),
            ]
        );
        assert_eq!(
            standings(Strict, NOW_REAL),
            [
                ("AMD-SB-3015", true, alias()),
                ("AMD-SB-3020", true, sb3020()),
                ("AMD-SB-3027", true, sb3027()),
            ]
        );
        // 2026-06-01: AMD-SB-3020 not a year old, named only; 2027-01-15
        // and after: AMD-SB-3027 imposed too.
        let named: Vec<bool> = standings(Transparent, 1_780_272_000)
            .iter()
            .map(|s| s.1)
            .collect();
        assert_eq!(named, [true, false, false]);
        let all: Vec<bool> = standings(
            Transparent,
            AMD_SB_3027.published_unix + SNP_BULLETIN_GRACE_SECS,
        )
        .iter()
        .map(|s| s.1)
        .collect();
        assert_eq!(all, [true, true, true]);
    }

    /// SECURITY (2026-10-28): under `Strict`, and under `Transparent` once every
    /// bulletin is a year old, the minimum of every CPU line is
    /// `default_minimum`'s, field by field; before that, `Transparent` asks
    /// no more than it.
    #[test]
    fn the_bulletin_floors_add_up_to_the_default_minimum() {
        use crate::tcb::TcbMode::{Strict, Transparent};
        let lines = [
            (SnpProduct::Milan, 0x01, 1),
            (SnpProduct::Milan, 0x01, 2),
            (SnpProduct::Genoa, 0x11, 1),
            (SnpProduct::Genoa, 0x11, 2),
            (SnpProduct::Genoa, 0xA0, 2),
            (SnpProduct::Turin, 0x02, 1),
            (SnpProduct::Turin, 0x11, 0),
        ];
        let later = AMD_SB_3027.published_unix + SNP_BULLETIN_GRACE_SECS;
        for (product, model, stepping) in lines {
            for cpuid in [
                Some(Cpuid {
                    family: 0x19,
                    model,
                    stepping,
                }),
                None,
            ] {
                let (default, source) = default_minimum(product, cpuid).unwrap();
                let (strict, strict_source, ..) =
                    mode_minimum(product, cpuid, Strict, NOW_REAL).unwrap();
                assert_eq!(
                    (strict, strict_source.as_str()),
                    (default, source),
                    "{product:?} {cpuid:?}"
                );
                let (all, ..) = mode_minimum(product, cpuid, Transparent, later).unwrap();
                assert_eq!(all, default, "{product:?} {cpuid:?}");
                let (early, ..) = mode_minimum(product, cpuid, Transparent, 1_780_272_000).unwrap();
                assert_eq!(early.raised_to(default), default, "{product:?} {cpuid:?}");
            }
        }
    }

    /// SECURITY: Milan is refused by default (W79, arXiv 2605.12990), on a
    /// real Milan report whose chain and signature are AMD's.
    #[test]
    fn a_real_milan_report_is_refused_by_default_and_verifies_only_when_allowed() {
        let p = evidence_payload(VIRTEE_MILAN_REPORT, VIRTEE_MILAN_VCEK);
        assert_eq!(
            production(&p, &smt_allowed()).unwrap_err(),
            SnpRejected::MilanRefused
        );
        // Allowed, it is still held to Milan's default minimum (2023 firmware).
        let allowed = SnpPolicy {
            allow_milan: true,
            ..smt_allowed()
        };
        let e = production(&p, &allowed).unwrap_err();
        assert!(matches!(e, SnpRejected::TcbTooLow(_)), "{e}");
        // Allowed at its own level, without the alias check its firmware predates.
        let at_its_level = SnpPolicy {
            minimum: Some(SnpTcbMinimum {
                fmc: 0,
                boot_loader: 3,
                tee: 0,
                snp: 8,
                microcode: 0x73,
                require_alias_check: false,
            }),
            ..allowed
        };
        let v = production(&p, &at_its_level).expect("AMD's Milan chain, signature and CRL");
        assert_eq!(v.product, SnpProduct::Milan);
        assert_eq!(v.report.version, 2);
    }

    /// A real VLEK (AWS, trustee test data): it chains to AMD's pinned
    /// Milan ASVK and signs its report, and is refused only by name until
    /// the policy names its provider; then the report's VMPL 1 refuses it.
    /// At 2025-06-01: the VLEK expired on 2025-12-10.
    #[test]
    fn a_real_aws_vlek_chains_to_amds_asvk_and_its_vmpl1_report_is_refused() {
        const IN_ITS_WINDOW: u64 = 1_748_736_000;
        let p = evidence_payload(AWS_VLEK_REPORT, AWS_VLEK);
        let run = |policy: &SnpPolicy| {
            verify_snp_report(
                &p,
                AMD_MILAN_CRL,
                &SnpAnchors::production(),
                policy,
                IN_ITS_WINDOW,
            )
            .unwrap_err()
        };
        let milan = SnpPolicy {
            allow_milan: true,
            ..smt_allowed()
        };
        let e = run(&milan);
        assert!(
            matches!(&e, SnpRejected::SigningKey(m) if m.contains("cc-us-east-2.amazonaws.com")),
            "{e}"
        );
        let aws = SnpPolicy {
            vlek_csp_ids: vec!["CN=cc-us-east-2.amazonaws.com".into()],
            ..milan
        };
        let e = run(&aws);
        assert!(
            matches!(&e, SnpRejected::Vmpl(m) if m.contains("VMPL 1")),
            "{e}"
        );
        // One bit of the report, and the VLEK's signature no longer holds.
        let mut tampered = p.clone();
        tampered[0x90] ^= 1;
        assert_eq!(
            verify_snp_report(
                &tampered,
                AMD_MILAN_CRL,
                &SnpAnchors::production(),
                &aws,
                IN_ITS_WINDOW
            )
            .unwrap_err(),
            SnpRejected::ReportSignature
        );
    }

    /// A real Turin VCEK chains to the pinned Turin ASK, and its 8-byte
    /// hwID and FMC extension are read (no real Turin report in hand).
    #[test]
    fn a_real_turin_vcek_chains_to_the_pinned_turin_root() {
        let facts = verify_vcek(VIRTEE_TURIN_VCEK, &SnpAnchors::production(), NOW)
            .expect("AMD's Turin chain");
        assert_eq!(facts.product, SnpProduct::Turin);
        assert_eq!(facts.hwid.as_ref().map(Vec::len), Some(8));
        assert_eq!(facts.csp_id, None);
        assert_eq!(facts.tcb_raw, [0, 0, 0, 0, 0, 0, 0, 9]);
        // Under another product's root it does not chain.
        let mut genoa_only = SnpAnchors::production();
        genoa_only.roots[2].ask_der = AMD_ASK_GENOA_DER;
        genoa_only.roots[2].ask_spki_sha256_hex = AMD_ASK_GENOA_SPKI_SHA256_HEX.into();
        let e = verify_vcek(VIRTEE_TURIN_VCEK, &genoa_only, NOW).unwrap_err();
        assert!(matches!(e, SnpRejected::Chain(_)), "{e}");
    }

    // ── Measurements ────────────────────────────────────────────────────

    /// Required refusal: a measurement that is not the pinned one.
    #[test]
    fn the_verdicts_measurements_are_the_reports_fields() {
        let v = verify(&genoa_evidence(|s| s.host_data = [0x48; 32])).unwrap();
        let m = snp_measurements(&v);
        let names: Vec<_> = m.iter().map(|m| m.register.as_str()).collect();
        assert_eq!(
            names,
            [
                "vcek_sha256",
                "snp_measurement",
                "snp_host_data",
                "snp_id_key_digest",
                "snp_author_key_digest"
            ]
        );
        assert_eq!(m[1].value_hex, "4d".repeat(48));
        assert_eq!(m[2].value_hex, "48".repeat(32));
        let pin = parse_expected_snp_measurement(&format!("snp_measurement={}", "4D".repeat(48)))
            .unwrap();
        assert_eq!(unmet_snp(&m, &[pin]), None);
        let other = parse_expected_snp_measurement(&format!("snp_measurement={}", "4e".repeat(48)))
            .unwrap();
        assert!(unmet_snp(&m, &[other]).unwrap().contains("not the image"));
    }

    #[test]
    fn an_expectation_that_could_never_match_is_refused_when_typed() {
        for bad in [
            "mrtd=00",
            "snp_measurement",
            &format!("snp_measurement={}", "0".repeat(95)),
            &format!("snp_host_data={}", "0".repeat(96)),
            &format!("snp_measurement={}", "g".repeat(96)),
        ] {
            assert!(parse_expected_snp_measurement(bad).is_err(), "{bad}");
        }
        let e = parse_expected_snp_measurement("rtmr1=SECRET").unwrap_err();
        assert!(!e.contains("SECRET"));
    }
}
