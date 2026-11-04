//! The attestation vocabulary of x0ne Engine, as a crate of its own:
//! `x0ne-infer` (the server and the relay) and `x0ne-sdk` (the caller's side)
//! share these types, so a verdict, a key release or a receipt means the
//! same on both sides. A small, dependency-light crate.
//!
//! Contents:
//! - [`lease`] — bounded authority across time (window, budget, revocation,
//!   provenance; ceilings as policy).
//! - [`model`] — the confidential-runtime model: environments, measurements,
//!   workload identity, attestation evidence, verdicts, protected artifacts,
//!   attestation-bound lease-bounded key release, receipts.
//! - [`policy`] — verification policy and the structural (non-cryptographic,
//!   named, fail-closed) verifier.
//! - [`tdx_quote`] — parsing an Intel TDX quote into measurements and
//!   checking the nonce it echoes. Structural only: the Intel PCS signature
//!   chain and the TCB level are checked by `x0ne-infer`
//!   (`pcs::verify_tdx_chain`, `tcb`), whose files `x0ne-sdk` compiles too.
//! - [`snp_report`] — parsing an AMD SEV-SNP attestation report (versions
//!   2, 3 and 5). Structural only: the VCEK chain, the signature and the
//!   guest policy are judged by `x0ne-infer::snp`.

pub mod lease;
pub mod model;
pub mod policy;
pub mod snp_report;
pub mod tdx_quote;
