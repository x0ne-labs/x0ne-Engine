//! x0ne Core runtime: Qwen3 dense bf16, one request, one GPU.
//! E0 holds the parts that run without a GPU (arena, loader, ceiling, references)
//! and the E1 surfaces (`ops`, `engine`) that its red tests call.
//!
//! Every `unsafe` block and impl carries a `// SAFETY:` comment, and an `unsafe fn` states
//! each unsafe operation in its own block.
#![deny(clippy::undocumented_unsafe_blocks)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod arena;
pub mod config;
pub mod cuda;
pub mod engine;
pub mod error;
pub mod gpu;
pub mod metrics;
pub mod ops;
pub mod refs;
pub mod roofline;
pub mod st;
pub mod weights;

pub use error::{Error, Result};

/// Kernel set hash and compiler of the embedded cubins, for receipts.
pub fn kernel_set() -> (&'static str, &'static str) {
    (
        x0ne_core_kernels::KERNEL_SET_SHA256,
        x0ne_core_kernels::NVCC_RELEASE,
    )
}
