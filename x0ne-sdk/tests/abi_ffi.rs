//! The C ABI's exports as a native host calls them: through `x0ne_call` and
//! `x0ne_result_ptr`, not `abi::call`. This file is a process of its own:
//! the panic below drops every handle of the instance, which the tests of
//! `abi.rs` share.

#![allow(unsafe_code)]

use serde_json::Value;
use std::sync::{Arc, Barrier};
use x0ne_sdk::abi::{self, OP_TEST_PANIC};

/// `x0ne_call` on `input`, then the result read where `x0ne_result_ptr`
/// says, as a native host reads it: `call` and `read` apart, so a test can
/// put another thread's call between them.
fn call(op: u32, input: &[u8]) -> usize {
    let ptr = abi::x0ne_alloc(input.len());
    assert!(!ptr.is_null());
    // SAFETY: `ptr` holds `input.len()` bytes, just allocated.
    unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), ptr, input.len()) };
    // SAFETY: an x0ne_alloc buffer of that length.
    let n = unsafe { abi::x0ne_call(op, ptr, input.len()) };
    // SAFETY: the pointer and length of the allocation above.
    unsafe { abi::x0ne_free(ptr, input.len()) };
    n
}

fn read(n: usize) -> Value {
    let at = abi::x0ne_result_ptr();
    assert!(!at.is_null());
    // SAFETY: the ABI's contract: `n` bytes at x0ne_result_ptr until this
    // thread's next x0ne_call.
    let bytes = unsafe { std::slice::from_raw_parts(at, n) };
    serde_json::from_slice(bytes).unwrap()
}

/// SECURITY: two threads of one host, each calling in turn, each read their
/// own result: never the other's answer, never a buffer the other's call
/// freed. A failure: one thread reading another's plaintext, or freed memory.
#[test]
fn two_threads_each_read_their_own_result() {
    const ROUNDS: usize = 50;
    let turn = Arc::new(Barrier::new(2));
    let threads: Vec<_> = [0x4000u32, 0x5000]
        .into_iter()
        .enumerate()
        .map(|(i, base)| {
            let turn = Arc::clone(&turn);
            std::thread::spawn(move || {
                for round in 0..ROUNDS {
                    let op = base + round as u32;
                    // Thread 0 calls, then thread 1; both read after both called.
                    if i == 1 {
                        turn.wait();
                    }
                    let n = call(op, b"{}");
                    if i == 0 {
                        turn.wait();
                    }
                    turn.wait();
                    let out = read(n);
                    assert_eq!(
                        out["error"],
                        format!("unknown op {op}"),
                        "thread {i}, round {round}: {out}"
                    );
                    turn.wait();
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

/// SECURITY: a panic inside the instance is an error the host reads, with
/// the code a panicking Rust process exits with (101) and the step
/// `panic`; the instance serves the next call. Without it, the panic
/// crosses `extern "C"` and aborts the host. (That the handles are dropped
/// is tested on a verified key, in a process of its own:
/// `e2e_sdk_tests::a_panic_in_the_c_abi_drops_every_handle`.)
#[test]
fn a_panic_is_an_error_the_host_reads_not_an_abort() {
    let out = read(call(OP_TEST_PANIC, b"{}"));
    assert_eq!(out["ok"], false, "{out}");
    assert_eq!(out["code"], 101, "{out}");
    assert_eq!(out["step"], "panic", "{out}");
    assert!(
        out["error"].as_str().unwrap().contains("every handle"),
        "{out}"
    );
    // The lock the panic poisoned does not refuse the next caller.
    let next = read(call(99, b"{}"));
    assert_eq!(next["error"], "unknown op 99", "{next}");
    let closed = read(call(abi::OP_CLOSE, br#"{"handle": 1}"#));
    assert_eq!(closed["error"], "no handle 1", "{closed}");
}
