//! Prototype of the non-targeting layer of a shared x0ne service. No x0ne binary
//! uses it, and it changes nothing in what x0ne Engine guarantees today.
//!
//! The aim: someone who breaks into one server, physically or otherwise, reads the
//! requests that land on it, but cannot make a chosen user's requests land there.
//!
//! - [`client`] encapsulates each request with Oblivious HTTP (RFC 9458) to a key
//!   configuration it pinned beforehand ([`config`]), on a node it drew itself,
//!   uniformly, from a published node set ([`nodes`]); it sends once, through the
//!   relay, and never redraws after a failure.
//! - [`relay`] sees who is asking, not what; it forwards the body and nothing that
//!   identifies the client.
//! - [`gateway`] sees what is asked, not by whom; it opens the request, refuses a
//!   replay within the date window, and seals the answer.
//! - [`message`] carries requests and answers in Binary HTTP (RFC 9292), padded to
//!   size buckets: the length shows only through its bucket. Header fields have
//!   exact values, and the date goes to the minute.
//!
//! The cryptography is the `ohttp` and `bhttp` crates, by the RFCs' author;
//! `tests/rfc9458.rs` checks them against RFC 9458's test vector with an
//! independent implementation.

pub mod client;
pub mod config;
pub mod gateway;
pub mod message;
pub mod nodes;
pub mod relay;

/// RFC 9458 section 9.
pub const REQUEST_MEDIA_TYPE: &str = "message/ohttp-req";
pub const RESPONSE_MEDIA_TYPE: &str = "message/ohttp-res";
pub const KEYS_MEDIA_TYPE: &str = "application/ohttp-keys";
