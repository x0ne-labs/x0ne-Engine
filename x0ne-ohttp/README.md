# x0ne-ohttp — prototype of the non-targeting layer

**Prototype. No x0ne binary uses it, no package ships it, and it changes nothing
in what x0ne Engine guarantees today.** It exists so that the design of a shared
service can be measured before it is built: a client that encapsulates each
request with Oblivious HTTP (RFC 9458) to a key it pinned beforehand, a relay
that sees who asks but not what, a gateway that sees what is asked but not by
whom, and a node set from which the client draws its server itself.

## The property it is after

Someone who breaks into one server, physically or otherwise, reads the requests
that land on it. The property is that nobody can make a chosen user's requests
land there: not the operator in front of the servers, not the host, not the
relay. Apple calls it non-targetability, Meta "should not be able to target a
particular user for compromise". The TEE does not give it; the protocol around
the TEE does.

## Topology

```
client ──(message/ohttp-req)──> relay ──> gateway ──> target
   ^                              │          │  (x0ne-infer, inside the TD)
   │ pins one key configuration   │          └ one HPKE key per TD, generated there
   │ draws one node, uniformly    └ forwards the body and nothing else; no retry
   └ never retries, never redraws
```

Modules: `config` (pinned key configurations, `application/ohttp-keys`),
`message` (Binary HTTP, RFC 9292, padded to power-of-two buckets; header fields
with exact values, then the date to the minute), `client`, `relay`, `gateway`
(replay cache bounded by the date window, one authority, listed paths), `nodes`
(the published node set, its epoch, and the uniform draw).

Cryptography: the `ohttp` and `bhttp` crates (RFC 9458 and RFC 9292, by the
RFC's author), HPKE from the `hpke` crate. KEMs offered: DHKEM(X25519,
HKDF-SHA256) and X-Wing (ML-KEM-768 + X25519). One symmetric suite
(HKDF-SHA256, AES-128-GCM): a configuration that does not offer it is refused,
and the client uses it whatever else a configuration lists, so every client
writes the same suite in the clear.

## What the tests attack

Every guarantee has a test that tries to break it (`tests/`):

| Guarantee | Test |
|---|---|
| The relay reads neither question, answer, path nor authority | `the_relay_sees_neither_the_question_nor_the_answer` |
| The relay forwards no header of the client's | `identifying_headers_never_reach_the_gateway` |
| Inside the encapsulation, the client stack sends no field that names the client: `content-type` and `accept` are `application/json`, `authorization` is one well-formed anonymous token (RFC 9577, type 0x0002), each once and in one order, then the date to the minute; the gateway refuses anything else | `the_client_puts_no_identity_inside_the_encapsulation`, `an_authorization_value_that_is_not_a_token_never_leaves_the_client`, `free_form_accept_and_content_type_values_never_leave_the_client`, `the_gateway_refuses_any_field_a_hostile_client_could_name_itself_with` |
| A key served to one user only is refused (Trail of Bits, TOB-WAPI-3) | `a_key_served_to_one_user_only_is_refused`, `the_key_configuration_fetched_through_the_relay_must_match_the_pin` |
| One suite for every client | `a_configuration_without_aes_128_gcm_is_refused`, `the_client_encapsulates_with_aes_128_gcm_whatever_the_order_of_the_suites` |
| A node set made for one user is refused; so is too small a set, or one of another epoch than the current one or the next | `a_set_made_for_one_user_is_refused`, `a_set_too_small_to_hide_anyone_is_refused`, `a_set_of_a_retired_or_later_epoch_is_refused` |
| The draw is the client's own and has no modulo bias; a node drawn 5 % more often than its share fails the chi-squared test | `the_index_draw_rejects_the_biased_tail`, `the_draw_is_uniform_over_the_nodes`, `the_uniformity_test_sees_one_node_favoured_by_five_percent` |
| Tampering, swapping or replaying by the relay is refused | `a_byte_flipped_by_the_relay_in_the_request_is_refused_before_the_target`, `a_byte_flipped_by_the_relay_in_the_response_is_refused_by_the_client`, `the_answer_to_another_request_is_refused`, `a_request_replayed_by_the_relay_reaches_the_target_once`, `a_replay_is_refused_inside_the_window_and_after_it` |
| Holding the public key is not enough to fill the replay cache: it takes only requests that passed every check, and forgets each one when its date leaves the window | `requests_the_gateway_refuses_do_not_fill_the_replay_cache`, `a_full_replay_cache_empties_as_its_requests_leave_the_window`, `a_request_dated_outside_the_window_is_refused` |
| One attempt, no retry (RFC 9458 section 6.5) | `a_failing_relay_gets_one_attempt_and_an_error` |
| Sizes on the wire follow the power-of-two bucket of the question and of the answer (1 KiB at least), not their length within it | `sizes_on_the_wire_hide_the_length_within_a_bucket`, `sizes_on_the_wire_follow_only_the_bucket_of_the_question` |
| The gateway serves one authority and listed paths only | `the_gateway_serves_only_its_own_target` |
| Wire format is RFC 9458's: Appendix A reproduced byte for byte by an independent implementation, which then talks to this crate | `tests/rfc9458.rs` |

```
cargo test -p x0ne-ohttp --locked
```

## What it does not do

- No attestation: the gateway's key is not yet bound to a TDX quote, and the
  client does not verify the node it drew. That binding is the next brick.
- No anonymous tokens: `authorization: PrivateToken …` (RFC 9577) is checked for
  its form, not issued nor verified. Until it is verified, its 32-byte nonce is
  the client's to fill, like the body: a client that wants to name itself can.
  What the stack guarantees is that it adds nothing that does.
- A flood of valid requests fills the replay cache and stops the node until
  they leave the date window. A client that gets errors keeps its node; a person
  who opens a new session draws again, so whoever stops every node but one steers
  those people to it. Tokens are what will make each request cost something.
- The date tells the gateway the client's clock to the minute: a client several
  minutes off stands apart from the others (RFC 9458 section 6.8).
- The node set's epoch is checked against the one the caller says is current;
  where that comes from (the log's latest entry, a clock) is left to the caller.
- No streaming: known-length messages only; chunked OHTTP is a draft.
- No transparency log: the node set's digest is given to the client, not
  proven included anywhere.
- A relay that colludes with the gateway, or a party that watches both sides,
  correlates by timing. The relay sees the client's address and request volume.
- Nothing here has run outside the loopback.
