# Tests

## Tests

```sh
cargo test -p x0ne-infer            # unit + integration, includes REAL fixtures
cargo test -p x0ne-infer --features nvml
```

The TCB tests run against a genuine TDX v4 quote and Intel PCS production
collateral, both vendored on 2026-08-20 under `x0ne-infer/tests/pcs_live/`,
and evaluated at a fixed date inside that collateral's validity window. The
flagship test asserts that a quote with a **perfect chain to the real Intel
root** is refused because its platform's TCB is below every level the
vendored collateral lists for it (all but the newest are out of date).
`tests/tcb_policy_live.rs` runs the `strict` TCB policy on the quote of the GCE run of
2026-10-18 and Intel PCS collateral vendored on 2026-10-22
(`pcs_live/2026-10-22/`): accepted under evaluation 22, refused under
Intel's default `standard` evaluation 20, refused once the collateral or the
PCK CRL is more than a day old or expired. `tests/root_crl_live.rs` verifies
Intel's root CA CRL of 2026-10-23 (`pcs_live/2026-10-23/`) under the pinned
root for the same quote and chain, refuses it outside its window or with one
bit flipped, and reads the production URL out of the root certificate's CRL
distribution point. `tests/real_model.rs` needs the model files and is ignored by
default. `tests/process_hardening.rs` probes the server from another process
of the same user: run as root, it prints `SKIPPED` and passes, because root
passes every check; `X0NE_REQUIRE_NON_ROOT=1` turns that into a failure, and
`make ci` sets it.

`x0ne-sdk` is a prototype of the caller's side of the TD's `/v2/e2e/*` routes
(the caller checks the TD's key and seals to it; `docs/sdk-e2e.md`).
`cargo test -p x0ne-sdk`, `cargo test -p x0ne-infer --lib
-- e2e_sdk_tests` (against the simulated TD), `cargo test -p x0ne-infer --lib
-- e2e_td_tests` (the TD's routes on synthetic TDX evidence, through a
recording carrier) and `make sdk-wasm` (the wasm32 build in Node) cover it.
`make e2e-local` adds the wasm32 build through the carrier and, with
`X0NE_E2E_QWEN3_DIR`, Qwen3-0.6B on CPU; no network.
`x0ne-sdk/tests/real_quote_2026_10_18.rs` replays the whole quote of the GCE
run of 2026-10-18 through the SDK, offline, on Intel's collateral of
2026-10-22 at a fixed clock: `verify_key` refuses it at `binding` (its
`report_data` binds the relay's channel), `verify_quote` accepts it to the
end on its own event log, Google's endorsement of its MRTD and the run's
reference, without pins, and an altered signature bit, pin, `report_data`,
collateral, PCK CRL, root CA CRL, endorsement, log event or PCK chain is
refused at its step.
