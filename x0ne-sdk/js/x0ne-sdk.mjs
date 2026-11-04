// x0ne-sdk in a browser or in Node 22: loads the wasm32 build of the crate and
// wraps its C ABI (src/abi.rs). Prototype. Every check runs in the wasm
// module; this file only moves JSON in and out, draws randomness with
// crypto.getRandomValues, reads the clock, and fetches.
//
//   const sdk = await load(wasmBytes);
//   const session = await connect(sdk, baseUrl, policy, { headers }); // verified, or throws
//   // the log's last checkpoint is kept in localStorage (options.tlog: another store)
//   session.verified.session_config;  // the TD's model, expires_at_unix, relay key: read first
//   const answer = await session.chat({ messages: [...] }); // receipt checked
//   const streamed = await session.chatStream({ stream: true, messages: [...] },
//     (delta) => show(delta));  // deltas as they come, NOT VERIFIED; the
//                               // result once the final receipt checked
//   session.close();                                       // frees the key's slot
//
// A result holds `text` and `receipt`, and under `unverified` what the TD
// said that no receipt signs (`body`; a stream's `finish_reason`, `usage`).
// An answer with no checked receipt throws: `e.unverified` holds the TD's
// status and its body or error, unsigned.
//
// The page that serves this file and the wasm module can replace both: a
// browser checks nothing about the code it was handed (see LIMITATIONS).

const OP_VERIFY_KEY = 1;
const OP_SEAL_CHAT = 2;
const OP_OPEN = 3;
const OP_CLOSE = 4;
const OP_NRAS_REQUEST = 5;
const OP_SEAL_CHAT_STREAM = 6;
const OP_STREAM_PUSH = 7;
const OP_STREAM_FINISH = 8;

const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");

function random32() {
  return crypto.getRandomValues(new Uint8Array(32));
}

function toBase64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

function fromBase64(text) {
  const s = atob(text);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

const nowSeconds = () => Math.floor(Date.now() / 1000);

const TLOG_KEY = "x0ne-sdk/tlog-checkpoint";

/**
 * Where the loader keeps the log's last verified checkpoint between pages:
 * `localStorage` under one key when the page has it, else nothing kept past
 * this run. `options.tlog` replaces it with any `{ load(), save(note) }`.
 * The module itself keeps no memory: it returns `tlog_checkpoint` with every
 * verified key and takes it back on the next call. Whoever writes this
 * storage can empty it (the next checkpoint is then kept as the first) or
 * put an older genuine checkpoint back; a forged one is refused.
 */
function defaultTlogStore() {
  let memory = null;
  const storage = typeof localStorage === "undefined" ? null : localStorage;
  return {
    load: () => (storage ? storage.getItem(TLOG_KEY) : memory),
    save: (note) => {
      if (storage) storage.setItem(TLOG_KEY, note);
      else memory = note;
    },
  };
}

/** The signed checkpoint note at the end of a `c2sp.org/tlog-proof@v1`. */
function checkpointOf(tlogProof) {
  const at = tlogProof.indexOf("\n\n");
  return at < 0 ? null : tlogProof.slice(at + 2);
}

/** The size line of a checkpoint note, as a number; null when unreadable. */
function sizeOf(note) {
  const lines = String(note).split("\n");
  return lines.length > 2 && /^(0|[1-9][0-9]*)$/.test(lines[1]) ? Number(lines[1]) : null;
}

/**
 * Instantiates the module (no imports) and returns `call(op, object)`. A
 * panic in wasm32 is a trap that leaves the module's state half changed
 * (src/abi.rs): after one, every call throws; load the module again.
 */
export async function load(wasmBytes) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports;
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  let trapped = false;
  function call(op, input) {
    if (trapped) throw new Error("the module trapped earlier: load it again");
    const bytes = enc.encode(JSON.stringify(input));
    const ptr = ex.x0ne_alloc(bytes.length);
    // Null over the module's cap: writing there would overwrite its memory.
    if (ptr === 0) throw new Error(`input of ${bytes.length} bytes: over the module's cap`);
    new Uint8Array(ex.memory.buffer, ptr, bytes.length).set(bytes);
    let out;
    try {
      const n = ex.x0ne_call(op, ptr, bytes.length);
      out = new Uint8Array(ex.memory.buffer, ex.x0ne_result_ptr(), n).slice();
    } catch (e) {
      if (e instanceof WebAssembly.RuntimeError) trapped = true;
      throw e;
    } finally {
      if (!trapped) ex.x0ne_free(ptr, bytes.length);
    }
    const v = JSON.parse(dec.decode(out));
    if (!v.ok) {
      // A refused key carries the relay's exit code for the same refusal;
      // an answer with no checked receipt, what the TD said, unsigned.
      const e = new Error(v.error);
      if (v.code !== undefined) e.code = v.code;
      if (v.step !== undefined) e.step = v.step;
      if (v.unverified !== undefined) e.unverified = v.unverified;
      throw e;
    }
    return v;
  }
  return { call };
}

/**
 * Fetches the TD's key document (quote, event log, and for a GPU TD the GPU
 * evidence with NVIDIA's verdict on it) for a fresh nonce and the relayed
 * collateral (Intel's two answers, Google's launch endorsement and the
 * firmware binary it names), and has the wasm module verify them under
 * `policy` ({reference_json, reference_sig_b64, reference_tlog_proof,
 * machine_type, model_digest_hex, attest: "cpu+gpu" ("cpu" is refused),
 * tcb?: "transparent" | "strict", nras_ocsp?: "transparent" | "required",
 * mrtd_hex?, rtmr0_hex?, rtmr1_hex?, rtmr2_hex?, rtmr3_hex?,
 * max_collateral_age_secs?}). The reference must
 * pin RTMR0 (`rtmr0.policy` `gce-endorsed`): one that reports it is
 * refused (code 104, as the relay refuses it).
 * `options.headers` go with every request: the relay's local endpoint wants
 * its bearer (`{ authorization: "Bearer ..." }`).
 *
 * A key document with GPU evidence and no verdict of its own (a TD before
 * 2026-10-28) gets NVIDIA's verdict through the carrier: the module builds
 * the request (`POST ${baseUrl}/v2/e2e/nras`, forwarded to NRAS), the answer
 * is checked with the rest. A carrier that fails it leaves it empty: refused.
 *
 * `reference_json`, with its signature and log proof, chooses the image the
 * TD must run, and `model_digest_hex` the model: take them from the
 * integrator's own build or a source it trusts, never from `baseUrl`. The
 * module admits the reference only signed by x0ne's archive key and shown in
 * x0ne's log. The collateral comes from `baseUrl`: its carrier can relay
 * genuine collateral up to a day old, and Intel's root CA CRL until it
 * expires (see LIMITATIONS); `max_collateral_age_secs` shortens the day.
 *
 * The log's last verified checkpoint is kept by this loader
 * (`options.tlog`, `localStorage` by default) and handed back to the module,
 * which holds the reference's checkpoint to it: the same size must show the
 * same root, a smaller one is refused, a larger one needs a consistency
 * proof from the remembered size (a tlog-witness `add-checkpoint` body).
 * `options.consistency(oldSize, remembered, checkpoint)` returns that body
 * (or null) when the reference's checkpoint is larger than the remembered
 * one: the integrator fetches it from the log or its own backend; the
 * loader does not read the log. Without it, a grown log is refused with
 * code 100 at `log-consistency` until the memory is cleared or the body
 * supplied.
 */
export async function connect(sdk, baseUrl, policy, options = {}) {
  const headers = options.headers || {};
  const tlog = options.tlog || defaultTlogStore();
  const get = async (path) => {
    const resp = await fetch(`${baseUrl}${path}`, { headers });
    if (!resp.ok) throw new Error(`${path.split("?")[0]} ${resp.status}: ${await resp.text()}`);
    return resp.json();
  };
  const nonce = random32();
  const document = await get(`/v2/e2e/key?nonce=${hex(nonce)}`);
  const collateral = await get("/v2/e2e/collateral");
  if (document && document.gpu_evidence_list && document.nras_answer === undefined) {
    const { request } = sdk.call(OP_NRAS_REQUEST, { document, client_nonce_hex: hex(nonce) });
    const resp = await fetch(`${baseUrl}/v2/e2e/nras`, {
      method: "POST",
      headers: { ...headers, "content-type": "application/json" },
      body: JSON.stringify(request),
    });
    collateral.nras_answer = resp.ok ? await resp.text() : "";
  }
  const remembered = tlog.load() || undefined;
  let consistency;
  if (remembered && options.consistency) {
    const checkpoint = checkpointOf(policy.reference_tlog_proof);
    const oldSize = sizeOf(remembered);
    const newSize = checkpoint === null ? null : sizeOf(checkpoint);
    if (oldSize !== null && newSize !== null && newSize > oldSize) {
      consistency = (await options.consistency(oldSize, remembered, checkpoint)) || undefined;
    }
  }
  const v = sdk.call(OP_VERIFY_KEY, {
    document,
    collateral,
    policy,
    client_nonce_hex: hex(nonce),
    now: nowSeconds(),
    tlog_checkpoint: remembered,
    tlog_consistency: consistency,
  });
  // The checkpoint to keep: the module verified it, this loader only stores it.
  if (typeof v.verified.tlog_checkpoint === "string") tlog.save(v.verified.tlog_checkpoint);
  return {
    verified: v.verified,
    /**
     * One chat completion, sealed to the verified key; its receipt checked
     * (`text`, `receipt`; the body, unsigned, under `unverified`). The TD's
     * refusal throws, its status and body in `e.unverified`.
     */
    async chat(body) {
      const s = sdk.call(OP_SEAL_CHAT, {
        key: v.key,
        body,
        ikm_hex: hex(random32()),
        now: nowSeconds(),
      });
      // OP_OPEN takes the pending handle, answer checked or not; until then
      // a failure here leaves it held, so it is closed.
      let taken = false;
      try {
        const request = fromBase64(s.request_b64);
        const resp = await fetch(`${baseUrl}/v2/e2e/chat`, {
          method: "POST",
          headers: { ...headers, "content-type": "message/ohttp-req" },
          body: request,
        });
        if (!resp.ok) throw new Error(`gateway ${resp.status}: ${await resp.text()}`);
        const sealed = new Uint8Array(await resp.arrayBuffer());
        taken = true;
        const opened = sdk.call(OP_OPEN, {
          pending: s.pending,
          response_b64: toBase64(sealed),
          now: nowSeconds(),
        });
        opened.request_len = request.length;
        opened.response_len = sealed.length;
        return opened;
      } finally {
        if (!taken) sdk.call(OP_CLOSE, { handle: s.pending });
      }
    },
    /**
     * One streamed chat completion (`body.stream` must be true), sealed to
     * the verified key as a chunked request. `onDelta(text)` gets each delta
     * as its frame comes off the wire, NOT VERIFIED: the stream may still be
     * refused. The result comes once the receipt of the final chunk checked
     * over the whole text (`text`, `receipt`, `final_deltas`, `chunks`;
     * `finish_reason` and `usage`, unsigned, under `unverified`); a frame the
     * relay altered, dropped or reordered throws here, as a stream the TD
     * refused or ended in an error event (`e.unverified`: its status and
     * error), and the deltas shown must be discarded.
     */
    async chatStream(body, onDelta) {
      const s = sdk.call(OP_SEAL_CHAT_STREAM, {
        key: v.key,
        body,
        ikm_hex: hex(random32()),
        now: nowSeconds(),
      });
      // OP_STREAM_FINISH takes the stream handle; until then a failure
      // leaves it held, so it is closed.
      let taken = false;
      try {
        const request = fromBase64(s.request_b64);
        const resp = await fetch(`${baseUrl}/v2/e2e/chat`, {
          method: "POST",
          headers: { ...headers, "content-type": "message/ohttp-chunked-req" },
          body: request,
        });
        if (!resp.ok) throw new Error(`gateway ${resp.status}: ${await resp.text()}`);
        const reader = resp.body.getReader();
        let frames = 0;
        let bytes = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          frames += 1;
          bytes += value.length;
          const pushed = sdk.call(OP_STREAM_PUSH, { stream: s.stream, bytes_b64: toBase64(value) });
          if (onDelta) for (const delta of pushed.unverified_deltas) onDelta(delta);
        }
        taken = true;
        const finished = sdk.call(OP_STREAM_FINISH, { stream: s.stream, now: nowSeconds() });
        finished.request_len = request.length;
        finished.response_len = bytes;
        finished.frames = frames;
        return finished;
      } finally {
        if (!taken) sdk.call(OP_CLOSE, { handle: s.stream });
      }
    },
    /** Drops the verified key: no question is sealed to it afterwards. */
    close() {
      sdk.call(OP_CLOSE, { handle: v.key });
    },
  };
}
