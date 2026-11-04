// Node smoke test of the wasm32 build against the simulated TD or a carrier
// in front of the TD's routes (x0ne-infer src/e2e_sdk_tests.rs and
// src/e2e_td_tests.rs, wasm_* tests; scripts/sdk-wasm.sh, scripts/e2e-local.sh).
// Usage: node smoke.mjs <x0ne_sdk.wasm> <base url> <policy json> <question> [bearer file]
// The bearer file (the relay's local key file) is read, never passed in argv.
// Prints one JSON line: the verified key's summary and the opened answer,
// or {"error": ..., "code"?: ..., "unverified"?: ...} with exit code 1. With X0NE_SMOKE_STREAM=1
// the question goes as a sealed stream (`chatStream`): `streamed` counts the
// frames and the unverified deltas, and says whether they matched the
// verified text. With X0NE_SMOKE_TLOG_FILE=<file>, the log's last verified
// checkpoint is kept in that file between runs, as a page keeps it in
// localStorage (x0ne-infer src/e2e_sdk_tests.rs, wasm_in_node_keeps_…).
import { readFile } from "node:fs/promises";
import { readFileSync, writeFileSync } from "node:fs";
import { load, connect } from "./x0ne-sdk.mjs";

/** A `{ load, save }` store over one file; nothing kept when it is unset. */
function fileStore(path) {
  if (!path) return undefined;
  return {
    load: () => {
      try {
        return readFileSync(path, "utf8");
      } catch (e) {
        if (e && e.code === "ENOENT") return null;
        throw e;
      }
    },
    save: (note) => writeFileSync(path, note),
  };
}

const [wasmPath, baseUrl, policyJson, question, bearerFile] = process.argv.slice(2);
try {
  const bytes = await readFile(wasmPath);
  const imports = WebAssembly.Module.imports(new WebAssembly.Module(bytes));
  const sdk = await load(bytes);
  const headers = bearerFile
    ? { authorization: `Bearer ${(await readFile(bearerFile, "utf8")).trim()}` }
    : {};
  const tlog = fileStore(process.env.X0NE_SMOKE_TLOG_FILE);
  const session = await connect(sdk, baseUrl, JSON.parse(policyJson), { headers, tlog });
  // No `model`: the request is answered under the one model the TD serves.
  const messages = [{ role: "user", content: question }];
  let answer;
  if (process.env.X0NE_SMOKE_STREAM === "1") {
    const deltas = [];
    answer = await session.chatStream({ stream: true, messages }, (d) => deltas.push(d));
    answer.streamed = {
      frames: answer.frames,
      deltas: deltas.length,
      unverified_matches: deltas.join("") + answer.final_deltas.join("") === answer.text,
    };
  } else {
    answer = await session.chat({ messages });
  }
  session.close();
  let closed = false;
  try {
    await session.chat({ messages: [{ role: "user", content: question }] });
  } catch (e) {
    closed = String(e.message).includes("no verified key");
  }
  console.log(
    JSON.stringify({
      wasm_bytes: bytes.length,
      wasm_imports: imports.length,
      verified: session.verified,
      closed,
      ...answer,
    }),
  );
} catch (e) {
  const out = { error: String(e && e.message ? e.message : e) };
  if (e && e.code !== undefined) out.code = e.code;
  if (e && e.step !== undefined) out.step = e.step;
  if (e && e.unverified !== undefined) out.unverified = e.unverified;
  console.log(JSON.stringify(out));
  process.exit(1);
}
