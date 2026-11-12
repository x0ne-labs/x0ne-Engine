//! The engine on small models built in memory, against the CPU reference forward and against
//! itself: determinism, the routes that must agree, a session against a model of it, and what
//! is left of a request once it is over.
//!
//! These tests need the GPU and are ignored without it: `scripts/core-gpu.sh` runs them
//! (`--include-ignored`, one thread).
//!
//! Two kinds of comparison. Bit for bit, where x0ne Core's own suite establishes it: the same
//! request twice, a request after others, a session's logits and the greedy pass's, the device's
//! greedy tokens and the host's. Within a tolerance, where two kernels or two routes compute
//! the same quantity (prefill and decode, graph and eager, per-head and split-K) and against
//! the reference: `LOGITS_ATOL_OF_MAX` and `LOGITS_KL` below. Those two were chosen from the
//! kernels' tolerances and have not been calibrated on a device: the first run on a GPU prints
//! the worst ratio of each comparison (`MEASURE`), and they are to be set from it.

mod common;

use common::{file_of, model_tensors, reference, tiny, worst_ratio, Rng};
use std::panic::{catch_unwind, AssertUnwindSafe};
use x0ne_core_oracle::model::{Cache, Config, Model};
use x0ne_core_oracle::{self as oracle, bf16};
use x0ne_core_runtime::engine::{
    DecodeAttention, DecodeMode, Engine, EngineOptions, Route, SplitMerge, GREEDY_AHEAD_MAX,
    PREFILL_CHUNK,
};
use x0ne_core_runtime::st::SliceSource;
use x0ne_core_runtime::{metrics, Error};

/// A logit within this share of the row's largest magnitude. Not calibrated on a device.
const LOGITS_ATOL_OF_MAX: f64 = 0.05;
/// KL(reference || engine) of a logits row, in nats, at most. Not calibrated on a device.
const LOGITS_KL: f64 = 0.02;

struct Fixture {
    cfg: Config,
    file: Vec<u8>,
    model: Model,
}

impl Fixture {
    fn new(cfg: Config, seed: u64) -> Fixture {
        Self::with(cfg, seed, |_, _| {})
    }

    /// As `new`, with the values of each tensor open to a change before the file is made.
    fn with(cfg: Config, seed: u64, mut change: impl FnMut(&str, &mut Vec<f32>)) -> Fixture {
        let mut tensors = model_tensors(&cfg, &mut Rng::new(seed));
        for t in &mut tensors {
            let mut values = bf16::from_bytes(&t.data);
            change(&t.name, &mut values);
            t.data = bf16::bytes(&values);
        }
        Fixture {
            model: reference(&cfg, &tensors),
            file: file_of(&tensors),
            cfg,
        }
    }

    fn engine(&self, max_ctx: usize) -> Engine {
        let src = SliceSource::new(None, &[("model.safetensors", &self.file)]).expect("checkpoint");
        Engine::load_from(&self.cfg.to_json(), &src, EngineOptions { max_ctx })
            .expect("a GPU the kernels are built for")
    }

    fn tokens(&self, seed: u64, n: usize) -> Vec<u32> {
        Rng::new(seed).tokens(n, self.cfg.vocab_size)
    }
}

fn wide() -> Config {
    Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 3,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        ..tiny(3, 16, 8, true)
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn top_two(row: &[f32]) -> (f32, f32) {
    row.iter().fold((f32::MIN, f32::MIN), |(a, b), &x| {
        if x > a {
            (x, a)
        } else {
            (a, b.max(x))
        }
    })
}

/// `got` against `want` within the model-level tolerance; the argmax too, when the reference's
/// first two logits are further apart than the tolerance can move them.
fn assert_logits(what: &str, got: &[f32], want: &[f32]) {
    let scale = want.iter().fold(0f64, |m, &x| m.max(x.abs() as f64));
    let atol = LOGITS_ATOL_OF_MAX * scale;
    let (ratio, at) = worst_ratio(got, want, atol, 0.0);
    let kl = oracle::kl(want, got);
    eprintln!("MEASURE {what}: worst logit ratio {ratio:.4} of {atol:.4}, KL {kl:.6}");
    assert!(
        ratio <= 1.0,
        "{what}: logit {at}: got {} want {} (ratio {ratio:.3})",
        got[at],
        want[at]
    );
    assert!(kl <= LOGITS_KL, "{what}: KL {kl}");
    let (first, second) = top_two(want);
    if (first - second) as f64 > 2.0 * atol {
        assert_eq!(metrics::argmax(got), oracle::argmax(want), "{what}: argmax");
    }
}

fn scrubbed(e: &Engine) -> bool {
    e.client_data_is_zero().expect("read back") && e.host_buffers_are_zero()
}

/// Both routes and both decode modes against the reference at every position, on three
/// geometries, tied and untied.
#[test]
#[ignore = "needs the GPU"]
fn every_route_gives_the_reference_logits_at_every_position() {
    for (name, f) in [
        ("1 layer 2/1", Fixture::new(tiny(1, 2, 1, false), 1)),
        ("2 layers 4/2", Fixture::new(tiny(2, 4, 2, false), 2)),
        ("3 layers 16/8 tied", Fixture::new(wide(), 3)),
    ] {
        let mut e = f.engine(96);
        eprintln!("MEASURE {name}: {}", e.describe());
        let ids = f.tokens(10, 40);
        let want = f.model.logits(&ids);
        let all: Vec<usize> = (0..ids.len()).collect();
        let prefill = e.logits_at(&ids, &all, Route::Prefill).expect("prefill");
        let graph = e
            .logits_at(&ids, &all, Route::Decode)
            .expect("decode, graph");
        e.set_decode_mode(DecodeMode::Eager).expect("eager");
        let eager = e
            .logits_at(&ids, &all, Route::Decode)
            .expect("decode, eager");
        e.set_decode_mode(DecodeMode::Graph).expect("graph");
        for p in 0..ids.len() {
            assert_logits(&format!("{name} prefill {p}"), &prefill[p], &want[p]);
            assert_logits(&format!("{name} decode {p}"), &graph[p], &want[p]);
            assert_logits(&format!("{name} eager {p}"), &eager[p], &want[p]);
            assert_logits(
                &format!("{name} prefill against decode {p}"),
                &prefill[p],
                &graph[p],
            );
        }
        eprintln!(
            "MEASURE {name}: graph and eager give the same bits: {}",
            graph.iter().zip(&eager).all(|(a, b)| bits(a) == bits(b))
        );
        // The statistics of a pass are those of its logits rows.
        let stats = e.stats(&ids, Route::Prefill).expect("stats");
        for (p, s) in stats.iter().enumerate() {
            assert_eq!(
                Some(s.argmax),
                metrics::argmax(&prefill[p]),
                "{name}: argmax at {p}"
            );
            let lse = metrics::logsumexp(&prefill[p]);
            assert!(
                (s.logsumexp as f64 - lse).abs() <= 1e-4 * lse.abs().max(1.0),
                "{name}: log-sum-exp at {p}"
            );
            match ids.get(p + 1) {
                Some(&next) => {
                    let lp = prefill[p][next as usize] as f64 - lse;
                    assert!(
                        (s.next_logprob as f64 - lp).abs() <= 1e-3,
                        "{name}: log-probability at {p}"
                    );
                }
                None => assert!(s.next_logprob.is_nan()),
            }
        }
        assert!(scrubbed(&e), "{name}: client bytes after the passes");
    }
}

/// The same request twice gives the same bits, on every route, and so does a request made
/// after other requests.
#[test]
#[ignore = "needs the GPU"]
fn the_same_request_twice_gives_the_same_bits() {
    let f = Fixture::new(tiny(2, 4, 2, false), 4);
    let mut e = f.engine(700);
    let ids = f.tokens(11, 600);
    let keep = [0, 1, 255, 511, 512, 513, 599];
    for route in [Route::Prefill, Route::Decode] {
        let first = e.logits_at(&ids, &keep, route).expect("first");
        let _ = e.greedy(&f.tokens(12, 30), 20).expect("another request");
        let _ = e
            .logits_at(&f.tokens(13, 77), &[76], Route::Prefill)
            .expect("another request");
        let second = e.logits_at(&ids, &keep, route).expect("second");
        for (i, (a, b)) in first.iter().zip(&second).enumerate() {
            assert_eq!(
                bits(a),
                bits(b),
                "{route:?}: position {} differs between two runs",
                keep[i]
            );
        }
    }
    let (a, la) = e.greedy_with_last_logits(&ids[..100], 50).expect("greedy");
    let (b, lb) = e.greedy_with_last_logits(&ids[..100], 50).expect("greedy");
    assert_eq!(a, b);
    assert_eq!(bits(&la), bits(&lb));
    // A second engine on the same checkpoint: the same bits as the first.
    let mut other = f.engine(700);
    let (c, lc) = other
        .greedy_with_last_logits(&ids[..100], 50)
        .expect("greedy");
    assert_eq!(a, c, "two engines, two answers");
    assert_eq!(bits(&la), bits(&lc));
}

/// A prompt that fills one prefill chunk and one that crosses one, two or three chunk
/// boundaries: the logits at the positions around each boundary are the reference's, and the
/// decode route's. A prefill is never split otherwise than at `PREFILL_CHUNK`, so "one chunk
/// against k chunks" is: the prefix of one chunk against the same positions of the long prompt.
#[test]
#[ignore = "needs the GPU"]
fn a_prefill_across_chunk_boundaries_is_a_prefill_within_one() {
    let f = Fixture::new(tiny(2, 4, 2, false), 5);
    let mut e = f.engine(2048);
    let c = PREFILL_CHUNK;
    let ids = f.tokens(14, 3 * c + 40);
    let want = f.model.logits(&ids);
    for len in [
        1,
        2,
        c - 1,
        c,
        c + 1,
        2 * c - 1,
        2 * c,
        2 * c + 1,
        3 * c,
        3 * c + 40,
    ] {
        let keep: Vec<usize> = [
            0,
            1,
            c - 2,
            c - 1,
            c,
            c + 1,
            2 * c - 1,
            2 * c,
            2 * c + 1,
            3 * c - 1,
            3 * c,
            len - 1,
        ]
        .into_iter()
        .filter(|&p| p < len)
        .collect();
        let got = e
            .logits_at(&ids[..len], &keep, Route::Prefill)
            .expect("prefill");
        for (row, &p) in got.iter().zip(&keep) {
            assert_logits(&format!("prefill of {len}, position {p}"), row, &want[p]);
        }
        // The positions of the first chunk, computed alone.
        let first: Vec<usize> = keep.iter().copied().filter(|&p| p < c.min(len)).collect();
        let alone = e
            .logits_at(&ids[..c.min(len)], &first, Route::Prefill)
            .expect("one chunk");
        for ((row, one), &p) in got.iter().zip(&alone).zip(&first) {
            assert_logits(
                &format!("prefill of {len} against one chunk, position {p}"),
                row,
                one,
            );
        }
        eprintln!(
            "MEASURE prefill of {len}: the first chunk's rows have the bits of a one-chunk prefill: {}",
            got.iter().zip(&alone).all(|(a, b)| bits(a) == bits(b))
        );
    }
    // The decode route over the same boundaries.
    let len = 2 * c + 3;
    let keep = [c - 1, c, c + 1, 2 * c - 1, 2 * c, 2 * c + 2];
    let prefill = e
        .logits_at(&ids[..len], &keep, Route::Prefill)
        .expect("prefill");
    let decode = e
        .logits_at(&ids[..len], &keep, Route::Decode)
        .expect("decode");
    for ((a, b), &p) in prefill.iter().zip(&decode).zip(&keep) {
        assert_logits(&format!("decode at {p}"), b, &want[p]);
        assert_logits(&format!("prefill against decode at {p}"), a, b);
    }
}

/// A session: its logits after the prompt and after each step are the reference's with the
/// same cache, a fresh session's on the whole sequence (a recompute), and the bits of the
/// greedy pass's last row.
#[test]
#[ignore = "needs the GPU"]
fn a_session_with_its_cache_is_a_recompute() {
    for f in [
        Fixture::new(tiny(2, 4, 2, false), 6),
        Fixture::new(wide(), 7),
    ] {
        let mut e = f.engine(128);
        let ids = f.tokens(15, 60);
        for split in [1, 2, 31, 59] {
            let mut cache = Cache::default();
            let mut want = f.model.extend(&mut cache, &ids[..split]);
            let mut rows = Vec::new();
            {
                let mut s = e.session(&ids[..split]).expect("session");
                assert_eq!(s.position(), split);
                assert_logits(
                    &format!("session after a prompt of {split}"),
                    s.logits(),
                    &want,
                );
                for (i, &t) in ids[split..].iter().enumerate() {
                    s.step(t).expect("step");
                    want = f.model.step(&mut cache, t);
                    assert_eq!(s.position(), split + i + 1);
                    assert_logits(&format!("session of {split}, step {i}"), s.logits(), &want);
                    rows.push(s.logits().to_vec());
                }
                s.finish().expect("finish");
            }
            assert!(scrubbed(&e));
            // A recompute: a session opened on the whole prefix, for a few of the steps.
            for i in [0, rows.len() / 2, rows.len() - 1] {
                let s = e.session(&ids[..split + i + 1]).expect("session");
                assert_logits(
                    &format!("cache of {split} + {} against a recompute", i + 1),
                    &rows[i],
                    s.logits(),
                );
                s.finish().expect("finish");
            }
        }
        // The session's first row has the bits of the greedy pass's (x0ne Core's X1b).
        let (tokens, last) = e.greedy_with_last_logits(&ids[..20], 1).expect("greedy");
        let s = e.session(&ids[..20]).expect("session");
        assert_eq!(bits(s.logits()), bits(&last));
        assert_eq!(metrics::argmax(s.logits()), Some(tokens[0]));
        s.finish().expect("finish");
    }
}

/// Greedy decoding four ways: the engine's pass, a session stepped by the host on the argmax
/// of each row, the device's greedy loop at each number of steps ahead, and the reference.
/// The first three give the same tokens; the reference agrees up to the first position where
/// its two best logits are within the tolerance.
#[test]
#[ignore = "needs the GPU"]
fn greedy_on_the_device_is_greedy_on_the_host() {
    for (f, max_ctx) in [
        (Fixture::new(tiny(2, 4, 2, false), 8), 256),
        (Fixture::new(wide(), 9), 256),
    ] {
        let mut e = f.engine(max_ctx);
        for (prompt_len, n) in [(1, 1), (1, 40), (17, 2), (30, 100), (200, 57)] {
            let prompt = f.tokens(16 + prompt_len as u64, prompt_len);
            let engine = e.greedy(&prompt, n).expect("greedy");
            assert_eq!(engine.len(), n);

            let mut host = Vec::new();
            let mut s = e.session(&prompt).expect("session");
            for i in 0..n {
                let t = metrics::argmax(s.logits()).expect("finite logits");
                host.push(t);
                if i + 1 < n {
                    s.step(t).expect("step");
                }
            }
            s.finish().expect("finish");
            assert_eq!(
                host, engine,
                "prompt {prompt_len}, {n} tokens: host greedy against the engine's"
            );

            for ahead in [1, 2, 3, GREEDY_AHEAD_MAX] {
                let mut g = e
                    .session(&prompt)
                    .expect("session")
                    .into_greedy_ahead(n, ahead)
                    .expect("greedy session");
                let mut device = Vec::new();
                for i in 0..n {
                    device.push(g.next_token().expect("next token"));
                    assert_eq!(g.taken(), i + 1);
                    assert!(
                        g.position() < prompt_len + n,
                        "a step past the tokens asked"
                    );
                    assert!(
                        g.position() <= prompt_len + i + ahead,
                        "more than {ahead} steps ahead"
                    );
                }
                assert!(
                    matches!(g.next_token(), Err(Error::Format(_))),
                    "a token past those asked"
                );
                g.finish().expect("finish");
                assert_eq!(
                    device, engine,
                    "prompt {prompt_len}, {n} tokens, {ahead} ahead: device greedy"
                );
                assert!(scrubbed(&e));
            }

            // The reference, until its choice is within the tolerance of another.
            let mut cache = Cache::default();
            let mut logits = f.model.extend(&mut cache, &prompt);
            for (i, &t) in engine.iter().enumerate() {
                let scale = logits.iter().fold(0f32, |m, x| m.max(x.abs()));
                let (first, second) = top_two(&logits);
                if ((first - second) as f64) <= 2.0 * LOGITS_ATOL_OF_MAX * scale as f64 {
                    eprintln!("MEASURE greedy prompt {prompt_len}: compared {i} of {n} tokens with the reference");
                    break;
                }
                assert_eq!(
                    Some(t),
                    oracle::argmax(&logits),
                    "prompt {prompt_len}: token {i} against the reference"
                );
                if i + 1 < n {
                    logits = f.model.step(&mut cache, t);
                }
            }
        }
    }
}

/// A session opened after every other kind of request, finished or abandoned, has the bits of
/// the first session of a newly loaded engine.
#[test]
#[ignore = "needs the GPU"]
fn a_session_after_other_requests_is_a_session_on_a_fresh_engine() {
    let f = Fixture::new(tiny(2, 4, 2, false), 10);
    let prompt = f.tokens(20, 90);
    let steps = f.tokens(21, 10);
    let run = |e: &mut Engine| -> Vec<Vec<u32>> {
        let mut s = e.session(&prompt).expect("session");
        let mut rows = vec![bits(s.logits())];
        for &t in &steps {
            s.step(t).expect("step");
            rows.push(bits(s.logits()));
        }
        s.finish().expect("finish");
        rows
    };
    let fresh = run(&mut f.engine(600));
    let mut e = f.engine(600);
    let noise = f.tokens(22, 600);
    // Longer, shorter, abandoned, refused, greedy stopped early: each leaves nothing behind.
    let _ = e
        .logits_at(&noise, &[599], Route::Prefill)
        .expect("a long prefill");
    assert_eq!(run(&mut e), fresh, "after a long prefill");
    drop(e.session(&noise[..300]).expect("session"));
    assert_eq!(run(&mut e), fresh, "after a session dropped without finish");
    {
        let mut s = e.session(&noise[..5]).expect("session");
        s.step(1).expect("step");
        assert!(s.step(f.cfg.vocab_size as u32).is_err());
    }
    assert_eq!(run(&mut e), fresh, "after a refused step");
    assert!(e.session(&[f.cfg.vocab_size as u32]).is_err());
    assert!(e.session(&[]).is_err());
    assert_eq!(run(&mut e), fresh, "after refused sessions");
    {
        let mut g = e
            .session(&noise[..40])
            .expect("session")
            .into_greedy(200)
            .expect("greedy");
        for _ in 0..7 {
            g.next_token().expect("token");
        }
    }
    assert_eq!(
        run(&mut e),
        fresh,
        "after a greedy session stopped early and dropped"
    );
    let _ = e
        .logits_at(&noise[..50], &[49], Route::Decode)
        .expect("decode route");
    e.set_decode_mode(DecodeMode::Eager).expect("eager");
    let _ = e
        .logits_at(&noise[..50], &[49], Route::Decode)
        .expect("eager decode");
    e.set_decode_mode(DecodeMode::Graph).expect("graph");
    assert_eq!(run(&mut e), fresh, "after the decode routes");
}

/// The client bytes, on the device and on the host, are zero after every way out of a
/// request: finished, dropped, refused, failed, unwound by a panic.
#[test]
#[ignore = "needs the GPU"]
fn client_bytes_are_zero_after_every_way_out_of_a_request() {
    let f = Fixture::new(tiny(2, 4, 2, false), 11);
    let mut e = f.engine(128);
    assert!(scrubbed(&e), "a new engine");
    let ids = f.tokens(23, 100);
    {
        let s = e.session(&ids).expect("session");
        assert!(s.logits().iter().any(|&x| x != 0.0));
        s.finish().expect("finish");
    }
    assert!(scrubbed(&e), "after finish");
    drop(e.session(&ids).expect("session"));
    assert!(scrubbed(&e), "after a drop");
    let _ = e.greedy(&ids, 20).expect("greedy");
    assert!(scrubbed(&e), "after a greedy pass");
    let _ = e.run(&ids, Route::Decode, &[3]).expect("run");
    assert!(scrubbed(&e), "after a teacher-forced pass");
    assert!(e.run(&ids, Route::Prefill, &[100]).is_err());
    assert!(e.greedy(&ids, 30).is_err(), "100 + 30 tokens in 128");
    assert!(scrubbed(&e), "after refused requests");
    {
        let mut g = e
            .session(&ids[..10])
            .expect("session")
            .into_greedy(50)
            .expect("greedy");
        g.next_token().expect("token");
    }
    assert!(
        scrubbed(&e),
        "after a greedy session dropped with steps in flight"
    );
    let unwound = catch_unwind(AssertUnwindSafe(|| {
        let mut s = e.session(&ids[..50]).expect("session");
        s.step(3).expect("step");
        panic!("the caller panics with a session open");
    }));
    assert!(unwound.is_err());
    assert!(scrubbed(&e), "after a panic with a session open");
    // The engine still serves.
    let s = e
        .session(&ids[..50])
        .expect("a session after all of the above");
    assert_logits(
        "after every way out",
        s.logits(),
        &f.model.logits(&ids[..50])[49],
    );
    s.finish().expect("finish");
}

/// A session against a model of it, on random operations: where it is, what it holds, what it
/// refuses. The model is the reference forward and its cache.
#[test]
#[ignore = "needs the GPU"]
fn random_sessions_match_a_model_of_them() {
    let f = Fixture::new(tiny(2, 4, 2, false), 12);
    let max_ctx = 48;
    let mut e = f.engine(max_ctx);
    let vocab = f.cfg.vocab_size as u32;
    for seed in 0..40 {
        let mut rng = Rng::new(1000 + seed);
        let len = *rng.pick(&[1usize, 2, 7, 40, 47, 48]);
        let prompt = rng.tokens(len, vocab as usize);
        let mut cache = Cache::default();
        let mut want = f.model.extend(&mut cache, &prompt);
        let mut s = e.session(&prompt).expect("session");
        for op in 0..60 {
            let what = format!("seed {seed} op {op} at {}", cache.len());
            assert_eq!(s.position(), cache.len(), "{what}: position");
            match rng.below(8) {
                0 => {
                    // A token outside the vocabulary: refused, nothing moves.
                    let bad = *rng.pick(&[vocab, vocab + 1, u32::MAX]);
                    let before = bits(s.logits());
                    assert!(
                        matches!(s.step(bad), Err(Error::Format(_))),
                        "{what}: token {bad}"
                    );
                    assert_eq!(
                        bits(s.logits()),
                        before,
                        "{what}: a refused step changed the logits"
                    );
                }
                1 => assert_logits(&what, s.logits(), &want),
                _ => {
                    let t = rng.below(vocab as usize) as u32;
                    if cache.len() == max_ctx {
                        let before = bits(s.logits());
                        assert!(
                            matches!(s.step(t), Err(Error::Format(_))),
                            "{what}: a step in a full context"
                        );
                        assert_eq!(bits(s.logits()), before);
                    } else {
                        s.step(t).unwrap_or_else(|err| panic!("{what}: {err}"));
                        want = f.model.step(&mut cache, t);
                    }
                }
            }
        }
        assert_logits(&format!("seed {seed}: at the end"), s.logits(), &want);
        // The way out, at random: finished, dropped, or turned into a greedy session.
        match rng.below(3) {
            0 => s.finish().expect("finish"),
            1 => drop(s),
            _ => {
                let room = max_ctx - cache.len();
                assert!(
                    s.into_greedy(room + 2).is_err(),
                    "seed {seed}: more greedy tokens than the context holds"
                );
            }
        }
        assert!(scrubbed(&e), "seed {seed}: client bytes after the session");
    }
}

/// The edges of the context: a prompt of one token, a prompt of the whole context, the last
/// step that fits and the first that does not, the model's whole position range.
#[test]
#[ignore = "needs the GPU"]
fn the_edges_of_the_context_are_served_and_bounded() {
    let f = Fixture::new(tiny(1, 2, 1, false), 13);
    let full = f.cfg.max_position_embeddings;
    let mut e = f.engine(full);
    assert_eq!(e.max_ctx(), full);
    let ids = f.tokens(24, full);
    let want = f.model.logits(&ids);

    let one = e
        .logits_at(&ids[..1], &[0], Route::Prefill)
        .expect("one token");
    assert_logits("a prompt of one token", &one[0], &want[0]);
    let one = e
        .logits_at(&ids[..1], &[0], Route::Decode)
        .expect("one token");
    assert_logits("a prompt of one token, decode", &one[0], &want[0]);
    assert_eq!(e.greedy(&ids[..1], 1).expect("one in, one out").len(), 1);

    // The whole context in one prefill, and filled step by step up to its last position.
    let last = e
        .logits_at(&ids, &[full - 2, full - 1], Route::Prefill)
        .expect("the whole context");
    assert_logits("position max - 2", &last[0], &want[full - 2]);
    assert_logits("position max - 1", &last[1], &want[full - 1]);
    let mut s = e.session(&ids[..full - 3]).expect("session");
    for p in full - 3..full {
        s.step(ids[p]).expect("a step inside the context");
        assert_logits(&format!("stepped to {p}"), s.logits(), &want[p]);
    }
    assert_eq!(s.position(), full);
    assert!(
        matches!(s.step(0), Err(Error::Format(_))),
        "a step past the context"
    );
    assert!(s.into_greedy(2).is_err(), "a greedy step past the context");
    let g = e
        .session(&ids)
        .expect("a prompt of the whole context")
        .into_greedy(1)
        .expect("one token needs no step");
    drop(g);

    // The bounds of each entry point.
    assert!(e.session(&ids).is_ok());
    let mut longer = ids.clone();
    longer.push(0);
    assert!(matches!(e.session(&longer), Err(Error::Format(_))));
    assert!(e.logits_at(&longer, &[0], Route::Prefill).is_err());
    assert!(
        e.greedy(&ids, 1).is_ok(),
        "the last position's argmax needs no step"
    );
    assert!(e.greedy(&ids, 2).is_err());
    assert!(e.greedy(&ids[..full - 9], 10).is_ok());
    assert!(e.greedy(&ids[..full - 9], 11).is_err());
    assert!(e.greedy(&ids[..4], 0).is_err());
    assert!(e.greedy(&[], 4).is_err());
    for ahead in [0, GREEDY_AHEAD_MAX + 1, usize::MAX] {
        assert!(
            e.session(&ids[..4])
                .expect("session")
                .into_greedy_ahead(4, ahead)
                .is_err(),
            "{ahead} ahead"
        );
    }
    assert!(e
        .session(&ids[..4])
        .expect("session")
        .into_greedy(0)
        .is_err());
    assert!(scrubbed(&e));

    // A context of one position.
    let mut small = f.engine(1);
    let row = small
        .logits_at(&ids[..1], &[0], Route::Prefill)
        .expect("max_ctx 1");
    assert_logits("max_ctx 1", &row[0], &want[0]);
    assert!(small.logits_at(&ids[..2], &[0], Route::Prefill).is_err());
    let mut s = small.session(&ids[..1]).expect("session");
    assert!(s.step(0).is_err());
}

/// The decode attention kernels and the two merges inside the engine: per-head and split-K
/// agree within the tolerance; the two merges give the same bits where both exist.
#[test]
#[ignore = "needs the GPU"]
fn the_decode_attention_kernels_agree_inside_the_engine() {
    for f in [
        Fixture::new(tiny(2, 4, 2, false), 14),
        Fixture::new(wide(), 15),
    ] {
        let mut e = f.engine(700);
        let ids = f.tokens(25, 520);
        let keep = [0, 127, 128, 129, 300, 519];
        let want = f.model.logits(&ids);
        assert_eq!(e.decode_attention(), DecodeAttention::SplitK);
        let split = e.logits_at(&ids, &keep, Route::Decode).expect("split-K");
        e.set_decode_attention(DecodeAttention::PerHead)
            .expect("per head");
        let per_head = e.logits_at(&ids, &keep, Route::Decode).expect("per head");
        e.set_decode_attention(DecodeAttention::SplitK)
            .expect("split-K");
        for ((a, b), &p) in split.iter().zip(&per_head).zip(&keep) {
            assert_logits(&format!("split-K at {p}"), a, &want[p]);
            assert_logits(&format!("per-head at {p}"), b, &want[p]);
            assert_logits(&format!("split-K against per-head at {p}"), a, b);
        }
        eprintln!("MEASURE P = {}, merge {:?}", e.splits(), e.split_merge());
        match e.set_split_merge(SplitMerge::LastBlock) {
            Ok(()) => {
                let last = e
                    .logits_at(&ids, &keep, Route::Decode)
                    .expect("last-block merge");
                for ((a, b), &p) in split.iter().zip(&last).zip(&keep) {
                    assert_eq!(bits(a), bits(b), "the two merges differ at {p}");
                }
                e.set_split_merge(SplitMerge::Kernel).expect("merge kernel");
            }
            Err(err) => eprintln!("MEASURE no last-block merge for this geometry: {err}"),
        }
        let again = e
            .logits_at(&ids, &keep, Route::Decode)
            .expect("split-K again");
        assert_eq!(
            split.iter().map(|r| bits(r)).collect::<Vec<_>>(),
            again.iter().map(|r| bits(r)).collect::<Vec<_>>()
        );
    }
}

/// A checkpoint with an infinity in one embedding row: a request that feeds that token has no
/// finite logits, its greedy pass is refused and leaves nothing; requests that do not feed it
/// are served, before and after, with the same bits.
#[test]
#[ignore = "needs the GPU"]
fn a_token_whose_embedding_is_not_finite_poisons_its_request_only() {
    const BAD: u32 = 7;
    for special in [f32::INFINITY, f32::NAN] {
        let cfg = tiny(2, 4, 2, false);
        let hidden = cfg.hidden_size;
        let f = Fixture::with(cfg, 16, |name, values| {
            if name == "model.embed_tokens.weight" {
                values[BAD as usize * hidden + 3] = special;
            }
        });
        let mut e = f.engine(64);
        let clean: Vec<u32> = f
            .tokens(26, 30)
            .into_iter()
            .map(|t| if t == BAD { BAD + 1 } else { t })
            .collect();
        let (before, row_before) = e
            .greedy_with_last_logits(&clean, 10)
            .expect("a clean request");
        assert!(row_before.iter().all(|x| x.is_finite()));

        let mut dirty = clean.clone();
        dirty[12] = BAD;
        assert!(
            matches!(e.greedy(&dirty, 5), Err(Error::Gpu(_))),
            "{special}: a greedy token from a NaN row"
        );
        assert!(scrubbed(&e), "{special}: after the refused greedy pass");
        // The teacher-forced pass takes the argmax of every row: it refuses the request too.
        assert!(
            matches!(
                e.logits_at(&dirty, &[11, 12, 29], Route::Prefill),
                Err(Error::Gpu(_))
            ),
            "{special}: a teacher-forced pass over a NaN row"
        );
        assert!(scrubbed(&e), "{special}: after the refused pass");
        let rows = e
            .logits_at(&dirty[..12], &[11], Route::Prefill)
            .expect("the tokens before it are a request of their own");
        assert!(
            rows[0].iter().all(|x| x.is_finite()),
            "{special}: a position before the token"
        );
        {
            let s = e
                .session(&dirty[..13])
                .expect("a session opens on any logits");
            assert!(
                s.logits().iter().all(|x| x.is_nan()),
                "{special}: the token's own position"
            );
            s.finish().expect("the session ends");
        }
        // The reference says the same of which rows have a value.
        let want = f.model.logits(&dirty);
        assert!(
            want[11].iter().all(|x| x.is_finite())
                && want[12].iter().all(|x| x.is_nan())
                && want[29].iter().all(|x| x.is_nan())
        );
        {
            let mut s = e.session(&dirty).expect("a session opens on any logits");
            assert!(s.logits().iter().all(|x| x.is_nan()));
            assert!(metrics::argmax(s.logits()).is_none());
            s.step(1).expect("a step on a poisoned cache is a step");
            let g = s.into_greedy(3);
            assert!(
                g.is_err() || g.is_ok_and(|mut g| g.next_token().is_err()),
                "{special}: a greedy token from a NaN row"
            );
        }
        assert!(scrubbed(&e), "{special}: after the poisoned session");

        let (after, row_after) = e
            .greedy_with_last_logits(&clean, 10)
            .expect("a clean request afterwards");
        assert_eq!(
            before, after,
            "{special}: the poisoned request changed a later one"
        );
        assert_eq!(bits(&row_before), bits(&row_after));
    }
}
