//! The generation loop over a backend that computes and the engine that decides.
//!
//! A [`LogitsSession`] holds one request on a backend: the prompt is prefilled
//! when it opens, [`LogitsSession::logits`] are the f32 logits of the last
//! position, and [`LogitsSession::step`] feeds the token the engine chose.
//! Everything else is here, once for every such backend, with the semantics of
//! the candle loop (`candle_backend::generate_streaming`): the budget, the
//! choice of the token (greedy, or [`crate::sampler`] when sampling: candle's
//! draws, without candle),
//! the `json_schema` mask ([`crate::constrain`]), EOS, stop sequences, the
//! stream, cancellation, the prefix check, and [`LogitsSession::finish`] on
//! every path once a session is open.
//!
//! A greedy request without a schema lets a backend that can pick the token
//! on its device do so ([`DevicePick::Greedy`]): the host then reads the
//! token, not a row of logits, and the token is the one [`argmax`] picks.

use tokenizers::Tokenizer;

use crate::backend::{
    Cancelled, Completion, GenerationParams, Held, JsonSchemaUnsupported, NotPrefixStable,
    TokenSink,
};
use crate::constrain::{Constraint, TokenTable};
use crate::decode_stream::StreamDecoder;
use crate::sampler::Sampler;

/// A rule by which a backend picks the next token on its device. Greedy only:
/// a seeded draw would join it once its tokens are proven to be the host's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DevicePick {
    /// The row's [`argmax`], fed to the model on the device.
    Greedy,
}

/// One request on a backend that hands out logits and takes the next token.
pub trait LogitsSession {
    /// f32 logits of the last position, one per vocabulary entry.
    fn logits(&self) -> &[f32];
    /// Feeds `token`: the logits become those of the position after it.
    fn step(&mut self, token: u32) -> anyhow::Result<()>;
    /// Ends the request. The backend clears what it held for it, and says so
    /// when it could not.
    fn finish(self) -> anyhow::Result<()>;
    /// Whether the backend picks every next token by `rule` on its device.
    /// Asked once, when the session is open; none does by default.
    fn picks_on_device(&self, rule: DevicePick) -> bool {
        let _ = rule;
        false
    }
    /// The next token by `rule`, already fed to the model. `budget` is the
    /// most tokens the request takes. Called instead of `logits` and `step`,
    /// and only when [`Self::picks_on_device`] said yes.
    fn next_on_device(&mut self, rule: DevicePick, budget: usize) -> anyhow::Result<u32> {
        let _ = (rule, budget);
        anyhow::bail!("this backend picks no token on its device")
    }
}

/// The index of the largest value, the smallest index on a tie; `None` for an
/// empty row or one that holds a NaN. The rule of x0ne Core's argmax kernel,
/// so greedy tokens picked here are the ones Core picks on the device.
pub fn argmax(row: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in row.iter().enumerate() {
        if v.is_nan() {
            return None;
        }
        // Strictly greater: the first of equal values stays, -inf included.
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.and_then(|(i, _)| u32::try_from(i).ok())
}

/// The token `row` picks: greedy ([`argmax`]), or `sampler`'s draw over it.
/// A row that holds a NaN picks none.
pub fn pick(row: &[f32], sampler: Option<&mut Sampler>) -> anyhow::Result<u32> {
    match sampler {
        None => argmax(row).ok_or_else(|| anyhow::anyhow!("the logits hold a NaN: no token")),
        Some(s) => s.sample(row),
    }
}

/// The token table a `json_schema` request needs, read from `tokenizer`;
/// `None` without a schema. A schema beside stop sequences, or over a
/// tokenizer whose tokens are not their text, is refused before any device.
pub fn table_for(
    tokenizer: &Tokenizer,
    eos_ids: &[u32],
    params: &GenerationParams,
) -> Result<Option<TokenTable>, JsonSchemaUnsupported> {
    if params.json_schema.is_none() {
        return Ok(None);
    }
    refuse_stop(params)?;
    TokenTable::from_tokenizer(tokenizer, eos_ids)
        .map(Some)
        .map_err(JsonSchemaUnsupported)
}

/// A stop sequence would cut a document: refused beside a schema, as the
/// request parser refuses it.
pub fn refuse_stop(params: &GenerationParams) -> Result<(), JsonSchemaUnsupported> {
    if params.json_schema.is_some() && !params.stop.is_empty() {
        return Err(JsonSchemaUnsupported(
            "stop: not with json_schema (a stop sequence would cut the document)".into(),
        ));
    }
    Ok(())
}

/// A request tokenized and budgeted, before any lock or device: the refusals
/// that cost nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    pub prompt_ids: Vec<u32>,
    /// `max_tokens` cut to what the context leaves ([`crate::backend::generation_budget`]).
    pub max_tokens: usize,
}

/// Tokenizes `prompt` as the candle backend does ([`crate::prompt::encode`]:
/// a [`crate::prompt::text_only`] tokenizer) and applies the budget.
pub fn prepare(
    tokenizer: &Tokenizer,
    prompt: &crate::prompt::Prompt,
    max_tokens: usize,
    context_tokens: usize,
) -> anyhow::Result<Prepared> {
    let prompt_ids = crate::prompt::encode(tokenizer, prompt)?;
    let max_tokens =
        crate::backend::generation_budget(prompt_ids.len(), max_tokens, context_tokens)?;
    Ok(Prepared {
        prompt_ids,
        max_tokens,
    })
}

/// Generates `req` through the session `open` returns, called after
/// `sink.started()` with the prompt's ids. `eos_ids` end the answer. Under a
/// `json_schema`, the token table is read from `tokenizer` per request
/// (79 ms for Qwen3's 151,669 ids, measured 2026-10-30).
pub fn run<S: LogitsSession>(
    tokenizer: &Tokenizer,
    eos_ids: &[u32],
    req: &Prepared,
    params: &GenerationParams,
    sink: &mut dyn TokenSink,
    open: impl FnOnce(&[u32]) -> anyhow::Result<S>,
) -> anyhow::Result<Completion> {
    let table = table_for(tokenizer, eos_ids, params)?;
    let mut constraint = params
        .json_schema
        .as_ref()
        .zip(table.as_ref())
        .map(|(schema, table)| Constraint::new(schema, table, eos_ids));
    let mut sampler = params.sampling.map(Sampler::new);
    // Every refusal is past: the 200 may go before the prefill.
    if sink.started().is_break() {
        return Err(Cancelled.into());
    }
    if req.prompt_ids.is_empty() {
        anyhow::bail!("empty prompt");
    }
    let mut session = open(&req.prompt_ids)?;
    // The token argmax would pick from the row, read without the row.
    let on_device = constraint.is_none()
        && sampler.as_ref().is_none_or(Sampler::is_greedy)
        && session.picks_on_device(DevicePick::Greedy);
    let mut emitted = String::new();
    // A closure, so that no `?` inside it can skip the finish below.
    // The document closed: no byte may follow, the answer ends without EOS.
    let mut closed = false;
    let mut decide = || -> anyhow::Result<Vec<u32>> {
        let mut decoder = StreamDecoder::new(tokenizer, &params.stop);
        let mut generated = Vec::new();
        for step in 0..req.max_tokens {
            if !sink.alive() {
                return Err(Cancelled.into());
            }
            let next = match constraint.as_ref() {
                None if on_device => session.next_on_device(DevicePick::Greedy, req.max_tokens)?,
                None => pick(session.logits(), sampler.as_mut())?,
                Some(c) => {
                    let mut row = session.logits().to_vec();
                    c.mask_logits(&mut row)?;
                    pick(&row, sampler.as_mut())?
                }
            };
            if eos_ids.contains(&next) {
                break;
            }
            if let Some(c) = constraint.as_mut() {
                c.advance(next)?;
                closed = c.done();
            }
            generated.push(next);
            let (piece, stop) = match decoder.push(next)? {
                Held::Emit(piece) => (piece, false),
                Held::Stop(piece) => (piece, true),
            };
            if !piece.is_empty() {
                if sink.delta(&piece).is_break() {
                    return Err(Cancelled.into());
                }
                emitted.push_str(&piece);
            }
            if stop || closed || step + 1 == req.max_tokens {
                break;
            }
            if !on_device {
                session.step(next)?;
            }
        }
        Ok(generated)
    };
    let generated = decide();
    // Unconditional, and its failure is the request's: an answer is never
    // returned over a session the backend could not clear.
    let finished = session.finish();
    let generated = generated?;
    finished?;
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    let (text, stopped) = crate::backend::cut_at_stop(&text, &params.stop);
    let Some(rest) = text.strip_prefix(emitted.as_str()) else {
        return Err(NotPrefixStable.into());
    };
    if constraint.as_ref().is_some_and(|c| !c.joins_to(&text)) {
        anyhow::bail!("json_schema: the decoded text is not the bytes the mask let through");
    }
    if !rest.is_empty() && sink.delta(rest).is_break() {
        return Err(Cancelled.into());
    }
    Ok(Completion {
        text,
        prompt_tokens: req.prompt_ids.len() as u32,
        completion_tokens: generated.len() as u32,
        stopped,
        length: !closed && generated.len() >= req.max_tokens,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::{NoSink, Sampling};
    use std::cell::RefCell;
    use std::ops::ControlFlow;
    use std::rc::Rc;

    /// What a sink saw, and where it stops reading.
    #[derive(Default)]
    pub(crate) struct Recorder {
        pub(crate) started: usize,
        pub(crate) deltas: Vec<String>,
        /// `Break` on this delta (0-based).
        pub(crate) break_at: Option<usize>,
        /// `alive()` false from this many calls on.
        pub(crate) dead_after: Option<usize>,
        pub(crate) alive_calls: std::cell::Cell<usize>,
        pub(crate) refuse_start: bool,
    }

    impl TokenSink for Recorder {
        fn started(&mut self) -> ControlFlow<()> {
            self.started += 1;
            if self.refuse_start {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
        fn delta(&mut self, text: &str) -> ControlFlow<()> {
            let n = self.deltas.len();
            self.deltas.push(text.to_string());
            if self.break_at == Some(n) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
        fn alive(&self) -> bool {
            let n = self.alive_calls.get();
            self.alive_calls.set(n + 1);
            self.dead_after.is_none_or(|d| n < d)
        }
    }

    /// What the scripted backend was asked.
    #[derive(Default, Debug)]
    struct Log {
        opened: Vec<Vec<u32>>,
        fed: Vec<u32>,
        finished: usize,
        /// Rows read by the loop.
        rows_read: usize,
        /// Budgets given to `next_on_device`.
        budgets: Vec<usize>,
    }

    /// A backend whose next token is `(last fed + 1) % vocab`, peaked at `peak`;
    /// `fail_at`: the step that fails. `device`: it picks greedy tokens on its
    /// device, feeding each one it returns.
    struct Scripted {
        vocab: usize,
        row: Vec<f32>,
        log: Rc<RefCell<Log>>,
        fail_at: Option<usize>,
        finish_fails: bool,
        device: bool,
    }

    fn peak(vocab: usize, at: u32) -> Vec<f32> {
        let mut row = vec![0.0f32; vocab];
        row[at as usize] = 10.0;
        row
    }

    impl LogitsSession for Scripted {
        fn logits(&self) -> &[f32] {
            self.log.borrow_mut().rows_read += 1;
            &self.row
        }
        fn step(&mut self, token: u32) -> anyhow::Result<()> {
            let n = self.log.borrow().fed.len();
            if self.fail_at == Some(n) {
                anyhow::bail!("device failure");
            }
            self.log.borrow_mut().fed.push(token);
            self.row = peak(self.vocab, (token + 1) % self.vocab as u32);
            Ok(())
        }
        fn finish(self) -> anyhow::Result<()> {
            self.log.borrow_mut().finished += 1;
            if self.finish_fails {
                anyhow::bail!("scrub failed");
            }
            Ok(())
        }
        fn picks_on_device(&self, rule: DevicePick) -> bool {
            self.device && rule == DevicePick::Greedy
        }
        fn next_on_device(&mut self, rule: DevicePick, budget: usize) -> anyhow::Result<u32> {
            assert!(self.device && rule == DevicePick::Greedy);
            self.log.borrow_mut().budgets.push(budget);
            let next = argmax(&self.row).unwrap();
            // The device feeds the token it picks, the last one included.
            let n = self.log.borrow().fed.len();
            if self.fail_at == Some(n) {
                anyhow::bail!("device failure");
            }
            self.log.borrow_mut().fed.push(next);
            self.row = peak(self.vocab, (next + 1) % self.vocab as u32);
            Ok(next)
        }
    }

    /// The ids `prepare` hands a backend (Core): each marker of the template its
    /// special token, a marker typed in a message text. A failure means Core
    /// reads a control token a caller wrote, the defect found 2026-10-26.
    #[test]
    fn prepare_reads_a_marker_typed_in_a_message_as_text() {
        let tok = crate::prompt::text_only(
            Tokenizer::from_bytes(crate::prompt::tests::chatml_tokenizer_json()).unwrap(),
        );
        let prompt = crate::arch::Architecture::Qwen3.chat_prompt(&[crate::server::ChatMessage {
            role: "user".into(),
            content: crate::prompt::tests::FORGED.into(),
        }]);
        let p = prepare(&tok, &prompt, 4, 64).unwrap();
        assert_eq!(p.prompt_ids, crate::prompt::tests::wanted(&prompt));
    }

    pub(crate) fn tokenizer(n: usize) -> Tokenizer {
        crate::prompt::text_only(
            Tokenizer::from_bytes(crate::candle_backend::tests::word_tokenizer(n)).unwrap(),
        )
    }

    struct Case {
        prompt: &'static str,
        max_tokens: usize,
        context: usize,
        eos: Vec<u32>,
        stop: Vec<String>,
        fail_at: Option<usize>,
        finish_fails: bool,
        device: bool,
        sampling: Option<Sampling>,
    }

    impl Default for Case {
        fn default() -> Self {
            Self {
                prompt: "t1 t2 t3",
                max_tokens: 5,
                context: 64,
                eos: vec![],
                stop: vec![],
                fail_at: None,
                finish_fails: false,
                device: false,
                sampling: None,
            }
        }
    }

    fn go(c: &Case, sink: &mut dyn TokenSink) -> (anyhow::Result<Completion>, Log) {
        let tk = tokenizer(16);
        let log = Rc::new(RefCell::new(Log::default()));
        let req = prepare(&tk, &c.prompt.into(), c.max_tokens, c.context).unwrap();
        let mut params = GenerationParams::greedy(c.max_tokens);
        params.stop = c.stop.clone();
        params.sampling = c.sampling;
        let l = log.clone();
        let out = run(&tk, &c.eos, &req, &params, sink, |ids: &[u32]| {
            l.borrow_mut().opened.push(ids.to_vec());
            Ok(Scripted {
                vocab: 16,
                row: peak(16, (ids[ids.len() - 1] + 1) % 16),
                log: l.clone(),
                fail_at: c.fail_at,
                finish_fails: c.finish_fails,
                device: c.device,
            })
        });
        drop(l);
        let log = Rc::try_unwrap(log).unwrap().into_inner();
        (out, log)
    }

    /// The prompt opens the session, each chosen token is fed back but the
    /// last, and the session is finished once.
    #[test]
    fn the_loop_feeds_each_token_but_the_last_and_finishes_once() {
        let mut sink = Recorder::default();
        let (c, log) = go(&Case::default(), &mut sink);
        let c = c.unwrap();
        assert_eq!(c.text, "t4 t5 t6 t7 t8");
        assert_eq!((c.prompt_tokens, c.completion_tokens), (3, 5));
        assert!(c.length && !c.stopped);
        assert_eq!(log.opened, vec![vec![1, 2, 3]]);
        assert_eq!(log.fed, vec![4, 5, 6, 7]);
        assert_eq!(log.finished, 1);
        assert_eq!(sink.started, 1);
        assert_eq!(sink.deltas.concat(), c.text);
    }

    /// EOS ends the answer before it is fed or emitted.
    #[test]
    fn eos_ends_the_answer_unfed() {
        let case = Case {
            eos: vec![6],
            ..Case::default()
        };
        let (c, log) = go(&case, &mut NoSink);
        let c = c.unwrap();
        assert_eq!(c.text, "t4 t5");
        assert_eq!(c.completion_tokens, 2);
        assert!(!c.length && !c.stopped);
        assert_eq!(log.fed, vec![4, 5]);
        assert_eq!(log.finished, 1);
    }

    /// A stop sequence ends the loop at the token that completed it.
    #[test]
    fn a_stop_sequence_ends_the_loop() {
        let case = Case {
            stop: vec!["t6".into()],
            ..Case::default()
        };
        let mut sink = Recorder::default();
        let (c, log) = go(&case, &mut sink);
        let c = c.unwrap();
        assert_eq!(c.text, "t4 t5 ");
        assert!(c.stopped);
        assert_eq!(c.completion_tokens, 3);
        assert_eq!(log.fed, vec![4, 5]);
        assert_eq!(log.finished, 1);
        assert_eq!(sink.deltas.concat(), c.text);
    }

    /// The context window cuts the budget, and that is `length`.
    #[test]
    fn the_context_cuts_the_budget() {
        let case = Case {
            context: 5,
            ..Case::default()
        };
        let (c, _) = go(&case, &mut NoSink);
        let c = c.unwrap();
        assert_eq!(c.completion_tokens, 2);
        assert!(c.length);
        let full = Case {
            context: 3,
            ..Case::default()
        };
        let tk = tokenizer(16);
        assert!(
            prepare(&tk, &full.prompt.into(), 5, full.context).is_err(),
            "a prompt that fills the context"
        );
    }

    /// Every way out after the session opened finishes it, once.
    #[test]
    fn every_path_finishes_the_open_session() {
        let fail = Case {
            fail_at: Some(2),
            ..Case::default()
        };
        let (c, log) = go(&fail, &mut NoSink);
        assert!(c.is_err());
        assert_eq!((log.fed.len(), log.finished), (2, 1), "a failed step");

        let mut gone = Recorder {
            dead_after: Some(2),
            ..Recorder::default()
        };
        let (c, log) = go(&Case::default(), &mut gone);
        assert!(c.unwrap_err().downcast_ref::<Cancelled>().is_some());
        assert_eq!(log.finished, 1, "a reader gone");

        let mut broke = Recorder {
            break_at: Some(1),
            ..Recorder::default()
        };
        let (c, log) = go(&Case::default(), &mut broke);
        assert!(c.unwrap_err().downcast_ref::<Cancelled>().is_some());
        assert_eq!(log.finished, 1, "a delta refused");

        let bad_finish = Case {
            finish_fails: true,
            ..Case::default()
        };
        let (c, log) = go(&bad_finish, &mut NoSink);
        assert!(c.is_err(), "a failed scrub is an error, not an answer");
        assert_eq!(log.finished, 1);
    }

    /// On the device path the loop reads no row and feeds nothing: the answer,
    /// its end, the stream and the finish are the host path's, for every case.
    #[test]
    fn a_greedy_pick_on_the_device_is_the_hosts() {
        let cases = [
            Case::default(),
            Case {
                eos: vec![6],
                ..Case::default()
            },
            Case {
                stop: vec!["t6".into()],
                ..Case::default()
            },
            Case {
                context: 5,
                ..Case::default()
            },
            Case {
                max_tokens: 1,
                ..Case::default()
            },
        ];
        for (k, host) in cases.into_iter().enumerate() {
            let device = Case {
                device: true,
                ..host
            };
            let (mut hs, mut ds) = (Recorder::default(), Recorder::default());
            let (h, hlog) = go(&device_off(&device), &mut hs);
            let (d, dlog) = go(&device, &mut ds);
            let (h, d) = (h.unwrap(), d.unwrap());
            assert_eq!(
                (&d.text, d.completion_tokens, d.stopped, d.length),
                (&h.text, h.completion_tokens, h.stopped, h.length),
                "case {k}"
            );
            assert_eq!(ds.deltas, hs.deltas, "case {k}");
            assert_eq!((dlog.rows_read, dlog.finished), (0, 1), "case {k}");
            // One pick per token decided, EOS included, each told the request's budget.
            let budget = prepare(
                &tokenizer(16),
                &device.prompt.into(),
                device.max_tokens,
                device.context,
            )
            .unwrap()
            .max_tokens;
            assert_eq!(dlog.budgets, vec![budget; dlog.fed.len()], "case {k}");
            assert!(dlog.fed.starts_with(&hlog.fed), "case {k}");
            assert!(hlog.rows_read > 0 && hlog.budgets.is_empty(), "case {k}");
        }
    }

    fn device_off(c: &Case) -> Case {
        Case {
            prompt: c.prompt,
            max_tokens: c.max_tokens,
            context: c.context,
            eos: c.eos.clone(),
            stop: c.stop.clone(),
            fail_at: c.fail_at,
            finish_fails: c.finish_fails,
            device: false,
            sampling: c.sampling,
        }
    }

    /// A draw, or a failure on the device, keeps the rules: sampling reads the
    /// rows on the host; a failed device pick finishes the session once.
    #[test]
    fn the_device_picks_only_greedy_tokens() {
        let sampled = Case {
            device: true,
            sampling: Some(Sampling {
                temperature: 1.0,
                top_p: 1.0,
                seed: 3,
            }),
            ..Case::default()
        };
        let (c, log) = go(&sampled, &mut NoSink);
        c.unwrap();
        assert!(log.budgets.is_empty() && log.rows_read == 5);
        let null_nucleus = Case {
            device: true,
            sampling: Some(Sampling {
                temperature: 1.0,
                top_p: 0.0,
                seed: 3,
            }),
            ..Case::default()
        };
        let (c, log) = go(&null_nucleus, &mut NoSink);
        assert_eq!(c.unwrap().text, "t4 t5 t6 t7 t8");
        assert_eq!((log.rows_read, log.budgets.len()), (0, 5));
        let failing = Case {
            device: true,
            fail_at: Some(2),
            ..Case::default()
        };
        let (c, log) = go(&failing, &mut NoSink);
        assert!(c.is_err());
        assert_eq!((log.fed.len(), log.finished), (2, 1));
    }

    /// A start refused opens nothing; an opening that fails has nothing to finish.
    #[test]
    fn nothing_is_opened_before_the_start() {
        let mut refused = Recorder {
            refuse_start: true,
            ..Recorder::default()
        };
        let (c, log) = go(&Case::default(), &mut refused);
        assert!(c.unwrap_err().downcast_ref::<Cancelled>().is_some());
        assert!(log.opened.is_empty() && log.finished == 0);

        let tk = tokenizer(16);
        let req = prepare(&tk, &"t1".into(), 3, 64).unwrap();
        let r = run::<Scripted>(
            &tk,
            &[],
            &req,
            &GenerationParams::greedy(3),
            &mut NoSink,
            |_| anyhow::bail!("out of memory"),
        );
        assert!(r.is_err());
    }

    /// Sampling draws through [`Sampler`]: a seed repeats its draw, `top_p` 0
    /// is greedy.
    #[test]
    fn sampling_repeats_by_seed_and_a_null_nucleus_is_greedy() {
        let tk = tokenizer(16);
        let req = prepare(&tk, &"t1".into(), 6, 64).unwrap();
        // Close, distinct logits: every token is a possible draw, one is the largest.
        let draw = |s: Option<Sampling>| {
            let mut p = GenerationParams::greedy(6);
            p.sampling = s;
            run(&tk, &[], &req, &p, &mut NoSink, |_| {
                Ok(Flat {
                    row: (0..16).map(|i| ((i * 7) % 16) as f32 * 0.05).collect(),
                })
            })
            .unwrap()
            .text
        };
        let s = |seed| {
            Some(Sampling {
                temperature: 1.0,
                top_p: 1.0,
                seed,
            })
        };
        assert_eq!(draw(s(3)), draw(s(3)));
        assert!((0..8).any(|seed| draw(s(seed)) != draw(s(seed + 1))));
        let greedy = draw(None);
        assert_eq!(
            draw(Some(Sampling {
                temperature: 1.0,
                top_p: 0.0,
                seed: 3
            })),
            greedy
        );
        assert_eq!(greedy, "t9 t9 t9 t9 t9 t9");
    }

    struct Flat {
        row: Vec<f32>,
    }

    impl LogitsSession for Flat {
        fn logits(&self) -> &[f32] {
            &self.row
        }
        fn step(&mut self, _: u32) -> anyhow::Result<()> {
            Ok(())
        }
        fn finish(self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    /// Core's kernel rule (`argmax.cu`): largest value, smallest index on a
    /// tie, a NaN anywhere refuses the row, -inf ties like any value.
    #[test]
    fn argmax_is_the_rule_of_cores_kernel() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax(&[3.0, 1.0, 3.0]), Some(0));
        assert_eq!(argmax(&[f32::NEG_INFINITY, -1e30]), Some(1));
        assert_eq!(argmax(&[f32::INFINITY, f32::INFINITY]), Some(0));
        assert_eq!(argmax(&[f32::NEG_INFINITY; 4]), Some(0));
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), None);
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), None);
        assert_eq!(argmax(&[]), None);
        assert_eq!(argmax(&[-0.0, 0.0]), Some(0), "-0 == +0: the smaller index");
    }

    /// A row with no maximum is an error, not token 0 or u32::MAX.
    #[test]
    fn a_row_without_a_maximum_is_refused() {
        let tk = tokenizer(16);
        let req = prepare(&tk, &"t1".into(), 3, 64).unwrap();
        let r = run(
            &tk,
            &[],
            &req,
            &GenerationParams::greedy(3),
            &mut NoSink,
            |_| {
                Ok(Flat {
                    row: vec![f32::NAN; 16],
                })
            },
        );
        assert!(r.is_err());
    }
}
