//! The rules every backend passes before a server may serve with it, written
//! once and run on each backend ([`crate::backends`]).
//!
//! - **gate**: the model comes only through [`crate::backend::InferenceBackend`],
//!   whose loads take the released key: no weight is loaded before the gate.
//! - **digest**: the served digest is the digest of the files given.
//! - **isolation**: the backend declares [`Isolation::NONE_OF_IT`], and what is
//!   observable of it holds: an answer does not change after other requests
//!   (another prompt, a shared prefix, sampling, a request cancelled or
//!   panicking mid-way), nor when two requests arrive at once.
//!
//! Not observable here: timing side channels of a prefix cache, and logging
//! (each crate sweeps its own sources for log statements that name the text).

use std::ops::ControlFlow;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::backend::{
    Cancelled, Completion, GenerationParams, Isolation, LoadedModel, Sampling, TokenSink,
    WeightFiles,
};
use crate::backends::NamedBackend;
use crate::keys::ReleasedKey;
use crate::model_spec::ModelArtifacts;
use crate::prompt::Prompt;
use crate::server::ChatMessage;

/// What the rules generate: two prompts and a length, greedy.
#[derive(Debug, Clone)]
pub struct Probe {
    pub prompts: [Prompt; 2],
    pub max_tokens: usize,
}

impl Probe {
    /// Two short questions in the model's own chat template, 8 tokens each.
    pub fn chat(model: &dyn LoadedModel) -> Self {
        let ask = |q: &str| {
            model.chat_prompt(&[ChatMessage {
                role: "user".into(),
                content: q.into(),
            }])
        };
        Self {
            prompts: [
                ask("Name the largest planet of the solar system."),
                ask("Write one sentence about rivers."),
            ],
            max_tokens: 8,
        }
    }
}

/// Breaks on its first delta: a reader gone mid-answer.
struct Leaves;

impl TokenSink for Leaves {
    fn started(&mut self) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    fn delta(&mut self, _: &str) -> ControlFlow<()> {
        ControlFlow::Break(())
    }
}

/// Panics on its first delta: a request that dies inside the backend's loop.
struct Panics;

impl TokenSink for Panics {
    fn started(&mut self) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    fn delta(&mut self, _: &str) -> ControlFlow<()> {
        panic!("conformance: a reader that panics");
    }
}

/// The rules on a loaded model. `Err` names each rule broken.
pub fn check_model(
    model: &dyn LoadedModel,
    expected_digest_hex: &str,
    probe: &Probe,
) -> Result<(), Vec<String>> {
    let mut broken = Vec::new();
    if model.model_digest_hex() != expected_digest_hex {
        broken.push("digest: the served digest is not the digest of the files given".into());
    }
    if model.isolation() != Isolation::NONE_OF_IT {
        broken.push("isolation: the backend does not declare Isolation::NONE_OF_IT".into());
    }
    let greedy = GenerationParams::greedy(probe.max_tokens);
    // Errors are named by rule, never quoted: a backend's message may hold text.
    let answer = |p: &Prompt| model.generate_with(p, &greedy).ok();
    let [p0, p1] = &probe.prompts;
    let (Some(a0), Some(a1)) = (answer(p0), answer(p1)) else {
        broken.push("generate: the probe's prompts are not answered".into());
        return Err(broken);
    };
    let mut same = |rule: &str, prompt: &Prompt, want: &Completion| {
        if answer(prompt).as_ref() != Some(want) {
            broken.push(format!("{rule}: the answer changed"));
        }
    };
    same("repeat", p0, &a0);
    let _ = answer(p1);
    same("after-another-prompt", p0, &a0);
    let mut shared = p0.clone();
    shared.push_prompt(p1);
    let _ = answer(&shared);
    same("after-a-shared-prefix", p0, &a0);
    let mut sampled = greedy.clone();
    sampled.sampling = Some(Sampling {
        temperature: 1.0,
        top_p: 0.9,
        seed: 7,
    });
    // A greedy-only backend refuses sampling: the rule is the next answer.
    let _ = model.generate_with(p1, &sampled);
    same("after-sampling", p0, &a0);
    let cancelled = model.generate_streaming(p1, &greedy, &mut Leaves);
    let cancelled_ok = cancelled.is_err_and(|e| e.downcast_ref::<Cancelled>().is_some());
    same("after-a-cancelled-request", p0, &a0);
    let _ = catch_unwind(AssertUnwindSafe(|| {
        model.generate_streaming(p1, &greedy, &mut Panics)
    }));
    same("after-a-panic", p0, &a0);
    let (r0, r1) = std::thread::scope(|s| {
        let t0 = s.spawn(|| answer(p0));
        let t1 = s.spawn(|| answer(p1));
        (t0.join().ok().flatten(), t1.join().ok().flatten())
    });
    if r0.as_ref() != Some(&a0) || r1.as_ref() != Some(&a1) {
        broken.push("concurrent-requests: an answer changed when two arrived at once".into());
    }
    if !cancelled_ok && a1.completion_tokens > 0 {
        broken.push("cancel: a reader gone did not end the request in Cancelled".into());
    }
    if broken.is_empty() {
        Ok(())
    } else {
        Err(broken)
    }
}

/// The whole suite on `backend`: the model is loaded through the released
/// key, its expected digest computed here from the files, then
/// [`check_model`] with the probe `probe` builds for it.
pub fn check(
    backend: &dyn NamedBackend,
    artifacts: ModelArtifacts,
    weights: &WeightFiles<'_>,
    device: &str,
    key: &ReleasedKey,
    probe: impl FnOnce(&dyn LoadedModel) -> Probe,
) -> Result<(), Vec<String>> {
    let expected = crate::model_spec::served_digest(
        &artifacts.config_json,
        &artifacts.tokenizer_json,
        weights,
    )
    .map_err(|e| vec![format!("digest: {e:#}")])?;
    let opened = backend
        .open(artifacts, device)
        .map_err(|e| vec![format!("open: {e:#}")])?;
    let mut broken = Vec::new();
    // A blob that claims the seal: only `load`, which unseals with the key, opens it.
    let mut claim = crate::seal::MAGIC.to_vec();
    claim.push(crate::seal::VERSION);
    claim.resize(64, 0);
    let sealed = WeightFiles {
        index_json: None,
        files: vec![(crate::shards::SINGLE_FILE, &claim[..])],
    };
    if opened.load_files(&sealed, key).is_ok() {
        broken.push("gate: load_files opened weights that claim the seal".into());
    }
    let model = match opened.load_files(weights, key) {
        Ok(m) => m,
        Err(e) => {
            broken.push(format!("load: {e:#}"));
            return Err(broken);
        }
    };
    let want_device = if device == "cuda" { "cuda:0" } else { device };
    if model.compute_device() != want_device {
        broken.push(format!(
            "device: asked {device}, computes on {}",
            model.compute_device()
        ));
    }
    let p = probe(model.as_ref());
    if let Err(b) = check_model(model.as_ref(), &expected, &p) {
        broken.extend(b);
    }
    if broken.is_empty() {
        Ok(())
    } else {
        Err(broken)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InferenceBackend;
    use crate::candle_backend::tests::{fixture, word_tokenizer, words};
    use std::sync::Mutex;

    fn candle_probe() -> Probe {
        let (_, _, reference) = fixture("qwen3");
        let ids: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
        Probe {
            prompts: [words(&ids), "t5 t9 t2 t31 t7".into()],
            max_tokens: 10,
        }
    }

    /// candle passes, loaded through the released key from the fixture's files.
    #[test]
    fn candle_passes_the_suite() {
        let (config_json, weights, _) = fixture("qwen3");
        let artifacts = ModelArtifacts {
            config_json,
            tokenizer_json: word_tokenizer(40),
        };
        let files = WeightFiles {
            index_json: None,
            files: vec![(crate::shards::SINGLE_FILE, &weights[..])],
        };
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let r = check(
            &crate::backends::Candle,
            artifacts,
            &files,
            "cpu",
            &key,
            |_| candle_probe(),
        );
        assert_eq!(r, Ok(()));
    }

    /// A model that answers what the echo backend answers, with one defect.
    struct Faulty {
        inner: Box<dyn LoadedModel>,
        remembers: bool,
        claims_shared_kv: bool,
        last: Mutex<usize>,
    }

    impl LoadedModel for Faulty {
        fn generate(
            &self,
            prompt: &crate::prompt::Prompt,
            max_tokens: usize,
        ) -> anyhow::Result<(String, (u32, u32))> {
            let (text, n) = self.inner.generate(prompt, max_tokens)?;
            let mut last = self.last.lock().unwrap();
            let before = *last;
            *last = prompt.len();
            if self.remembers {
                return Ok((format!("{text} {before}"), n));
            }
            Ok((text, n))
        }
        fn model_digest_hex(&self) -> String {
            self.inner.model_digest_hex()
        }
        fn compute_device(&self) -> String {
            self.inner.compute_device()
        }
        fn isolation(&self) -> Isolation {
            Isolation {
                kv_cache_shared_between_requests: self.claims_shared_kv,
                ..Isolation::NONE_OF_IT
            }
        }
    }

    fn echo() -> Box<dyn LoadedModel> {
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        crate::backend::EchoBackend.load(b"w", &key).unwrap()
    }

    fn faulty(remembers: bool, claims_shared_kv: bool) -> Faulty {
        Faulty {
            inner: echo(),
            remembers,
            claims_shared_kv,
            last: Mutex::new(0),
        }
    }

    fn probe() -> Probe {
        Probe {
            prompts: ["a b c".into(), "d e f g".into()],
            max_tokens: 3,
        }
    }

    /// The echo backend passes; each defect fails the rule that names it.
    #[test]
    fn each_defect_fails_its_rule() {
        let good = faulty(false, false);
        let digest = good.model_digest_hex();
        assert_eq!(check_model(&good, &digest, &probe()), Ok(()));

        let broken = check_model(&faulty(true, false), &digest, &probe()).unwrap_err();
        assert!(
            broken.iter().any(|r| r.starts_with("after-another-prompt")),
            "{broken:?}"
        );

        let broken = check_model(&faulty(false, true), &digest, &probe()).unwrap_err();
        assert!(
            broken.iter().any(|r| r.starts_with("isolation")),
            "{broken:?}"
        );

        let broken = check_model(&good, &"0".repeat(64), &probe()).unwrap_err();
        assert!(broken.iter().any(|r| r.starts_with("digest")), "{broken:?}");
    }

    /// The probe's prompts are the model's own template around two questions.
    #[test]
    fn the_chat_probe_uses_the_models_template() {
        let model = faulty(false, false);
        let p = Probe::chat(&model);
        assert!(p.max_tokens > 0);
        assert_ne!(p.prompts[0], p.prompts[1]);
        assert!(
            p.prompts.iter().all(|q| q.contains("<|im_start|>user")),
            "{p:?}"
        );
    }
}
