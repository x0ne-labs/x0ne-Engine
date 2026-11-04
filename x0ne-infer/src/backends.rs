//! The backends a server can be started with, by name: an open list.
//!
//! A backend is registered once, as a [`NamedBackend`]; `--backend <name>`
//! picks it. `core` (x0ne Core, with `cuda`) is the default and the one
//! backend of a release build; `candle` is the CPU reference of the tests
//! (`candle-reference`). Before a server serves with a registered backend,
//! the backend passes [`crate::conformance`].

use crate::backend::InferenceBackend;
#[cfg(feature = "candle-reference")]
use crate::candle_backend::CandleBackend;
use crate::model_spec::ModelArtifacts;

/// The backend of a server started without `--backend`: x0ne Core.
pub const DEFAULT: &str = "core";

/// The name of candle, the CPU reference of the tests.
pub const CANDLE: &str = "candle";

/// One entry of the list.
pub trait NamedBackend: Send + Sync {
    /// The value of `--backend`.
    fn name(&self) -> &'static str;
    /// The backend over a model's metadata files on `device` (`cpu`, `cuda`,
    /// `cuda:N`). Weights then come only through [`InferenceBackend::load`]
    /// or [`InferenceBackend::load_files`], which take the released key.
    fn open(
        &self,
        artifacts: ModelArtifacts,
        device: &str,
    ) -> anyhow::Result<Box<dyn InferenceBackend + Send + Sync>>;
}

/// candle, on the CPU: the reference of the tests.
#[cfg(feature = "candle-reference")]
pub struct Candle;

#[cfg(feature = "candle-reference")]
impl NamedBackend for Candle {
    fn name(&self) -> &'static str {
        CANDLE
    }
    fn open(
        &self,
        artifacts: ModelArtifacts,
        device: &str,
    ) -> anyhow::Result<Box<dyn InferenceBackend + Send + Sync>> {
        Ok(Box::new(candle_on(artifacts, device)?))
    }
}

/// `cpu` only: candle is the reference of the tests; the GPU is x0ne Core's.
#[cfg(feature = "candle-reference")]
fn candle_on(artifacts: ModelArtifacts, device: &str) -> anyhow::Result<CandleBackend> {
    if device == "cpu" {
        return Ok(CandleBackend::cpu(artifacts));
    }
    anyhow::bail!(
        "--device {device:?}: candle is the CPU reference of the tests; \
         the GPU is x0ne Core's (--backend core)"
    )
}

/// The registered backends, `core` first.
pub struct Registry {
    entries: Vec<Box<dyn NamedBackend>>,
}

impl Default for Registry {
    /// What this build links: `core` with `cuda`, `candle` with
    /// `candle-reference` (the tests). A release build without `cuda` has none.
    fn default() -> Self {
        let entries: Vec<Box<dyn NamedBackend>> = vec![
            #[cfg(feature = "cuda")]
            Box::new(crate::core_backend::CoreEntry),
            #[cfg(feature = "candle-reference")]
            Box::new(Candle),
        ];
        Self { entries }
    }
}

impl Registry {
    /// Adds `backend`; a name already taken is refused.
    pub fn with(mut self, backend: Box<dyn NamedBackend>) -> Result<Self, String> {
        if self.get(backend.name()).is_some() {
            return Err(format!("two backends named {:?}", backend.name()));
        }
        self.entries.push(backend);
        Ok(self)
    }
    /// The names, in registration order.
    pub fn names(&self) -> Vec<&'static str> {
        self.entries.iter().map(|b| b.name()).collect()
    }
    /// The backend registered under `name`.
    pub fn get(&self, name: &str) -> Option<&dyn NamedBackend> {
        self.entries
            .iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Other(&'static str);

    impl NamedBackend for Other {
        fn name(&self) -> &'static str {
            self.0
        }
        fn open(
            &self,
            _: ModelArtifacts,
            _: &str,
        ) -> anyhow::Result<Box<dyn InferenceBackend + Send + Sync>> {
            Ok(Box::new(crate::backend::EchoBackend))
        }
    }

    /// core is the default and comes with `cuda`; candle, the tests'
    /// reference, with `candle-reference`; a backend joins by registration
    /// alone; two backends never share a name.
    #[test]
    fn the_list_is_open_and_core_is_the_default() {
        assert_eq!(DEFAULT, "core");
        let r = Registry::default();
        let linked = if cfg!(feature = "cuda") {
            vec!["core", CANDLE]
        } else {
            vec![CANDLE]
        };
        assert_eq!(r.names(), linked);
        assert_eq!(r.get(CANDLE).map(|b| b.name()), Some("candle"));
        let r = r.with(Box::new(Other("other"))).unwrap();
        assert_eq!(r.get("other").map(|b| b.name()), Some("other"));
        assert!(
            r.with(Box::new(Other("candle"))).is_err(),
            "a second candle"
        );
    }

    /// The candle entry opens candle on the CPU, and only there.
    #[test]
    fn the_candle_entry_opens_candle() {
        let (config_json, _, _) = crate::candle_backend::tests::fixture("qwen3");
        let artifacts = || ModelArtifacts {
            config_json: config_json.clone(),
            tokenizer_json: crate::candle_backend::tests::word_tokenizer(40),
        };
        let b = Candle.open(artifacts(), "cpu").unwrap();
        assert_eq!(b.name(), "candle-qwen3");
        assert!(Candle.open(artifacts(), "tpu").is_err());
        for gpu in ["cuda", "cuda:0"] {
            let e = Candle.open(artifacts(), gpu).err().expect(gpu).to_string();
            assert!(e.contains("CPU reference"), "{e}");
        }
    }
}
