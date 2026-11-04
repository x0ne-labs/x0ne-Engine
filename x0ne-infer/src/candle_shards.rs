//! candle's view of a checked set of weight files ([`crate::shards::Routing`]):
//! a `SimpleBackend` that reads each tensor from the one file the routing
//! names.

use candle_core::safetensors::SliceSafetensors;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;

use crate::shards::Routing;

/// The weight files of one model, parsed in place, behind candle's
/// `SimpleBackend`: each tensor read from the one file the routing names.
pub struct ShardedSlices<'a> {
    files: Vec<SliceSafetensors<'a>>,
    routing: Routing,
}

impl<'a> ShardedSlices<'a> {
    /// `files`: `(name, bytes)` of every weight file, in any order.
    pub fn new(index_json: Option<&[u8]>, files: &[(&str, &'a [u8])]) -> anyhow::Result<Self> {
        let routing = Routing::check(index_json, files)?;
        let files = files
            .iter()
            .map(|(name, bytes)| {
                SliceSafetensors::new(bytes).map_err(|e| anyhow::anyhow!("{name}: {e}"))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self { files, routing })
    }

    pub fn len(&self) -> usize {
        self.routing.len()
    }

    /// Every tensor name, in no order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.routing.names()
    }

    /// The file that holds `name`, if `name` is stored as BF16, F16 or F32.
    /// An integer or FP8 tensor is quantized storage this engine does not
    /// dequantize: cast to a float, it would serve another model.
    fn file(&self, name: &str) -> candle_core::Result<&SliceSafetensors<'a>> {
        let f = self
            .routing
            .file_of(name)
            .map(|i| &self.files[i])
            .ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.into() }.bt())?;
        let stored = f.get(name)?.dtype();
        match DType::try_from(stored) {
            Ok(DType::BF16 | DType::F16 | DType::F32) => Ok(f),
            _ => Err(candle_core::Error::Msg(format!(
                "{name}: stored as {stored:?}; only BF16, F16 and F32 weights are read"
            ))
            .bt()),
        }
    }
}

impl SimpleBackend for ShardedSlices<'_> {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        SimpleBackend::get(self.file(name)?, s, name, h, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        SimpleBackend::get_unchecked(self.file(name)?, name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.routing.file_of(name).is_some()
    }
}
