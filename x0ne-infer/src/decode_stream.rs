//! Token ids to the text a client is streamed, for every backend: whole
//! characters from the tokenizer's `DecodeStream`, then stop sequences held
//! back ([`StopHoldback`]). No model code here.
//!
//! [`StopHoldback`]: crate::backend::StopHoldback

use tokenizers::Tokenizer;

use crate::backend::{Held, NotPrefixStable, StopHoldback};

type TokenStream<'t> = tokenizers::DecodeStream<
    't,
    tokenizers::ModelWrapper,
    tokenizers::NormalizerWrapper,
    tokenizers::PreTokenizerWrapper,
    tokenizers::PostProcessorWrapper,
    tokenizers::DecoderWrapper,
>;

/// Token ids in, final text out: whole characters from `DecodeStream`, then
/// stop sequences held back by [`StopHoldback`].
pub(crate) struct StreamDecoder<'t> {
    stream: TokenStream<'t>,
    holdback: StopHoldback,
}

impl<'t> StreamDecoder<'t> {
    /// `skip_special_tokens` as the final decode has it, or the prefix check fails.
    pub(crate) fn new(tokenizer: &'t Tokenizer, stop: &[String]) -> Self {
        Self {
            stream: tokenizer.decode_stream(true),
            holdback: StopHoldback::new(stop),
        }
    }

    pub(crate) fn push(&mut self, id: u32) -> anyhow::Result<Held> {
        match self.stream.step(id).map_err(decode_error)? {
            Some(piece) => Ok(self.holdback.push(&piece)),
            None => Ok(Held::Emit(String::new())),
        }
    }
}

/// A decoder failure without its message: `InvalidPrefix` quotes decoded text.
fn decode_error(e: tokenizers::Error) -> anyhow::Error {
    if e.downcast_ref::<tokenizers::DecodeStreamError>().is_some() {
        NotPrefixStable.into()
    } else {
        anyhow::anyhow!("decode failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T19.
    /// `InvalidPrefix` quotes decoded text in its message; what leaves the
    /// decoder must not, since it may reach a client or a log.
    #[test]
    fn a_decoder_error_never_carries_generated_text() {
        let witness = "WITNESS-7f3a";
        let raw: tokenizers::Error = Box::new(tokenizers::DecodeStreamError::InvalidPrefix {
            token_id: 7,
            expected_prefix: witness.to_string(),
            actual_string: format!("{witness}?"),
        });
        assert!(raw.to_string().contains(witness), "the raw error quotes it");
        let e = decode_error(raw);
        assert!(e.downcast_ref::<NotPrefixStable>().is_some(), "{e}");
        let shown = format!("{e} {e:#} {e:?}");
        assert!(!shown.contains(witness), "{shown}");
        // The error event a stream ends with, after the 200.
        assert_eq!(
            crate::stream::failure_after_start(&e),
            crate::stream::NOT_PREFIX_STABLE
        );
        let other: tokenizers::Error = format!("near {witness}").into();
        let shown = format!("{:?}", decode_error(other));
        assert!(!shown.contains(witness), "{shown}");
    }
}
