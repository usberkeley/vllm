// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use crate::incremental::DecodeStream;

mod byte_level_decode;
#[macro_use]
mod error;
mod hf;
mod incremental;
mod tekken;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
mod tiktoken;

pub use error::{Result, TokenizerError};
pub use hf::HuggingFaceTokenizer;
pub use incremental::{DecodedText, IncrementalDecoder, TokenAnchor, TokenAttribution};
pub use tekken::TekkenTokenizer;
pub use tiktoken::TiktokenTokenizer;

pub trait Tokenizer: Send + Sync {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Equivalent to `encode(text, false)`, except that every added,
    /// special, and control-token matcher is bypassed.
    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>>;

    /// Encode a text pair the way cross-encoder scoring models expect, joining
    /// both sequences with the tokenizer's own pair template.
    ///
    /// Backends without a pair template report an error; callers should treat
    /// that as "this model cannot be served by the cross-encoder score path".
    fn encode_pair(
        &self,
        text: &str,
        text_pair: &str,
        add_special_tokens: bool,
    ) -> Result<PairEncoding> {
        let _ = (text, text_pair, add_special_tokens);
        Err(tokenizer_error!(
            "this tokenizer backend does not support text-pair encoding"
        ))
    }

    /// Decode one token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Convert one token string into a token ID, returning `None` if the token
    /// is not in the tokenizer vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Convert one token ID into the tokenizer's raw token string.
    fn id_to_token(&self, id: u32) -> Option<String>;

    /// Borrow tokenizer added-vocabulary entries as `(token, id)` pairs.
    ///
    /// Backends that do not expose the distinction between model vocabulary
    /// and added vocabulary return an empty list.
    // TODO: add support to all tokenizer backends
    fn added_vocab(&self) -> &[(String, u32)] {
        &[]
    }

    /// Return the vocabulary size. Backends that cannot report it fall back to
    /// `usize::MAX`, an effectively unbounded value used only by test stubs.
    fn vocab_size(&self) -> usize {
        usize::MAX
    }

    /// Return whether the given token ID is special.
    fn is_special_id(&self, _token_id: u32) -> bool {
        false
    }

    /// Create a stateful incremental decoder primed with the given prompt
    /// tokens.
    ///
    /// The prompt tokens provide left context for the first generated token;
    /// the decoder does not re-emit prompt text.
    fn create_decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(DecodeStream::new(
            self,
            prompt_token_ids,
            skip_special_tokens,
            min_bytes_to_buffer,
        ))
    }
}

pub type DynTokenizer = Arc<dyn Tokenizer>;

/// Token IDs of one encoded text pair, as cross-encoder models consume them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairEncoding {
    /// Token IDs of the joined pair, including the special tokens inserted by
    /// the tokenizer's own pair template.
    pub token_ids: Vec<u32>,
    /// Index of the first token that belongs to the second sequence.
    ///
    /// Token type IDs of a pair are a run of zeros followed by a run of ones,
    /// so this single boundary carries the same information. It equals
    /// `token_ids.len()` when the tokenizer emits no token type IDs.
    ///
    /// Original Python compression:
    /// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/utils.py#L253-L269>
    pub token_type_boundary: usize,
}

impl PairEncoding {
    /// Build a pair encoding from parallel token IDs and token type IDs.
    ///
    /// Token type IDs must be a run of zeros followed by a run of ones, which
    /// is what every pair post-processor produces.
    pub fn from_token_type_ids(token_ids: Vec<u32>, token_type_ids: &[u32]) -> Result<Self> {
        if token_type_ids.len() != token_ids.len() {
            return Err(tokenizer_error!(
                "token type ids length {} does not match token ids length {}",
                token_type_ids.len(),
                token_ids.len()
            ));
        }
        let boundary = token_type_ids.partition_point(|&type_id| type_id == 0);
        if token_type_ids[boundary..].iter().any(|&type_id| type_id != 1) {
            return Err(tokenizer_error!(
                "token type ids are expected to be a sequence of zeros followed by a \
                 sequence of ones"
            ));
        }
        Ok(Self {
            token_ids,
            token_type_boundary: boundary,
        })
    }
}
