// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::task::PoolingTask;
use crate::protocol::OpaqueValue;

/// Additional model-specific pooling arguments forwarded verbatim to
/// engine-core.
///
/// Original Python field:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/pooling_params.py#L70>
pub type PoolingExtraKwargs = BTreeMap<String, OpaqueValue>;

/// Key under which cross-encoder scoring passes the query/document boundary.
///
/// Engine-core rebuilds the full token type ID vector from this single index.
///
/// Original Python producer:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/io_processor.py#L558>
pub const COMPRESSED_TOKEN_TYPE_IDS_KEY: &str = "compressed_token_type_ids";

/// API parameters for pooling models.
///
/// This is the supported positional prefix of Python's array-like
/// `PoolingParams`. Python fills the omitted internal suffix with its declared
/// defaults.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/pooling_params.py#L38-L73>
///
/// Original Python field documentation:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/config/pooler.py#L51-L115>
#[derive(Debug, Clone, Default, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct EngineCorePoolingParams {
    /// Whether to apply activation function to the pooler outputs.
    /// `None` lets engine-core resolve the model's default.
    pub use_activation: Option<bool>,
    /// Reduce the dimensions of embeddings if model support matryoshka
    /// representation.
    /// `None` lets engine-core resolve the model's default.
    pub dimensions: Option<u32>,
    /// If set, only the score corresponding to the `step_tag_id` in the
    /// generated sentence should be returned. Otherwise, the scores for all
    /// tokens are returned.
    pub step_tag_id: Option<u32>,
    /// A list of indices for the vocabulary dimensions to be extracted,
    /// such as the token IDs of `good_token` and `bad_token` in the
    /// `math-shepherd-mistral-7b-prm` model.
    pub returned_token_ids: Option<Vec<u32>>,
    /// The task used for pooling.
    pub task: PoolingTask,
    /// Whether the pooler needs the prompt token IDs alongside the hidden
    /// states. Internal to Python; kept only to reach the fields after it.
    pub requires_token_ids: bool,
    /// Whether to skip reading the prefix cache for this request.
    /// `None` lets engine-core resolve the task's default.
    pub skip_reading_prefix_cache: Option<bool>,
    /// Worker-side late-interaction metadata. Not yet modeled in Rust, so it
    /// is always sent unset.
    // TODO: type this once late-interaction scoring is supported.
    pub late_interaction_params: Option<OpaqueValue>,
    /// Additional model-specific arguments, such as
    /// [`COMPRESSED_TOKEN_TYPE_IDS_KEY`] for cross-encoder scoring.
    pub extra_kwargs: Option<PoolingExtraKwargs>,
}

impl EngineCorePoolingParams {
    /// Attach the query/document boundary consumed by cross-encoder models.
    pub fn with_compressed_token_type_ids(mut self, boundary: usize) -> Self {
        self.extra_kwargs.get_or_insert_default().insert(
            COMPRESSED_TOKEN_TYPE_IDS_KEY.to_string(),
            OpaqueValue::from(boundary as u64),
        );
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{decode_msgpack, encode_msgpack};

    #[test]
    fn pooling_params_encode_as_a_positional_python_prefix() {
        let params = EngineCorePoolingParams {
            task: PoolingTask::Classify,
            ..Default::default()
        }
        .with_compressed_token_type_ids(3);

        let encoded = encode_msgpack(&params).expect("encode pooling params");
        let wire: Vec<OpaqueValue> = decode_msgpack(&encoded).expect("decode as an array");

        assert_eq!(wire.len(), 9);
        assert_eq!(wire[4].as_str(), Some("classify"));
        assert_eq!(
            wire[8].as_map().expect("extra kwargs map")[0].1.as_u64(),
            Some(3)
        );
        assert_eq!(
            decode_msgpack::<EngineCorePoolingParams>(&encoded).unwrap(),
            params
        );
    }
}
