// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Scoring routes: `/score` and `/rerank`, plus their versioned aliases.
//!
//! Both endpoints share one pipeline. The request's two sides are passed to
//! the shared text facade without expanding a shared query. The facade picks
//! the cross-encoder or bi-encoder path from what the served model's pooler
//! supports. Only the request and response shapes differ, so each endpoint owns
//! just those.
//!
//! Original Python package:
//! <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/>

mod rerank;
mod score;

pub use rerank::rerank;
pub use score::score;
use serde::Deserialize;
use validator::Validate;
use vllm_text::{Prompt, PromptTruncation, ScoreParams, TruncationSide};

use crate::error::{ApiError, bail_invalid_request, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::types::default_true;
use crate::utils::ResolvedRequestContext;

/// Request fields shared by every scoring endpoint.
///
/// Original Python mixin:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/protocol.py>
#[derive(Debug, Clone, Deserialize, Validate)]
pub(crate) struct ScoringParams {
    /// ID of the model to use. An omitted or empty value selects the default.
    pub model: Option<String>,
    /// Whether to apply activation to pooler outputs. `None` uses the pooler's
    /// model-aware default.
    pub use_activation: Option<bool>,
    /// Whether tokenization adds special tokens such as BOS.
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
    /// Number of prompt tokens to retain. `-1` uses the available input
    /// budget; `None` leaves the prompt untruncated.
    pub truncate_prompt_tokens: Option<i64>,
    /// Side from which excess prompt tokens are discarded.
    pub truncation_side: Option<TruncationSide>,
    /// Request ID used throughout inference and returned in the response.
    pub request_id: Option<String>,
    /// Scheduling priority; lower values are handled earlier.
    pub priority: Option<i32>,
    /// Random salt used to isolate prefix-cache entries across users.
    pub cache_salt: Option<String>,
}

/// One or more inputs on either side of a scored pair.
///
/// Python models this as `ScoreInput | list[ScoreInput]`; the token-ID variants
/// are this frontend's pre-tokenized escape hatch and are accepted only by the
/// bi-encoder path.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub(crate) enum ScoreInput {
    TokenIds(Vec<u32>),
    TokenIdBatch(Vec<Vec<u32>>),
    Text(String),
    TextBatch(Vec<String>),
}

impl ScoreInput {
    fn into_prompts(self) -> Vec<Prompt> {
        match self {
            Self::TokenIds(token_ids) => vec![Prompt::TokenIds(token_ids)],
            Self::TokenIdBatch(batch) => batch.into_iter().map(Prompt::TokenIds).collect(),
            Self::Text(text) => vec![Prompt::Text(text)],
            Self::TextBatch(batch) => batch.into_iter().map(Prompt::Text).collect(),
        }
    }
}

/// Pairs lowered from one request, ready to submit.
struct PreparedRequest {
    response_id: String,
    response_model: String,
    request: vllm_text::ScoreRequest<Vec<Prompt>>,
}

/// One scored pair, keyed by its position in the request.
struct PairScore {
    index: usize,
    prompt_token_count: usize,
    score: f32,
}

/// Every pair of one request, scored.
struct ScoredPairs {
    response_id: String,
    response_model: String,
    /// In request order, one entry per submitted pair.
    scores: Vec<PairScore>,
}

impl ScoredPairs {
    /// Total prompt tokens across every pair, which is what both endpoints
    /// report as usage.
    fn prompt_tokens(&self) -> usize {
        self.scores.iter().map(|pair| pair.prompt_token_count).sum()
    }
}

/// Validate both sides without expanding a shared query across documents.
///
/// Original Python validation:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/utils.py#L94-L108>
fn validate_inputs(data_1: &[Prompt], data_2: &[Prompt]) -> Result<(), ApiError> {
    if data_1.is_empty() {
        bail_invalid_request!(param = "data_1", "At least one text element must be given");
    }
    if data_2.is_empty() {
        bail_invalid_request!(
            param = "data_2",
            "At least one text_pair element must be given"
        );
    }
    if data_1.len() > 1 && data_1.len() != data_2.len() {
        bail_invalid_request!("Input lengths must be either 1:1, 1:N or N:N");
    }

    Ok(())
}

/// Validate one scoring request and lower its pairs into text-layer requests.
///
/// Python gives every scoring endpoint the same `score` request-ID prefix, so
/// both response IDs are built the same way here.
fn prepare_pairs(
    common: ScoringParams,
    data_1: Vec<Prompt>,
    data_2: Vec<Prompt>,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<PreparedRequest, ApiError> {
    if let Some(model) = common.model.as_ref().filter(|model| !model.is_empty())
        && !lora_resolution.model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.clone()));
    }

    let prompt_truncation = common
        .truncate_prompt_tokens
        .map(|limit| {
            PromptTruncation::from_wire(
                limit,
                common.truncation_side.unwrap_or(TruncationSide::Right),
            )
        })
        .transpose()
        .map_err(|error| text_submit_error("invalid prompt truncation", error))?;
    validate_inputs(&data_1, &data_2)?;

    let response_id = format!("score-{}", ctx.request_id);
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| lora_resolution.model_names.first().cloned().unwrap_or_default());
    let params = ScoreParams {
        use_activation: common.use_activation,
    };
    let priority = ctx.priority.or(common.priority).unwrap_or(0);
    let request = vllm_text::ScoreRequest {
        request_id: response_id.clone(),
        query: data_1,
        document: data_2,
        params,
        prompt_truncation,
        add_special_tokens: common.add_special_tokens,
        priority,
        cache_salt: common.cache_salt,
        trace_headers: None,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id: ctx.session_id,
        lora_request: lora_resolution.lora_request.clone(),
        arrival_time: None,
    };

    Ok(PreparedRequest {
        response_id,
        response_model,
        request,
    })
}

/// Score every pair concurrently and return them in request order.
async fn run_pairs(
    text: &vllm_text::TextLlm,
    prepared: PreparedRequest,
) -> Result<ScoredPairs, ApiError> {
    let outputs = text
        .score_batch(prepared.request)
        .await
        .map_err(|error| text_submit_error("failed to submit score request", error))?;
    let scores = outputs
        .into_iter()
        .enumerate()
        .map(|(index, output)| PairScore {
            index,
            prompt_token_count: output.prompt_token_ids.len(),
            score: output.score,
        })
        .collect();

    Ok(ScoredPairs {
        response_id: prepared.response_id,
        response_model: prepared.response_model,
        scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(values: &[&str]) -> Vec<Prompt> {
        values.iter().map(|value| Prompt::Text(value.to_string())).collect()
    }

    #[test]
    fn mismatched_sides_are_rejected() {
        let cases = [
            (
                text(&["q1", "q2"]),
                text(&["d1", "d2", "d3"]),
                "Input lengths must be either 1:1, 1:N or N:N",
            ),
            (Vec::new(), text(&["d"]), "At least one text element"),
            (text(&["q"]), Vec::new(), "At least one text_pair element"),
        ];

        for (data_1, data_2, expected) in cases {
            let error = validate_inputs(&data_1, &data_2).unwrap_err();
            assert!(
                error.to_error_response().error.message.contains(expected),
                "expected {expected:?} in {:?}",
                error.to_error_response().error.message
            );
        }
    }
}
