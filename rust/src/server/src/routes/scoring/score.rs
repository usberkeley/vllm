// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `POST /score` and `POST /v1/score` (root paths, matching Python).

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{ScoreInput, ScoredPairs, ScoringParams, prepare_pairs, run_pairs};
use crate::routes::openai::utils::types::{Normalizable, Usage};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

/// Score request accepted by `POST /score` and `POST /v1/score`.
///
/// The field aliases cover Python's four `ScoreRequest` variants, which differ
/// only in what they name the two sides of the pair.
///
/// Original Python definitions:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/protocol.py>
#[derive(Debug, Clone, Deserialize, Validate)]
pub(crate) struct ScoreRequest {
    #[serde(flatten)]
    pub common: ScoringParams,
    /// The query side of the pair, or a batch of them.
    #[serde(alias = "text_1", alias = "queries")]
    pub data_1: ScoreInput,
    /// The document side of the pair, or a batch of them.
    #[serde(alias = "text_2", alias = "documents", alias = "items")]
    pub data_2: ScoreInput,
}

impl Normalizable for ScoreRequest {}

#[derive(Debug, Clone, Serialize)]
struct ScoreResponseData {
    index: usize,
    object: &'static str,
    score: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ScoreResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    data: Vec<ScoreResponseData>,
    usage: Usage,
}

/// Serve one score request through the shared scoring pipeline.
pub async fn score(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ScoreRequest>,
) -> Response {
    let requested_model = body.common.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model_with_loras(requested_model).await;
    let ctx = resolve_request_context(&headers, body.common.request_id.as_deref());

    let prepared = match prepare_pairs(
        body.common,
        body.data_1.into_prompts(),
        body.data_2.into_prompts(),
        &lora_resolution,
        ctx,
    ) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };

    match run_pairs(state.chat.text(), prepared).await {
        Ok(scored) => Json(build_response(scored)).into_response(),
        Err(error) => error.into_response(),
    }
}

fn build_response(scored: ScoredPairs) -> ScoreResponse {
    ScoreResponse {
        id: scored.response_id.clone(),
        object: "list",
        created: unix_timestamp(),
        model: scored.response_model.clone(),
        data: scored
            .scores
            .iter()
            .map(|pair| ScoreResponseData {
                index: pair.index,
                object: "score",
                score: pair.score,
            })
            .collect(),
        usage: Usage::from_counts(scored.prompt_tokens(), 0, None, 0),
    }
}

#[cfg(test)]
mod tests {
    use vllm_text::Prompt;

    use super::*;

    #[test]
    fn every_python_score_request_variant_deserializes() {
        let bodies = [
            r#"{"text_1": "q", "text_2": ["d1", "d2"]}"#,
            r#"{"queries": "q", "documents": ["d1", "d2"]}"#,
            r#"{"queries": "q", "items": ["d1", "d2"]}"#,
            r#"{"data_1": "q", "data_2": ["d1", "d2"]}"#,
        ];

        for body in bodies {
            let request: ScoreRequest =
                serde_json::from_str(body).unwrap_or_else(|error| panic!("{body}: {error}"));
            assert_eq!(
                request.data_1.into_prompts(),
                vec![Prompt::Text("q".into())]
            );
            assert_eq!(
                request.data_2.into_prompts(),
                vec![Prompt::Text("d1".into()), Prompt::Text("d2".into())],
                "{body}"
            );
        }
    }

    #[test]
    fn shared_scoring_params_are_flattened_into_the_request() {
        let request: ScoreRequest = serde_json::from_str(
            r#"{"model": "m", "text_1": "q", "text_2": "d",
                "use_activation": false, "add_special_tokens": false,
                "truncate_prompt_tokens": -1, "priority": 3}"#,
        )
        .unwrap();

        assert_eq!(request.common.model.as_deref(), Some("m"));
        assert_eq!(request.common.use_activation, Some(false));
        assert!(!request.common.add_special_tokens);
        assert_eq!(request.common.truncate_prompt_tokens, Some(-1));
        assert_eq!(request.common.priority, Some(3));
    }

    #[test]
    fn add_special_tokens_defaults_to_true() {
        let request: ScoreRequest =
            serde_json::from_str(r#"{"text_1": "q", "text_2": "d"}"#).unwrap();

        assert!(request.common.add_special_tokens);
    }

    #[test]
    fn token_id_prompts_are_accepted_on_both_sides() {
        let request: ScoreRequest =
            serde_json::from_str(r#"{"text_1": [1, 2], "text_2": [[3], [4]]}"#).unwrap();

        assert_eq!(
            (request.data_1.into_prompts(), request.data_2.into_prompts()),
            (
                vec![Prompt::TokenIds(vec![1, 2])],
                vec![Prompt::TokenIds(vec![3]), Prompt::TokenIds(vec![4])]
            )
        );
    }
}
