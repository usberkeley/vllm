// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `POST /rerank`, `POST /v1/rerank`, and `POST /v2/rerank` (root paths,
//! matching Python). The response conforms to the JinaAI rerank API.

use std::cmp::Ordering;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{ScoreInput, ScoredPairs, ScoringParams, run_pairs};
use crate::routes::openai::utils::types::Normalizable;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::resolve_request_context;

/// Rerank request accepted by the `/rerank` routes.
///
/// Unlike `/score`, the query is always a single input, so every document is
/// scored against it.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/protocol.py>
#[derive(Debug, Clone, Deserialize, Validate)]
pub(crate) struct RerankRequest {
    #[serde(flatten)]
    pub common: ScoringParams,
    /// The single query every document is scored against.
    pub query: super::ScoreItem,
    /// The documents to rank.
    pub documents: ScoreInput,
    /// How many of the highest-scoring documents to return. `0` returns all.
    #[serde(default)]
    pub top_n: usize,
}

impl Normalizable for RerankRequest {}

/// The document a result refers to, echoed back to the caller.
///
/// `text` is absent for pre-tokenized documents, which have no text to return.
/// Structured content is echoed in `multi_modal`.
#[derive(Debug, Clone, Serialize)]
struct RerankDocument {
    text: Option<String>,
    multi_modal: Option<Vec<crate::routes::openai::utils::types::ContentPart>>,
}

#[derive(Debug, Clone, Serialize)]
struct RerankResult {
    /// Position of this document in the request, preserved through sorting.
    index: usize,
    document: RerankDocument,
    relevance_score: f32,
}

/// Rerank reports only prompt tokens, so it does not reuse the OpenAI usage
/// shape.
#[derive(Debug, Clone, Serialize)]
struct RerankUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RerankResponse {
    id: String,
    model: String,
    usage: RerankUsage,
    results: Vec<RerankResult>,
}

/// Serve one rerank request through the shared scoring pipeline.
pub async fn rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<RerankRequest>,
) -> Response {
    let requested_model = body.common.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model_with_loras(requested_model).await;
    let ctx = resolve_request_context(&headers, body.common.request_id.as_deref());

    let documents = body.documents.into_items();
    let echoed_documents = documents.iter().map(document_echo).collect::<Vec<_>>();
    let top_n = body.top_n;
    let prepared = match super::multimodal::prepare(
        body.common,
        vec![body.query],
        documents,
        &lora_resolution,
        ctx,
        &state.chat,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };

    match run_pairs(state.chat.text(), prepared).await {
        Ok(scored) => Json(build_response(scored, &echoed_documents, top_n)).into_response(),
        Err(error) => error.into_response(),
    }
}

fn document_echo(document: &super::ScoreItem) -> RerankDocument {
    match document {
        super::ScoreItem::Text(text) => RerankDocument {
            text: Some(text.clone()),
            multi_modal: None,
        },
        super::ScoreItem::TokenIds(_) => RerankDocument {
            text: None,
            multi_modal: None,
        },
        super::ScoreItem::Content { content } => RerankDocument {
            text: None,
            multi_modal: Some(content.clone()),
        },
    }
}

/// Rank the scored pairs, keeping the highest `top_n` and reporting usage
/// across every document that was scored.
///
/// Original Python response builder:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/serving.py#L138-L185>
fn build_response(
    scored: ScoredPairs,
    documents: &[RerankDocument],
    top_n: usize,
) -> RerankResponse {
    let usage = RerankUsage {
        prompt_tokens: scored.prompt_tokens(),
        total_tokens: scored.prompt_tokens(),
    };
    let mut results = scored
        .scores
        .iter()
        .map(|pair| RerankResult {
            index: pair.index,
            document: documents[pair.index].clone(),
            relevance_score: pair.score,
        })
        .collect::<Vec<_>>();

    // A stable sort by descending score leaves tied documents in request
    // order, matching Python's `list.sort`.
    results.sort_by(|left, right| {
        right
            .relevance_score
            .partial_cmp(&left.relevance_score)
            .unwrap_or(Ordering::Equal)
    });
    if top_n > 0 && top_n < results.len() {
        results.truncate(top_n);
    }

    RerankResponse {
        id: scored.response_id,
        model: scored.response_model,
        usage,
        results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routes::scoring::PairScore;

    fn scored(scores: &[f32]) -> ScoredPairs {
        ScoredPairs {
            response_id: "score-abc".to_string(),
            response_model: "test-model".to_string(),
            scores: scores
                .iter()
                .enumerate()
                .map(|(index, &score)| PairScore {
                    index,
                    prompt_token_count: 2,
                    score,
                })
                .collect(),
        }
    }

    fn documents(texts: &[&str]) -> Vec<RerankDocument> {
        texts
            .iter()
            .map(|text| document_echo(&super::super::ScoreItem::Text(text.to_string())))
            .collect()
    }

    #[test]
    fn results_are_ranked_by_descending_score_but_keep_request_indices() {
        let response = build_response(scored(&[0.1, 0.9, 0.5]), &documents(&["a", "b", "c"]), 0);

        assert_eq!(
            response
                .results
                .iter()
                .map(|result| (result.index, result.relevance_score))
                .collect::<Vec<_>>(),
            vec![(1, 0.9), (2, 0.5), (0, 0.1)]
        );
        assert_eq!(response.results[0].document.text.as_deref(), Some("b"));
    }

    #[test]
    fn top_n_keeps_only_the_best_documents_but_usage_covers_all() {
        let response = build_response(scored(&[0.1, 0.9, 0.5]), &documents(&["a", "b", "c"]), 2);

        assert_eq!(
            response.results.iter().map(|result| result.index).collect::<Vec<_>>(),
            vec![1, 2]
        );
        // All three documents were scored at 2 prompt tokens each.
        assert_eq!(response.usage.prompt_tokens, 6);
        assert_eq!(response.usage.total_tokens, 6);
    }

    #[test]
    fn a_top_n_of_zero_or_beyond_the_document_count_returns_everything() {
        for top_n in [0, 3, 10] {
            let response = build_response(
                scored(&[0.1, 0.9, 0.5]),
                &documents(&["a", "b", "c"]),
                top_n,
            );
            assert_eq!(response.results.len(), 3, "top_n={top_n}");
        }
    }

    #[test]
    fn tied_scores_keep_their_request_order() {
        let response = build_response(scored(&[0.5, 0.5, 0.9]), &documents(&["a", "b", "c"]), 0);

        assert_eq!(
            response.results.iter().map(|result| result.index).collect::<Vec<_>>(),
            vec![2, 0, 1]
        );
    }

    #[test]
    fn pre_tokenized_documents_echo_no_text() {
        let response = build_response(
            scored(&[0.5]),
            &[document_echo(&super::super::ScoreItem::TokenIds(vec![1]))],
            0,
        );

        assert!(response.results[0].document.text.is_none());
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(
            json["results"][0]["document"]["text"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn a_single_query_and_document_list_deserialize() {
        let request: RerankRequest = serde_json::from_str(
            r#"{"model": "m", "query": "q", "documents": ["d1", "d2"], "top_n": 1}"#,
        )
        .unwrap();

        assert_eq!(
            request.query,
            super::super::ScoreItem::Text("q".to_string())
        );
        assert_eq!(request.documents.clone().into_items().len(), 2);
        assert_eq!(request.top_n, 1);
    }

    #[test]
    fn top_n_defaults_to_returning_everything() {
        let request: RerankRequest =
            serde_json::from_str(r#"{"query": "q", "documents": "d"}"#).unwrap();

        assert_eq!(request.top_n, 0);
    }

    #[test]
    fn a_negative_top_n_is_rejected() {
        let error = serde_json::from_str::<RerankRequest>(
            r#"{"query": "q", "documents": "d", "top_n": -1}"#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("-1"), "{error}");
    }
}
