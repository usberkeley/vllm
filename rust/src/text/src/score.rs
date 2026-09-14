// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Query/document scoring on top of the pooling facade.
//!
//! Two model families answer the same request shape. Cross-encoders read the
//! pair as one prompt and expose a classifier score directly; bi-encoders embed
//! each side and the frontend takes the cosine similarity. The served model's
//! pooling tasks decide which one runs.

use std::collections::BTreeMap;

use futures::{StreamExt as _, TryStreamExt as _, stream};
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_llm::{
    EncodeOutput, EncodeRequest, EngineTask, PoolingParams as LlmPoolingParams, PoolingTask,
};

use crate::error::{Error, Result};
use crate::{Prompt, PromptTruncation, TextLlm, TextRequestProcessor, TruncationSide};

/// Denominator floor used by cosine similarity, matching `torch`'s default so
/// zero-magnitude embeddings score `0.0` instead of `NaN`.
const COSINE_SIMILARITY_EPS: f32 = 1e-8;

#[derive(Debug, thiserror::Error)]
pub enum ScoreError {
    #[error("Input lengths must be nonempty and either 1:1, 1:N or N:N")]
    InvalidInputLengths,
    #[error(
        "model `{model_name}` supports no scoring task; \
         scoring needs `classify` (cross-encoder) or `embed` (bi-encoder), \
         but the engine reports {supported_tasks:?}"
    )]
    UnsupportedModel {
        model_name: String,
        supported_tasks: Vec<EngineTask>,
    },
    #[error(
        "cross-encoder scoring joins both sides with the tokenizer's pair \
         template, so `{parameter}` must be text rather than token IDs"
    )]
    TokenIdsUnsupported { parameter: &'static str },
    #[error("score request `{request_id}` expected a single pooled score, got shape {shape:?}")]
    OutputShape {
        request_id: String,
        shape: Vec<usize>,
    },
}

impl ScoreError {
    pub(crate) fn is_request_validation_error(&self) -> bool {
        !matches!(self, Self::OutputShape { .. })
    }
}

/// How the served model turns one query/document pair into a score.
///
/// Original Python mapping:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/tasks.py#L34-L38>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreMode {
    /// Embed both sides separately and score their cosine similarity.
    BiEncoder,
    /// Encode the pair as one prompt and read the classifier output.
    CrossEncoder,
}

impl ScoreMode {
    /// Pick the scoring mode from the pooling tasks engine-core reports.
    ///
    /// This mirrors the priority Python falls back to when the pooler config
    /// pins no task. In practice a pooler exposes either the embedding or the
    /// classification family and never both, so the order rarely decides.
    ///
    /// Original Python priority:
    /// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/config/model.py#L1792-L1800>
    fn from_supported_tasks(supported_tasks: &[EngineTask]) -> Option<Self> {
        [
            (PoolingTask::Embed, Self::BiEncoder),
            (PoolingTask::Classify, Self::CrossEncoder),
        ]
        .into_iter()
        .find(|(task, _)| supported_tasks.contains(&EngineTask::Pooling(*task)))
        .map(|(_, mode)| mode)
    }
}

/// User-facing parameters for one scoring operation.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/protocol.py>
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoreParams {
    /// Whether to apply activation function to the pooler outputs.
    /// `None` uses the pooler's default, which is `true` in most cases.
    pub use_activation: Option<bool>,
}

/// One query/document pair scored by [`TextLlm::score`].
/// Use `ScoreRequest<Vec<Prompt>>` with [`TextLlm::score_batch`] for 1:N or N:N
/// scoring without duplicating a shared query's encoding.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreRequest<T = Prompt> {
    /// Request ID used throughout inference and returned in the response.
    pub request_id: String,
    /// The query side of the pair.
    pub query: T,
    /// The document side of the pair.
    pub document: T,
    /// Scoring parameters forwarded to engine-core for model-aware resolution.
    pub params: ScoreParams,
    /// Optional typed prompt-truncation policy, applied to the joined pair for
    /// cross-encoders and to each side independently for bi-encoders.
    pub prompt_truncation: Option<PromptTruncation>,
    /// Whether to add special tokens, such as BOS, during tokenization.
    pub add_special_tokens: bool,
    /// Request scheduling priority. Lower values are handled earlier.
    pub priority: i32,
    /// Random salt used to isolate prefix-cache entries across users.
    pub cache_salt: Option<String>,
    /// Optional tracing headers forwarded to engine-core.
    pub trace_headers: Option<BTreeMap<String, String>>,
    /// Override data-parallel rank routing.
    pub data_parallel_rank: Option<u32>,
    /// Stable session identity shared by related requests.
    pub session_id: Option<String>,
    /// LoRA adapter selected for this request.
    pub lora_request: Option<LoraRequest>,
    /// Wall-clock unix timestamp when this request arrived at the frontend.
    pub arrival_time: Option<f64>,
}

impl ScoreRequest {
    /// Return one minimal request fixture for tests.
    pub fn for_test() -> Self {
        Self {
            request_id: "test-score".to_string(),
            query: Prompt::Text("query".to_string()),
            document: Prompt::Text("document".to_string()),
            params: ScoreParams::default(),
            prompt_truncation: None,
            add_special_tokens: true,
            priority: 0,
            cache_salt: None,
            trace_headers: None,
            data_parallel_rank: None,
            session_id: None,
            lora_request: None,
            arrival_time: None,
        }
    }

    fn validate(&self) -> Result<()> {
        for (parameter, prompt) in [("query", &self.query), ("document", &self.document)] {
            if matches!(prompt, Prompt::TokenIds(token_ids) if token_ids.is_empty()) {
                return Err(Error::EmptyPromptTokenIds {
                    request_id: format!("{}-{parameter}", self.request_id),
                });
            }
        }
        Ok(())
    }
}

impl<T> ScoreRequest<T> {
    /// Split off the fields shared by every request this pair lowers into.
    fn split(self) -> (String, T, T, ScoreRequestCommon) {
        (
            self.request_id,
            self.query,
            self.document,
            ScoreRequestCommon {
                params: self.params,
                prompt_truncation: self.prompt_truncation,
                add_special_tokens: self.add_special_tokens,
                priority: self.priority,
                cache_salt: self.cache_salt,
                trace_headers: self.trace_headers,
                data_parallel_rank: self.data_parallel_rank,
                session_id: self.session_id,
                lora_request: self.lora_request,
                arrival_time: self.arrival_time,
            },
        )
    }
}

/// Request fields copied onto each pooling request a score request lowers into.
#[derive(Debug, Clone)]
struct ScoreRequestCommon {
    params: ScoreParams,
    prompt_truncation: Option<PromptTruncation>,
    add_special_tokens: bool,
    priority: i32,
    cache_salt: Option<String>,
    trace_headers: Option<BTreeMap<String, String>>,
    data_parallel_rank: Option<u32>,
    session_id: Option<String>,
    lora_request: Option<LoraRequest>,
    arrival_time: Option<f64>,
}

impl ScoreRequestCommon {
    fn into_encode_request(
        self,
        request_id: String,
        prompt_token_ids: Vec<u32>,
        task: PoolingTask,
        compressed_token_type_ids: Option<usize>,
    ) -> EncodeRequest {
        EncodeRequest {
            request_id,
            prompt_token_ids,
            mm_features: None,
            task,
            pooling_params: LlmPoolingParams {
                use_activation: self.params.use_activation,
                // Matryoshka truncation is an embedding-only control and would
                // change what a score means, so it is never forwarded here.
                dimensions: None,
                step_tag_id: None,
                returned_token_ids: None,
                compressed_token_type_ids,
            },
            arrival_time: self.arrival_time,
            cache_salt: self.cache_salt,
            trace_headers: self.trace_headers,
            priority: self.priority,
            data_parallel_rank: self.data_parallel_rank,
            session_id: self.session_id,
            lora_request: self.lora_request,
        }
    }
}

/// Final score for one query/document pair.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreOutput {
    /// Stable caller-supplied request ID.
    pub request_id: String,
    /// Token IDs the model saw. Bi-encoders report both sides joined by the
    /// padding token, mirroring how Python accounts for pair prompt tokens.
    pub prompt_token_ids: Vec<u32>,
    /// The similarity score for the pair.
    pub score: f32,
    /// Number of prompt tokens served from cache.
    pub cached_token_count: usize,
}

/// Read the single scalar a scoring pooler emits.
///
/// Original Python conversion:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/outputs.py#L363-L372>
fn scalar_score(request_id: &str, output: &EncodeOutput) -> Result<f32> {
    match output.output.data.as_slice() {
        [score] => Ok(*score),
        _ => Err(ScoreError::OutputShape {
            request_id: request_id.to_string(),
            shape: output.output.shape.clone(),
        }
        .into()),
    }
}

/// Cosine similarity of two equally sized embeddings.
fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let magnitude = |values: &[f32]| values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let denominator = (magnitude(left) * magnitude(right)).max(COSINE_SIMILARITY_EPS);
    let dot: f32 = left.iter().zip(right).map(|(left, right)| left * right).sum();
    dot / denominator
}

impl TextRequestProcessor {
    /// Tokenize the pair as one prompt and lower it for a cross-encoder.
    ///
    /// The tokenizer's own pair template inserts the separators the model was
    /// trained with, and the resulting query/document boundary travels to
    /// engine-core so it can rebuild the token type IDs.
    pub fn prepare_cross_encoder_score(&self, request: ScoreRequest) -> Result<EncodeRequest> {
        request.validate()?;
        let (request_id, query, document, mut common) = request.split();

        let text = |prompt: Prompt, parameter: &'static str| match prompt {
            Prompt::Text(text) => Ok(text),
            Prompt::TokenIds(_) => Err(ScoreError::TokenIdsUnsupported { parameter }),
        };
        let query = text(query, "query")?;
        let document = text(document, "document")?;

        let mut encoded =
            self.backend
                .tokenizer()
                .encode_pair(&query, &document, common.add_special_tokens)?;
        if let Some(prompt_truncation) = common.prompt_truncation.take() {
            let original_len = encoded.token_ids.len();
            prompt_truncation.apply(&mut encoded.token_ids, self.max_model_len)?;
            encoded.token_type_boundary = truncated_boundary(
                encoded.token_type_boundary,
                original_len,
                encoded.token_ids.len(),
                prompt_truncation.side,
            );
        }
        self.validate_prompt_tokens(&request_id, &encoded.token_ids)?;

        Ok(common.into_encode_request(
            request_id,
            encoded.token_ids,
            PoolingTask::Classify,
            Some(encoded.token_type_boundary),
        ))
    }

    /// Tokenize each side separately and lower both for a bi-encoder.
    pub fn prepare_bi_encoder_score(&self, request: ScoreRequest) -> Result<[EncodeRequest; 2]> {
        request.validate()?;
        let (request_id, query, document, common) = request.split();

        let prepare = |parameter: &str, prompt: Prompt| -> Result<EncodeRequest> {
            let side_request_id = format!("{request_id}-{parameter}");
            let prompt_token_ids = self.prepare_prompt_tokens(
                prompt,
                common.add_special_tokens,
                common.prompt_truncation,
                None,
            )?;
            self.validate_prompt_tokens(&side_request_id, &prompt_token_ids)?;
            Ok(common.clone().into_encode_request(
                side_request_id,
                prompt_token_ids,
                PoolingTask::Embed,
                None,
            ))
        };

        Ok([prepare("query", query)?, prepare("document", document)?])
    }
}

/// Move a query/document boundary through the same truncation applied to the
/// prompt tokens.
///
/// Original Python equivalent:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/scoring/io_processor.py#L53-L84>
pub(crate) fn truncated_boundary(
    boundary: usize,
    original_len: usize,
    truncated_len: usize,
    side: TruncationSide,
) -> usize {
    match side {
        TruncationSide::Right => boundary.min(truncated_len),
        TruncationSide::Left => boundary.saturating_sub(original_len - truncated_len),
    }
}

impl TextLlm {
    /// Score preprocessed pooling inputs, reusing each bi-encoder query once.
    ///
    /// Bi-encoder inputs are ordered as queries followed by documents;
    /// cross-encoder inputs are already paired and use `n_queries = 0`.
    pub async fn score_encoded_batch(
        &self,
        request_id: String,
        mode: ScoreMode,
        n_queries: usize,
        requests: Vec<crate::TextEncodeRequest>,
    ) -> Result<Vec<ScoreOutput>> {
        let valid = match mode {
            ScoreMode::BiEncoder => {
                n_queries > 0
                    && requests.len() > n_queries
                    && (n_queries == 1 || requests.len() - n_queries == n_queries)
            }
            ScoreMode::CrossEncoder => n_queries == 0 && !requests.is_empty(),
        };
        if !valid {
            return Err(ScoreError::InvalidInputLengths.into());
        }
        let outputs = self.encode_batch(requests).await?;
        match mode {
            ScoreMode::BiEncoder => Ok(outputs[n_queries..]
                .iter()
                .enumerate()
                .map(|(index, document)| {
                    self.bi_encoder_output(
                        format!("{request_id}-{index}"),
                        &outputs[if n_queries == 1 { 0 } else { index }],
                        document,
                    )
                })
                .collect()),
            ScoreMode::CrossEncoder => outputs
                .into_iter()
                .enumerate()
                .map(|(index, output)| {
                    let id = format!("{request_id}-{index}");
                    Ok(ScoreOutput {
                        score: scalar_score(&id, &output)?,
                        request_id: id,
                        prompt_token_ids: output.prompt_token_ids,
                        cached_token_count: output.cached_token_count,
                    })
                })
                .collect(),
        }
    }

    /// Score a batch, encoding a shared bi-encoder query only once.
    /// Results retain document order and account for prompt tokens per pair.
    /// Both sides must be nonempty, with one query or equal input counts.
    pub async fn score_batch(
        &self,
        request: ScoreRequest<Vec<Prompt>>,
    ) -> Result<Vec<ScoreOutput>> {
        const MAX_CONCURRENT_REQUESTS: usize = 32;
        let (request_id, queries, documents, common) = request.split();
        if queries.is_empty()
            || documents.is_empty()
            || (queries.len() != 1 && queries.len() != documents.len())
        {
            return Err(ScoreError::InvalidInputLengths.into());
        }
        let mode = self.score_mode().await?;
        let n_queries = queries.len();
        if mode == ScoreMode::BiEncoder {
            let requests = queries
                .into_iter()
                .map(|prompt| ("query", prompt))
                .chain(documents.into_iter().map(|prompt| ("document", prompt)))
                .enumerate()
                .map(|(index, (side, prompt))| {
                    let id = format!("{request_id}-{index}-{side}");
                    let tokens = self.processor.prepare_prompt_tokens(
                        prompt,
                        common.add_special_tokens,
                        common.prompt_truncation,
                        None,
                    )?;
                    self.processor.validate_prompt_tokens(&id, &tokens)?;
                    Ok(common.clone().into_encode_request(id, tokens, PoolingTask::Embed, None))
                })
                .collect::<Result<Vec<_>>>()?;
            let outputs = stream::iter(requests)
                .map(|request| self.llm.encode(request))
                .buffered(MAX_CONCURRENT_REQUESTS)
                .try_collect::<Vec<_>>()
                .await?;
            return Ok(outputs[n_queries..]
                .iter()
                .enumerate()
                .map(|(index, document)| {
                    let query = &outputs[if n_queries == 1 { 0 } else { index }];
                    self.bi_encoder_output(format!("{request_id}-{index}"), query, document)
                })
                .collect());
        }
        stream::iter(documents.into_iter().enumerate())
            .map(|(index, document)| {
                let query = queries[if n_queries == 1 { 0 } else { index }].clone();
                let common = common.clone();
                let request_id = format!("{request_id}-{index}");
                async move {
                    self.cross_encoder_score(ScoreRequest {
                        request_id,
                        query,
                        document,
                        params: common.params,
                        prompt_truncation: common.prompt_truncation,
                        add_special_tokens: common.add_special_tokens,
                        priority: common.priority,
                        cache_salt: common.cache_salt,
                        trace_headers: common.trace_headers,
                        data_parallel_rank: common.data_parallel_rank,
                        session_id: common.session_id,
                        lora_request: common.lora_request,
                        arrival_time: common.arrival_time,
                    })
                    .await
                }
            })
            .buffered(MAX_CONCURRENT_REQUESTS)
            .try_collect()
            .await
    }

    /// Return how this model scores pairs, discovered from engine-core.
    pub async fn score_mode(&self) -> Result<ScoreMode> {
        let supported_tasks = self.engine_core_client().get_supported_tasks().await?;
        ScoreMode::from_supported_tasks(supported_tasks).ok_or_else(|| {
            ScoreError::UnsupportedModel {
                model_name: self.model_id().to_string(),
                supported_tasks: supported_tasks.to_vec(),
            }
            .into()
        })
    }

    /// Score one query/document pair through the model's scoring pooler.
    pub async fn score(&self, request: ScoreRequest) -> Result<ScoreOutput> {
        match self.score_mode().await? {
            ScoreMode::CrossEncoder => self.cross_encoder_score(request).await,
            ScoreMode::BiEncoder => self.bi_encoder_score(request).await,
        }
    }

    async fn cross_encoder_score(&self, request: ScoreRequest) -> Result<ScoreOutput> {
        let encode_request = self.processor.prepare_cross_encoder_score(request)?;
        let request_id = encode_request.request_id.clone();
        let output = self.llm.encode(encode_request).await?;

        Ok(ScoreOutput {
            score: scalar_score(&request_id, &output)?,
            request_id,
            prompt_token_ids: output.prompt_token_ids,
            cached_token_count: output.cached_token_count,
        })
    }

    async fn bi_encoder_score(&self, request: ScoreRequest) -> Result<ScoreOutput> {
        let request_id = request.request_id.clone();
        let [query, document] = self.processor.prepare_bi_encoder_score(request)?;
        let (query, document) =
            futures::try_join!(self.llm.encode(query), self.llm.encode(document))?;

        Ok(self.bi_encoder_output(request_id, &query, &document))
    }

    fn bi_encoder_output(
        &self,
        request_id: String,
        query: &EncodeOutput,
        document: &EncodeOutput,
    ) -> ScoreOutput {
        let mut prompt_token_ids = query.prompt_token_ids.clone();
        prompt_token_ids.extend(self.processor.backend.pad_token_id());
        prompt_token_ids.extend_from_slice(&document.prompt_token_ids);

        ScoreOutput {
            request_id,
            prompt_token_ids,
            score: cosine_similarity(&query.output.data, &document.output.data),
            cached_token_count: query.cached_token_count + document.cached_token_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use expect_test::expect;
    use vllm_llm::{GenerationTask, PoolingOutput};
    use vllm_tokenizer::DynTokenizer;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;
    use crate::{PromptTruncationLimit, TextBackend};

    struct FakeScoreBackend {
        tokenizer: DynTokenizer,
    }

    impl TextBackend for FakeScoreBackend {
        fn tokenizer(&self) -> DynTokenizer {
            self.tokenizer.clone()
        }

        fn model_id(&self) -> &str {
            "test-score-model"
        }

        fn model_vocab_size(&self) -> usize {
            512
        }

        fn tokenizer_vocab_size(&self) -> usize {
            512
        }

        fn pad_token_id(&self) -> Option<u32> {
            Some(258)
        }
    }

    fn processor(max_model_len: u32) -> TextRequestProcessor {
        let tokenizer = TestTokenizer::new()
            .with_bos_token("<cls>", 256)
            .with_pair_separator_token("<sep>", 257)
            .with_regular_token("<pad>", 258)
            .with_vocab_size(512);
        TextRequestProcessor::new(
            Arc::new(FakeScoreBackend {
                tokenizer: Arc::new(tokenizer),
            }),
            max_model_len,
        )
    }

    fn request(query: Prompt, document: Prompt) -> ScoreRequest {
        ScoreRequest {
            query,
            document,
            arrival_time: Some(42.5),
            ..ScoreRequest::for_test()
        }
    }

    fn encode_output(data: Vec<f32>) -> EncodeOutput {
        EncodeOutput {
            request_id: "internal-id".to_string(),
            prompt_token_ids: vec![1, 2],
            output: PoolingOutput {
                shape: vec![data.len()],
                data,
            },
            cached_token_count: 0,
        }
    }

    #[test]
    fn score_mode_follows_the_engine_reported_pooling_tasks() {
        let mode = |tasks: &[EngineTask]| ScoreMode::from_supported_tasks(tasks);

        assert_eq!(
            mode(&[
                PoolingTask::Classify.into(),
                PoolingTask::TokenClassify.into()
            ]),
            Some(ScoreMode::CrossEncoder)
        );
        assert_eq!(
            mode(&[PoolingTask::TokenEmbed.into(), PoolingTask::Embed.into()]),
            Some(ScoreMode::BiEncoder)
        );
        assert_eq!(mode(&[GenerationTask::Generate.into()]), None);
    }

    #[test]
    fn cross_encoder_preparation_joins_the_pair_and_marks_its_boundary() {
        let prepared = processor(16)
            .prepare_cross_encoder_score(request(
                Prompt::Text("ab".to_string()),
                Prompt::Text("c".to_string()),
            ))
            .unwrap();

        expect![[r#"
            EncodeRequest {
                request_id: "test-score",
                prompt_token_ids: [
                    256,
                    97,
                    98,
                    257,
                    99,
                    257,
                ],
                mm_features: None,
                task: Classify,
                pooling_params: PoolingParams {
                    use_activation: None,
                    dimensions: None,
                    step_tag_id: None,
                    returned_token_ids: None,
                    compressed_token_type_ids: Some(
                        4,
                    ),
                },
                arrival_time: Some(
                    42.5,
                ),
                cache_salt: None,
                trace_headers: None,
                priority: 0,
                data_parallel_rank: None,
                session_id: None,
                lora_request: None,
            }
        "#]]
        .assert_debug_eq(&prepared);
    }

    #[test]
    fn cross_encoder_preparation_forwards_use_activation() {
        let mut score_request =
            request(Prompt::Text("a".to_string()), Prompt::Text("b".to_string()));
        score_request.params.use_activation = Some(false);

        let prepared = processor(16).prepare_cross_encoder_score(score_request).unwrap();

        assert_eq!(prepared.pooling_params.use_activation, Some(false));
    }

    #[test]
    fn cross_encoder_preparation_rejects_token_id_prompts() {
        let error = processor(16)
            .prepare_cross_encoder_score(request(
                Prompt::TokenIds(vec![1, 2]),
                Prompt::Text("b".to_string()),
            ))
            .unwrap_err();

        assert!(matches!(
            error,
            Error::Score(ScoreError::TokenIdsUnsupported { parameter: "query" })
        ));
    }

    #[test]
    fn truncating_the_joined_pair_moves_the_boundary_with_it() {
        // `<cls> a b <sep> c d <sep>`, so the document starts at index 4.
        let boundary = 4;
        let cases = [
            (TruncationSide::Right, 3, 3),
            (TruncationSide::Right, 6, 4),
            (TruncationSide::Left, 3, 0),
            (TruncationSide::Left, 6, 2),
        ];

        for (side, truncated_len, expected) in cases {
            assert_eq!(
                truncated_boundary(boundary, 8, truncated_len, side),
                expected,
                "side={side:?} truncated_len={truncated_len}"
            );
        }
    }

    #[test]
    fn cross_encoder_preparation_applies_truncation_to_prompt_and_boundary() {
        let mut score_request = request(
            Prompt::Text("ab".to_string()),
            Prompt::Text("cd".to_string()),
        );
        score_request.prompt_truncation = Some(PromptTruncation {
            limit: PromptTruncationLimit::Fixed(3),
            side: TruncationSide::Right,
        });

        let prepared = processor(16).prepare_cross_encoder_score(score_request).unwrap();

        assert_eq!(prepared.prompt_token_ids, vec![256, 97, 98]);
        assert_eq!(prepared.pooling_params.compressed_token_type_ids, Some(3));
    }

    #[test]
    fn cross_encoder_preparation_enforces_the_context_window() {
        let error = processor(4)
            .prepare_cross_encoder_score(request(
                Prompt::Text("abc".to_string()),
                Prompt::Text("def".to_string()),
            ))
            .unwrap_err();

        expect![[
            "this model's maximum context length is 4 tokens, but the prompt contains 9 input tokens"
        ]]
        .assert_eq(&error.to_string());
    }

    #[test]
    fn bi_encoder_preparation_lowers_each_side_independently() {
        let prepared = processor(16)
            .prepare_bi_encoder_score(request(
                Prompt::Text("ab".to_string()),
                Prompt::TokenIds(vec![7, 8]),
            ))
            .unwrap();

        let summary = prepared
            .iter()
            .map(|request| {
                (
                    request.request_id.clone(),
                    request.task,
                    request.prompt_token_ids.clone(),
                    request.pooling_params.compressed_token_type_ids,
                )
            })
            .collect::<Vec<_>>();

        expect![[r#"
            [
                (
                    "test-score-query",
                    Embed,
                    [
                        256,
                        97,
                        98,
                    ],
                    None,
                ),
                (
                    "test-score-document",
                    Embed,
                    [
                        7,
                        8,
                    ],
                    None,
                ),
            ]
        "#]]
        .assert_debug_eq(&summary);
    }

    #[test]
    fn bi_encoder_preparation_rejects_out_of_vocabulary_prompt_ids() {
        let error = processor(16)
            .prepare_bi_encoder_score(request(
                Prompt::Text("a".to_string()),
                Prompt::TokenIds(vec![512]),
            ))
            .unwrap_err();

        assert!(matches!(
            error,
            Error::TokenIds(crate::TokenIdsError::OutOfVocab {
                parameter: "prompt",
                token_ids,
                vocab_size: 512,
            }) if token_ids == vec![512]
        ));
    }

    #[test]
    fn empty_token_id_prompts_are_rejected_per_side() {
        let error = processor(16)
            .prepare_bi_encoder_score(request(
                Prompt::Text("a".to_string()),
                Prompt::TokenIds(Vec::new()),
            ))
            .unwrap_err();

        assert!(matches!(
            error,
            Error::EmptyPromptTokenIds { request_id } if request_id == "test-score-document"
        ));
    }

    #[test]
    fn a_scoring_pooler_must_emit_exactly_one_value() {
        assert_eq!(
            scalar_score("external-id", &encode_output(vec![0.75])).unwrap(),
            0.75
        );

        let error = scalar_score("external-id", &encode_output(vec![0.25, -0.5])).unwrap_err();
        assert!(matches!(
            error,
            Error::Score(ScoreError::OutputShape { request_id, shape })
                if request_id == "external-id" && shape == vec![2]
        ));
    }

    #[test]
    fn cosine_similarity_matches_torch_including_the_zero_vector_case() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert!((cosine_similarity(&[1.0, 1.0], &[1.0, 0.0]) - 0.5_f32.sqrt()).abs() < 1e-6);
    }
}
