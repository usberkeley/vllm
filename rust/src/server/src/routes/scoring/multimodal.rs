// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_chat::{ChatContent, ChatOptions};
use vllm_llm::{PoolingParams, PoolingTask};
use vllm_text::{Prompt, ScoreMode, TextEncodeRequest};

use super::{PreparedRequest, ScoreItem, ScoringParams, prepare_pairs};
use crate::error::{ApiError, bail_invalid_request, chat_submit_error, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::chat_completions::convert::convert_content;
use crate::routes::openai::utils::types::MessageContent;
use crate::utils::ResolvedRequestContext;

pub(super) struct EncodedScores {
    pub mode: ScoreMode,
    pub n_queries: usize,
    pub requests: Vec<TextEncodeRequest>,
}

impl ScoreItem {
    fn has_media(&self) -> bool {
        matches!(self, Self::Content { content } if content.iter().any(|part| !matches!(part, crate::routes::openai::utils::types::ContentPart::Text { .. })))
    }

    fn content(self) -> Result<ChatContent, ApiError> {
        match self {
            Self::Text(text) => Ok(ChatContent::Text(text)),
            Self::Content { content } => convert_content(MessageContent::Parts(content)),
            Self::TokenIds(_) => bail_invalid_request!(
                "cross-encoder scoring requires text or content inputs, not token IDs"
            ),
        }
    }

    fn text_prompt(&self) -> Prompt {
        match self {
            Self::Text(text) => Prompt::Text(text.clone()),
            Self::TokenIds(ids) => Prompt::TokenIds(ids.clone()),
            // Only used to validate lengths and resolve shared request metadata.
            Self::Content { .. } => Prompt::Text(String::new()),
        }
    }
}

pub(super) async fn prepare(
    mut common: ScoringParams,
    queries: Vec<ScoreItem>,
    documents: Vec<ScoreItem>,
    resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    chat: &vllm_chat::ChatLlm,
) -> Result<PreparedRequest, ApiError> {
    let structured = queries
        .iter()
        .chain(&documents)
        .any(|item| matches!(item, ScoreItem::Content { .. }));
    let templated = common.chat_template.is_some();
    if let Some(instruction) = common.instruction.take() {
        common
            .chat_template_kwargs
            .entry("instruction".into())
            .or_insert(instruction.into());
    }
    if !templated && !common.chat_template_kwargs.is_empty() {
        bail_invalid_request!(
            "chat_template_kwargs and instruction require an explicit scoring chat_template"
        );
    }
    let options = ChatOptions {
        chat_template: common.chat_template.clone(),
        template_kwargs: common.chat_template_kwargs.clone(),
        ..Default::default()
    };
    let mut prepared = prepare_pairs(
        common,
        queries.iter().map(ScoreItem::text_prompt).collect(),
        documents.iter().map(ScoreItem::text_prompt).collect(),
        resolution,
        ctx,
    )?;
    if !structured && !templated {
        return Ok(prepared);
    }
    if prepared.request.prompt_truncation.is_some()
        && queries.iter().chain(&documents).any(ScoreItem::has_media)
    {
        bail_invalid_request!("truncate_prompt_tokens is not supported for multimodal requests");
    }
    let mode = chat
        .text()
        .score_mode()
        .await
        .map_err(|error| text_submit_error("failed to discover scoring task", error))?;
    if templated && mode == ScoreMode::BiEncoder {
        bail_invalid_request!(
            "scoring chat_template applies to cross-encoders; use messages with embeddings for a bi-encoder chat template"
        );
    }
    let n_queries = queries.len();
    let inputs = match mode {
        ScoreMode::BiEncoder => queries
            .into_iter()
            .chain(documents)
            .map(|input| (input, None))
            .collect::<Vec<_>>(),
        ScoreMode::CrossEncoder => documents
            .into_iter()
            .enumerate()
            .map(|(index, document)| {
                (
                    queries[if n_queries == 1 { 0 } else { index }].clone(),
                    Some(document),
                )
            })
            .collect(),
    };
    let mut requests = Vec::with_capacity(inputs.len());
    for (index, (query, document)) in inputs.into_iter().enumerate() {
        let task = match mode {
            ScoreMode::BiEncoder => PoolingTask::Embed,
            ScoreMode::CrossEncoder => PoolingTask::Classify,
        };
        let (prompt, mm_features, boundary) = match (query, document) {
            (ScoreItem::TokenIds(ids), None) => (Prompt::TokenIds(ids), None, None),
            (query, document) => {
                let output = chat
                    .request_processor()
                    .prepare_score_prompt(
                        query.content()?,
                        document.map(ScoreItem::content).transpose()?,
                        chat.text().tokenizer(),
                        options.clone(),
                        prepared.request.add_special_tokens,
                    )
                    .await
                    .map_err(|error| chat_submit_error("failed to prepare scoring media", error))?;
                (
                    Prompt::TokenIds(output.prompt_token_ids),
                    output.mm_features,
                    output.token_type_boundary,
                )
            }
        };
        let suffix = match mode {
            ScoreMode::CrossEncoder => index.to_string(),
            ScoreMode::BiEncoder => format!(
                "{index}-{}",
                if index < n_queries {
                    "query"
                } else {
                    "document"
                }
            ),
        };
        requests.push(TextEncodeRequest {
            request_id: format!("{}-{suffix}", prepared.response_id),
            prompt,
            mm_features,
            task,
            pooling_params: PoolingParams {
                use_activation: prepared.request.params.use_activation,
                compressed_token_type_ids: boundary,
                ..Default::default()
            },
            add_special_tokens: prepared.request.add_special_tokens,
            prompt_truncation: prepared.request.prompt_truncation,
            arrival_time: Some(vllm_llm::current_unix_timestamp_secs()),
            cache_salt: prepared.request.cache_salt.clone(),
            trace_headers: prepared.request.trace_headers.clone(),
            priority: prepared.request.priority,
            data_parallel_rank: prepared.request.data_parallel_rank,
            session_id: prepared.request.session_id.clone(),
            lora_request: prepared.request.lora_request.clone(),
        });
    }
    prepared.encoded = Some(EncodedScores {
        mode,
        n_queries: if mode == ScoreMode::BiEncoder {
            n_queries
        } else {
            0
        },
        requests,
    });
    Ok(prepared)
}
