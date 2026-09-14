// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_text::Prompt;

use crate::{ChatMessage, ChatOptions, ChatRequest, ChatRequestProcessor, MmFeatures, Result};

/// Scoring input after media expansion; the boundary indexes the final tokens.
pub struct PreparedScorePrompt {
    pub prompt_token_ids: Vec<u32>,
    pub mm_features: Option<MmFeatures>,
    pub token_type_boundary: Option<usize>,
}

impl ChatRequestProcessor {
    /// Prepare one independent scoring input or a cross-encoder pair.
    pub async fn prepare_score_prompt(
        &self,
        query: crate::ChatContent,
        document: Option<crate::ChatContent>,
        tokenizer: vllm_text::tokenizer::DynTokenizer,
        options: ChatOptions,
        add_special_tokens: bool,
    ) -> Result<PreparedScorePrompt> {
        use crate::renderer::hf::{MultimodalRenderInfo, to_template_string_content};
        use llm_multimodal::Modality;

        let info = self.backend.multimodal_model_info();
        let render_info = info.map(|info| MultimodalRenderInfo {
            image_token: info.placeholder_token(Modality::Image).map(str::to_owned),
            video_token: info.placeholder_token(Modality::Video).map(str::to_owned),
            audio_token: info.placeholder_token(Modality::Audio).map(str::to_owned),
        });
        let (mut tokens, mut boundary) = if let Some(document) = document.as_ref() {
            if options.chat_template.is_some() {
                let rendered =
                    self.backend.chat_renderer().render_score(&query, document, &options)?;
                let tokens = match rendered.prompt {
                    Prompt::Text(text) => tokenizer
                        .encode(&text, add_special_tokens)
                        .map_err(|error| crate::Error::Text(error.into()))?,
                    Prompt::TokenIds(tokens) => tokens,
                };
                (tokens, None)
            } else {
                let query = to_template_string_content(&query, render_info.as_ref())?;
                let document = to_template_string_content(document, render_info.as_ref())?;
                let pair = tokenizer
                    .encode_pair(&query, &document, add_special_tokens)
                    .map_err(|error| crate::Error::Text(error.into()))?;
                (pair.token_ids, Some(pair.token_type_boundary))
            }
        } else {
            let text = to_template_string_content(&query, render_info.as_ref())?;
            (
                tokenizer
                    .encode(&text, add_special_tokens)
                    .map_err(|error| crate::Error::Text(error.into()))?,
                None,
            )
        };
        let mut messages = vec![ChatMessage::user(query)];
        if let Some(document) = document {
            messages.push(ChatMessage::user(document));
        }
        let request = pooling_request(messages, options, add_special_tokens);
        let media = crate::multimodal::extract_media_parts(&request)?;
        let mm_features = if media.is_empty() {
            None
        } else {
            let info = info.ok_or(crate::Error::UnsupportedMultimodalRenderer)?;
            let dtype = self.model_dtype.ok_or(crate::Error::UnsupportedMultimodalRenderer)?;
            Some(
                info.prepare_multimodal_with_boundary(media, &mut tokens, dtype, &mut boundary)
                    .await?,
            )
        };
        Ok(PreparedScorePrompt {
            prompt_token_ids: tokens,
            mm_features,
            token_type_boundary: boundary,
        })
    }

    /// Render a pooling conversation and reuse chat media preprocessing.
    pub async fn prepare_pooling_prompt(
        &self,
        messages: Vec<ChatMessage>,
        chat_options: ChatOptions,
        add_special_tokens: bool,
    ) -> Result<(Prompt, Option<MmFeatures>)> {
        let request = pooling_request(messages, chat_options, add_special_tokens);
        request.validate()?;
        let rendered = self.backend.chat_renderer().render(&request)?;
        self.finalize_rendered_prompt(&request, rendered).await
    }
}

fn pooling_request(
    messages: Vec<ChatMessage>,
    chat_options: ChatOptions,
    add_special_tokens: bool,
) -> ChatRequest {
    ChatRequest {
        request_id: String::new(),
        messages,
        chat_options,
        add_special_tokens,
        sampling_params: Default::default(),
        tool_context: Default::default(),
        decode_options: Default::default(),
        intermediate: false,
        prompt_truncation: None,
        priority: 0,
        documents: None,
        cache_salt: None,
        data_parallel_rank: None,
        session_id: None,
        lora_request: None,
    }
}
