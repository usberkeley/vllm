// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;
use vllm_chat::{ChatOptions, GenerationPromptMode, MmFeatures};
use vllm_text::Prompt;

use crate::error::{ApiError, bail_invalid_request, chat_submit_error};
use crate::routes::openai::chat_completions::convert::{convert_content, convert_message};
use crate::routes::openai::utils::types::{ChatMessage, ContentPart, MessageContent};

/// A raw pooling input; media is lowered above the text layer.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum Input {
    TokenIds(Vec<u32>),
    TokenIdBatch(Vec<Vec<u32>>),
    Text(String),
    TextBatch(Vec<String>),
    Content { content: Vec<ContentPart> },
    Messages(Vec<ChatMessage>),
    ContentBatch(Vec<ContentInput>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum ContentInput {
    Text(String),
    Content { content: Vec<ContentPart> },
}

/// Conversation inputs use the model's chat template without an assistant prefix.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatInputOptions {
    pub messages: Option<Vec<ChatMessage>>,
    pub chat_template: Option<String>,
    #[serde(default)]
    pub chat_template_kwargs: HashMap<String, Value>,
    #[serde(default)]
    pub add_generation_prompt: bool,
}

pub(crate) struct PreparedInput {
    pub prompt: Prompt,
    pub mm_features: Option<MmFeatures>,
}

impl ChatInputOptions {
    pub async fn prepare(
        self,
        input: Option<Input>,
        chat: &vllm_chat::ChatLlm,
        add_special_tokens: bool,
        truncate_prompt_tokens: Option<i64>,
    ) -> Result<Vec<PreparedInput>, ApiError> {
        let inputs = match (input, self.messages) {
            (Some(_), Some(_)) => bail_invalid_request!("provide either input or messages"),
            (None, None) => bail_invalid_request!("input or messages is required"),
            (None, Some(messages)) | (Some(Input::Messages(messages)), None) => {
                vec![Either::Messages(messages)]
            }
            (Some(Input::Text(text)), None) => vec![Either::Prompt(Prompt::Text(text))],
            (Some(Input::TextBatch(batch)), None) => {
                batch.into_iter().map(|s| Either::Prompt(Prompt::Text(s))).collect()
            }
            (Some(Input::TokenIds(ids)), None) => vec![Either::Prompt(Prompt::TokenIds(ids))],
            (Some(Input::TokenIdBatch(batch)), None) => {
                batch.into_iter().map(|ids| Either::Prompt(Prompt::TokenIds(ids))).collect()
            }
            (Some(Input::Content { content }), None) => vec![Either::Content(content)],
            (Some(Input::ContentBatch(batch)), None) => batch
                .into_iter()
                .map(|input| match input {
                    ContentInput::Text(text) => Either::Prompt(Prompt::Text(text)),
                    ContentInput::Content { content } => Either::Content(content),
                })
                .collect(),
        };
        let options = ChatOptions {
            generation_prompt_mode: if self.add_generation_prompt {
                GenerationPromptMode::StartNewAssistant
            } else {
                GenerationPromptMode::NoGenerationPrompt
            },
            chat_template: self.chat_template,
            template_kwargs: self.chat_template_kwargs,
            ..Default::default()
        };
        let mut prepared = Vec::with_capacity(inputs.len());
        for input in inputs {
            let messages = match input {
                Either::Prompt(prompt) => {
                    prepared.push(PreparedInput {
                        prompt,
                        mm_features: None,
                    });
                    continue;
                }
                Either::Messages(messages) => {
                    messages.into_iter().map(convert_message).collect::<Result<Vec<_>, _>>()?
                }
                Either::Content(parts) => vec![vllm_chat::ChatMessage::user(convert_content(
                    MessageContent::Parts(parts),
                )?)],
            };
            if truncate_prompt_tokens.is_some()
                && messages.iter().any(|message| message.has_multimodal())
            {
                bail_invalid_request!(
                    "truncate_prompt_tokens is not supported for multimodal requests"
                );
            }
            let (prompt, mm_features) = chat
                .request_processor()
                .prepare_pooling_prompt(messages, options.clone(), add_special_tokens)
                .await
                .map_err(|error| chat_submit_error("failed to prepare pooling input", error))?;
            prepared.push(PreparedInput {
                prompt,
                mm_features,
            });
        }
        Ok(prepared)
    }
}

enum Either {
    Prompt(Prompt),
    Messages(Vec<ChatMessage>),
    Content(Vec<ContentPart>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_conversations_preserve_roles_instead_of_becoming_a_content_batch() {
        let input: Input = serde_json::from_value(serde_json::json!([
            {"role": "system", "content": [{"type": "text", "text": "embed"}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image.png"}}]}
        ]))
        .unwrap();
        let Input::Messages(messages) = input else {
            panic!("conversation was treated as separate pooling inputs");
        };
        assert!(matches!(messages[0], ChatMessage::System { .. }));
        assert!(matches!(messages[1], ChatMessage::User { .. }));
    }
}
