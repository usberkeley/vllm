// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_text::Prompt;

use crate::{ChatMessage, ChatOptions, ChatRequest, ChatRequestProcessor, MmFeatures, Result};

impl ChatRequestProcessor {
    /// Render a pooling conversation and reuse chat media preprocessing.
    pub async fn prepare_pooling_prompt(
        &self,
        messages: Vec<ChatMessage>,
        chat_options: ChatOptions,
        add_special_tokens: bool,
    ) -> Result<(Prompt, Option<MmFeatures>)> {
        let request = ChatRequest {
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
        };
        request.validate()?;
        let rendered = self.backend.chat_renderer().render(&request)?;
        self.finalize_rendered_prompt(&request, rendered).await
    }
}
