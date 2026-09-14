// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt::Write as _;

use tracing::{Level, enabled, info};
use vllm_chat::{AssistantContentBlock, AssistantMessage};

use super::types::ChatMessageDelta;

pub(super) fn log_message(
    request_id: &str,
    message: &AssistantMessage,
    include_reasoning: bool,
    finish_reason: &str,
    streaming: bool,
) {
    if !enabled!(Level::INFO) {
        return;
    }
    let mut output = String::new();
    for block in &message.content {
        match block {
            AssistantContentBlock::Text { text } => output.push_str(text),
            AssistantContentBlock::Reasoning { text } if include_reasoning => {
                let _ = write!(output, "[reasoning: {text}]");
            }
            AssistantContentBlock::ToolCall(call) => {
                let _ = write!(output, "[tool_calls: {}({})]", call.name, call.arguments);
            }
            AssistantContentBlock::Reasoning { .. } => {}
        }
    }
    info!(
        request_id,
        ?output,
        finish_reason,
        streaming,
        delta = false,
        "Generated response"
    );
}

pub(super) fn log_delta(request_id: &str, delta: &ChatMessageDelta) {
    if !enabled!(Level::INFO) {
        return;
    }
    let mut output = delta.content.clone().unwrap_or_default();
    if let Some(reasoning) = &delta.reasoning
        && !reasoning.is_empty()
    {
        let _ = write!(output, "[reasoning: {reasoning}]");
    }
    if let Some(calls) = &delta.tool_calls {
        for call in calls {
            if let Some(function) = &call.function
                && let Some(arguments) = &function.arguments
                && !arguments.is_empty()
            {
                let _ = write!(output, "[tool_calls: {arguments}]");
            }
        }
    }
    if !output.is_empty() {
        info!(
            request_id,
            ?output,
            streaming = true,
            delta = true,
            "Generated response"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    use futures::{StreamExt as _, stream};
    use tracing::instrument::WithSubscriber as _;
    use vllm_chat::{AssistantBlockKind, AssistantToolCall, ChatEvent, FinishReason};

    use super::*;
    use crate::config::ApiServerOptions;
    use crate::routes::openai::chat_completions::{ResponseOptions, chat_completion_chunk_stream};

    #[derive(Clone, Default)]
    struct LogBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for LogBuffer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl LogBuffer {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }

        fn subscriber(&self, level: Level) -> impl tracing::Subscriber + Send + Sync + 'static {
            let buffer = self.clone();
            tracing_subscriber::fmt()
                .without_time()
                .with_ansi(false)
                .with_max_level(level)
                .with_writer(move || buffer.clone())
                .finish()
        }
    }

    fn message() -> AssistantMessage {
        AssistantMessage {
            content: vec![
                AssistantContentBlock::Reasoning {
                    text: "secret".into(),
                },
                AssistantContentBlock::Text {
                    text: "你好".into(),
                },
                AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: "call-1".into(),
                    name: "weather".into(),
                    arguments: "{}".into(),
                }),
            ],
        }
    }

    #[test]
    fn complete_output_includes_tools_and_respects_hidden_reasoning() {
        let logs = LogBuffer::default();
        tracing::subscriber::with_default(logs.subscriber(Level::INFO), || {
            log_message("request-1", &message(), false, "tool_calls", false);
        });
        let text = logs.text();
        assert!(text.contains("你好[tool_calls: weather({})]"));
        assert!(text.contains("request-1"));
        assert!(text.contains("streaming=false delta=false"));
        assert!(text.contains("tool_calls"));
        assert!(!text.contains("secret"));
    }

    #[tokio::test]
    async fn output_logging_switches_preserve_stream_responses() {
        for include_reasoning in [false, true] {
            let mut baseline = None;
            for (outputs, deltas, level) in [
                (false, true, Level::INFO),
                (true, true, Level::INFO),
                (true, false, Level::INFO),
                (true, true, Level::WARN),
            ] {
                let events = stream::iter(vec![
                    Ok(ChatEvent::BlockDelta {
                        index: 0,
                        kind: AssistantBlockKind::Reasoning,
                        delta: "secret".into(),
                        token_count: None,
                    }),
                    Ok(ChatEvent::BlockDelta {
                        index: 1,
                        kind: AssistantBlockKind::Text,
                        delta: "你好".into(),
                        token_count: None,
                    }),
                    Ok(ChatEvent::ToolCallStart {
                        index: 2,
                        id: "call-1".into(),
                        name: "weather".into(),
                    }),
                    Ok(ChatEvent::ToolCallArgumentsDelta {
                        index: 2,
                        delta: "{}".into(),
                    }),
                    Ok(ChatEvent::Done {
                        message: message(),
                        usage: Default::default(),
                        finish_reason: FinishReason::stop_eos(),
                        kv_transfer_params: None,
                        ec_transfer_params: None,
                    }),
                ]);
                let logs = LogBuffer::default();
                let chunks = chat_completion_chunk_stream(
                    events,
                    "request-1".into(),
                    "model".into(),
                    1,
                    ApiServerOptions {
                        enable_log_requests: true,
                        enable_log_outputs: outputs,
                        enable_log_deltas: deltas,
                        ..Default::default()
                    },
                    ResponseOptions {
                        include_reasoning,
                        ..Default::default()
                    },
                )
                .collect::<Vec<_>>()
                .with_subscriber(logs.subscriber(level))
                .await;
                let response = serde_json::to_value(
                    chunks.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
                )
                .unwrap();
                assert_eq!(baseline.get_or_insert(response.clone()), &response);
                let text = logs.text();
                let enabled = outputs && level == Level::INFO;
                assert_eq!(text.contains("Generated response"), enabled);
                assert_eq!(text.contains("delta=true"), enabled && deltas);
                assert_eq!(text.contains("delta=false"), enabled);
                assert_eq!(text.contains("secret"), enabled && include_reasoning);
                if enabled {
                    assert!(text.contains("你好[tool_calls: weather({})]"));
                }
            }
        }
    }
}
