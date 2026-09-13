# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils import deep_gemm


@pytest.mark.parametrize("model_type", ["qwen3_5_text", "qwen3_5_moe_text"])
@pytest.mark.parametrize("use_e8m0", [True, False])
def test_qwen35_auto_disable_requires_e8m0(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    use_e8m0: bool,
) -> None:
    monkeypatch.setattr(
        deep_gemm.current_platform,
        "is_device_capability_family",
        lambda family: family == 100,
    )
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", str(int(use_e8m0)))

    assert deep_gemm.should_auto_disable_deep_gemm(model_type) is use_e8m0
