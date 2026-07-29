# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import TypeAdapter

from vllm.config import AFDConfig, VllmConfig


def test_default_disabled():
    cfg = AFDConfig()
    assert cfg.role == "none"


def test_parse_from_json():
    cfg = TypeAdapter(AFDConfig).validate_json(
        '{"role": "ffn", "afd_connector": "LoopbackAFDConnector"}'
    )
    assert cfg.is_ffn
    assert cfg.afd_connector == "LoopbackAFDConnector"


def test_role_requires_connector():
    with pytest.raises(ValueError):
        AFDConfig(role="attention")


def test_invalid_role():
    with pytest.raises(ValueError):
        AFDConfig(role="bogus")


def test_role_affects_hash():
    a = AFDConfig(role="attention", afd_connector="LoopbackAFDConnector")
    f = AFDConfig(role="ffn", afd_connector="LoopbackAFDConnector")
    assert a.compute_hash() != f.compute_hash()


def test_scheduler_knob_defaults():
    cfg = AFDConfig()
    assert cfg.max_inflight_batches == 1
    assert cfg.age_limit_ms == 10.0


@pytest.mark.parametrize("capacity", [0, 2])
def test_phase0_requires_one_inflight_batch(capacity):
    with pytest.raises(ValueError, match="must be 1"):
        AFDConfig(max_inflight_batches=capacity)


def test_negative_age_limit_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        AFDConfig(age_limit_ms=-1.0)


def test_attention_role_registers_only_the_pure_fx_cut():
    config = VllmConfig(
        afd_config=AFDConfig(
            role="attention",
            afd_connector="LoopbackAFDConnector",
        )
    )

    splitting_ops = config.compilation_config.splitting_ops or []
    assert "vllm::afd_cut" in splitting_ops
    assert not {"vllm::afd_dispatch", "vllm::afd_recv", "vllm::afd_send"} & set(
        splitting_ops
    )
