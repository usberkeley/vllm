# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import AFDConfig
from vllm.distributed.afd_transfer.connector import (
    AFDConnectorFactory,
    AFDConnectorRole,
)
from vllm.distributed.afd_transfer.connector.loopback import LoopbackAFDConnector


def _config(**kw):
    return SimpleNamespace(afd_config=AFDConfig(**kw))


def test_builtin_connectors_registered():
    for name in ("LoopbackAFDConnector", "P2PAFDConnector", "NixlAFDConnector"):
        assert name in AFDConnectorFactory._registry


def test_create_loopback():
    cfg = _config(role="attention", afd_connector="LoopbackAFDConnector")
    conn = AFDConnectorFactory.create_connector(cfg, AFDConnectorRole.ATTENTION)
    assert isinstance(conn, LoopbackAFDConnector)
    assert conn.role is AFDConnectorRole.ATTENTION


def test_unknown_connector_raises():
    cfg = AFDConfig(role="ffn", afd_connector="DoesNotExist")
    with pytest.raises(ValueError, match="Unsupported AFD connector"):
        AFDConnectorFactory.get_connector_class(cfg)


def test_module_path_takes_priority():
    # The external module path is resolved instead of the registry; pointing it
    # at the loopback module with the matching class name must succeed.
    cfg = AFDConfig(
        role="attention",
        afd_connector="LoopbackAFDConnector",
        afd_connector_module_path=("vllm.distributed.afd_transfer.connector.loopback"),
    )
    assert AFDConnectorFactory.get_connector_class(cfg) is LoopbackAFDConnector


def test_empty_module_path_rejected():
    cfg = AFDConfig(
        role="attention",
        afd_connector="LoopbackAFDConnector",
        afd_connector_module_path="",
    )
    with pytest.raises(ValueError, match="cannot be an empty string"):
        AFDConnectorFactory.get_connector_class(cfg)
