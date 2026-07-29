# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.distributed.afd_transfer.connector.base import (
    AFDConnectorBase,
    AFDConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.afd import AFDConfig

logger = init_logger(__name__)


class AFDConnectorFactory:
    """Name -> class registry for AFD connectors.

    Mirrors ``KVConnectorFactory``: connectors are registered lazily (only the
    selected connector's module is imported), and an out-of-tree
    ``afd_connector_module_path`` takes priority over the built-in registry.
    """

    _registry: dict[str, Callable[[], type[AFDConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"AFD connector '{name}' is already registered.")

        def loader() -> type[AFDConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_connector_class(cls, afd_config: "AFDConfig") -> type[AFDConnectorBase]:
        connector_name = afd_config.afd_connector
        if connector_name is None:
            raise ValueError("afd_connector is not set in AFDConfig")

        module_path = afd_config.afd_connector_module_path
        if module_path is not None and not module_path:
            raise ValueError("afd_connector_module_path cannot be an empty string.")

        if module_path:
            # External module path takes priority over the built-in registry.
            module = importlib.import_module(module_path)
            try:
                connector_cls = getattr(module, connector_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Class {connector_name} not found in {module_path}"
                ) from e
            return connector_cls

        if connector_name in cls._registry:
            return cls._registry[connector_name]()

        raise ValueError(f"Unsupported AFD connector: {connector_name}")

    @classmethod
    def create_connector(
        cls, config: "VllmConfig", role: AFDConnectorRole
    ) -> AFDConnectorBase:
        afd_config = config.afd_config
        if afd_config is None:
            raise ValueError("afd_config must be set to create an AFD connector")
        connector_cls = cls.get_connector_class(afd_config)
        logger.info(
            "Creating AFD connector '%s' for role %s",
            afd_config.afd_connector,
            role.value,
        )
        return connector_cls(config, role)


# Built-in connectors, registered lazily.
AFDConnectorFactory.register_connector(
    "LoopbackAFDConnector",
    "vllm.distributed.afd_transfer.connector.loopback",
    "LoopbackAFDConnector",
)
AFDConnectorFactory.register_connector(
    "P2PAFDConnector",
    "vllm.distributed.afd_transfer.connector.p2p",
    "P2PAFDConnector",
)
AFDConnectorFactory.register_connector(
    "NixlAFDConnector",
    "vllm.distributed.afd_transfer.connector.nixl",
    "NixlAFDConnector",
)
