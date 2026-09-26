# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Readiness must only cover dependencies required to accept local traffic."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.routers import system


class _ConfigWithForbiddenEmbeddingAccess:
    @property
    def embedding(self):
        raise AssertionError("readiness must not access the embedding configuration")


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_healthy, expected_status", [(True, 200), (False, 503)])
async def test_readiness_does_not_call_embedding_provider(
    monkeypatch, storage_healthy, expected_status
):
    storage = SimpleNamespace(health_check=AsyncMock(return_value=storage_healthy))
    viking_fs = SimpleNamespace(_get_vector_store=lambda: storage)
    monkeypatch.setattr(system, "get_service", lambda: SimpleNamespace(_initialized=True))
    monkeypatch.setattr(
        system,
        "_probe_agfs_readiness",
        AsyncMock(
            return_value={
                "status": "ok",
                "checks": {"filesystem": "ok", "multiwrite_sync": "ok"},
            }
        ),
    )
    monkeypatch.setattr(system, "get_viking_fs", lambda: viking_fs)

    from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton
    from openviking_cli.utils import ollama

    monkeypatch.setattr(
        OpenVikingConfigSingleton,
        "get_instance",
        staticmethod(lambda: _ConfigWithForbiddenEmbeddingAccess()),
    )
    monkeypatch.setattr(ollama, "detect_ollama_in_config", lambda _config: (False, None, None))

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(api_key_manager=object()))
    )
    response = await system.readiness_check(request)

    assert response.status_code == expected_status
    assert json.loads(response.body)["checks"] == {
        "agfs": {
            "status": "ok",
            "checks": {"filesystem": "ok", "multiwrite_sync": "ok"},
        },
        "vectordb": "ok" if storage_healthy else "unhealthy",
        "api_key_manager": "ok",
        "ollama": "not_configured",
    }
