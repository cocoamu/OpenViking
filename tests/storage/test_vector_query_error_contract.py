# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""EXEPD-229746：向量库查询失败不得伪装成“0 命中”。

2026-10-29 生产：引擎到 VikingDB 的 keep-alive 连接被对端关闭，`query()` 捕获异常后
`return []`、HTTP 仍 200，knowledge-service 只能判 NO_MATCH，用户被告知“知识库中暂无该产品”。

契约：
- 只读检索（`search()` → `strict=True`）：连接类错误重试 1 次；仍失败抛 `UnavailableError`(503)；
  超时不重试。
- 元数据/写入路径（`filter()`、`query()` 默认 `strict=False`）：保持旧行为——记日志、返回 []。
"""

import pytest

from openviking.storage.viking_vector_index_backend import (
    _SingleAccountBackend,
    _is_transient_transport_error,
)
from openviking_cli.exceptions import UnavailableError


class _FakeAsyncAdapter:
    def __init__(self, effects):
        self.effects = list(effects)
        self.calls = 0

    async def call(self, method, **kwargs):
        self.calls += 1
        effect = self.effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect


def _backend(effects):
    backend = object.__new__(_SingleAccountBackend)
    backend._bound_account_id = None
    backend._async_adapter = _FakeAsyncAdapter(effects)
    return backend


@pytest.mark.asyncio
async def test_empty_result_stays_success_empty():
    backend = _backend([[]])

    assert await backend.search() == []
    assert backend._async_adapter.calls == 1


@pytest.mark.asyncio
async def test_transient_error_retried_once_and_recovers():
    rows = [{"uri": "viking://a", "_score": 0.9}]
    backend = _backend([ConnectionError("Remote end closed connection"), rows])

    result = await backend.search()

    assert result == rows
    assert backend._async_adapter.calls == 2


@pytest.mark.asyncio
async def test_two_transient_errors_raise_unavailable():
    backend = _backend([ConnectionError("boom-1"), ConnectionError("boom-2")])

    with pytest.raises(UnavailableError) as excinfo:
        await backend.search()

    assert excinfo.value.code == "UNAVAILABLE"
    assert excinfo.value.details["service"] == "vectordb"
    assert backend._async_adapter.calls == 2


@pytest.mark.asyncio
async def test_timeout_is_not_retried():
    # 超时可能意味着服务端已经执行：重试会放大负载，直接失败。
    backend = _backend([TimeoutError("read timeout"), [{"uri": "should-not-be-used"}]])

    with pytest.raises(UnavailableError):
        await backend.search()

    assert backend._async_adapter.calls == 1


@pytest.mark.asyncio
async def test_non_strict_path_keeps_swallowing_behaviour():
    # filter() 承载入库前清理 / reindex prune / 列目录补 tag：吞掉异常返回空是既有契约。
    backend = _backend([ConnectionError("boom")])

    assert await backend.filter(filter={"op": "must", "field": "uri", "conds": ["x"]}) == []
    assert backend._async_adapter.calls == 1


@pytest.mark.asyncio
async def test_generic_error_strict_raises_without_retry():
    backend = _backend([ValueError("bad filter"), []])

    with pytest.raises(UnavailableError):
        await backend.search()

    assert backend._async_adapter.calls == 1


def test_transient_classifier_walks_cause_chain():
    root = ConnectionError("inner")
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = root

    assert _is_transient_transport_error(wrapped) is True
    assert _is_transient_transport_error(TimeoutError("t")) is False
