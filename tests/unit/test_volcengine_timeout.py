# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Exercise the real Ark HTTP client against a slow local provider."""

import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from openviking.models.embedder.volcengine_embedders import VolcengineDenseEmbedder


@pytest.mark.parametrize("async_call", [False, True])
def test_slow_provider_is_bounded_by_timeout_and_openviking_retry(async_call):
    calls = []

    class SlowProvider(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append(self.path)
            time.sleep(0.5)
            self.close_connection = True

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowProvider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    embedder = VolcengineDenseEmbedder(
        model_name="test",
        api_key="local-test-key",
        api_base=f"http://127.0.0.1:{server.server_port}/api/v3",
        dimension=2,
        config={"timeout": 0.1, "max_retries": 1},
    )

    async def run_async():
        client = embedder._get_async_client()
        try:
            await embedder.embed_async("timeout probe")
        finally:
            await client.close()

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="Volcengine embedding failed"):
            if async_call:
                asyncio.run(run_async())
            else:
                embedder.embed("timeout probe")
        assert time.monotonic() - started < 4.0
        assert calls == ["/api/v3/embeddings/multimodal"] * 2
    finally:
        embedder.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
