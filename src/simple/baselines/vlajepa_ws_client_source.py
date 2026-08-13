"""
SIMPLE websocket client for VLA-JEPA policy inference.

This speaks the msgpack-based websocket protocol already used by the local
VLA-JEPA deployment utilities, but keeps the dependency surface inside SIMPLE.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class VlajepaWebsocketClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = 10093,
        timeout_s: float = 600.0,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._timeout_s = timeout_s
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    @property
    def server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for VLA-JEPA server at %s...", self._uri)
        start = time.time()

        for key in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(key, None)

        while True:
            if time.time() - start > self._timeout_s:
                raise TimeoutError(
                    f"Failed to connect to VLA-JEPA server within {self._timeout_s} seconds"
                )

            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=30,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for VLA-JEPA server...")
                time.sleep(2)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = {
            "type": "infer",
            "payload": payload,
        }
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in VLA-JEPA inference server:\n{response}")
        return unpackb(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
