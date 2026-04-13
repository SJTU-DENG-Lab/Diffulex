from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


def _load_msgpack():
    try:
        import msgpack
    except ImportError as exc:
        raise RuntimeError("Diffulex ZMQ serving requires msgpack. Install the project dependencies first.") from exc
    return msgpack


def _load_zmq():
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("Diffulex ZMQ serving requires pyzmq. Install the project dependencies first.") from exc
    return zmq


def _load_zmq_asyncio():
    try:
        import zmq.asyncio
    except ImportError as exc:
        raise RuntimeError("Diffulex ZMQ serving requires pyzmq. Install the project dependencies first.") from exc
    return zmq.asyncio


class ZmqPushQueue(Generic[T]):
    def __init__(self, addr: str, *, create: bool, encoder: Callable[[T], dict]):
        zmq = _load_zmq()
        self._msgpack = _load_msgpack()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T) -> None:
        event = self._msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self) -> None:
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(self, addr: str, *, create: bool, decoder: Callable[[dict], T]):
        zmq = _load_zmq()
        self._msgpack = _load_msgpack()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        event = self.socket.recv()
        return self.decoder(self._msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self) -> None:
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(self, addr: str, *, create: bool, encoder: Callable[[T], dict]):
        zmq_asyncio = _load_zmq_asyncio()
        self._msgpack = _load_msgpack()
        self.context = zmq_asyncio.Context()
        zmq = _load_zmq()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T) -> None:
        event = self._msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self) -> None:
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(self, addr: str, *, create: bool, decoder: Callable[[dict], T]):
        zmq_asyncio = _load_zmq_asyncio()
        self._msgpack = _load_msgpack()
        self.context = zmq_asyncio.Context()
        zmq = _load_zmq()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        event = await self.socket.recv()
        return self.decoder(self._msgpack.unpackb(event, raw=False))

    def stop(self) -> None:
        self.socket.close()
        self.context.term()
