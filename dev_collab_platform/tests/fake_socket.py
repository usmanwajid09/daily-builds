"""A tiny in-memory stand-in for socket.socket, used to unit-test the
hand-rolled WebSocket framing in app/ws/protocol.py without opening any
real network connections."""


class FakeSocket:
    def __init__(self, initial_bytes: bytes = b""):
        self._inbox = bytearray(initial_bytes)
        self.sent = bytearray()

    def feed(self, data: bytes) -> None:
        self._inbox += data

    def recv(self, n: int) -> bytes:
        if not self._inbox:
            return b""  # simulates a closed connection
        chunk = bytes(self._inbox[:n])
        del self._inbox[:n]
        return chunk

    def sendall(self, data: bytes) -> None:
        self.sent += data
