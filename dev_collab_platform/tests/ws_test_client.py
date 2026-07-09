"""A minimal, hand-rolled RFC 6455 client used only by the integration
tests. Reuses the same protocol.py the server uses, but drives a real
socket end to end -- deliberately not the `websockets` pip package, so
the whole client/server round trip in this repo is dependency-free."""
import json
import socket

from app.ws import protocol


class WSClient:
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._handshake(host, port)

    def _handshake(self, host: str, port: int) -> None:
        key = protocol.generate_client_key()
        request = protocol.build_handshake_request(f"{host}:{port}", "/", key)
        self.sock.sendall(request)

        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(1)
            if not chunk:
                raise ConnectionError("server closed connection during handshake")
            response += chunk
        if b"101" not in response.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"handshake rejected: {response!r}")

    def send_json(self, message: dict) -> None:
        payload = json.dumps(message).encode("utf-8")
        self.sock.sendall(protocol.encode_frame(payload, opcode=protocol.OPCODE_TEXT, mask=True))

    def recv_json(self) -> dict:
        opcode, payload = protocol.decode_frame(self.sock)
        if opcode != protocol.OPCODE_TEXT:
            raise ValueError(f"expected a text frame, got opcode {opcode}")
        return json.loads(payload.decode("utf-8"))

    def settimeout(self, seconds: float) -> None:
        self.sock.settimeout(seconds)

    def close(self) -> None:
        try:
            self.sock.sendall(protocol.encode_frame(b"", opcode=protocol.OPCODE_CLOSE, mask=True))
        except OSError:
            pass
        self.sock.close()
