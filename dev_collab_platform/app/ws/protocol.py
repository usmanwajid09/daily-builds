"""Minimal RFC 6455 WebSocket protocol implementation from raw sockets.

No `websockets` / `websocket-client` library is used anywhere in this
module (server or test client) -- just `socket`, `hashlib`, `base64`,
and `struct`. This mirrors the rest of the repo's habit of implementing
the interesting protocol/algorithm by hand instead of importing it.

Only what this project needs is implemented: the opening handshake, and
text/close/ping/pong data frames up to 64-bit lengths. Extensions
(permessage-deflate etc.) are not negotiated or supported.
"""
import base64
import hashlib
import socket
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OPCODE_CONTINUATION = 0x0
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2
OPCODE_CLOSE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA


class WebSocketError(Exception):
    """Raised on a malformed handshake or frame, or an unexpected close."""


class ConnectionClosed(WebSocketError):
    """Raised when the peer closed the connection (clean or otherwise)."""


def compute_accept_key(sec_websocket_key: str) -> str:
    digest = hashlib.sha1((sec_websocket_key + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise ConnectionClosed if the peer hangs up."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionClosed("socket closed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_line(sock: socket.socket) -> bytes:
    """Read up to and including the next b'\\r\\n' (used only for the
    plain-HTTP handshake, one byte at a time -- handshakes are tiny and
    infrequent so this isn't a hot path worth buffering)."""
    line = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionClosed("socket closed during handshake")
        line += b
        if line.endswith(b"\r\n"):
            return bytes(line)


def parse_handshake_request(sock: socket.socket) -> dict:
    """Read an HTTP Upgrade request off `sock` and return its headers
    (lower-cased keys) plus 'method' and 'path'."""
    request_line = _recv_line(sock).decode("iso-8859-1").rstrip("\r\n")
    parts = request_line.split(" ")
    if len(parts) != 3:
        raise WebSocketError(f"malformed request line: {request_line!r}")
    method, path, _version = parts

    headers = {}
    while True:
        line = _recv_line(sock).decode("iso-8859-1").rstrip("\r\n")
        if line == "":
            break
        if ":" not in line:
            raise WebSocketError(f"malformed header line: {line!r}")
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    headers["method"] = method
    headers["path"] = path
    return headers


def validate_handshake_headers(headers: dict) -> None:
    if headers.get("method") != "GET":
        raise WebSocketError("handshake must be a GET request")
    if headers.get("upgrade", "").lower() != "websocket":
        raise WebSocketError("missing/invalid Upgrade header")
    if "upgrade" not in headers.get("connection", "").lower():
        raise WebSocketError("missing/invalid Connection header")
    if "sec-websocket-key" not in headers:
        raise WebSocketError("missing Sec-WebSocket-Key header")


def build_handshake_response(sec_websocket_key: str) -> bytes:
    accept_key = compute_accept_key(sec_websocket_key)
    lines = [
        "HTTP/1.1 101 Switching Protocols",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Accept: {accept_key}",
        "\r\n",
    ]
    return "\r\n".join(lines).encode("ascii")


def build_handshake_request(host: str, path: str, sec_websocket_key: str) -> bytes:
    """Client-side handshake request (used by the hand-rolled test client)."""
    lines = [
        f"GET {path} HTTP/1.1",
        f"Host: {host}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {sec_websocket_key}",
        "Sec-WebSocket-Version: 13",
        "\r\n",
    ]
    return "\r\n".join(lines).encode("ascii")


def generate_client_key() -> str:
    import os
    return base64.b64encode(os.urandom(16)).decode("ascii")


def encode_frame(payload: bytes, opcode: int = OPCODE_TEXT, mask: bool = False) -> bytes:
    """Encode a single, final (FIN=1) frame. Servers must send unmasked
    frames (mask=False); clients must send masked frames (mask=True)."""
    header = bytearray()
    header.append(0x80 | (opcode & 0x0F))  # FIN=1, RSV=0, opcode

    length = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header.append(mask_bit | length)
    elif length < 65536:
        header.append(mask_bit | 126)
        header += struct.pack(">H", length)
    else:
        header.append(mask_bit | 127)
        header += struct.pack(">Q", length)

    if mask:
        import os
        masking_key = os.urandom(4)
        header += masking_key
        masked = bytes(b ^ masking_key[i % 4] for i, b in enumerate(payload))
        return bytes(header) + masked

    return bytes(header) + payload


def decode_frame(sock: socket.socket) -> tuple:
    """Read exactly one frame from `sock`. Returns (opcode, payload_bytes).
    Raises ConnectionClosed if the peer hung up mid-frame."""
    first_two = _recv_exact(sock, 2)
    b0, b1 = first_two[0], first_two[1]

    fin = (b0 & 0x80) != 0
    opcode = b0 & 0x0F
    if not fin:
        # Fragmented messages aren't needed for this app's small JSON
        # control messages; treat as a protocol error rather than
        # silently mishandling reassembly.
        raise WebSocketError("fragmented frames are not supported")

    masked = (b1 & 0x80) != 0
    length = b1 & 0x7F

    if length == 126:
        length = struct.unpack(">H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack(">Q", _recv_exact(sock, 8))[0]

    if masked:
        masking_key = _recv_exact(sock, 4)
        raw = _recv_exact(sock, length)
        payload = bytes(b ^ masking_key[i % 4] for i, b in enumerate(raw))
    else:
        payload = _recv_exact(sock, length)

    return opcode, payload
