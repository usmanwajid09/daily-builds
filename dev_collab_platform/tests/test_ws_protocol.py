import pytest

from app.ws import protocol
from .fake_socket import FakeSocket


def test_accept_key_matches_rfc6455_example():
    # The worked example straight from RFC 6455 section 1.3.
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    assert protocol.compute_accept_key(key) == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_handshake_request_response_roundtrip():
    sock = FakeSocket()
    request = protocol.build_handshake_request("localhost:8765", "/", "dGhlIHNhbXBsZSBub25jZQ==")
    sock.feed(request)

    headers = protocol.parse_handshake_request(sock)
    protocol.validate_handshake_headers(headers)
    assert headers["path"] == "/"
    assert headers["sec-websocket-key"] == "dGhlIHNhbXBsZSBub25jZQ=="

    response = protocol.build_handshake_response(headers["sec-websocket-key"])
    assert response.startswith(b"HTTP/1.1 101 Switching Protocols")
    assert b"s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" in response


@pytest.mark.parametrize("method", ["POST", "PUT"])
def test_handshake_rejects_non_get(method):
    headers = {"method": method, "path": "/", "upgrade": "websocket",
               "connection": "Upgrade", "sec-websocket-key": "x"}
    with pytest.raises(protocol.WebSocketError):
        protocol.validate_handshake_headers(headers)


def test_handshake_rejects_missing_upgrade_header():
    headers = {"method": "GET", "path": "/", "connection": "Upgrade", "sec-websocket-key": "x"}
    with pytest.raises(protocol.WebSocketError):
        protocol.validate_handshake_headers(headers)


def test_handshake_rejects_missing_key():
    headers = {"method": "GET", "path": "/", "upgrade": "websocket", "connection": "Upgrade"}
    with pytest.raises(protocol.WebSocketError):
        protocol.validate_handshake_headers(headers)


@pytest.mark.parametrize("payload_len", [0, 1, 125, 126, 300, 65535, 65536, 70000])
def test_frame_roundtrip_various_lengths_unmasked(payload_len):
    payload = bytes(range(256)) * (payload_len // 256 + 1)
    payload = payload[:payload_len]

    encoded = protocol.encode_frame(payload, opcode=protocol.OPCODE_TEXT, mask=False)
    sock = FakeSocket(encoded)
    opcode, decoded = protocol.decode_frame(sock)
    assert opcode == protocol.OPCODE_TEXT
    assert decoded == payload


def test_frame_roundtrip_masked_like_a_real_client():
    payload = b'{"type": "subscribe", "project_id": 42}'
    encoded = protocol.encode_frame(payload, opcode=protocol.OPCODE_TEXT, mask=True)
    # A masked frame must have the mask bit set and differ from the raw payload.
    assert encoded[1] & 0x80
    sock = FakeSocket(encoded)
    opcode, decoded = protocol.decode_frame(sock)
    assert decoded == payload


def test_decode_frame_raises_on_truncated_connection():
    sock = FakeSocket(b"\x81")  # only 1 of the required 2 header bytes
    with pytest.raises(protocol.ConnectionClosed):
        protocol.decode_frame(sock)


def test_decode_frame_rejects_fragmented_frames():
    # FIN=0 (0x01 instead of 0x81), text opcode, zero-length payload.
    sock = FakeSocket(b"\x01\x00")
    with pytest.raises(protocol.WebSocketError):
        protocol.decode_frame(sock)


def test_ping_and_close_opcodes_roundtrip():
    for opcode in (protocol.OPCODE_PING, protocol.OPCODE_PONG, protocol.OPCODE_CLOSE):
        encoded = protocol.encode_frame(b"", opcode=opcode)
        sock = FakeSocket(encoded)
        got_opcode, payload = protocol.decode_frame(sock)
        assert got_opcode == opcode
        assert payload == b""
