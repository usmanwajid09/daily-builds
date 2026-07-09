"""Real-time task-board WebSocket server.

Protocol (JSON text frames over the raw RFC 6455 connection implemented
in `protocol.py`):

  client -> server:
    {"type": "auth", "token": "<jwt>"}
        Must be the first message. Verifies the JWT (same secret/claims
        as the REST API) and binds this connection to a workspace_id.
    {"type": "subscribe", "project_id": <int>}
        Joins the broadcast channel for that project. The project must
        belong to the connection's authenticated workspace_id (checked
        against the DB), so a token for workspace A can never subscribe
        to workspace B's project traffic.
    {"type": "unsubscribe", "project_id": <int>}

  server -> client:
    {"type": "auth_ok", "workspace_id": <int>, "role": "<role>"}
    {"type": "subscribed", "project_id": <int>}
    {"type": "task_created", "project_id": <int>, "task": {...}}
    {"type": "task_updated", "project_id": <int>, "task": {...}}
    {"type": "task_deleted", "project_id": <int>, "task_id": <int>}
    {"type": "error", "message": "<str>"}

The REST API (app/routes/task_routes.py) calls `Broadcaster.broadcast`
directly after committing each task mutation, so REST writers and
WebSocket subscribers share one process-wide in-memory registry -- no
separate message queue needed for a single-process deployment.
"""
import json
import logging
import socket
import threading

from .. import auth, db
from . import protocol

logger = logging.getLogger("dev_collab_platform.ws")


class Broadcaster:
    """Thread-safe registry of project_id -> set of subscribed sockets."""

    def __init__(self):
        self._lock = threading.Lock()
        self._channels = {}  # project_id -> set[socket.socket]

    def subscribe(self, project_id: int, sock: socket.socket) -> None:
        with self._lock:
            self._channels.setdefault(project_id, set()).add(sock)

    def unsubscribe(self, project_id: int, sock: socket.socket) -> None:
        with self._lock:
            subs = self._channels.get(project_id)
            if subs is not None:
                subs.discard(sock)
                if not subs:
                    del self._channels[project_id]

    def unsubscribe_all(self, sock: socket.socket) -> None:
        with self._lock:
            for project_id in list(self._channels.keys()):
                self._channels[project_id].discard(sock)
                if not self._channels[project_id]:
                    del self._channels[project_id]

    def subscriber_count(self, project_id: int) -> int:
        with self._lock:
            return len(self._channels.get(project_id, ()))

    def broadcast(self, project_id: int, message: dict, exclude: socket.socket = None) -> int:
        """Send `message` (JSON) to every socket subscribed to project_id.
        Returns the number of sockets it was actually sent to. Dead
        sockets are dropped silently -- broadcast is best-effort."""
        with self._lock:
            subs = list(self._channels.get(project_id, ()))
        payload = protocol.encode_frame(json.dumps(message).encode("utf-8"))
        sent = 0
        for sock in subs:
            if sock is exclude:
                continue
            try:
                sock.sendall(payload)
                sent += 1
            except OSError:
                self.unsubscribe(project_id, sock)
        return sent


class ConnectionState:
    def __init__(self):
        self.authenticated = False
        self.user_id = None
        self.workspace_id = None
        self.role = None


def _send_json(sock: socket.socket, message: dict) -> None:
    sock.sendall(protocol.encode_frame(json.dumps(message).encode("utf-8")))


class WebSocketServer:
    """A minimal threaded WebSocket server: one accept loop, one thread
    per connection. Fine for a demo/dev-scale task board; a production
    version would use an event loop (asyncio/selectors) instead of a
    thread per connection."""

    def __init__(self, host: str, port: int, conn_factory, jwt_secret: str,
                 broadcaster: Broadcaster = None):
        self.host = host
        self.port = port
        self.conn_factory = conn_factory  # callable() -> sqlite3.Connection
        self.jwt_secret = jwt_secret
        self.broadcaster = broadcaster if broadcaster is not None else Broadcaster()
        self._server_sock = None
        self._stop = threading.Event()
        self._ready = threading.Event()

    def serve_forever(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self.port = self._server_sock.getsockname()[1]  # resolves port=0 to the OS-assigned port
        self._server_sock.listen(16)
        self._server_sock.settimeout(0.5)
        logger.info("ws server listening on %s:%s", self.host, self.port)
        self._ready.set()
        while not self._stop.is_set():
            try:
                client_sock, _addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True)
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass

    def wait_ready(self, timeout: float = 2.0) -> None:
        """Blocks until the listening socket is bound (and self.port is
        resolved, if port=0 was requested). Intended for tests that start
        serve_forever() on a background thread."""
        if not self._ready.wait(timeout):
            raise TimeoutError("ws server did not become ready in time")

    def _handle_client(self, sock: socket.socket) -> None:
        state = ConnectionState()
        try:
            headers = protocol.parse_handshake_request(sock)
            protocol.validate_handshake_headers(headers)
            response = protocol.build_handshake_response(headers["sec-websocket-key"])
            sock.sendall(response)
        except protocol.WebSocketError as exc:
            logger.debug("handshake failed: %s", exc)
            sock.close()
            return
        except protocol.ConnectionClosed:
            sock.close()
            return

        try:
            self._message_loop(sock, state)
        except protocol.ConnectionClosed:
            pass
        except protocol.WebSocketError as exc:
            logger.debug("protocol error, closing: %s", exc)
        finally:
            self.broadcaster.unsubscribe_all(sock)
            try:
                sock.close()
            except OSError:
                pass

    def _message_loop(self, sock: socket.socket, state: ConnectionState) -> None:
        conn = self.conn_factory()
        try:
            while True:
                opcode, payload = protocol.decode_frame(sock)

                if opcode == protocol.OPCODE_CLOSE:
                    raise protocol.ConnectionClosed("client sent close frame")
                if opcode == protocol.OPCODE_PING:
                    sock.sendall(protocol.encode_frame(payload, opcode=protocol.OPCODE_PONG))
                    continue
                if opcode == protocol.OPCODE_PONG:
                    continue
                if opcode != protocol.OPCODE_TEXT:
                    _send_json(sock, {"type": "error", "message": "only text frames are supported"})
                    continue

                try:
                    message = json.loads(payload.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    _send_json(sock, {"type": "error", "message": "invalid JSON"})
                    continue

                self._dispatch(sock, state, conn, message)
        finally:
            conn.close()

    def _dispatch(self, sock, state: ConnectionState, conn, message: dict) -> None:
        msg_type = message.get("type")

        if msg_type == "auth":
            self._handle_auth(sock, state, message)
            return

        if not state.authenticated:
            _send_json(sock, {"type": "error", "message": "not authenticated, send {type: auth, token} first"})
            return

        if msg_type == "subscribe":
            self._handle_subscribe(sock, state, conn, message)
        elif msg_type == "unsubscribe":
            project_id = message.get("project_id")
            if isinstance(project_id, int):
                self.broadcaster.unsubscribe(project_id, sock)
            _send_json(sock, {"type": "unsubscribed", "project_id": project_id})
        else:
            _send_json(sock, {"type": "error", "message": f"unknown message type: {msg_type!r}"})

    def _handle_auth(self, sock, state: ConnectionState, message: dict) -> None:
        token = message.get("token")
        if not token:
            _send_json(sock, {"type": "error", "message": "missing token"})
            return
        try:
            claims = auth.decode_token(self.jwt_secret, token)
        except auth.AuthError as exc:
            _send_json(sock, {"type": "error", "message": str(exc)})
            return
        state.authenticated = True
        state.user_id = claims["sub"]
        state.workspace_id = claims["workspace_id"]
        state.role = claims["role"]
        _send_json(sock, {"type": "auth_ok", "workspace_id": state.workspace_id, "role": state.role})

    def _handle_subscribe(self, sock, state: ConnectionState, conn, message: dict) -> None:
        project_id = message.get("project_id")
        if not isinstance(project_id, int):
            _send_json(sock, {"type": "error", "message": "project_id must be an int"})
            return
        project = db.get_project(conn, project_id)
        if project is None or project["workspace_id"] != state.workspace_id:
            # Deliberately the same error for "doesn't exist" and "wrong
            # workspace" so a client can't use this to enumerate other
            # workspaces' project ids.
            _send_json(sock, {"type": "error", "message": "project not found"})
            return
        self.broadcaster.subscribe(project_id, sock)
        _send_json(sock, {"type": "subscribed", "project_id": project_id})
