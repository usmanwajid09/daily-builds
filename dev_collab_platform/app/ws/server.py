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
    {"type": "subscribe_notifications"}
        Joins this connection's own per-user notification channel --
        no project_id needed, it's keyed off the authenticated user_id.
    {"type": "unsubscribe_notifications"}

  server -> client:
    {"type": "auth_ok", "workspace_id": <int>, "role": "<role>"}
    {"type": "subscribed", "project_id": <int>}
    {"type": "task_created", "project_id": <int>, "task": {...}}
    {"type": "task_updated", "project_id": <int>, "task": {...}}
    {"type": "task_deleted", "project_id": <int>, "task_id": <int>}
    {"type": "comment_created", "project_id": <int>, "task_id": <int>, "comment": {...}}
    {"type": "comment_deleted", "project_id": <int>, "task_id": <int>, "comment_id": <int>}
    {"type": "project_deleted", "project_id": <int>}
    {"type": "subscribed_notifications"}
    {"type": "notification", "notification": {...}}
        Pushed when a mention or role change creates a notification for
        this connection's user, if they're subscribed to their channel.
    {"type": "error", "message": "<str>"}

The REST API calls `Broadcaster.broadcast` directly after committing
each mutation, so REST writers and WebSocket subscribers share one
process-wide in-memory registry -- no separate message queue needed for
a single-process deployment. Broadcaster channels are keyed generically
(see its docstring): task-board events use an int project_id, personal
notifications use a string key f"user:{user_id}" -- same class, same
subscribe/broadcast/unsubscribe_all machinery, no second broadcaster
needed since Python dicts don't care about key type.
"""
import json
import logging
import socket
import threading

from .. import auth, db
from . import protocol

logger = logging.getLogger("dev_collab_platform.ws")


class ClientConnection:
    """Wraps one accepted socket plus a write lock.

    A connection's own thread (replying to auth/subscribe messages) and
    the Broadcaster (pushing task events from whichever REST request
    thread triggered them) can both want to write to the *same*
    connection at close to the same instant. Two threads calling
    sock.sendall() concurrently on one socket isn't safe -- the kernel
    can interleave the two write() calls, corrupting the WebSocket frame
    stream on the wire even though each individual sendall() call looks
    atomic in Python. Every write to this connection, regardless of
    which thread it comes from, must go through `send()` so they're
    serialized by `write_lock`.

    (Plain socket.socket objects don't support arbitrary attribute
    assignment -- there's no per-socket __dict__ in CPython's C
    implementation -- hence this wrapper instead of e.g. `sock.lock = ...`.)
    """

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.write_lock = threading.Lock()

    def send(self, data: bytes) -> None:
        with self.write_lock:
            self.sock.sendall(data)

    def send_json(self, message: dict) -> None:
        self.send(protocol.encode_frame(json.dumps(message).encode("utf-8")))

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class Broadcaster:
    """Thread-safe registry of channel key -> set of subscribed connections.

    A channel key is any hashable value -- an int project_id for
    task-board/comment events, or a string f"user:{user_id}" for a
    person's own notification channel. Nothing here cares which; it's
    just a dict, so one registry and one set of methods serves both use
    cases instead of a second Broadcaster-like class for notifications.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._channels = {}  # channel key -> set[ClientConnection]

    def subscribe(self, channel, conn: ClientConnection) -> None:
        with self._lock:
            self._channels.setdefault(channel, set()).add(conn)

    def unsubscribe(self, channel, conn: ClientConnection) -> None:
        with self._lock:
            subs = self._channels.get(channel)
            if subs is not None:
                subs.discard(conn)
                if not subs:
                    del self._channels[channel]

    def unsubscribe_all(self, conn: ClientConnection) -> None:
        with self._lock:
            for channel in list(self._channels.keys()):
                self._channels[channel].discard(conn)
                if not self._channels[channel]:
                    del self._channels[channel]

    def subscriber_count(self, channel) -> int:
        with self._lock:
            return len(self._channels.get(channel, ()))

    def broadcast(self, channel, message: dict, exclude: ClientConnection = None) -> int:
        """Send `message` (JSON) to every connection subscribed to
        `channel`. Returns how many it was actually sent to. Dead
        connections are dropped silently -- broadcast is best-effort."""
        with self._lock:
            subs = list(self._channels.get(channel, ()))
        payload = protocol.encode_frame(json.dumps(message).encode("utf-8"))
        sent = 0
        for conn in subs:
            if conn is exclude:
                continue
            try:
                conn.send(payload)
                sent += 1
            except OSError:
                self.unsubscribe(channel, conn)
        return sent


class ConnectionState:
    def __init__(self):
        self.authenticated = False
        self.user_id = None
        self.workspace_id = None
        self.role = None


class WebSocketServer:
    """A minimal threaded WebSocket server: one accept loop, one thread
    per connection. Fine for a demo/dev-scale task board; a production
    version would use an event loop (asyncio/selectors) instead of a
    thread per connection."""

    def __init__(self, host: str, port: int, conn_factory, jwt_secret: str,
                 broadcaster: Broadcaster = None):
        self.host = host
        self.port = port
        self.conn_factory = conn_factory  # callable() -> sqlite3.Connection (DB, not socket)
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
        try:
            headers = protocol.parse_handshake_request(sock)
            protocol.validate_handshake_headers(headers)
            response = protocol.build_handshake_response(headers["sec-websocket-key"])
            sock.sendall(response)  # handshake response: no other writer exists yet
        except protocol.WebSocketError as exc:
            logger.debug("handshake failed: %s", exc)
            sock.close()
            return
        except protocol.ConnectionClosed:
            sock.close()
            return

        conn = ClientConnection(sock)
        state = ConnectionState()
        try:
            self._message_loop(conn, state)
        except protocol.ConnectionClosed:
            pass
        except protocol.WebSocketError as exc:
            logger.debug("protocol error, closing: %s", exc)
        finally:
            self.broadcaster.unsubscribe_all(conn)
            conn.close()

    def _message_loop(self, conn: ClientConnection, state: ConnectionState) -> None:
        db_conn = self.conn_factory()
        try:
            while True:
                opcode, payload = protocol.decode_frame(conn.sock)

                if opcode == protocol.OPCODE_CLOSE:
                    try:
                        conn.send(protocol.encode_frame(payload, opcode=protocol.OPCODE_CLOSE))
                    except OSError:
                        pass  # peer may already be gone; closing below is still correct
                    raise protocol.ConnectionClosed("client sent close frame")
                if opcode == protocol.OPCODE_PING:
                    conn.send(protocol.encode_frame(payload, opcode=protocol.OPCODE_PONG))
                    continue
                if opcode == protocol.OPCODE_PONG:
                    continue
                if opcode != protocol.OPCODE_TEXT:
                    conn.send_json({"type": "error", "message": "only text frames are supported"})
                    continue

                try:
                    message = json.loads(payload.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    conn.send_json({"type": "error", "message": "invalid JSON"})
                    continue

                self._dispatch(conn, state, db_conn, message)
        finally:
            db_conn.close()

    def _dispatch(self, conn: ClientConnection, state: ConnectionState, db_conn, message: dict) -> None:
        msg_type = message.get("type")

        if msg_type == "auth":
            self._handle_auth(conn, state, message)
            return

        if not state.authenticated:
            conn.send_json({"type": "error", "message": "not authenticated, send {type: auth, token} first"})
            return

        if msg_type == "subscribe":
            self._handle_subscribe(conn, state, db_conn, message)
        elif msg_type == "unsubscribe":
            project_id = message.get("project_id")
            if isinstance(project_id, int) and not isinstance(project_id, bool):
                self.broadcaster.unsubscribe(project_id, conn)
            conn.send_json({"type": "unsubscribed", "project_id": project_id})
        elif msg_type == "subscribe_notifications":
            self.broadcaster.subscribe(f"user:{state.user_id}", conn)
            conn.send_json({"type": "subscribed_notifications"})
        elif msg_type == "unsubscribe_notifications":
            self.broadcaster.unsubscribe(f"user:{state.user_id}", conn)
            conn.send_json({"type": "unsubscribed_notifications"})
        else:
            conn.send_json({"type": "error", "message": f"unknown message type: {msg_type!r}"})

    def _handle_auth(self, conn: ClientConnection, state: ConnectionState, message: dict) -> None:
        token = message.get("token")
        if not token:
            conn.send_json({"type": "error", "message": "missing token"})
            return
        try:
            claims = auth.decode_token(self.jwt_secret, token)
        except auth.AuthError as exc:
            conn.send_json({"type": "error", "message": str(exc)})
            return
        state.authenticated = True
        state.user_id = int(claims["sub"])  # "sub" is a string on the wire (RFC 7519)
        state.workspace_id = claims["workspace_id"]
        state.role = claims["role"]
        conn.send_json({"type": "auth_ok", "workspace_id": state.workspace_id, "role": state.role})

    def _handle_subscribe(self, conn: ClientConnection, state: ConnectionState, db_conn, message: dict) -> None:
        project_id = message.get("project_id")
        if not isinstance(project_id, int) or isinstance(project_id, bool):
            conn.send_json({"type": "error", "message": "project_id must be an int"})
            return
        project = db.get_project(db_conn, project_id)
        if project is None or project["workspace_id"] != state.workspace_id:
            # Deliberately the same error for "doesn't exist" and "wrong
            # workspace" so a client can't use this to enumerate other
            # workspaces' project ids.
            conn.send_json({"type": "error", "message": "project not found"})
            return
        self.broadcaster.subscribe(project_id, conn)
        conn.send_json({"type": "subscribed", "project_id": project_id})
