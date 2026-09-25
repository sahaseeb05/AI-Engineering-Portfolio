"""
Decoy SSH server.

Accepts any username/password combination, records each credential attempt as
a JSON line in LOG_FILE, then presents a minimal fake shell before disconnecting.
"""

import json
import logging
import os
import socket
import threading
from datetime import datetime, timezone

import paramiko

LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "2222"))
LOG_FILE = os.environ.get("LOG_FILE", "/data/attacks.log")
HOST_KEY_FILE = os.environ.get("HOST_KEY_FILE", "/data/ssh_host_rsa_key")
SERVER_BANNER = "SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.6"
SESSION_TIMEOUT = 60  # seconds an authenticated session may stay open
MAX_INPUT_LINE = 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("decoy_ssh")

_log_lock = threading.Lock()


def record_attempt(source_ip: str, username: str, password: str) -> None:
    """Append one structured JSON record to the shared attack log."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_ip": source_ip,
        "username": username,
        "password": password,
    }
    line = json.dumps(entry, ensure_ascii=False)
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    log.info("Login attempt from %s user=%r", source_ip, username)


def load_host_key() -> paramiko.RSAKey:
    """Load the persisted host key, generating it on first run for a stable fingerprint."""
    if os.path.exists(HOST_KEY_FILE):
        return paramiko.RSAKey(filename=HOST_KEY_FILE)
    key = paramiko.RSAKey.generate(2048)
    key.write_private_key_file(HOST_KEY_FILE)
    log.info("Generated new host key at %s", HOST_KEY_FILE)
    return key


class DecoyServer(paramiko.ServerInterface):
    def __init__(self, source_ip: str):
        self.source_ip = source_ip
        self.shell_requested = threading.Event()

    def get_allowed_auths(self, username):
        return "password"

    def check_auth_password(self, username, password):
        record_attempt(self.source_ip, username, password)
        return paramiko.AUTH_SUCCESSFUL

    def check_auth_publickey(self, username, key):
        return paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        if kind == "session":
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_pty_request(self, channel, term, width, height, pixelwidth, pixelheight, modes):
        return True

    def check_channel_shell_request(self, channel):
        self.shell_requested.set()
        return True

    def check_channel_exec_request(self, channel, command):
        self.shell_requested.set()
        return True


def fake_shell(channel: paramiko.Channel, username: str) -> None:
    """Very small interactive shell imitation: echoes input, rejects every command."""
    prompt = f"{username}@srv-prod-01:~$ ".encode()
    channel.settimeout(SESSION_TIMEOUT)
    channel.send(b"Welcome to Ubuntu 22.04.4 LTS (GNU/Linux 5.15.0-105-generic x86_64)\r\n\r\n")
    channel.send(prompt)

    buffer = b""
    while True:
        data = channel.recv(1024)
        if not data:
            return
        for byte in data:
            ch = bytes([byte])
            if ch in (b"\r", b"\n"):
                channel.send(b"\r\n")
                command = buffer.decode(errors="replace").strip()
                buffer = b""
                if command in ("exit", "logout", "quit"):
                    channel.send(b"logout\r\n")
                    return
                if command:
                    name = command.split()[0]
                    channel.send(f"-bash: {name}: command not found\r\n".encode())
                channel.send(prompt)
            elif ch in (b"\x7f", b"\x08"):
                if buffer:
                    buffer = buffer[:-1]
                    channel.send(b"\b \b")
            elif ch in (b"\x03", b"\x04"):  # Ctrl-C / Ctrl-D
                return
            elif len(buffer) < MAX_INPUT_LINE:
                buffer += ch
                channel.send(ch)


def handle_client(client: socket.socket, addr, host_key: paramiko.RSAKey) -> None:
    source_ip = addr[0]
    transport = None
    try:
        transport = paramiko.Transport(client)
        transport.local_version = SERVER_BANNER
        transport.add_server_key(host_key)
        server = DecoyServer(source_ip)
        transport.start_server(server=server)

        channel = transport.accept(SESSION_TIMEOUT)
        if channel is None:
            return
        if not server.shell_requested.wait(10):
            return
        fake_shell(channel, transport.get_username() or "root")
    except (paramiko.SSHException, EOFError, socket.error, socket.timeout) as exc:
        log.debug("Connection from %s ended: %s", source_ip, exc)
    except Exception:
        log.exception("Unexpected error handling %s", source_ip)
    finally:
        try:
            if transport is not None:
                transport.close()
            client.close()
        except Exception:
            pass


def main() -> None:
    host_key = load_host_key()
    # Make sure the log exists so the dashboard can read it before the first attack.
    open(LOG_FILE, "a", encoding="utf-8").close()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LISTEN_HOST, LISTEN_PORT))
    sock.listen(100)
    log.info("Decoy SSH listening on %s:%d, logging to %s", LISTEN_HOST, LISTEN_PORT, LOG_FILE)

    while True:
        client, addr = sock.accept()
        client.settimeout(SESSION_TIMEOUT)
        threading.Thread(target=handle_client, args=(client, addr, host_key), daemon=True).start()


if __name__ == "__main__":
    main()
