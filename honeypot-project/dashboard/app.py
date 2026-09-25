"""Honeypot web dashboard: serves the UI and exposes parsed attack logs as JSON."""

import json
import os

from flask import Flask, jsonify, render_template, request

LOG_FILE = os.environ.get("LOG_FILE", "/data/attacks.log")
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000

app = Flask(__name__)


def read_attacks() -> list[dict]:
    """Parse the JSON-lines attack log, skipping blank or malformed lines."""
    records = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    records.append({
                        "timestamp": str(entry.get("timestamp", "")),
                        "source_ip": str(entry.get("source_ip", "")),
                        "username": str(entry.get("username", "")),
                        "password": str(entry.get("password", "")),
                    })
    except FileNotFoundError:
        pass
    return records


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/logs")
def api_logs():
    try:
        limit = int(request.args.get("limit", DEFAULT_LIMIT))
    except ValueError:
        limit = DEFAULT_LIMIT
    limit = max(1, min(limit, MAX_LIMIT))

    records = read_attacks()
    latest = records[-1] if records else None
    return jsonify({
        "total_attempts": len(records),
        "unique_ips": len({r["source_ip"] for r in records}),
        "last_source_ip": latest["source_ip"] if latest else None,
        "last_seen": latest["timestamp"] if latest else None,
        "logs": list(reversed(records[-limit:])),  # newest first
    })


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
