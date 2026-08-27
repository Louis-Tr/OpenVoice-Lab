from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .config import PipelineConfig, load_config
from .io_utils import read_jsonl
from .review import VALID_DECISIONS
from .text import audit_model_text


def _handler(config: PipelineConfig):
    review_dir = config.path("review")
    queue = {
        row["sample_id"]: row for row in read_jsonl(review_dir / "review_queue.jsonl")
    }
    decisions = review_dir / str(config.section("review")["decisions_file"])
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path == "/review_queue.html":
                self._send_file(
                    review_dir / "review_queue.html", "text/html; charset=utf-8"
                )
                return
            if self.path.startswith("/audio/"):
                sample_id = unquote(self.path.removeprefix("/audio/"))
                item = queue.get(sample_id)
                if item:
                    self._send_file(config.repository_root / item["audio"], "audio/wav")
                    return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/decision":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                sample_id = payload.get("sample_id")
                decision = payload.get("decision")
                if sample_id not in queue or decision not in VALID_DECISIONS:
                    raise ValueError("unknown sample_id or decision")
                action = {
                    "sample_id": sample_id,
                    "decision": decision,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
                if decision == "edit":
                    edited = str(payload.get("edited_model_input_text", "")).strip()
                    if not edited or not audit_model_text(edited)["safe"]:
                        raise ValueError(
                            "edited text must be nonempty and tokenizer-safe"
                        )
                    action["edited_model_input_text"] = edited
                decisions.parent.mkdir(parents=True, exist_ok=True)
                with lock, decisions.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(action, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self.send_response(204)
                self.end_headers()
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the append-only medical speech review UI"
    )
    parser.add_argument("--config", default="training/config/dataset.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    config = load_config(args.config)
    server = ThreadingHTTPServer((args.host, args.port), _handler(config))
    print(f"Review UI: http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
