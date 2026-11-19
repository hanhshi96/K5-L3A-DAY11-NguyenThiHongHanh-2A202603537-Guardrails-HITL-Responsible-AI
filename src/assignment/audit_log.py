"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}   # key -> thời điểm bắt đầu (để tính latency)
        self._inputs: dict[str, dict] = {}  # key -> thông tin câu hỏi đang chờ câu trả lời

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        """Ghép cặp input/output: ưu tiên request_id, không có thì dùng user_id."""
        return request_id or user_id

    @staticmethod
    def _safe(text: str) -> str:
        """Che PII/secret trước khi ghi log — log cũng không được chứa secret."""
        return content_filter(text or "")["redacted"]

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Lưu câu hỏi + thời điểm bắt đầu."""
        key = self._key(user_id, request_id)
        self._open[key] = time.perf_counter()
        self._inputs[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": self._safe(text),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu câu trả lời, có bị chặn không, lớp nào chặn, latency -> thêm vào self.logs."""
        key = self._key(user_id, request_id)
        start = self._open.pop(key, None)
        entry = self._inputs.pop(key, {"request_id": request_id, "user_id": user_id,
                                       "timestamp": utc_now_iso(), "input": None})
        entry.update({
            "output": self._safe(text),
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1) if start else None,
        })
        self.logs.append(entry)

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)  # tự tạo outputs/ nếu chưa có
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
