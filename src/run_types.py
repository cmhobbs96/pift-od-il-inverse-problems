"""Shared run lifecycle types for frontend/backend orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass(slots=True)
class RunRecord:
    run_id: str
    created_at: str
    updated_at: str
    method: str
    status: RunStatus
    device_requested: str
    device_used: str
    config_hash: str
    runtime_sec: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
