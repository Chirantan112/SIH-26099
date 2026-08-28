"""Lightweight human-review governance for the CPSE harmonization prototype.

The governance layer records analysis and reviewer actions without changing the
existing deterministic mapping authority. It is intentionally dependency-free
and session-friendly so the Streamlit prototype can use it without a database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    """Immutable record of an analysis or human-review action."""

    timestamp: str
    event_type: str
    legacy_material_code: str
    input_description: str
    deterministic_decision: str
    deterministic_material_id: str | None
    ai_candidates: tuple[str, ...] = ()
    selected_material_id: str | None = None
    reviewer_action: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditTrail:
    """Small in-memory audit trail suitable for the prototype UI.

    The trail is deliberately independent of the decision engine. Events are
    append-only for the lifetime of the application/session and can be exported
    by the UI. A production deployment can replace this storage implementation
    with a database without changing the event contract.
    """

    def __init__(self, events: tuple[AuditEvent, ...] = ()) -> None:
        self._events = list(events)

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    def record_analysis(
        self,
        *,
        legacy_material_code: str,
        input_description: str,
        deterministic_decision: str,
        deterministic_material_id: str | None,
        ai_candidates: tuple[str, ...],
    ) -> AuditEvent:
        event = AuditEvent(
            timestamp=_utc_now(),
            event_type="analysis",
            legacy_material_code=legacy_material_code,
            input_description=input_description,
            deterministic_decision=deterministic_decision,
            deterministic_material_id=deterministic_material_id,
            ai_candidates=tuple(ai_candidates),
        )
        self._events.append(event)
        return event

    def record_review(
        self,
        *,
        legacy_material_code: str,
        input_description: str,
        deterministic_decision: str,
        deterministic_material_id: str | None,
        candidate_material_id: str | None,
        action: str,
        note: str = "",
    ) -> AuditEvent:
        normalized_action = action.strip().upper()
        if normalized_action not in {"APPROVE", "REJECT", "REVIEW"}:
            raise ValueError("review action must be APPROVE, REJECT, or REVIEW")
        event = AuditEvent(
            timestamp=_utc_now(),
            event_type="human_review",
            legacy_material_code=legacy_material_code,
            input_description=input_description,
            deterministic_decision=deterministic_decision,
            deterministic_material_id=deterministic_material_id,
            selected_material_id=candidate_material_id,
            reviewer_action=normalized_action,
            note=note,
        )
        self._events.append(event)
        return event


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_audit_trail(existing: Any | None = None) -> AuditTrail:
    """Return an existing compatible trail or a fresh one."""
    if isinstance(existing, AuditTrail):
        return existing
    return AuditTrail()
