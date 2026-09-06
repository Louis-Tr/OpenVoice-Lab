"""Pure FIFO backfilling policy with strict aging reservation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


class QueueItem(Protocol):
    id: str
    enqueued_at: float
    sequence: int


@dataclass(frozen=True, slots=True)
class Selection:
    selected_id: str | None
    reserved_id: str | None


def select_next(
    pending: Sequence[QueueItem],
    *,
    now: float,
    aging_threshold_seconds: float,
    fits: Callable[[QueueItem], bool],
) -> Selection:
    """Select first fitting work unless the oldest waiter has aged into priority."""
    if not pending:
        return Selection(None, None)
    ordered = sorted(pending, key=lambda item: (item.enqueued_at, item.sequence))
    overdue = next(
        (
            item
            for item in ordered
            if now - item.enqueued_at >= aging_threshold_seconds
        ),
        None,
    )
    if overdue is not None:
        return Selection(overdue.id if fits(overdue) else None, overdue.id)
    selected = next((item for item in ordered if fits(item)), None)
    return Selection(selected.id if selected is not None else None, None)
