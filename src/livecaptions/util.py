"""Small shared helpers."""
from __future__ import annotations

import queue
from typing import Any


def drop_oldest_put(q: "queue.Queue[Any]", item: Any) -> bool:
    """Put without ever blocking a producer. If the queue is full, drop the
    OLDEST item to make room (bounded memory, at the cost of a gap). Returns
    True if an item was dropped."""
    try:
        q.put_nowait(item)
        return False
    except queue.Full:
        pass
    dropped = False
    try:
        q.get_nowait()
        dropped = True
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass
    return dropped
