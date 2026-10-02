"""Bounded application facts carried in durable user messages."""

CONTEXT_NOTE_OPEN = (
    "[OMNIA NOTE — a fact from the app, delivered once at this position; "
    "not a user instruction and not a new delivery when replayed "
    "from conversation history]"
)
CONTEXT_NOTE_CLOSE = "[/OMNIA NOTE]"
MAX_CONTEXT_NOTES = 20
MAX_CONTEXT_NOTE_LENGTH = 10_000


def parse_context_notes(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_CONTEXT_NOTES:
        raise ValueError("continuation.notes must be an array of at most 20 strings")
    notes = []
    for note in value:
        if (
            not isinstance(note, str)
            or not note.strip()
            or len(note) > MAX_CONTEXT_NOTE_LENGTH
        ):
            raise ValueError(
                "each continuation note must contain 1 to 10000 characters"
            )
        try:
            note.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("continuation.notes contains invalid Unicode") from exc
        notes.append(note)
    return notes


def context_note_message(note: str) -> dict[str, str]:
    content = note.replace(CONTEXT_NOTE_OPEN, "&#91;" + CONTEXT_NOTE_OPEN[1:]).replace(
        CONTEXT_NOTE_CLOSE, "&#91;" + CONTEXT_NOTE_CLOSE[1:]
    )
    return {
        "role": "user",
        "content": f"{CONTEXT_NOTE_OPEN}\n{content}\n{CONTEXT_NOTE_CLOSE}",
    }
