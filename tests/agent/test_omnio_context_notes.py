import pytest

from agent.context_notes import CONTEXT_NOTE_OPEN, context_note_message, is_context_note_only


def _note(text):
    return context_note_message(text)["content"]


@pytest.mark.parametrize("content", [_note("Project framing"), f"{_note('Framing')}\n\n{_note('Change')}"])
def test_recognizes_standalone_context_notes(content):
    assert is_context_note_only(content)


@pytest.mark.parametrize("content", [
    None,
    "",
    [{"type": "text", "text": _note("An attachment caption")}],
    f"{_note('Framing')}\n\nStart a new request",
    f"User input\n\n{_note('Change')}",
    f"{CONTEXT_NOTE_OPEN}\nIncomplete note",
    _note("Literal marker").replace("[OMNIA NOTE", "&#91;OMNIA NOTE"),
])
def test_does_not_classify_user_input_as_a_context_note(content):
    assert not is_context_note_only(content)
