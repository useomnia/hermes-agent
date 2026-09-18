"""Complete recovery for skill bodies omitted by tool-result budgets.

Adapted from Mira Solari's upstream Hermes PR #98736 (2fce577e5535).
The fork has no repeat-view dedup cache; that upstream machinery is omitted.
"""

import json
import re

SKILL_INCOMPLETE_MARKER_PREFIX = "[SKILL_INCOMPLETE:"
# Bounds on the receipt. It is itself a tool result, so it must be small enough
# that it never becomes a spill candidate in its own right.
_MAX_INCOMPLETE_SECTIONS = 40
_MAX_SECTION_HEADING_CHARS = 80
_MAX_INCOMPLETE_RECEIPT_CHARS = 20_000
# Ceiling on a single retrieved section, well under the per-result
# threshold even with six-character JSON escaping, so a section answer is not
# itself a spill candidate. A section
# whose full content exceeds this is NOT truncated -- it is answered with
# navigation only (see _apply_section_selection).
_MAX_SECTION_CHARS = 12_000
# Label for the entry covering everything before the first heading. Skills
# routinely open with prose (and always with frontmatter); without an entry of
# its own that text would have no retrieval route at all.
_PREAMBLE_HEADING = "(document preamble)"
# Metadata worth carrying into an incomplete receipt: what the skill is, whether
# it is usable, and what setup it still needs. Copied verbatim when present.
_INCOMPLETE_RECEIPT_METADATA_KEYS = (
    "description",
    "file",
    "path",
    "skill_dir",
    "readiness_status",
    "setup_needed",
    "setup_skipped",
    "setup_help",
    "setup_note",
    "gateway_setup_hint",
    "missing_required_environment_variables",
    "missing_credential_files",
    "missing_required_commands",
    "required_environment_variables",
    "required_commands",
    "usage_hint",
    "org_provenance",
    "compatibility",
    "tags",
    "related_skills",
    "linked_files",
    "metadata",
)
# Dropped first, in this order, if the receipt still exceeds its cap.
_INCOMPLETE_RECEIPT_DROPPABLE_KEYS = (
    "metadata",
    "linked_files",
    "related_skills",
    "tags",
    "required_environment_variables",
    "org_provenance",
)


def render_skill_result(payload: dict, max_chars: int = 95_000) -> str:
    """Bound ancillary metadata without dropping instructions or navigation."""
    rendered = json.dumps(payload, ensure_ascii=False)
    for key in _INCOMPLETE_RECEIPT_METADATA_KEYS:
        if len(rendered) <= max_chars:
            break
        if key == "file" or key not in payload:
            continue
        payload.pop(key)
        payload["metadata_omitted"] = True
        rendered = json.dumps(payload, ensure_ascii=False)
    return rendered


def _skill_incomplete_marker(skill_name: str, file_path: str | None = None) -> str:
    """Return the canonical incomplete-load marker for *skill_name*.

    *file_path* names the LINKED FILE this receipt is about, when it is about
    one. It must then appear in the continuation call, because a linked file is
    a different document from SKILL.md with its own heading index: a follow-up
    that drops file_path resolves against SKILL.md instead, where the same
    heading text and the same ``#n`` both exist, and answers out of the wrong
    file with section_found true and no marker. The instruction the model is
    handed is the only thing standing between it and that answer.

    Used verbatim by the emit site (the persisted-result formatter) and matched
    by _SKILL_INCOMPLETE_MARKER_RE below -- one string, no drift. The tool-call
    literal is quoted the same way the JSON receipt carrying it is, so what the
    model reads back is what the detector matches.
    """
    subject = f'file "{file_path}" of this skill' if file_path else "this skill"
    target = f'name="{skill_name}"'
    if file_path:
        target += f', file_path="{file_path}"'
    return (
        f"{SKILL_INCOMPLETE_MARKER_PREFIX} {subject} is NOT loaded. Its body was "
        f"removed to protect the context window, so none of its instructions were "
        f"delivered. Do not act on it. Retrieve all required instructions one section at a "
        f'time with skill_view({target}, section="<heading>")]'
    )


def _payload_file(payload: dict) -> str | None:
    """The linked file a skill_view payload is about, or None for SKILL.md.

    Only the linked-file branches set "file"; a main-skill payload carries
    "path" instead. That difference is the whole discriminator -- nothing here
    infers a file from a name.
    """
    file_path = payload.get("file")
    return file_path if isinstance(file_path, str) and file_path else None


def _continuation_hint(payload: dict) -> str:
    """Sentence keeping a linked-file continuation on the same linked file.

    Empty for SKILL.md, where the bare call shape is already correct. A section
    that WAS delivered carries no incomplete marker, so on that path this is the
    only place the next call's shape is stated at all.
    """
    file_path = _payload_file(payload)
    if not file_path:
        return ""
    return (
        f' These selectors are positions in file "{file_path}", not in '
        f"SKILL.md, so keep file_path on every follow-up: "
        f'skill_view(name="{payload.get("name") or ""}", '
        f'file_path="{file_path}", section="<heading>").'
    )


# Matches the canonical marker and captures the skill name. Anchored on the
# shared prefix constant so a wording change updates the emit helper and this
# extractor together. The quotes are optionally backslash-escaped because the
# marker is normally read back out of the JSON receipt that carries it, where
# json.dumps has escaped every double quote inside the string.
_SKILL_INCOMPLETE_MARKER_RE = re.compile(
    re.escape(SKILL_INCOMPLETE_MARKER_PREFIX)
    + r'[^\]]*?skill_view\(name=\\?"([^"\\]+)\\?"'
    + r'(?:, file_path=\\?"[^"\\]*\\?")?, section='
)


def has_skill_incomplete_marker(text: str) -> bool:
    """True when *text* carries a canonical incomplete-load marker."""
    return bool(text) and bool(_SKILL_INCOMPLETE_MARKER_RE.search(text))


def extract_incomplete_skill_names(text: str) -> list[str]:
    """Return skill names referenced by incomplete-load markers in *text*."""
    names: list[str] = []
    for match in _SKILL_INCOMPLETE_MARKER_RE.finditer(text or ""):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


def _skill_entries(body: str) -> list[dict]:
    """Ordered navigation entries for markdown *body*.

    Entry 0 is the document preamble -- everything before the first heading,
    frontmatter included -- and exists only when that text is not blank. Entries
    1..N are the ATX heading occurrences in document order. A repeated heading
    is two DIFFERENT places in the file, so it gets two entries; nothing here is
    deduplicated by text.

    Each entry carries two spans:

    ``own_start``/``own_end``
        the heading line plus its own prose, ending at the next heading of ANY
        level. Parts tile this span.
    ``start``/``end``
        the logical section: down to the next heading of the SAME OR A HIGHER
        level, so descendants are included. This is what a section request
        returns when it fits.

    An entry's own span plus its children's logical spans tile its logical span
    exactly -- that is what makes retrieval lossless: whatever is too large to
    return is always reachable as parts (own text) plus child sections.

    Fenced code blocks are skipped so a shell comment inside an example is not
    mistaken for a heading. Single source of truth for the index, for named
    retrieval and for every selector.
    """
    text = body or ""
    heads: list[tuple[int, str, int]] = []  # (level, heading, start offset)
    offset = 0
    in_fence = False
    fence_marker = ""
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped[:3] in ("```", "~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            offset += len(line)
            continue
        if not in_fence and stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped[level:].strip().rstrip("#").strip()
            if 1 <= level <= 6 and heading:
                heads.append((level, heading, offset))
        offset += len(line)

    # Logical end of each heading: the next heading at the same or a higher
    # level closes it. One pass with a stack -- an O(n^2) scan is noticeable on
    # the 400-heading scraped reference dumps this feature exists for.
    ends = [len(text)] * len(heads)
    open_stack: list[int] = []
    for i, (level, _heading, start) in enumerate(heads):
        while open_stack and heads[open_stack[-1]][0] >= level:
            ends[open_stack.pop()] = start
        open_stack.append(i)

    entries: list[dict] = []
    first = heads[0][2] if heads else len(text)
    if text[:first].strip():
        entries.append(
            {
                "number": 0,
                "heading": _PREAMBLE_HEADING,
                "level": 0,
                "start": 0,
                "end": first,
                "own_start": 0,
                "own_end": first,
                "parent": None,
            }
        )
    for i, (level, heading, start) in enumerate(heads):
        entries.append(
            {
                "number": i + 1,
                "heading": heading,
                "level": level,
                "start": start,
                "end": ends[i],
                "own_start": start,
                "own_end": heads[i + 1][2] if i + 1 < len(heads) else len(text),
                "parent": None,
            }
        )

    # Immediate parent = nearest preceding heading of a strictly smaller level.
    parent_stack: list[dict] = []
    for entry in entries:
        if entry["number"] == 0:
            continue
        while parent_stack and parent_stack[-1]["level"] >= entry["level"]:
            parent_stack.pop()
        entry["parent"] = parent_stack[-1]["number"] if parent_stack else None
        parent_stack.append(entry)
    return entries


def _section_part_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Deterministic part offsets tiling ``text[start:end]``.

    Parts break on line boundaries so each one reads as prose; a single line
    longer than the ceiling is chunked at fixed offsets rather than dropped.
    Concatenating the parts in order reproduces ``text[start:end]`` exactly --
    no gap, no overlap, no character unreachable.
    """
    limit = _MAX_SECTION_CHARS
    if end - start <= limit:
        return [(start, end)]
    spans: list[tuple[int, int]] = []
    cursor = pos = start
    while pos < end:
        newline = text.find("\n", pos, end)
        line_end = end if newline == -1 else newline + 1
        if line_end - cursor > limit:
            if cursor < pos:
                spans.append((cursor, pos))
                cursor = pos
            while line_end - cursor > limit:  # one line longer than the ceiling
                spans.append((cursor, cursor + limit))
                cursor += limit
        pos = line_end
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _selector_of(entry: dict) -> str:
    """Stable selector for *entry*, unique within this exact skill body."""
    return f"#{entry['number']}"


def _leaf_target(entry: dict) -> dict:
    """One retrievable navigation target, as the model sees it."""
    return {
        "selector": _selector_of(entry),
        "heading": entry["heading"][:_MAX_SECTION_HEADING_CHARS],
        "level": entry["level"],
        "chars": entry["end"] - entry["start"],
    }


def _group_span(count: int) -> int:
    """Entries per group so that at most _MAX_INCOMPLETE_SECTIONS groups fit.

    Always strictly smaller than *count* when *count* overflows the cap, so
    expanding a group makes progress and the recursion terminates.
    """
    span = _MAX_INCOMPLETE_SECTIONS
    while span * _MAX_INCOMPLETE_SECTIONS < count:
        span *= _MAX_INCOMPLETE_SECTIONS
    return span


def _chunk_groups(items: list, span: int, selector_of, covers_of) -> list[dict]:
    """Split *items* into bounded groups, each with its own selector."""
    groups = []
    for at in range(0, len(items), span):
        chunk = items[at : at + span]
        groups.append(
            {
                "selector": selector_of(chunk[0], chunk[-1]),
                "covers": covers_of(chunk[0], chunk[-1], at, len(chunk)),
            }
        )
    return groups


def _entry_range_groups(entries: list[dict], total: int, span: int | None = None) -> list[dict]:
    """Group selectors over a run of entries, expressed as ``#first-last``."""
    return _chunk_groups(
        entries,
        span or _group_span(len(entries)),
        lambda lo, hi: f"#{lo['number']}-{hi['number']}",
        lambda lo, hi, _at, n: (
            f"{n} entries, headings {lo['number']}-{hi['number']} of {total}"
            f" — starts at {lo['heading'][:60]!r}"
        ),
    )


def _entry_navigation(entries: list[dict], total: int) -> tuple[list, list]:
    """(leaf targets, group targets) for *entries* -- whichever stays bounded.

    Never both, and never neither while *entries* is non-empty: every entry is
    reachable either directly or by expanding exactly one returned group.
    """
    if len(entries) <= _MAX_INCOMPLETE_SECTIONS:
        return [_leaf_target(e) for e in entries], []
    return [], _entry_range_groups(entries, total)


def _part_navigation(entry: dict, numbered: list, total_parts: int) -> tuple[list, list]:
    """(leaf targets, group targets) for parts of *entry*'s own span.

    *numbered* is ``[(real part number, (start, end)), ...]`` -- the real number
    travels with each span so that expanding a part range keeps naming parts by
    their true position, not by their offset inside the page being expanded.
    """
    selector = _selector_of(entry)
    heading = entry["heading"][:_MAX_SECTION_HEADING_CHARS]
    if len(numbered) <= _MAX_INCOMPLETE_SECTIONS:
        return (
            [
                {
                    "selector": f"{selector}.part{n}",
                    "heading": f"part {n} of {total_parts} of {heading!r}",
                    "level": entry["level"],
                    "chars": hi - lo,
                }
                for n, (lo, hi) in numbered
            ],
            [],
        )
    return [], _chunk_groups(
        numbered,
        _group_span(len(numbered)),
        lambda lo, hi: f"{selector}.part{lo[0]}-{hi[0]}",
        lambda lo, hi, _at, n: (
            f"{n} parts, parts {lo[0]}-{hi[0]} of {total_parts} of {heading!r}"
        ),
    )


# Selector grammar. These are the ONLY shapes the model ever needs, and it
# never has to construct one -- every selector it can use was returned to it by
# a previous result. A heading whose literal text looks like a selector is
# still reachable through its own numeric selector, so nothing becomes
# unretrievable; the selector reading simply wins.
_SELECTOR_ONE_RE = re.compile(r"^#(\d+)$")
_SELECTOR_RANGE_RE = re.compile(r"^#(\d+)-(\d+)$")
_SELECTOR_PART_RE = re.compile(r"^#(\d+)\.part(\d+)$")
_SELECTOR_PART_RANGE_RE = re.compile(r"^#(\d+)\.part(\d+)-(\d+)$")


def _skill_section_navigation(body: str) -> tuple[list, list, int]:
    """Bounded, fully routed index of *body*: (leaves, groups, total entries)."""
    entries = _skill_entries(body)
    leaves, groups = _entry_navigation(entries, len(entries))
    return leaves, groups, len(entries)


def _set_navigation(payload: dict, notice: str, leaves: list, groups: list, total: int) -> None:
    """Answer with navigation only: no body, canonical marker, honest status.

    Every response that withholds content goes through here, so "content is
    absent" and "the incomplete marker is present" cannot come apart.
    """
    payload.pop("content", None)
    payload["load_status"] = "incomplete"
    payload["content_returned"] = False
    payload["notice"] = (
        f"{notice} "
        f"{_skill_incomplete_marker(str(payload.get('name') or ''), _payload_file(payload))}"
    )
    if leaves:
        payload["sections"] = leaves
    else:
        payload.pop("sections", None)
    if groups:
        payload["section_groups"] = groups
    else:
        payload.pop("section_groups", None)
    payload["sections_total"] = total


def _apply_section_selection(payload: dict, section: str) -> None:
    """Narrow *payload*'s content to the requested *section*, in place.

    Three kinds of request, one rule: content comes back only when the WHOLE of
    what was asked for fits. Otherwise the answer is navigation -- selectors
    that between them cover every character of the thing that did not fit --
    and no fragment of the body at all.

    * an exact heading, when it occurs exactly once;
    * ``#n`` / ``#n.partK`` -- a heading occurrence or one part of its own text;
    * ``#a-b`` / ``#n.partA-B`` -- an index page, navigation only.

    A miss, an ambiguous heading and an oversized section are all navigation
    answers rather than tool errors: the model's next move is another
    ``section=`` call either way.
    """
    wanted = str(section).strip()
    body = payload.get("content")
    if not isinstance(body, str):
        return
    payload["section"] = wanted
    entries = _skill_entries(body)
    total = len(entries)
    by_number = {e["number"]: e for e in entries}

    def index_answer(notice: str) -> None:
        """Nothing here answers to what was asked: whole index, no body.

        This is the ONLY path that reports section_found False. An ambiguous
        heading and an oversized section both exist -- they just cannot be
        returned as asked -- so they leave the key off rather than deny them.
        """
        leaves, groups = _entry_navigation(entries, total)
        _set_navigation(payload, notice, leaves, groups, total)
        payload["section_found"] = False

    # ── index pages: navigation only, never an instruction body ──────────
    match = _SELECTOR_RANGE_RE.match(wanted)
    if match:
        lo, hi = int(match.group(1)), int(match.group(2))
        chosen = [e for e in entries if lo <= e["number"] <= hi]
        if not chosen:
            return index_answer(
                f"Selector {wanted!r} covers no headings in this skill."
            )
        leaves, groups = _entry_navigation(chosen, total)
        return _set_navigation(
            payload,
            f"Headings {lo}-{hi} of {total} in this skill. This is an index, "
            f"not skill content — request an entry by its selector.",
            leaves,
            groups,
            total,
        )

    match = _SELECTOR_PART_RANGE_RE.match(wanted)
    if match:
        number, lo, hi = (int(g) for g in match.groups())
        entry = by_number.get(number)
        if entry is None:
            return index_answer(f"Selector {wanted!r} names no section here.")
        spans = _section_part_spans(body, entry["own_start"], entry["own_end"])
        numbered = [(n, s) for n, s in enumerate(spans, start=1) if lo <= n <= hi]
        if not numbered:
            return index_answer(f"Selector {wanted!r} covers no parts here.")
        leaves, groups = _part_navigation(entry, numbered, len(spans))
        return _set_navigation(
            payload,
            f"Parts {lo}-{hi} of {len(spans)} of section "
            f"{entry['heading'][:_MAX_SECTION_HEADING_CHARS]!r}. This is an index, not skill content.",
            leaves,
            groups,
            total,
        )

    # ── one part of a section's own text ─────────────────────────────────
    match = _SELECTOR_PART_RE.match(wanted)
    if match:
        number, wanted_part = int(match.group(1)), int(match.group(2))
        entry = by_number.get(number)
        if entry is None:
            return index_answer(f"Selector {wanted!r} names no section here.")
        spans = _section_part_spans(body, entry["own_start"], entry["own_end"])
        if not 1 <= wanted_part <= len(spans):
            return index_answer(
                f"Section {_selector_of(entry)} has {len(spans)} part(s); "
                f"{wanted!r} is out of range."
            )
        lo, hi = spans[wanted_part - 1]
        leaves, groups = _part_navigation(
            entry, list(enumerate(spans, start=1)), len(spans)
        )
        payload["section_found"] = True
        payload["content"] = body[lo:hi]
        payload["section_part"] = wanted_part
        payload["section_part_count"] = len(spans)
        payload["part_of"] = {
            "selector": _selector_of(entry),
            "heading": entry["heading"][:_MAX_SECTION_HEADING_CHARS],
            "level": entry["level"],
        }
        payload["notice"] = (
            f"This is part {wanted_part} of {len(spans)} of the text of section "
            f"{entry['heading'][:_MAX_SECTION_HEADING_CHARS]!r} ({_selector_of(entry)}). This part is "
            f"complete; the section is NOT. The remaining parts are listed in "
            f'"sections".' + _continuation_hint(payload)
        )
        payload["sections"] = leaves
        if groups:
            payload["section_groups"] = groups
        payload["sections_total"] = total
        return

    # ── a single heading occurrence ──────────────────────────────────────
    entry = None
    match = _SELECTOR_ONE_RE.match(wanted)
    if match:
        entry = by_number.get(int(match.group(1)))
        if entry is None:
            return index_answer(f"Selector {wanted!r} names no section here.")
    else:
        hits = [e for e in entries if e["heading"].strip() == wanted]
        if not hits:
            return index_answer(
                f"Section {wanted!r} was not found in this skill. The headings "
                f'that do exist are listed in "sections"; request one of those '
                f"exactly, or by its selector."
            )
        if len(hits) > 1:
            leaves, groups = _entry_navigation(hits, total)
            payload["section_ambiguous"] = True
            return _set_navigation(
                payload,
                f"Heading {wanted!r} occurs {len(hits)} times in this skill, so "
                f"an exact-heading request is ambiguous and no body was "
                f'returned. Each occurrence has its own selector in "sections" '
                f"— request the one you want.",
                leaves,
                groups,
                total,
            )
        entry = hits[0]

    # Both arms above either returned or bound a real entry.
    size = entry["end"] - entry["start"]
    if size <= _MAX_SECTION_CHARS:
        leaves, groups = _entry_navigation(entries, total)
        payload["section_found"] = True
        payload["section_selector"] = _selector_of(entry)
        payload["content"] = body[entry["start"] : entry["end"]]
        payload["notice"] = (
            f"Section {entry['heading'][:_MAX_SECTION_HEADING_CHARS]!r} ({_selector_of(entry)}) is returned "
            f"in full. It is one entry of {total} in this skill; the rest of "
            f'the skill is NOT loaded. The others are listed in "sections".'
            + _continuation_hint(payload)
        )
        payload["sections"] = leaves
        if groups:
            payload["section_groups"] = groups
        payload["sections_total"] = total
        return

    # Too large to return whole. NO fragment: the own text becomes parts and
    # the child headings become their own targets, and between them they cover
    # every character of the section.
    own_parts = _section_part_spans(body, entry["own_start"], entry["own_end"])
    part_leaves, part_groups = _part_navigation(
        entry, list(enumerate(own_parts, start=1)), len(own_parts)
    )
    children = [e for e in entries if e["parent"] == entry["number"]]
    child_leaves, child_groups = _entry_navigation(children, total)
    payload["section_selector"] = _selector_of(entry)
    payload["section_chars"] = size
    _set_navigation(
        payload,
        f"Section {entry['heading'][:_MAX_SECTION_HEADING_CHARS]!r} ({_selector_of(entry)}) is {size:,} "
        f"characters, larger than the {_MAX_SECTION_CHARS:,}-character result "
        f"ceiling, so NO part of it was returned. Its own text is available as "
        f'parts and its sub-sections as their own selectors, listed in '
        f'"sections"; together they cover the whole section.',
        part_leaves + child_leaves,
        part_groups + child_groups,
        total,
    )


def _skill_view_incomplete_result(content: str, *, tool_name: str = "skill_view") -> str | None:
    """Replacement receipt for a skill_view result whose body was NOT delivered.

    Called by tools/tool_result_storage.py only after that module has already
    decided, on its own, to persist this result -- i.e. the body is gone from
    the model's view no matter what we return. Emits an index-only receipt:
    typed marker, honest status, bounded heading index, availability metadata,
    and no fragment of the skill body.

    Returns None (meaning "use the generic receipt") for anything that is not a
    successful skill_view payload carrying content. Never raises.
    """
    try:
        payload, end = json.JSONDecoder().raw_decode(content)
    except ValueError:
        # Not a JSON payload at all (already-truncated text, a synthetic error
        # string, ...). The generic receipt is the honest answer.
        return None
    if not isinstance(payload, dict) or not payload.get("success"):
        return None
    body = payload.get("content")
    if not isinstance(body, str):
        # Nothing was going to be delivered anyway (dedup stub, error shape).
        return None

    # run_agent.append_toolguard_guidance can append text AFTER the JSON before
    # persistence. That guidance is a live instruction to the model; carry it.
    trailing = content[end:]


    name = str(payload.get("name") or "")
    leaves, groups, total = _skill_section_navigation(body)
    if payload.get("section"):
        # A turn-budget spill must retry the ORIGINAL selector, never re-index
        # the selected fragment against the full document on the next call.
        leaves = [{"selector": payload["section"], "heading": "Retry this section alone"}]
        groups = []
        total = 1
    receipt = {
        "success": True,
        "name": name,
        "load_status": "incomplete",
        "content_returned": False,
        "notice": _skill_incomplete_marker(name, _payload_file(payload)),
        "sections_total": total,
    }
    if leaves:
        receipt["sections"] = leaves
    if groups:
        receipt["section_groups"] = groups
    for key in _INCOMPLETE_RECEIPT_METADATA_KEYS:
        if key in payload and payload[key] is not None:
            receipt[key] = payload[key]

    rendered = json.dumps(receipt, ensure_ascii=False)
    for key in _INCOMPLETE_RECEIPT_DROPPABLE_KEYS:
        if len(rendered) <= _MAX_INCOMPLETE_RECEIPT_CHARS:
            break
        if receipt.pop(key, None) is not None:
            rendered = json.dumps(receipt, ensure_ascii=False)

    # Still over the cap: coarsen the index into fewer, wider groups rather
    # than dropping entries off the end. A dropped entry is a heading the model
    # is told exists and given no way to reach; a wider group is one extra hop.
    entries = [] if payload.get("section") else _skill_entries(body)
    span = None
    while len(rendered) > _MAX_INCOMPLETE_RECEIPT_CHARS and entries:
        span = _group_span(total) if span is None else span * _MAX_INCOMPLETE_SECTIONS
        receipt.pop("sections", None)
        receipt["section_groups"] = _entry_range_groups(entries, total, span)
        rendered = json.dumps(receipt, ensure_ascii=False)
        if len(receipt["section_groups"]) <= 1:
            break  # one group already covers everything; nothing left to merge

    return render_skill_result(receipt, _MAX_INCOMPLETE_RECEIPT_CHARS) + trailing


