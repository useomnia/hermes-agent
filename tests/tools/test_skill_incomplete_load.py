"""A mandatory skill that did not reach the model is never reported as loaded.

`skill_view` can have its body removed by two independent context-protection
decisions in `tools/tool_result_storage.py`: the per-result threshold (layer 2)
and aggregate turn-budget enforcement (layer 3). Both used to leave a generic
preview that still opened with `"success": true` plus a truncated fragment of
the skill body, and let partial instructions appear complete. These tests pin the honest receipt instead.
"""

import json
import re

import pytest

from tools.budget_config import DEFAULT_BUDGET
from tools.registry import registry
import tools.skills_tool  # registers the production handler
from tools.tool_result_storage import maybe_persist_tool_result

FINAL_RULE = "ALWAYS run the migration before restarting the service."
BODY_LINE = "Some ordinary instruction prose that fills the body of the skill file."


def _write_skill(skills_dir, name, target_chars):
    """Write a SKILL.md of roughly *target_chars* with a known final rule."""
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    head = (
        f"---\nname: {name}\ndescription: A skill used by the incomplete-load tests\n"
        f"---\n\n# {name}\n\n## Overview\n\nWhat this skill is for.\n\n## Procedure\n\n"
    )
    tail = f"\n## Final Governing Rule\n\n{FINAL_RULE}\n"
    filler = BODY_LINE + "\n"
    n = max(0, (target_chars - len(head) - len(tail)) // len(filler))
    (d / "SKILL.md").write_text(head + (filler * n) + tail, encoding="utf-8")
    return d / "SKILL.md"


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    _write_skill(skills, "oversized-skill", 120_000)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


def _view_through_executor(name, task_id, **kw):
    """Exactly what agent/tool_executor.py does: registered handler, then layer 2."""
    handler = registry.get_entry("skill_view").handler
    raw = handler({"name": name, **kw}, task_id=task_id)
    return raw, maybe_persist_tool_result(
        content=raw,
        tool_name="skill_view",
        tool_use_id=f"tu_{task_id}",
        env=None,
        config=DEFAULT_BUDGET,
    )


class TestPerResultSpill:
    def test_oversized_skill_view_returns_index_only_receipt(self, skills_home):
        raw, delivered = _view_through_executor("oversized-skill", "t-per-result")

        assert len(raw) > DEFAULT_BUDGET.resolve_threshold("skill_view")
        assert delivered != raw, "layer 2 must still remove the oversized body"

        payload = json.loads(delivered)
        assert "[SKILL_INCOMPLETE:" in payload["notice"]
        assert payload["load_status"] == "incomplete"
        assert payload["content_returned"] is False
        assert "content" not in payload

        headings = [s["heading"] for s in payload["sections"]]
        assert "Final Governing Rule" in headings
        assert 0 < len(payload["sections"]) <= 40

        # No skill body reaches the model -- not one line of it.
        assert BODY_LINE not in delivered
        assert FINAL_RULE not in delivered


class _FakeEnv:
    """Minimal remote sandbox env: records the write and reports success.

    *probe_readable* False makes the readability probe on the host-side
    spillover copy fail, which is what drives layer 2 down to the in-sandbox
    write fallback.
    """

    def __init__(self, probe_readable=True):
        self.written = None
        self.probe_readable = probe_readable

    def get_temp_dir(self):
        return "/tmp"

    def execute(self, cmd, timeout=30, stdin_data=None):
        if cmd.startswith("test -r "):
            return {"returncode": 0 if self.probe_readable else 1}
        self.written = stdin_data
        return {"returncode": 0}


class TestPerResultSpillWithSandbox:
    """Layer 2 has more than one return path once a sandbox env is present.
    None of them may emit the generic block for a skill_view result."""

    def _spill(self, env, tool_use_id):
        handler = registry.get_entry("skill_view").handler
        raw = handler({"name": "oversized-skill"}, task_id="t-sandbox")
        return raw, maybe_persist_tool_result(
            content=raw,
            tool_name="skill_view",
            tool_use_id=tool_use_id,
            env=env,
            config=DEFAULT_BUDGET,
        )

    def _assert_index_only(self, delivered):
        payload = json.loads(delivered)
        assert "[SKILL_INCOMPLETE:" in payload["notice"]
        assert payload["load_status"] == "incomplete"
        assert "content" not in payload
        assert "<persisted-output>" not in delivered
        assert BODY_LINE not in delivered
        assert FINAL_RULE not in delivered

    def test_in_sandbox_write_still_yields_index_only_receipt(self, skills_home):
        """Remote backend whose sandbox cannot read the spillover copy, so
        layer 2 falls back to writing into the sandbox temp dir."""
        env = _FakeEnv(probe_readable=False)
        raw, delivered = self._spill(env, "tu_sandbox_write")

        # The full result still reaches disk unchanged -- nothing is lost.
        assert env.written == raw

        self._assert_index_only(delivered)


class TestAggregateSpill:
    """A skill_view result that is legal under every per-result cap can still be
    removed later, by aggregate turn-budget enforcement. That decision is made
    after skill_view has returned, and it used to discard the tool identity --
    so the skill was replaced by the generic preview and no marker was possible.
    """

    def test_legal_size_skill_view_spilled_by_turn_budget_gets_the_marker(
        self, tmp_path, monkeypatch
    ):
        from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, enforce_turn_budget
        from agent.tool_dispatch_helpers import make_tool_result_message

        home = tmp_path / ".hermes"
        skills = home / "skills"
        skills.mkdir(parents=True)
        _write_skill(skills, "legal-skill", 90_000)
        monkeypatch.setenv("HERMES_HOME", str(home))

        handler = registry.get_entry("skill_view").handler
        raw = handler({"name": "legal-skill"}, task_id="t-aggregate")

        # Legal at layer 2: it survives the per-result threshold untouched.
        assert len(raw) < DEFAULT_BUDGET.resolve_threshold("skill_view")
        assert maybe_persist_tool_result(
            content=raw, tool_name="skill_view", tool_use_id="tu_a",
            env=None, config=DEFAULT_BUDGET,
        ) == raw

        messages = [
            make_tool_result_message("skill_view", raw, "tu_a"),
            make_tool_result_message("search_files", "x" * 60_000, "tu_b"),
            make_tool_result_message("read_file", "y" * 60_000, "tu_c"),
        ]
        assert sum(len(m["content"]) for m in messages) > DEFAULT_BUDGET.turn_budget

        enforce_turn_budget(messages, env=None, config=DEFAULT_BUDGET)
        delivered = messages[0]["content"]

        assert delivered != raw, "the turn budget must still have spilled it"
        payload = json.loads(delivered)
        assert "[SKILL_INCOMPLETE:" in payload["notice"]
        assert payload["load_status"] == "incomplete"
        assert payload["content_returned"] is False
        assert "content" not in payload
        assert "Final Governing Rule" in [s["heading"] for s in payload["sections"]]
        assert PERSISTED_OUTPUT_TAG not in delivered
        assert BODY_LINE not in delivered
        assert FINAL_RULE not in delivered




class TestMarkerCannotDrift:
    """PR #44166 shipped a marker whose emit side said "[SKILL_PRUNED:" while
    its check side matched "[SKILL_PRUNED]"; the re-injection then fired even
    when the marker had survived. Mirrors tests/agent/test_ghost_skill_pruning.py
    for the incomplete-load marker: one constant, both sides.
    """

    def test_emit_and_detect_share_one_constant(self):
        import tools.skill_delivery as st

        marker = st._skill_incomplete_marker("some-skill")
        assert marker.startswith(st.SKILL_INCOMPLETE_MARKER_PREFIX)
        assert st.has_skill_incomplete_marker(marker)
        assert st.extract_incomplete_skill_names(marker) == ["some-skill"]
        # The detector is BUILT from the prefix, not from a copy of it.
        import re as _re
        assert _re.escape(st.SKILL_INCOMPLETE_MARKER_PREFIX) in \
            st._SKILL_INCOMPLETE_MARKER_RE.pattern

    def test_the_marker_the_model_receives_is_the_one_we_detect(self, skills_home):
        _raw, delivered = _view_through_executor("oversized-skill", "t-marker")
        import tools.skill_delivery as st

        assert st.has_skill_incomplete_marker(delivered)
        assert st.extract_incomplete_skill_names(delivered) == ["oversized-skill"]
        assert json.loads(delivered)["notice"] == st._skill_incomplete_marker(
            "oversized-skill"
        )


class TestSectionRetrieval:
    """The index receipt is only useful if a named heading can be fetched."""

    def test_named_section_returns_exactly_that_section(self, skills_home):
        handler = registry.get_entry("skill_view").handler
        r = json.loads(
            handler(
                {"name": "oversized-skill", "section": "Final Governing Rule"},
                task_id="t-section",
            )
        )
        assert r["success"] is True
        assert r["section"] == "Final Governing Rule"
        assert r["section_found"] is True
        assert FINAL_RULE in r["content"]
        assert r["content"].lstrip().startswith("## Final Governing Rule")
        # Exactly that section: none of the preceding body comes with it.
        assert BODY_LINE not in r["content"]

    def test_unknown_heading_returns_the_index_and_a_plain_notice(self, skills_home):
        handler = registry.get_entry("skill_view").handler
        r = json.loads(
            handler(
                {"name": "oversized-skill", "section": "No Such Heading"},
                task_id="t-section-miss",
            )
        )
        assert r["success"] is True
        assert r["section_found"] is False
        assert "content" not in r
        assert "not found" in r["notice"].lower()
        assert "error" not in r
        assert "Final Governing Rule" in [s["heading"] for s in r["sections"]]

    def test_section_request_is_not_answered_from_the_dedup_stub(self, skills_home):
        handler = registry.get_entry("skill_view").handler
        # A full, fitting view first, so a dedup record exists for this skill.
        _write_skill(skills_home / "skills", "small-skill", 400)
        handler({"name": "small-skill"}, task_id="t-section-dedup")
        r = json.loads(
            handler(
                {"name": "small-skill", "section": "Final Governing Rule"},
                task_id="t-section-dedup",
            )
        )
        assert r.get("dedup") is not True
        assert FINAL_RULE in r["content"]

    def test_omitting_section_preserves_current_behaviour(self, skills_home):
        handler = registry.get_entry("skill_view").handler
        _write_skill(skills_home / "skills", "small-skill", 400)
        r = json.loads(handler({"name": "small-skill"}, task_id="t-section-none"))
        assert "section" not in r
        assert "section_found" not in r
        assert FINAL_RULE in r["content"]
        assert BODY_LINE in r["content"]


class TestGuardrailSuffixAndBadInput:
    """run_agent.append_toolguard_guidance can append text AFTER the JSON before
    persistence. That guidance is a live instruction to the model, so the
    receipt must carry it; and anything that does not parse must fall through to
    the generic block rather than raise inside the storage layer."""

    def test_trailing_guardrail_guidance_survives(self, skills_home):
        from agent.tool_guardrails import ToolGuardrailDecision, append_toolguard_guidance
        import tools.skill_delivery as st

        handler = registry.get_entry("skill_view").handler
        raw = handler({"name": "oversized-skill"}, task_id="t-guard")
        decision = ToolGuardrailDecision(
            action="warn",
            code="repeat_tool_calls",
            count=4,
            message="Vary the arguments or change approach.",
        )
        with_guidance = append_toolguard_guidance(raw, decision)
        assert with_guidance != raw

        delivered = maybe_persist_tool_result(
            content=with_guidance, tool_name="skill_view", tool_use_id="tu_guard",
            env=None, config=DEFAULT_BUDGET,
        )
        assert "Vary the arguments or change approach." in delivered
        assert delivered.rstrip().endswith("]")
        assert st.has_skill_incomplete_marker(delivered)
        assert BODY_LINE not in delivered

        payload, end = json.JSONDecoder().raw_decode(delivered)
        assert payload["load_status"] == "incomplete"
        assert "Tool loop warning" in delivered[end:]

    @pytest.mark.parametrize(
        "content",
        [
            "not json at all",
            "",
            "[1, 2, 3]",
            '{"success": false, "error": "nope"}',
            '{"success": true, "name": "x"}',                 # no content key
            '{"success": true, "name": "x", "content": null}',
            '{"success": true, "status": "unchanged", "dedup": true, '
            '"content_returned": false, "message": "..."}',   # the dedup stub
        ],
    )
    def test_unparseable_or_bodyless_input_falls_through_without_raising(
        self, content, tmp_path, monkeypatch
    ):
        """A payload the formatter declines gets exactly the generic receipt --
        byte for byte the same string the storage layer produces with no
        formatter registered at all."""
        import tools.oversized_result_formatters as orf
        import tools.skill_delivery as st

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        assert st._skill_view_incomplete_result(content) is None

        padded = content + "p" * 100_001

        def _persist():
            return maybe_persist_tool_result(
                content=padded, tool_name="skill_view", tool_use_id="tu_fall",
                env=None, config=DEFAULT_BUDGET,
            )

        delivered = _persist()
        saved = dict(orf._FORMATTERS)
        monkeypatch.setattr(orf, "_FORMATTERS", {})
        try:
            generic = _persist()
        finally:
            monkeypatch.setattr(orf, "_FORMATTERS", saved)

        assert delivered == generic
        assert "[SKILL_INCOMPLETE:" not in delivered




# ── Batched-correction coverage (review findings B1-B4) ─────────────────────
# Each class below reproduces one blocking finding from the exact-commit review
# of 5e2c4dd988 and pins the corrected behaviour.

def _write_raw_skill(skills_dir, name, body):
    """Write a SKILL.md whose body is *body* verbatim, after frontmatter."""
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: A skill used by the incomplete-load "
        f"tests\n---\n\n{body}",
        encoding="utf-8",
    )
    return path


def _entries(payload):
    """Leaf navigation targets a response offered, in order."""
    return list(payload.get("sections") or [])


def _groups(payload):
    """Group navigation targets a response offered, in order."""
    return list(payload.get("section_groups") or [])


def _crawl_navigation(handler, name, task_id, payload, budget=4000):
    """Follow ONLY selectors the tool returned; never construct one.

    Returns (leaf entries reached, largest response seen, number of calls).
    """
    leaves, seen, calls, biggest = {}, set(), 0, len(json.dumps(payload))
    queue = [("leaf", e) for e in _entries(payload)]
    queue += [("group", g) for g in _groups(payload)]
    while queue:
        kind, entry = queue.pop(0)
        selector = entry["selector"]
        if kind == "leaf":
            leaves[selector] = entry
            continue
        if selector in seen:
            continue
        seen.add(selector)
        calls += 1
        assert calls <= budget, "navigation did not converge"
        raw = handler({"name": name, "section": selector}, task_id=task_id)
        biggest = max(biggest, len(raw))
        sub = json.loads(raw)
        # A group selector is navigation only -- never an instruction body.
        assert "content" not in sub, f"group selector {selector} returned a body"
        queue += [("leaf", e) for e in _entries(sub)]
        queue += [("group", g) for g in _groups(sub)]
    return leaves, biggest, calls




class TestEveryHeadingIsReachable:
    """B2: more than 40 headings. Every occurrence must be reachable by
    recursively following selectors the tool itself returned, and every
    response must stay bounded.
    """

    HEADINGS = 120

    def _skill(self, skills_home):
        body = "".join(
            f"## Heading {i:03d}\n\nprose for {i:03d}\n" + ("x" * 900) + "\n\n"
            for i in range(1, self.HEADINGS + 1)
        )
        _write_raw_skill(skills_home / "skills", "many-headings", body)
        return "many-headings"

    def test_recursive_navigation_reaches_every_heading(self, skills_home):
        name = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        raw, delivered = _view_through_executor(name, "t-many")
        assert len(raw) > 100_000, "fixture must trip the per-result threshold"
        receipt = json.loads(delivered)
        assert receipt["load_status"] == "incomplete"
        assert "content" not in receipt

        leaves, biggest, _calls = _crawl_navigation(
            handler, name, "t-many-nav", receipt
        )
        found = {e["heading"] for e in leaves.values()}
        expected = {f"Heading {i:03d}" for i in range(1, self.HEADINGS + 1)}
        assert expected <= found, sorted(expected - found)[:10]
        assert biggest <= 100_000, "a navigation response became a spill candidate"

    def test_no_omitted_count_without_a_route(self, skills_home):
        name = self._skill(skills_home)
        _raw, delivered = _view_through_executor(name, "t-many-omit")
        receipt = json.loads(delivered)
        assert "sections_omitted" not in receipt
        assert _entries(receipt) or _groups(receipt)

    def test_every_returned_selector_resolves_to_its_own_heading(self, skills_home):
        name = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        _raw, delivered = _view_through_executor(name, "t-many-resolve")
        leaves, _b, _c = _crawl_navigation(
            handler, name, "t-many-resolve", json.loads(delivered)
        )
        for selector, entry in sorted(leaves.items()):
            if not entry["heading"].startswith("Heading "):
                continue
            got = json.loads(
                handler({"name": name, "section": selector}, task_id="t-many-resolve")
            )
            assert got["section_found"] is True, selector
            assert got["content"].lstrip().startswith(f"## {entry['heading']}")


class TestOversizedSectionHasNoPartialBody:
    """B3: a section too large to return must return NO body fragment, and its
    part selectors must reconstruct the original span byte-for-byte.
    """

    def _skill(self, skills_home):
        body = "## Big Leaf\n\n" + ("y" * 79 + "\n") * 1_700
        path = _write_raw_skill(skills_home / "skills", "big-leaf", body)
        return "big-leaf", path

    def test_oversized_leaf_returns_a_marker_and_no_fragment(self, skills_home):
        name, _path = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler({"name": name, "section": "Big Leaf"}, task_id="t-leaf")
        )
        assert "content" not in got, "an oversized section returned a body fragment"
        assert got["load_status"] == "incomplete"
        assert "[SKILL_INCOMPLETE:" in got["notice"]
        assert got.get("section_found") is not True
        assert _entries(got) or _groups(got)

    def test_part_selectors_reconstruct_the_span_exactly(self, skills_home):
        name, path = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        full = path.read_text(encoding="utf-8")
        span = full[full.index("## Big Leaf") :]

        got = json.loads(
            handler({"name": name, "section": "Big Leaf"}, task_id="t-leaf-parts")
        )
        leaves, biggest, _calls = _crawl_navigation(handler, name, "t-leaf-parts", got)
        assert biggest <= 100_000

        rebuilt = ""
        parts = [s for s in leaves if ".part" in s]
        assert len(parts) > 1, "an oversized leaf must be split into parts"
        for selector in sorted(parts, key=lambda s: int(s.split(".part")[1])):
            piece = json.loads(
                handler({"name": name, "section": selector}, task_id="t-leaf-parts")
            )
            assert piece["section_found"] is True, selector
            assert piece["section_part_count"] == len(parts)
            assert piece["section_part"] == int(selector.split(".part")[1])
            assert "[SKILL_INCOMPLETE:" not in json.dumps(piece)
            assert len(piece["content"]) <= 60_000
            rebuilt += piece["content"]
        assert rebuilt == span, "parts must tile the span with no gap or overlap"


class TestPreambleIsReachable:
    """Skills routinely open with prose before the first heading -- and always
    with frontmatter. Without an entry of its own that text has no selector, so
    an index that names only headings still strands it.
    """

    def test_the_text_before_the_first_heading_has_its_own_selector(self, skills_home):
        _write_raw_skill(
            skills_home / "skills",
            "with-preamble",
            "Read this before anything else.\n\n## Later\n\nlater prose\n",
        )
        handler = registry.get_entry("skill_view").handler
        index = json.loads(
            handler(
                {"name": "with-preamble", "section": "No Such Heading"},
                task_id="t-preamble",
            )
        )
        preamble = [e for e in _entries(index) if e["selector"] == "#0"]
        assert preamble, [e["selector"] for e in _entries(index)]

        got = json.loads(
            handler({"name": "with-preamble", "section": "#0"}, task_id="t-preamble")
        )
        assert got["section_found"] is True
        assert "Read this before anything else." in got["content"]
        assert "later prose" not in got["content"]


class TestSectionsAreHierarchical:
    """B3's root cause: a section that ended at the next heading of ANY level
    could only overflow the ceiling by containing no headings at all, which is
    exactly why "request its sub-headings" was unsatisfiable. A section runs to
    the next heading of the SAME OR A HIGHER level, so sub-sections come with
    their parent -- and an oversized parent has real children to hand back.
    """

    def test_a_parent_section_carries_its_sub_sections(self, skills_home):
        body = (
            "# Parent\n\nparent prose\n\n"
            "## Child One\n\nchild one prose\n\n"
            "### Grandchild\n\ngrandchild prose\n\n"
            "# Sibling\n\nsibling prose\n"
        )
        _write_raw_skill(skills_home / "skills", "nested", body)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler({"name": "nested", "section": "Parent"}, task_id="t-nest")
        )
        assert got["section_found"] is True
        for expected in ("parent prose", "child one prose", "grandchild prose"):
            assert expected in got["content"], expected
        # ...and stops at the next heading of the same level.
        assert "sibling prose" not in got["content"]

    def test_an_oversized_parent_hands_back_its_children(self, skills_home):
        filler = ("k" * 99 + "\n") * 350  # 35,000 chars per child
        body = (
            "# Parent\n\nparent prose\n\n"
            + "".join(f"## Child {i}\n\n{filler}\n" for i in range(1, 4))
        )
        _write_raw_skill(skills_home / "skills", "big-parent", body)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler({"name": "big-parent", "section": "Parent"}, task_id="t-bigparent")
        )
        assert "content" not in got
        assert got["load_status"] == "incomplete"
        assert got["section_chars"] > 60_000
        offered = [e["heading"] for e in _entries(got)]
        for i in range(1, 4):
            assert f"Child {i}" in offered, offered
        # Its own prose is reachable too, as a part -- nothing is stranded.
        assert any(".part" in e["selector"] for e in _entries(got))
        own = json.loads(
            handler(
                {"name": "big-parent", "section": [e for e in _entries(got)
                                                   if ".part" in e["selector"]][0]["selector"]},
                task_id="t-bigparent",
            )
        )
        assert "parent prose" in own["content"]


class TestDuplicateHeadingsAreHonest:
    """B4: a repeated heading is ambiguous, not silently the first match."""

    def _skill(self, skills_home):
        body = (
            "## Alpha\n\nfirst alpha body\n\n"
            "## Beta\n\nbeta body\n\n"
            "## Alpha\n\nsecond alpha body\n"
        )
        _write_raw_skill(skills_home / "skills", "dupe-headings", body)
        return "dupe-headings"

    def test_exact_name_reports_ambiguity_and_returns_no_body(self, skills_home):
        name = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler({"name": name, "section": "Alpha"}, task_id="t-dupe")
        )
        assert got.get("section_found") is not True
        assert "content" not in got
        assert got["section_ambiguous"] is True
        assert got["load_status"] == "incomplete"
        assert "[SKILL_INCOMPLETE:" in got["notice"]
        selectors = [e["selector"] for e in _entries(got)]
        assert len(selectors) == 2, selectors

    def test_each_occurrence_selector_returns_its_own_content(self, skills_home):
        name = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler({"name": name, "section": "Alpha"}, task_id="t-dupe-each")
        )
        bodies = []
        for entry in _entries(got):
            one = json.loads(
                handler(
                    {"name": name, "section": entry["selector"]},
                    task_id="t-dupe-each",
                )
            )
            assert one["section_found"] is True
            bodies.append(one["content"])
        assert "first alpha body" in bodies[0]
        assert "second alpha body" in bodies[1]
        assert "second alpha body" not in bodies[0]
        assert "first alpha body" not in bodies[1]

    def test_a_unique_heading_still_answers_to_its_exact_text(self, skills_home):
        name = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler({"name": name, "section": "Beta"}, task_id="t-dupe-unique")
        )
        assert got["section_found"] is True
        assert "beta body" in got["content"]
        assert "[SKILL_INCOMPLETE:" not in json.dumps(got)

    def test_the_index_keeps_duplicate_occurrences_separate(self, skills_home):
        name = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler({"name": name, "section": "No Such Heading"}, task_id="t-dupe-idx")
        )
        headings = [e["heading"] for e in _entries(got)]
        assert headings.count("Alpha") == 2
        selectors = [e["selector"] for e in _entries(got)]
        assert len(selectors) == len(set(selectors))


# ── Linked-file continuation (terminal review, 2026-08-23) ──────────────
# A linked file is a DIFFERENT document from SKILL.md with its own heading
# index. A receipt for one that does not carry file_path sends the model back
# to SKILL.md, where the same heading text and the same #n both resolve -- so
# the answer is a confident, unmarked section of the WRONG file. These tests
# follow only the call literal the receipt itself hands the model.

_FOLLOW_RE = re.compile(
    r'skill_view\(name="([^"]+)"(?:,\s*file_path="([^"]+)")?,\s*section='
)

MAIN_OVERVIEW = "MAIN-OVERVIEW-BODY"
MAIN_PROCEDURE = "MAIN-PROCEDURE-BODY"
LINKED_OVERVIEW = "LINKED-OVERVIEW-BODY"
LINKED_OMEGA = "LINKED-OMEGA-BODY"
BIG_FILE = "references/big.md"


def _follow_the_notice(payload):
    """The call the receipt literally tells the model to make next.

    Reads ONLY the actionable call literal out of ``notice``: that string is
    all the model has, so anything this helper infers from elsewhere in the
    payload would be evidence the model never gets.
    """
    notice = payload.get("notice") or ""
    match = _FOLLOW_RE.search(notice)
    assert match, f"no actionable skill_view call in notice: {notice!r}"
    args = {"name": match.group(1)}
    if match.group(2):
        args["file_path"] = match.group(2)
    return args


def _selector_for(payload, heading):
    """The selector this payload's own index gives for *heading*."""
    hits = [e for e in _entries(payload) if e["heading"] == heading]
    assert len(hits) == 1, f"{heading!r} not uniquely advertised: {_entries(payload)}"
    return hits[0]["selector"]


class TestLinkedFileContinuation:
    """Requirements 1-4: a linked-file continuation stays on the linked file."""

    def _skill(self, skills_home):
        """SKILL.md and references/big.md that collide on BOTH grammars.

        Heading text: 'Overview' is a heading in each file.
        Numeric: '#3' is 'Procedure' in SKILL.md and 'Omega' in the linked
        file -- unrelated positions, which is the normal case for two
        different documents, not a contrived one.
        """
        skills = skills_home / "skills"
        _write_raw_skill(
            skills,
            "linked-skill",
            f"# linked-skill\n\n## Overview\n\n{MAIN_OVERVIEW}\n\n"
            f"## Procedure\n\n{MAIN_PROCEDURE}\n",
        )
        big = skills / "linked-skill" / BIG_FILE
        big.parent.mkdir(parents=True, exist_ok=True)
        big.write_text(
            f"# Reference\n\nreference intro\n\n"
            f"## Overview\n\n{LINKED_OVERVIEW}\n\n"
            f"## Omega\n\n{LINKED_OMEGA}\n\n"
            f"## Bulk\n\n" + ("filler line for the linked reference file\n" * 3000),
            encoding="utf-8",
        )
        return "linked-skill"

    def _receipt(self, skills_home, task_id):
        name = self._skill(skills_home)
        raw, delivered = _view_through_executor(
            name, task_id, file_path=BIG_FILE
        )
        assert len(raw) > DEFAULT_BUDGET.resolve_threshold("skill_view"), (
            "fixture must trip the per-result threshold"
        )
        receipt = json.loads(delivered)
        assert receipt["load_status"] == "incomplete"
        assert receipt["file"] == BIG_FILE
        return name, receipt

    def test_the_receipt_instructs_a_continuation_that_keeps_file_path(
        self, skills_home
    ):
        _name, receipt = self._receipt(skills_home, "t-linked-marker")
        assert f'file_path="{BIG_FILE}"' in receipt["notice"], (
            "the only actionable instruction in the receipt drops file_path"
        )
        assert _follow_the_notice(receipt).get("file_path") == BIG_FILE

    def test_a_colliding_heading_returns_the_linked_file_not_the_main_skill(
        self, skills_home
    ):
        """Terminal-review failure 1: 'Overview' is in both files."""
        _name, receipt = self._receipt(skills_home, "t-linked-collide")
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler(
                {**_follow_the_notice(receipt), "section": "Overview"},
                task_id="t-linked-collide",
            )
        )
        assert got["section_found"] is True
        assert LINKED_OVERVIEW in got["content"]
        assert MAIN_OVERVIEW not in got["content"], (
            "the receipt's own instruction resolved against SKILL.md"
        )
        assert got["file"] == BIG_FILE

    def test_a_numeric_selector_is_scoped_to_the_linked_files_own_index(
        self, skills_home
    ):
        """Terminal-review failure 2: #3 means different sections per file."""
        _name, receipt = self._receipt(skills_home, "t-linked-numeric")
        selector = _selector_for(receipt, "Omega")
        assert selector == "#3", f"fixture must collide on #3, got {selector}"
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler(
                {**_follow_the_notice(receipt), "section": selector},
                task_id="t-linked-numeric",
            )
        )
        assert got["section_found"] is True
        assert LINKED_OMEGA in got["content"]
        for sentinel in (MAIN_OVERVIEW, MAIN_PROCEDURE):
            assert sentinel not in got["content"], (
                f"advertised {selector} as 'Omega' and answered out of SKILL.md, "
                f"whose {selector} is an unrelated section"
            )

    def test_every_advertised_selector_resolves_inside_the_linked_file(
        self, skills_home
    ):
        _name, receipt = self._receipt(skills_home, "t-linked-all")
        handler = registry.get_entry("skill_view").handler
        follow = _follow_the_notice(receipt)
        for entry in _entries(receipt):
            got = json.loads(
                handler(
                    {**follow, "section": entry["selector"]},
                    task_id="t-linked-all",
                )
            )
            assert got["file"] == BIG_FILE, entry
            assert MAIN_OVERVIEW not in json.dumps(got), entry
            assert MAIN_PROCEDURE not in json.dumps(got), entry

    def test_a_section_answer_keeps_file_path_in_its_continuation(
        self, skills_home
    ):
        """Requirement 2: the hop AFTER a successful section stays on the file."""
        name = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler(
                {"name": name, "file_path": BIG_FILE, "section": "Omega"},
                task_id="t-linked-answer",
            )
        )
        assert got["section_found"] is True
        assert LINKED_OMEGA in got["content"]
        assert f'file_path="{BIG_FILE}"' in got["notice"]
        assert _follow_the_notice(got)["file_path"] == BIG_FILE

    def test_a_navigation_answer_keeps_file_path_in_its_continuation(
        self, skills_home
    ):
        name = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler(
                {"name": name, "file_path": BIG_FILE, "section": "No Such Heading"},
                task_id="t-linked-nav",
            )
        )
        assert got["section_found"] is False
        assert f'file_path="{BIG_FILE}"' in got["notice"]
        assert _follow_the_notice(got)["file_path"] == BIG_FILE

    def test_an_oversized_linked_section_keeps_file_path_in_its_continuation(
        self, skills_home
    ):
        """The no-fragment path for a section too large to return whole."""
        name = self._skill(skills_home)
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler(
                {"name": name, "file_path": BIG_FILE, "section": "Bulk"},
                task_id="t-linked-oversized",
            )
        )
        assert got.get("section_found") is not True
        assert "content" not in got
        assert f'file_path="{BIG_FILE}"' in got["notice"]
        follow = _follow_the_notice(got)
        assert follow["file_path"] == BIG_FILE
        part = json.loads(
            handler(
                {**follow, "section": _entries(got)[0]["selector"]},
                task_id="t-linked-oversized",
            )
        )
        assert part["file"] == BIG_FILE
        assert f'file_path="{BIG_FILE}"' in part["notice"]

    def test_the_marker_still_reports_the_skill_name_it_always_did(
        self, skills_home
    ):
        """Requirement 5: marker detection and name extraction do not drift."""
        import tools.skill_delivery as st

        name, receipt = self._receipt(skills_home, "t-linked-detect")
        rendered = json.dumps(receipt)
        assert st.has_skill_incomplete_marker(rendered)
        assert st.extract_incomplete_skill_names(rendered) == [name]


class TestMainSkillContinuationIsUnchanged:
    """Requirement 5: nothing above may alter the main-SKILL.md path."""

    def test_a_main_skill_receipt_instructs_a_call_with_no_file_path(
        self, skills_home
    ):
        _raw, delivered = _view_through_executor("oversized-skill", "t-main-shape")
        receipt = json.loads(delivered)
        assert "file_path=" not in receipt["notice"]
        assert _follow_the_notice(receipt) == {"name": "oversized-skill"}

    def test_a_main_skill_section_answer_instructs_no_file_path(self, skills_home):
        handler = registry.get_entry("skill_view").handler
        got = json.loads(
            handler(
                {"name": "oversized-skill", "section": "Final Governing Rule"},
                task_id="t-main-answer",
            )
        )
        assert got["section_found"] is True
        assert "file_path=" not in got["notice"]


def test_json_escaped_parts_can_all_pass_through_the_real_result_budget(skills_home):
    body = '# Escaped\n' + '\x01' * 80_000 + '\nFINAL RULE\n'
    _write_raw_skill(skills_home / 'skills', 'escaped', body)
    _, receipt = _view_through_executor('escaped', 'escaped-test')
    handler = registry.get_entry('skill_view').handler
    queue = list(json.loads(receipt)['sections'])
    recovered = []
    seen = set()
    while queue:
        selector = queue.pop(0)['selector']
        if selector in seen:
            continue
        seen.add(selector)
        raw, delivered = _view_through_executor('escaped', 'escaped-test', section=selector)
        payload = json.loads(delivered)
        if 'content' in payload:
            assert delivered == raw
            recovered.append(payload['content'])
        else:
            queue.extend(payload.get('sections', []))
    assert ''.join(recovered).endswith(body)


def test_aggregate_spill_of_a_part_retries_the_original_selector(skills_home):
    from tools.tool_result_storage import enforce_turn_budget
    from tools.budget_config import BudgetConfig
    _write_raw_skill(skills_home / 'skills', 'retry-part', '# First\none\n# Second\n' + 'x' * 50_000)
    raw, _ = _view_through_executor('retry-part', 'part-test', section='#2.part2')
    messages = [{'role': 'tool', 'name': 'skill_view', 'tool_call_id': 'part', 'content': raw},
                {'role': 'tool', 'name': 'other', 'tool_call_id': 'other', 'content': 'y' * 10_000}]
    enforce_turn_budget(messages, config=BudgetConfig(turn_budget=15_000))
    payload = json.loads(messages[0]['content'])
    assert payload['sections'][0]['selector'] == '#2.part2'
    again, delivered = _view_through_executor('retry-part', 'part-test', section=payload['sections'][0]['selector'])
    assert again == delivered == raw
