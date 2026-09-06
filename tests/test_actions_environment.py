from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from vapa.actions import ActionParseError, format_action, make_action, parse_action
from vapa.environment import CalculatorRegistry, CalculatorSpec, InputSpec, StateManager
from vapa.prompts import render_tool_return
from vapa.rollouts import exact_outcome_scorer
from vapa.schemas import (
    ActionKind,
    Domain,
    Episode,
    RecordEvent,
    ReturnCode,
    TaskSpec,
    TimeWindow,
    ToolReturn,
)

CUTOFF = datetime(2025, 1, 10, tzinfo=UTC)


def _event(
    pointer: str,
    timestamp: str,
    field: str,
    value: object,
    *,
    unit: str | None = None,
    text: str | None = None,
) -> RecordEvent:
    return RecordEvent(
        pointer=pointer,
        patient_id="patient-1",
        timestamp=timestamp,
        domain=Domain.LAB,
        field=field,
        value=value,
        unit=unit,
        text=text,
        source="synthetic",
    )


def _episode(*events: RecordEvent, window: str = "all") -> Episode:
    task = TaskSpec(
        instance_id="instance-1",
        patient_id="patient-1",
        instruction="Return the latest HbA1c.",
        cutoff=CUTOFF,
        family="latest_lab",
        requested_fields=("hba1c",),
        requested_window=TimeWindow.parse(window),
    )
    return Episode(task=task, events=tuple(events), gold_answer=6.4)


def _memory_item(item_id: str, field: str, value: float, pointer: str) -> dict[str, object]:
    return {
        "id": item_id,
        "field": field,
        "value": value,
        "unit": "%",
        "status": "current",
        "validity_scope": "all",
        "evidence_pointers": [pointer],
    }


def test_action_parser_handles_nested_arguments_and_requires_action_on_final_line():
    output = """I will bind both inputs without flattening the nested payload.
Calculate(bmi, {"weight_kg":{"pointer":"e#1"},"height_m":["e#2","fallback, (x)"],"strict":true})
"""
    action = parse_action(output)

    assert action.kind is ActionKind.CALCULATE
    assert action.arguments["calculator"] == "bmi"
    assert action.arguments["bindings"] == {
        "weight_kg": {"pointer": "e#1"},
        "height_m": ["e#2", "fallback, (x)"],
        "strict": True,
    }
    assert parse_action(format_action(action)) == action

    with pytest.raises(ActionParseError, match="final line"):
        parse_action("Answer(6.4, [e#1])\nThis text is forbidden after the action.")
    with pytest.raises(ActionParseError, match="unterminated|unbalanced"):
        parse_action('Calculate(bmi, {"weight": ["e#1"})')


@pytest.mark.parametrize("value", ["true", "false", "null", "none", "1", "-2.5", "1e3"])
def test_action_parser_and_formatter_preserve_keyword_and_numeric_strings(value: str):
    action = parse_action(f"Answer({json.dumps(value)}, [])")

    assert action.arguments["prediction"] == value
    assert parse_action(format_action(action)) == action


def test_action_parser_accepts_documented_pointer_lists_and_rejects_non_json_values():
    action = parse_action("Answer(ok, [e#1,e#2])")
    assert action.arguments["evidence"] == ["e#1", "e#2"]

    with pytest.raises(ActionParseError, match="non-finite"):
        parse_action("Answer(1e9999, [])")
    with pytest.raises(ActionParseError, match="invalid structured"):
        parse_action("Answer(ok, [{1, 2}])")
    with pytest.raises(ValueError, match="finite"):
        make_action(ActionKind.ANSWER, prediction=float("inf"), evidence=[])


def test_record_access_is_cutoff_restricted_for_query_and_retrieval():
    episode = _episode(
        _event("e#old", "2024-12-01T00:00:00Z", "hba1c", 7.2, unit="%"),
        _event("e#recent", "2025-01-05T00:00:00Z", "hba1c", 6.4, unit="%"),
        _event(
            "e#future",
            "2025-01-11T00:00:00Z",
            "hba1c",
            99.0,
            unit="%",
            text="POST-CUTOFF SECRET",
        ),
    )
    manager = StateManager(episode)

    query = manager.step(
        make_action(ActionKind.QUERY_FIELD, field="hba1c", window="all")
    ).tool_return
    assert query.code is ReturnCode.FOUND
    assert query.value == 6.4
    assert query.evidence_pointers == ("e#recent",)

    retrieved = manager.step(
        make_action(ActionKind.RETRIEVE, query="hba1c", window="all", domain="lab")
    ).tool_return
    assert set(retrieved.evidence_pointers) == {"e#old", "e#recent"}
    assert "e#future" not in retrieved.evidence_pointers
    assert all(event.timestamp <= CUTOFF for event in retrieved.events)


def test_curriculum_action_mask_controls_observation_and_transition_legality():
    manager = StateManager(_episode())
    manager.set_action_mask({ActionKind.RETRIEVE, ActionKind.ANSWER})
    assert manager.observation().legal_actions == (ActionKind.RETRIEVE, ActionKind.ANSWER)

    rejected = manager.step(make_action(ActionKind.QUERY_FIELD, field="hba1c", window="all"))
    assert not rejected.accepted
    assert rejected.cost == 1
    assert rejected.tool_return.code is ReturnCode.REJECTED

    with pytest.raises(ValueError, match="Answer"):
        manager.set_action_mask({ActionKind.RETRIEVE})
    manager.set_action_mask(None)
    assert ActionKind.QUERY_FIELD in manager.observation().legal_actions


def test_scratchpad_is_one_turn_while_memory_preserves_explicitly_written_evidence():
    episode = _episode(_event("e#1", "2025-01-05T00:00:00Z", "hba1c", 6.4, unit="%"))
    manager = StateManager(episode)

    queried = manager.step(make_action(ActionKind.QUERY_FIELD, field="hba1c", window="all"))
    assert queried.observation.last_return is queried.tool_return
    assert queried.observation.last_return.value == 6.4

    written = manager.step(
        make_action(
            ActionKind.UPDATE_MEMORY,
            item=_memory_item("a1c", "hba1c", 6.4, "e#1"),
        )
    )
    assert written.tool_return.code is ReturnCode.OK
    assert written.observation.last_return is written.tool_return
    assert written.observation.last_return.value is None
    assert written.observation.last_return.events == ()
    assert written.observation.memory[0].value == 6.4
    assert written.observation.memory[0].evidence_pointers == ("e#1",)


def test_full_memory_rejects_new_item_without_silent_eviction():
    episode = _episode(
        _event("e#1", "2025-01-05T00:00:00Z", "hba1c", 6.4),
        _event("e#2", "2025-01-06T00:00:00Z", "glucose", 101.0),
    )
    manager = StateManager(
        episode,
        memory_capacity=1,
        action_budget=3,
        turn_cap=4,
    )
    manager.step(make_action(ActionKind.RETRIEVE, query="", window="all", domain="lab"))
    first_action = make_action(
        ActionKind.UPDATE_MEMORY,
        item=_memory_item("first", "hba1c", 6.4, "e#1"),
    )
    assert manager.step(first_action).accepted
    original = manager.memory[0]
    assert ActionKind.UPDATE_MEMORY in manager.legal_actions()

    rejected = manager.step(
        make_action(
            ActionKind.UPDATE_MEMORY,
            item=_memory_item("second", "glucose", 101.0, "e#2"),
        )
    )
    assert rejected.tool_return.code is ReturnCode.REJECTED
    assert not rejected.accepted
    assert rejected.cost == 1
    assert manager.memory == (original,)
    assert manager.memory[0].item_id == "first"


def test_compact_history_excludes_values_text_pointers_and_memory_contents():
    episode = _episode(
        _event(
            "e#private",
            "2025-01-05T00:00:00Z",
            "hba1c",
            6.4,
            text="PRIVATE CLINICAL TEXT",
        )
    )
    manager = StateManager(episode)
    manager.step(make_action(ActionKind.QUERY_FIELD, field="hba1c", window="all"))
    observation = manager.step(
        make_action(
            ActionKind.UPDATE_MEMORY,
            item=_memory_item("secret-item", "hba1c", 6.4, "e#private"),
        )
    ).observation

    serialized_history = json.dumps([entry.to_dict() for entry in observation.history])
    assert "hba1c:all" in serialized_history
    for private_value in ("6.4", "PRIVATE CLINICAL TEXT", "e#private", "secret-item"):
        assert private_value not in serialized_history


def test_logged_prefix_replay_reconstructs_identical_observation():
    episode = _episode(_event("e#1", "2025-01-05T00:00:00Z", "hba1c", 6.4, unit="%"))
    manager = StateManager(episode)
    actions = [
        make_action(ActionKind.QUERY_FIELD, field="hba1c", window="all"),
        make_action(
            ActionKind.UPDATE_MEMORY,
            item=_memory_item("a1c", "hba1c", 6.4, "e#1"),
        ),
        make_action(
            ActionKind.MARK_STATUS,
            item_id="a1c",
            status="stale",
            scope="historical",
            why="synthetic replay check",
        ),
    ]
    for action in actions:
        manager.step(action)
    expected = manager.observation()

    replayed = StateManager(episode).replay(actions)
    replayed_again = StateManager(episode).replay(actions)
    assert replayed.canonical_bytes() == expected.canonical_bytes()
    assert replayed.state_hash == expected.state_hash == replayed_again.state_hash


def test_query_field_distinguishes_all_four_record_return_codes():
    episode = _episode(
        _event("e#found", "2025-01-09T12:00:00Z", "hba1c", 6.4),
        _event("e#old", "2025-01-01T00:00:00Z", "creatinine", 1.1),
        _event("e#tie-a", "2025-01-08T00:00:00Z", "sodium", 130.0),
        _event("e#tie-b", "2025-01-08T00:00:00Z", "sodium", 140.0),
    )
    manager = StateManager(episode, action_budget=6, turn_cap=7)

    def query(field: str, window: str = "all") -> ReturnCode:
        result = manager.step(make_action(ActionKind.QUERY_FIELD, field=field, window=window))
        return result.tool_return.code

    assert query("hba1c") is ReturnCode.FOUND
    assert query("potassium") is ReturnCode.NOT_RECORDED
    assert query("creatinine", "last 1d") is ReturnCode.OUT_OF_WINDOW
    assert query("sodium") is ReturnCode.AMBIGUOUS


def test_calculator_evaluates_restricted_formula_and_rejects_code_execution_syntax():
    weight = InputSpec("weight_kg", ("weight",), unit="kg")
    height = InputSpec("height_m", ("height",), unit="m")
    bmi = CalculatorSpec(
        name="bmi",
        inputs=(weight, height),
        expression="weight_kg / (height_m ** 2)",
        output_unit="kg/m2",
        output_decimals=2,
    )
    episode = _episode(
        _event("e#w", "2025-01-05T00:00:00Z", "weight", 70.0, unit="kg"),
        _event("e#h", "2025-01-05T00:00:00Z", "height", 1.75, unit="m"),
    )
    manager = StateManager(episode, calculators=CalculatorRegistry((bmi,)))
    manager.step(make_action(ActionKind.RETRIEVE, query="", window="all", domain="lab"))
    result = manager.step(
        make_action(
            ActionKind.CALCULATE,
            calculator="bmi",
            bindings={"weight_kg": "e#w", "height_m": "e#h"},
        )
    ).tool_return
    assert result.code is ReturnCode.VALUE
    assert result.value == 22.86
    assert result.evidence_pointers == ("e#w", "e#h")

    with pytest.raises(ValueError, match="unsupported|unknown"):
        CalculatorSpec(
            name="unsafe",
            inputs=(weight,),
            expression="__import__('os').system('id')",
        )
    with pytest.raises(ValueError, match="unsupported"):
        CalculatorSpec(
            name="introspection",
            inputs=(weight,),
            expression="weight_kg.__class__",
        )


def test_calculator_can_bind_only_visible_evidence_and_requires_source_units():
    value_input = InputSpec("x", ("weight",), unit="kg")
    identity = CalculatorSpec("identity", (value_input,), "x", output_unit="kg")
    event = _event("e#weight", "2025-01-05T00:00:00Z", "weight", 70.0, unit="kg")
    episode = _episode(event)
    hidden = StateManager(episode, calculators=CalculatorRegistry((identity,)))

    unresolved = hidden.step(
        make_action(
            ActionKind.CALCULATE,
            calculator="identity",
            bindings={"x": "e#weight"},
        )
    ).tool_return
    assert unresolved.code is ReturnCode.UNRESOLVED

    visible = StateManager(episode, calculators=CalculatorRegistry((identity,)))
    visible.step(make_action(ActionKind.RETRIEVE, query="weight", window="all", domain="lab"))
    resolved = visible.step(
        make_action(
            ActionKind.CALCULATE,
            calculator="identity",
            bindings={"x": "e#weight"},
        )
    ).tool_return
    assert resolved.code is ReturnCode.VALUE
    assert resolved.value == 70.0

    missing_unit_episode = _episode(_event("e#unitless", "2025-01-05T00:00:00Z", "weight", 70.0))
    unitless = StateManager(missing_unit_episode, calculators=CalculatorRegistry((identity,)))
    unitless.step(make_action(ActionKind.RETRIEVE, query="weight", window="all", domain="lab"))
    ambiguous = unitless.step(
        make_action(
            ActionKind.CALCULATE,
            calculator="identity",
            bindings={"x": "e#unitless"},
        )
    ).tool_return
    assert ambiguous.code is ReturnCode.AMBIGUOUS
    assert "source unit is required" in ambiguous.message


def test_memory_writes_require_visible_pointers_and_status_changes_use_mark_status():
    episode = _episode(_event("e#1", "2025-01-05T00:00:00Z", "hba1c", 6.4, unit="%"))
    manager = StateManager(episode)
    current = _memory_item("a1c", "hba1c", 6.4, "e#1")

    hidden = manager.step(make_action(ActionKind.UPDATE_MEMORY, item=current))
    assert hidden.tool_return.code is ReturnCode.REJECTED
    assert not manager.memory

    manager.step(make_action(ActionKind.QUERY_FIELD, field="hba1c", window="all"))
    assert manager.step(make_action(ActionKind.UPDATE_MEMORY, item=current)).accepted
    replacement = dict(current, status="stale", validity_scope="historical")
    rejected = manager.step(make_action(ActionKind.UPDATE_MEMORY, item=replacement))
    assert rejected.tool_return.code is ReturnCode.REJECTED
    assert manager.memory[0].status.value == "current"
    assert manager.memory[0].validity_scope == "all"

    changed = manager.step(
        make_action(
            ActionKind.MARK_STATUS,
            item_id="a1c",
            status="stale",
            scope="historical",
            why="synthetic status update",
        )
    )
    assert changed.accepted
    assert manager.memory[0].status.value == "stale"
    assert manager.memory[0].validity_scope == "historical"


def test_answer_requires_visible_pointers_and_invalid_terminal_answers_cannot_match_none():
    episode = _episode(_event("e#1", "2025-01-05T00:00:00Z", "hba1c", 6.4, unit="%"))
    hidden = StateManager(episode)
    rejected = hidden.step(make_action(ActionKind.ANSWER, prediction=6.4, evidence=["e#1"]))
    assert rejected.terminated
    assert not rejected.accepted
    assert rejected.tool_return.code is ReturnCode.REJECTED
    none_gold = Episode(episode.task, episode.events, gold_answer=None)
    assert exact_outcome_scorer(none_gold, None, ()) == -1.0
    assert exact_outcome_scorer(none_gold, hidden.prediction, hidden.answer_evidence) == -1.0

    visible = StateManager(episode)
    visible.step(make_action(ActionKind.QUERY_FIELD, field="hba1c", window="all"))
    answered = visible.step(make_action(ActionKind.ANSWER, prediction=6.4, evidence=["e#1"]))
    assert answered.accepted
    assert answered.tool_return.code is ReturnCode.ANSWERED

    missing = StateManager(episode)
    failed = missing.step(make_action(ActionKind.ANSWER, prediction=None, evidence=[]))
    assert failed.terminated
    assert not failed.accepted
    assert exact_outcome_scorer(none_gold, missing.prediction, missing.answer_evidence) == -1.0


def test_malformed_answer_and_non_answer_at_required_answer_state_terminate():
    episode = _episode(_event("e#1", "2025-01-05T00:00:00Z", "hba1c", 6.4))
    malformed = StateManager(episode)
    failed = malformed.step_text("Answer(6.4)")
    assert failed.terminated
    assert not failed.accepted
    assert malformed.prediction is not None

    trailing = StateManager(episode)
    trailed = trailing.step_text("Answer(6.4, [])\ntrailing text")
    assert trailed.terminated
    assert not trailed.accepted

    multiple = StateManager(episode)
    multi_action = multiple.step_text("Answer(6.4, [])\nQueryField(hba1c, all)")
    assert multi_action.terminated
    assert not multi_action.accepted
    assert multiple.prediction is not None

    exhausted = StateManager(episode, action_budget=1, turn_cap=2)
    exhausted.step(make_action(ActionKind.QUERY_FIELD, field="hba1c", window="all"))
    required = exhausted.step(make_action(ActionKind.QUERY_FIELD, field="hba1c", window="all"))
    assert required.terminated
    assert not required.accepted
    assert exhausted.prediction is not None


def test_prompt_tool_events_exclude_patient_ids_free_text_and_source_metadata():
    event = _event(
        "e#private",
        "2025-01-05T00:00:00Z",
        "hba1c",
        6.4,
        unit="%",
        text="PRIVATE CLINICAL TEXT",
    )
    rendered = render_tool_return(
        ToolReturn(
            ReturnCode.FOUND,
            value=event.value,
            unit=event.unit,
            evidence_pointers=(event.pointer,),
            events=(event,),
        )
    )
    event_payload = json.loads(rendered)["events"][0]

    assert set(event_payload) == {
        "pointer",
        "timestamp",
        "domain",
        "field",
        "value",
        "unit",
        "code",
    }
    assert event_payload["pointer"] == "e#private"
    assert "patient-1" not in rendered
    assert "PRIVATE CLINICAL TEXT" not in rendered
    assert "synthetic" not in rendered
