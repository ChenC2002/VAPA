from dataclasses import replace

import pytest

from vapa.actions import make_action
from vapa.demo import tiny_episode
from vapa.environment.state_manager import StateManager
from vapa.evaluation.scoring import TaskScorer
from vapa.inference import factory_identity
from vapa.policies.heuristic import ScriptedPolicy
from vapa.rollouts import RolloutRunner
from vapa.schemas import ActionKind, ReturnCode


def scorer(**changes):
    rule = {
        "method": "numeric",
        "grounding": "exact_references",
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.05,
        "missing_value": "NOTRECORDED",
        "allow_missing_without_evidence": False,
    }
    rule.update(changes)
    return TaskScorer(
        {"schema_version": "vapa-task-scorer-v1", "task_rules": {"latest_laboratory_value": rule}}
    )


def test_grounded_numeric_scoring_and_invalid_predictions():
    episode = tiny_episode()
    score = scorer().evaluation_scorer()
    assert score(episode, 6.4, ("e#1002",)) == 1
    assert score(episode, 6.5, ("e#1002",)) == 1
    for answer, evidence in [
        (7, ("e#1002",)),
        (True, ("e#1002",)),
        ("6.4", ("e#1002",)),
        (6.4, ()),
        (6.4, ("e#future",)),
        (6.4, ("e#1001",)),
        (6.4, ("e#1002", "e#1002")),
        (6.4, ("unknown",)),
    ]:
        assert score(episode, answer, evidence) == -1


def test_missing_gold_and_unknown_tasks_are_not_negative_measurements():
    episode = tiny_episode()
    with pytest.raises(ValueError, match="missing gold"):
        scorer()(replace(episode, gold_answer=None), None, ())
    unknown = replace(episode, task=replace(episode.task, metadata={}))
    with pytest.raises(ValueError, match="no declared"):
        scorer()(unknown, 6.4, ("e#1002",))
    with pytest.raises(ValueError, match="reference evidence"):
        scorer().evaluation_scorer()(
            replace(episode, reference_evidence=("e#future",)), 6.4, ("e#1002",)
        )


@pytest.mark.parametrize("evidence", [(), ("e#1002",), ("unknown",), ("e#future",)])
def test_terminal_training_reward_does_not_include_grounding(evidence):
    episode = tiny_episode()
    assert scorer()(episode, 6.4, evidence) == 1
    assert scorer()(episode, 7.0, evidence) == -1


def missing_episode():
    return replace(tiny_episode(), events=(), gold_answer="NOTRECORDED", reference_evidence=())


def missing_rollout(episode, *, query=True, field="hba1c", window="last 90d", evaluation=True):
    def choose(observation, rng):
        if query and observation.last_return is None:
            return make_action(ActionKind.QUERY_FIELD, field=field, window=window)
        return make_action(ActionKind.ANSWER, prediction="NOTRECORDED", evidence=[])

    score = scorer(allow_missing_without_evidence=True)
    runner = RolloutRunner(
        StateManager, outcome_scorer=score.evaluation_scorer() if evaluation else score
    )
    return runner.run(episode, ScriptedPolicy(choose), rollout_id="missing", seed=7)


def test_missing_sentinel_requires_declared_exemption_and_executed_query():
    episode = missing_episode()
    assert scorer()(episode, "NOTRECORDED", ()) == 1
    assert scorer().evaluation_scorer()(episode, "NOTRECORDED", ()) == -1
    score = scorer(allow_missing_without_evidence=True).evaluation_scorer()
    with pytest.raises(ValueError, match="executed tool trace"):
        score(episode, "NOTRECORDED", ())
    assert missing_rollout(episode, query=False).outcome_reward == -1
    assert missing_rollout(episode, query=False, evaluation=False).outcome_reward == 1
    rollout = missing_rollout(episode)
    assert rollout.turns[0].tool_return.code is ReturnCode.NOT_RECORDED
    assert rollout.outcome_reward == 1
    assert score(episode, "NOTRECORDED", ("e#future",)) == -1


@pytest.mark.parametrize("query", [dict(field="glucose"), dict(window="last 30d")])
def test_unrelated_absence_return_cannot_ground_missingness(query):
    assert missing_rollout(missing_episode(), **query).outcome_reward == -1


def test_out_of_window_return_confirms_no_qualifying_event():
    episode = replace(missing_episode(), events=(tiny_episode().events[0],))
    rollout = missing_rollout(episode)
    assert rollout.turns[0].tool_return.code is ReturnCode.OUT_OF_WINDOW
    assert rollout.outcome_reward == 1


def test_observed_value_cannot_be_claimed_missing():
    episode = replace(tiny_episode(), gold_answer="NOTRECORDED", reference_evidence=())
    assert missing_rollout(episode).outcome_reward == -1


def test_trace_cannot_be_reused_for_a_different_task():
    episode = missing_episode()
    trace = missing_rollout(episode).turns
    other = replace(episode, task=replace(episode.task, instance_id="another-instance"))
    with pytest.raises(ValueError, match="different task"):
        scorer(allow_missing_without_evidence=True).evaluation_scorer().score_with_trace(
            other, "NOTRECORDED", (), tuple(trace)
        )


def test_multifield_missingness_requires_task_adapter():
    episode = missing_episode()
    episode = replace(episode, task=replace(episode.task, requested_fields=("hba1c", "glucose")))
    with pytest.raises(ValueError, match="task-specific scorer"):
        missing_rollout(episode)


def test_exact_scoring_is_type_strict_and_manifest_bound():
    episode = replace(tiny_episode(), gold_answer=1)
    exact = scorer(method="exact", relative_tolerance=0)
    assert exact(episode, 1, ("e#1002",)) == 1
    assert exact(episode, True, ("e#1002",)) == -1
    assert factory_identity(scorer()) != factory_identity(scorer(relative_tolerance=0))
    assert factory_identity(scorer()) != factory_identity(scorer().evaluation_scorer())
    assert factory_identity(scorer().evaluation_scorer()) != factory_identity(
        scorer(relative_tolerance=0).evaluation_scorer()
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(relative_tolerance=True),
        dict(grounding="guess"),
        dict(method="exact"),
        dict(absolute_tolerance=-1),
        dict(extra_field=1),
    ],
)
def test_bad_rules_fail_closed(change):
    with pytest.raises(ValueError):
        scorer(**change)
