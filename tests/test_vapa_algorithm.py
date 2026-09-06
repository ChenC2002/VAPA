from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from vapa.actions import make_action
from vapa.policies.base import PolicyDecision
from vapa.rollouts import Rollout, Turn
from vapa.schemas import (
    ActionKind,
    Domain,
    Episode,
    MemoryItem,
    MemoryStatus,
    Observation,
    RecordEvent,
    ReturnCode,
    TaskSpec,
    ToolReturn,
)
from vapa.training.advantages import (
    assign_step_advantages,
    assign_trajectory_advantages,
    credit_return,
)
from vapa.training.groups import GroupTier, StepGroup, TurnRef, assign_step_groups
from vapa.training.replay import select_fork_anchors
from vapa.verifiers import (
    Predicate,
    VerifierCatalog,
    VerifierContext,
    VerifierFamily,
    VerifierTrace,
)

TASK = TaskSpec(
    instance_id="instance-1",
    patient_id="patient-1",
    instruction="Return the requested value.",
    cutoff=datetime(2025, 1, 10, tzinfo=UTC),
    family="synthetic",
)
EPISODE = Episode(task=TASK, events=(), gold_answer="ok")


@pytest.mark.parametrize("index", [-1, True, 0.5])
def test_turn_reference_cannot_index_an_unrelated_turn(index: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        TurnRef("rollout", index)


@pytest.mark.parametrize(
    "updates",
    [
        {"epsilon": float("nan")},
        {"beta": float("inf")},
        {"cost_weight": -1.0},
        {"gamma": float("nan")},
    ],
)
def test_step_advantages_reject_invalid_coefficients(updates: dict) -> None:
    rollout = _rollout("base", [_turn(0, _observation(None))])
    with pytest.raises(ValueError):
        assign_step_advantages([rollout], [], [], **updates)


@pytest.mark.parametrize("index", [-1, True, 1])
def test_credit_return_cannot_silently_use_wrong_suffix(index: object) -> None:
    rollout = _rollout("base", [_turn(0, _observation(None))])
    with pytest.raises(ValueError, match="existing turn"):
        credit_return(rollout, index, process_weight=0.05, cost_weight=0.02, gamma=1.0)


def test_local_comparison_cannot_mix_episode_occurrences() -> None:
    first = _rollout("first", [_turn(0, _observation(None))])
    second = replace(_rollout("second", [_turn(0, _observation(None))]), instance_id="other")
    group = StepGroup("invalid", GroupTier.TURN_INDEX, (TurnRef("first", 0), TurnRef("second", 0)))
    with pytest.raises(ValueError, match="cannot mix episode occurrences"):
        assign_step_advantages([first, second], [], [group])


def _memory(count: int) -> tuple[MemoryItem, ...]:
    return tuple(
        MemoryItem(
            item_id=f"m{index}",
            field=f"field_{index}",
            value=float(index),
            status=MemoryStatus.CURRENT,
            validity_scope="all",
            evidence_pointers=(f"e#{index}",),
        )
        for index in range(count)
    )


def _observation(
    tag: str | None,
    *,
    budget: int = 12,
    memory_count: int = 0,
) -> Observation:
    return Observation(
        task=TASK,
        last_return=None if tag is None else ToolReturn(ReturnCode.OK, message=tag),
        memory=_memory(memory_count),
        history=(),
        budget_remaining=budget,
        budget_cap=12,
        turn=13 - budget,
        turn_cap=16,
        memory_capacity=8,
        legal_actions=(ActionKind.QUERY_FIELD, ActionKind.ANSWER),
    )


def _action(kind: ActionKind = ActionKind.QUERY_FIELD):
    if kind is ActionKind.ANSWER:
        return make_action(ActionKind.ANSWER, prediction="ok", evidence=[])
    return make_action(ActionKind.QUERY_FIELD, field="value", window="all")


def _turn(
    index: int,
    observation: Observation,
    *,
    kind: ActionKind = ActionKind.QUERY_FIELD,
    cost: int = 0,
    process_reward: float = 0.0,
    copied_prefix: bool = False,
    loss_mask: bool = True,
    token_count: int = 1,
) -> Turn:
    action = _action(kind)
    return Turn(
        index=index,
        observation=observation,
        decision=PolicyDecision(action=action, token_count=token_count),
        action=action,
        tool_return=ToolReturn(ReturnCode.ANSWERED if kind is ActionKind.ANSWER else ReturnCode.OK),
        cost=cost,
        accepted=True,
        process_reward=process_reward,
        copied_prefix=copied_prefix,
        loss_mask=loss_mask,
    )


def _rollout(
    rollout_id: str,
    turns: list[Turn],
    *,
    outcome: float = 0.0,
    is_base: bool = True,
    parent: str | None = None,
    fork_turn: int | None = None,
) -> Rollout:
    return Rollout(
        rollout_id=rollout_id,
        instance_id=TASK.instance_id,
        turns=turns,
        is_base=is_base,
        parent_rollout_id=parent,
        fork_turn=fork_turn,
        outcome_reward=outcome,
    )


def test_verifier_abstentions_remain_in_fixed_family_denominator():
    positive = Predicate(
        name="fires",
        family=VerifierFamily.GROUNDING,
        weight=2.0,
        actions=frozenset({ActionKind.QUERY_FIELD}),
        function=lambda context, rollout, index: 1.0,
    )
    abstaining = Predicate(
        name="abstains",
        family=VerifierFamily.TEMPORAL,
        weight=1.0,
        actions=frozenset({ActionKind.QUERY_FIELD}),
        function=lambda context, rollout, index: None,
    )
    catalog = VerifierCatalog((positive, abstaining), catalog_id="test-catalog", paper_exact=False)
    rollout = _rollout("base", [_turn(0, _observation(None))])

    score = catalog.score_turn(EPISODE, rollout, 0)
    assert score.denominator == 3.0
    assert score.reward == pytest.approx(2.0 / 3.0)
    assert score.fired_count == 1
    assert [result.fired for result in score.results] == [True, False]


def test_custom_verifier_predicate_cannot_access_gold_or_post_cutoff_events():
    task = TaskSpec(
        instance_id="private-instance",
        patient_id="private-patient",
        instruction="Return the value recorded before cutoff.",
        cutoff=datetime(2025, 1, 10, tzinfo=UTC),
        family="synthetic",
        requested_fields=("value",),
    )
    admissible = RecordEvent(
        pointer="e#past",
        patient_id=task.patient_id,
        timestamp=datetime(2025, 1, 9, tzinfo=UTC),
        domain=Domain.LAB,
        field="value",
        value=1.0,
    )
    post_cutoff = RecordEvent(
        pointer="e#future-secret",
        patient_id=task.patient_id,
        timestamp=datetime(2025, 1, 11, tzinfo=UTC),
        domain=Domain.LAB,
        field="value",
        value=999.0,
    )
    episode = Episode(
        task=task,
        events=(admissible, post_cutoff),
        gold_answer="gold-secret",
        reference_evidence=(post_cutoff.pointer,),
    )
    captured: list[VerifierContext] = []
    traces: list[VerifierTrace] = []

    def probe(context: VerifierContext, trace: VerifierTrace, index: int) -> float:
        captured.append(context)
        traces.append(trace)
        return 1.0

    catalog = VerifierCatalog(
        (
            Predicate(
                name="boundary_probe",
                family=VerifierFamily.GROUNDING,
                weight=1.0,
                actions=frozenset({ActionKind.QUERY_FIELD}),
                function=probe,
            ),
        ),
        catalog_id="boundary-test",
        paper_exact=False,
    )

    private_observation = replace(_observation(None), task=task)
    score = catalog.score_turn(episode, _rollout("private", [_turn(0, private_observation)]), 0)

    assert score.reward == 1.0
    assert captured[0].task is task
    assert captured[0].admissible_events == (admissible,)
    for forbidden_attribute in ("events", "gold_answer", "reference_evidence"):
        with pytest.raises(AttributeError):
            getattr(captured[0], forbidden_attribute)
    assert not hasattr(captured[0], "__dict__")
    for forbidden_attribute in (
        "outcome_reward",
        "prediction",
        "answer_evidence",
        "rollout_id",
    ):
        with pytest.raises(AttributeError):
            getattr(traces[0], forbidden_attribute)
    for forbidden_attribute in (
        "process_reward",
        "local_advantage",
        "episode_advantage",
        "normalized_advantage",
    ):
        with pytest.raises(AttributeError):
            getattr(traces[0].turns[0], forbidden_attribute)
    assert not hasattr(traces[0], "__dict__")
    with pytest.raises(ValueError, match="post-cutoff"):
        VerifierContext(task, (post_cutoff,))
    with pytest.raises(ValueError, match="another task"):
        catalog.score_turn(episode, _rollout("wrong-task", [_turn(0, _observation(None))]), 0)
    malicious_turn = _turn(0, private_observation)
    malicious_turn.tool_return = ToolReturn(ReturnCode.FOUND, events=(post_cutoff,))
    with pytest.raises(ValueError, match="inadmissible"):
        catalog.score_turn(episode, _rollout("malicious", [malicious_turn]), 0)


def test_exact_fork_tier_preempts_collision_while_other_peers_still_collide():
    common = _observation("same-state")
    base_anchor = _rollout("base-anchor", [_turn(0, common)])
    peer_one = _rollout("peer-one", [_turn(0, common)])
    peer_two = _rollout("peer-two", [_turn(0, common)])
    sibling = _rollout(
        "sibling",
        [_turn(0, common)],
        is_base=False,
        parent="base-anchor",
        fork_turn=0,
    )

    groups = assign_step_groups(
        [base_anchor, peer_one, peer_two],
        [sibling],
    )
    exact = next(group for group in groups if group.tier is GroupTier.EXACT_FORK)
    collision = next(group for group in groups if group.tier is GroupTier.COLLISION)

    assert set(exact.members) == {
        TurnRef("base-anchor", 0),
        TurnRef("sibling", 0),
    }
    assert set(collision.members) == {
        TurnRef("peer-one", 0),
        TurnRef("peer-two", 0),
    }
    assert base_anchor.turns[0].group_tier == GroupTier.EXACT_FORK.value


def test_collision_then_turn_index_fallback_are_nonoverlapping():
    shared = _observation("shared", budget=12)
    left = _observation("left", budget=11)
    right = _observation("right", budget=11, memory_count=1)
    first = _rollout("first", [_turn(0, shared), _turn(1, left)])
    second = _rollout("second", [_turn(0, shared), _turn(1, right)])

    groups = assign_step_groups([first, second], [])
    collision = next(group for group in groups if group.tier is GroupTier.COLLISION)
    pooled = next(group for group in groups if group.tier is GroupTier.TURN_INDEX)

    assert set(collision.members) == {TurnRef("first", 0), TurnRef("second", 0)}
    assert set(pooled.members) == {TurnRef("first", 1), TurnRef("second", 1)}
    assert set(collision.members).isdisjoint(pooled.members)
    assert first.turns[1].group_tier == GroupTier.TURN_INDEX.value


def test_exact_fork_rejects_observations_that_only_share_collision_signature():
    base_observation = _observation("same-message")
    reordered_observation = replace(
        base_observation,
        legal_actions=tuple(reversed(base_observation.legal_actions)),
    )
    base = _rollout("base", [_turn(0, base_observation)])
    branch = _rollout(
        "branch",
        [_turn(0, reordered_observation)],
        is_base=False,
        parent="base",
        fork_turn=0,
    )

    assert (
        base.turns[0].observation.canonical_bytes() == branch.turns[0].observation.canonical_bytes()
    )
    assert base.turns[0].observation.exact_bytes() != branch.turns[0].observation.exact_bytes()
    with pytest.raises(ValueError, match="pre-action observation"):
        assign_step_groups([base], [branch])


def test_collision_signature_includes_structured_return_and_legal_mask():
    observation = _observation("same")
    different_return = replace(
        observation,
        last_return=ToolReturn(ReturnCode.VALUE, value=42.0, unit="MG / DL"),
    )
    different_mask = replace(observation, legal_actions=(ActionKind.ANSWER,))

    assert observation.state_hash != different_return.state_hash
    assert observation.state_hash != different_mask.state_hash


def test_equation_4_discounted_suffix_return_is_exact():
    rollout = _rollout(
        "credit",
        [
            _turn(0, _observation("t0"), cost=1, process_reward=1.0),
            _turn(1, _observation("t1", budget=11), cost=1, process_reward=0.5),
            _turn(
                2,
                _observation("t2", budget=10),
                kind=ActionKind.ANSWER,
                process_reward=-1.0,
            ),
        ],
        outcome=1.0,
    )

    assert credit_return(
        rollout,
        0,
        process_weight=0.2,
        cost_weight=0.1,
        gamma=0.5,
    ) == pytest.approx(1.0)
    assert credit_return(
        rollout,
        1,
        process_weight=0.2,
        cost_weight=0.1,
        gamma=0.5,
    ) == pytest.approx(0.9)


def test_equations_5_6_7_use_leave_one_out_and_population_standard_deviation():
    positive = _rollout("positive", [_turn(0, _observation("positive"))], outcome=1.0)
    negative = _rollout("negative", [_turn(0, _observation("negative"))], outcome=-1.0)
    group = StepGroup(
        "comparison",
        GroupTier.TURN_INDEX,
        (TurnRef("positive", 0), TurnRef("negative", 0)),
    )

    summary = assign_step_advantages(
        [positive, negative],
        [],
        [group],
        process_weight=0.0,
        cost_weight=0.0,
        beta=1.0,
        epsilon=1.0,
    )

    assert positive.turns[0].local_advantage == 2.0
    assert negative.turns[0].local_advantage == -2.0
    assert summary.local_scale == 2.0
    assert summary.episode_scales[TASK.instance_id] == 1.0
    assert positive.turns[0].episode_advantage == 0.5
    assert negative.turns[0].episode_advantage == -0.5
    assert positive.turns[0].normalized_advantage == pytest.approx(7.0 / 6.0)
    assert negative.turns[0].normalized_advantage == pytest.approx(-7.0 / 6.0)
    assert summary.local_nonzero == 2


def test_valid_tool_rejections_remain_grouped_and_loss_bearing():
    observation = _observation("rejected")
    left_turn = _turn(0, observation)
    right_turn = _turn(0, observation)
    for turn in (left_turn, right_turn):
        turn.accepted = False
        turn.tool_return = ToolReturn(ReturnCode.REJECTED, message="invalid binding")
    left = _rollout("rejected-left", [left_turn], outcome=-1.0)
    right = _rollout("rejected-right", [right_turn], outcome=1.0)

    groups = assign_step_groups([left, right], [])
    assert len(groups) == 1
    assign_step_advantages([left, right], [], groups)

    assert left.turns[0].loss_mask
    assert right.turns[0].loss_mask


def test_branch_masks_keep_only_fork_actions_and_collision_grouped_continuations():
    shared_prefix = _observation("prefix", budget=12)
    anchor_zero = _observation("anchor-zero", budget=11)
    anchor_one = _observation("anchor-one", budget=11)
    continuation = _observation("continuation", budget=10)
    unmatched = _observation("unmatched", budget=9)

    base_zero = _rollout(
        "base-zero",
        [_turn(0, shared_prefix, cost=1), _turn(1, anchor_zero, kind=ActionKind.ANSWER)],
        outcome=1.0,
    )
    base_one = _rollout(
        "base-one",
        [_turn(0, shared_prefix, cost=1), _turn(1, anchor_one, kind=ActionKind.ANSWER)],
        outcome=-1.0,
    )
    branch_zero = _rollout(
        "branch-zero",
        [
            _turn(0, shared_prefix, copied_prefix=True, loss_mask=False),
            _turn(1, anchor_zero, cost=1),
            _turn(2, continuation, cost=1, loss_mask=False),
            _turn(3, unmatched, kind=ActionKind.ANSWER, loss_mask=False),
        ],
        outcome=-1.0,
        is_base=False,
        parent="base-zero",
        fork_turn=1,
    )
    branch_one = _rollout(
        "branch-one",
        [
            _turn(0, shared_prefix, copied_prefix=True, loss_mask=False),
            _turn(1, anchor_one, cost=1),
            _turn(2, continuation, kind=ActionKind.ANSWER, loss_mask=False),
        ],
        outcome=1.0,
        is_base=False,
        parent="base-one",
        fork_turn=1,
    )

    groups = assign_step_groups(
        [base_zero, base_one],
        [branch_zero, branch_one],
    )
    continuation_group = next(
        group
        for group in groups
        if group.tier is GroupTier.COLLISION and TurnRef("branch-zero", 2) in group.members
    )
    assert set(continuation_group.members) == {
        TurnRef("branch-zero", 2),
        TurnRef("branch-one", 2),
    }

    assign_step_advantages(
        [base_zero, base_one],
        [branch_zero, branch_one],
        groups,
        epsilon=1.0,
    )
    assert not branch_zero.turns[0].loss_mask
    assert branch_zero.turns[1].loss_mask
    assert branch_zero.turns[2].loss_mask
    assert not branch_zero.turns[3].loss_mask
    assert not branch_one.turns[0].loss_mask
    assert branch_one.turns[1].loss_mask
    assert branch_one.turns[2].loss_mask


def test_anchor_selection_draws_early_and_late_and_weights_memory_pressure():
    rollout = _rollout(
        "base",
        [
            _turn(0, _observation("early", budget=10)),
            _turn(1, _observation("late", budget=6, memory_count=6)),
        ],
    )
    selected = select_fork_anchors([rollout], seed=7)

    assert [(anchor.turn_index, anchor.stratum) for anchor in selected] == [
        (0, "early"),
        (1, "late"),
    ]
    assert [anchor.weight for anchor in selected] == [1.0, 2.0]
    assert select_fork_anchors([rollout], seed=7) == selected

    early_only = _rollout(
        "early-only",
        [
            _turn(0, _observation("first", budget=12)),
            _turn(1, _observation("second", budget=11)),
        ],
    )
    carried = select_fork_anchors([early_only], seed=3)
    assert len(carried) == 2
    assert {anchor.stratum for anchor in carried} == {"early"}
    assert len({anchor.turn_index for anchor in carried}) == 2

    # Eligibility is a property of the pre-action state, not the sampled action's
    # parse/acceptance outcome.
    rollout.turns[0].accepted = False
    rollout.turns[1].action = None
    selected_after_outcomes = select_fork_anchors([rollout], seed=7)
    assert [(anchor.turn_index, anchor.stratum) for anchor in selected_after_outcomes] == [
        (0, "early"),
        (1, "late"),
    ]


def test_trajectory_advantage_is_blind_to_within_trajectory_reward_placement():
    def pair(first_rewards: tuple[float, float]) -> tuple[Rollout, Rollout]:
        rewarded = _rollout(
            "rewarded",
            [
                _turn(0, _observation("rewarded-0"), process_reward=first_rewards[0]),
                _turn(1, _observation("rewarded-1"), process_reward=first_rewards[1]),
            ],
        )
        baseline = _rollout(
            "baseline",
            [
                _turn(0, _observation("baseline-0")),
                _turn(1, _observation("baseline-1")),
            ],
        )
        return rewarded, baseline

    early, early_baseline = pair((1.0, 0.0))
    late, late_baseline = pair((0.0, 1.0))
    assign_trajectory_advantages(
        [early, early_baseline],
        process_weight=0.5,
        cost_weight=0.0,
        epsilon=1.0,
    )
    assign_trajectory_advantages(
        [late, late_baseline],
        process_weight=0.5,
        cost_weight=0.0,
        epsilon=1.0,
    )

    early_advantages = [
        turn.normalized_advantage for rollout in (early, early_baseline) for turn in rollout.turns
    ]
    late_advantages = [
        turn.normalized_advantage for rollout in (late, late_baseline) for turn in rollout.turns
    ]
    assert early_advantages == pytest.approx(late_advantages)
    assert early.turns[0].normalized_advantage == early.turns[1].normalized_advantage
