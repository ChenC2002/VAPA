"""Unified command line for validation and the dependency-free method demo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from vapa import __version__
from vapa.config import ExperimentConfig, load_config
from vapa.data import load_episode_objects, sha256_file
from vapa.demo import publish_demo_results, run_demo, run_demo_suite
from vapa.environment.calculators import CalculatorRegistry
from vapa.training.curriculum import STAGES


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False))


def _validate_config(args: argparse.Namespace) -> int:
    config = load_config(args.path)
    _json({"status": "ok", "config": asdict(config)})
    return 0


def _demo(args: argparse.Namespace) -> int:
    if args.output is None and not args.publish:
        _json(run_demo(args.seed, compact=args.compact))
        return 0
    output = args.output or Path("runs/demo") / f"run-{uuid4().hex[:12]}"
    result = run_demo_suite(args.examples, output, seed=args.seed, compact=args.compact)
    if args.publish:
        publish_demo_results(output, Path.cwd())
    _json(
        {
            "status": "ok",
            "run_directory": str(output.resolve()),
            "published": args.publish,
            "records": len(result["records"]),
        }
    )
    return 0


def _validate_episodes(args: argparse.Namespace) -> int:
    episodes = load_episode_objects(args.path)
    _json(
        {
            "status": "ok",
            "sha256": sha256_file(args.path),
            "episodes": len(episodes),
            "patients": len({episode.task.patient_id for episode in episodes}),
            "admissible_events": sum(len(episode.admissible_events) for episode in episodes),
            "post_cutoff_events": sum(
                len(episode.events) - len(episode.admissible_events) for episode in episodes
            ),
        }
    )
    return 0


def _validate_calculators(args: argparse.Namespace) -> int:
    registry = CalculatorRegistry.from_json(args.path)
    _json({"status": "ok", "calculators": len(registry), "sha256": sha256_file(args.path)})
    return 0


def _show_training_plan(args: argparse.Namespace) -> int:
    config = ExperimentConfig() if args.config is None else load_config(args.config)
    _json(
        {
            "experiment": config.name,
            "model": config.model.name,
            "steps": [
                "construct patient-disjoint episodes from credentialed sources",
                "generate public or import author-supplied reference-program demonstrations",
                "train one-epoch action-token SFT LoRA",
                "sample complete base groups and optional replay branches",
                "score deterministic verifiers and build tiered advantages",
                "take one actor update per intact accumulated batch",
                "evaluate the final token-budget-matched checkpoint without replay or verifiers",
            ],
            "curriculum": [
                {
                    "name": stage.name,
                    "fraction": [stage.start_fraction, stage.end_fraction],
                    "history_quartile": stage.max_history_quartile,
                    "actions": sorted(action.value for action in stage.actions),
                }
                for stage in STAGES
            ],
            "sampled_token_budget": config.optimization.sampled_token_budget,
            "paper_scale_backend_included": True,
            "paper_exact_author_artifacts_included": False,
        }
    )
    return 0


def _prepare_data(args: argparse.Namespace) -> int:
    from vapa.data.pipeline import prepare_dataset

    result = prepare_dataset(
        args.manifest,
        args.output,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    _json(result.to_dict())
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from vapa.evaluation.runner import evaluate_episodes
    from vapa.inference import heuristic_policy_factory, import_factory, path_checkpoint_factory

    episodes = load_episode_objects(args.episodes)
    checkpoint_factory = (
        path_checkpoint_factory
        if args.checkpoint_factory is None
        else import_factory(args.checkpoint_factory)
    )
    policy_factory = (
        heuristic_policy_factory
        if args.policy_factory is None
        else import_factory(args.policy_factory)
    )
    checkpoint_manager_factory = (
        None
        if args.checkpoint_manager_factory is None
        else import_factory(args.checkpoint_manager_factory)
    )
    outcome_scorer = None if args.outcome_scorer is None else import_factory(args.outcome_scorer)
    result = evaluate_episodes(
        episodes,
        args.output,
        seed=args.seed,
        max_instances=args.max_instances,
        task_cap=args.task_cap,
        selection_seed=args.selection_seed,
        greedy=not args.stochastic,
        checkpoint_path=args.checkpoint,
        checkpoint_factory=checkpoint_factory,
        policy_factory=policy_factory,
        checkpoint_manager_factory=checkpoint_manager_factory,
        **({} if outcome_scorer is None else {"outcome_scorer": outcome_scorer}),
        policy_id=args.policy_id,
        binary_readout=args.binary_readout,
        analysis_seed=args.analysis_seed,
        continue_on_error=args.continue_on_error,
        retry_errors=args.retry_errors,
        content_kind=args.content_kind,
        allow_synthetic_outcome_scorer=args.allow_synthetic_outcome_scorer,
    )
    _json(result.to_dict())
    return 0


def _validate_public_core(args: argparse.Namespace) -> int:
    from vapa.releases import validate_public_core_release

    report = validate_public_core_release(args.release_dir)
    _json(report.to_dict())
    return 0 if report.ok() else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vapa", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="strictly validate a TOML experiment")
    validate.add_argument("path", type=Path)
    validate.set_defaults(handler=_validate_config)

    demo = subparsers.add_parser("demo", help="run environment, replay, verifiers, and advantages")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--compact", action="store_true", help="use a smaller base/sibling group")
    demo.add_argument(
        "--output", type=Path, help="run the full synthetic lifecycle in a new directory"
    )
    demo.add_argument(
        "--examples", type=Path, default=Path("examples"), help="public fixture directory"
    )
    demo.add_argument(
        "--publish",
        action="store_true",
        help="run the full demo and update results/demo_results.json and logs/demo_results.jsonl",
    )
    demo.set_defaults(handler=_demo)

    episodes = subparsers.add_parser(
        "validate-episodes", help="validate canonical JSON/JSONL episodes"
    )
    episodes.add_argument("path", type=Path)
    episodes.set_defaults(handler=_validate_episodes)

    calculators = subparsers.add_parser(
        "validate-calculators", help="validate a calculator manifest"
    )
    calculators.add_argument("path", type=Path)
    calculators.set_defaults(handler=_validate_calculators)

    plan = subparsers.add_parser(
        "show-training-plan", help="show the paper-scale stages and limits"
    )
    plan.add_argument(
        "--config",
        type=Path,
        help="experiment TOML (defaults to the built-in paper-scale configuration)",
    )
    plan.set_defaults(handler=_show_training_plan)

    prepare = subparsers.add_parser(
        "prepare-data",
        help="build cutoff-safe patient-disjoint episodes from an explicit manifest",
    )
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--seed", type=int)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(handler=_prepare_data)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="run deterministic resume-safe evaluation over prepared episodes",
    )
    evaluate.add_argument("episodes", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--max-instances", type=int)
    evaluate.add_argument("--task-cap", type=int)
    evaluate.add_argument("--selection-seed", type=int)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--checkpoint-factory")
    evaluate.add_argument("--policy-factory")
    evaluate.add_argument("--checkpoint-manager-factory")
    evaluate.add_argument("--policy-id")
    evaluate.add_argument("--binary-readout", action="store_true")
    evaluate.add_argument("--analysis-seed", type=int)
    evaluate.add_argument("--outcome-scorer")
    evaluate.add_argument("--allow-synthetic-outcome-scorer", action="store_true")
    evaluate.add_argument("--stochastic", action="store_true")
    evaluate.add_argument("--continue-on-error", action="store_true")
    evaluate.add_argument("--retry-errors", action="store_true")
    evaluate.add_argument(
        "--content-kind",
        choices=("public", "derived", "credentialed", "raw_records"),
        default="credentialed",
    )
    evaluate.set_defaults(handler=_evaluate)

    release = subparsers.add_parser(
        "validate-public-core",
        help="verify the bundled versioned public-core prompts and schemas",
    )
    release.add_argument("--release-dir", type=Path)
    release.set_defaults(handler=_validate_public_core)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
