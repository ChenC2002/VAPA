"""Deterministic public reference-program demonstrations for SFT.

The paper's author demonstrations are not public.  This module closes the executable
pipeline gap without pretending to recreate them: it runs the documented synthetic
heuristic against cutoff-safe episodes and emits the exact strict JSONL consumed by
``vapa-train-sft``.  Every output is accompanied by a content-addressed manifest.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from vapa.actions import format_action
from vapa.artifacts import (
    ArtifactContentKind,
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    fingerprint_file,
    guard_artifact_write_path,
    validate_output_paths,
)
from vapa.data.episodes import load_episode_objects
from vapa.environment.state_manager import StateManager
from vapa.policies.heuristic import HeuristicPolicy
from vapa.prompts import render_chat
from vapa.provenance import package_code_fingerprint
from vapa.rollouts import RolloutRunner
from vapa.training.sft_train import parse_sft_demonstrations

DEMONSTRATION_MANIFEST_SCHEMA_VERSION = "vapa-demonstrations-v1"
PUBLIC_REFERENCE_PROGRAM_ID = "vapa-public-latest-field-reference-v1"


@dataclass(frozen=True, slots=True)
class DemonstrationGenerationResult:
    output_path: Path
    manifest_path: Path
    episodes: int
    examples: int
    output_sha256: str


def _instance_seed(seed: int, instance_id: str) -> int:
    digest = artifact_fingerprint(
        {"purpose": "public-reference-program-v1", "seed": seed, "instance_id": instance_id}
    )
    return int(digest[:16], 16)


def generate_sft_demonstrations(
    episodes_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 0,
    scaffold: str = "",
    overwrite: bool = False,
    content_kind: ArtifactContentKind | str = ArtifactContentKind.CREDENTIALED,
    repository_root: str | Path | None = None,
) -> DemonstrationGenerationResult:
    """Generate strict SFT JSONL with the public non-author reference program.

    The built-in program intentionally supports the latest-field episode family used by
    the public fixtures.  It fails closed when any action is rejected or the terminal
    answer is incorrect, so unsupported task families cannot silently become bad SFT data.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(scaffold, str):
        raise TypeError("scaffold must be a string")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")

    source = Path(episodes_path).expanduser().resolve()
    source_before = fingerprint_file(source)
    episodes = load_episode_objects(source)
    source_after = fingerprint_file(source)
    if source_before != source_after:
        raise RuntimeError("episode source changed while demonstrations were being prepared")
    if not episodes:
        raise ValueError("demonstration generation requires at least one episode")

    destination = guard_artifact_write_path(
        output_path,
        content_kind=content_kind,
        repository_root=repository_root,
    )
    manifest_path = guard_artifact_write_path(
        destination.with_name(destination.name + ".manifest.json"),
        content_kind=content_kind,
        repository_root=repository_root,
    )
    validate_output_paths([destination, manifest_path], inputs=[source], overwrite=overwrite)
    implementation = package_code_fingerprint()

    policy = HeuristicPolicy()
    runner = RolloutRunner(StateManager)
    rows: list[dict[str, object]] = []
    ordered_episodes = sorted(episodes, key=lambda episode: episode.task.instance_id)
    for episode in ordered_episodes:
        instance_id = episode.task.instance_id
        rollout = runner.run(
            episode,
            policy,
            rollout_id=f"reference:{instance_id}",
            seed=_instance_seed(seed, instance_id),
            greedy=True,
        )
        if rollout.outcome_reward != 1.0:
            raise RuntimeError(
                f"public reference program did not solve episode {instance_id!r}; "
                "provide author/task-specific demonstrations instead"
            )
        for turn in rollout.turns:
            if not turn.accepted or turn.action is None:
                raise RuntimeError(
                    f"public reference program produced an invalid action for {instance_id!r}"
                )
            rows.append(
                {
                    "messages": render_chat(turn.observation, scaffold),
                    "action": format_action(turn.action),
                    "group": f"{PUBLIC_REFERENCE_PROGRAM_ID}:{instance_id}",
                }
            )

    payload = "".join(canonical_json_dumps(row) + "\n" for row in rows)
    generated = parse_sft_demonstrations(payload)
    if len(generated) != len(rows):
        raise RuntimeError("generated demonstration count changed during validation")
    encoded = payload.encode("utf-8")
    output_sha256 = hashlib.sha256(encoded).hexdigest()
    if fingerprint_file(source) != source_before or package_code_fingerprint() != implementation:
        raise RuntimeError("demonstration inputs or implementation changed during generation")
    manifest = {
        "schema_version": DEMONSTRATION_MANIFEST_SCHEMA_VERSION,
        "generator_id": PUBLIC_REFERENCE_PROGRAM_ID,
        "paper_exact": False,
        "warning": "These deterministic public demonstrations are not the author SFT data.",
        "seed": seed,
        "scaffold_sha256": artifact_fingerprint({"scaffold": scaffold}),
        "source": source_before.to_dict(),
        "output": {"sha256": output_sha256, "size_bytes": len(encoded)},
        "content_kind": ArtifactContentKind(content_kind).value,
        "episodes": len(ordered_episodes),
        "examples": len(rows),
        "package_code": implementation,
    }
    atomic_write_text(destination, payload, overwrite=overwrite)
    atomic_write_text(manifest_path, canonical_json_dumps(manifest) + "\n", overwrite=overwrite)
    return DemonstrationGenerationResult(
        output_path=destination,
        manifest_path=manifest_path,
        episodes=len(ordered_episodes),
        examples=len(rows),
        output_sha256=output_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate non-author public SFT demonstrations from episode JSON/JSONL."
    )
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scaffold", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--content-kind",
        choices=[item.value for item in ArtifactContentKind],
        default=ArtifactContentKind.CREDENTIALED.value,
    )
    parser.add_argument("--repository-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = generate_sft_demonstrations(
        arguments.episodes,
        arguments.output,
        seed=arguments.seed,
        scaffold=arguments.scaffold,
        overwrite=arguments.overwrite,
        content_kind=arguments.content_kind,
        repository_root=arguments.repository_root,
    )
    print(
        canonical_json_dumps(
            {
                "episodes": result.episodes,
                "examples": result.examples,
                "manifest": str(result.manifest_path),
                "output": str(result.output_path),
                "sha256": result.output_sha256,
            }
        )
    )
    return 0


__all__ = [
    "DEMONSTRATION_MANIFEST_SCHEMA_VERSION",
    "DemonstrationGenerationResult",
    "PUBLIC_REFERENCE_PROGRAM_ID",
    "generate_sft_demonstrations",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
