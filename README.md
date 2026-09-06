# Where Credit Lands: Step-Level Advantages for Bounded-Memory EHR Agents

**VAPA** (**V**erifier-**A**nchored **P**rocess **A**dvantage) trains bounded-memory EHR agents using verifier rewards and forked replay to assign credit to individual actions. At inference, the agent retrieves evidence, manages memory, and answers under an action budget without replay or verifiers.

## Method Map

| Stage | Implementation | Responsibility |
| --- | --- | --- |
| Data preparation | [`adapters.py`](src/vapa/data/adapters.py), [`pipeline.py`](src/vapa/data/pipeline.py) | Construct cutoff-valid tasks and patient-disjoint splits |
| Bounded-memory agent | [`actions.py`](src/vapa/actions.py), [`environment/`](src/vapa/environment/) | Enforce action, memory, retrieval, and budget constraints |
| Process supervision | [`verifiers.py`](src/vapa/verifiers.py), [`rollouts.py`](src/vapa/rollouts.py) | Score actions and replay exact pre-action states |
| Step credit | [`groups.py`](src/vapa/training/groups.py), [`advantages.py`](src/vapa/training/advantages.py), [`trainer.py`](src/vapa/training/trainer.py) | Combine local and episode advantages |
| SFT and RL | [`sft_train.py`](src/vapa/training/sft_train.py), [`rl_train.py`](src/vapa/training/rl_train.py) | Train action tokens and save resumable checkpoints |
| Evaluation | [`runner.py`](src/vapa/evaluation/runner.py), [`analysis.py`](src/vapa/evaluation/analysis.py), [`reporting.py`](src/vapa/evaluation/reporting.py) | Evaluate checkpoints and analyze experiment results |

Examples are in [`examples/`](examples/), settings in [`configs/`](configs/), and tests
in [`tests/`](tests/). Versioned prompts and schemas live in
[`public_core_v1/`](src/vapa/releases/public_core_v1/).

## Quick Start

### Install

From the repository root, using Python 3.11 or newer:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The core has no runtime dependencies; optional training and development extras are
defined in [`pyproject.toml`](pyproject.toml).

### Validate

```bash
vapa validate-public-core
vapa validate-config configs/vapa_qwen.toml
vapa demo --compact
```

These checks need no GPU, model weights, credentials, or network access.

## Results and Outputs

| Results | Summary | Log |
| --- | --- | --- |
| Paper-reported references | [`paper_results.json`](results/paper_results.json) | [`paper_results.jsonl`](logs/paper_results.jsonl) |
| Tiny-model training checks | [`training_results.json`](results/training_results.json) | [`training_results.jsonl`](logs/training_results.jsonl) |
| Demo checks | [`demo_results.json`](results/demo_results.json) | [`demo_results.jsonl`](logs/demo_results.jsonl) |

Paper numbers are manuscript-reported, not independently reproduced here; tiny-model
training and synthetic demo results are execution checks, not benchmark performance.

Runs retain prepared splits, demonstrations, checkpoints, predictions, and analysis
reports in their selected output directories; keep patient-level data on approved storage.

<details>
<summary>Verify or regenerate result snapshots</summary>

Verify paper references:

```bash
python scripts/export_paper_results.py
```

Reproduce the CPU training checks in a separate Python 3.13 environment:

```bash
python3.13 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -e '.[train]' -c configs/model_test_constraints.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train_tiny.py --publish
```

Regenerate demo snapshots with `vapa demo --compact --publish`.

`--publish` updates the corresponding files above; underlying runs remain in `runs/`.

</details>

## Required Inputs

Full-model runs require authorized EHR data, task-specific adapters and scoring,
verifier/calculator catalogs, and pinned model/tokenizer revisions with LoRA targets.

| Input | Example |
| --- | --- |
| Dataset paths, field mappings, and patient splits | [`tiny_dataset_manifest.json`](examples/tiny_dataset_manifest.json) |
| Event records and task instances | [`tiny_events.csv`](examples/tiny_events.csv), [`tiny_tasks.jsonl`](examples/tiny_tasks.jsonl) |
| Event-only task construction | [`tiny_latest_field_manifest.json`](examples/tiny_latest_field_manifest.json) |
| Action-token demonstrations | [`tiny_sft.jsonl`](examples/tiny_sft.jsonl) |
| Verifier and calculator schemas | [`demo_verifier_catalog.json`](examples/demo_verifier_catalog.json), [`tiny_calculators.json`](examples/tiny_calculators.json) |
| Analysis and reporting manifests | [`tiny_analysis_manifest.json`](examples/tiny_analysis_manifest.json), [`tiny_experiment_report_manifest.json`](examples/tiny_experiment_report_manifest.json) |

Use [`vapa_qwen.toml`](configs/vapa_qwen.toml) for the main experiment, or the
[GPT-OSS](configs/vapa_gpt_oss.toml) and [matched no-fork](configs/no_fork_matched_qwen.toml) variants.

## Workflow

The default commands use bundled examples; full-model templates are expandable below.

### Step 1: Preprocess

```bash
vapa prepare-data examples/tiny_dataset_manifest.json runs/demo/prepared
vapa validate-episodes runs/demo/prepared/train.jsonl
```

Preparation writes patient-disjoint splits; replace the example manifest with your own mapped sources.

### Step 2: Build Demonstrations

```bash
vapa-generate-sft runs/demo/prepared/train.jsonl runs/demo/sft.jsonl \
  --content-kind public
vapa-train-sft --data runs/demo/sft.jsonl --dry-run
```

The generator supports the example latest-field tasks; other task families need their own demonstrations.

### Step 3: Train

Check the RL setup without loading a model:

```bash
vapa show-training-plan --config configs/vapa_qwen.toml
vapa-train configs/vapa_qwen.toml examples/tiny_episode.json runs/demo/rl \
  --dry-run --demo-catalogs
```

<details>
<summary>Full-model SFT and RL</summary>

These templates require your own inputs: replace `AUTHOR_*`, `path/to/...`, and `your_release.*`.

```bash
python -m pip install -e '.[train]'

vapa-train-sft \
  --data path/to/demonstrations.jsonl \
  --output-dir runs/sft \
  --model-revision AUTHOR_MODEL_COMMIT \
  --tokenizer-revision AUTHOR_TOKENIZER_COMMIT \
  --model-kind multimodal --use-processor \
  --lora-target-module AUTHOR_MODULE
```

Initialize the RL actor and frozen KL reference from the resulting SFT checkpoint:

```bash
vapa-train configs/vapa_qwen.toml data/processed/train.jsonl runs/vapa \
  --sft-checkpoint path/to/final-sft-checkpoint \
  --model-revision AUTHOR_MODEL_COMMIT \
  --tokenizer-revision AUTHOR_TOKENIZER_COMMIT \
  --model-kind multimodal --use-processor yes \
  --lora-target-module AUTHOR_MODULE \
  --verifier-manifest path/to/verifiers.json \
  --verifier-factory your_release.verifiers:factory \
  --calculator-manifest path/to/calculators.json \
  --outcome-scorer your_release.scoring:outcome_scorer
```

Repeat `--lora-target-module` for each target; use SFT `--resume-from` or RL `--resume`
to continue a checkpoint in its original run.

</details>

### Step 4: Evaluate

```bash
vapa evaluate runs/demo/prepared/train.jsonl runs/demo/evaluation
```

This evaluates the reference policy on the prepared examples, not a trained checkpoint.

<details>
<summary>Trained-checkpoint evaluation</summary>

Use the RL run's `checkpoint_path` and your task-specific scorer:

```bash
vapa evaluate data/processed/test.jsonl results/evaluation \
  --checkpoint path/to/final-vapa-checkpoint \
  --checkpoint-factory vapa.model.inference:checkpoint_factory \
  --policy-factory vapa.model.inference:policy_factory \
  --checkpoint-manager-factory vapa.model.inference:checkpoint_manager_factory \
  --outcome-scorer your_release.scoring:outcome_scorer \
  --policy-id vapa-transformers-v1
```

</details>

### Step 5: Analyze

```bash
vapa-analyze examples/tiny_analysis_manifest.json \
  examples/tiny_binary_predictions.jsonl runs/demo/analysis.json --content-kind public
vapa-report examples/tiny_experiment_report_manifest.json \
  examples/tiny_experiment_metrics.jsonl runs/demo/report.json --content-kind public
```

Replace the fixture inputs with your predictions and metric rows for experiment reporting.

<details>
<summary>Advanced analysis and checkpoint notes</summary>

- Binary evaluation uses `--binary-readout --policy-id SYSTEM_ID`; trained systems also require `--analysis-seed SEED`.
- Aggregate repeated episodes to one value per system/seed/task/patient/metric/coordinate/profile cell before reporting; paired comparisons need matched panels.
- Use `decision_depths` / `decision_depth` for per-step slopes, not the legacy per-day horizon fields.
- `confirmatory_tests` selects the Holm correction family; omit it for all computed tests or use `[]` for none.
- Checkpoint loading verifies file and tokenizer fingerprints; older tokenizer-v1 checkpoints are incompatible.
- Existing data/report outputs require explicit `--overwrite`; use `--help` for each command's options.

</details>

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check src tests scripts
ruff format --check src tests scripts
python scripts/validate_release.py
python -m build
```

`pytest -m model` selects the optional CPU training integration test; it requires the
model dependencies above. Release validation checks result/log agreement and fingerprints.
