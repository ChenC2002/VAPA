# Where Credit Lands: Step-Level Advantages for Bounded-Memory EHR Agents

**VAPA** (**V**erifier-**A**nchored **P**rocess **A**dvantage) learns how to retrieve cutoff-valid evidence, maintain a bounded memory, and answer EHR questions under an action budget. Deterministic verifiers score checkable actions, forked replay creates local comparisons, and a two-level estimator combines step credit with terminal answer rewards. Replay and verifiers are used only during training.

## Method Map

| Stage | Implementation | Responsibility |
| --- | --- | --- |
| Data preparation | [`adapters.py`](src/vapa/data/adapters.py), [`pipeline.py`](src/vapa/data/pipeline.py) | Map source records, construct tasks, filter by cutoff, and split by patient |
| Bounded-memory agent | [`actions.py`](src/vapa/actions.py), [`environment/`](src/vapa/environment/) | Parse the eight-action grammar and enforce memory, retrieval, calculator, and budget contracts |
| Process supervision | [`verifiers.py`](src/vapa/verifiers.py), [`rollouts.py`](src/vapa/rollouts.py) | Score observable evidence and reproduce exact pre-action states for replay |
| Step credit | [`groups.py`](src/vapa/training/groups.py), [`advantages.py`](src/vapa/training/advantages.py), [`trainer.py`](src/vapa/training/trainer.py) | Build exact-fork, state-collision, and turn-index groups; combine local and episode advantages |
| SFT and RL | [`sft_train.py`](src/vapa/training/sft_train.py), [`rl_train.py`](src/vapa/training/rl_train.py) | Train action tokens, track sampled-token budgets, and save resumable checkpoints |
| Evaluation | [`runner.py`](src/vapa/evaluation/runner.py), [`analysis.py`](src/vapa/evaluation/analysis.py), [`reporting.py`](src/vapa/evaluation/reporting.py) | Run checkpoint-backed inference and compute binary, factorial, horizon, and profile statistics |

Versioned prompts and manifest schemas live in
[`public_core_v1/`](src/vapa/releases/public_core_v1/). Test example is in
[`examples/`](examples/), experiment settings in [`configs/`](configs/), and regression
tests in [`tests/`](tests/). The `scripts/` entrypoints use the same implementations as
the installed commands below.

## Quick Start

### Install

From the repository root, using Python 3.11 or newer:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The core has no runtime dependencies. Install `'.[train]'` for Transformers/PEFT
training, `'.[dev]'` for development tools, or `'.[all]'` for both. Dependency ranges
are in [`pyproject.toml`](pyproject.toml); freeze a platform-specific environment before
running a research experiment.

### Validate

```bash
vapa validate-public-core
vapa validate-config configs/vapa_qwen.toml
vapa demo --compact
python scripts/validate_release.py
```

These checks need no GPU, model weights, credentials, or network access.

## Results

| Evidence | Summary | Log | Meaning |
| --- | --- | --- | --- |
| Paper-reported | [`paper_results.json`](results/paper_results.json) | [`paper_results.jsonl`](logs/paper_results.jsonl) | Paper results |
| Executed model training | [`training_results.json`](results/training_results.json) | [`training_results.jsonl`](logs/training_results.jsonl) | Real CPU LoRA/SFT updates, checkpoint resume, and a shared-reference objective check on a tiny random model |
| Executed demo | [`demo_results.json`](results/demo_results.json) | [`demo_results.jsonl`](logs/demo_results.jsonl) | Dependency-free test preparation, method, evaluation, and analysis checks |

Verify the snapshots without the manuscript, or check against the original sources:

```bash
python scripts/export_paper_results.py
python scripts/export_paper_results.py --source /path/to/manuscript-review-v1
```

To reproduce the recorded CPU stack, use a separate Python 3.13 environment:

```bash
python3.13 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -e '.[train]' -c configs/model_test_constraints.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train_tiny.py --publish
```

Each run keeps its random base model, resumable SFT checkpoints, and original metrics
under a new `runs/tiny-training/run-.../` directory.

The executed summary is in [`results/demo_results.json`](results/demo_results.json),
with a test result-event log in [`logs/demo_results.jsonl`](logs/demo_results.jsonl).
Regenerate both public snapshots and retain all underlying run files:

```bash
vapa demo --compact --publish
```


## Required Inputs

### EHR and Task Files

| Input | Contract / example |
| --- | --- |
| Dataset manifest | Explicit source paths, column mappings, disclosure class, adapter, and patient split settings; [`tiny_dataset_manifest.json`](examples/tiny_dataset_manifest.json) |
| Event records | CSV, JSON, or JSONL mapped to `pointer`, `patient_id`, ISO-8601 `timestamp`, `domain`, `field`, and scalar `value`; optional `unit`, `text`, `source`, `code` |
| Task records | Instance/patient IDs, instruction, cutoff, family, requested fields/window, answer type, gold answer, and reference pointers; [`tiny_tasks.jsonl`](examples/tiny_tasks.jsonl) |
| Event-only task template | `latest_field_events` constructs latest-value tasks from mapped events and an explicit field/window template; [`tiny_latest_field_manifest.json`](examples/tiny_latest_field_manifest.json) |


### Training and Evaluation Artifacts

| Input | Contract / example |
| --- | --- |
| SFT demonstrations | JSONL with exactly `messages`, an `action` string, and a nonempty `group`; [`tiny_sft.jsonl`](examples/tiny_sft.jsonl) |
| Model and tokenizer | Immutable commit revisions, explicit processor selection, and explicit LoRA target modules |
| Verifiers | Checksum-bound catalog and task-specific implementation; [`demo_verifier_catalog.json`](examples/demo_verifier_catalog.json) shows the public schema |
| Calculators and scoring | Frozen formula/unit/tolerance definitions and a task-specific outcome scorer; [`tiny_calculators.json`](examples/tiny_calculators.json) is synthetic |
| Analysis inputs | Predictions or patient/task metric rows plus a checksum-bound manifest; [`tiny_analysis_manifest.json`](examples/tiny_analysis_manifest.json), [`tiny_experiment_report_manifest.json`](examples/tiny_experiment_report_manifest.json) |

## Workflow

### Step 1: Preprocess

```bash
vapa prepare-data examples/tiny_dataset_manifest.json runs/demo/prepared
vapa validate-episodes runs/demo/prepared/train.jsonl
```

Replace the example manifest with your mapped sources for a dataset run. To construct
tasks without a separate task file, use `examples/tiny_latest_field_manifest.json`.
Preparation writes patient-disjoint splits and a manifest recording source, adapter,
configuration, and output fingerprints. Existing outputs require explicit `--overwrite`.

### Step 2: Build Demonstrations

```bash
vapa-generate-sft runs/demo/prepared/train.jsonl runs/demo/sft.jsonl \
  --content-kind public
vapa-train-sft --data runs/demo/sft.jsonl --dry-run
```

The bundled reference program supports the synthetic latest-field tasks. Every action
and terminal answer is checked before demonstrations are published; unsupported tasks
fail instead of producing training labels. Provide task-specific demonstrations for
other families. The generator preserves existing output unless `--overwrite` is passed
and writes its checksum manifest last.

### Step 3: Train

Install the model stack on suitable compute:

```bash
python -m pip install -e '.[train]'
```

Run action-token SFT. Replace `AUTHOR_*` and `path/to/...` with your frozen inputs:

```bash
vapa-train-sft \
  --data path/to/demonstrations.jsonl \
  --output-dir runs/sft \
  --model-revision AUTHOR_MODEL_COMMIT \
  --tokenizer-revision AUTHOR_TOKENIZER_COMMIT \
  --model-kind multimodal --use-processor \
  --lora-target-module AUTHOR_MODULE
```

Prompts are context-only; the loss covers action tokens. Demonstration groups remain
intact during accumulation. Dry-run token counts are estimates; real planning uses the
pinned tokenizer. SFT resume uses `--resume-from` with a checkpoint in the same run.

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

Repeat `--lora-target-module` for each target. Use `--dry-run` to validate inputs before
loading weights, and `--resume` to continue an RL checkpoint in its original run.
The dependency-free synthetic launch is:

```bash
vapa-train configs/vapa_qwen.toml examples/tiny_episode.json runs/demo/rl \
  --dry-run --demo-catalogs
```

RL uses hierarchical task sampling, curriculum gates, batch-wide advantage normalization,
intact comparison groups, and a sampled-token ledger. Generation and actor/reference
likelihoods use the same legal action-token support. Checkpoints bind the model,
tokenizer, LoRA topology, data, environment, optimizer, and runtime cursor.

### Step 4: Evaluate

Run the public reference policy on prepared synthetic episodes:

```bash
vapa evaluate runs/demo/prepared/train.jsonl runs/demo/evaluation
```

For a trained model, use the `checkpoint_path` reported by the RL run:

```bash
vapa evaluate data/processed/test.jsonl results/evaluation \
  --checkpoint path/to/final-vapa-checkpoint \
  --checkpoint-factory vapa.model.inference:checkpoint_factory \
  --policy-factory vapa.model.inference:policy_factory \
  --checkpoint-manager-factory vapa.model.inference:checkpoint_manager_factory \
  --outcome-scorer your_release.scoring:outcome_scorer \
  --policy-id vapa-transformers-v1
```

The loader verifies checkpoint contracts and file checksums before constructing the
model. Evaluation restores the training environment without replay or verifier access,
uses stable per-instance seeds, and supports resume without changing completed answers.
Research evaluation requires a task-specific scorer; the built-in exact-string scorer
is intended for synthetic fixtures.

### Step 5: Analyze

```bash
vapa-analyze examples/tiny_analysis_manifest.json \
  examples/tiny_binary_predictions.jsonl runs/demo/analysis.json --content-kind public
vapa-report examples/tiny_experiment_report_manifest.json \
  examples/tiny_experiment_metrics.jsonl runs/demo/report.json --content-kind public
```

For binary endpoints, evaluation accepts `--binary-readout --policy-id SYSTEM_ID` and
`--analysis-seed SEED` for trained systems. It freezes a label-blind patient-preserving
roster, capped at 2,500 instances per task by default, and emits `binary_predictions.jsonl`
with fixed-answer-position probabilities. Set `metadata.evaluation_task_id` when the
evaluation task is narrower than the episode family.

`vapa-report` expects one value per system/seed/task/patient/metric/coordinate/profile cell;
aggregate repeated episodes within that cell before reporting. Paired comparisons
require matched task and patient panels. Reports include equal-task factorial contrasts,
paired-seed t intervals, Holm-adjusted p values, decision-depth slopes, and joint patient
bootstraps. Profile intervals average seeds first, retain the declared task set, and
report how many bootstrap draws were undefined because a task had no sampled patients.
Both analysis commands preserve existing reports unless `--overwrite` is explicit.

The horizon section uses `decision_depths`, with `decision_depth` in metric rows;
slopes are per decision step. The separate legacy `horizons_days` / `horizon_days`
contract yields per-day slopes. Do not mix the two axes. Systems must share patients
at each coordinate, but different depths may use different cohorts.
Optional `confirmatory_tests` lists the exact test IDs to include in the report's
Holm family, as in the example manifest. Omit it to adjust all computed tests, or
use `[]` for no confirmatory family. Nonmembers retain nominal p values and have
`holm_p_value: null`.

## Outputs

| Stage | Output |
| --- | --- |
| Public demo snapshots | `results/demo_results.json`, `logs/demo_results.jsonl` |
| Paper reference tables | `results/paper_results.json`, `logs/paper_results.jsonl` |
| Measured tiny-model training | `results/training_results.json`, `logs/training_results.jsonl` |
| Retained demo run | `runs/demo/run-.../results.json`, `events.jsonl`, and stage outputs |
| Preparation | `train.jsonl`, `validation.jsonl`, `test.jsonl`, `prepared_manifest.json` |
| Demonstrations | Action-token JSONL and adjacent `.manifest.json` |
| SFT / RL | Run manifest, metrics, result summary, and checksummed `checkpoints/`; RL also records token/replay accounting |
| Evaluation | Run manifest, predictions, errors, aggregate metrics, and optional binary predictions |
| Analysis | JSON estimates, intervals, comparisons, and input fingerprints |

Keep patient-level inputs and outputs on approved storage under `data/`, `runs/`,
`results/`, or an external protected path. Generated runs and weights are ignored;
only the six reviewed result/log snapshots are allowlisted. Ignoring is not access control.
Training/evaluation default to `credentialed` and reject sensitive writes
into public source/example locations. Use `public` only for reviewed synthetic data;
`derived` denotes public-safe aggregates, not patient-level derived records.

## Experiment Configuration

[`vapa_qwen.toml`](configs/vapa_qwen.toml) contains the main A4 settings: Qwen3.5-9B,
rank-16 LoRA, 32,768 context tokens, 512 generated tokens per turn, memory capacity 8,
action budget 12, turn cap 16, 8 base rollouts, up to 2 replay anchors with 3 siblings,
process weight 0.05, action-cost weight 0.02, learning rate `5e-6`, and KL weight 0.01.
Updates target at least `2^17` sampled actor tokens within a `2^25` total budget.

```bash
vapa show-training-plan --config configs/vapa_qwen.toml
```

The four arms cross outcome-only versus outcome-plus-process rewards with trajectory
versus VAPA step credit. The step-credit factor changes replay, groups, masks, and the
update together. [`no_fork_matched_qwen.toml`](configs/no_fork_matched_qwen.toml) uses a
cumulative replay-token quota for the matched no-fork control.
[`vapa_gpt_oss.toml`](configs/vapa_gpt_oss.toml) selects the GPT-OSS arm with explicit
MXFP4 dequantization, recorded in the checkpoint load plan.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check src tests scripts
ruff format --check src tests scripts
python scripts/validate_release.py
python -m build
```

CI also installs the wheel and executes the full demo outside the checkout. A separate
CPU-model job runs genuine SFT/resume and shared-reference checks. `pytest -m model`
selects that integration test; it skips when optional model libraries are absent.
Release validation reparses every paper-result source row, checks result/log agreement,
and rejects stale demo and tiny-training fingerprints. Regenerate both executed
snapshots after changing their implementation.

Tokenizer fingerprints now bind the token mapping and encoding rules, not just the
name and vocabulary size. Older tokenizer-v1 checkpoints fail the stricter identity
check; they are not silently accepted as compatible.

To build a review archive with neutral timestamps and no Git history, caches, private
runs, or model files:

```bash
python scripts/package_review.py --output ../VAPA-review.zip
```

The packager checks for local user paths, email addresses, account links, and prohibited
files. Use repeatable `--deny-text` arguments for additional names or identifiers that
must not appear. Automated checks cannot guarantee anonymity against outside knowledge
or matching public material. The archive preserves evidence provenance and does not
turn synthetic tests into benchmark results.
