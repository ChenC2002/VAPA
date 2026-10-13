# Where Credit Lands: Verifiable Step Credit for Bounded-Memory EHR Agents

**VAPA** (Verifier-Anchored Process Advantage) is a training method for bounded-memory
EHR agents. It uses record-grounded verifiers and replay branches from the same agent
state to assign credit to individual actions. At test time, the agent retrieves
evidence, manages its memory, and answers within an action budget, without replay
or verifier calls.

## Getting started

Clone the repository and install the core package with Python 3.11 or newer:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Then check the setup and run the example:

```bash
vapa validate-public-core
vapa validate-config configs/vapa_qwen.toml
vapa demo --compact
```

The core package has no runtime dependencies. The example runs on CPU and does not
download model weights or need access to patient data.

## Results

| Results | Summary | Log |
| --- | --- | --- |
| Paper-reported references (earlier manuscript) | [`paper_results.json`](results/paper_results.json) | [`paper_results.jsonl`](logs/paper_results.jsonl) |
| Tiny-model training checks | [`training_results.json`](results/training_results.json) | [`training_results.jsonl`](logs/training_results.jsonl) |
| Synthetic demo checks | [`demo_results.json`](results/demo_results.json) | [`demo_results.jsonl`](logs/demo_results.jsonl) |

Each run saves its outputs in the directory you choose. Keep patient-level data on
approved storage.

<details>
<summary>Rerun the CPU checks</summary>

For the tiny-model training run, use a separate Python 3.13 environment:

```bash
python3.13 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -e '.[train]' -c configs/model_test_constraints.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train_tiny.py --publish
```

To update the demo results, run `vapa demo --compact --publish`.
The `--publish` flag replaces the summary and log linked above; full run files stay in `runs/`.

</details>

## Data and configuration

The files in [`examples/`](examples/) show the expected input formats:

| Input | Example |
| --- | --- |
| Dataset paths, field mappings, and patient splits | [`tiny_dataset_manifest.json`](examples/tiny_dataset_manifest.json) |
| Event records and task instances | [`tiny_events.csv`](examples/tiny_events.csv), [`tiny_tasks.jsonl`](examples/tiny_tasks.jsonl) |
| Event-only task construction | [`tiny_latest_field_manifest.json`](examples/tiny_latest_field_manifest.json) |
| Action-token demonstrations | [`tiny_sft.jsonl`](examples/tiny_sft.jsonl) |
| Verifier and calculator schemas | [`demo_verifier_catalog.json`](examples/demo_verifier_catalog.json), [`tiny_calculators.json`](examples/tiny_calculators.json) |
| Analysis and reporting manifests | [`tiny_analysis_manifest.json`](examples/tiny_analysis_manifest.json), [`tiny_experiment_report_manifest.json`](examples/tiny_experiment_report_manifest.json) |

For your own experiments, supply authorized EHR data, task adapters and scorers,
verifier and calculator implementations, and fixed model/tokenizer revisions.
The [Table 7 layout](src/vapa/releases/paper_verifier_layout.json) lists the names,
weights, and action assignments of all fourteen predicates. Their detailed decision and
abstention rules need the companion specification; the demo uses seven predicates.

Start with [`vapa_qwen.toml`](configs/vapa_qwen.toml) for the main experiment.
The [`configs/`](configs/) directory also includes
[GPT-OSS](configs/vapa_gpt_oss.toml) and [matched no-fork](configs/no_fork_matched_qwen.toml)
settings. Training follows the forward-KL objective in Eq. 9.

## Running the pipeline

The commands below use the small example dataset. Expand the training and evaluation
sections when you are ready to use your own data and checkpoints.

### 1. Prepare the data

```bash
vapa prepare-data examples/tiny_dataset_manifest.json runs/demo/prepared
vapa validate-episodes runs/demo/prepared/train.jsonl
```

This creates splits with no patients shared between them. To use another dataset,
edit the paths and field mappings in the input manifest.

### 2. Build SFT demonstrations

```bash
vapa-generate-sft runs/demo/prepared/train.jsonl runs/demo/sft.jsonl \
  --content-kind public
vapa-train-sft --data runs/demo/sft.jsonl --dry-run
```

The included generator handles latest-field queries. Add demonstrations for any
other task families you plan to train on.

### 3. Train

First, inspect the training plan and check the configuration without loading a model:

```bash
vapa show-training-plan --config configs/vapa_qwen.toml
vapa-train configs/vapa_qwen.toml examples/tiny_episode.json runs/demo/rl \
  --dry-run --demo-catalogs
```

<details>
<summary>Train a model with SFT, then RL</summary>

Replace `AUTHOR_*`, `path/to/...`, and `your_release.*` with your revisions, paths,
and task implementations. Set the LoRA target modules for your chosen model.

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

Use the SFT checkpoint to initialize both the RL actor and its frozen KL reference:

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

Repeat `--lora-target-module` for each target module. To resume training in the same
run directory, use `--resume-from` for SFT or `--resume` for RL.

</details>

### 4. Evaluate

Run the reference policy on the prepared examples:

```bash
vapa evaluate runs/demo/prepared/train.jsonl runs/demo/evaluation
```

<details>
<summary>Evaluate a trained checkpoint</summary>

Point `--checkpoint` to the RL run's saved checkpoint and provide your task scorer:

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

### 5. Analyze the results

These examples show how to generate analysis and experiment reports. Replace the
example files with your predictions and metric rows for your own runs.

```bash
vapa-analyze examples/tiny_analysis_manifest.json \
  examples/tiny_binary_predictions.jsonl runs/demo/analysis.json --content-kind public
vapa-report examples/tiny_experiment_report_manifest.json \
  examples/tiny_experiment_metrics.jsonl runs/demo/report.json --content-kind public
```

## Reproducing results

Use the [example configuration](examples/tiny_seed_audit.json) as a starting point
for comparing results across seeds:

```bash
vapa-audit-seeds examples/tiny_seed_audit.json examples/tiny_seed_metrics.jsonl \
  runs/demo/seed_audit.json
```

See `vapa-audit-seeds --help` for comparison options and `vapa-run-receipt --help`
for checking a run's configuration, logs, and checkpoint records.

## Code layout

| Component | Where to look |
| --- | --- |
| Data loading and preparation | [`src/vapa/data/`](src/vapa/data/) |
| Actions, tools, and memory | [`actions.py`](src/vapa/actions.py), [`environment/`](src/vapa/environment/) |
| Verifiers and replay | [`verifiers.py`](src/vapa/verifiers.py), [`rollouts.py`](src/vapa/rollouts.py) |
| Step advantages, SFT, and RL | [`src/vapa/training/`](src/vapa/training/) |
| Evaluation and statistical analysis | [`src/vapa/evaluation/`](src/vapa/evaluation/) |
| Versioned prompts and schemas | [`public_core_v1/`](src/vapa/releases/public_core_v1/) |

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check src tests scripts
ruff format --check src tests scripts
python scripts/validate_release.py
python -m build
```

To run just the CPU model integration test, install the training dependencies and
use `pytest -m model`.
