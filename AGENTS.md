# Engineering guidelines

## Ownership

Keep VAPA responsible for the research method: environment semantics, verifiers,
fork/replay, comparison groups, advantages, eligible-token accounting, and Eq. 9.
Use PyTorch, Transformers, and PEFT for tensors, models, adapters, and optimizers;
use vLLM for optional high-throughput generation. Prefer documented library APIs
over custom implementations of capabilities the dependency already provides.

The current execution path is synchronous and single-process. For multi-GPU work,
evaluate an established stack such as verl with FSDP2 and vLLM before adding custom
worker scheduling, distributed collectives, IPC, or checkpoint sharding. Do not
claim an integration exists until it has executable coverage on the target stack.

## Algorithm compatibility

- Preserve prompt token IDs, legal-token support, fork replay bytes, loss masks,
  group membership, curriculum admission, and sampled-token charges.
- Keep the loss as the token-mean log-policy term plus exact KL(actor || reference)
  over the legal vocabulary. PPO clipping, sampled KL, importance correction,
  asynchronous stale-policy sampling, and changed normalization are experimental
  variants, not infrastructure-only substitutions. Obtain agreement before adding them.
- Accumulate gradients using the global eligible-token denominator. Test unequal
  group sizes against a dense reference, including gradients and frozen-reference behavior.
- Keep one implementation of shared semantics; serial and batched execution should
  differ only where scheduling requires it. Preserve the serial API's call order.

## Validation and operations

- Separate dependency-light contracts, real CPU model tests, and opt-in CUDA tests.
  Run pytest and Ruff for changed code, then the complete suite before release.
- Backend changes need tests for token identity, masks, adapter-version barriers,
  failure cleanup, and resume. Pin version-specific integrations and isolate any
  unavoidable private-API use. Do not broaden supported versions without testing.
- Record host phase timings and behavior-policy versions. Benchmark on the actual
  GPU/model before claiming throughput, memory capacity, or batch-invariant generation;
  a per-request seed alone does not prove those properties.
- Keep raw patient data and prompts on approved storage; do not add external telemetry
  by default. Preserve accurate result provenance and anonymous release metadata.
- When package code changes, rerun the demo and tiny training publication commands,
  validate release fingerprints, and build the package. Do not invent measurements.
- Keep the README short and task-oriented. Licensing remains pending.
