# Veya Codex operating contract

## Outcome

Develop Veya as one coherent decision-model codebase while using external
research sources only as evidence, teachers, benchmark targets, or legally
compatible implementation references.

## Naming boundary

First-party runtime code under `veya/` must use Veya naming only. Do not place
upstream project or model names in Python identifiers, docstrings, comments,
config constants, user-facing error messages, or package metadata.

Upstream names and URLs are allowed only in:

- `research/provenance/` for attribution, license records, and reproducibility;
- `.veya-foundry/` for ignored local source snapshots and generated bridges;
- test fixtures whose purpose is provenance/audit validation.

Never remove or rewrite upstream copyright, license, or NOTICE material in a
source snapshot. The Veya-only naming rule does not override legal attribution.

## Source integration

1. Read `research/provenance/sources.json` before using any external source.
2. `integration_policy=integrate`: implementation ideas may be adapted subject
   to the recorded license and any dependency/model/data licenses.
3. `integration_policy=reference_only`: do not copy source code. Use public
   behavior, documentation, papers, independently generated outputs, or a clean
   reimplementation from specifications.
4. Keep external dependencies process-isolated behind the Veya JSON-lines
   bridge. Do not import external research packages into Veya runtime modules.
5. Prefer source-specific bridge code inside ignored `.veya-foundry/` workspaces.
6. Every ported mechanism needs a Veya-native test and a short provenance note.

## Architecture

All experts implement the Veya Foundry contract:

`VeyaRequest -> VeyaExpert.predict -> VeyaResponse`

Use `VeyaExpertPool` for parallel evaluation and disagreement mining. Convert
multi-expert consensus into Veya training records with the distillation module.

## Benchmark integrity

- Never train on frozen private-test examples.
- Record dataset version, source revision, prompt/schema hash, hardware, dtype,
  batch size, context length, latency percentiles, accuracy, Brier score, and
  calibration error for every headline run.
- Compare models only on identical examples and scoring rules.
- Prefer paired confidence intervals for model-to-model comparisons.

## Verification

For every change:

1. run focused tests for the touched Veya modules;
2. run the Veya naming audit;
3. inspect the diff for copied source names or license leakage;
4. broaden tests when public APIs, packaging, or training formats change.

Do not claim a benchmark or integration works unless the relevant command was
actually executed and its output observed.
