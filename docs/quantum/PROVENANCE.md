# Provenance and licensing boundary

This file is a note for legal review. It does not relicense the repository, and it does not rewrite `artifacts/release/license_inventory.json` or any frozen audit artifact.

## First party

The `quantum/` package, its tests, and these notes are BenchBrex LLC integration code. They call Veya over the existing model HTTP API. They do not contain a copy of the Veya weights.

The tree still carries `LICENSE` (Apache-2.0) and `license = { text = "Apache-2.0" }` in `pyproject.toml`. This change does not add a second open-source grant, and it does not remove the grant that is already in the tree. Whether the project is held out as proprietary, and whether that Apache file still applies to new files, is a legal question. Do not describe this change as having closed that question.

## NVIDIA Ising

`nvidia/ising-calibration-1.5-31b` is a third-party model. This repository does not vendor its weights, its container, or its source. The runtime calls an operator-configured NIM or OpenAI-compatible HTTP endpoint. Hosted calls use `https://integrate.api.nvidia.com/v1` only when that URL is configured. Self-hosted calls use whatever base URL the operator set.

NVIDIA's public model card describes the model as a multimodal model built on Gemma 4 31B, served as a NIM. That description is not a BenchBrex copyright claim, and the Gemma or NVIDIA license texts are not copied here. Those texts were not fetched into the tree.

## What is not cleared

Legal review still needs to decide:

- NVIDIA API and NIM terms for sending laboratory plots to a hosted endpoint.
- Whether Ising outputs, or outputs of a Gemma-based model, may be stored and used as distillation targets.
- The project's own Apache-2.0 file versus a proprietary BenchBrex posture.
- The safety-guard model's output contract. The adapter accepts only `{"flagged": true|false}`. A different contract defers. The guard is not a calibration oracle, and it is off by default.

No NVIDIA notice was stripped, because no NVIDIA notice file was added. Do not relabel Ising, Gemma, or NIM as BenchBrex technology.

## Historical inventory

`artifacts/release/license_inventory.json` remains the 2026-09-29 dependency and dataset inventory. It was not edited to describe this integration. The Ising model is an external service, not a redistributed asset in that inventory.
