# Research note: a compact multimodal Veya student

Status: **not implemented**. The served Veya backbone stays text-only. This note records a later distillation target. It is not a claim that the student exists or that it matches Ising.

## Target

```text
calibration image
    |
    v
compact vision encoder
    |
    v
projection into the Veya representation
    |
    v
existing choice / score / noul heads
```

The student would be small relative to the 31B Ising specialist. Ising would remain an external teacher. Training rows would come from `assemble_teacher_record`: human-verified labels, separate from the teacher JSON, with dataset version and hashes. Unreviewed Ising output and unreviewed hard negatives would stay out of that set.

A vision tower, a projection, and any fusion belong in a new experimental module if a benchmark is funded. They do not belong in the audited checkpoint, and they do not belong in this change.

## Evaluation that would be required first

The offline harness in `quantum/benchmark.py` can score later prediction files. It does not call a model. Modes to compare, once labeled data exists:

1. Structured measurements into Veya, with no image.
2. Ising alone, judged against instrument values and human labels, not against another language model.
3. Ising, then Veya.
4. Ising with bounded in-context examples, then Veya.
5. Ising, text critics, then Veya.

Hold out by device, experiment family, hardware generation, lab, or time. Do not fit on that holdout. Plot perturbations in `quantum/perturb.py` change presentation only. Ground truth has to be identical before and after a perturbation, and stability is a separate measurement.

An LLM judge may be reported as a supplement. It is not the scientific label. Parameter error is computed against the numeric record. A safety classifier is not a test of whether a fit is physically right.

No number in this note is a measured result.
