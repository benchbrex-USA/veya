---
kind: external_dependency
name: Hugging Face Hub — model weights and demo space host
slug: huggingface-hub
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
---

Veya loads its three checkpoints (`veyadynamics/veya`, `veyadynamics/veya-multilingual`, `veyadynamics/veya-typed-decisions`) from Hugging Face Hub via `huggingface_hub` and publishes fine-tuned results back there. The release bundle also resolves models by repo id; the web demo is hosted at `veyadynamics/veya-demo`. Verify exact download/push API against the current `huggingface_hub` docs.