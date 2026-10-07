---
kind: external_dependency
name: PyPI — distribution channel for the `veya` package
slug: pypi
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
---

The project is published as the `veya` Python package on PyPI (proprietary license, built with setuptools). Consumers install via `pip install veya`; the package exposes `veya.load`, `veya.predict_shortlist`, `veya.Router`, and the `veya.experimental` subpackage.