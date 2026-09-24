# Veya research provenance

This directory is the only committed location where upstream project names and
repository URLs are intentionally preserved. Veya needs this provenance for
license compliance, reproducibility, scientific attribution, and source audits.

The production Python package under `veya/` must remain Veya-only. Upstream
source snapshots are synchronized into `.veya-foundry/sources/source_###` and
are excluded from packaging and version control.

A source marked `reference_only` must not be copied into Veya unless a later
review verifies an explicit compatible license. It may still be benchmarked or
studied through independently written bridges and public interfaces.
