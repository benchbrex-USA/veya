"""Ordinal Score dataset adapter.

Production Score stays the categorical head until a real ordinal table is
present and a held-out comparison is recorded. This loader does not download
data and does not invent labels.
"""
import csv
import os


class OrdinalDataUnavailable(FileNotFoundError):
    """No genuine ordinal table is on disk."""


def load_ordinal_table(path, label_order):
    """Load a CSV with columns ``text`` and ``label``.

    ``label_order`` is the low-to-high level names. Rows whose label is outside
    that list are rejected rather than coerced.
    """
    if not path or not os.path.isfile(path):
        raise OrdinalDataUnavailable(
            "ordinal Score has no dataset at %r. Production Score stays categorical. "
            "Expected a CSV with columns text,label and an explicit low-to-high label order."
            % (path,)
        )
    if not label_order or len(label_order) < 2:
        raise ValueError("ordinal label_order needs at least two levels")
    index = {name: i for i, name in enumerate(label_order)}
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("ordinal CSV must have columns text,label")
        for row in reader:
            label = row["label"]
            if label not in index:
                raise ValueError("label %r is not in the ordinal order" % label)
            rows.append((row["text"], index[label], label))
    if len(rows) < 32:
        raise ValueError("ordinal CSV has %d rows; refusing a comparison below 32" % len(rows))
    return rows
