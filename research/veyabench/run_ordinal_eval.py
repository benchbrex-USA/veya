"""Ordinal Score check.

Exits 2 when no dataset is configured. Does not print invented accuracy, MAE,
or calibration numbers. Production Score remains categorical.
"""
import os
import sys

from veya.experimental.ordinal_data import OrdinalDataUnavailable, load_ordinal_table


def main():
    path = os.environ.get("VEYA_ORDINAL_CSV", "")
    order = os.environ.get("VEYA_ORDINAL_LABELS", "")
    labels = [piece for piece in order.split(",") if piece]
    try:
        rows = load_ordinal_table(path, labels)
    except OrdinalDataUnavailable as exc:
        print("ORDINAL SCORE: IMPLEMENTED / LOCALLY VERIFIED / NOT RESEARCH VALIDATED / NOT PRODUCTION")
        print(str(exc))
        return 2
    print("ordinal rows", len(rows))
    print("A comparison against categorical Score is not implemented in this command until a reviewed training run exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
