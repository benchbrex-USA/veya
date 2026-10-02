"""Packaging metadata must match what the dependencies actually need (#34).

Text parsing, not tomllib: the floor is 3.10 and tomllib arrives in 3.11.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def version_tuple(text):
    return tuple(int(part) for part in text.split("."))


pyproject = read("pyproject.toml")
setup_py = read("setup.py")

requires_python = re.search(r'requires-python\s*=\s*"[>=~^]*\s*([\d.]+)"', pyproject)
check_true("pyproject/declares requires-python", requires_python is not None)
floor = version_tuple(requires_python.group(1)) if requires_python else (0, 0)

classifier_versions = [
    version_tuple(v)
    for v in re.findall(r'"Programming Language :: Python :: (\d+\.\d+)"', pyproject)
]
check_true("pyproject/advertises specific Python versions", len(classifier_versions) > 0)
below_floor = [".".join(str(p) for p in v) for v in classifier_versions if v < floor]
check("classifiers/none below requires-python", below_floor, [])

# Checkpoints run on answerdotai/ModernBERT-large, which transformers only knows from 4.48.
# The floor regex tolerates an optional upper bound, e.g. "transformers>=4.48.0,<6".
transformers_floor = re.search(r'"transformers>=([\d.]+)', pyproject)
check_true("pyproject/pins a transformers floor", transformers_floor is not None)
check_true(
    "transformers/floor covers ModernBERT",
    transformers_floor is not None and version_tuple(transformers_floor.group(1)) >= (4, 48),
    "ModernBERT support starts in transformers 4.48",
)

# Runtime deps must cap the next major so a surprise release cannot break a
# pinned checkpoint. Every core dependency line needs a ",<N" upper bound.
core_deps = ["torch", "transformers", "safetensors", "huggingface_hub", "numpy"]
uncapped = [
    dep
    for dep in core_deps
    if not re.search(r'"%s>=[\d.]+,<\d+"' % re.escape(dep), pyproject)
]
check("pyproject/caps every core dependency major", uncapped, [])

for field in ("python_requires", "install_requires", "classifiers"):
    check_true(
        "setup.py/does not duplicate %s" % field,
        field not in setup_py,
        "metadata belongs in pyproject.toml only",
    )

workflow = read(os.path.join(".github", "workflows", "ci.yml"))
ci_versions = [version_tuple(v) for v in re.findall(r'"(\d+\.\d+)"', workflow)]
stale = [".".join(str(p) for p in v) for v in ci_versions if v < floor]
check("ci/tests no Python below requires-python", stale, [])

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all packaging tests passed")


def test_recorded_checks_pass():
    assert not FAIL, "\n".join(FAIL)


if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)
