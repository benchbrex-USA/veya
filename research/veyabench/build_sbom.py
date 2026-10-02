"""Software bill of materials for the release environment (gate G14).

Walks the distributions installed in the running interpreter and records the
version, license, and source URL of each. Categories follow the dependency
closure of pyproject.toml: direct ``[project.dependencies]`` and their
transitive requirements are ``runtime``, ``[project.optional-dependencies]``
groups keep their extra name (e.g. ``train``), everything else installed is
``dev`` tooling, and the project distribution itself is ``self``. The output
is ``artifacts/release/sbom.json`` in the ``veya-sbom-v1`` schema.
"""
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from importlib import metadata

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(ROOT, "artifacts", "release", "sbom.json")

# The project itself is installed from the repo root (veya.egg-info). Script
# execution puts the script directory on sys.path instead, so add the root
# before scanning distributions or the project package is never discovered.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_NAME_HEAD = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXTRA_MARKER = re.compile(r"extra\s*==")

_LICENSE_CLASSIFIERS = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: BSD License": "BSD",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
}


def canonical(name):
    """PEP 503 normalized project name."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _requirement_name(requirement):
    match = _NAME_HEAD.match(requirement)
    return canonical(match.group(1)) if match else None


def declared_dependencies(pyproject_path):
    """Return (base dependencies, {extra: dependencies}) as canonical names."""
    with open(pyproject_path, "rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project") or {}
    base = [name for name in (_requirement_name(req) for req in project.get("dependencies") or []) if name]
    extras = {}
    for extra, requirements in (project.get("optional-dependencies") or {}).items():
        extras[extra] = [name for name in (_requirement_name(req) for req in requirements) if name]
    return base, extras


def base_requirements(dist):
    """Requirements of one installed distribution, ignoring extra-gated ones."""
    names = set()
    for requirement in dist.requires or []:
        if _EXTRA_MARKER.search(requirement):
            continue
        name = _requirement_name(requirement)
        if name:
            names.add(name)
    return names


def license_of(meta):
    """License-Expression first, then a short License field, then classifiers."""
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    plain = (meta.get("License") or "").strip()
    if 0 < len(plain) <= 40 and plain.upper() not in ("UNKNOWN", "NONE"):
        return plain
    for classifier in meta.get_all("Classifier") or []:
        if classifier in _LICENSE_CLASSIFIERS:
            return _LICENSE_CLASSIFIERS[classifier]
        if classifier.startswith("License :: OSI Approved :: "):
            return classifier.rsplit(" :: ", 1)[-1]
    return "UNKNOWN"


def source_url(meta, name):
    """Project-URL homepage, then the Home-page field, then any project URL."""
    labelled = {}
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        if url.strip():
            labelled.setdefault(label.strip().lower(), url.strip())
    if "homepage" in labelled:
        return labelled["homepage"]
    home_page = (meta.get("Home-page") or "").strip()
    if home_page:
        return home_page
    for label in ("source", "repository", "code", "github"):
        if label in labelled:
            return labelled[label]
    if labelled:
        return next(iter(labelled.values()))
    return "https://pypi.org/project/%s/" % name


def classify(dists, base, extras):
    """Map canonical name -> category via dependency closure over installed dists.

    Only installed distributions are traversed, so requirements that are absent
    from this environment (platform-specific wheels, undeclared tooling) do not
    appear. Runtime wins over extras, extras win over dev tooling.
    """
    installed = {}
    for dist in dists:
        name = dist.metadata["Name"]
        if name:
            installed.setdefault(canonical(name), dist)
    membership = {}

    def closure(seeds, label):
        queue = [name for name in seeds if name in installed]
        while queue:
            name = queue.pop()
            if name in membership:
                continue
            membership[name] = label
            queue.extend(
                dep for dep in base_requirements(installed[name])
                if dep in installed and dep not in membership
            )

    closure(base, "runtime")
    for extra in sorted(extras):
        closure(extras[extra], extra)
    for name in installed:
        if name == "veya":
            membership[name] = "self"
        elif name not in membership:
            membership[name] = "dev"
    return membership


def build():
    base, extras = declared_dependencies(os.path.join(ROOT, "pyproject.toml"))
    dists = list(metadata.distributions())
    membership = classify(dists, base, extras)
    declared = set(base) | {name for requirements in extras.values() for name in requirements}
    packages = []
    seen = set()
    for dist in dists:
        meta = dist.metadata
        name = meta["Name"]
        if not name:
            continue
        key = canonical(name)
        if key in seen:
            continue
        seen.add(key)
        packages.append({
            "name": name,
            "version": dist.version,
            "license": license_of(meta),
            "source": source_url(meta, key),
            "category": membership.get(key, "dev"),
            "direct": key in declared,
        })
    packages.sort(key=lambda package: package["name"].lower())
    return {
        "format": "veya-sbom-v1",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": platform.python_version(),
        "packages": packages,
    }


def main():
    sbom = build()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(sbom, handle, indent=2)
        handle.write("\n")
    by_category = {}
    for package in sbom["packages"]:
        by_category[package["category"]] = by_category.get(package["category"], 0) + 1
    print("wrote %s: %d packages %s" % (OUTPUT, len(sbom["packages"]), by_category))


if __name__ == "__main__":
    main()
