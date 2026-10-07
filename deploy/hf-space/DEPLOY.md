# Deploying the Veya serving stack

This is the operator runbook. It lives in `deploy/`, which `assemble.py` never
copies into the upload tree, so nothing written here is published.

The reader-facing description of the deployed service is
[`README.md`](./README.md) in this directory — that file becomes the Space
repository root README, and Hugging Face parses its YAML frontmatter for
`sdk: docker` and `app_port: 7860`.

## What this deploys

Two artifacts, deliberately separated:

| Artifact | Visibility | Contents |
| --- | --- | --- |
| The Space repository | see "The visibility decision" below | `veya/`, `web/`, `space/`, small `models/production/` files, four `artifacts/release/` JSONs, `requirements.lock`, `pyproject.toml`, `setup.py`, `README.md`, `Dockerfile` |
| The checkpoint | stays private | `model.safetensors`, 804 MB, fetched at boot from a pinned revision |

The checkpoint is never committed to the Space. `space/fetch_checkpoint.py`
downloads that one file at boot and asserts
`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c` — the
`base_checkpoint_hash` the release manifest already pins — before the model
service is allowed to start. `veya.release_bundle.load_bundle` hashes the same
file again independently. A wrong or truncated download fails fast with both
digests printed rather than ending a four-minute cold start in an opaque load
error.

Nothing is editable here. Every manifest-pinned file is copied byte-for-byte;
if a digest would change, the release has to be re-minted under the
repository's change protocol first, not patched during deployment.

## Preconditions — both were open when this was written

**1. The Hugging Face namespace.** The repository assumes `veyadynamics`
everywhere: `pyproject.toml`'s Homepage, the README badges, `MASTER.md`,
`tests/test_download.py`, `tests/test_router.py`, and
`space/fetch_checkpoint.py`'s `DEFAULT_REPO`. As of this writing that namespace
does not resolve:

```
GET https://huggingface.co/veyadynamics                                -> 404
GET https://huggingface.co/api/organizations/veyadynamics/overview     -> 404
GET https://huggingface.co/api/models?author=veyadynamics              -> 200 []
GET https://huggingface.co/api/models/veyadynamics/veya                -> 401
```

The control probes for `answerdotai` and `huggingface` return 200 on the same
two endpoints, so this is not a broken probe. The 401 on the model repo is
ambiguous — Hugging Face returns 401 for a missing repo as readily as for a
private one, to avoid leaking existence. Either the namespace has to be
created, or the real name substituted. Set `ORG` below to whatever is correct.

`space/fetch_checkpoint.py` reads `VEYA_CHECKPOINT_REPO` and
`VEYA_CHECKPOINT_REVISION` from the environment, so a namespace change needs no
code edit — but `DEFAULT_REPO` and the docs should be corrected too, or the
next operator hits the same wall.

**2. Authentication, and whether a free plan can create a Docker Space at
all.** The Hugging Face docs currently state, in a warning box on the Spaces
overview page:

> static Spaces are free for everyone. Gradio and Docker Spaces run on compute
> and require a paid plan to create: PRO for personal accounts, Team or
> Enterprise for organizations. Free personal accounts in good standing can
> still host up to 2 Gradio Spaces running on ZeroGPU.

The pricing page lists "Host ZeroGPU, Gradio & Docker Spaces" under **PRO
Account** features, which corroborates it. This has not been verified against a
live free account, because verifying it means attempting to create the Space,
which means being logged in. **Treat step 3 as the test.** If creation is
refused, see "If a free Docker Space is refused" before doing anything else.

Authenticate in your own terminal. Do not paste a token into a chat, a commit
message, or a shell history:

```bash
hf auth login     # prompts; the token is never echoed
hf auth whoami    # confirm the identity that owns ORG
```

## The runbook

```bash
export ORG=veyadynamics        # <- correct this first; see preconditions
export SPACE_ID="$ORG/veya-serve"
export SPACE_URL="https://${ORG}-veya-serve.hf.space"
export WORKER_ORIGIN="https://veya.veya-8d0.workers.dev"
```

### 1. Stage and verify the upload tree

```bash
python3 deploy/hf-space/assemble.py
python3 deploy/hf-space/assemble.py --check
```

`assemble.py` wipes and rebuilds `build/hf-space/`, then re-verifies every
manifest-pinned digest **against the staged bytes** rather than the working
tree, using the production helpers (`veya.release_bundle.resolve_manifest_path`,
`veya.shortlist_artifact.file_sha256`) so the check cannot drift from the gate
it is meant to mirror. It also asserts the checkpoint is *absent*, and re-checks
every staged path against `DENY_GLOBS`.

Expected output, from the last verified run:

```
staged: build_inputs=3, model_files=9, release_artifacts=4, space_scaffolding=4, veya=24, web=23
staged 67 files, 7.18 MB -> build/hf-space
verified 24 manifest digests against the staged bytes
checkpoint: absent as required (boot fetch asserts 891102d372688fc2…)
```

It then prints the manifest references it deliberately did not stage — dataset
lineage, blind-evaluation inputs, and the phase-9 research heads.
`load_release_manifest` does not verify those and nothing in the serving path
reads them. They are printed, not silently dropped, so a future manifest change
that starts depending on one shows up as a new line rather than a boot failure.

`--check` is the guard against contaminating the tree. Running anything that
imports the staged package writes `__pycache__` into it, and `--check` will then
fail with `staged file matches deny pattern '*.pyc'`. That is the deny list
working. Rebuild rather than loosening it.

**Re-run `assemble.py` after editing `README.md`.** The staged copy is a file
copy, not a symlink, and a stale README is easy to miss.

### 2. Prove the staged bytes actually serve

Staging checks digests; it does not check that the artifact runs. Verify the
tree end-to-end before publishing it, by hardlinking the checkpoint in (no
843 MB copy), importing the *staged* package, and asserting that is what got
imported:

```python
assert Path(veya.__file__).parent == STAGE / "veya"
bundle = load_release_manifest(str(STAGE / "artifacts/release/manifest.json"),
                               device="cpu", root=str(STAGE), timings=timings)
```

Then exercise all three primitives and both routing paths, and unlink the
hardlink in a `finally`. The last verified run latched in 68.6 s and returned
`dispute a duplicate charge` on both the packed route (5 candidates) and the
shortlist route (25 candidates), `noul = 0.827`, a float score, and
`input_tokens = 206`.

This is the step that distinguishes "the digests match" from "the deployment
works". Do not skip it.

### 3. Create the Space

```bash
hf repos create "$SPACE_ID" --type space --sdk docker --flavor cpu-basic --public --exist-ok
```

Notes on the flags, verified against `hf` 1.33.0:

- The hardware flag on `repos create` is `--flavor`, not `--hardware`.
  (`hf spaces settings` uses `--hardware`. They are not the same flag.)
- Do **not** pass `--region`. Its help text says "Requires Team plan or above".
- There is no `--hardware` on create and no way to set sleep behaviour here.

If this step fails with a plan or billing error, stop — see "If a free Docker
Space is refused".

### 4. Upload

```bash
hf upload "$SPACE_ID" build/hf-space . --type space \
  --commit-message "Deploy Veya serving stack 0.3.5-audit-baseline.1"
```

The upload tree is 7.18 MB across 67 files. The largest file is
`models/production/tokenizer/tokenizer.json` at 3,583,228 bytes, so nothing
crosses the 10 MB LFS threshold and no LFS setup is needed. Confirm before
pushing:

```bash
find build/hf-space -type f -size +9M          # must be empty
find build/hf-space \( -name '*.safetensors' -o -name '*.pyc' \)   # must be empty
```

### 5. Configure the runtime

Variables are public and non-sensitive:

```bash
hf spaces variables add "$SPACE_ID" \
  -e VEYA_ALLOWED_ORIGINS="$WORKER_ORIGIN" \
  -e VEYA_DEVICE=cpu \
  -e VEYA_WEB_HOST=0.0.0.0 \
  -e VEYA_WEB_PORT=7860
```

`VEYA_PUBLIC_ORIGIN` does not need to be set: `space/boot.py` derives it from
the runtime-injected `SPACE_HOST`, which keeps it from being a hand-copied
string that can drift from the actual Space name. Set it explicitly only if you
are fronting the Space with a custom domain.

The checkpoint credential is a **secret**, not a variable. Note that
`hf spaces secrets add -s NAME` reads the value from an *environment variable
of that name* — it does not prompt, and `-s NAME=value` writes the value into
your shell history. So:

```bash
read -rs VEYA_HF_TOKEN && export VEYA_HF_TOKEN
hf spaces secrets add "$SPACE_ID" -s VEYA_HF_TOKEN
unset VEYA_HF_TOKEN
```

The token needs read access to the checkpoint repo and nothing more.

Hugging Face also injects `HF_TOKEN` into Spaces owned by an identity that has
one, and `fetch_checkpoint.py` falls back to it (`VEYA_HF_TOKEN`, then
`HF_TOKEN`, then `HUGGING_FACE_HF_TOKEN`). **Do not rely on that fallback.**
Set the secret explicitly; the fallback is a convenience for same-owner Spaces,
not a guarantee, and the boot log names the exact variable to set if all three
are empty.

### 6. Watch the build

```bash
hf spaces logs "$SPACE_ID" --build -f      # image build
hf spaces wait "$SPACE_ID" --timeout 20m   # exits 0 iff RUNNING
```

A `BUILD_ERROR` is almost always the torch install. The Dockerfile installs
`torch==2.14.0` from the PyTorch CPU index *before* `requirements.lock`, because
the lock was resolved on macOS and pins `torch==2.14.0` with zero `nvidia-*`
entries. Installing it as-is on Linux would pull the 554 MB CUDA wheel plus
roughly 3 GB of nvidia runtime packages. The CPU wheel for the same pinned
version is 196,227,330 bytes, and PEP 440 lets a specifier without a local
segment match a wheel that has one, so `torch==2.14.0` stays satisfied.

### 7. Verify from outside

```bash
curl -s "$SPACE_URL/healthz"          # 200 within seconds of boot
curl -s "$SPACE_URL/startupz"         # which component, if any, failed the gate
curl -s "$SPACE_URL/readyz"           # 503 until the release latches, then 200
curl -s "$SPACE_URL/api/version"      # release bindings the gateway observed
```

`/readyz` returning 503 early is the design, not a fault. `space/boot.py` binds
the gateway first precisely so the runtime's `app_port` probe succeeds while the
804 MB download and the ~80 s ModernBERT-large load are still running. Only
`/startupz` naming a component means something actually failed.

Then a real inference. Choice `criteria` must be an **object** of `id →
description` at the gateway, not a list — `web/contracts.py` rejects the list
form even though the internal `ReleaseBundle.public_choice` accepts it:

```bash
curl -s "$SPACE_URL/api/infer" -H 'content-type: application/json' -d '{
  "primitive": "choice",
  "state": "Customer was charged twice for the same subscription in March.",
  "instructions": "Which intent best matches this message?",
  "criteria": {
    "duplicate_charge": "dispute a duplicate charge",
    "cancel_subscription": "cancel the subscription"
  }
}'
```

Expect `decision: "answer"`, `answer: "duplicate_charge"`, a probability
distribution summing to one, `route: "packed"`, and `model: "Veya Core"`.

### 8. Repoint the Worker

`wrangler.jsonc` currently points at a dead ephemeral tunnel:

```jsonc
"vars": { "VEYA_ORIGIN": "https://educated-entertainment-bookmarks-aging.trycloudflare.com" }
```

Change it to `$SPACE_URL`, commit, and push — the GitHub-connected build
deploys automatically:

```bash
GIT_AUTHOR_EMAIL=261865777+benchbrex-USA@users.noreply.github.com \
GIT_COMMITTER_EMAIL=261865777+benchbrex-USA@users.noreply.github.com \
  git commit -m "deploy: repoint VEYA_ORIGIN at the hosted Veya Space"
```

Set the author and committer identity per-command like that. Do not write it
into the repository's git config.

Then confirm the edge, with the laptop taken out of the path first:

```bash
curl -s "$WORKER_ORIGIN/__edge_health"    # {"edge":"ok","origin_configured":true}
curl -s "$WORKER_ORIGIN/api/version"
```

The Worker forwards the browser's `Origin` header untouched, so
`VEYA_ALLOWED_ORIGINS` on the Space must contain `$WORKER_ORIGIN` or every
model request comes back 400 `invalid_input`. This is the single most likely
misconfiguration and the least obvious from the Space side, because direct
`curl` has no `Origin` header and succeeds.

Stop the local stack before believing any of this:

```bash
pkill -f 'web.model_server'; pkill -f 'web.server'; pkill -f cloudflared
```

## Known constraints

**Cold start.** Free hardware sleeps when idle and
`hf spaces settings --sleep-time -1` says "Only available on upgraded hardware",
so sleep cannot be disabled on `cpu-basic`. Every wake is a full cold start:
804 MB download, ~11 s to hash it, ~80 s to load ModernBERT-large on CPU, then
the first `/version` probe. Measured locally, the release latched in 68.6 s from
a warm cache; `artifacts/release/runtime_audit.json` records 90.802 s and
1638.9 MB RSS. Budget three to four minutes in the Space and expect the first
visitor after an idle period to wait it out. `space/boot.py` allows 900 s.

**The rate limit is one shared budget, not per visitor.** The limiter is a
sliding window of 12 cost units per 60 seconds keyed on
`self.client_address[0]`, the raw TCP peer address. It ignores `X-Forwarded-For`
and every other forwarding header on purpose — those are client-controlled, and
trusting them would let any caller choose their own bucket. That is correct for
a directly-exposed server and wrong behind an ingress. Inside the container
every external request arrives from the same platform address, so the window is
effectively global: a crawler that finds the public Space URL spends the budget
the website needs, and legitimate traffic starts getting 429 without anyone
being malicious. `MAX_INFLIGHT = 2` bounds concurrency separately.

`web/README.md` already records this limitation — the limiter is process-local,
and public multi-instance deployment needs a shared limiter at the ingress. The
Cloudflare Worker *is* that ingress and is where real client IPs are available.
Fixing it there is the recommended follow-up. It has deliberately **not** been
done in `web/server.py`: that file is audited and tested, and although `web/*.py`
is not manifest-pinned, editing it silently would put the deployed gateway out
of step with the one the test suite covers.

**No inbound authentication.** The gateway has no bearer or API-key check.
`VEYA_MODEL_TOKEN` is outbound, to the loopback model service. The blast radius
is bounded by the request contract, the two-inference concurrency cap, the rate
limiter, and the model service never being published — and the shared rate
budget is the weakest of those, not the strongest.

## The visibility decision

`gh repo view benchbrex-USA/veya` reports `{"isPrivate":true}`. A **public**
Space would therefore publish `veya/` and `web/` source as the namespace's first
public artifact — new exposure, not re-publication. Both are proprietary (all rights reserved),
so a public Space would also be a decision to publish proprietary source code.

What a public Space does *not* expose: the checkpoint (private repo, fetched at
boot), the legal and licence material, the SBOM, the readiness matrix, the
activation procedure, the data lineage, the blind-evaluation inputs, the phase-9
research heads, the multilingual and typed-decisions checkpoints (647 MB and
807 MB, excluded), the internal `web/*.md` audit write-ups, and every
credential pattern in `.gitignore`. That exclusion set is `DENY_GLOBS` in
`assemble.py`, it fails the build rather than warning, and `--check` re-verifies
it against every staged file.

The middle option — **protected** visibility, which keeps source private while
the app stays reachable at `*.hf.space` — is exactly what this deployment wants
and is documented by Hugging Face as "part of PRO or Team & Enterprise plans".
It is not available on a free account either.

If the Space must be public and that is not acceptable, the alternative is a
private Space plus an `Authorization` header injected by the Worker. That path is
described here and **not implemented**, and Hugging Face's support for
authenticated Space-to-Space calls from a third-party proxy has not been
verified.

## If a free Docker Space is refused

The artifact is host-agnostic — it is a Dockerfile and a boot supervisor, not
Hugging Face machinery — so the work is not wasted. `space/boot.py` already
reads `PORT` before `VEYA_WEB_PORT`, and derives `VEYA_PUBLIC_ORIGIN` from an
environment variable with `SPACE_HOST` only as a convenience.

One portability fact is already verified: the PyTorch CPU index serves
`torch-2.14.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl`, so the exact pinned
version installs on ARM64 Linux too. An Ampere-based host needs no change to
the torch install line.

The constraint that rules most free tiers out is memory, not CPU: 1638.9 MB RSS
from `runtime_audit.json`. A 512 MB or 1 GB free container cannot hold
ModernBERT-large, and loading it in fp16 to fit would change the numerics the
release calibration was fitted against, which means re-auditing the release.
That is out of scope for a deployment task.

Candidates worth evaluating, none verified here:

| Host | Free tier | Verdict |
| --- | --- | --- |
| Hugging Face, PRO | $9/mo personal | Everything already built for it. Zero additional work. |
| Oracle Cloud Always Free | 4 ARM A1 cores, 24 GB, free forever | Fits comfortably; aarch64 wheel confirmed. Needs a credit card at signup, plus instance, DNS and TLS setup. |
| Render / Koyeb / Railway | 512 MB – 1 GB | Too little memory. |
| Google Cloud `e2-micro` | 1 GB | Too little memory. |
| Cloudflare Workers / Pages / Containers | — | Cannot run PyTorch; Containers are paid-only. |
| Local + `cloudflared` tunnel | free | Works today, but the URL is ephemeral and the laptop has to stay awake. This is what is being replaced. |

## Rollback

Rolling back the Space is a `git revert` on the Space repository — each push is
a commit, and every commit triggers a rebuild. Rolling back the *website* is
independent: restore the previous `VEYA_ORIGIN` in `wrangler.jsonc` and push.
The two are only coupled through that one variable.

Because the checkpoint is pinned by revision and asserted by digest, no rollback
can silently serve different weights. If the pinned revision is ever deleted
from the model repo, `fetch_checkpoint.py` fails, the model service is never
started, `/readyz` stays 503, and the container exits so the runtime restarts
it. That is the intended failure mode: unavailable, never wrong.

To take the Space down without deleting it:

```bash
hf spaces pause "$SPACE_ID"
```

Paused time is not billed, and the Worker starts returning 502
`origin_unreachable` immediately, which the website already handles as a
distinct state.
