# Updating `/Volumes/Parth Patel/Veya` safely (run these yourself in Terminal on the Mac)

The drive-only work is already backed up on GitHub (`backup/local-drive-20261007`, commit `11bb9dd`) and in R2. These steps keep a local copy too.

```bash
cd "/Volumes/Parth Patel/Veya"

# 0. Ignore local wrangler dev state (the miniflare sqlite in .wrangler/) before stashing
grep -qxF '.wrangler/' .gitignore || printf '\n# local wrangler/miniflare dev state\n.wrangler/\n' >> .gitignore

# 1. Update the remote refs (fetch only; your files stay as they are)
git fetch origin --prune

# 2. Keep a pointer to where you are now, and stash ALL uncommitted work, including untracked files
git branch local/pre-sync-20261007 HEAD          # = a76bc65
git stash push -u -m "drive work 2026-10-07 (mirrors origin/backup/local-drive-20261007)"
git stash list                                    # expect stash@{0}

# 3. Fast-forward production/veya-dynamics: a76bc65 -> 426fd88 (local has no commits of its own)
git switch production/veya-dynamics
git merge --ff-only origin/production/veya-dynamics

# 4. Bring the work back
git stash pop
```

**Expected conflict: only `.github/workflows/ci.yml`.** I checked this by applying the backup commit onto `426fd88` on the box. `.gitignore` merges cleanly. README, pyproject, hf-space and the docs don't conflict.

Resolve it by keeping both sides. Keep the upstream block (`Cloudflare config separation…`, `Full offline test suite`, and the whole `worker-tests:` job). Put your two steps (`Quantum boundary tests…` and `Free-tier Cloudflare proxy tests`) **directly after `Full offline test suite` and before `worker-tests:`**, so they stay inside the same job. A pre-resolved file is in the backup branch at `_salvage/ci.yml.resolved-onto-426fd88`:

```bash
git show origin/backup/local-drive-20261007:_salvage/ci.yml.resolved-onto-426fd88 > .github/workflows/ci.yml
git add .github/workflows/ci.yml
git restore --staged .        # stash pop leaves the resolved files staged; unstage them to keep the working tree as before
git stash drop                # only after `git status` shows your files are back (a conflicted pop does NOT drop the stash)
```

Optional tidy-ups:
- `.gitignore` now has `.dev.vars` / `.dev.vars.*` twice (your edit added them in two places). Delete one pair.
- `main` (local `c96d5f7`) can simply be fast-forwarded: `git switch main && git merge --ff-only origin/main && git switch production/veya-dynamics`.
- If anything goes wrong: `git switch -C production/veya-dynamics local/pre-sync-20261007`. The stash is still there until you drop it, and every file is also on `origin/backup/local-drive-20261007`. The two large `.npz` files are in R2.
- The copilot worktree (`benchbrex-usa-remove-website`, 64 changes) and the `refs/copilot*` snapshots are untouched. Their patches and bundle are in `_salvage/` on the backup branch.
