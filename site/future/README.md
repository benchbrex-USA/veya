# Not deployed

This Worker, its Wrangler config, and the D1 migration are not part of the investor deployment. They do not issue project keys, and the edge Worker does not verify them.

The only deploy target is the repository root `wrangler.jsonc` (`veya-staging`): one Worker, one `veya-core` container, and the private R2 checkpoint.
