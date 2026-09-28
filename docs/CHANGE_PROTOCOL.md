# Incremental website change protocol

For each future request:

1. Parse the requested delta and its explicit exclusions.
2. Map it to token, content, component, page, or API ownership using `WEBSITE_ARCHITECTURE.md`.
3. Read the primary file and its direct dependents. Avoid a full-site rewrite for a local change.
4. Preserve unrelated route, model, navigation, and accessibility behavior.
5. Apply the smallest coherent patch. Reuse shared rendering primitives instead of copying them.
6. Run focused tests. Expand to the gateway suite when contracts or shared client code change.
7. View the actual affected page at a relevant desktop and mobile width. Check keyboard behavior for interactive edits.
8. Inspect the diff for unrelated changes, stale claims, mock output, secrets, and local paths.
9. Report exactly what changed, what was verified, and any remaining blocker.

Change impact graph: requested change → owning file → direct dependents → focused tests → rendered view. A copy change normally touches one content file. A shared color change normally touches a token. A route addition touches the registry and one content page. An API contract change can cross the browser client, gateway, service, and tests, and warrants the broader verification.

Do not regenerate the website, replace its runtime, or redesign unaffected pages to implement a small patch. Keep audited model claims tied to the current release manifest and benchmark artifact.
