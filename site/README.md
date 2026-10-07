# Veya Dynamics web

Public site, console, and docs for Veya. This package is not part of the model container.

```bash
npm install
npm test
npm run dev
npm run build
```

`npm run dev` serves the interface at `http://127.0.0.1:4173`. `/console` and `/docs` are the other surfaces. The Playground calls `/api/playground` on the console Worker. Without that Worker it reports that no decision was made.

The Worker config is `wrangler.jsonc`. It targets the veya Cloudflare account and the staging API service `veya-staging`. D1, Turnstile, and the service binding are declared and not deployed. HSTS is off. The local Turnstile site key in `vars` is Cloudflare's always-pass test key.

Do not copy an infrastructure bearer token into the browser. Customer keys use the `veya_test_` or `veya_live_` prefix and are stored as SHA-256 hashes.
