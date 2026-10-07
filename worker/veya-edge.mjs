/**
 * Customer Worker `veya` entrypoint (wrangler.jsonc).
 *
 * Serves the Veya website and same-origin API from Cloudflare. Inference goes to
 * the production Worker `veya-production` over the VEYA_API service binding with
 * the server-side project key VEYA_SITE_KEY. There is no dependency on the former
 * Mac / Tailscale Funnel origin. All logic lives in ./veya_site_edge.mjs.
 */

import benchmark from "../artifacts/release/blind_clinc150_val_result.json";
import siteConfig from "../web/static/site-config.json";
import indexHtml from "../web/static/index.html";
import { createHandler } from "./veya_site_edge.mjs";

const handle = createHandler({ indexHtml, siteConfig, benchmark });

export default {
  async fetch(request, env, ctx) {
    return handle(request, env, ctx);
  },
};
