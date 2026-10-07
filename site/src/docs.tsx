import { API_ORIGIN, RELEASE } from "./release.ts";
import { Mark } from "./mark.tsx";

export function Docs({ onNavigate }: { onNavigate: (path: string) => void }) {
  return (
    <>
      <header className="site-header">
        <a className="brand" href="/" onClick={(event) => { event.preventDefault(); onNavigate("/"); }}><Mark /> VEYA DYNAMICS</a>
        <nav className="nav">
          <button className="btn" onClick={() => onNavigate("/console")}>Open console</button>
        </nav>
      </header>
      <article className="docs">
        <p className="kicker">Documentation</p>
        <h1>Veya API</h1>
        <p>These pages describe release {RELEASE.id}. They do not describe a generally available product.</p>
        <h2>Quickstart</h2>
        <p>Send POST {API_ORIGIN}/api/infer with a JSON object. Choice needs state, instructions, and a criteria object of id to description. Noul omits criteria. Score sends an ordered list of 2–10 labels.</p>
        <h2>Authentication</h2>
        <p>POST /api/infer uses Authorization: Bearer with a temporary demo token. That token is not the infrastructure credential and is not a project key. Console-issued veya_test_ and veya_live_ keys are not accepted. The Playground keeps the demo token in memory only.</p>
        <h2>Choice</h2>
        <p>{RELEASE.choice.note} Packed through {RELEASE.choice.packedMax} candidates. Shortlist K={RELEASE.choice.k} from {RELEASE.choice.shortlistAt} candidates.</p>
        <h2>Noul</h2>
        <p>{RELEASE.noul.note} The response answer is a probability from 0 to 1.</p>
        <h2>Score</h2>
        <p>{RELEASE.score.note} Probabilities are keyed by level index. The legend maps those indexes to labels.</p>
        <h2>Errors</h2>
        <p>Validation failures use code invalid_input and error_code VALIDATION_ERROR. Rate limits use 429, code rate_limited, and Retry-After. Overload uses model_busy. A defer is a successful decision, not an HTTP error.</p>
        <h2>Rate limits</h2>
        <p>POST /api/infer is limited by the demo credential: about 20 requests a minute, and about 6 a minute for heavier requests. The isolate-local limiter is only a fallback when the Cloudflare rate-limit binding is absent, and it uses that same identity. Do not trust X-Forwarded-For.</p>
        <h2>Versioning</h2>
        <p>The served release id is {RELEASE.id}. Architecture {RELEASE.architecture}. A version response from the model gateway exposes release and checkpoint hashes after the model is ready. /readyz stays 503 until then.</p>
        <h2>Security</h2>
        <p>Model weights stay in private R2. Static responses send a content security policy without unsafe-inline, nosniff, frame-ancestors none, and X-Robots-Tag noindex. Worker responses for the API and health routes send the same noindex header. HSTS is off until a production hostname is confirmed.</p>
        <h2>Release status</h2>
        <p>{RELEASE.verdict} Automation is {RELEASE.automation}.</p>
      </article>
    </>
  );
}
