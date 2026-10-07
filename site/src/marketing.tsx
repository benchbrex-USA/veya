import { useState } from "react";
import { API_ORIGIN, BENCHMARK, RELEASE } from "./release.ts";
import { Lattice, Mark } from "./mark.tsx";
import { toggleFaq } from "./security.ts";

const FAQ = [
  ["What is Veya?", "Veya is a typed decision model. One forward pass scores a structured question. It does not generate free text."],
  ["What is Veya Core?", `Veya Core is release ${RELEASE.id}. English Choice is a ${RELEASE.choice.status}. It is not a release candidate.`],
  ["Which primitives are available?", "The manifest serves choice, noul, and categorical score. Noul and categorical score are research only. Ordinal score, multilingual primitives, and Phase 9 are not served."],
  ["How does high-cardinality Choice work?", `One to ${RELEASE.choice.packedMax} candidates use the packed decision head. ${RELEASE.choice.shortlistAt} or more use a learned shortlist, keep K=${RELEASE.choice.k}, then that same head.`],
  ["Are probabilities calibrated?", "Choice has packed and shortlist calibration artifacts. Noul and categorical Score use checkpoint temperature only. Automation is disabled, so a probability is not a production risk guarantee."],
  ["When does Veya defer?", "The model can return decision=defer. Confidently detected non-English text is deferred as an unsupported language. English is the only served language."],
  ["Which languages are supported?", "English only in this release."],
  ["How is API data handled?", "The console does not store raw state or candidate text for its history. Usage telemetry records primitive, status, route, release, and a latency bucket."],
  ["How do API keys work?", "This deployment uses a temporary demo token on POST /api/infer. It is not a project key. The edge does not accept console-issued project keys."],
  ["Can I try Veya without writing code?", "The console Playground sends POST /api/infer. It returns a decision only when Veya Core is ready. It does not invent a result. The demo token stays in memory for that page."],
] as const;

export function Marketing({ onNavigate }: { onNavigate: (path: string) => void }) {
  const [open, setOpen] = useState<string | null>(null);
  const [menu, setMenu] = useState(false);
  const [tab, setTab] = useState<"curl" | "python" | "ts">("curl");
  return (
    <>
      <header className="site-header">
        <a className="brand" href="#top"><Mark /> VEYA DYNAMICS</a>
        <button className="menu-button" aria-expanded={menu} onClick={() => setMenu((value) => !value)}>Menu</button>
        <nav className={menu ? "nav open" : "nav"} aria-label="Primary">
          <a href="#veya">Veya</a>
          <a href="#developers">Developers</a>
          <a href="#research">Research</a>
          <button className="linkish" onClick={() => onNavigate("/docs")}>Docs</button>
          <a href="#company">Company</a>
          <button className="btn" onClick={() => onNavigate("/console")}>Open console</button>
        </nav>
      </header>
      <main id="top">
        <section className="hero">
          <div className="hero-inner">
            <Lattice />
            <p className="kicker">Veya Core</p>
            <h1>Veya is a decision API for intelligent systems</h1>
            <p className="lede">Fast, calibrated decisions over structured choices, scores, and routing. English Choice in this release is a research release, not a production certification.</p>
            <div className="actions">
              <a className="btn solid" href="#developers">Start building</a>
              <button className="btn" onClick={() => onNavigate("/console")}>Open console</button>
              <button className="btn quiet" onClick={() => onNavigate("/docs")}>Docs</button>
            </div>
          </div>
        </section>
        <section className="section" id="veya">
          <p className="kicker">Served primitives</p>
          <h2>What this release actually scores</h2>
          <div className="caps">
            <article className="cap">
              <div className="status">{RELEASE.choice.status}</div>
              <h3>Choice</h3>
              <p>Pick one candidate. Small sets stay packed. Large sets shortlist to K={RELEASE.choice.k}.</p>
            </article>
            <article className="cap">
              <div className="status">{RELEASE.noul.status}</div>
              <h3>Noul</h3>
              <p>A single probability. No separate audited calibration artifact ships with this release.</p>
            </article>
            <article className="cap">
              <div className="status">{RELEASE.score.status}</div>
              <h3>Score</h3>
              <p>Categorical labels only. Ordinal Score is research and is not served.</p>
            </article>
          </div>
        </section>
        <section className="section">
          <p className="kicker">Example workflows</p>
          <h2>How a team can use a typed decision</h2>
          <div className="uses">
            <article className="use"><h3>Routing</h3><p>Select a capability before a more expensive model call. The gateway accepts a bounded candidate map, not an open prompt.</p></article>
            <article className="use"><h3>Selective action</h3><p>Read the probability and the defer flag. Automation is disabled in this release, so nothing here is an approval to act.</p></article>
            <article className="use"><h3>Agent control</h3><p>Choose among tools or escalation paths when the options are an explicit set.</p></article>
            <article className="use"><h3>Specialist handoff</h3><p>Keep domain experts outside the decision head. Veya scores the typed question. It does not replace those systems.</p></article>
          </div>
        </section>
        <section className="section" id="research">
          <p className="kicker">Veya Core</p>
          <h2>{RELEASE.id}</h2>
          <table className="meta">
            <tbody>
              <tr><th>Status</th><td>{RELEASE.verdict}</td></tr>
              <tr><th>Architecture</th><td>{RELEASE.architecture}</td></tr>
              <tr><th>Scorer</th><td>{RELEASE.scorer}</td></tr>
              <tr><th>Languages</th><td>English</td></tr>
              <tr><th>High-N</th><td>{RELEASE.choice.shortlistAt}+ candidates, K={RELEASE.choice.k}, then the packed head</td></tr>
              <tr><th>Automation</th><td>{RELEASE.automation}. {RELEASE.automationReason}</td></tr>
            </tbody>
          </table>
        </section>
        <section className="section">
          <p className="kicker">Sealed evaluation</p>
          <h2>One blind Choice measurement</h2>
          <table className="meta">
            <tbody>
              <tr><th>Dataset</th><td>{BENCHMARK.dataset} {BENCHMARK.split}, {BENCHMARK.examples} examples, {BENCHMARK.candidates} candidates</td></tr>
              <tr><th>Recall@8</th><td>{BENCHMARK.recallAt8}</td></tr>
              <tr><th>Conditional accuracy</th><td>{BENCHMARK.conditionalAccuracy.toFixed(4)} given a retrieval hit</td></tr>
              <tr><th>End-to-end accuracy</th><td>{BENCHMARK.endToEnd}</td></tr>
              <tr><th>End-to-end ECE</th><td>{BENCHMARK.ece.toFixed(4)}</td></tr>
              <tr><th>Release</th><td>{BENCHMARK.release}</td></tr>
              <tr><th>Status</th><td>{BENCHMARK.status}. English Choice remains a research release.</td></tr>
            </tbody>
          </table>
          <p className="caveat">{BENCHMARK.caveat}</p>
        </section>
        <section className="section" id="developers">
          <p className="kicker">API</p>
          <h2>POST /api/infer</h2>
          <div className="tabs" role="tablist">
            {(["curl", "python", "ts"] as const).map((item) => (
              <button key={item} role="tab" aria-selected={tab === item} onClick={() => setTab(item)}>{item}</button>
            ))}
          </div>
          <pre><code>{example(tab)}</code></pre>
          <p className="caveat">POST /api/infer expects Authorization: Bearer with a temporary demo token. That token is not the infrastructure credential and is not a project key. Do not put either secret in browser code.</p>
        </section>
        <section className="section explore">
          <a href="#docs" onClick={(event) => { event.preventDefault(); onNavigate("/docs"); }}><p className="kicker">01</p><h3>API documentation</h3><p>Schemas, errors, and the release boundary.</p></a>
          <a href="#research"><p className="kicker">02</p><h3>Research status</h3><p>What is served, and what stays research.</p></a>
          <a href="#console" onClick={(event) => { event.preventDefault(); onNavigate("/console"); }}><p className="kicker">03</p><h3>Playground</h3><p>Build a real request. No fabricated decision.</p></a>
        </section>
        <section className="section faq">
          <h2>Questions</h2>
          {FAQ.map(([question, answer]) => {
            const id = question.slice(0, 16);
            const expanded = open === id;
            return (
              <div key={question}>
                <button aria-expanded={expanded} onClick={() => setOpen(toggleFaq(open, id))}>{question}</button>
                {expanded ? <p>{answer}</p> : null}
              </div>
            );
          })}
        </section>
      </main>
      <footer className="footer" id="company">
        <div><strong>Product</strong><a href="#veya">Veya</a><button onClick={() => onNavigate("/console")}>Console</button></div>
        <div><strong>Developers</strong><button onClick={() => onNavigate("/docs")}>Docs</button><a href="#developers">API</a></div>
        <div><strong>Company</strong><span>Veya Dynamics</span></div>
        <div><strong>Legal</strong><span>Research release. Not a production certification.</span></div>
      </footer>
    </>
  );
}

function example(tab: "curl" | "python" | "ts"): string {
  const body = `{
  "primitive": "choice",
  "state": "Please refund the duplicate March invoice.",
  "instructions": "Which label best fits the text?",
  "criteria": {"billing": "Invoice and refund questions", "access": "Login and permission questions"}
}`;
  if (tab === "python") {
    return `import json, urllib.request\nreq = urllib.request.Request(\n    "${API_ORIGIN}/api/infer",\n    data=json.dumps(${body}).encode(),\n    headers={"content-type": "application/json", "authorization": "Bearer <demo token>"},\n)\nurllib.request.urlopen(req)`;
  }
  if (tab === "ts") {
    return `await fetch("${API_ORIGIN}/api/infer", {\n  method: "POST",\n  headers: {\n    "content-type": "application/json",\n    authorization: "Bearer <demo token>",\n  },\n  body: JSON.stringify(${body}),\n});`;
  }
  return `curl ${API_ORIGIN}/api/infer \\\n  -H "content-type: application/json" \\\n  -H "authorization: Bearer <demo token>" \\\n  -d '${body.replaceAll("\n", "")}'`;
}
