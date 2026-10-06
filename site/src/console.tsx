import { useState } from "react";
import { buildInferenceRequest, choiceRoute, type ServedPrimitive } from "./contract.ts";
import { Mark } from "./mark.tsx";
import { MAX_FUSION, RELEASE } from "./release.ts";

type View = "overview" | "projects" | "requests" | "playground" | "evaluations" | "usage" | "keys" | "organization";

const VIEWS: View[] = ["overview", "projects", "requests", "playground", "evaluations", "usage", "keys", "organization"];

export function Console({ onNavigate, theme, setTheme }: { onNavigate: (path: string) => void; theme: string; setTheme: (value: string) => void }) {
  const [view, setView] = useState<View>("overview");
  const [menu, setMenu] = useState(false);
  return (
    <div className="shell">
      <aside className="rail">
        <a className="brand" href="/" onClick={(event) => { event.preventDefault(); onNavigate("/"); }}><Mark /> VEYA</a>
        <button className="menu-button" aria-expanded={menu} onClick={() => setMenu((value) => !value)}>Sections</button>
        <nav className={menu ? "open" : undefined} aria-label="Console">
          {VIEWS.map((item) => (
            <a key={item} href={`#${item}`} aria-current={view === item ? "page" : undefined} onClick={(event) => { event.preventDefault(); setView(item); setMenu(false); }}>
              {item[0].toUpperCase() + item.slice(1)}
            </a>
          ))}
        </nav>
        <div className="spacer" />
        <button className="linkish" onClick={() => onNavigate("/docs")}>Docs</button>
        <button className="linkish" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>Theme</button>
      </aside>
      <main className="console-main">
        {view === "playground" ? <Playground /> : null}
        {view === "overview" ? <Overview onPlay={() => setView("playground")} onDocs={() => onNavigate("/docs")} /> : null}
        {view === "projects" ? <Projects /> : null}
        {view === "requests" ? <Empty title="Requests" body="No requests have been recorded. Raw state and candidate text are not stored for this view." /> : null}
        {view === "evaluations" ? <Evaluations /> : null}
        {view === "usage" ? <Empty title="Usage" body="No usage has been recorded. This page is not a billing invoice." /> : null}
        {view === "keys" ? <Keys /> : null}
        {view === "organization" ? <Empty title="Organization" body="No identity provider is connected. The console does not keep a password database. Create a project when D1 and a session are available." /> : null}
      </main>
    </div>
  );
}

function Overview({ onPlay, onDocs }: { onPlay: () => void; onDocs: () => void }) {
  return (
    <>
      <p className="kicker">Overview</p>
      <div className="split">
        <div>
          <h1>Veya Core</h1>
          <p className="lede">Structured decisions, with the release boundary left visible.</p>
          <p className="status">{RELEASE.choice.status}</p>
          <p>{RELEASE.id}. Health and readiness are reported by the API Worker, not by this page.</p>
          <div className="actions start">
            <button className="btn solid" onClick={onPlay}>Open playground</button>
            <button className="btn" onClick={onDocs}>View docs</button>
          </div>
        </div>
        <p className="caveat">API base {`https://api.veyadynamics.com`}. Served primitives: choice, noul, score-categorical. SystemOne is not listed as served.</p>
      </div>
      <div className="panel"><h2>Recent requests</h2><p className="empty">No requests yet.</p></div>
      <div className="panel"><h2>API keys</h2><p className="empty">No API keys yet.</p></div>
    </>
  );
}

function Empty({ title, body }: { title: string; body: string }) {
  return <section><p className="kicker">{title}</p><h1>{title}</h1><p className="empty">{body}</p></section>;
}

function Projects() {
  return (
    <section>
      <p className="kicker">Projects</p>
      <h1>Projects</h1>
      <p className="empty">No projects yet. A project groups keys and usage. Creating one requires the console Worker and D1.</p>
      <ProjectForm />
    </section>
  );
}

function ProjectForm() {
  const [message, setMessage] = useState("Nothing saved.");
  return (
    <form onSubmit={async (event) => {
      event.preventDefault();
      const data = new FormData(event.currentTarget);
      const response = await fetch("/api/projects", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ organization: data.get("organization"), name: data.get("name") }),
      });
      const payload = await response.json().catch(() => ({}));
      setMessage(response.ok ? "Project stored." : payload.detail || "The console API did not store a project.");
    }}>
      <label className="field"><span>Organization</span><input name="organization" required maxLength={80} /></label>
      <label className="field"><span>Project</span><input name="name" required maxLength={80} /></label>
      <button className="btn solid" type="submit">Create project</button>
      <p>{message}</p>
    </form>
  );
}

function Keys() {
  const [revealed, setRevealed] = useState<string | null>(null);
  const [message, setMessage] = useState("Keys are shown once. The stored value is a hash.");
  return (
    <section>
      <p className="kicker">API keys</p>
      <h1>API keys</h1>
      <p className="empty">{message}</p>
      {revealed ? <p className="ok">Copy this now. It will not be shown again. {revealed}</p> : null}
      <form onSubmit={async (event) => {
        event.preventDefault();
        const data = new FormData(event.currentTarget);
        const response = await fetch("/api/keys", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ project_id: data.get("project_id") }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          setRevealed(null);
          setMessage(payload.detail || "No key was created.");
          return;
        }
        setRevealed(payload.token);
        setMessage("The full key was returned once by the Worker.");
      }}>
        <label className="field"><span>Project id</span><input name="project_id" required /></label>
        <button className="btn solid" type="submit">Create key</button>
      </form>
    </section>
  );
}

function Evaluations() {
  return (
    <section>
      <p className="kicker">Evaluations</p>
      <h1>Release evidence</h1>
      <table className="meta">
        <tbody>
          <tr><th>English Choice</th><td>{RELEASE.choice.status}. Sealed CLINC150 val end-to-end accuracy 0.349. Not a release candidate.</td></tr>
          <tr><th>English Noul</th><td>{RELEASE.noul.status}</td></tr>
          <tr><th>Categorical Score</th><td>{RELEASE.score.status}</td></tr>
          <tr><th>Max fusion</th><td>{MAX_FUSION.status}. Promoted: {String(MAX_FUSION.promoted)}. Recommendation: {MAX_FUSION.recommendation}.</td></tr>
        </tbody>
      </table>
    </section>
  );
}

function Playground() {
  const [primitive, setPrimitive] = useState<ServedPrimitive>("choice");
  const [state, setState] = useState("");
  const [instructions, setInstructions] = useState("Which label best fits the text?");
  const [rows, setRows] = useState([{ id: "billing", description: "Invoice and refund questions" }, { id: "access", description: "Login and permission questions" }]);
  const [labels, setLabels] = useState("low, high");
  const [raw, setRaw] = useState(false);
  const [output, setOutput] = useState("No result yet. A run calls the console Worker. It does not invent a decision.");
  const [error, setError] = useState("");
  const criteria = primitive === "score"
    ? labels.split(",").map((label) => label.trim()).filter(Boolean)
    : rows;
  const built = buildInferenceRequest({ primitive, state, instructions, criteria });
  return (
    <section>
      <p className="kicker">Playground</p>
      <h1>Run a structured decision</h1>
      <p className="status">{primitive === "choice" ? RELEASE.choice.status : primitive === "noul" ? RELEASE.noul.status : RELEASE.score.status}</p>
      <div className="split">
        <form onSubmit={async (event) => {
          event.preventDefault();
          if (!built.ok) { setError(built.error); setOutput(""); return; }
          setError("");
          const started = performance.now();
          try {
            const response = await fetch("/api/playground", {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify(built.body),
            });
            const text = await response.text();
            setOutput(`${response.status} in ${Math.round(performance.now() - started)} ms\n${text}`);
          } catch {
            setOutput("");
            setError("The console Worker did not respond. No decision was invented.");
          }
        }}>
          <label className="field"><span>Primitive</span>
            <select value={primitive} onChange={(event) => setPrimitive(event.target.value as ServedPrimitive)}>
              <option value="choice">choice</option>
              <option value="noul">noul</option>
              <option value="score">score</option>
            </select>
          </label>
          <label className="field"><span>State</span><textarea value={state} onChange={(event) => setState(event.target.value)} required /></label>
          <label className="field"><span>Instructions</span><input value={instructions} onChange={(event) => setInstructions(event.target.value)} required /></label>
          {primitive === "choice" ? (
            <div className="rows">
              {rows.map((row, index) => (
                <div className="row" key={index}>
                  <input aria-label={`Candidate ID ${index + 1}`} value={row.id} onChange={(event) => setRows(rows.map((item, itemIndex) => itemIndex === index ? { ...item, id: event.target.value } : item))} />
                  <input aria-label={`Description ${index + 1}`} value={row.description} onChange={(event) => setRows(rows.map((item, itemIndex) => itemIndex === index ? { ...item, description: event.target.value } : item))} />
                  <button className="btn quiet" type="button" onClick={() => setRows(rows.filter((_, itemIndex) => itemIndex !== index))}>Remove</button>
                </div>
              ))}
              <button className="btn quiet" type="button" onClick={() => setRows([...rows, { id: "", description: "" }])}>Add candidate</button>
            </div>
          ) : null}
          {primitive === "score" ? <label className="field"><span>Labels</span><input value={labels} onChange={(event) => setLabels(event.target.value)} /></label> : null}
          {primitive === "choice" && built.ok ? <p>Route preview: {choiceRoute(Object.keys(built.body.criteria as object).length)}.</p> : null}
          <button className="btn solid" type="submit">Run</button>
          {error ? <p className="error">{error}</p> : null}
        </form>
        <div>
          <button className="btn quiet" type="button" onClick={() => setRaw((value) => !value)}>{raw ? "Human view" : "Raw JSON"}</button>
          <pre><code>{raw ? JSON.stringify(built.ok ? built.body : { error: built.error }, null, 2) : output}</code></pre>
        </div>
      </div>
    </section>
  );
}
