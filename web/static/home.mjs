import { homeContent } from './home-content.mjs?v=lab-1';
import { quickstart } from './examples.mjs?v=docs-2';
import { VeyaClient } from './model-client.mjs';
import { codeTabs, esc, statusBadge, wireCodeTabs, wireCopyButtons } from './ui.mjs?v=docs-2';
import { renderArchitectureExplorer, setArchitectureRoute, wireArchitectureExplorer } from './architecture-explorer.mjs';

const lines = value => value.split('\n').map(esc).join('<br>');
const label = (number, title) => `<span class="home-label"><span>${esc(number)}</span> ${esc(title)}</span>`;

function hero() {
  return `<section class="hero home-hero" aria-labelledby="home-title">
    <div class="hero-copy">
      <span class="eyebrow">${esc(homeContent.eyebrow)}</span>
      <h1 id="home-title">${lines(homeContent.title)}</h1>
      <p>${esc(homeContent.summary)}</p>
      <div class="hero-actions"><a class="button" href="#live" data-run-example>Run a live example <span aria-hidden="true">↗</span></a><a class="button outline" href="#research">Read the research <span aria-hidden="true">↓</span></a></div>
      <div class="hero-release">${statusBadge('Research')}<span>English checkpoint · automation disabled</span></div>
    </div>
    <div class="hero-art" aria-hidden="true"><div class="hero-decision-field">
      <div class="hero-field-top"><span>VEYA / DECISION FIELD</span><span>FIG. 01 / 04 → 01</span></div>
      <div class="hero-field-plane">
        <svg class="hero-field-paths" viewBox="0 0 500 500" preserveAspectRatio="none" focusable="false"><path d="M65 85 H180 L222 207"/><path d="M435 85 H320 L278 207"/><path d="M65 415 H180 L222 293"/><path d="M435 415 H320 L278 293"/></svg>
        <span class="hero-field-axis hero-field-axis-x"></span><span class="hero-field-axis hero-field-axis-y"></span>
        <span class="hero-field-point hero-field-point-a"><i></i><b>01</b></span><span class="hero-field-point hero-field-point-b"><i></i><b>02</b></span>
        <span class="hero-field-point hero-field-point-c"><i></i><b>03</b></span><span class="hero-field-point hero-field-point-d"><i></i><b>04</b></span>
        <div class="hero-field-resolution"><span class="hero-field-resolution-corner"></span><img src="/static/veya-symbol.png" alt=""></div>
        <span class="hero-field-coordinate hero-field-coordinate-x">X / ALTERNATIVES</span><span class="hero-field-coordinate hero-field-coordinate-y">Y / EVIDENCE</span>
      </div>
      <div class="hero-field-bottom"><span>POSSIBILITIES</span><span>STRUCTURE</span><span>DECISION / DEFER</span></div>
    </div></div>
    <div class="home-hero-index" aria-hidden="true">001 / 009</div>
  </section>`;
}

function thesis() {
  const comparison = (name, items, className) => `<div class="thesis-track ${className}"><span>${esc(name)}</span><ol>${items.map((item, index) => `<li><small>${String(index + 1).padStart(2, '0')}</small>${esc(item)}</li>`).join('')}</ol></div>`;
  return `<section class="home-thesis home-pad" id="thesis" aria-labelledby="thesis-title">
    ${label('01', 'A DIFFERENT OUTPUT')}
    <div class="home-thesis-head"><h2 id="thesis-title">${esc(homeContent.thesis.title)}</h2><p>${esc(homeContent.thesis.summary)}</p></div>
    <div class="thesis-comparison">${comparison('GENERIC GENERATIVE INTERFACE', homeContent.thesis.generative, 'thesis-generative')}${comparison('VEYA / STRUCTURED INFERENCE', homeContent.thesis.veya, 'thesis-veya')}</div>
  </section>`;
}

function principles() {
  return `<section class="home-principles home-pad" aria-labelledby="principles-title">${label('02', 'THREE PRINCIPLES')}
    <h2 id="principles-title" class="visually-hidden">Veya principles</h2>
    ${homeContent.principles.map(item => `<article class="home-principle"><span class="home-principle-number">${esc(item.number)}</span><h3>${lines(item.title)}</h3><p>${esc(item.detail)}</p><span class="home-principle-axis" aria-hidden="true"></span></article>`).join('')}
  </section>`;
}

function live() {
  return `<section class="home-live home-pad" id="live" aria-labelledby="live-title">${label('03', 'THE MODEL / LIVE')}
    <div class="home-section-head"><h2 id="live-title">Give the model<br>a decision.</h2><div><p>Run the same validated Veya gateway used by the dedicated playground. This is a real response from the served research checkpoint.</p><a href="/experience" class="home-text-link">Open the full experience ↗</a></div></div>
    <div class="home-playground" id="home-playground"></div>
  </section>`;
}

function research() {
  const metric = (key, labelText, note) => `<div class="home-research-metric"><strong data-research="${esc(key)}">—</strong><span>${esc(labelText)}</span><p>${esc(note)}</p></div>`;
  return `<section class="home-research home-pad" id="research" aria-labelledby="research-title">${label('04', 'SEALED EVALUATION')}
    <div class="home-section-head"><h2 id="research-title">The bottleneck<br>is visible.</h2><p>When the correct label reaches the shortlist, the decision head usually finds it. On an unseen domain, retrieval often misses it. These numbers must be read together.</p></div>
    <div class="home-research-grid" aria-live="polite">
      ${metric('recall_at_8', 'Blind Recall@8', 'The target reached the eight-candidate shortlist.')}
      ${metric('conditional_accuracy', 'Conditional accuracy', 'Correct once the target reached that shortlist. Not end-to-end accuracy.')}
      ${metric('accuracy', 'Blind end-to-end accuracy', 'Counts retrieval misses and scoring errors across all examples.')}
      ${metric('ece', 'End-to-end ECE', 'Calibration error on the same blind run; lower is better.')}
    </div>
    <figure class="home-evidence-fate" aria-labelledby="home-fate-title">
      <figcaption><span>THE BLIND ROUTE / ACCOUNTED FOR</span><h3 id="home-fate-title"><span data-fate="n">—</span> cases.<br>Three outcomes.</h3><p>Follow the target through retrieval and scoring. Each segment is proportional to the sealed evaluation sample.</p></figcaption>
      <div class="home-fate-visual"><p data-fate-fallback>Checking the sealed outcome breakdown…</p><div data-fate-content hidden><div class="home-fate-bar" role="img" aria-label="Evaluation outcome distribution"><span data-fate-bar="retrieval_misses" class="home-fate-segment is-retrieval"></span><span data-fate-bar="scoring_misses" class="home-fate-segment is-scoring"></span><span data-fate-bar="correct_on_hit" class="home-fate-segment is-correct"></span></div><div class="home-fate-legend"><div><i class="is-retrieval" aria-hidden="true"></i><strong data-fate="retrieval_misses">—</strong><span>Lost at retrieval</span><p>The true label never entered the top eight.</p></div><div><i class="is-scoring" aria-hidden="true"></i><strong data-fate="scoring_misses">—</strong><span>Lost after retrieval</span><p>The label reached the shortlist; scoring missed it.</p></div><div><i class="is-correct" aria-hidden="true"></i><strong data-fate="correct_on_hit">—</strong><span>Correct end to end</span><p>The target survived retrieval and was selected.</p></div></div></div></div>
    </figure>
    <div class="home-research-bottom"><p>Current research focus: <strong>domain-general retrieval.</strong></p><details><summary>Methodology and evidence boundary</summary><p data-research-meta>Checking the sealed aggregate artifact…</p><p>The result describes one English high-cardinality Choice setting. It does not establish general reliability for Noul, Score, multilingual use, or automation.</p></details><a href="/benchmarks" class="home-text-link">Full benchmark record ↗</a></div>
  </section>`;
}

function architecture() {
  return `<section class="home-architecture home-pad" id="architecture-story" aria-labelledby="architecture-title">${label('05', 'THE SERVED PATH')}
    <div class="home-section-head"><h2 id="architecture-title">A route for<br>the question.</h2><p>Choice uses the packed head for small sets. At 20 or more candidates, the served model retrieves eight before the base decision head and route calibration. The experimental hypothesis scorer is not served.</p></div>
    ${renderArchitectureExplorer()}
    <div class="home-architecture-foot"><span>SERVED / PACKED + LEARNED K8</span><span>RESEARCH / PHASE 9 HYPOTHESIS SCORER</span><span>RETIRED / PHASE 6 RERANKER HEADS</span><a href="/architecture">Trace the architecture ↗</a></div>
  </section>`;
}

function developer() {
  return `<section class="home-developer home-pad" id="developers-preview" aria-labelledby="developer-title">${label('06', 'FOR BUILDERS')}
    <div class="home-section-head"><h2 id="developer-title">A decision<br>your software<br>can inspect.</h2><div><p>A small JSON request goes through the same-origin website gateway. The response keeps its decision, probabilities, defer reason, and technical metadata.</p><a href="/developers/api" class="home-text-link">Read the API reference ↗</a></div></div>
    <div class="home-developer-code">${codeTabs([{label:'curl',language:'bash',code:quickstart.curl},{label:'Python',language:'python',code:quickstart.python},{label:'TypeScript',language:'typescript',code:quickstart.typescript}],'home-developer-example')}</div>
  </section>`;
}

function status() {
  return `<section class="home-status home-pad" id="model-status-preview" aria-labelledby="status-title">${label('07', 'RELEASE STATE')}
    <div class="home-section-head"><h2 id="status-title">What is served.<br>What is research.</h2><p>Release labels are part of the product, not fine print. The live service version is checked separately from the audited manifest.</p></div>
    <div class="home-status-list">${homeContent.status.map(item => `<div><span>${esc(item.label)}</span><strong${item.liveKey ? ` data-status="${esc(item.liveKey)}"` : ''}>${esc(item.value)}</strong></div>`).join('')}</div>
    <p class="home-status-live" data-status="live" role="status">Checking active release…</p><a href="/model-status" class="home-text-link">Full model status ↗</a>
  </section>`;
}

function articles() {
  return `<section class="home-articles home-pad" id="research-notes" aria-labelledby="articles-title">${label('08', 'RESEARCH NOTES')}
    <div class="home-section-head"><h2 id="articles-title">Read beneath<br>the result.</h2><p>Architecture, evidence, and limits are documented alongside the live instrument.</p></div>
    <div class="home-article-list">${homeContent.articles.map(item => `<a href="${esc(item.href)}" class="home-article"><span>${esc(item.number)} / ${esc(item.category)}</span><h3>${esc(item.title)}</h3><p>${esc(item.description)}</p><b aria-hidden="true">↗</b></a>`).join('')}</div>
  </section>`;
}

function faq() {
  return `<section class="home-faq home-pad" id="faq" aria-labelledby="faq-title">${label('09', 'QUESTIONS / ANSWERS')}
    <div class="home-section-head"><h2 id="faq-title">The important<br>questions.</h2><p>Specific answers for the current audited release.</p></div>
    <div class="home-faq-list">${homeContent.faq.map(item => `<details><summary>${esc(item.question)}<span aria-hidden="true">+</span></summary><p>${esc(item.answer)}</p></details>`).join('')}</div>
    <div class="home-closing"><img src="/static/veya-symbol.png" alt="" loading="lazy"><p>Decisions, made legible.</p><a class="button" href="/experience">Experience Veya ↗</a></div>
  </section>`;
}

async function loadResearch(root) {
  const metadata = root.querySelector('[data-research-meta]');
  try {
    const response = await fetch('/api/benchmark', { cache: 'no-store' });
    if (!response.ok) throw new Error('Research artifact unavailable');
    const audit = await response.json();
    const rateKeys = ['recall_at_8','conditional_accuracy','accuracy','ece'];
    const countKeys = ['retrieval_misses','scoring_misses','correct_on_hit'];
    if (rateKeys.some(key => typeof audit[key] !== 'number' || !Number.isFinite(audit[key]) || audit[key] < 0 || audit[key] > 1)) throw new Error('Invalid research rates');
    if (!Number.isSafeInteger(audit.n) || audit.n <= 0 || countKeys.some(key => !Number.isSafeInteger(audit[key]) || audit[key] < 0 || audit[key] > audit.n) || countKeys.reduce((total, key) => total + audit[key], 0) !== audit.n) throw new Error('Invalid research counts');
    if (typeof audit.dataset !== 'string' || typeof audit.release !== 'string' || typeof audit.split_status !== 'string' || !Number.isSafeInteger(audit.candidate_count)) throw new Error('Invalid research metadata');
    for (const key of rateKeys) {
      const value = key === 'ece' ? audit[key].toFixed(3) : `${(audit[key] * 100).toFixed(key === 'conditional_accuracy' ? 2 : 1)}%`;
      root.querySelector(`[data-research="${key}"]`).textContent = value;
    }
    root.querySelector('[data-fate="n"]').textContent = audit.n.toLocaleString();
    for (const key of countKeys) {
      root.querySelector(`[data-fate="${key}"]`).textContent = audit[key].toLocaleString();
      root.querySelector(`[data-fate-bar="${key}"]`).style.flexBasis = `${(audit[key] / audit.n * 100).toFixed(5)}%`;
    }
    root.querySelector('.home-fate-bar').setAttribute('aria-label', `${audit.n.toLocaleString()} blind cases: ${audit.retrieval_misses.toLocaleString()} missed at retrieval, ${audit.scoring_misses.toLocaleString()} missed after retrieval, ${audit.correct_on_hit.toLocaleString()} correct end to end.`);
    root.querySelector('[data-fate-fallback]').hidden = true;
    root.querySelector('[data-fate-content]').hidden = false;
    metadata.textContent = `${audit.dataset} · ${Number(audit.n).toLocaleString()} examples · ${audit.candidate_count} candidate labels · sealed ${audit.split_status.toLowerCase().replaceAll('_',' ')} · release ${audit.release}. The top-eight retriever was trained on Banking77; this CLINC150 validation is a cross-domain evaluation.`;
  } catch {
    root.querySelector('[data-fate-fallback]').textContent = 'Outcome breakdown unavailable while the sealed aggregate artifact cannot be verified.';
    metadata.textContent = 'The sealed aggregate artifact is unavailable. See the benchmark record for methodology; no figures are shown here without the source.';
  }
}

async function loadStatus(root) {
  try {
    const version = await new VeyaClient().version();
    for (const item of homeContent.status.filter(item => item.liveKey)) {
      root.querySelector(`[data-status="${item.liveKey}"]`).textContent = version.release_status?.[item.liveKey] || 'Status not reported';
    }
    root.querySelector('[data-status="live"]').textContent = `Active service: ${version.release || version.model_version} · languages: ${version.supported_languages?.join(', ') || 'not reported'}. Audited manifest: automation disabled.`;
  } catch {
    root.querySelector('[data-status="live"]').textContent = 'Model service unavailable. The statuses above reflect the last audited release manifest.';
  }
}

export function renderHome() {
  const root = document.querySelector('#main');
  root.innerHTML = [hero(), thesis(), principles(), live(), research(), architecture(), developer(), status(), articles(), faq()].join('');
  wireCopyButtons(root);
  wireCodeTabs(root);
  wireArchitectureExplorer(root);
  root.querySelector('[data-run-example]').addEventListener('click', () => {
    root.querySelector('#load-example')?.click();
    root.querySelector('#submit')?.click();
  });
  root.querySelector('#home-playground').addEventListener('veya:decision', event => {
    const result = event.detail;
    if (result?.primitive !== 'choice' || typeof result.route !== 'string') return;
    setArchitectureRoute(root, result.route, { route: result.route, calibration: result.calibration, requestId: result.requestId });
  });
  loadResearch(root);
  loadStatus(root);
}
