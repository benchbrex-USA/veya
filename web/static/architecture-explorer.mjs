/**
 * A schematic of the Choice routes served by the audited 0.3.5 release.
 * Source: artifacts/release/manifest.json (choice, calibration, known_limitations).
 * This component never makes an inference or depicts inferred candidates.
 */

const routes = {
  packed: {
    title: 'Packed route',
    range: '2–19 candidates',
    summary: 'Every submitted candidate reaches the base decision head.',
    boundary: 'The returned Choice distribution is calibrated for the packed route. A probability is an estimate for this request, not a correctness guarantee.',
    steps: [
      ['01 / VALIDATE', 'Structured input', 'Check state, question, candidate count, and unique candidate IDs.'],
      ['02 / DECIDE', 'Packed decision head', 'Evaluate the submitted candidates in one packed decision pass.'],
      ['03 / CALIBRATE', 'Packed calibration', 'Apply the calibration artifact for this route.'],
      ['04 / RETURN', 'Answer or defer', 'Return the decision, distribution, defer state, and route metadata.'],
    ],
  },
  shortlist: {
    title: 'Shortlist route',
    range: '20–100 candidates',
    summary: 'A learned shortlist selects eight candidates before the base decision head.',
    boundary: 'The distribution is over the returned shortlist, not every submitted candidate. A retrieval miss cannot be repaired by the decision head; cross-domain retrieval is a measured limit.',
    steps: [
      ['01 / VALIDATE', 'Structured input', 'Check state, question, candidate count, and unique candidate IDs.'],
      ['02 / SHORTLIST', 'Learned K=8', 'Select eight candidates from the submitted set.'],
      ['03 / DECIDE', 'Base decision head', 'Evaluate the eight candidates in a packed decision pass.'],
      ['04 / CALIBRATE', 'High-N calibration', 'Apply the separate shortlist-route calibration artifact.'],
      ['05 / RETURN', 'Answer or defer', 'Return the decision, shortlist distribution, defer state, and route metadata.'],
    ],
  },
};

function routePanel(key, route) {
  return `<div class="archx-route-panel" id="archx-panel-${key}" role="region" aria-labelledby="archx-button-${key}" ${key === 'packed' ? '' : 'hidden'} data-archx-panel="${key}">
    <div class="archx-panel-heading"><div><span class="archx-kicker">SERVED CHOICE PATH / ${route.range}</span><h4>${route.title}</h4></div><p>${route.summary}</p></div>
    <ol class="archx-flow" style="--archx-steps:${route.steps.length}">
      ${route.steps.map(([index, title, detail], i) => `<li class="archx-step" style="--archx-index:${i}"><span class="archx-point" aria-hidden="true"></span>${i < route.steps.length - 1 ? '<span class="archx-path" aria-hidden="true"></span>' : ''}<span class="archx-step-index">${index}</span><strong>${title}</strong><p>${detail}</p></li>`).join('')}
    </ol>
    <div class="archx-boundary"><span>READ THE OUTPUT</span><p>${route.boundary}</p></div>
  </div>`;
}

export function renderArchitectureExplorer() {
  return `<div class="archx" data-architecture-explorer data-route="packed">
    <div class="archx-topline"><span>ROUTE INSPECTOR / CHOICE</span><span>SCHEMATIC · NOT A PREDICTION</span></div>
    <div class="archx-controls"><div><h3>Follow the decision path.</h3><p>Candidate count selects a real served route. Inspect what happens before a result is returned.</p></div>
      <div class="archx-toggle" role="group" aria-label="Choice route by candidate count">
        <button type="button" id="archx-button-packed" data-archx-route="packed" aria-pressed="true" aria-controls="archx-panel-packed"><span>ROUTE A</span><strong>2–19</strong><small>packed</small></button>
        <button type="button" id="archx-button-shortlist" data-archx-route="shortlist" aria-pressed="false" aria-controls="archx-panel-shortlist"><span>ROUTE B</span><strong>20–100</strong><small>shortlist</small></button>
      </div>
    </div>
    <div class="archx-field" aria-hidden="true"><span class="archx-field-label">STATE + QUESTION</span><span class="archx-axis archx-axis-h"></span><span class="archx-axis archx-axis-v"></span><span class="archx-field-point archx-field-point-a"></span><span class="archx-field-point archx-field-point-b"></span><span class="archx-field-point archx-field-point-c"></span><span class="archx-field-point archx-field-point-d"></span><span class="archx-field-end">CANDIDATE SET ↓</span></div>
    ${routePanel('packed', routes.packed)}
    ${routePanel('shortlist', routes.shortlist)}
    <div class="archx-observed" data-archx-observed hidden><span>LAST CHOICE RESPONSE / OBSERVED ROUTE</span><dl data-archx-observed-fields></dl></div>
    <p class="archx-sr" role="status" aria-live="polite" data-archx-announcement></p>
    <div class="archx-footer"><span>English research release · automation disabled</span><a href="/architecture/high-cardinality-choice">Examine the served architecture <span aria-hidden="true">↗</span></a></div>
  </div>`;
}

function findExplorer(root) {
  return root.matches?.('[data-architecture-explorer]') ? root : root.querySelector('[data-architecture-explorer]');
}

function activate(explorer, route) {
  if (!Object.hasOwn(routes, route) || explorer.dataset.route === route) return;
  const buttons = [...explorer.querySelectorAll('[data-archx-route]')];
  const panels = [...explorer.querySelectorAll('[data-archx-panel]')];
  const announcement = explorer.querySelector('[data-archx-announcement]');
  explorer.dataset.route = route;
  for (const button of buttons) button.setAttribute('aria-pressed', String(button.dataset.archxRoute === route));
  for (const panel of panels) panel.hidden = panel.dataset.archxPanel !== route;
  announcement.textContent = `${routes[route].range}: ${routes[route].summary}`;
}

/**
 * Call after an actual Choice response. Unknown routes leave the explainer as-is.
 * metadata may include route, calibration, and requestId/request_id. Values are
 * inserted as text nodes only. Without metadata, no observed-result box appears.
 */
export function setArchitectureRoute(root, route, metadata = {}) {
  const explorer = findExplorer(root);
  const key = route === 'packed' ? 'packed' : route === 'shortlist_k8_packed_rerank' || route === 'shortlist' ? 'shortlist' : null;
  if (!explorer || !key) return false;
  activate(explorer, key);

  const observed = explorer.querySelector('[data-archx-observed]');
  const fields = observed.querySelector('[data-archx-observed-fields]');
  const values = [
    ['Route', metadata?.route],
    ['Calibration', metadata?.calibration ?? metadata?.calibration_version],
    ['Request ID', metadata?.requestId ?? metadata?.request_id],
  ].filter(([, value]) => typeof value === 'string' && value.trim());
  fields.replaceChildren();
  for (const [label, value] of values) {
    const pair = explorer.ownerDocument.createElement('div');
    const term = explorer.ownerDocument.createElement('dt');
    const detail = explorer.ownerDocument.createElement('dd');
    term.textContent = label;
    detail.textContent = value.slice(0, 160);
    pair.append(term, detail);
    fields.append(pair);
  }
  observed.hidden = values.length === 0;
  if (values.length) explorer.querySelector('[data-archx-announcement]').textContent = `Last Choice response used the ${routes[key].title.toLowerCase()}.`;
  return true;
}

export function wireArchitectureExplorer(root = document) {
  const explorer = findExplorer(root);
  if (!explorer) return () => {};

  const buttons = [...explorer.querySelectorAll('[data-archx-route]')];

  function onClick(event) {
    activate(explorer, event.currentTarget.dataset.archxRoute);
    explorer.querySelector('[data-archx-observed]').hidden = true;
  }

  for (const button of buttons) button.addEventListener('click', onClick);
  return () => {
    for (const button of buttons) button.removeEventListener('click', onClick);
  };
}
