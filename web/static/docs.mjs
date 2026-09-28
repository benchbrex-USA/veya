import { docsContent } from './content.mjs?v=docs-2';
import { VeyaClient } from './model-client.mjs';
import { esc, renderSection, slug, statusBadge, wireCodeTabs, wireCopyButtons } from './ui.mjs?v=docs-2';

const client = new VeyaClient();
const groups = ['Overview','Start','Core concepts','System','Evidence','Developers'];

function sidebar(config, path) {
  const grouped = groups.map(group => {
    const pages = config.pages.filter(page => page.section === group).sort((a,b) => a.order - b.order);
    if (!pages.length) return '';
    return `<div class="docs-nav-group"><h2>${esc(group)}</h2>${pages.map(page => `<a href="${esc(page.href)}" data-doc-link="${esc((page.navLabel+' '+page.description).toLowerCase())}" ${page.href === path ? 'aria-current="page"' : ''}>${esc(page.navLabel)}</a>`).join('')}</div>`;
  }).join('');
  return `<aside class="docs-sidebar" id="docs-sidebar" aria-label="Documentation navigation"><label for="docs-search">Search documentation</label><input id="docs-search" type="search" placeholder="Search pages…" autocomplete="off" aria-controls="docs-nav-results"><div id="docs-nav-results" aria-live="polite" class="docs-search-count"></div><nav class="docs-nav" aria-label="Documentation">${grouped}</nav></aside>`;
}

export function renderDocs(path, config) {
  const page = docsContent[path];
  if (!page) throw new Error(`Documentation page is missing: ${path}`);
  const root = document.querySelector('#main');
  const toc = page.sections.map(section => `<a href="#${slug(section.title)}">${esc(section.title)}</a>`).join('');
  const crumb = path === '/principles' ? 'Overview' : config.pages.find(item => item.href === path)?.section || 'Documentation';
  root.innerHTML = `<div class="documentation"><button class="docs-drawer-toggle" type="button" aria-controls="docs-sidebar" aria-expanded="false">Browse documentation <span aria-hidden="true">☰</span></button><div class="documentation-layout">${sidebar(config,path)}<article class="docs-article"><div class="docs-breadcrumb"><a href="/">Veya</a><span aria-hidden="true">/</span>${esc(crumb)}</div><header class="docs-article-header"><span class="eyebrow">${esc(page.category)}</span><h1>${esc(page.heading)}</h1><p>${esc(page.summary)}</p>${statusBadge(page.status)}</header>${page.sections.map(renderSection).join('')}<div class="docs-end"><a href="/experience">Try the live model <span aria-hidden="true">↗</span></a><a href="/limitations">Read limitations <span aria-hidden="true">↗</span></a></div></article><aside class="docs-toc" aria-label="On this page"><strong>On this page</strong><nav>${toc}</nav></aside></div></div>`;
  const toggle = root.querySelector('.docs-drawer-toggle');
  toggle.addEventListener('click', () => {
    const open = root.querySelector('.documentation').classList.toggle('docs-drawer-open');
    toggle.setAttribute('aria-expanded',String(open));
    if (open) root.querySelector('#docs-search').focus();
  });
  const search = root.querySelector('#docs-search');
  search.addEventListener('input', () => {
    const term = search.value.trim().toLowerCase();
    const links = [...root.querySelectorAll('[data-doc-link]')];
    let visible = 0;
    links.forEach(link => { const show = !term || link.dataset.docLink.includes(term); link.hidden = !show; if (show) visible++; });
    root.querySelectorAll('.docs-nav-group').forEach(group => { group.hidden = !group.querySelector('a:not([hidden])'); });
    root.querySelector('#docs-nav-results').textContent = term ? `${visible} matching pages` : '';
  });
  document.addEventListener('keydown', event => {
    const editable = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
    if ((event.key === '/' && !editable) || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k')) {
      event.preventDefault();
      root.querySelector('.documentation').classList.add('docs-drawer-open');
      toggle.setAttribute('aria-expanded','true');
      search.focus();
    }
    if (event.key === 'Escape' && document.activeElement === search) {
      search.value = '';
      search.dispatchEvent(new Event('input'));
      root.querySelector('.documentation').classList.remove('docs-drawer-open');
      toggle.setAttribute('aria-expanded','false');
      toggle.focus();
    }
  });
  wireCopyButtons(root);
  wireCodeTabs(root);
  if (path === '/benchmarks') loadBenchmarks(root);
  if (path === '/model-status') loadVersion(root);
}

async function loadBenchmarks(root) {
  const values = root.querySelectorAll('.docs-metric strong');
  try {
    const response = await fetch('/api/benchmark');
    if (!response.ok) throw new Error('Benchmark artifact unavailable');
    const audit = await response.json();
    const metrics = [audit.recall_at_8,audit.conditional_accuracy,audit.accuracy,audit.ece];
    if (metrics.length !== values.length || metrics.some(value => typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)) throw new Error('Invalid benchmark artifact');
    metrics.forEach((value,index) => { values[index].textContent = index === 3 ? value.toFixed(3) : `${(value*100).toFixed(index === 1 ? 2 : 1)}%`; });
  } catch {
    values.forEach(value => { value.textContent = 'Unavailable'; });
  }
}

async function loadVersion(root) {
  const section = root.querySelector('#live-release-metadata');
  const detail = document.createElement('div');
  detail.className = 'docs-live-version';
  detail.setAttribute('role','status');
  detail.textContent = 'Checking active model…';
  section.append(detail);
  try {
    const version = await client.version();
    const languages = Array.isArray(version.supported_languages) ? version.supported_languages.join(', ') : 'Not reported';
    detail.textContent = `Active release ${version.release || 'not reported'} · model version ${version.model_version} · languages ${languages}. ${version.release_status?.english_choice || 'Research status not reported'}.`;
  } catch {
    detail.textContent = 'Model unavailable. The matrix above reflects the last audited manifest.';
  }
}
