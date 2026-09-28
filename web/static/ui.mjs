/** Shared, intentionally small rendering primitives. Content is data, not markup. */
export const $ = (selector, root = document) => root.querySelector(selector);
export const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
export const slug = value => String(value).toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');

export function statusBadge(status) {
  const key = slug(status);
  return `<span class="status-badge status-${esc(key)}">${esc(status)}</span>`;
}
export function callout(text, tone = 'info') {
  return `<aside class="docs-callout docs-callout-${esc(tone)}">${esc(text)}</aside>`;
}
export function codeBlock(code, language = 'text', title = '') {
  return `<div class="docs-code"><div class="docs-code-bar"><span>${esc(title || language)}</span><button type="button" class="copy-code" aria-label="Copy ${esc(title || language)} example">Copy</button></div><pre><code>${esc(code)}</code></pre></div>`;
}
export function codeTabs(options, id) {
  return `<div class="docs-code-tabs" id="${esc(id)}"><div class="docs-code-tablist" role="tablist" aria-label="Code language">${options.map((option, index) => `<button type="button" role="tab" id="${esc(id)}-tab-${index}" aria-controls="${esc(id)}-panel" aria-selected="${index === 0}" tabindex="${index === 0 ? 0 : -1}" data-code-tab="${index}">${esc(option.label)}</button>`).join('')}</div><div class="docs-code-panel" role="tabpanel" id="${esc(id)}-panel" aria-labelledby="${esc(id)}-tab-0">${codeBlock(options[0].code,options[0].language,options[0].label)}</div><script type="application/json" class="docs-code-data">${JSON.stringify(options).replace(/</g,'\\u003c')}</script></div>`;
}
export function metric(label, value, note = '') {
  return `<div class="docs-metric"><strong>${esc(value)}</strong><span>${esc(label)}</span>${note ? `<small>${esc(note)}</small>` : ''}</div>`;
}
export function renderSection(section) {
  const id = slug(section.title);
  const blocks = section.blocks.map((block, index) => {
    if (block.type === 'p') return `<p>${esc(block.text)}</p>`;
    if (block.type === 'list') return `<ul>${block.items.map(item => `<li>${esc(item)}</li>`).join('')}</ul>`;
    if (block.type === 'code') return codeBlock(block.code, block.language, block.title);
    if (block.type === 'codeTabs') return codeTabs(block.options,`${id}-code-${index}`);
    if (block.type === 'callout') return callout(block.text, block.tone);
    if (block.type === 'flow') return `<ol class="docs-flow">${block.items.map(item => `<li>${esc(item)}</li>`).join('')}</ol>`;
    if (block.type === 'table') return `<div class="docs-table-wrap"><table><thead><tr>${block.headers.map(h => `<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${block.rows.map(row => `<tr>${row.map(cell => `<td>${esc(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    if (block.type === 'links') return `<div class="docs-link-row">${block.links.map(link => `<a href="${esc(link.href)}">${esc(link.label)} <span aria-hidden="true">↗</span></a>`).join('')}</div>`;
    if (block.type === 'metrics') return `<div class="docs-metric-grid">${block.items.map(item => metric(item.label,item.value,item.note)).join('')}</div>`;
    return '';
  }).join('');
  return `<section class="docs-section" id="${id}"><h2>${esc(section.title)}</h2>${blocks}</section>`;
}
export function wireCopyButtons(root = document) {
  root.querySelectorAll('.copy-code').forEach(button => {
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(button.closest('.docs-code').querySelector('code').textContent);
        button.textContent = 'Copied';
        window.setTimeout(() => { button.textContent = 'Copy'; }, 1500);
      } catch { button.textContent = 'Copy failed'; }
    });
  });
}
export function wireCodeTabs(root = document) {
  root.querySelectorAll('.docs-code-tabs').forEach(group => {
    const options = JSON.parse(group.querySelector('.docs-code-data').textContent);
    const tabs = [...group.querySelectorAll('[role="tab"]')];
    const panel = group.querySelector('[role="tabpanel"]');
    const select = index => {
      tabs.forEach((tab, current) => {
        tab.setAttribute('aria-selected', String(current === index));
        tab.tabIndex = current === index ? 0 : -1;
      });
      panel.setAttribute('aria-labelledby', tabs[index].id);
      panel.innerHTML = codeBlock(options[index].code,options[index].language,options[index].label);
      wireCopyButtons(panel);
      tabs[index].focus();
    };
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => select(index));
      tab.addEventListener('keydown', event => {
        let next = index;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = tabs.length - 1;
        else return;
        event.preventDefault();
        select(next);
      });
    });
  });
}
