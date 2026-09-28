import { renderHome } from './home.mjs?v=lab-2';
import { experience } from './playground.mjs?v=lab-2';
import { renderDocs } from './docs.mjs?v=docs-2';
import { esc } from './ui.mjs?v=docs-2';

const root = document.querySelector('#main');
const path = location.pathname;
if (new URLSearchParams(location.search).get('motion') === 'reduce') document.documentElement.classList.add('force-reduced');

async function start() {
  let config;
  try {
    const response = await fetch('/static/site-config.json', {cache:'no-store'});
    if (!response.ok) throw new Error('Site configuration unavailable');
    config = await response.json();
  } catch {
    root.innerHTML = '<section class="page-hero"><h1>Site temporarily unavailable.</h1><p>Please reload this page.</p></section>';
    return;
  }
  const page = config.pages.find(item => item.href === path);
  if (!page) { root.innerHTML = '<section class="page-hero"><h1>Page unavailable.</h1></section>'; return; }
  document.title = page.title;
  const entries = config.primaryNavigation.map(href => config.pages.find(item => item.href === href)).filter(Boolean);
  document.querySelector('#nav').innerHTML = entries.map(item => `<a href="${esc(item.href)}" ${item.href === path || (path.startsWith(item.href+'/') && item.href !== '/') ? 'aria-current="page" class="active"' : ''}>${esc(item.topLabel || item.navLabel)}</a>`).join('');
  document.querySelector('#footer-links').innerHTML = config.footerNavigation.map(href => config.pages.find(item => item.href === href)).filter(Boolean).map(item => `<a href="${esc(item.href)}">${esc(item.navLabel)}</a>`).join('');
  const menu = document.querySelector('#menu-toggle');
  menu.addEventListener('click', () => {
    const expanded = document.querySelector('#nav').classList.toggle('open');
    menu.setAttribute('aria-expanded',String(expanded));
  });
  menu.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    document.querySelector('#nav').classList.remove('open');
    menu.setAttribute('aria-expanded','false');
  });
  document.querySelector('#nav').addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    document.querySelector('#nav').classList.remove('open');
    menu.setAttribute('aria-expanded','false');
    menu.focus();
  });
  let scrollScheduled = false;
  window.addEventListener('scroll', () => {
    if (scrollScheduled) return;
    scrollScheduled = true;
    requestAnimationFrame(() => {
      document.querySelector('.site-header').classList.toggle('is-scrolled', window.scrollY > 24);
      scrollScheduled = false;
    });
  }, { passive: true });
  if (page.kind === 'home') { renderHome(); experience(document.querySelector('#home-playground')); }
  else if (page.kind === 'playground') experience();
  else renderDocs(path,config);
}

start();
