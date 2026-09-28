import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { docsContent } from '../web/static/content.mjs';
import { homeContent } from '../web/static/home-content.mjs';
import { choiceRequest, noulRequest, scoreRequest, systemOneRequest } from '../web/static/examples.mjs';

const config = JSON.parse(readFileSync(new URL('../web/static/site-config.json', import.meta.url), 'utf8'));
const release = JSON.parse(readFileSync(new URL('../artifacts/release/manifest.json', import.meta.url), 'utf8'));
const pages = new Map(config.pages.map(page => [page.href, page]));

test('every public route has unique metadata and an appropriate content owner', () => {
  assert.equal(pages.size, config.pages.length);
  assert.equal(config.pages.filter(page => page.kind === 'home').length, 1);
  assert.equal(config.pages.filter(page => page.kind === 'playground').length, 1);
  for (const page of config.pages) {
    assert.match(page.href, /^\//);
    for (const field of ['title', 'description', 'section', 'navLabel']) assert.ok(page[field]);
    if (page.kind === 'docs') {
      const content = docsContent[page.href];
      assert.ok(content, `Missing documentation for ${page.href}`);
      assert.ok(content.sections.length >= 2);
      assert.equal(new Set(content.sections.map(section => section.title)).size, content.sections.length);
    }
  }
  assert.deepEqual(Object.keys(docsContent).sort(), config.pages.filter(page => page.kind === 'docs').map(page => page.href).sort());
  for (const href of config.primaryNavigation) assert.ok(pages.has(href), `Missing navigation target ${href}`);
  for (const href of config.footerNavigation) assert.ok(pages.has(href), `Missing footer target ${href}`);
});

test('research homepage links and benchmark labels retain their evidence boundary', () => {
  for (const article of homeContent.articles) assert.ok(pages.has(article.href), `Missing research note ${article.href}`);
  assert.equal(homeContent.status.find(item => item.label === 'ENGLISH CHOICE')?.liveKey, 'english_choice');
  assert.equal(homeContent.status.find(item => item.label === 'ENGLISH CHOICE')?.value.toUpperCase(), release.release_status.english_choice);
  assert.equal(homeContent.status.find(item => item.label === 'AUTOMATION')?.value.toUpperCase(), release.automation.status);
  assert.equal(homeContent.status.find(item => item.label === 'ORDINAL SCORE')?.value.toUpperCase(), release.research_artifacts_not_served.ordinal_score);
  const benchmark = docsContent['/benchmarks'];
  const metrics = benchmark.sections.flatMap(section => section.blocks.filter(block => block.type === 'metrics').flatMap(block => block.items));
  assert.deepEqual(metrics.map(item => item.label), ['Recall@8', 'Conditional accuracy', 'End-to-end accuracy', 'End-to-end ECE']);
  assert.match(benchmark.sections.map(section => section.blocks.filter(block => block.type === 'p').map(block => block.text).join(' ')).join(' '), /conditional accuracy on shortlist hits as full-system accuracy/i);
});

test('documentation links resolve to registered routes', () => {
  for (const [route, content] of Object.entries(docsContent)) {
    for (const section of content.sections) {
      for (const block of section.blocks) {
        if (block.type !== 'links') continue;
        for (const link of block.links) assert.ok(pages.has(link.href), `${route} links to missing ${link.href}`);
      }
    }
  }
});

test('public examples use the currently served gateway request shapes', () => {
  assert.equal(choiceRequest.primitive, 'choice');
  assert.equal(noulRequest.primitive, 'noul');
  assert.equal(scoreRequest.primitive, 'score');
  assert.equal(systemOneRequest.primitive, 'systemone');
  assert.ok(Object.keys(choiceRequest.criteria).length >= 2);
  assert.ok(Array.isArray(scoreRequest.criteria));
  assert.ok(Object.keys(systemOneRequest.questions).length >= 2);
});

test('published quickstarts target the hosting site rather than a visitor loopback', async () => {
  const previous = globalThis.location;
  try {
    globalThis.location = { protocol: 'https:', origin: 'https://demo.veya.example' };
    const { quickstart } = await import('../web/static/examples.mjs?deployed-origin');
    assert.match(quickstart.curl, /https:\/\/demo\.veya\.example\/api\/infer/);
    assert.match(quickstart.python, /https:\/\/demo\.veya\.example\/api\/infer/);
    assert.doesNotMatch(quickstart.curl, /127\.0\.0\.1/);
  } finally {
    if (previous === undefined) delete globalThis.location;
    else globalThis.location = previous;
  }
});
