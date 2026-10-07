import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHandler, createLimiter, weight, MAX_REQUEST_BYTES} from '../worker/veya_site_edge.mjs';

const root = new URL('..', import.meta.url);
const indexHtml = readFileSync(new URL('web/static/index.html', root), 'utf8');
const siteConfig = JSON.parse(readFileSync(new URL('web/static/site-config.json', root), 'utf8'));
const benchmark = JSON.parse(readFileSync(new URL('artifacts/release/blind_clinc150_val_result.json', root), 'utf8'));
const KEY = 'test-site-key-not-real';
const HOST = 'veya.example.workers.dev';
const choice = {primitive: 'choice', state: 'duplicate charge', instructions: 'Which team?', criteria: {billing: 'refunds', technical: 'bugs'}};
const answer = {decision: 'answer', deferred: false, primitive: 'choice', answer: 'billing', probabilities: {billing: 0.9, technical: 0.1}, request_id: 'r1', gateway_latency_ms: 12};

function jsonResponse(status, body, headers = {}) {
  return new Response(JSON.stringify(body), {status, headers: {'content-type': 'application/json', ...headers}});
}

function setup({upstream, env = {}, logs = []} = {}) {
  const calls = [];
  const assets = [];
  const api = {
    async fetch(request) {
      calls.push({url: request.url, method: request.method, auth: request.headers.get('authorization'), body: request.method === 'POST' ? await request.text() : null});
      const path = new URL(request.url).pathname;
      if (upstream) return upstream(path, request);
      if (path === '/healthz') return jsonResponse(200, {status: 'ok', edge: 'ok', release: '0.3.5-audit-baseline.1'});
      if (path === '/v1/version') return jsonResponse(200, {release: '0.3.5-audit-baseline.1', checkpoint_sha256: 'abc', release_claims: {english_choice: 'RESEARCH RELEASE'}});
      return jsonResponse(200, answer, {'x-veya-replica': '0'});
    },
  };
  const ASSETS = {
    async fetch(request) {
      const path = new URL(request.url).pathname;
      assets.push(path);
      if (path === '/site.css') return new Response('body{}', {status: 200, headers: {etag: '"x"'}});
      return new Response('nope', {status: 404});
    },
  };
  const handle = createHandler({indexHtml, siteConfig, benchmark}, {limiter: createLimiter(), log: (entry) => logs.push(entry)});
  const fullEnv = {VEYA_API: api, ASSETS, VEYA_SITE_KEY: KEY, VEYA_RELEASE_ID: '0.3.5-audit-baseline.1', VEYA_MODEL_VERSION: '0.3.5', ...env};
  const run = (path, init = {}) => handle(new Request(`https://${HOST}${path}`, init), fullEnv, {waitUntil() {}});
  return {run, calls, assets, logs};
}

const post = (body, headers = {}) => ({method: 'POST', headers: {'content-type': 'application/json', host: HOST, 'cf-connecting-ip': '198.51.100.7', ...headers}, body: typeof body === 'string' ? body : JSON.stringify(body)});

test('serves the website pages with gateway substitutions and headers', async () => {
  const {run} = setup();
  const home = await run('/');
  assert.equal(home.status, 200);
  assert.match(home.headers.get('content-type'), /text\/html/);
  const html = await home.text();
  assert.ok(!html.includes('__VEYA_TITLE__'));
  assert.ok(html.includes('<title>Veya — Structured decisions</title>'));
  assert.equal(home.headers.get('x-frame-options'), 'DENY');
  assert.match(home.headers.get('content-security-policy'), /default-src 'self'/);
  assert.equal(home.headers.get('strict-transport-security'), 'max-age=31536000; includeSubDomains');
  for (const page of siteConfig.pages) assert.equal((await run(page.href)).status, 200, page.href);
  const missing = await run('/no-such-page');
  assert.equal(missing.status, 404);
  assert.match(await missing.text(), /Page not found/);
  const head = await run('/', {method: 'HEAD'});
  assert.equal(head.status, 200);
  assert.equal(await head.text(), '');
});

test('static files go through the suffix allow-list', async () => {
  const {run, assets} = setup();
  const css = await run('/static/site.css');
  assert.equal(css.status, 200);
  assert.equal(css.headers.get('content-type'), 'text/css; charset=utf-8');
  assert.equal(css.headers.get('cache-control'), 'no-store');
  assert.deepEqual(assets, ['/site.css']);
  for (const bad of ['/static/index.html', '/static/missing.js', '/static/', '/static/a%2F..%2F..%2Fx.js']) {
    const response = await run(bad);
    assert.equal(response.status, 404, bad);
    assert.equal((await response.json()).code, 'not_found');
  }
  // URL normalisation turns /static/../x into /x, which is an unknown page (HTML 404).
  assert.equal((await run('/static/../wrangler.jsonc')).status, 404);
});

test('api benchmark, robots and unknown api paths match the gateway', async () => {
  const {run} = setup();
  const bench = await (await run('/api/benchmark')).json();
  assert.equal(bench.n, benchmark.n);
  assert.equal(bench.accuracy, benchmark.choice.end_to_end_accuracy.rate);
  assert.equal(await (await run('/robots.txt')).text(), 'User-agent: *\nDisallow: /\n');
  assert.equal((await run('/sitemap.xml')).status, 404);
  const unknown = await run('/api/nope');
  assert.equal(unknown.status, 404);
  assert.equal((await unknown.json()).error_code, 'NOT_FOUND');
  assert.equal((await run('/api/infer', {method: 'PUT'})).status, 405);
  assert.equal((await run('/elsewhere', post(choice))).status, 404);
});

test('inference uses the service binding with the server-side key and passes the body through', async () => {
  const logs = [];
  const {run, calls} = setup({logs});
  const response = await run('/api/infer', post(choice));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), answer);
  const call = calls.find((c) => c.method === 'POST');
  assert.equal(new URL(call.url).pathname, '/v1/decisions');
  assert.equal(call.auth, `Bearer ${KEY}`);
  assert.deepEqual(JSON.parse(call.body), choice);
  assert.equal(logs.at(-1).status, 200);
  assert.ok(!JSON.stringify(logs).includes(KEY), 'key must never be logged');
});

test('request validation: size, content type, empty body, JSON', async () => {
  const {run, calls} = setup();
  const big = JSON.stringify({...choice, state: 'x'.repeat(MAX_REQUEST_BYTES)});
  const declared = await run('/api/infer', post(big));
  assert.equal(declared.status, 413);
  assert.equal((await declared.json()).code, 'request_too_large');
  const chunks = new ReadableStream({start(c) { for (let i = 0; i < 5; i += 1) c.enqueue(new TextEncoder().encode('y'.repeat(10_000))); c.close(); }});
  const streamed = await run('/api/infer', {method: 'POST', headers: {'content-type': 'application/json', host: HOST}, body: chunks, duplex: 'half'});
  assert.equal(streamed.status, 413);
  const text = await run('/api/infer', post(choice, {'content-type': 'text/plain'}));
  assert.equal(text.status, 400);
  const body = await text.json();
  assert.equal(body.code, 'invalid_input');
  assert.equal(body.error_code, 'VALIDATION_ERROR');
  assert.equal(typeof body.request_id, 'string');
  assert.equal(text.headers.get('x-request-id'), body.request_id);
  assert.equal((await run('/api/infer', post(''))).status, 400);
  assert.equal((await run('/api/infer', post('{not json'))).status, 400);
  assert.equal((await run('/api/infer', post('[1,2]'))).status, 400);
  assert.equal(calls.filter((c) => c.method === 'POST').length, 0, 'invalid requests never reach production');
});

test('same-origin check with VEYA_ALLOWED_ORIGINS', async () => {
  const {run} = setup({env: {VEYA_ALLOWED_ORIGINS: 'https://veyadynamics.com'}});
  assert.equal((await run('/api/infer', post(choice, {origin: `https://${HOST}`}))).status, 200);
  assert.equal((await run('/api/infer', post(choice, {origin: 'https://veyadynamics.com'}))).status, 200);
  const cross = await run('/api/infer', post(choice, {origin: 'https://evil.example'}));
  assert.equal(cross.status, 400);
  assert.equal((await cross.json()).detail, 'Cross-origin model requests are not allowed.');
});

test('per-IP weighted rate limit (12 units / 60 s), local fallback', async () => {
  const {run} = setup();
  for (let i = 0; i < 12; i += 1) assert.equal((await run('/api/infer', post(choice))).status, 200, `request ${i}`);
  const blocked = await run('/api/infer', post(choice));
  assert.equal(blocked.status, 429);
  assert.ok(Number(blocked.headers.get('retry-after')) >= 1);
  assert.equal((await blocked.json()).code, 'rate_limited');
  assert.equal((await run('/api/infer', post(choice, {'cf-connecting-ip': '203.0.113.9'}))).status, 200);
  const wide = Object.fromEntries(Array.from({length: 60}, (_, i) => [`c${i}`, `candidate ${i}`]));
  assert.equal(weight({primitive: 'choice', criteria: wide}), 4); // 1 + ceil((60 - 19) / 20), as web/server.py
  assert.equal(weight({primitive: 'noul'}), 1);
});

test('rate-limit binding is used when bound', async () => {
  const keys = [];
  let allow = 2;
  const binding = {async limit({key}) { keys.push(key); allow -= 1; return {success: allow >= 0}; }};
  const {run} = setup({env: {VEYA_IP_RATE: binding}});
  assert.equal((await run('/api/infer', post(choice))).status, 200);
  assert.equal((await run('/api/infer', post(choice))).status, 200);
  assert.equal((await run('/api/infer', post(choice))).status, 429);
  assert.deepEqual([...new Set(keys)], ['198.51.100.7']);
});

test('upstream failures are always JSON with gateway codes', async () => {
  const cases = [
    [() => jsonResponse(503, {code: 'model_busy'}, {'retry-after': '2'}), 503, 'model_busy', '2'],
    [() => jsonResponse(504, {code: 'upstream_timeout'}), 504, 'timeout', null],
    [() => new Response('<html>bad gateway</html>', {status: 502}), 503, 'model_unavailable', '5'],
    [() => new Response('oops', {status: 500}), 503, 'model_unavailable', '5'],
    [() => jsonResponse(401, {code: 'unauthorized'}), 503, 'model_unavailable', '5'],
    [() => jsonResponse(429, {code: 'rate_limited'}, {'retry-after': '60'}), 429, 'rate_limited', '60'],
    [() => { throw new Error('binding down'); }, 503, 'model_unavailable', '5'],
  ];
  for (const [upstream, status, code, retry] of cases) {
    const {run} = setup({upstream: (path) => (path === '/v1/decisions' ? upstream() : jsonResponse(200, {status: 'ok'}))});
    const response = await run('/api/infer', post(choice));
    assert.equal(response.status, status, code);
    assert.match(response.headers.get('content-type'), /application\/json/);
    const body = await response.json();
    assert.equal(body.code, code);
    assert.ok(body.error_code && body.request_id && body.error);
    assert.equal(response.headers.get('retry-after'), retry);
    assert.ok(!JSON.stringify(body).includes('unauthorized'), 'upstream auth details are not leaked');
  }
});

test('upstream validation errors pass through; invalid success body is a JSON 502', async () => {
  const reject = setup({upstream: () => jsonResponse(400, {code: 'unsupported_primitive', detail: 'SystemOne is not served by this release.', request_id: 'up1'})});
  const r = await reject.run('/api/infer', post({primitive: 'systemone', state: 's', questions: {}}));
  assert.equal(r.status, 400);
  const body = await r.json();
  assert.equal(body.code, 'unsupported_primitive');
  assert.equal(body.error_code, 'VALIDATION_ERROR');
  const broken = setup({upstream: () => new Response('not json', {status: 200})});
  const b = await broken.run('/api/infer', post(choice));
  assert.equal(b.status, 502);
  assert.equal((await b.json()).code, 'invalid_upstream_response');
});

test('inference deadline returns a JSON 504', async () => {
  const {run} = setup({env: {VEYA_INFER_TIMEOUT_MS: '1000'}, upstream: () => new Promise((resolve) => setTimeout(() => resolve(jsonResponse(200, answer)), 1500))});
  const response = await run('/api/infer', post(choice));
  assert.equal(response.status, 504);
  assert.equal((await response.json()).code, 'timeout');
});

test('missing key or binding is a JSON 503, never a crash', async () => {
  for (const env of [{VEYA_SITE_KEY: ''}, {VEYA_API: undefined}]) {
    const {run} = setup({env});
    const response = await run('/api/infer', post(choice));
    assert.equal(response.status, 503);
    assert.equal((await response.json()).code, 'model_unavailable');
  }
});

test('__edge_health, /api/health and /readyz really probe veya-production', async () => {
  const healthy = setup();
  const edge = await healthy.run('/__edge_health');
  assert.equal(edge.status, 200);
  const e = await edge.json();
  assert.equal(e.upstream.status, 200);
  assert.equal(e.upstream.ok, true);
  assert.equal(e.site_key_configured, true);
  assert.ok(!JSON.stringify(e).includes(KEY));
  assert.ok(healthy.calls.some((c) => new URL(c.url).pathname === '/healthz'));
  assert.equal((await (await healthy.run('/api/health')).json()).status, 'ok');
  assert.equal((await healthy.run('/readyz')).status, 200);
  assert.deepEqual(await (await healthy.run('/healthz')).json(), {status: 'alive'});

  const down = setup({upstream: () => { throw new Error('down'); }});
  const d = await down.run('/__edge_health');
  assert.equal(d.status, 503);
  assert.equal((await d.json()).upstream.ok, false);
  const h = await down.run('/api/health');
  assert.equal(h.status, 503);
  assert.deepEqual(await h.json(), {status: 'unavailable', code: 'model_unavailable', error: 'model_unavailable', error_code: 'INTERNAL_ERROR'});
  assert.equal((await down.run('/readyz')).status, 503);
});

test('/api/version maps the production version to the gateway shape', async () => {
  const {run} = setup();
  const response = await run('/api/version');
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.model_version, '0.3.5');
  assert.equal(body.release, '0.3.5-audit-baseline.1');
  assert.deepEqual(body.supported_primitives, ['choice', 'noul', 'score']);
  assert.equal(body.release_status.english_choice, 'RESEARCH RELEASE');
  const down = setup({upstream: () => jsonResponse(401, {code: 'unauthorized'})});
  assert.equal((await down.run('/api/version')).status, 503);
});
