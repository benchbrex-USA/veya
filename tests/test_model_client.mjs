import test from 'node:test';
import assert from 'node:assert/strict';
import {VeyaClient, VeyaError, parseDecision} from '../web/static/model-client.mjs';

const choice = {primitive:'choice',state:'duplicate charge',instructions:'Which team?',criteria:{billing:'refunds',technical:'bugs'}};
const reply = {decision:'answer',deferred:false,primitive:'choice',answer:'billing',probabilities:{billing:0.8,technical:0.2},request_id:'abc',gateway_latency_ms:100};
const response = (status, data, headers = new Headers()) => ({ok:status >= 200 && status < 300,status,headers,json:async()=>data});

test('serializes real Choice payload to same-origin gateway', async () => {
  let captured;
  const client = new VeyaClient(async (url, init) => {captured={url,init};return response(200,reply)});
  const result = await client.infer(choice);
  assert.equal(captured.url,'/api/infer');
  assert.equal(captured.init.method,'POST');
  assert.equal(captured.init.credentials,'same-origin');
  assert.deepEqual(JSON.parse(captured.init.body),choice);
  assert.equal(result.answer,'billing');
});

test('preserves SystemOne question association in submitted order', async () => {
  const request={primitive:'systemone',state:'x',questions:{q2:{type:'noul',instructions:'True?'},q1:{type:'noul',instructions:'True?'}}};
  const result=parseDecision({decision:'answer',deferred:false,primitive:'systemone',answers:{q1:{type:'noul',noul:.4},q2:{type:'noul',noul:.6}},request_id:'id',gateway_latency_ms:1},request);
  assert.deepEqual(Object.keys(result.answers),['q2','q1']);
});

test('rejects malformed success payload', async () => {
  const client=new VeyaClient(async()=>response(200,{decision:'answer',primitive:'choice'}));
  await assert.rejects(client.infer(choice),e=>e instanceof VeyaError&&e.code==='invalid_response');
});

test('surfaces rate limit and retry-after', async () => {
  const client=new VeyaClient(async()=>response(429,{code:'rate_limited',detail:'Too many model requests.'},new Headers({'Retry-After':'17'})));
  await assert.rejects(client.infer(choice),e=>e.code==='rate_limited'&&e.retryAfter===17);
});

test('cancellation is not converted to an outage', async () => {
  const client=new VeyaClient(async()=>{throw new DOMException('Cancelled','AbortError')});
  await assert.rejects(client.infer(choice),e=>e.name==='AbortError');
});

test('defer requires a reason', () => {
  const result=parseDecision({decision:'defer',deferred:true,primitive:'choice',defer_reason:'unsupported_language',request_id:'id',gateway_latency_ms:1},choice);
  assert.equal(result.defer_reason,'unsupported_language');
  assert.throws(()=>parseDecision({...result,defer_reason:null},choice),VeyaError);
});
