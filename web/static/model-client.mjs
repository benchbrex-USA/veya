/** Typed, same-origin client for the public Veya website contract. No mock adapter exists. */
/** @typedef {'choice'|'noul'|'score'|'systemone'} Primitive */
/** @typedef {{primitive:Primitive,state:string|object|Array<unknown>,instructions?:string,criteria?:Record<string,string>|string[],questions?:Record<string,{type:'choice'|'noul'|'score',instructions:string,criteria?:Record<string,string>|string[]}>}} VeyaRequest */
/** @typedef {{decision:'answer'|'defer',deferred:boolean,primitive:Primitive,answer?:string|number|null,probabilities?:Record<string,number>|null,answers?:Record<string,object>,defer_reason?:string|null,route?:string|null,model?:string|null,model_version?:string,release?:string,request_id:string,gateway_latency_ms:number}} VeyaResponse */

export class VeyaError extends Error {
  /** @param {string} code @param {string} message @param {number} status @param {number|null} retryAfter */
  constructor(code, message, status = 0, retryAfter = null) {
    super(message);
    this.name = 'VeyaError';
    this.code = code;
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

const primitiveSet = new Set(['choice', 'noul', 'score', 'systemone']);
const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const isProbability = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;

/** @param {unknown} value @param {VeyaRequest} request @returns {VeyaResponse} */
export function parseDecision(value, request) {
  if (!isObject(value) || !['answer', 'defer'].includes(value.decision) || value.primitive !== request.primitive || typeof value.request_id !== 'string' || typeof value.gateway_latency_ms !== 'number') {
    throw new VeyaError('invalid_response', 'The model returned an invalid response.', 502);
  }
  if (value.deferred !== (value.decision === 'defer')) {
    throw new VeyaError('invalid_response', 'The model returned conflicting decision states.', 502);
  }
  if (value.decision === 'defer') {
    if (typeof value.defer_reason !== 'string') throw new VeyaError('invalid_response', 'The model did not provide a defer reason.', 502);
    return /** @type {VeyaResponse} */ (value);
  }
  if (request.primitive === 'systemone') {
    if (!isObject(value.answers) || !isObject(request.questions)) throw new VeyaError('invalid_response', 'SystemOne results are incomplete.', 502);
    const ids = Object.keys(request.questions);
    if (ids.length !== Object.keys(value.answers).length || ids.some(id => !isObject(value.answers[id]))) throw new VeyaError('invalid_response', 'SystemOne question IDs do not match.', 502);
    value.answers = Object.fromEntries(ids.map(id => [id, value.answers[id]]));
    return /** @type {VeyaResponse} */ (value);
  }
  if (request.primitive === 'noul') {
    if (!isProbability(value.answer)) throw new VeyaError('invalid_response', 'Noul probability is invalid.', 502);
  } else if (request.primitive === 'choice') {
    if (typeof value.answer !== 'string' || !isObject(value.probabilities) || !isProbability(value.probabilities[value.answer])) throw new VeyaError('invalid_response', 'Choice result is invalid.', 502);
  } else if (request.primitive === 'score') {
    if (typeof value.answer !== 'number' || !Number.isFinite(value.answer) || !isObject(value.probabilities)) throw new VeyaError('invalid_response', 'Score result is invalid.', 502);
  }
  return /** @type {VeyaResponse} */ (value);
}

export class VeyaClient {
  /** @param {(input:RequestInfo|URL, init?:RequestInit)=>Promise<Response>} fetcher */
  constructor(fetcher = (input, init) => fetch(input, init)) { this.fetcher = fetcher; }

  async health() {
    try {
      const response = await this.fetcher('/api/health', {credentials:'same-origin'});
      const data = await response.json();
      return response.ok && data?.status === 'ok';
    } catch { return false; }
  }

  async version() {
    const response = await this.fetcher('/api/version', {credentials:'same-origin'});
    if (!response.ok) throw new VeyaError('model_unavailable', 'Model version unavailable.', response.status);
    const data = await response.json();
    if (!isObject(data) || typeof data.model_version !== 'string') throw new VeyaError('invalid_response', 'Model version response is invalid.', 502);
    return data;
  }

  /** @param {VeyaRequest} request @param {{signal?:AbortSignal}} options @returns {Promise<VeyaResponse>} */
  async infer(request, options = {}) {
    if (!isObject(request) || !primitiveSet.has(request.primitive)) throw new VeyaError('invalid_input', 'Select a supported primitive.', 400);
    let response;
    try {
      response = await this.fetcher('/api/infer', {method:'POST', credentials:'same-origin', headers:{'Content-Type':'application/json'}, body:JSON.stringify(request), signal:options.signal});
    } catch (error) {
      if (error?.name === 'AbortError') throw error;
      throw new VeyaError('network_error', 'Could not reach the Veya website. Please retry.');
    }
    let data;
    try { data = await response.json(); }
    catch { throw new VeyaError('invalid_response', 'The website returned an invalid response.', response.status); }
    if (!response.ok) {
      const retryHeader = Number(response.headers?.get?.('Retry-After'));
      const retryAfter = Number.isFinite(retryHeader) && retryHeader > 0 ? retryHeader : Number.isFinite(data?.retry_after) ? data.retry_after : null;
      const error = new VeyaError(typeof data?.code === 'string' ? data.code : 'request_failed', typeof data?.detail === 'string' ? data.detail : 'The request failed.', response.status, retryAfter);
      error.deferReason = typeof data?.defer_reason === 'string' ? data.defer_reason : null;
      error.requestId = typeof data?.request_id === 'string' ? data.request_id : null;
      throw error;
    }
    return parseDecision(data, request);
  }
}
