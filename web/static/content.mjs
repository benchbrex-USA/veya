import { choiceRequest, noulRequest, scoreRequest, systemOneRequest, quickstart } from './examples.mjs?v=docs-2';

const p = text => ({type:'p',text});
const list = (...items) => ({type:'list',items});
const code = (value, language='json', title='') => ({type:'code',code:typeof value==='string'?value:JSON.stringify(value,null,2),language,title});
const note = (text,tone='info') => ({type:'callout',text,tone});
const links = (...items) => ({type:'links',links:items});
const table = (headers, rows) => ({type:'table',headers,rows});
const flow = (...items) => ({type:'flow',items});
const section = (title,...blocks) => ({title,blocks});

/** Page copy and examples. Layout lives in docs.mjs and ui.mjs. */
export const docsContent = {
  '/getting-started': {
    category:'START HERE', heading:'Getting started', summary:'Make one structured decision, inspect its distribution, then decide whether the result is suitable for your use case.', status:'Research',
    sections:[
      section('Run the local services',p('From the repository root, start the audited model and website gateway in separate terminals. The model service binds loopback; the browser calls the website gateway.'),code('VEYA_DEVICE=cpu python3 -m web.model_server\npython3 -m web.server','bash','Local startup')),
      section('Make a Choice request',p('Send a state, one question, and at least two named candidates to the same-origin gateway. The live playground uses this exact request shape.'),{type:'codeTabs',options:[{label:'curl',language:'bash',code:quickstart.curl},{label:'Python',language:'python',code:quickstart.python},{label:'TypeScript',language:'typescript',code:quickstart.typescript}]}),
      section('Read the result',p('An answer carries a selected candidate and a distribution. A defer carries a reason and no selected candidate. Probabilities describe model output; they are not a correctness or automation guarantee.'),flow('Check decision: answer or defer.','If answered, inspect answer and probabilities.','Inspect route, release, latency, and request ID.','Keep a human in the loop; automation is disabled for this research release.'),links({href:'/experience',label:'Try the live model'},{href:'/developers/api',label:'Read the API contract'})),
      section('Continue',p('Choice, Noul, Score, and SystemOne share the same state/question discipline, but have different output semantics.'),links({href:'/concepts/choice',label:'Explore core concepts'},{href:'/limitations',label:'Read current limits'}))
    ]
  },
  '/concepts/choice': {
    category:'CORE CONCEPT', heading:'Choice', summary:'Select one outcome from a defined candidate set and inspect the probabilities assigned to the served candidates.', status:'Research',
    sections:[
      section('What it is',p('Choice takes a state, a question, and a map of candidate IDs to descriptions. The answer is one submitted ID or a defer.')), 
      section('How it works',p('The audited English checkpoint uses a packed route for 2–19 candidates. At 20–100 candidates, a learned top-eight shortlist feeds the base decision head and route-specific calibration.'),note('For high-N Choice, the returned distribution covers the eight shortlisted candidates. Omitted candidates are not represented in that distribution.','warning')),
      section('Example request',code(choiceRequest)),
      section('Response',p('Inspect decision, answer, probabilities, candidate_count, route, calibration, release, and request_id. The live result drawer shows the exact API response.'),links({href:'/experience',label:'Run Choice'})),
      section('When to use it',p('Use Choice when the outcome space is explicit and mutually exclusive, such as routing a request among named teams.')),
      section('Limits',p('The website accepts 2–100 candidates with unique IDs and bounded text. English is the only served language; unsupported language can defer. The high-N shortlist has limited cross-domain retrieval evidence.'),links({href:'/architecture/high-cardinality-choice',label:'High-N architecture'},{href:'/benchmarks',label:'See blind results'}))
    ]
  },
  '/concepts/noul': {
    category:'CORE CONCEPT', heading:'Noul', summary:'Evaluate the probability of one proposition against the supplied state.', status:'Research',
    sections:[
      section('What it is',p('Noul returns a number between 0 and 1 for the proposition expressed in the question. It is a probability estimate, not an absolute true/false claim.')),
      section('How it works',p('The current English checkpoint has a Noul head. This audited release does not include an independent Noul calibration artifact.')),
      section('Example request',code(noulRequest)),
      section('Response',p('For an answer, the numeric answer is the estimated true probability. A defer carries defer_reason and no probability answer. The website presents the value as a percentage while preserving the raw number in JSON.'),links({href:'/experience',label:'Run Noul'})),
      section('When to use it',p('Use Noul for one explicit proposition, such as whether a customer has threatened to cancel.')),
      section('Limits',p('English only in this served checkpoint. The probability is not certified for automated decisions; Noul is research only.'),links({href:'/model-status',label:'Model status'}))
    ]
  },
  '/concepts/score': {
    category:'CORE CONCEPT', heading:'Score', summary:'Estimate a position on a supplied ordered categorical scale and inspect the category distribution.', status:'Research',
    sections:[
      section('What it is',p('Score accepts ordered labels and returns a categorical expected value plus probabilities over the indexed labels. The current served path is not a validated ordinal head.')),
      section('How it works',p('The checkpoint scores the provided categories. The website restores your labels in the displayed distribution.')),
      section('Example request',code(scoreRequest)),
      section('Response',p('The answer is a numeric expected category index. Review the full distribution and label legend before interpreting it; a number such as 1.8 on a three-level scale is not an absolute statement.')),
      section('When to use it',p('Use Score when the labels themselves form a meaningful ordered scale, such as routine, soon, and critical.')),
      section('Limits',p('The website accepts 2–10 unique labels. Categorical Score is research only; ordinal Score has not been promoted.'),links({href:'/limitations',label:'Read limitations'}))
    ]
  },
  '/concepts/systemone': {
    category:'CORE CONCEPT', heading:'SystemOne', summary:'Ask several typed questions over one shared state and keep every result attached to its question ID.', status:'Research',
    sections:[
      section('What it is',p('SystemOne combines Choice, Noul, and Score questions in one request. It returns answers keyed by the submitted question IDs.')),
      section('How it works',p('The current API responds atomically. A partial-success schema is not served, and question-conditioned attention means each question is evaluated separately against the shared state.')),
      section('Example request',code(systemOneRequest)),
      section('Response',p('Read answers.department and answers.churn_risk separately. The gateway checks that every submitted ID has a corresponding typed output.')),
      section('When to use it',p('Use SystemOne when several related decisions depend on the same state, such as routing and churn-risk assessment.')),
      section('Limits',p('The website accepts 1–8 questions. Any invalid or unsupported question can prevent an atomic answer.'),links({href:'/experience',label:'Run SystemOne'}))
    ]
  },
  '/architecture': {
    category:'SYSTEM', heading:'Architecture', summary:'Trace an actual request from the browser to an answer or defer.', status:'Research',
    sections:[
      section('Serving path',flow('Browser calls the same-origin website gateway.','Gateway validates schema, request size, candidate limits, and rate limit.','Veya service applies language gate and primitive route.','The served checkpoint scores the relevant outcome space.','Route-specific calibration produces returned probabilities where available.','The response is an answer or defer with safe technical metadata.'),note('The website never calls the model service directly and never receives the server-side token.')),
      section('Primitive routes',table(['Primitive','Current path'],[['Choice 2–19','Packed decision head'],['Choice 20–100','Learned top-8 shortlist → packed rerank'],['Noul','Proposition probability head'],['Score','Categorical score head'],['SystemOne','Typed questions keyed by ID']])),
      section('Calibration and policy',p('Choice uses route-specific calibration metadata. Noul and categorical Score do not have independent audited calibration artifacts in this release. Automation policy is disabled; the service may answer or defer.'),links({href:'/model-status',label:'Current model status'},{href:'/limitations',label:'Limitations'})),
      section('Inspect a real response',p('The playground technical drawer exposes route, calibration, release, latency, candidate count, and request ID when supplied by the API.'),links({href:'/experience',label:'Open the model'}))
    ]
  },
  '/architecture/high-cardinality-choice': {
    category:'SYSTEM / CHOICE', heading:'High-cardinality Choice', summary:'At 20 or more candidates, Veya retrieves eight candidate IDs before scoring the shortlist.', status:'Research',
    sections:[
      section('Current served route',flow('Validate 20–100 named candidates.','Use the learned K=8 shortlist.','Score those eight through the base packed decision head.','Apply the high-N route calibration.','Return an answer or defer plus route and shortlist metadata.')),
      section('What is not served',p('The experimental Phase 9 hypothesis scorer is not loaded by the audited manifest. The website does not present it as an active production route.')),
      section('Measured limit',p('The sealed CLINC150 validation run found shortlist retrieval to be the main bottleneck. The benchmark page loads the aggregate artifact and explains the sample, split, and metrics.'),links({href:'/benchmarks',label:'Benchmark methodology'})),
      section('Interpretation',note('A normalized distribution over the returned shortlist is not a distribution over every submitted candidate. Retrieval misses are the main blind bottleneck.','warning'))
    ]
  },
  '/benchmarks': {
    category:'EVIDENCE', heading:'Benchmarks', summary:'A sealed aggregate run shows both the current model’s strengths and its retrieval limit.', status:'Research',
    sections:[
      section('Sealed CLINC150 validation',p('The metrics below are loaded from the audited aggregate artifact. Dataset: CLINC150 validation, in-scope examples. Split: val. Sample count: 3,000. Candidate universe: 150 labels. Release: 0.3.5-audit-baseline.1.'),{type:'metrics',items:[{label:'Recall@8',value:'—',note:'True label reached the shortlist'},{label:'Conditional accuracy',value:'—',note:'Correct on shortlist hits only'},{label:'End-to-end accuracy',value:'—',note:'Including retrieval misses'},{label:'End-to-end ECE',value:'—',note:'Calibration error; lower is better'}]},note('These are research benchmark results, not a production quality guarantee. Automation is disabled.','warning')),
      section('Methodology',p('The untouched CLINC150 val partition was frozen as a replacement blind set and evaluated once with aggregate-only output. An earlier CLINC150 test run failed during persistence and was not rerun. Banking77 test rows used in prior phases are development-exposed and cannot certify this release.')),
      section('What the numbers mean',p('Recall@8 measures whether the true candidate survived retrieval. End-to-end accuracy counts both retrieval and scoring errors. ECE compares confidence with observed correctness; lower is better. Do not interpret conditional accuracy on shortlist hits as full-system accuracy.')),
      section('Evidence boundary',p('This benchmark is one high-cardinality English Choice setting. It does not establish Noul, Score, multilingual, domain-general, or automation reliability.'),links({href:'/limitations',label:'Read limitations'},{href:'/model-status',label:'Model status'}))
    ]
  },
  '/limitations': {
    category:'EVIDENCE', heading:'Limitations', summary:'The current checkpoint is useful for research exploration. These limits remain visible by design.', status:'Research',
    sections:[
      section('Supported scope',list('English is the only served language in the audited checkpoint.','Choice is a research release; Noul and categorical Score are research only.','Automation is disabled. A probability is not a correctness or risk guarantee.')),
      section('Input and serving limits',list('The website accepts up to 32 KB per request and 12 KB for state.','Choice accepts 2–100 candidates; Score accepts 2–10 labels; SystemOne accepts 1–8 questions.','Inference times out after 20 seconds. The public demo limiter is per process and per socket IP.')),
      section('Model limits',list('The learned K=8 shortlist is Banking77-trained; cross-domain retrieval is weak in the sealed CLINC150 run.','Noul and categorical Score lack independent audited calibration artifacts.','Ordinal Score and multilingual calibration are not served as validated production paths.','Experimental Phase 9 hypothesis scoring is not loaded.')),
      section('Security and deployment',p('Candidate and state text are untrusted. The gateway validates structure and escapes rendered output, but this does not certify prompt-injection robustness of every model input. A public multi-instance deployment needs shared rate limiting, authentication/ingress policy, TLS, and a privacy review.'),links({href:'/developers/api',label:'Gateway limits'}))
    ]
  },
  '/model-status': {
    category:'EVIDENCE', heading:'Model status', summary:'A concise view of what the audited checkpoint serves and what remains research only.', status:'Research',
    sections:[
      section('Current status',p('The matrix below starts from the audited manifest and checks /api/version for the active release. If the model is offline, the page says so rather than inventing live status.'),table(['Component','Audited status'],[['English Choice','Research release'],['English Noul','Research only'],['English categorical Score','Research only'],['Multilingual primitives','Unsupported by served checkpoint'],['Ordinal Score','Research only'],['Phase 9 hypothesis scorer','Not served']])),
      section('Live release metadata',p('Active model version, release, supported languages, and primitive metadata are shown when the service is reachable. The underlying checkpoint remains a research artifact.')),
      section('Next evidence needed',p('Independent calibration and stronger blind retrieval evidence are required before stronger reliability or automation claims would be justified.'),links({href:'/benchmarks',label:'Inspect benchmark evidence'}))
    ]
  },
  '/developers': {
    category:'DEVELOPERS', heading:'Developer guide', summary:'Use a small, typed JSON request and inspect the structured response.', status:'Research',
    sections:[
      section('One gateway',p('The browser calls POST /api/infer on the website origin. The gateway validates requests and responses, applies rate limits and timeouts, then calls the Veya service. Keep model credentials on the server.')),
      section('Quick request',{type:'codeTabs',options:[{label:'curl',language:'bash',code:quickstart.curl},{label:'Python',language:'python',code:quickstart.python},{label:'TypeScript',language:'typescript',code:quickstart.typescript}]}),
      section('Read the response',p('Check decision before reading answer. For Choice, probabilities identify the served candidate distribution. Noul returns a true probability; Score returns a categorical expected value. SystemOne returns outputs keyed by question ID.'),links({href:'/developers/api',label:'Full API reference'},{href:'/experience',label:'Inspect live JSON'})),
      section('Errors',p('The gateway sends structured errors with code, detail, and request_id. Rate limits include retry_after and a Retry-After header. Model outages never fall back to fake output.'))
    ]
  },
  '/developers/api': {
    category:'DEVELOPERS', heading:'API reference', summary:'The current website gateway contract for the audited research release.', status:'Research',
    sections:[
      section('Endpoints',table(['Method','Path','Purpose'],[['POST','/api/infer','Validated inference for all four primitives'],['GET','/api/health','Model reachability'],['GET','/api/version','Bounded release and capability metadata'],['GET','/api/benchmark','Sealed aggregate Choice benchmark']])),
      section('Request',p('Every inference request has primitive and state. Choice adds instructions and criteria object; Noul adds instructions; Score adds instructions and an ordered criteria array; SystemOne adds questions keyed by ID.'),code(choiceRequest)),
      section('Response',p('Successful results include decision, deferred, primitive, request_id, gateway_latency_ms, and release/version metadata when available. Answer and probability fields depend on the primitive. Defer includes defer_reason. The exact live JSON can be inspected in the playground.')),
      section('Validation limits',table(['Field','Limit'],[['Request body','32 KB'],['State','12 KB JSON encoding'],['Question','500 characters'],['Choice candidates','2–100, unique IDs'],['Candidate ID','80 characters'],['Candidate description','240 characters'],['Score labels','2–10'],['SystemOne questions','1–8']])),
      section('Errors and retry',table(['HTTP','Code','Meaning'],[['400','invalid_input / model_rejected','Invalid request or structured model rejection'],['404','not_found','Unknown API path'],['413','request_too_large','Request exceeds size limit'],['429','rate_limited','Try again after Retry-After'],['502','model_error / invalid_upstream_response','Model failure or invalid response'],['503','model_unavailable / model_busy','Unavailable or at concurrency limit'],['504','timeout','Model exceeded gateway timeout']])),
      section('Versioning and privacy',p('Read /api/version for the active release. The gateway logs only request ID, primitive, candidate count, route, defer state, status, latency, model version, and error code by default. It does not log raw state or candidate text.'))
    ]
  },
  '/principles': {
    category:'OVERVIEW', heading:'Principles', summary:'Veya makes its outcome space and uncertainty visible so decisions can be inspected.', status:'Research',
    sections:[
      section('Structure first',p('Define the question and the possible outcomes before inference. Typed inputs make a decision reproducible and inspectable.')),
      section('Probabilities in context',p('A high score is a model output, not a guarantee. Keep the distribution, route, and evidence boundary beside the result.')),
      section('Defer honestly',p('An unsupported language or withheld decision should remain a defer, not be converted into a guessed answer.')),
      section('Evidence before claims',p('Separate development-exposed results from sealed evaluation, and keep release status current.'),links({href:'/benchmarks',label:'See the evidence'}))
    ]
  }
};
