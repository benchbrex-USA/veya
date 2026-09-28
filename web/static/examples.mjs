/** Documentation examples use the same gateway shape as the live playground. */
export const choiceRequest = {
  primitive: 'choice',
  state: 'A customer was charged twice and requests a refund.',
  instructions: 'Which team should handle this?',
  criteria: { billing: 'Payments and refunds', technical: 'Bugs and outages', other: 'Other requests' }
};
export const noulRequest = {
  primitive: 'noul',
  state: 'The customer says they will cancel if the duplicate charge remains.',
  instructions: 'Does the customer threaten to cancel?'
};
export const scoreRequest = {
  primitive: 'score',
  state: 'The production dashboard is down and customers are blocked.',
  instructions: 'How urgent is this request?',
  criteria: ['Routine', 'Needs attention soon', 'Critical / blocking']
};
export const systemOneRequest = {
  primitive: 'systemone',
  state: 'A customer was charged twice and says they will cancel.',
  questions: {
    department: { type: 'choice', instructions: 'Which team?', criteria: { billing: 'Payments and refunds', other: 'Other' } },
    churn_risk: { type: 'noul', instructions: 'Does the customer threaten to cancel?' }
  }
};
export const playgroundExamples = {
  choice: choiceRequest,
  noul: noulRequest,
  score: scoreRequest,
  systemone: systemOneRequest
};
const body = JSON.stringify(choiceRequest, null, 2);
const gatewayOrigin = typeof location !== 'undefined' && /^https?:$/.test(location.protocol)
  ? location.origin
  : 'http://127.0.0.1:3000';
export const quickstart = {
  curl: `curl -X POST '${gatewayOrigin}/api/infer' \\\n  -H 'Content-Type: application/json' \\\n  -d '${body}'`,
  python: `import json\nfrom urllib.request import Request, urlopen\n\nbody = ${JSON.stringify(choiceRequest, null, 2)}\nrequest = Request(\n    ${JSON.stringify(`${gatewayOrigin}/api/infer`)},\n    data=json.dumps(body).encode(),\n    headers={"Content-Type": "application/json"},\n)\nwith urlopen(request, timeout=25) as response:\n    print(json.load(response))`,
  typescript: `const response = await fetch('/api/infer', {\n  method: 'POST',\n  headers: { 'Content-Type': 'application/json' },\n  body: JSON.stringify(${JSON.stringify(choiceRequest, null, 2)})\n});\nif (!response.ok) throw new Error('Veya request failed');\nconst result = await response.json();\nconsole.log(result.decision, result.answer, result.probabilities);`
};
