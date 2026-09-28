/** Homepage narrative. Model figures are fetched from the sealed artifact. */
export const homeContent = {
  eyebrow: 'VEYA / DECISION RESEARCH',
  title: 'AI should know\nwhen not to decide.',
  summary: 'Veya is a probabilistic model for structured decisions. Give it state, a question, and possible outcomes. Inspect the answer, distribution, or defer.',
  thesis: {
    label: 'THE THESIS / 01',
    title: 'Decisions have a shape.',
    summary: 'Generative interfaces produce language. Veya works over a defined outcome space and returns a typed result that software can inspect. Both approaches have uses; their outputs are different.',
    generative: ['Input', 'Tokens', 'String'],
    veya: ['State + question', 'Alternatives', 'Probabilities', 'Answer / defer']
  },
  principles: [
    { number: '01', title: 'Decisions,\nnot prose.', detail: 'Choice selects from named alternatives. Noul evaluates a proposition. Score estimates over an ordered categorical scale.' },
    { number: '02', title: 'Probabilities,\nin context.', detail: 'Every number belongs to a model route and an evidence boundary. A high probability is not a correctness guarantee.' },
    { number: '03', title: 'Defer,\nwhen the path cannot answer.', detail: 'Unsupported language and other withheld decisions stay visible as deferrals instead of becoming guessed answers.' }
  ],
  // Audited defaults from artifacts/release/manifest.json; English Choice is refreshed from /api/version.
  status: [
    { label: 'ENGLISH CHOICE', value: 'Research release', liveKey: 'english_choice' },
    { label: 'AUTOMATION', value: 'Disabled' },
    { label: 'PHASE 9 SCORER', value: 'Research only · not served' },
    { label: 'ORDINAL SCORE', value: 'Research only' },
    { label: 'MULTILINGUAL', value: 'Not served by this checkpoint' }
  ],
  articles: [
    { number: '01', category: 'ARCHITECTURE NOTE', title: 'Where the shortlist loses the answer', description: 'Trace the served high-N path and its cross-domain retrieval limit.', href: '/architecture/high-cardinality-choice' },
    { number: '02', category: 'EVALUATION NOTE', title: 'What the sealed run actually measured', description: 'Read the sample, split, retrieval, end-to-end, and calibration results together.', href: '/benchmarks' },
    { number: '03', category: 'RELEASE NOTE', title: 'Served today. Research tomorrow.', description: 'See what the checkpoint serves, and which experiments are not active.', href: '/model-status' }
  ],
  faq: [
    { question: 'What is Veya?', answer: 'Veya is a research model for typed, structured decisions over supplied state and outcomes. The current website exposes Choice, Noul, categorical Score, and SystemOne through a validated gateway.' },
    { question: 'How is this different from a conversational model?', answer: 'The request defines a question and an outcome space. The response is a structured decision with probabilities or a defer, rather than an open-ended passage of text.' },
    { question: 'What does a probability mean?', answer: 'It is the served model’s estimated weight for an outcome on that request and route. It is not a universal chance of correctness or a guarantee that an automated action is safe.' },
    { question: 'When does Veya defer?', answer: 'The service can withhold an answer, including for confidently detected unsupported language. The response carries a reason where available.' },
    { question: 'Is Veya ready to automate decisions?', answer: 'No. Automation is disabled in this audited research release. The sealed high-N evaluation shows a substantial cross-domain retrieval limitation.' },
    { question: 'Which languages are served?', answer: 'English is the only served language in the audited checkpoint. The separate multilingual work is not a validated release path.' },
    { question: 'Where can I inspect the exact API response?', answer: 'Run a request in the Experience page or the live instrument below, then open the technical drawer and select JSON.' }
  ]
};
