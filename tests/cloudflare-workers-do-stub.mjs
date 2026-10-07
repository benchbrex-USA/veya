// Minimal `cloudflare:workers` stub whose DurableObject keeps ctx/env, so a
// Container subclass can be constructed in Node tests.
export class DurableObject {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
  }
}
export class WorkerEntrypoint {}
