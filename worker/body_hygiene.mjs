/**
 * Container response hygiene.
 *
 * @cloudflare/containers (0.3.7, Container.containerFetch) counts a proxied
 * request as in flight until its response body has been piped to the end:
 *
 *   this.inflightRequests++;
 *   ...
 *   res.body?.pipeTo(writable).finally(() => this.decrementInflight());
 *   return new Response(readable, res);   // readable side of an IdentityTransformStream
 *
 * An IdentityTransformStream has no buffer, so the pipe only finishes when the
 * caller reads the body to the end or cancels it. While inflightRequests > 0,
 * isActivityExpired() renews the sleepAfter timer and returns false, so a single
 * response that nobody reads keeps the container awake indefinitely.
 *
 * Rule: every container response the edge does not hand to the client must be
 * read or cancelled, including responses that arrive after the edge stopped
 * waiting (timeouts, abandoned attempts).
 */

/** Cancel a response body we will not use. Safe on null, bodiless, used or locked responses. */
export async function discard(response) {
  if (!response || !response.body) return;
  try {
    await response.body.cancel();
  } catch {
    // already read, locked by a reader, or errored: nothing left to release
  }
}

/** Cancel the body of a response that may still arrive (timed-out or abandoned fetch). */
export function discardLate(pending) {
  return Promise.resolve(pending).then((response) => discard(response), () => undefined);
}

/**
 * Track the container fetches made for one unit of work so the ones we stop
 * waiting for can be cancelled whenever they settle. `release(keep)` discards
 * every tracked response except `keep` (the one handed to the client).
 */
export function fetchTracker() {
  const pending = [];
  return {
    fetch(stub, request) {
      const call = Promise.resolve().then(() => stub.fetch(request));
      pending.push(call);
      return call;
    },
    release(keep) {
      return Promise.all(pending.map((call) => call.then(
        (response) => (response === keep ? undefined : discard(response)),
        () => undefined,
      )));
    },
    get size() {
      return pending.length;
    },
  };
}
