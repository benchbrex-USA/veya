import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";
import test from "node:test";

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === "cloudflare:workers") {
      return {
        shortCircuit: true,
        url: new URL("./cloudflare-workers-stub.mjs", import.meta.url).href,
      };
    }
    if ((specifier.startsWith("./") || specifier.startsWith("../")) && context.parentURL) {
      const exact = fileURLToPath(new URL(specifier, context.parentURL));
      if (!existsSync(exact) && existsSync(exact + ".js")) {
        return {
          shortCircuit: true,
          url: new URL(specifier + ".js", context.parentURL).href,
        };
      }
    }
    return nextResolve(specifier, context);
  },
});

test("the worker entrypoint exports ContainerProxy", async () => {
  const worker = await import("../worker/veya-container.mjs");
  assert.equal(typeof worker.ContainerProxy, "function");
  assert.equal(worker.ContainerProxy.name, "ContainerProxy");
  assert.equal(typeof worker.VeyaContainer, "function");
  assert.equal(typeof worker.default.fetch, "function");
});
