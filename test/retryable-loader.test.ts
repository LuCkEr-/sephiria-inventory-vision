import assert from "node:assert/strict";
import test from "node:test";

import { RetryableLoader } from "../src/retryable-loader.js";

test("retryable loader deduplicates, retries failures, and caches success", async () => {
  let attempts = 0;
  const loader = new RetryableLoader(async () => {
    attempts += 1;
    await Promise.resolve();
    if (attempts === 1) throw new Error("transient");
    return 42;
  }, "Dependency initialization failed");

  const first = loader.get();
  assert.equal(loader.get(), first);
  await assert.rejects(first, (error: unknown) => {
    assert(error instanceof Error);
    assert.equal(error.message, "Dependency initialization failed");
    assert.match(String(error.cause), /transient/);
    return true;
  });

  const retry = loader.get();
  assert.equal(loader.get(), retry);
  assert.equal(await retry, 42);
  assert.equal(attempts, 2);
  assert.equal(await loader.get(), 42);
  assert.equal(attempts, 2);
});

test("retryable loader preserves unwrapped synchronous failures", async () => {
  const failure = new Error("synchronous");
  const loader = new RetryableLoader<number>(() => {
    throw failure;
  });

  await assert.rejects(loader.get(), (error: unknown) => error === failure);
  await assert.rejects(loader.get(), (error: unknown) => error === failure);
});
