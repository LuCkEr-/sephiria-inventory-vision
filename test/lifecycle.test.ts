import assert from "node:assert/strict";
import { join, resolve } from "node:path";
import test from "node:test";

import { ResourceLifecycle } from "../src/lifecycle.js";
import {
  createInventoryCascadeDetector,
  createInventoryDetector,
  createInventoryVisionDetector,
} from "../src/index.js";

const fixture = join(
  resolve(import.meta.dirname, ".."),
  "test",
  "fixtures",
  "real",
  "all",
  "16-08-37.png",
);

test("resource lifecycle defers release until every active operation exits", () => {
  let releases = 0;
  const lifecycle = new ResourceLifecycle("TestResource", () => {
    releases += 1;
  });
  const leaveFirst = lifecycle.enter();
  const leaveSecond = lifecycle.enter();

  lifecycle.dispose();
  lifecycle.dispose();
  assert.equal(releases, 0);
  assert.throws(() => lifecycle.enter(), /TestResource has been disposed/);

  leaveFirst();
  leaveFirst();
  assert.equal(releases, 0);
  leaveSecond();
  assert.equal(releases, 1);
});

test("every detector defers native release for in-flight detection", async (context) => {
  const factories = [
    ["InventoryDetector", createInventoryDetector],
    ["InventoryVisionDetector", createInventoryVisionDetector],
    ["InventoryCascadeDetector", createInventoryCascadeDetector],
  ] as const;
  for (const [name, create] of factories) {
    await context.test(name, async () => {
      const detector = await create();
      const inFlight = detector.detect(fixture);
      detector.dispose();

      const result = await inFlight;
      assert.equal(result.slots.length, 24);
      await assert.rejects(detector.detect(fixture), new RegExp(`${name} has been disposed`));
      detector.dispose();
    });
  }
});
