import assert from "node:assert/strict";
import test from "node:test";

import { evaluateGameAssetOpenSet } from "../scripts/lib/game-asset-open-set.js";
import { DEFAULT_VISION_ACCEPTANCE_THRESHOLD } from "../src/vision-calibration.js";

test("every unseen extracted game-item template is rejected by the vision model", async () => {
  const report = await evaluateGameAssetOpenSet();
  assert.equal(report.acceptanceThreshold, DEFAULT_VISION_ACCEPTANCE_THRESHOLD);
  assert.equal(report.queries, 461);
  assert.equal(report.rejected, report.queries);
  assert.equal(report.accepted, 0);
  assert.equal(report.falseAcceptRate, 0);
  assert.deepEqual(report.acceptedExamples, []);
});
