import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

import sharp from "sharp";

import {
  createInventoryCascadeDetector,
  createInventoryVisionDetector,
  type VisionModel,
} from "../src/index.js";
import {
  DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
  calibrateVisionAcceptanceThresholds,
} from "../src/vision-calibration.js";

const root = resolve(import.meta.dirname, "..");
const fixtureRoot = join(root, "test", "fixtures", "real", "all");
const manifest = JSON.parse(
  await readFile(join(root, "test", "fixtures", "real", "ground-truth.json"), "utf8"),
) as {
  screenshots: {
    fixtureFile: string;
    slots: { item: { name: string } | null }[];
  }[];
};

test("vision slots expose confidence and top-two margin diagnostics", async () => {
  const capture = manifest.screenshots.find(
    (candidate) => candidate.fixtureFile === "16-08-37.png",
  );
  assert.ok(capture);
  const detector = await createInventoryVisionDetector();
  try {
    const result = await detector.detect(join(fixtureRoot, capture.fixtureFile));
    assert.equal(result.slots.length, 24);
    for (const slot of result.slots) {
      assert.ok(slot.classification);
      const secondScore = slot.classification.secondScore;
      assert.ok(secondScore !== null);
      assert.ok(slot.classification.bestScore >= secondScore);
      assert.ok(slot.classification.margin >= 0);
      assert.ok(slot.classification.acceptanceThreshold >= DEFAULT_VISION_ACCEPTANCE_THRESHOLD);
      assert.ok(slot.classification.minimumMargin >= 0);
      assert.equal(slot.classification.accepted, slot.item !== null);
      assert.equal(slot.classification.acceptanceReason, slot.item ? "accepted" : "empty");
    }
  } finally {
    detector.dispose();
  }
});

test("cascade can force and merge extracted-game-asset verification for every slot", async () => {
  const capture = manifest.screenshots.find(
    (candidate) => candidate.fixtureFile === "16-08-37.png",
  );
  assert.ok(capture);
  const detector = await createInventoryCascadeDetector();
  try {
    await assert.rejects(
      detector.detect(join(fixtureRoot, capture.fixtureFile), { cascadeMargin: Number.NaN }),
      /cascadeMargin must be a finite number/,
    );
    await assert.rejects(
      detector.detect(join(fixtureRoot, capture.fixtureFile), {
        verifyVisionMatches: "yes" as unknown as boolean,
      }),
      /verifyVisionMatches must be boolean/,
    );
    const result = await detector.detect(join(fixtureRoot, capture.fixtureFile), {
      cascadeMargin: 1,
    });
    assert.equal(result.cascade.fallbackSlots, 24);
    assert.equal(result.cascade.templateCheckedSlots, 24);
    assert.equal(result.cascade.assetVerifiedSlots, 0);
    assert.deepEqual(
      result.slots.map((slot) => slot.item?.name ?? null),
      capture.slots.map((slot) => slot.item?.name ?? null),
    );
    assert.ok(result.matchedItems.every((slot) => slot.item?.classifier === "template"));
    assert.ok(result.slots.every((slot) => slot.cascade.backend === "template"));
  } finally {
    detector.dispose();
  }
});

test("cascade reports every fallback decision and supports a vision-only fast path", async () => {
  const capture = manifest.screenshots.find(
    (candidate) => candidate.fixtureFile === "16-08-37.png",
  );
  assert.ok(capture);
  const input = join(fixtureRoot, capture.fixtureFile);
  const detector = await createInventoryCascadeDetector();
  try {
    const visionOnly = await detector.detect(input, {
      cascadeConfidence: 0,
      cascadeMargin: 0,
      itemThreshold: 0,
      verifyVisionMatches: false,
    });
    assert.equal(visionOnly.cascade.fallbackSlots, 0);
    assert.equal(visionOnly.cascade.templateCheckedSlots, 0);
    assert.equal(visionOnly.cascade.assetVerificationEnabled, false);
    assert.ok(visionOnly.slots.every((slot) => slot.cascade.backend === "vision-features"));

    const missing = await detector.detect(input, {
      slots: [{ x: 10_000, y: 10_000, width: 32, height: 32 }],
      cascadeConfidence: 0,
      cascadeMargin: 0,
    });
    const missingSlot = missing.slots[0];
    assert.ok(missingSlot);
    assert.equal(missingSlot.cascade.fallbackReason, "missing-classification");
    assert.equal(missingSlot.item, null);

    const jpeg = await sharp(input).jpeg({ quality: 80 }).toBuffer();
    const rejected = await detector.detect(jpeg, {
      cascadeConfidence: 0,
      cascadeMargin: 0,
      itemThreshold: 1,
    });
    assert.ok(rejected.slots.some((slot) => slot.cascade.fallbackReason === "vision-rejected"));

    const lowConfidence = await detector.detect(jpeg, {
      cascadeConfidence: 1,
      cascadeMargin: 0,
      itemThreshold: 0,
    });
    assert.ok(lowConfidence.slots.some((slot) => slot.cascade.fallbackReason === "low-confidence"));
  } finally {
    detector.dispose();
  }

  await assert.rejects(detector.detect(input), /has been disposed/);
  detector.dispose();
});

test("calibration and game-asset fallback correct every unseen visual alias", async () => {
  const capture = manifest.screenshots.find((candidate) => candidate.fixtureFile === "image.png");
  assert.ok(capture);
  const model = JSON.parse(
    await readFile(join(root, "assets", "vision", "model.json"), "utf8"),
  ) as VisionModel;
  const references = model.references.filter(
    (reference) => reference.sourceScreenshot !== "image.png",
  );
  const labels = [...new Set(references.map((reference) => reference.label))];
  const holdoutModel: VisionModel = {
    ...model,
    generatedAt: new Date().toISOString(),
    sourceScreenshots: model.sourceScreenshots.filter((source) => source !== "image.png"),
    labels,
    references,
    acceptanceThresholds: calibrateVisionAcceptanceThresholds(references),
  };
  const temporaryRoot = await mkdtemp(join(tmpdir(), "sephiria-cascade-test-"));
  const modelPath = join(temporaryRoot, "holdout-model.json");
  await writeFile(modelPath, JSON.stringify(holdoutModel), "utf8");

  const detector = await createInventoryCascadeDetector({ modelPath });
  try {
    const result = await detector.detect(join(fixtureRoot, capture.fixtureFile), {
      cascadeConfidence: 0,
      cascadeMargin: 0,
      itemThreshold: 0,
    });
    const advance = result.slots.find((slot) => slot.row === 3 && slot.column === 1);
    assert.ok(advance);
    assert.equal(advance.classification?.bestLabel, "Golden Handbell");
    assert.equal(advance.classification.accepted, false);
    assert.equal(advance.classification.acceptanceReason, "low-confidence");
    assert.equal(advance.item?.name, "Advance");
    assert.equal(advance.cascade.backend, "template");
    assert.equal(advance.cascade.fallbackReason, "vision-rejected");
    assert.equal(advance.cascade.assetVerified, null);

    const past = result.slots.find((slot) => slot.row === 1 && slot.column === 0);
    assert.ok(past);
    assert.equal(past.classification?.bestLabel, "Future");
    assert.equal(past.classification.accepted, false);
    assert.equal(past.classification.acceptanceReason, "low-confidence");
    assert.equal(past.item?.name, "Past");
    assert.equal(past.cascade.backend, "template");
    assert.equal(past.cascade.fallbackReason, "vision-rejected");
    assert.equal(past.cascade.assetVerified, null);
  } finally {
    detector.dispose();
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});
