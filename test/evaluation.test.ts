import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import test from "node:test";

import sharp from "sharp";

import {
  createInventoryDetector,
  createInventoryVisionDetector,
  type VisionReference,
} from "../src/index.js";
import {
  DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
  calibrateVisionAcceptanceThresholds,
  countVisionLabelSources,
  effectiveVisionAcceptanceThreshold,
  evaluateVisionAcceptance,
  visionConfidence,
} from "../src/vision-calibration.js";

const projectRoot = resolve(import.meta.dirname, "..");
const fixtureRoot = join(projectRoot, "test", "fixtures", "real", "all");
const manifest = JSON.parse(
  await readFile(join(projectRoot, "test", "fixtures", "real", "ground-truth.json"), "utf8"),
) as {
  screenshots: {
    sourceFile: string;
    fixtureFile: string;
    family: string;
    image: { width: number; height: number };
    grid: { x: number; y: number; width: number; height: number };
    slots: {
      row: number;
      column: number;
      item: { name: string; rotationDegrees: number } | null;
    }[];
  }[];
};

const expectedNames = (screenshot: (typeof manifest.screenshots)[number]) =>
  screenshot.slots.map((slot) => slot.item?.name ?? null);

test("both backends are measured against frozen ground truth, not each other", async () => {
  const template = await createInventoryDetector();
  const vision = await createInventoryVisionDetector();
  try {
    for (const screenshot of manifest.screenshots) {
      const input = join(fixtureRoot, screenshot.fixtureFile);
      const expected = expectedNames(screenshot);
      const templateResult = await template.detect(input);
      const visionResult = await vision.detect(input);
      assert.deepEqual(
        templateResult.slots.map((slot) => slot.item?.name ?? null),
        expected,
        `template ${screenshot.fixtureFile}`,
      );
      assert.deepEqual(
        visionResult.slots.map((slot) => slot.item?.name ?? null),
        expected,
        `vision ${screenshot.fixtureFile}`,
      );
    }
  } finally {
    vision.dispose();
    template.dispose();
  }
});

function nearest(
  query: VisionReference,
  training: readonly VisionReference[],
): { label: string; score: number } | null {
  return rankLabels(query, training)[0] ?? null;
}

function rankLabels(
  query: VisionReference,
  training: readonly VisionReference[],
): { label: string; score: number }[] {
  const bestByLabel = new Map<string, number>();
  for (const reference of training) {
    const score = visionConfidence(query.features, reference.features);
    if (score > (bestByLabel.get(reference.label) ?? -1)) {
      bestByLabel.set(reference.label, score);
    }
  }
  return [...bestByLabel]
    .map(([label, score]) => ({ label, score }))
    .sort((a, b) => b.score - a.score);
}

test("leave-one-screenshot-out vision score remains above the reviewed floor", async () => {
  const detector = await createInventoryVisionDetector();
  try {
    const references = detector.model.references;
    const queries = references.filter((reference) => (reference.variant ?? "base") === "base");
    let coveredCells = 0;
    let correctCells = 0;
    let coveredItems = 0;
    let correctItems = 0;
    let unseenClassSlots = 0;
    for (const query of queries) {
      const training = references.filter(
        (reference) => reference.sourceScreenshot !== query.sourceScreenshot,
      );
      if (!training.some((reference) => reference.label === query.label)) {
        unseenClassSlots += 1;
        continue;
      }
      const prediction = nearest(query, training);
      coveredCells += 1;
      if (prediction?.label === query.label) correctCells += 1;
      if (query.label !== "__empty__") {
        coveredItems += 1;
        if (prediction?.label === query.label) correctItems += 1;
      }
    }
    assert.equal(unseenClassSlots, 18);
    assert.equal(coveredCells, 270);
    assert.ok(correctCells / coveredCells >= 0.985);
    assert.equal(coveredItems, 93);
    assert.ok(correctItems / coveredItems >= 0.96);
  } finally {
    detector.dispose();
  }
});

test("family holdout prevents adjacent near-duplicate captures from leaking", async () => {
  const detector = await createInventoryVisionDetector();
  try {
    const familyByScreenshot = new Map(
      manifest.screenshots.map((screenshot) => [screenshot.sourceFile, screenshot.family]),
    );
    const references = detector.model.references;
    const queries = references.filter((reference) => (reference.variant ?? "base") === "base");
    let coveredItems = 0;
    let correctItems = 0;
    for (const query of queries) {
      if (query.label === "__empty__") continue;
      const queryFamily = familyByScreenshot.get(query.sourceScreenshot);
      const training = references.filter(
        (reference) => familyByScreenshot.get(reference.sourceScreenshot) !== queryFamily,
      );
      if (!training.some((reference) => reference.label === query.label)) continue;
      coveredItems += 1;
      if (nearest(query, training)?.label === query.label) correctItems += 1;
    }
    assert.equal(coveredItems, 30);
    assert.ok(correctItems / coveredItems >= 0.9);
  } finally {
    detector.dispose();
  }
});

test("removing every non-grid pixel does not change icon detections", async () => {
  const screenshot = manifest.screenshots.find(
    (candidate) => candidate.fixtureFile === "16-08-37.png",
  );
  assert.ok(screenshot);
  const input = join(fixtureRoot, screenshot.fixtureFile);
  const metadata = await sharp(input).metadata();
  assert.ok(metadata.width && metadata.height);
  const scaleX = metadata.width / screenshot.image.width;
  const scaleY = metadata.height / screenshot.image.height;
  const logicalGrid = {
    left: Math.round(screenshot.grid.x * scaleX),
    top: Math.round(screenshot.grid.y * scaleY),
    width: Math.round(screenshot.grid.width * scaleX),
    height: Math.round(screenshot.grid.height * scaleY),
  };
  const gridPixels = await sharp(input).extract(logicalGrid).png().toBuffer();
  const isolated = await sharp({
    create: {
      width: metadata.width,
      height: metadata.height,
      channels: 4,
      background: "#000000ff",
    },
  })
    .composite([{ input: gridPixels, left: logicalGrid.left, top: logicalGrid.top }])
    .png()
    .toBuffer();

  const detector = await createInventoryVisionDetector();
  try {
    const result = await detector.detect(isolated);
    assert.deepEqual(
      result.slots.map((slot) => slot.item?.name ?? null),
      expectedNames(screenshot),
    );
  } finally {
    detector.dispose();
  }
});

test("rotation identity survives screenshot holdout", async () => {
  const detector = await createInventoryVisionDetector();
  try {
    const binaryStars = detector.model.references.filter(
      (reference) => reference.label === "Binary Star" && (reference.variant ?? "base") === "base",
    );
    assert.equal(binaryStars.length, 5);
    for (const query of binaryStars) {
      const training = detector.model.references.filter(
        (reference) => reference.sourceScreenshot !== query.sourceScreenshot,
      );
      assert.equal(nearest(query, training)?.label, "Binary Star");
      assert.equal(query.item?.rotationDegrees, 90);
    }
  } finally {
    detector.dispose();
  }
});

test("vision transformation robustness is measured explicitly", async (context) => {
  const screenshot = manifest.screenshots.find(
    (candidate) => candidate.fixtureFile === "16-08-37.png",
  );
  assert.ok(screenshot);
  const input = join(fixtureRoot, screenshot.fixtureFile);
  const metadata = await sharp(input).metadata();
  assert.ok(metadata.width && metadata.height);
  const expected = expectedNames(screenshot);
  const original = sharp(input);
  const variants = [
    {
      name: "brightness-85",
      buffer: await original.clone().modulate({ brightness: 0.85 }).png().toBuffer(),
    },
    {
      name: "jpeg-80",
      buffer: await original.clone().jpeg({ quality: 80 }).toBuffer(),
    },
    {
      name: "nearest-150",
      buffer: await original
        .clone()
        .resize(Math.round(metadata.width * 1.5), Math.round(metadata.height * 1.5), {
          kernel: "nearest",
        })
        .png()
        .toBuffer(),
    },
    {
      name: "bilinear-150",
      buffer: await original
        .clone()
        .resize(Math.round(metadata.width * 1.5), Math.round(metadata.height * 1.5), {
          kernel: "cubic",
        })
        .png()
        .toBuffer(),
    },
  ];
  const simulatedSource = await original
    .clone()
    .resize(Math.round(metadata.width * 5), Math.round(metadata.height * 5), {
      kernel: "nearest",
    })
    .png()
    .toBuffer();
  const sourceCubic150 = await sharp(simulatedSource)
    .resize(Math.round(metadata.width * 7.5), Math.round(metadata.height * 7.5), {
      kernel: "cubic",
    })
    .png()
    .toBuffer();

  const detector = await createInventoryVisionDetector();
  try {
    const sourcesByLabel = new Map<string, Set<string>>();
    for (const reference of detector.model.references) {
      const sources = sourcesByLabel.get(reference.label) ?? new Set<string>();
      sources.add(reference.sourceScreenshot);
      sourcesByLabel.set(reference.label, sources);
    }
    const riskySingletons: string[] = [];
    for (const variant of variants) {
      const result = await detector.detect(variant.buffer);
      const actual = result.slots.map((slot) => slot.item?.name ?? null);
      for (const [index, slot] of result.slots.entries()) {
        const label = expected[index];
        if (!label || sourcesByLabel.get(label)?.size !== 1 || !slot.classification) continue;
        if (slot.classification.bestScore < 0.99 && slot.classification.margin < 0.04) {
          riskySingletons.push(
            `${variant.name}:${label}:${slot.classification.bestScore.toFixed(6)}:${slot.classification.margin.toFixed(6)}`,
          );
        }
      }
      const correct = actual.filter((name, index) => name === expected[index]).length;
      context.diagnostic(`${variant.name}: ${correct}/24`);
      assert.equal(correct, 24, `${variant.name}: ${correct}/24`);
    }

    const sourceOrdered = await detector.detect(sourceCubic150);
    for (const [index, slot] of sourceOrdered.slots.entries()) {
      const label = expected[index];
      if (!label || sourcesByLabel.get(label)?.size !== 1 || !slot.classification) continue;
      if (slot.classification.bestScore < 0.99 && slot.classification.margin < 0.04) {
        riskySingletons.push(
          `source-cubic-150:${label}:${slot.classification.bestScore.toFixed(6)}:${slot.classification.margin.toFixed(6)}`,
        );
      }
    }
    const sourceOrderedCorrect = sourceOrdered.slots.filter(
      (slot, index) => (slot.item?.name ?? null) === expected[index],
    ).length;
    context.diagnostic(`source-cubic-150: ${sourceOrderedCorrect}/24`);
    for (const [index, slot] of sourceOrdered.slots.entries()) {
      if ((slot.item?.name ?? null) === expected[index]) continue;
      context.diagnostic(
        `source-cubic-150 miss ${index}: expected ${expected[index]}, best ${slot.classification?.bestLabel ?? "none"}, score ${slot.classification?.bestScore ?? 0}, threshold ${slot.classification?.acceptanceThreshold ?? 0}`,
      );
    }
    assert.deepEqual(riskySingletons, []);
    assert.equal(sourceOrderedCorrect, 24);
  } finally {
    detector.dispose();
  }
});

test("class-aware calibration reports both in-model and leakage-free open-set behavior", async (context) => {
  const detector = await createInventoryVisionDetector();
  try {
    const occupied = detector.model.references.filter(
      (reference) => reference.label !== "__empty__" && (reference.variant ?? "base") === "base",
    );
    let uncalibratedFalseAccepts = 0;
    let inModelFalseAccepts = 0;
    let holdoutFalseAccepts = 0;
    const holdoutThresholds = new Map<string, Readonly<Record<string, number>>>();
    const fullSourceCounts = countVisionLabelSources(detector.model.references);
    const holdoutSourceCounts = new Map<string, ReadonlyMap<string, number>>();
    for (const query of occupied) {
      const otherClasses = detector.model.references.filter(
        (reference) => reference.label !== query.label && reference.label !== "__empty__",
      );
      const ranked = rankLabels(query, otherClasses);
      const prediction = ranked[0];
      if (!prediction || prediction.score < DEFAULT_VISION_ACCEPTANCE_THRESHOLD) continue;
      const margin = prediction.score - (ranked[1]?.score ?? 0);
      uncalibratedFalseAccepts += 1;
      const inModelThreshold = effectiveVisionAcceptanceThreshold(
        DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
        prediction.label,
        detector.model.acceptanceThresholds,
      );
      const inModelAcceptance = evaluateVisionAcceptance(
        prediction.label,
        prediction.score,
        margin,
        inModelThreshold,
        fullSourceCounts.get(prediction.label) ?? 0,
      );
      if (inModelAcceptance.accepted) inModelFalseAccepts += 1;
      let thresholds = holdoutThresholds.get(query.label);
      let sourceCounts = holdoutSourceCounts.get(query.label);
      if (!thresholds) {
        thresholds = calibrateVisionAcceptanceThresholds(otherClasses);
        holdoutThresholds.set(query.label, thresholds);
      }
      if (!sourceCounts) {
        sourceCounts = countVisionLabelSources(otherClasses);
        holdoutSourceCounts.set(query.label, sourceCounts);
      }
      const holdoutThreshold = effectiveVisionAcceptanceThreshold(
        DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
        prediction.label,
        thresholds,
      );
      const holdoutAcceptance = evaluateVisionAcceptance(
        prediction.label,
        prediction.score,
        margin,
        holdoutThreshold,
        sourceCounts.get(prediction.label) ?? 0,
      );
      if (holdoutAcceptance.accepted) holdoutFalseAccepts += 1;
    }
    context.diagnostic(
      `open-set false accepts: raw ${uncalibratedFalseAccepts}/${occupied.length}; in-model calibrated ${inModelFalseAccepts}/${occupied.length}; leakage-free class holdout ${holdoutFalseAccepts}/${occupied.length}`,
    );
    assert.equal(uncalibratedFalseAccepts, 19);
    assert.equal(inModelFalseAccepts, 0);
    assert.equal(holdoutFalseAccepts, 0);
  } finally {
    detector.dispose();
  }
});
