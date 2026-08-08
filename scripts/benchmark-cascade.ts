import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import {
  createInventoryCascadeDetector,
  createInventoryDetector,
  createInventoryVisionDetector,
  type CascadeDetectionResult,
  type DetectionResult,
  type VisionModel,
} from "../src/index.js";
import { calibrateVisionAcceptanceThresholds } from "../src/vision-calibration.js";

const root = resolve(import.meta.dirname, "..");
const fixtureRoot = join(root, "test", "fixtures", "real", "all");
const outputPath = resolve(process.argv[2] ?? join(root, "benchmarks", "cascade.json"));
const manifest = JSON.parse(
  await readFile(join(root, "test", "fixtures", "real", "ground-truth.json"), "utf8"),
) as {
  policy: string;
  screenshots: {
    sourceFile: string;
    fixtureFile: string;
    slots: { row: number; column: number; item: { name: string } | null }[];
  }[];
};
const capture = manifest.screenshots.find((candidate) => candidate.sourceFile.includes("16-08-37"));
if (!capture) throw new Error("Missing challenge screenshot");
const input = join(fixtureRoot, capture.fixtureFile);
const expected = capture.slots.map((slot) => slot.item?.name ?? null);
const fullModel = JSON.parse(
  await readFile(join(root, "assets", "vision", "model.json"), "utf8"),
) as VisionModel;
const references = fullModel.references.filter(
  (reference) => reference.sourceScreenshot !== capture.sourceFile,
);
const holdoutModel: VisionModel = {
  ...fullModel,
  generatedAt: new Date().toISOString(),
  sourceScreenshots: fullModel.sourceScreenshots.filter((name) => name !== capture.sourceFile),
  labels: [...new Set(references.map((reference) => reference.label))],
  references,
  acceptanceThresholds: calibrateVisionAcceptanceThresholds(references),
};

function score(result: DetectionResult) {
  const actual = result.slots.map((slot) => slot.item?.name ?? null);
  const correct = actual.filter((name, index) => name === expected[index]).length;
  return { correct, cells: expected.length, accuracy: correct / expected.length, actual };
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted[middle] ?? 0;
}

function summarizeTimings(results: readonly DetectionResult[]) {
  return {
    runs: results.length,
    medianTotalMs: median(results.map((result) => result.timingsMs.total)),
    medianIdentityMs: median(results.map((result) => result.timingsMs.matchItems)),
    totalRunsMs: results.map((result) => result.timingsMs.total),
  };
}

const temporaryRoot = await mkdtemp(join(tmpdir(), "sephiria-cascade-"));
const modelPath = join(temporaryRoot, "holdout-model.json");
await writeFile(modelPath, JSON.stringify(holdoutModel), "utf8");

try {
  const vision = await createInventoryVisionDetector({ modelPath });
  const template = await createInventoryDetector();
  const cascade = await createInventoryCascadeDetector({ modelPath });
  try {
    const visionRuns: DetectionResult[] = [];
    const templateRuns: DetectionResult[] = [];
    const cascadeRuns: CascadeDetectionResult[] = [];
    for (let run = 0; run < 3; run += 1) {
      visionRuns.push(await vision.detect(input));
      templateRuns.push(await template.detect(input));
      cascadeRuns.push(await cascade.detect(input));
    }
    const visionResult = visionRuns.at(-1);
    const templateResult = templateRuns.at(-1);
    const cascadeResult = cascadeRuns.at(-1);
    if (!visionResult || !templateResult || !cascadeResult) {
      throw new Error("Cascade benchmark produced no results");
    }
    const fallbacks = cascadeResult.slots.flatMap((slot) =>
      slot.cascade.backend === "template"
        ? [
            {
              row: slot.row,
              column: slot.column,
              expected: expected[(slot.row ?? 0) * 6 + (slot.column ?? 0)],
              result: slot.item?.name ?? null,
              reason: slot.cascade.fallbackReason,
              visionBest: slot.classification?.bestLabel ?? null,
              visionScore: slot.classification?.bestScore ?? null,
              visionMargin: slot.classification?.margin ?? null,
            },
          ]
        : [],
    );
    const report = {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      policy: manifest.policy,
      challengeScreenshot: capture.sourceFile,
      holdoutPolicy: "Every reference from the challenge screenshot is removed before detection.",
      cascadePolicy: {
        primary:
          "HOG + spatial color + normalized luminance + centered RGB signature with deterministic augmentation",
        fallback: "masked OpenCV matching against extracted game assets",
        acceptedMatchVerification: "predicted game asset; full catalog on mismatch",
        confidenceThreshold: cascadeResult.cascade.confidenceThreshold,
        marginThreshold: cascadeResult.cascade.marginThreshold,
      },
      visionOnly: {
        ...score(visionResult),
        timing: summarizeTimings(visionRuns),
      },
      templateOnly: {
        ...score(templateResult),
        timing: summarizeTimings(templateRuns),
      },
      cascade: {
        ...score(cascadeResult),
        timing: summarizeTimings(cascadeRuns),
        fallbackSlots: cascadeResult.cascade.fallbackSlots,
        visionSlots: cascadeResult.cascade.visionSlots,
        templateCheckedSlots: cascadeResult.cascade.templateCheckedSlots,
        assetVerifiedSlots: cascadeResult.cascade.assetVerifiedSlots,
        fallbacks,
      },
    };
    await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    console.log(JSON.stringify(report, null, 2));
    if (report.cascade.correct !== expected.length) {
      throw new Error(`Cascade accuracy regressed to ${report.cascade.correct}/${expected.length}`);
    }
    if (report.templateOnly.correct !== expected.length) {
      throw new Error(
        `Template accuracy regressed to ${report.templateOnly.correct}/${expected.length}`,
      );
    }
    if (report.cascade.timing.medianTotalMs >= report.templateOnly.timing.medianTotalMs) {
      throw new Error("Cascade median latency must remain below template-only latency");
    }
  } finally {
    cascade.dispose();
    template.dispose();
    vision.dispose();
  }
} finally {
  await rm(temporaryRoot, { recursive: true, force: true });
}
