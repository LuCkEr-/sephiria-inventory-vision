import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { createInventoryCascadeDetector, type VisionModel } from "../src/index.js";
import { calibrateVisionAcceptanceThresholds } from "../src/vision-calibration.js";

interface GroundTruthCapture {
  sourceFile: string;
  fixtureFile: string;
  family: string;
  slots: {
    row: number;
    column: number;
    item: { name: string } | null;
  }[];
}

const root = resolve(import.meta.dirname, "..");
const fixtureRoot = join(root, "test", "fixtures", "real", "all");
const outputPath = resolve(
  process.argv[2] ?? join(root, "benchmarks", "cascade-cross-validation.json"),
);
const manifest = JSON.parse(
  await readFile(join(root, "test", "fixtures", "real", "ground-truth.json"), "utf8"),
) as { policy: string; screenshots: GroundTruthCapture[] };
const fullModel = JSON.parse(
  await readFile(join(root, "assets", "vision", "model.json"), "utf8"),
) as VisionModel;
const temporaryRoot = await mkdtemp(join(tmpdir(), "sephiria-cascade-cv-"));

try {
  const captures = [];
  for (const capture of manifest.screenshots) {
    const references = fullModel.references.filter(
      (reference) => reference.sourceScreenshot !== capture.sourceFile,
    );
    const labels = [...new Set(references.map((reference) => reference.label))];
    const model: VisionModel = {
      ...fullModel,
      generatedAt: new Date().toISOString(),
      sourceScreenshots: fullModel.sourceScreenshots.filter(
        (source) => source !== capture.sourceFile,
      ),
      labels,
      references,
      acceptanceThresholds: calibrateVisionAcceptanceThresholds(references),
    };
    const modelPath = join(temporaryRoot, `${capture.fixtureFile}.json`);
    await writeFile(modelPath, JSON.stringify(model), "utf8");

    const detector = await createInventoryCascadeDetector({ modelPath });
    try {
      const result = await detector.detect(join(fixtureRoot, capture.fixtureFile));
      const expected = capture.slots.map((slot) => slot.item?.name ?? null);
      const actual = result.slots.map((slot) => slot.item?.name ?? null);
      const failures = capture.slots.flatMap((slot, index) =>
        actual[index] === expected[index]
          ? []
          : [
              {
                row: slot.row,
                column: slot.column,
                expected: expected[index],
                actual: actual[index],
                classSeenInTraining: slot.item ? labels.includes(slot.item.name) : true,
                backend: result.slots[index]?.cascade.backend ?? null,
                fallbackReason: result.slots[index]?.cascade.fallbackReason ?? null,
                visionBest: result.slots[index]?.classification?.bestLabel ?? null,
                visionScore: result.slots[index]?.classification?.bestScore ?? null,
                visionMargin: result.slots[index]?.classification?.margin ?? null,
                nearestReferenceScreenshot:
                  result.slots[index]?.alternatives[0]?.nearestReferenceScreenshot ?? null,
              },
            ],
      );
      const occupied = capture.slots.filter((slot) => slot.item !== null).length;
      const correctItems = capture.slots.filter(
        (slot, index) => slot.item !== null && actual[index] === expected[index],
      ).length;
      captures.push({
        screenshot: capture.sourceFile,
        fixture: capture.fixtureFile,
        family: capture.family,
        trainingReferences: references.length,
        trainingLabels: labels.length - 1,
        cells: capture.slots.length,
        correctCells: capture.slots.length - failures.length,
        occupied,
        correctItems,
        fallbackSlots: result.cascade.fallbackSlots,
        visionSlots: result.cascade.visionSlots,
        templateCheckedSlots: result.cascade.templateCheckedSlots,
        assetVerifiedSlots: result.cascade.assetVerifiedSlots,
        totalMs: result.timingsMs.total,
        failures,
      });
      console.log(
        `${capture.fixtureFile}: ${capture.slots.length - failures.length}/${capture.slots.length} cells, ${correctItems}/${occupied} items, ${result.cascade.fallbackSlots} fallbacks`,
      );
    } finally {
      detector.dispose();
    }
  }

  const totals = captures.reduce(
    (sum, capture) => ({
      cells: sum.cells + capture.cells,
      correctCells: sum.correctCells + capture.correctCells,
      occupied: sum.occupied + capture.occupied,
      correctItems: sum.correctItems + capture.correctItems,
      fallbackSlots: sum.fallbackSlots + capture.fallbackSlots,
      templateCheckedSlots: sum.templateCheckedSlots + capture.templateCheckedSlots,
      assetVerifiedSlots: sum.assetVerifiedSlots + capture.assetVerifiedSlots,
      totalMs: sum.totalMs + capture.totalMs,
    }),
    {
      cells: 0,
      correctCells: 0,
      occupied: 0,
      correctItems: 0,
      fallbackSlots: 0,
      templateCheckedSlots: 0,
      assetVerifiedSlots: 0,
      totalMs: 0,
    },
  );
  const report = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    policy: manifest.policy,
    holdoutPolicy:
      "For each fold, every learned reference from the evaluated screenshot is removed before end-to-end cascade detection.",
    folds: captures.length,
    totals: {
      ...totals,
      cellAccuracy: totals.correctCells / totals.cells,
      itemAccuracy: totals.correctItems / totals.occupied,
      averageMsPerScreenshot: totals.totalMs / captures.length,
    },
    captures,
  };
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(report.totals, null, 2));
  if (totals.correctCells !== totals.cells || totals.correctItems !== totals.occupied) {
    throw new Error(
      `Cascade cross-validation regressed: ${totals.correctCells}/${totals.cells} cells, ${totals.correctItems}/${totals.occupied} items`,
    );
  }
} finally {
  await rm(temporaryRoot, { recursive: true, force: true });
}
