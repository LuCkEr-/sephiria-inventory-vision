import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

import sharp from "sharp";

import { createInventoryDetector, createInventoryVisionDetector } from "../src/index.js";
import {
  DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
  calibrateVisionAcceptanceThresholds,
  countVisionLabelSources,
  effectiveVisionAcceptanceThreshold,
  evaluateVisionAcceptance,
  visionConfidence,
} from "../src/vision-calibration.js";

const projectRoot = resolve(import.meta.dirname, "..");
const screenshotRoot = resolve(
  process.argv[2] ?? String.raw`C:\dev\experiments\sephiria-inventory-screenshots`,
);
const outputPath = resolve(process.argv[3] ?? join(projectRoot, "benchmarks", "comparison.json"));
const groundTruth = JSON.parse(
  await readFile(join(projectRoot, "test", "fixtures", "real", "ground-truth.json"), "utf8"),
) as {
  screenshots: {
    sourceFile: string;
    family: string;
    grid: { x: number; y: number; width: number; height: number };
    slots: { item: { name: string } | null }[];
  }[];
};
const truthByScreenshot = new Map(
  groundTruth.screenshots.map((screenshot) => [screenshot.sourceFile, screenshot]),
);
const screenshots = (await readdir(screenshotRoot))
  .filter((name) => /\.(?:png|jpe?g|webp)$/i.test(name))
  .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));

const templateDetector = await createInventoryDetector();
const visionDetector = await createInventoryVisionDetector();
const perScreenshot: {
  screenshot: string;
  slots: number;
  agreements: number;
  templateCorrect: number;
  visionCorrect: number;
  templateMs: number;
  templateMatchMs: number;
  visionMs: number;
  visionMatchMs: number;
}[] = [];
let agreements = 0;
let comparedSlots = 0;
let templateCorrect = 0;
let visionCorrect = 0;
const robustness: {
  transformation: string;
  correctCells: number;
  cells: number;
  accuracy: number;
  failures: {
    index: number;
    expected: string | null;
    predicted: string | null;
    bestLabel: string | null;
    bestScore: number | null;
    acceptanceThreshold: number | null;
    margin: number | null;
    minimumMargin: number | null;
    acceptanceReason: string | null;
    nearestReferenceId: string | null;
    x: number;
    y: number;
  }[];
}[] = [];

try {
  for (const screenshot of screenshots) {
    const path = join(screenshotRoot, screenshot);
    const template = await templateDetector.detect(path);
    const vision = await visionDetector.detect(path);
    const truth = truthByScreenshot.get(screenshot);
    if (!truth) throw new Error(`No frozen ground truth for ${screenshot}`);
    const expected = truth.slots.map((slot) => slot.item?.name ?? null);
    let screenshotAgreements = 0;
    let screenshotTemplateCorrect = 0;
    let screenshotVisionCorrect = 0;
    for (const templateSlot of template.slots) {
      if (templateSlot.row === undefined || templateSlot.column === undefined) {
        throw new Error(`Template slot in ${screenshot} is missing grid coordinates`);
      }
      const expectedName = expected[templateSlot.row * 6 + templateSlot.column];
      const visionSlot = vision.slots.find(
        (slot) => slot.row === templateSlot.row && slot.column === templateSlot.column,
      );
      if ((templateSlot.item?.name ?? null) === (visionSlot?.item?.name ?? null))
        screenshotAgreements += 1;
      if ((templateSlot.item?.name ?? null) === expectedName) {
        screenshotTemplateCorrect += 1;
      }
      if ((visionSlot?.item?.name ?? null) === expectedName) {
        screenshotVisionCorrect += 1;
      }
    }
    agreements += screenshotAgreements;
    comparedSlots += template.slots.length;
    templateCorrect += screenshotTemplateCorrect;
    visionCorrect += screenshotVisionCorrect;
    perScreenshot.push({
      screenshot,
      slots: template.slots.length,
      agreements: screenshotAgreements,
      templateCorrect: screenshotTemplateCorrect,
      visionCorrect: screenshotVisionCorrect,
      templateMs: template.timingsMs.total,
      templateMatchMs: template.timingsMs.matchItems,
      visionMs: vision.timingsMs.total,
      visionMatchMs: vision.timingsMs.matchItems,
    });
  }

  const hardCapture = groundTruth.screenshots.find((screenshot) =>
    screenshot.sourceFile.includes("16-08-37"),
  );
  if (!hardCapture) throw new Error("Missing robustness capture");
  const hardPath = join(screenshotRoot, hardCapture.sourceFile);
  const hardImage = sharp(hardPath);
  const hardMetadata = await hardImage.metadata();
  if (!hardMetadata.width || !hardMetadata.height)
    throw new Error("Missing robustness image dimensions");
  const transformed = [
    {
      name: "brightness-85",
      buffer: await hardImage.clone().modulate({ brightness: 0.85 }).png().toBuffer(),
      scale: 1,
    },
    { name: "jpeg-80", buffer: await hardImage.clone().jpeg({ quality: 80 }).toBuffer(), scale: 1 },
    {
      name: "nearest-150",
      buffer: await hardImage
        .clone()
        .resize(Math.round(hardMetadata.width * 1.5), Math.round(hardMetadata.height * 1.5), {
          kernel: "nearest",
        })
        .png()
        .toBuffer(),
      scale: 1.5,
    },
    {
      name: "bilinear-150",
      buffer: await hardImage
        .clone()
        .resize(Math.round(hardMetadata.width * 1.5), Math.round(hardMetadata.height * 1.5), {
          kernel: "cubic",
        })
        .png()
        .toBuffer(),
      scale: 1.5,
    },
  ];
  const hardExpected = hardCapture.slots.map((slot) => slot.item?.name ?? null);
  for (const variant of transformed) {
    const result = await visionDetector.detect(variant.buffer);
    const failures = result.slots.flatMap((slot, index) => {
      const expected = hardExpected[index] ?? null;
      const predicted = slot.item?.name ?? null;
      if (predicted === expected) return [];
      return [
        {
          index,
          expected,
          predicted,
          bestLabel: slot.classification?.bestLabel ?? null,
          bestScore: slot.classification?.bestScore ?? null,
          acceptanceThreshold: slot.classification?.acceptanceThreshold ?? null,
          margin: slot.classification?.margin ?? null,
          minimumMargin: slot.classification?.minimumMargin ?? null,
          acceptanceReason: slot.classification?.acceptanceReason ?? null,
          nearestReferenceId: slot.alternatives[0]?.nearestReferenceId ?? null,
          x: slot.x,
          y: slot.y,
        },
      ];
    });
    const correctCells = 24 - failures.length;
    robustness.push({
      transformation: variant.name,
      correctCells,
      cells: 24,
      accuracy: correctCells / 24,
      failures,
    });
  }
} finally {
  templateDetector.dispose();
  visionDetector.dispose();
}

const references = visionDetector.model.references;
const baseReferences = references.filter((reference) => (reference.variant ?? "base") === "base");
const familyByScreenshot = new Map(
  groundTruth.screenshots.map((screenshot) => [screenshot.sourceFile, screenshot.family]),
);
let covered = 0;
let correct = 0;
let coveredItems = 0;
let correctItems = 0;
let unseenClassSlots = 0;
const failures = [];
for (const query of baseReferences) {
  const training = references.filter(
    (reference) => reference.sourceScreenshot !== query.sourceScreenshot,
  );
  if (!training.some((reference) => reference.label === query.label)) {
    unseenClassSlots += 1;
    continue;
  }
  covered += 1;
  if (query.label !== "__empty__") coveredItems += 1;
  const nearest = training
    .map((reference) => ({
      reference,
      score: visionConfidence(query.features, reference.features),
    }))
    .sort((left, right) => right.score - left.score)[0];
  const isCorrect = nearest?.reference.label === query.label;
  if (isCorrect) {
    correct += 1;
    if (query.label !== "__empty__") correctItems += 1;
  } else {
    failures.push({
      screenshot: query.sourceScreenshot,
      row: query.row,
      column: query.column,
      expected: query.label,
      predicted: nearest?.reference.label ?? null,
      score: nearest?.score ?? null,
    });
  }
}

let familyCoveredItems = 0;
let familyCorrectItems = 0;
for (const query of baseReferences) {
  if (query.label === "__empty__") continue;
  const queryFamily = familyByScreenshot.get(query.sourceScreenshot);
  const training = references.filter(
    (reference) => familyByScreenshot.get(reference.sourceScreenshot) !== queryFamily,
  );
  if (!training.some((reference) => reference.label === query.label)) continue;
  familyCoveredItems += 1;
  const prediction = training
    .map((reference) => ({
      label: reference.label,
      score: visionConfidence(query.features, reference.features),
    }))
    .sort((left, right) => right.score - left.score)[0];
  if (prediction?.label === query.label) familyCorrectItems += 1;
}

const occupiedReferences = baseReferences.filter((reference) => reference.label !== "__empty__");
let openSetFalseAccepts = 0;
let inModelOpenSetFalseAccepts = 0;
let holdoutOpenSetFalseAccepts = 0;
const holdoutThresholds = new Map<string, Readonly<Record<string, number>>>();
const fullSourceCounts = countVisionLabelSources(references);
const holdoutSourceCounts = new Map<string, ReadonlyMap<string, number>>();
for (const query of occupiedReferences) {
  const otherClasses = references.filter(
    (reference) => reference.label !== query.label && reference.label !== "__empty__",
  );
  const bestByLabel = new Map<string, number>();
  for (const reference of otherClasses) {
    const score = visionConfidence(query.features, reference.features);
    if (score > (bestByLabel.get(reference.label) ?? -1)) {
      bestByLabel.set(reference.label, score);
    }
  }
  const ranked = [...bestByLabel]
    .map(([label, score]) => ({ label, score }))
    .sort((left, right) => right.score - left.score);
  const bestOtherClass = ranked[0] ?? null;
  if (!bestOtherClass || bestOtherClass.score < DEFAULT_VISION_ACCEPTANCE_THRESHOLD) continue;
  const margin = bestOtherClass.score - (ranked[1]?.score ?? 0);
  openSetFalseAccepts += 1;
  const threshold = effectiveVisionAcceptanceThreshold(
    DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
    bestOtherClass.label,
    visionDetector.model.acceptanceThresholds,
  );
  const inModelAcceptance = evaluateVisionAcceptance(
    bestOtherClass.label,
    bestOtherClass.score,
    margin,
    threshold,
    fullSourceCounts.get(bestOtherClass.label) ?? 0,
  );
  if (inModelAcceptance.accepted) {
    inModelOpenSetFalseAccepts += 1;
  }
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
    bestOtherClass.label,
    thresholds,
  );
  const holdoutAcceptance = evaluateVisionAcceptance(
    bestOtherClass.label,
    bestOtherClass.score,
    margin,
    holdoutThreshold,
    sourceCounts.get(bestOtherClass.label) ?? 0,
  );
  if (holdoutAcceptance.accepted) holdoutOpenSetFalseAccepts += 1;
}

const average = (key: "templateMs" | "templateMatchMs" | "visionMs" | "visionMatchMs") =>
  perScreenshot.reduce((sum, result) => sum + result[key], 0) / perScreenshot.length;
const report = {
  generatedAt: new Date().toISOString(),
  screenshots: screenshots.length,
  comparisonPolicy: "inventory-grid pixels only; no tooltip, label, or OCR input",
  methods: {
    template: "masked OpenCV SQDIFF against 571 templates, including calibrated renders",
    vision: visionDetector.model.method,
  },
  runtime: {
    templateAverageTotalMs: average("templateMs"),
    templateAverageIdentityMs: average("templateMatchMs"),
    visionAverageTotalMs: average("visionMs"),
    visionAverageIdentityMs: average("visionMatchMs"),
    totalSpeedup: average("templateMs") / average("visionMs"),
    identitySpeedup: average("templateMatchMs") / average("visionMatchMs"),
  },
  sameSetAgreement: {
    note: "Vision references include these screenshots; use leave-one-screenshot-out below for generalization.",
    agreements,
    slots: comparedSlots,
    rate: agreements / comparedSlots,
  },
  frozenGroundTruth: {
    cells: comparedSlots,
    templateCorrect,
    templateAccuracy: templateCorrect / comparedSlots,
    visionCorrect,
    visionAccuracy: visionCorrect / comparedSlots,
  },
  leaveOneScreenshotOut: {
    note: "Every query is compared only with references from the other 11 screenshots.",
    coveredSlots: covered,
    correctSlots: correct,
    accuracy: correct / covered,
    coveredItemSlots: coveredItems,
    correctItemSlots: correctItems,
    itemAccuracy: correctItems / coveredItems,
    unseenClassSlots,
    failures,
  },
  familyHoldout: {
    note: "All screenshots from the query capture family are excluded together.",
    coveredItemSlots: familyCoveredItems,
    correctItemSlots: familyCorrectItems,
    itemAccuracy: familyCorrectItems / familyCoveredItems,
  },
  openSet: {
    note: "Each item's class is removed from matching. In-model calibration uses the deployed full model; strict holdout calibration is rebuilt without the query class.",
    queries: occupiedReferences.length,
    rawFalseAccepts: openSetFalseAccepts,
    rawFalseAcceptRate: openSetFalseAccepts / occupiedReferences.length,
    inModelCalibratedFalseAccepts: inModelOpenSetFalseAccepts,
    inModelCalibratedFalseAcceptRate: inModelOpenSetFalseAccepts / occupiedReferences.length,
    holdoutCalibratedFalseAccepts: holdoutOpenSetFalseAccepts,
    holdoutCalibratedFalseAcceptRate: holdoutOpenSetFalseAccepts / occupiedReferences.length,
  },
  robustness,
  perScreenshot,
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
