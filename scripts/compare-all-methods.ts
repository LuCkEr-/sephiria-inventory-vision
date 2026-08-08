import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import sharp from "sharp";

import { createInventoryDetector } from "../src/index.js";
import {
  CLASSICAL_METHODS,
  classifyPreparedCell,
  cosineDescriptorSimilarity,
  nonMaximumSuppression,
  prepareMethodCell,
  trainSiameseEmbedding,
  trainTinyCnn,
  type ClassicalMethod,
  type PreparedMethodCell,
  type ScoredBox,
} from "../src/lab.js";

interface TruthItem {
  name: string;
  rotationDegrees: number;
}

interface TruthCapture {
  id: string;
  sourceFile: string;
  fixtureFile: string;
  family: string;
  image: { width: number; height: number };
  grid: { x: number; y: number; width: number; height: number; rows: number; columns: number };
  slots: { row: number; column: number; item: TruthItem | null }[];
}

interface Cell extends PreparedMethodCell {
  screenshot: string;
  family: string;
  row: number;
  column: number;
}

interface AccuracyResult {
  queries: number;
  correct: number;
  accuracy: number;
  coveredQueries: number;
  coveredCorrect: number;
  coveredAccuracy: number;
  occupiedCovered: number;
  occupiedCorrect: number;
  occupiedAccuracy: number;
  milliseconds: number;
  millisecondsPerQuery: number;
  failures: { id: string; expected: string; predicted: string | null; score: number | null }[];
}

const root = resolve(import.meta.dirname, "..");
const fixtureRoot = join(root, "test", "fixtures", "real", "all");
const reportPath = resolve(process.argv[2] ?? join(root, "benchmarks", "all-methods.json"));
const markdownPath = reportPath.replace(/\.json$/i, ".md");
const manifest = JSON.parse(
  await readFile(join(root, "test", "fixtures", "real", "ground-truth.json"), "utf8"),
) as { policy: string; screenshots: TruthCapture[] };

async function extractCells(capture: TruthCapture): Promise<Cell[]> {
  const path = join(fixtureRoot, capture.fixtureFile);
  const metadata = await sharp(path).metadata();
  if (!metadata.width || !metadata.height)
    throw new Error(`Missing dimensions for ${capture.fixtureFile}`);
  const scaleX = metadata.width / capture.image.width;
  const scaleY = metadata.height / capture.image.height;
  const gridX = Math.round(capture.grid.x * scaleX);
  const gridY = Math.round(capture.grid.y * scaleY);
  const slotWidth = Math.round((capture.grid.width / capture.grid.columns) * scaleX);
  const slotHeight = Math.round((capture.grid.height / capture.grid.rows) * scaleY);
  const source = sharp(path);
  const cells: Cell[] = [];
  for (const slot of capture.slots) {
    const { data, info } = await source
      .clone()
      .extract({
        left: gridX + slot.column * slotWidth,
        top: gridY + slot.row * slotHeight,
        width: slotWidth,
        height: slotHeight,
      })
      .resize(32, 32, { kernel: "nearest" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    if (info.channels !== 3) throw new Error(`Unexpected ${info.channels}-channel fixture cell`);
    const id = `${capture.fixtureFile}:r${slot.row}:c${slot.column}`;
    cells.push({
      ...prepareMethodCell({ id, label: slot.item?.name ?? "__empty__", pixels: data }),
      screenshot: capture.fixtureFile,
      family: capture.family,
      row: slot.row,
      column: slot.column,
    });
  }
  return cells;
}

console.log(`Preparing ${manifest.screenshots.length} reviewed screenshots...`);
const cells = (await Promise.all(manifest.screenshots.map(extractCells))).flat();
const challengeCapture = manifest.screenshots.find((capture) =>
  capture.sourceFile.includes("16-08-37"),
);
if (!challengeCapture) throw new Error("Missing the 16-08-37 challenge capture");
const challenge = cells.filter((cell) => cell.screenshot === challengeCapture.fixtureFile);
const training = cells.filter((cell) => cell.screenshot !== challengeCapture.fixtureFile);

function summarizePredictions(
  queries: readonly Cell[],
  trainingCells: readonly Cell[],
  predictions: readonly { label: string | null; score: number | null }[],
  milliseconds: number,
): AccuracyResult {
  const known = new Set(trainingCells.map((cell) => cell.label));
  let correct = 0;
  let coveredQueries = 0;
  let coveredCorrect = 0;
  let occupiedCovered = 0;
  let occupiedCorrect = 0;
  const failures: AccuracyResult["failures"] = [];
  queries.forEach((query, index) => {
    const prediction = predictions[index] ?? { label: null, score: null };
    const matches = prediction.label === query.label;
    if (matches) correct += 1;
    if (known.has(query.label)) {
      coveredQueries += 1;
      if (matches) coveredCorrect += 1;
      if (query.label !== "__empty__") {
        occupiedCovered += 1;
        if (matches) occupiedCorrect += 1;
      }
    }
    if (!matches && failures.length < 30) {
      failures.push({
        id: query.id,
        expected: query.label,
        predicted: prediction.label,
        score: prediction.score,
      });
    }
  });
  return {
    queries: queries.length,
    correct,
    accuracy: correct / Math.max(1, queries.length),
    coveredQueries,
    coveredCorrect,
    coveredAccuracy: coveredCorrect / Math.max(1, coveredQueries),
    occupiedCovered,
    occupiedCorrect,
    occupiedAccuracy: occupiedCorrect / Math.max(1, occupiedCovered),
    milliseconds: Number(milliseconds.toFixed(3)),
    millisecondsPerQuery: Number((milliseconds / Math.max(1, queries.length)).toFixed(4)),
    failures,
  };
}

function evaluateClassical(
  method: ClassicalMethod,
  queries: readonly Cell[],
  trainingCells: readonly Cell[],
): AccuracyResult {
  const started = performance.now();
  const predictions = queries.map((query) => {
    const best = classifyPreparedCell(query, trainingCells, method, 1)[0];
    return { label: best?.label ?? null, score: best?.score ?? null };
  });
  return summarizePredictions(queries, trainingCells, predictions, performance.now() - started);
}

console.log("Running classical leave-one-screenshot-out benchmarks...");
const classicalLeaveOneOut: Record<string, AccuracyResult> = {};
for (const method of CLASSICAL_METHODS) {
  const started = performance.now();
  const predictions: { query: Cell; label: string | null; score: number | null }[] = [];
  for (const capture of manifest.screenshots) {
    const queries = cells.filter((cell) => cell.screenshot === capture.fixtureFile);
    const references = cells.filter((cell) => cell.screenshot !== capture.fixtureFile);
    for (const query of queries) {
      const best = classifyPreparedCell(query, references, method, 1)[0];
      predictions.push({ query, label: best?.label ?? null, score: best?.score ?? null });
    }
  }
  classicalLeaveOneOut[method] = summarizePredictions(
    predictions.map((prediction) => prediction.query),
    cells,
    predictions.map(({ label, score }) => ({ label, score })),
    performance.now() - started,
  );
  // Correct coverage for leave-one-out instead of the all-cells helper input.
  let coveredQueries = 0;
  let coveredCorrect = 0;
  let occupiedCovered = 0;
  let occupiedCorrect = 0;
  for (const prediction of predictions) {
    const references = cells.filter((cell) => cell.screenshot !== prediction.query.screenshot);
    const covered = references.some((reference) => reference.label === prediction.query.label);
    if (!covered) continue;
    coveredQueries += 1;
    if (prediction.label === prediction.query.label) coveredCorrect += 1;
    if (prediction.query.label !== "__empty__") {
      occupiedCovered += 1;
      if (prediction.label === prediction.query.label) occupiedCorrect += 1;
    }
  }
  const methodResult = classicalLeaveOneOut[method];
  Object.assign(methodResult, {
    coveredQueries,
    coveredCorrect,
    coveredAccuracy: coveredCorrect / Math.max(1, coveredQueries),
    occupiedCovered,
    occupiedCorrect,
    occupiedAccuracy: occupiedCorrect / Math.max(1, occupiedCovered),
  });
  console.log(`${method}: ${occupiedCorrect}/${occupiedCovered} covered occupied cells`);
}

const classicalChallenge = Object.fromEntries(
  CLASSICAL_METHODS.map((method) => [method, evaluateClassical(method, challenge, training)]),
);

function augment(pixels: readonly number[], variant: number): number[] {
  const shiftX = (variant % 3) - 1;
  const shiftY = (Math.floor(variant / 3) % 3) - 1;
  const brightness = 0.9 + (variant % 5) * 0.05;
  const output = new Array<number>(pixels.length).fill(0);
  for (let y = 0; y < 32; y += 1) {
    for (let x = 0; x < 32; x += 1) {
      const sourceX = Math.max(0, Math.min(31, x - shiftX));
      const sourceY = Math.max(0, Math.min(31, y - shiftY));
      for (let channel = 0; channel < 3; channel += 1) {
        const source = pixels[(sourceY * 32 + sourceX) * 3 + channel] ?? 0;
        output[(y * 32 + x) * 3 + channel] = Math.max(
          0,
          Math.min(255, Math.round(source * brightness)),
        );
      }
    }
  }
  return output;
}

function balancedNeuralSamples(source: readonly Cell[]): { label: string; pixels: number[] }[] {
  const byLabel = new Map<string, Cell[]>();
  for (const cell of source) {
    const bucket = byLabel.get(cell.label) ?? [];
    bucket.push(cell);
    byLabel.set(cell.label, bucket);
  }
  const samples: { label: string; pixels: number[] }[] = [];
  for (const [label, bucket] of byLabel) {
    const target = label === "__empty__" ? 24 : 10;
    for (let index = 0; index < target; index += 1) {
      const source = bucket[index % bucket.length];
      if (!source) throw new Error(`Missing source cell for ${label}`);
      samples.push({ label, pixels: augment(source.pixels, index) });
    }
  }
  return samples;
}

console.log("Training the tiny CNN classifier and embedding tower...");
const cnnTrainingStarted = performance.now();
const cnn = await trainTinyCnn(balancedNeuralSamples(training), {
  epochs: 8,
  batchSize: 32,
  learningRate: 0.004,
});
const cnnTrainingMs = performance.now() - cnnTrainingStarted;
const cnnStarted = performance.now();
const cnnPredictions = cnn.predictBatch(challenge.map((cell) => cell.pixels));
const cnnChallenge = summarizePredictions(
  challenge,
  training,
  cnnPredictions.map((prediction) => ({ label: prediction.label, score: prediction.confidence })),
  performance.now() - cnnStarted,
);

const trainingEmbeddings = training.map((cell) => ({ cell, embedding: cnn.embed(cell.pixels) }));
const embeddingStarted = performance.now();
const embeddingPredictions = challenge.map((query) => {
  const embedding = cnn.embed(query.pixels);
  let best: { label: string; score: number } | null = null;
  for (const reference of trainingEmbeddings) {
    const score = cosineDescriptorSimilarity(embedding, reference.embedding);
    if (!best || score > best.score) best = { label: reference.cell.label, score };
  }
  return best ?? { label: null, score: null };
});
const neuralEmbeddingChallenge = summarizePredictions(
  challenge,
  training,
  embeddingPredictions,
  performance.now() - embeddingStarted,
);

console.log("Training the contrastive Siamese embedding...");
const siameseTrainingStarted = performance.now();
const siamese = await trainSiameseEmbedding(
  training.map((cell) => ({ label: cell.label, features: cell.vision })),
  { epochs: 12, learningRate: 0.003 },
);
const siameseTrainingMs = performance.now() - siameseTrainingStarted;
const siameseReferences = training.map((cell) => ({ cell, embedding: siamese.embed(cell.vision) }));
const siameseStarted = performance.now();
const siamesePredictions = challenge.map((query) => {
  const embedding = siamese.embed(query.vision);
  let best: { label: string; score: number } | null = null;
  for (const reference of siameseReferences) {
    const score = cosineDescriptorSimilarity(embedding, reference.embedding);
    if (!best || score > best.score) best = { label: reference.cell.label, score };
  }
  return best ?? { label: null, score: null };
});
const siameseChallenge = summarizePredictions(
  challenge,
  training,
  siamesePredictions,
  performance.now() - siameseStarted,
);

console.log("Running the extracted-game-asset OpenCV baseline...");
const templateDetector = await createInventoryDetector();
let templateResult;
try {
  templateResult = await templateDetector.detect(join(fixtureRoot, challengeCapture.fixtureFile));
} finally {
  templateDetector.dispose();
}
const templateChallenge = summarizePredictions(
  challenge,
  training,
  templateResult.slots.map((slot) => ({
    label: slot.item?.name ?? "__empty__",
    score: slot.item?.confidence ?? 1,
  })),
  templateResult.timingsMs.matchItems,
);

async function normalizedGrid(
  capture: TruthCapture,
): Promise<{ pixels: Buffer; width: number; height: number }> {
  const path = join(fixtureRoot, capture.fixtureFile);
  const metadata = await sharp(path).metadata();
  if (!metadata.width || !metadata.height) throw new Error("Challenge fixture has no dimensions");
  const scaleX = metadata.width / capture.image.width;
  const scaleY = metadata.height / capture.image.height;
  const width = capture.grid.columns * 32;
  const height = capture.grid.rows * 32;
  const result = await sharp(path)
    .extract({
      left: Math.round(capture.grid.x * scaleX),
      top: Math.round(capture.grid.y * scaleY),
      width: Math.round(capture.grid.width * scaleX),
      height: Math.round(capture.grid.height * scaleY),
    })
    .resize(width, height, { kernel: "nearest" })
    .removeAlpha()
    .raw()
    .toBuffer();
  return { pixels: result, width, height };
}

function cropRaw(pixels: Buffer, imageWidth: number, x: number, y: number): number[] {
  const result = new Array<number>(32 * 32 * 3);
  for (let row = 0; row < 32; row += 1) {
    for (let column = 0; column < 32; column += 1) {
      for (let channel = 0; channel < 3; channel += 1) {
        result[(row * 32 + column) * 3 + channel] =
          pixels[((y + row) * imageWidth + x + column) * 3 + channel] ?? 0;
      }
    }
  }
  return result;
}

console.log("Running dense-window object detection inside the inventory ROI...");
const grid = await normalizedGrid(challengeCapture);
const windows: { x: number; y: number; pixels: number[] }[] = [];
for (let y = 0; y <= grid.height - 32; y += 8) {
  for (let x = 0; x <= grid.width - 32; x += 8)
    windows.push({ x, y, pixels: cropRaw(grid.pixels, grid.width, x, y) });
}
const objectStarted = performance.now();
const windowPredictions = cnn.predictBatch(windows.map((window) => window.pixels));
const candidateBoxes: ScoredBox[] = windows.flatMap((window, index) => {
  const prediction = windowPredictions[index];
  if (!prediction) throw new Error(`Missing CNN prediction for window ${index}`);
  return prediction.label === "__empty__" || prediction.confidence < 0.35
    ? []
    : [
        {
          x: window.x,
          y: window.y,
          width: 32,
          height: 32,
          label: prediction.label,
          confidence: prediction.confidence,
        },
      ];
});
const detections = nonMaximumSuppression(candidateBoxes, 0.35, 48);
const objectMs = performance.now() - objectStarted;
const truthBoxes = challenge
  .filter((cell) => cell.label !== "__empty__")
  .map((cell) => ({
    x: cell.column * 32,
    y: cell.row * 32,
    width: 32,
    height: 32,
    label: cell.label,
  }));
const iou = (left: { x: number; y: number; width: number; height: number }, right: typeof left) => {
  const x1 = Math.max(left.x, right.x);
  const y1 = Math.max(left.y, right.y);
  const x2 = Math.min(left.x + left.width, right.x + right.width);
  const y2 = Math.min(left.y + left.height, right.y + right.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return (
    intersection / Math.max(1, left.width * left.height + right.width * right.height - intersection)
  );
};
let localized = 0;
let localizedAndClassified = 0;
for (const truth of truthBoxes) {
  const matching = detections.filter((detection) => iou(truth, detection) >= 0.5);
  if (matching.length > 0) localized += 1;
  if (matching.some((detection) => detection.label === truth.label)) localizedAndClassified += 1;
}

cnn.dispose();
siamese.dispose();

const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  policy: manifest.policy,
  corpus: {
    screenshots: manifest.screenshots.length,
    cells: cells.length,
    occupiedCells: cells.filter((cell) => cell.label !== "__empty__").length,
    itemClasses: new Set(
      cells.filter((cell) => cell.label !== "__empty__").map((cell) => cell.label),
    ).size,
  },
  evaluation: {
    challengeScreenshot: challengeCapture.sourceFile,
    trainingScreenshots: manifest.screenshots.length - 1,
    note: "Challenge results exclude that screenshot from reference/training data. Covered accuracy omits item classes never seen in training; all-cell accuracy retains them as unavoidable misses.",
  },
  methods: {
    "opencv-game-asset-template": {
      kind: "masked pixel template matching against extracted game sprites",
      challenge: templateChallenge,
      runtimeIncludes: "identity matching; grid localization excluded",
    },
    ...Object.fromEntries(
      CLASSICAL_METHODS.map((method) => [
        method,
        {
          kind:
            method === "vision-features"
              ? "HOG + spatial color + normalized luminance + centered RGB nearest neighbor"
              : method === "perceptual-hash"
                ? "64-bit DCT perceptual hash nearest neighbor"
                : method === "edge-shape"
                  ? "Sobel contour + orientation descriptor nearest neighbor"
                  : method === "orb"
                    ? "oriented FAST/BRIEF local-feature nearest neighbor"
                    : "pHash eight-label shortlist followed by normalized pixel verification",
          challenge: classicalChallenge[method],
          leaveOneScreenshotOut: classicalLeaveOneOut[method],
        },
      ]),
    ),
    "tiny-cnn-classifier": {
      kind: "two-convolution closed-set classifier trained only on the other screenshots",
      trainingMilliseconds: Number(cnnTrainingMs.toFixed(3)),
      challenge: cnnChallenge,
    },
    "cnn-embedding": {
      kind: "nearest-neighbor retrieval using the CNN penultimate layer",
      sharedTrainingMilliseconds: Number(cnnTrainingMs.toFixed(3)),
      challenge: neuralEmbeddingChallenge,
    },
    "siamese-embedding": {
      kind: "shared-weight twin MLP trained with contrastive loss over icon descriptors",
      trainingMilliseconds: Number(siameseTrainingMs.toFixed(3)),
      challenge: siameseChallenge,
    },
    "dense-window-object-detector": {
      kind: "8-pixel-stride CNN scan plus non-maximum suppression inside the inventory grid ROI",
      tooltipOrLabelPixels: false,
      windows: windows.length,
      candidateBoxes: candidateBoxes.length,
      detections: detections.length,
      truthObjects: truthBoxes.length,
      localized,
      localizationRecall: localized / Math.max(1, truthBoxes.length),
      localizedAndClassified,
      identityRecall: localizedAndClassified / Math.max(1, truthBoxes.length),
      milliseconds: Number(objectMs.toFixed(3)),
      detectionsDetail: detections,
    },
  },
  interpretation: {
    limits: [
      "Only 111 occupied cells and 40 identities are labeled; deep-learning results are feasibility measurements, not production estimates.",
      "Adjacent screenshots can be visually similar, so leave-one-screenshot-out is less strict than a new gameplay-session holdout.",
      "Object detection is restricted to the inventory grid ROI to preserve the no-tooltip/no-label policy.",
      "A class absent from training cannot be recognized by the learned closed-set methods.",
    ],
  },
};

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const rows = Object.entries(report.methods).flatMap(([name, method]) => {
  if (!("challenge" in method)) return [];
  return `| ${name} | ${method.challenge.occupiedCorrect}/${method.challenge.occupiedCovered} | ${percent(method.challenge.occupiedAccuracy)} | ${method.challenge.millisecondsPerQuery.toFixed(3)} ms |`;
});
const markdown =
  `# Sephiria inventory method comparison\n\n` +
  `All methods use inventory-grid icon pixels only. Tooltip and label pixels are excluded.\n\n` +
  `Challenge capture: **${challengeCapture.sourceFile}**. It is excluded from training/reference data.\n\n` +
  `| Method | Correct covered items | Covered item accuracy | Identity time / cell |\n` +
  `|---|---:|---:|---:|\n${rows.join("\n")}\n\n` +
  `The dense-window detector localized ${localized}/${truthBoxes.length} occupied cells and localized plus classified ${localizedAndClassified}/${truthBoxes.length}.\n\n` +
  `Deep methods are feasibility results on a small corpus. The JSON report contains failures, all-cell scores, training times, and leave-one-screenshot-out classical scores.\n`;

await mkdir(resolve(reportPath, ".."), { recursive: true });
await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
await writeFile(markdownPath, markdown, "utf8");
console.log(`Wrote ${reportPath}`);
