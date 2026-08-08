import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import vm from "node:vm";

import sharp from "sharp";

const demoText = await readFile(resolve("../sephiria-inventory-detector-demo/data.js"), "utf8");
const sandbox = { window: {} };
vm.runInNewContext(demoText, sandbox);
const data = sandbox.window.SEPHIRIA_DEMO_DATA;
const screenshotRoot = resolve(String.raw`C:\dev\experiments\sephiria-inventory-screenshots`);

function normalize(vector) {
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => value / magnitude);
}

function cosine(left, right) {
  let score = 0;
  for (let index = 0; index < left.length; index += 1) score += left[index] * right[index];
  return score;
}

function extractFeatures(data, width, height) {
  const gray = new Float32Array(width * height);
  const red = new Float32Array(width * height);
  const green = new Float32Array(width * height);
  const blue = new Float32Array(width * height);
  let grayMean = 0;
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 3;
    red[index] = data[offset] / 255;
    green[index] = data[offset + 1] / 255;
    blue[index] = data[offset + 2] / 255;
    gray[index] = red[index] * 0.299 + green[index] * 0.587 + blue[index] * 0.114;
    grayMean += gray[index];
  }
  grayMean /= gray.length;
  let variance = 0;
  for (const value of gray) variance += (value - grayMean) ** 2;
  const grayStd = Math.sqrt(variance / gray.length) || 1;

  const features = [];
  const cells = 4;
  const cellWidth = width / cells;
  const cellHeight = height / cells;
  for (let cellY = 0; cellY < cells; cellY += 1) {
    for (let cellX = 0; cellX < cells; cellX += 1) {
      let r = 0;
      let g = 0;
      let b = 0;
      let count = 0;
      for (let y = Math.floor(cellY * cellHeight); y < Math.floor((cellY + 1) * cellHeight); y += 1) {
        for (let x = Math.floor(cellX * cellWidth); x < Math.floor((cellX + 1) * cellWidth); x += 1) {
          const index = y * width + x;
          r += red[index];
          g += green[index];
          b += blue[index];
          count += 1;
        }
      }
      const sum = r + g + b || 1;
      features.push((r / sum) * 0.7, (g / sum) * 0.7, (b / sum) * 0.7, ((r + g + b) / count / 3) * 0.25);
    }
  }

  const hogCells = 4;
  const bins = 9;
  const histograms = Array.from({ length: hogCells * hogCells }, () => new Float32Array(bins));
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const gx = gray[y * width + x + 1] - gray[y * width + x - 1];
      const gy = gray[(y + 1) * width + x] - gray[(y - 1) * width + x];
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      let angle = Math.atan2(gy, gx);
      if (angle < 0) angle += Math.PI;
      if (angle >= Math.PI) angle -= Math.PI;
      const bin = Math.min(bins - 1, Math.floor(angle / Math.PI * bins));
      const cellX = Math.min(hogCells - 1, Math.floor(x / width * hogCells));
      const cellY = Math.min(hogCells - 1, Math.floor(y / height * hogCells));
      histograms[cellY * hogCells + cellX][bin] += magnitude;
    }
  }
  for (const histogram of histograms) {
    const normalized = normalize([...histogram]);
    for (const value of normalized) features.push(value * 1.5);
  }

  const downsample = 8;
  for (let cellY = 0; cellY < downsample; cellY += 1) {
    for (let cellX = 0; cellX < downsample; cellX += 1) {
      let value = 0;
      let count = 0;
      for (let y = cellY * 4; y < (cellY + 1) * 4; y += 1) {
        for (let x = cellX * 4; x < (cellX + 1) * 4; x += 1) {
          value += (gray[y * width + x] - grayMean) / grayStd;
          count += 1;
        }
      }
      features.push(value / count * 0.35);
    }
  }
  return normalize(features);
}

const samples = [];
for (const capture of data.screenshots) {
  const source = sharp(resolve(screenshotRoot, capture.filename));
  for (const slot of capture.slots) {
    const item = slot.alternatives?.[0];
    const { data: pixels, info } = await source
      .clone()
      .extract({ left: slot.x, top: slot.y, width: slot.width, height: slot.height })
      .resize(32, 32, { kernel: "nearest" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    samples.push({
      screenshot: capture.filename,
      row: slot.row,
      column: slot.column,
      label: item && item.confidence >= data.defaultThreshold ? item.name : "__empty__",
      features: extractFeatures(pixels, info.width, info.height),
    });
  }
}

const predictions = [];
for (const query of samples) {
  const training = samples.filter((sample) => sample.screenshot !== query.screenshot);
  const labelsInTraining = new Set(training.map((sample) => sample.label));
  if (!labelsInTraining.has(query.label)) {
    predictions.push({ ...query, predicted: null, score: null, covered: false, correct: false });
    continue;
  }
  const scored = training
    .map((sample) => ({ label: sample.label, score: cosine(query.features, sample.features) }))
    .sort((left, right) => right.score - left.score);
  predictions.push({
    ...query,
    predicted: scored[0]?.label ?? null,
    score: scored[0]?.score ?? null,
    covered: true,
    correct: scored[0]?.label === query.label,
  });
}

const covered = predictions.filter((prediction) => prediction.covered);
const correct = covered.filter((prediction) => prediction.correct);
const failures = covered.filter((prediction) => !prediction.correct);
console.log(JSON.stringify({
  method: "HOG + spatial color + normalized luminance, 1-NN",
  evaluation: "leave-one-screenshot-out",
  totalSlots: predictions.length,
  coveredSlots: covered.length,
  unseenClassSlots: predictions.length - covered.length,
  correct: correct.length,
  accuracyOnCovered: correct.length / covered.length,
  failures: failures.map(({ screenshot, row, column, label, predicted, score }) => ({ screenshot, row, column, label, predicted, score })),
}, null, 2));
