import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyPreparedCell,
  cosineDescriptorSimilarity,
  extractEdgeShapeFeatures,
  extractOrbDescriptors,
  extractPerceptualHash,
  extractPixelVector,
  nonMaximumSuppression,
  orbSimilarity,
  perceptualHashSimilarity,
  prepareMethodCell,
  trainSiameseEmbedding,
  trainTinyCnn,
} from "../src/lab.js";

const blackCell = new Uint8Array(32 * 32 * 3);

test("classical feature methods define empty and malformed-input behavior", () => {
  assert.throws(
    () => prepareMethodCell({ id: "short", label: "Short", pixels: [] }),
    /does not contain 32x32 RGB pixels/,
  );
  assert.throws(() => extractPerceptualHash(blackCell, 31, 32), /Expected a 32x32 RGB cell/);
  assert.throws(() => extractPerceptualHash(blackCell, 32, 32, 3.5), /Expected a 32x32 RGB cell/);
  assert.throws(() => extractPerceptualHash([], 32, 32), /Expected 3072 pixel values/);
  const sparseCell = new Array<number>(32 * 32 * 3);
  assert.throws(() => extractPerceptualHash(sparseCell, 32, 32), /byte-valued RGB pixels/);
  const fractionalCell = new Float64Array(blackCell);
  fractionalCell[0] = 0.5;
  assert.throws(() => extractPixelVector(fractionalCell, 32, 32), /byte-valued RGB pixels/);
  assert.throws(() => extractOrbDescriptors(blackCell, 32, 32, 3, -1), /non-negative integer/);
  assert.deepEqual(extractOrbDescriptors(blackCell, 32, 32, 3, 0), []);
  assert.throws(() => cosineDescriptorSimilarity([1], [1, 2]), /equal vector lengths/);
  assert.throws(() => cosineDescriptorSimilarity([Number.NaN], [1]), /finite vectors/);
  assert.throws(
    () => prepareMethodCell({ id: "", label: "Bad", pixels: blackCell }),
    /id must be a non-empty string/,
  );
  assert.throws(
    () => prepareMethodCell({ id: "bad", label: "", pixels: blackCell }),
    /label must be a non-empty string/,
  );

  const prepared = prepareMethodCell({ id: "black", label: "Black", pixels: blackCell });
  assert.deepEqual(classifyPreparedCell(prepared, [], "vision-features"), []);
  assert.throws(
    () => classifyPreparedCell(prepared, [], "unsupported" as never),
    /Unsupported classical method/,
  );
  assert.throws(
    () => classifyPreparedCell(prepared, [], "vision-features", 0),
    /alternatives must be a positive integer/,
  );
  assert.deepEqual(extractOrbDescriptors(blackCell, 32, 32), []);
  assert.equal(orbSimilarity([], []), 0);
  assert.equal(cosineDescriptorSimilarity([], []), 0);
  assert.equal(extractEdgeShapeFeatures(blackCell, 32, 32).length, 192);

  const hash = extractPerceptualHash(blackCell, 32, 32);
  assert.equal(perceptualHashSimilarity(hash, hash), 1);
});

test("neural trainers reject invalid datasets and hyperparameters before allocating models", async () => {
  await assert.rejects(trainTinyCnn([]), /without samples/);
  await assert.rejects(
    trainTinyCnn([{ label: "Bad", pixels: [] }]),
    /must contain 32x32 RGB pixels/,
  );
  await assert.rejects(
    trainTinyCnn([{ label: "Black", pixels: blackCell }], { epochs: 0 }),
    /epochs must be a positive integer/,
  );
  await assert.rejects(
    trainTinyCnn([{ label: "Black", pixels: blackCell }], { batchSize: 0 }),
    /batchSize must be a positive integer/,
  );
  await assert.rejects(
    trainTinyCnn([{ label: "Black", pixels: blackCell }], { learningRate: Number.NaN }),
    /learningRate must be positive and finite/,
  );
  await assert.rejects(
    trainTinyCnn([{ label: "Black", pixels: blackCell }], { learningRate: 0 }),
    /learningRate must be positive and finite/,
  );
  await assert.rejects(
    trainTinyCnn([{ label: "Black", pixels: blackCell }], { seed: 1.5 }),
    /seed must be a safe integer/,
  );
  await assert.rejects(
    trainTinyCnn([{ label: "", pixels: blackCell }]),
    /label must be a non-empty string/,
  );
  const invalidPixels = new Float64Array(blackCell);
  invalidPixels[0] = 256;
  await assert.rejects(
    trainTinyCnn([{ label: "Invalid", pixels: invalidPixels }]),
    /byte-valued RGB pixels/,
  );
  await assert.rejects(trainSiameseEmbedding([]), /without sample pairs/);
  await assert.rejects(
    trainSiameseEmbedding(
      [
        { label: "A", features: [1] },
        { label: "B", features: [1, 2] },
      ],
      { seed: 1.5 },
    ),
    /seed must be a safe integer/,
  );
  await assert.rejects(
    trainSiameseEmbedding([
      { label: "A", features: [1] },
      { label: "B", features: [1, 2] },
    ]),
    /one non-zero shared size/,
  );
  await assert.rejects(
    trainSiameseEmbedding([
      { label: "A", features: [Number.NaN] },
      { label: "B", features: [1] },
    ]),
    /only finite values/,
  );
  await assert.rejects(
    trainSiameseEmbedding([
      { label: "", features: [0] },
      { label: "B", features: [1] },
    ]),
    /label must be a non-empty string/,
  );
});

test("non-maximum suppression validates thresholds and supports an explicit zero limit", () => {
  assert.throws(() => nonMaximumSuppression([], -0.1), /overlapThreshold must be between/);
  assert.throws(
    () => nonMaximumSuppression([], 0.5, 1.5),
    /maximum must be a non-negative integer/,
  );
  const box = { x: 0, y: 0, width: 1, height: 1, label: "A", confidence: 1 };
  assert.throws(
    () => nonMaximumSuppression([{ ...box, x: Number.NaN }]),
    /coordinates must be finite/,
  );
  assert.throws(
    () => nonMaximumSuppression([{ ...box, width: 0 }]),
    /dimensions must be positive and finite/,
  );
  assert.throws(
    () => nonMaximumSuppression([{ ...box, label: "" }]),
    /label must be a non-empty string/,
  );
  assert.throws(
    () => nonMaximumSuppression([{ ...box, confidence: 2 }]),
    /confidence must be between 0 and 1/,
  );
  assert.deepEqual(nonMaximumSuppression([box], 0.5, 0), []);
});
