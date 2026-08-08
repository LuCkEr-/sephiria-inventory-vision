import assert from "node:assert/strict";
import test from "node:test";

import {
  CLASSICAL_METHODS,
  classifyMethodCell,
  nonMaximumSuppression,
  prepareMethodCell,
  trainSiameseEmbedding,
  trainTinyCnn,
} from "../src/lab.js";

function patterned(seed: number, brightness = 1): number[] {
  let state = seed >>> 0;
  const pixels = new Array<number>(32 * 32 * 3);
  for (let y = 0; y < 32; y += 1) {
    for (let x = 0; x < 32; x += 1) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const shape = (x * ((seed % 7) + 1) + y * ((seed % 5) + 2)) % 17 < 7 ? 180 : 30;
      for (let channel = 0; channel < 3; channel += 1) {
        pixels[(y * 32 + x) * 3 + channel] = Math.min(
          255,
          Math.round((shape + ((state >>> (channel * 5)) & 31)) * brightness),
        );
      }
    }
  }
  return pixels;
}

test("all classical method-lab backends recognize a brightness-shifted known pattern", () => {
  const references = [
    prepareMethodCell({ id: "alpha", label: "Alpha", pixels: patterned(11) }),
    prepareMethodCell({ id: "beta", label: "Beta", pixels: patterned(29) }),
  ];
  for (const method of CLASSICAL_METHODS) {
    const result = classifyMethodCell(
      { id: "query", label: "Alpha", pixels: patterned(11, 0.92) },
      references,
      method,
      1,
    );
    assert.equal(result[0]?.label, "Alpha", method);
  }
});

test("tiny CNN exposes classifier, embedding, and batch inference", async () => {
  const samples = Array.from({ length: 12 }, (_, index) => ({
    label: index < 6 ? "Alpha" : "Beta",
    pixels: patterned(index < 6 ? 11 : 29, 0.9 + (index % 3) * 0.05),
  }));
  const model = await trainTinyCnn(samples, { epochs: 4, batchSize: 4, learningRate: 0.01 });
  try {
    const predictions = model.predictBatch([patterned(11), patterned(29)]);
    assert.equal(predictions.length, 2);
    assert.equal(predictions[0]?.embedding.length, 16);
    assert.equal(model.embed(patterned(11)).length, 16);
    const singlePrediction = model.predict(patterned(11));
    assert.ok(model.labels.includes(singlePrediction.label));
    assert.equal(predictions[0].label, singlePrediction.label);
    assert.ok(Math.abs(predictions[0].confidence - singlePrediction.confidence) < 1e-5);
    assert.deepEqual(model.predictBatch([]), []);
    assert.throws(() => model.predict([]), /must contain 32x32 RGB pixels/);
    assert.throws(() => model.predictBatch([[]]), /must contain 32x32 RGB pixels/);
  } finally {
    model.dispose();
  }
  assert.throws(() => model.predict(patterned(11)), /has been disposed/);
  model.dispose();
});

test("Siamese training supports a single identity class", async () => {
  const model = await trainSiameseEmbedding(
    [
      { label: "Alpha", features: [1, 0] },
      { label: "Alpha", features: [0.9, 0.1] },
    ],
    { epochs: 1, seed: 7 },
  );
  try {
    assert.equal(model.embed([1, 0]).length, 24);
  } finally {
    model.dispose();
  }
});

test("Siamese tower returns normalized learned embeddings", async () => {
  const samples = [
    { label: "Alpha", features: [1, 0, 0, 0] },
    { label: "Alpha", features: [0.9, 0.1, 0, 0] },
    { label: "Beta", features: [0, 0, 1, 0] },
    { label: "Beta", features: [0, 0, 0.9, 0.1] },
  ];
  const model = await trainSiameseEmbedding(samples, { epochs: 3, learningRate: 0.01 });
  try {
    const embedding = model.embed([1, 0, 0, 0]);
    assert.equal(embedding.length, 24);
    const magnitude = Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0));
    assert.ok(Math.abs(magnitude - 1) < 1e-4);
    assert.throws(() => model.embed([1, 0]), /Expected 4 Siamese input values/);
  } finally {
    model.dispose();
  }
  assert.throws(() => model.embed([1, 0, 0, 0]), /has been disposed/);
  model.dispose();
});

test("object-detector non-maximum suppression removes duplicate boxes", () => {
  const result = nonMaximumSuppression([
    { x: 0, y: 0, width: 32, height: 32, label: "Alpha", confidence: 0.9 },
    { x: 2, y: 2, width: 32, height: 32, label: "Alpha", confidence: 0.8 },
    { x: 40, y: 0, width: 32, height: 32, label: "Beta", confidence: 0.7 },
  ]);
  assert.equal(result.length, 2);
  assert.equal(result[0]?.confidence, 0.9);
});
