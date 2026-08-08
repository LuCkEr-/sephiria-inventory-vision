import assert from "node:assert/strict";
import { join, resolve } from "node:path";
import test from "node:test";

import type { Mat } from "@techstark/opencv-js";
import sharp from "sharp";

import {
  collectLocalMaxima,
  differenceConfidenceAt,
  finiteScore,
  scoreAt,
} from "../src/grid-locator.js";
import { disposeTemplate, locateInventoryGrid } from "../src/matching.js";
import { getOpenCv } from "../src/opencv.js";
import { imageToRgbaMat, loadTemplate } from "../src/runtime.js";
import type { CatalogTemplate } from "../src/types.js";

const root = resolve(import.meta.dirname, "..");
const catalogRoot = join(root, "assets", "catalog");
const slotFile = "ui/InventorySlot0__663.png";
const slotMetadata: CatalogTemplate = {
  id: "inventoryslot0-663",
  name: "InventorySlot0",
  file: slotFile,
  width: 32,
  height: 32,
  sourceFile: "resources.assets",
  pathId: 663,
};

test("grid score primitives clamp invalid data and search safely across image boundaries", () => {
  assert.equal(finiteScore(Number.NaN), 0);
  assert.equal(finiteScore(-1), 0);
  assert.equal(finiteScore(2), 1);
  const result = {
    cols: 2,
    rows: 2,
    data32F: new Float32Array([Number.NaN, 0.4, 0.9, 2]),
  } as Mat;
  assert.equal(scoreAt(result, -1, -1, 2), 1);
  assert.ok(Math.abs(differenceConfidenceAt(result, -1, -1, 2) - 0.6) < 1e-6);
  assert.deepEqual(collectLocalMaxima(result, 0.3, 1, 1, 1, "test"), [
    {
      x: 1,
      y: 1,
      width: 1,
      height: 1,
      scale: 1,
      localizationConfidence: 1,
      templateId: "test",
    },
  ]);
});

test("grid locator handles oversized, empty, and explicit-tolerance searches", async () => {
  const { cv } = await getOpenCv();
  const template = await loadTemplate(cv, catalogRoot, slotMetadata);
  const tiny = new cv.Mat(1, 1, Number(cv.CV_8UC3));
  const empty = cv.Mat.zeros(64, 64, Number(cv.CV_8UC3));
  let gridRgb: Mat | undefined;

  try {
    assert.equal(
      locateInventoryGrid(cv, tiny, template, {
        rows: 1,
        columns: 1,
        threshold: 0.97,
        minSupport: 1,
      }),
      null,
    );
    assert.equal(
      locateInventoryGrid(cv, empty, template, {
        rows: 2,
        columns: 2,
        threshold: 1,
        minSupport: 4,
        tolerance: 0,
      }),
      null,
    );

    const input = await sharp({
      create: { width: 64, height: 64, channels: 4, background: "#00000000" },
    })
      .composite(
        [
          [0, 0],
          [32, 0],
          [0, 32],
          [32, 32],
        ].map(([left, top]) => ({
          input: join(catalogRoot, slotFile),
          left,
          top,
        })),
      )
      .png()
      .toBuffer();
    const decoded = await imageToRgbaMat(cv, input);
    gridRgb = new cv.Mat();
    try {
      cv.cvtColor(decoded.mat, gridRgb, Number(cv.COLOR_RGBA2RGB));
    } finally {
      decoded.mat.delete();
    }

    const located = locateInventoryGrid(cv, gridRgb, template, {
      rows: 2,
      columns: 2,
      threshold: 0.97,
      minSupport: 4,
      tolerance: 0,
      preferredOriginY: 0,
    });
    assert.ok(located);
    assert.deepEqual(located.origin, { x: 0, y: 0 });
    assert.equal(located.support, 4);
    assert.equal(located.slots.length, 4);
  } finally {
    gridRgb?.delete();
    empty.delete();
    tiny.delete();
    disposeTemplate(template);
  }
});
