import assert from "node:assert/strict";
import test from "node:test";

import type { Mat } from "@techstark/opencv-js";
import sharp from "sharp";

import {
  makeBorderMask,
  rgbaMatToTemplate,
  scaleTemplate,
  type LoadedTemplate,
} from "../src/matching.js";
import type { OpenCv } from "../src/opencv.js";
import { imageToRgbaMat } from "../src/runtime.js";
import type { CatalogTemplate } from "../src/types.js";

const metadata: CatalogTemplate = {
  id: "failure-test",
  name: "Failure Test",
  file: "unused.png",
  width: 1,
  height: 1,
  sourceFile: "test",
  pathId: 1,
};

class FakeMat {
  readonly cols = 1;
  readonly rows = 1;
  readonly data = new Uint8Array(4);
  deleteCalls = 0;

  delete(): void {
    this.deleteCalls += 1;
  }
}

test("template conversion releases partial OpenCV allocations on failure", () => {
  const allocations: FakeMat[] = [];
  const alpha = new FakeMat();
  let vectorDeleteCalls = 0;
  let colorCalls = 0;
  let splitCalls = 0;
  const cv = {
    Mat: class extends FakeMat {
      constructor(..._arguments: unknown[]) {
        super();
        allocations.push(this);
      }
    },
    MatVector: class {
      get(): FakeMat {
        return alpha;
      }

      delete(): void {
        vectorDeleteCalls += 1;
      }
    },
    COLOR_RGBA2RGB: 1,
    THRESH_BINARY: 2,
    cvtColor() {
      colorCalls += 1;
    },
    split() {
      splitCalls += 1;
    },
    threshold() {
      throw new Error("injected threshold failure");
    },
  } as unknown as OpenCv;

  assert.throws(
    () => rgbaMatToTemplate(cv, new FakeMat() as unknown as Mat, metadata),
    /injected threshold failure/,
  );
  assert.equal(colorCalls, 1);
  assert.equal(splitCalls, 1);
  assert.equal(vectorDeleteCalls, 1);
  assert.equal(alpha.deleteCalls, 1);
  assert.deepEqual(
    allocations.map((mat) => mat.deleteCalls),
    [1, 1],
  );
});

test("template scaling releases both owned matrices when resize fails", () => {
  const allocations: FakeMat[] = [];
  let resizeCalls = 0;
  const cv = {
    Mat: class extends FakeMat {
      constructor(..._arguments: unknown[]) {
        super();
        allocations.push(this);
      }
    },
    Size: class {
      constructor(
        readonly width: number,
        readonly height: number,
      ) {}
    },
    INTER_NEAREST: 1,
    resize() {
      resizeCalls += 1;
      if (resizeCalls === 2) throw new Error("injected resize failure");
    },
  } as unknown as OpenCv;
  const sourceRgb = new FakeMat();
  const sourceMask = new FakeMat();
  const template: LoadedTemplate = {
    metadata,
    rgb: sourceRgb as unknown as Mat,
    mask: sourceMask as unknown as Mat,
    width: 1,
    height: 1,
  };

  assert.throws(() => scaleTemplate(cv, template, 2), /injected resize failure/);
  assert.equal(resizeCalls, 2);
  assert.deepEqual(
    allocations.map((mat) => mat.deleteCalls),
    [1, 1],
  );
  assert.equal(sourceRgb.deleteCalls, 0);
  assert.equal(sourceMask.deleteCalls, 0);
});

test("border-mask construction releases every partial allocation on failure", () => {
  const allocations: FakeMat[] = [];
  const ring = new FakeMat();
  const MatConstructor = class extends FakeMat {
    static zeros(): FakeMat {
      return ring;
    }

    constructor(..._arguments: unknown[]) {
      super();
      allocations.push(this);
    }
  };
  const cv = {
    Mat: MatConstructor,
    CV_8UC1: 1,
    bitwise_and() {
      throw new Error("injected bitwise failure");
    },
  } as unknown as OpenCv;

  assert.throws(
    () => makeBorderMask(cv, new FakeMat() as unknown as Mat, 1),
    /injected bitwise failure/,
  );
  assert.equal(ring.deleteCalls, 1);
  assert.deepEqual(
    allocations.map((mat) => mat.deleteCalls),
    [1],
  );
});

test("image decoding releases its matrix if the native buffer shape is inconsistent", async () => {
  const allocations: FakeMat[] = [];
  const cv = {
    Mat: class extends FakeMat {
      override readonly data = new Uint8Array(0);

      constructor(..._arguments: unknown[]) {
        super();
        allocations.push(this);
      }
    },
    CV_8UC4: 1,
  } as unknown as OpenCv;
  const image = await sharp({
    create: { width: 1, height: 1, channels: 4, background: "#ffffffff" },
  })
    .png()
    .toBuffer();

  await assert.rejects(imageToRgbaMat(cv, image), RangeError);
  assert.deepEqual(
    allocations.map((mat) => mat.deleteCalls),
    [1],
  );
});
