import assert from "node:assert/strict";
import { test } from "node:test";
import {
  rankVisionFeatures,
  type VisionFeatureReference,
} from "@lucker-/sephiria-inventory-vision/browser";

test("rankVisionFeatures keeps the strongest reference for each label", () => {
  const references: VisionFeatureReference<"binary-star" | "good-will" | "foundation">[] = [
    { label: "binary-star", features: [1, 0] },
    { label: "binary-star", features: [0.9, 0.1] },
    { label: "good-will", features: [0.7, 0.7] },
    { label: "foundation", features: [0, 1] },
  ];

  assert.deepEqual(rankVisionFeatures([1, 0], references, 2), [
    { label: "binary-star", score: 1 },
    { label: "good-will", score: Math.SQRT1_2 },
  ]);
});

test("rankVisionFeatures is deterministic and validates its limit", () => {
  assert.deepEqual(
    rankVisionFeatures(
      [1, 0],
      [
        { label: "zebra", features: [1, 0] },
        { label: "alpha", features: [1, 0] },
      ],
      2,
    ),
    [
      { label: "alpha", score: 1 },
      { label: "zebra", score: 1 },
    ],
  );
  assert.deepEqual(rankVisionFeatures([1], [], 1), []);
  assert.deepEqual(rankVisionFeatures([1], [{ label: "item", features: [1] }], 0), []);
  assert.throws(() => rankVisionFeatures([1], [], -1), /non-negative integer/);
});
