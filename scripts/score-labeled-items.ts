import { join, resolve } from "node:path";

import { createInventoryDetector } from "../src/index.js";

const screenshotRoot = resolve(
  process.argv[2] ?? String.raw`C:\dev\experiments\sephiria-inventory-screenshots`,
);
const originalCases = [
  { file: "Sephiria_2026-08-07_15-20-26.png", row: 0, column: 0, name: "Foundation" },
  { file: "Sephiria_2026-08-07_14-39-32.png", row: 0, column: 0, name: "Foundation" },
  { file: "Sephiria_2026-08-07_15-05-25.png", row: 0, column: 0, name: "Foundation" },
  { file: "image2.png", row: 0, column: 0, name: "Foundation" },
  { file: "Sephiria_2026-08-07_15-35-41.png", row: 0, column: 0, name: "Golden Maple Leaf" },
  { file: "Sephiria_2026-08-07_15-40-28.png", row: 0, column: 0, name: "Golden Maple Leaf" },
  { file: "Sephiria_2026-08-07_16-00-29.png", row: 0, column: 0, name: "Golden Maple Leaf" },
  { file: "Sephiria_2026-08-07_16-08-37.png", row: 1, column: 3, name: "Golden Maple Leaf" },
];
const fixtureCases = [
  { file: "inventory-1440p-logical.png", row: 0, column: 0, name: "Foundation" },
  { file: "inventory-4k-logical.png", row: 0, column: 0, name: "Foundation" },
  { file: "inventory-dense-1440p-logical.png", row: 1, column: 3, name: "Golden Maple Leaf" },
];
const cases = process.argv[3] === "fixtures" ? fixtureCases : originalCases;

const detector = await createInventoryDetector();
try {
  for (const expected of cases) {
    const result = await detector.detect(join(screenshotRoot, expected.file), {
      itemThreshold: 0,
      alternatives: detector.catalog.items.length,
    });
    const slot = result.slots.find(
      (candidate) => candidate.row === expected.row && candidate.column === expected.column,
    );
    const match = slot?.alternatives.find((candidate) => candidate.name === expected.name);
    console.log({
      ...expected,
      match,
      rank: match && slot ? slot.alternatives.indexOf(match) + 1 : null,
      top: slot?.alternatives.slice(0, 3),
    });
  }
} finally {
  detector.dispose();
}
