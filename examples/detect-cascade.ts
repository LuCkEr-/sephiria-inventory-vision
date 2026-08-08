import { resolve } from "node:path";

import { createInventoryCascadeDetector } from "../src/index.js";

const input = process.argv[2];
if (!input) throw new Error("Usage: npm run example:cascade -- <screenshot.png>");

const detector = await createInventoryCascadeDetector();
try {
  const result = await detector.detect(resolve(input));
  console.log(
    JSON.stringify(
      {
        timingsMs: result.timingsMs,
        cascade: result.cascade,
        slots: result.slots.map((slot) => ({
          row: slot.row,
          column: slot.column,
          item: slot.item?.name ?? null,
          confidence: slot.item?.confidence ?? null,
          rotationDegrees: slot.item?.rotationDegrees ?? 0,
          backend: slot.cascade.backend,
          fallbackReason: slot.cascade.fallbackReason,
          visionEvidence: slot.classification ?? null,
        })),
      },
      null,
      2,
    ),
  );
} finally {
  detector.dispose();
}
