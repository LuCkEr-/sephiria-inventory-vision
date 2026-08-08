import { readdir } from "node:fs/promises";
import { join, resolve } from "node:path";

import { createInventoryDetector } from "../src/index.js";

const screenshotDirectory = resolve(
  process.argv[2] ?? String.raw`C:\dev\experiments\sephiria-inventory-screenshots`,
);
const itemThreshold = process.argv[3] === undefined ? undefined : Number(process.argv[3]);
const filenameFilter = process.argv[4]?.toLowerCase();
const names = (await readdir(screenshotDirectory))
  .filter((name) => /\.(?:png|jpe?g|webp|bmp)$/i.test(name))
  .filter((name) => !filenameFilter || name.toLowerCase().includes(filenameFilter))
  .sort();

const detector = await createInventoryDetector();
try {
  for (const name of names) {
    const result = await detector.detect(
      join(screenshotDirectory, name),
      itemThreshold === undefined ? {} : { itemThreshold },
    );
    console.log(
      JSON.stringify({
        file: name,
        size: result.image,
        slots: result.slots.length,
        gridOrigin: result.slots[0] ? { x: result.slots[0].x, y: result.slots[0].y } : null,
        slotSize: result.slots[0]?.width ?? null,
        matched: result.matchedItems.map((slot) => ({
          row: slot.row,
          column: slot.column,
          slotConfidence: slot.localizationConfidence,
          item: slot.item?.name,
          confidence: slot.item?.confidence,
          ...(itemThreshold === undefined ? {} : { alternatives: slot.alternatives }),
        })),
        timingsMs: result.timingsMs,
      }),
    );
  }
} finally {
  detector.dispose();
}
