import { resolve } from "node:path";

import { createInventoryVisionDetector } from "@lucker-/sephiria-inventory-vision";

const screenshot = process.argv[2];
if (!screenshot) {
  console.error("Usage: npm run example -- <screenshot.png>");
  process.exit(1);
}

const detector = await createInventoryVisionDetector();
try {
  const result = await detector.detect(resolve(screenshot), {
    itemThreshold: 0.94,
    alternatives: 3,
  });
  console.log(JSON.stringify(result, null, 2));
} finally {
  detector.dispose();
}
