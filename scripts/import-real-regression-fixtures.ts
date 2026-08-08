import { mkdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const screenshotRoot = resolve(
  process.argv[2] ?? String.raw`C:\dev\experiments\sephiria-inventory-screenshots`,
);
const outputRoot = join(projectRoot, "test", "fixtures", "real", "all");

const fixtures = [
  ["Sephiria_2026-08-07_14-39-32.png", "14-39-32.png"],
  ["Sephiria_2026-08-07_15-05-25.png", "15-05-25.png"],
  ["Sephiria_2026-08-07_15-15-04.png", "15-15-04.png"],
  ["Sephiria_2026-08-07_15-16-10.png", "15-16-10.png"],
  ["Sephiria_2026-08-07_15-20-26.png", "15-20-26.png"],
  ["Sephiria_2026-08-07_15-35-41.png", "15-35-41.png"],
  ["Sephiria_2026-08-07_15-40-28.png", "15-40-28.png"],
  ["Sephiria_2026-08-07_15-48-31.png", "15-48-31.png"],
  ["Sephiria_2026-08-07_16-00-29.png", "16-00-29.png"],
  ["Sephiria_2026-08-07_16-08-37.png", "16-08-37.png"],
  ["image.png", "image.png"],
  ["image2.png", "image2.png"],
] as const;

await mkdir(outputRoot, { recursive: true });
for (const [source, output] of fixtures) {
  await sharp(join(screenshotRoot, source))
    .resize({ height: 288, kernel: "nearest" })
    .png()
    .toFile(join(outputRoot, output));
}

console.log(`Imported ${fixtures.length} real regression fixtures into ${outputRoot}`);
