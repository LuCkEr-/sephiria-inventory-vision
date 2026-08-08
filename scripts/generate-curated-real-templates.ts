import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const screenshotRoot = resolve(
  process.argv[2] ?? String.raw`C:\dev\experiments\sephiria-inventory-screenshots`,
);
const extractedRoot = resolve(
  process.argv[3] ??
    String.raw`C:\Program Files (x86)\Steam\steamapps\common\Sephiria\Extracted_Image_Assets`,
);
const outputRoot = join(projectRoot, "assets", "curated-items");

interface Calibration {
  name: string;
  key?: string;
  screenshot: string;
  origin: [number, number];
  slot: [number, number];
  offset?: [number, number];
  flipX?: boolean;
  rotationDegrees?: 0 | 90 | 180 | 270;
}

interface ItemIconMapping {
  itemId: number;
  displayName: string;
  iconSourceFile: string;
  iconPathId: number;
  spriteName: string;
  spritePath: string;
  width: number;
  height: number;
}

function slugify(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
}

const calibrations = JSON.parse(
  await readFile(join(outputRoot, "manifest.json"), "utf8"),
) as Calibration[];
const itemMappings = JSON.parse(
  await readFile(join(projectRoot, "assets", "item-icon-map.json"), "utf8"),
) as ItemIconMapping[];
const generatedTemplates = [];

await mkdir(outputRoot, { recursive: true });

for (const calibration of calibrations) {
  const mapping = itemMappings.find((candidate) => candidate.displayName === calibration.name);
  if (!mapping) throw new Error(`No extracted item icon mapping found for ${calibration.name}`);

  const screenshot = await sharp(join(screenshotRoot, calibration.screenshot))
    .resize({ height: 288, kernel: "nearest" })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const mask = await sharp(join(extractedRoot, mapping.spritePath))
    .rotate(calibration.rotationDegrees ?? 0)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const output = Buffer.alloc(mask.info.width * mask.info.height * 4);
  const [row, column] = calibration.slot;
  const [originX, originY] = calibration.origin;
  const [offsetX, offsetY] = calibration.offset ?? [
    Math.floor((32 - mask.info.width) / 2),
    Math.floor((32 - mask.info.height) / 2),
  ];

  for (let y = 0; y < mask.info.height; y += 1) {
    for (let x = 0; x < mask.info.width; x += 1) {
      const outputIndex = (y * mask.info.width + x) * 4;
      const screenshotX = originX + column * 32 + offsetX + x;
      const screenshotY = originY + row * 32 + offsetY + y;
      const screenshotIndex = (screenshotY * screenshot.info.width + screenshotX) * 4;
      output[outputIndex] = screenshot.data[screenshotIndex] ?? 0;
      output[outputIndex + 1] = screenshot.data[screenshotIndex + 1] ?? 0;
      output[outputIndex + 2] = screenshot.data[screenshotIndex + 2] ?? 0;
      const maskX = calibration.flipX ? mask.info.width - 1 - x : x;
      const maskIndex = (y * mask.info.width + maskX) * 4;
      output[outputIndex + 3] = mask.data[maskIndex + 3] ?? 0;
    }
  }

  const key = slugify(calibration.key ?? calibration.name);
  const filename = `${key}-rendered.png`;
  await sharp(output, {
    raw: { width: mask.info.width, height: mask.info.height, channels: 4 },
  })
    .png()
    .toFile(join(outputRoot, filename));

  generatedTemplates.push({
    id: `curated-${key}`,
    name: calibration.name,
    file: filename,
    width: mask.info.width,
    height: mask.info.height,
    sourceFile: mapping.iconSourceFile,
    pathId: mapping.iconPathId,
    spriteName: mapping.spriteName,
    displayNames: [calibration.name],
    itemIds: [mapping.itemId],
    ambiguousIdentity: false,
    rotationDegrees: calibration.rotationDegrees ?? 0,
  });
}

await writeFile(
  join(outputRoot, "generated-templates.json"),
  `${JSON.stringify(generatedTemplates, null, 2)}\n`,
  "utf8",
);

console.log(
  `Generated ${generatedTemplates.length} icon-only calibrated templates in ${outputRoot}`,
);
