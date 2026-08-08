import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

import sharp from "sharp";

import { calibrateVisionAcceptanceThresholds } from "../src/vision-calibration.js";
import { extractVisionFeatures } from "../src/vision-features.js";
import type {
  InventoryCatalog,
  MatchAlternative,
  VisionModel,
  VisionReference,
  VisionReferenceVariant,
} from "../src/types.js";

const projectRoot = resolve(import.meta.dirname, "..");
const screenshotRoot = resolve(
  process.argv[2] ?? String.raw`C:\dev\experiments\sephiria-inventory-screenshots`,
);
const outputPath = resolve(process.argv[3] ?? join(projectRoot, "assets", "vision", "model.json"));
const groundTruth = JSON.parse(
  await readFile(join(projectRoot, "test", "fixtures", "real", "ground-truth.json"), "utf8"),
) as {
  screenshots: {
    id: string;
    sourceFile: string;
    image: { width: number; height: number };
    grid: { x: number; y: number; width: number; height: number; rows: number; columns: number };
    slots: {
      row: number;
      column: number;
      item: { name: string; rotationDegrees: 0 | 90 | 180 | 270 } | null;
    }[];
  }[];
};
const catalog = JSON.parse(
  await readFile(join(projectRoot, "assets", "catalog", "catalog.json"), "utf8"),
) as InventoryCatalog;

function identity(item: { name: string; rotationDegrees: 0 | 90 | 180 | 270 }): MatchAlternative {
  const template =
    catalog.items.find(
      (candidate) =>
        candidate.name === item.name && (candidate.rotationDegrees ?? 0) === item.rotationDegrees,
    ) ?? catalog.items.find((candidate) => candidate.name === item.name);
  return {
    itemId: `vision-${item.name
      .normalize("NFKD")
      .replace(/[^a-zA-Z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .toLowerCase()}`,
    name: item.name,
    confidence: 1,
    offset: { x: 0, y: 0 },
    ...(template?.spriteName ? { spriteName: template.spriteName } : {}),
    ...(template?.displayNames ? { displayNames: template.displayNames } : {}),
    ...(template?.itemIds ? { itemIds: template.itemIds } : {}),
    ...(template?.ambiguousIdentity !== undefined
      ? { ambiguousIdentity: template.ambiguousIdentity }
      : {}),
    ...(template?.itemVariants ? { itemVariants: template.itemVariants } : {}),
    rotationDegrees: item.rotationDegrees,
    ...(template?.canonicalTemplateId ? { canonicalTemplateId: template.canonicalTemplateId } : {}),
    classifier: "vision-features",
  };
}

const references: VisionReference[] = [];
const seenAugmentedEmpty = new Set<string>();
const augmentationPolicy = [
  "base",
  "brightness-85",
  "jpeg-80",
  "source-jpeg-80",
  "nearest-150",
  "roundtrip-nearest-150",
  "roundtrip-cubic-150-y1",
  "cubic-150",
  "source-cubic-150",
  "source-cubic-150-y1",
] as const satisfies readonly VisionReferenceVariant[];

interface AugmentedScreenshot {
  input: Buffer;
  scale: number;
  offsetX?: number;
  offsetY?: number;
}

function resizeNearestLikeOpenCv(
  pixels: Uint8Array,
  width: number,
  height: number,
  channels: number,
): Buffer {
  if (channels < 3) throw new Error(`Expected RGB pixels, received ${channels} channels`);
  const output = Buffer.allocUnsafe(32 * 32 * 3);
  for (let y = 0; y < 32; y += 1) {
    const sourceY = Math.min(height - 1, Math.floor((y * height) / 32));
    for (let x = 0; x < 32; x += 1) {
      const sourceX = Math.min(width - 1, Math.floor((x * width) / 32));
      const sourceOffset = (sourceY * width + sourceX) * channels;
      const outputOffset = (y * 32 + x) * 3;
      output[outputOffset] = pixels[sourceOffset] ?? 0;
      output[outputOffset + 1] = pixels[sourceOffset + 1] ?? 0;
      output[outputOffset + 2] = pixels[sourceOffset + 2] ?? 0;
    }
  }
  return output;
}

async function augmentedScreenshots(
  input: Buffer,
  width: number,
  height: number,
  sourceJpeg80: Buffer,
  sourceCubic150: Buffer,
): Promise<Map<VisionReferenceVariant, AugmentedScreenshot>> {
  const nearest150 = await sharp(input)
    .resize(Math.round(width * 1.5), Math.round(height * 1.5), { kernel: "nearest" })
    .png()
    .toBuffer();
  const roundtripNearest150 = await sharp(nearest150)
    .resize(width, height, { kernel: "nearest" })
    .png()
    .toBuffer();
  const cubic150 = await sharp(input)
    .resize(Math.round(width * 1.5), Math.round(height * 1.5), { kernel: "cubic" })
    .png()
    .toBuffer();
  const roundtripCubic150 = await sharp(cubic150)
    .resize(width, height, { kernel: "nearest" })
    .png()
    .toBuffer();
  return new Map<VisionReferenceVariant, AugmentedScreenshot>([
    ["base", { input, scale: 1 }],
    [
      "brightness-85",
      { input: await sharp(input).modulate({ brightness: 0.85 }).png().toBuffer(), scale: 1 },
    ],
    ["jpeg-80", { input: await sharp(input).jpeg({ quality: 80 }).toBuffer(), scale: 1 }],
    ["source-jpeg-80", { input: sourceJpeg80, scale: 1 }],
    [
      "nearest-150",
      {
        input: nearest150,
        scale: 1.5,
      },
    ],
    [
      "roundtrip-nearest-150",
      {
        input: roundtripNearest150,
        scale: 1,
      },
    ],
    ["roundtrip-cubic-150-y1", { input: roundtripCubic150, scale: 1, offsetY: 1 }],
    [
      "cubic-150",
      {
        input: cubic150,
        scale: 1.5,
      },
    ],
    ["source-cubic-150", { input: sourceCubic150, scale: 1 }],
    ["source-cubic-150-y1", { input: sourceCubic150, scale: 1, offsetY: 1 }],
  ]);
}

for (const capture of groundTruth.screenshots) {
  const sourcePath = resolve(screenshotRoot, capture.sourceFile);
  const normalizedHeight = 288;
  const normalizedWidth = Math.round(
    (capture.image.width / capture.image.height) * normalizedHeight,
  );
  const normalizedScreenshot = await sharp(sourcePath)
    .resize({ width: normalizedWidth, height: normalizedHeight, kernel: "nearest" })
    .png()
    .toBuffer();
  const normalizedSourceJpeg80 = await sharp(
    await sharp(sourcePath).jpeg({ quality: 80 }).toBuffer(),
  )
    .resize({ width: normalizedWidth, height: normalizedHeight, kernel: "nearest" })
    .png()
    .toBuffer();
  const sourceCubic150 = await sharp(sourcePath)
    .resize(Math.round(capture.image.width * 1.5), Math.round(capture.image.height * 1.5), {
      kernel: "cubic",
    })
    .png()
    .toBuffer();
  const normalizedSourceCubic150 = await sharp(sourceCubic150)
    .resize({ width: normalizedWidth, height: normalizedHeight, kernel: "nearest" })
    .png()
    .toBuffer();
  const screenshots = await augmentedScreenshots(
    normalizedScreenshot,
    normalizedWidth,
    normalizedHeight,
    normalizedSourceJpeg80,
    normalizedSourceCubic150,
  );
  const scaleX = normalizedWidth / capture.image.width;
  const scaleY = normalizedHeight / capture.image.height;
  const gridX = Math.round(capture.grid.x * scaleX);
  const gridY = Math.round(capture.grid.y * scaleY);
  const slotWidth = Math.round((capture.grid.width / capture.grid.columns) * scaleX);
  const slotHeight = Math.round((capture.grid.height / capture.grid.rows) * scaleY);
  for (const slot of capture.slots) {
    for (const variant of augmentationPolicy) {
      if (!slot.item && variant !== "base") {
        const emptyKey = `${capture.sourceFile}:${variant}`;
        if (seenAugmentedEmpty.has(emptyKey)) continue;
        seenAugmentedEmpty.add(emptyKey);
      }
      const screenshot = screenshots.get(variant);
      if (!screenshot) throw new Error(`Missing generated ${variant} screenshot`);
      const scaledSlotWidth = Math.round(slotWidth * screenshot.scale);
      const scaledSlotHeight = Math.round(slotHeight * screenshot.scale);
      const { data: extractedPixels, info } = await sharp(screenshot.input)
        .extract({
          left:
            Math.round(gridX * screenshot.scale) +
            slot.column * scaledSlotWidth +
            (screenshot.offsetX ?? 0),
          top:
            Math.round(gridY * screenshot.scale) +
            slot.row * scaledSlotHeight +
            (screenshot.offsetY ?? 0),
          width: scaledSlotWidth,
          height: scaledSlotHeight,
        })
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      const variantPixels = resizeNearestLikeOpenCv(
        extractedPixels,
        info.width,
        info.height,
        info.channels,
      );
      references.push({
        id: `${capture.id}:r${slot.row}:c${slot.column}:${variant}`,
        sourceScreenshot: capture.sourceFile,
        row: slot.row,
        column: slot.column,
        label: slot.item?.name ?? "__empty__",
        item: slot.item ? identity(slot.item) : null,
        features: extractVisionFeatures(variantPixels, 32, 32, 3).map((value) =>
          Number(value.toFixed(6)),
        ),
        variant,
      });
    }
  }
}

const model: VisionModel = {
  schemaVersion: 1,
  method:
    "HOG + spatial chromaticity + normalized luminance + centered RGB signature, augmented 1-nearest-neighbor",
  inputPolicy:
    "Nearest-neighbor 288px logical-height normalization; 32x32 inventory-grid cells only; no OCR or text regions",
  slotSize: 32,
  generatedAt: new Date().toISOString(),
  sourceScreenshots: groundTruth.screenshots.map((screenshot) => screenshot.sourceFile),
  labels: [...new Set(references.map((reference) => reference.label))],
  references,
  augmentationPolicy,
  acceptanceThresholds: calibrateVisionAcceptanceThresholds(references),
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(model)}\n`, "utf8");
console.log(
  `Wrote ${references.length} references (${augmentationPolicy.length} variants/cell) across ${model.labels.length - 1} item classes to ${outputPath}`,
);
