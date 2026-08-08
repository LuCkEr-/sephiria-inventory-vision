import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import sharp from "sharp";

import { createInventoryVisionDetector } from "../../src/index.js";
import type { CatalogTemplate, InventoryCatalog, SlotRect } from "../../src/types.js";
import { DEFAULT_VISION_ACCEPTANCE_THRESHOLD } from "../../src/vision-calibration.js";

export interface OpenSetAcceptance {
  sourceItem: string;
  sourceTemplateId: string;
  predictedItem: string;
  score: number;
  threshold: number;
}

export interface GameAssetOpenSetReport {
  schemaVersion: 1;
  generatedAt: string;
  policy: string;
  acceptanceThreshold: number;
  queries: number;
  rejected: number;
  accepted: number;
  falseAcceptRate: number;
  acceptedExamples: OpenSetAcceptance[];
}

async function renderInventoryCell(
  catalogRoot: string,
  background: Buffer,
  item: CatalogTemplate,
): Promise<Buffer> {
  const icon = sharp(join(catalogRoot, item.file));
  const metadata = await icon.metadata();
  const scale = Math.min(1, 30 / Math.max(metadata.width, metadata.height));
  const width = Math.max(1, Math.round(metadata.width * scale));
  const height = Math.max(1, Math.round(metadata.height * scale));
  const resized = await icon.resize(width, height, { kernel: "nearest" }).png().toBuffer();
  return sharp(background)
    .composite([
      {
        input: resized,
        left: Math.floor((32 - width) / 2),
        top: Math.floor((32 - height) / 2),
      },
    ])
    .png()
    .toBuffer();
}

function createSlots(count: number, columns: number): SlotRect[] {
  return Array.from({ length: count }, (_, index) => ({
    x: (index % columns) * 32,
    y: Math.floor(index / columns) * 32,
    width: 32,
    height: 32,
  }));
}

export async function evaluateGameAssetOpenSet(
  projectRoot = resolve(import.meta.dirname, "..", ".."),
  acceptanceThreshold = DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
): Promise<GameAssetOpenSetReport> {
  const catalogRoot = join(projectRoot, "assets", "catalog");
  const catalog = JSON.parse(
    await readFile(join(catalogRoot, "catalog.json"), "utf8"),
  ) as InventoryCatalog;
  const detector = await createInventoryVisionDetector();
  try {
    const knownLabels = new Set(detector.model.labels);
    const unknownTemplates = catalog.items.filter((item) => !knownLabels.has(item.name));
    const background = await readFile(join(catalogRoot, "ui", "InventorySlot0__663.png"));
    const cells = await Promise.all(
      unknownTemplates.map((item) => renderInventoryCell(catalogRoot, background, item)),
    );
    const columns = 24;
    const rows = Math.ceil(cells.length / columns);
    const atlas = await sharp({
      create: {
        width: columns * 32,
        height: rows * 32,
        channels: 4,
        background: "#000000ff",
      },
    })
      .composite(
        cells.map((input, index) => ({
          input,
          left: (index % columns) * 32,
          top: Math.floor(index / columns) * 32,
        })),
      )
      .png()
      .toBuffer();
    const result = await detector.detect(atlas, {
      slots: createSlots(cells.length, columns),
      itemThreshold: acceptanceThreshold,
    });
    const acceptedExamples = result.slots.flatMap((slot, index) => {
      const source = unknownTemplates[index];
      if (!slot.item || !slot.classification || !source) return [];
      return [
        {
          sourceItem: source.name,
          sourceTemplateId: source.id,
          predictedItem: slot.item.name,
          score: slot.classification.bestScore,
          threshold: slot.classification.acceptanceThreshold,
        },
      ];
    });
    return {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      policy:
        "Every extracted game-item template absent from the vision model is rendered into an inventory slot; tooltip and label pixels are excluded.",
      acceptanceThreshold,
      queries: unknownTemplates.length,
      rejected: unknownTemplates.length - acceptedExamples.length,
      accepted: acceptedExamples.length,
      falseAcceptRate: acceptedExamples.length / unknownTemplates.length,
      acceptedExamples,
    };
  } finally {
    detector.dispose();
  }
}
