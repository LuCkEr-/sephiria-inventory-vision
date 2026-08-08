import type { Mat } from "@techstark/opencv-js";

import { collectLocalMaxima, finiteScore } from "./grid-locator.js";
import type { SlotCandidate } from "./grid-locator.js";
import type { OpenCv } from "./opencv.js";
import type { CatalogTemplate, MatchAlternative, Rect, SlotRect } from "./types.js";

export { locateInventoryGrid } from "./grid-locator.js";
export type { LocatedGrid, SlotCandidate } from "./grid-locator.js";

export interface LoadedTemplate {
  metadata: CatalogTemplate;
  rgb: Mat;
  mask: Mat;
  width: number;
  height: number;
}

interface ScaledTemplate {
  rgb: Mat;
  mask: Mat;
  width: number;
  height: number;
  owned: boolean;
}

export function rgbaMatToTemplate(
  cv: OpenCv,
  rgba: Mat,
  metadata: CatalogTemplate,
): LoadedTemplate {
  const rgb = new cv.Mat();
  let mask: Mat | undefined;
  try {
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    const channels = new cv.MatVector();
    try {
      cv.split(rgba, channels);
      const alpha = channels.get(3);
      try {
        mask = new cv.Mat();
        cv.threshold(alpha, mask, 0, 255, cv.THRESH_BINARY);
      } finally {
        alpha.delete();
      }
    } finally {
      channels.delete();
    }
    return {
      metadata,
      rgb,
      mask,
      width: rgba.cols,
      height: rgba.rows,
    };
  } catch (error) {
    mask?.delete();
    rgb.delete();
    throw error;
  }
}

export function scaleTemplate(cv: OpenCv, template: LoadedTemplate, scale: number): ScaledTemplate {
  const width = Math.max(1, Math.round(template.width * scale));
  const height = Math.max(1, Math.round(template.height * scale));
  if (width === template.width && height === template.height) {
    return { rgb: template.rgb, mask: template.mask, width, height, owned: false };
  }

  const rgb = new cv.Mat();
  let mask: Mat | undefined;
  try {
    mask = new cv.Mat();
    const size = new cv.Size(width, height);
    cv.resize(template.rgb, rgb, size, 0, 0, cv.INTER_NEAREST);
    cv.resize(template.mask, mask, size, 0, 0, cv.INTER_NEAREST);
    return { rgb, mask, width, height, owned: true };
  } catch (error) {
    mask?.delete();
    rgb.delete();
    throw error;
  }
}

function deleteScaled(template: ScaledTemplate): void {
  if (!template.owned) return;
  template.rgb.delete();
  template.mask.delete();
}

function bestMatch(
  cv: OpenCv,
  image: Mat,
  template: ScaledTemplate,
): { confidence: number; x: number; y: number } | null {
  if (template.width > image.cols || template.height > image.rows) return null;
  const expectedX = Math.round((image.cols - template.width) / 2);
  const expectedY = Math.round((image.rows - template.height) / 2);
  const tolerance = Math.max(2, Math.round(Math.min(image.cols, image.rows) * 0.12));
  const searchX = Math.max(0, expectedX - tolerance);
  const searchY = Math.max(0, expectedY - tolerance);
  const searchRight = Math.min(image.cols, expectedX + template.width + tolerance + 1);
  const searchBottom = Math.min(image.rows, expectedY + template.height + tolerance + 1);
  const search = image.roi(
    new cv.Rect(searchX, searchY, searchRight - searchX, searchBottom - searchY),
  );
  let result: Mat | undefined;
  let noMask: Mat | undefined;
  try {
    result = new cv.Mat();
    noMask = new cv.Mat();
    // Squared-difference matching is much less likely than correlation to
    // mistake a uniformly dark empty slot for a dark pixel-art icon.
    cv.matchTemplate(search, template.rgb, result, cv.TM_SQDIFF_NORMED, template.mask);
    // OpenCV.js expects an empty Mat in the optional mask position.
    const extrema = cv.minMaxLoc(result, noMask);
    return {
      confidence: finiteScore(1 - extrema.minVal),
      x: searchX + extrema.minLoc.x,
      y: searchY + extrema.minLoc.y,
    };
  } finally {
    search.delete();
    noMask?.delete();
    result?.delete();
  }
}

/** @internal */
export function makeBorderMask(cv: OpenCv, alphaMask: Mat, borderWidth: number): Mat {
  const width = alphaMask.cols;
  const height = alphaMask.rows;
  const border = Math.max(1, Math.min(borderWidth, Math.floor(Math.min(width, height) / 2)));
  const ring = cv.Mat.zeros(height, width, cv.CV_8UC1);
  let combined: Mat | undefined;
  try {
    const data = ring.data;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if (x < border || y < border || x >= width - border || y >= height - border) {
          data[y * width + x] = 255;
        }
      }
    }
    combined = new cv.Mat();
    cv.bitwise_and(alphaMask, ring, combined);
    return combined;
  } catch (error) {
    combined?.delete();
    throw error;
  } finally {
    ring.delete();
  }
}

function intersectionOverUnion(left: Rect, right: Rect): number {
  const x1 = Math.max(left.x, right.x);
  const y1 = Math.max(left.y, right.y);
  const x2 = Math.min(left.x + left.width, right.x + right.width);
  const y2 = Math.min(left.y + left.height, right.y + right.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (intersection === 0) return 0;
  const union = left.width * left.height + right.width * right.height - intersection;
  return intersection / union;
}

function assignGridCoordinates(slots: SlotCandidate[]): SlotCandidate[] {
  const sorted = [...slots].sort((left, right) => left.y - right.y || left.x - right.x);
  const rows: SlotCandidate[][] = [];

  for (const slot of sorted) {
    const centerY = slot.y + slot.height / 2;
    let row = rows.find((candidateRow) => {
      const meanCenter =
        candidateRow.reduce((sum, candidate) => sum + candidate.y + candidate.height / 2, 0) /
        candidateRow.length;
      const tolerance = Math.max(slot.height, candidateRow[0]?.height ?? slot.height) * 0.4;
      return Math.abs(meanCenter - centerY) <= tolerance;
    });
    if (!row) {
      row = [];
      rows.push(row);
    }
    row.push(slot);
  }

  rows.sort((left, right) => (left[0]?.y ?? 0) - (right[0]?.y ?? 0));
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    if (!row) continue;
    row.sort((left, right) => left.x - right.x);
    row.forEach((slot, column) => {
      slot.row = rowIndex;
      slot.column = column;
    });
  }

  return rows.flat();
}

export function locateSlots(
  cv: OpenCv,
  imageRgb: Mat,
  templates: readonly LoadedTemplate[],
  options: {
    scales: number[];
    threshold: number;
    borderWidth: number;
    maxSlots: number;
  },
): SlotCandidate[] {
  const candidates: SlotCandidate[] = [];

  for (const template of templates) {
    for (const scale of options.scales) {
      if (!Number.isFinite(scale) || scale <= 0) continue;
      const scaled = scaleTemplate(cv, template, scale);
      if (scaled.width > imageRgb.cols || scaled.height > imageRgb.rows) {
        deleteScaled(scaled);
        continue;
      }

      let borderMask: Mat | undefined;
      let result: Mat | undefined;
      try {
        borderMask = makeBorderMask(cv, scaled.mask, Math.round(options.borderWidth * scale));
        result = new cv.Mat();
        cv.matchTemplate(imageRgb, scaled.rgb, result, cv.TM_CCORR_NORMED, borderMask);
        const maxima = collectLocalMaxima(
          result,
          options.threshold,
          scaled.width,
          scaled.height,
          scale,
          template.metadata.id,
        )
          .sort((left, right) => right.localizationConfidence - left.localizationConfidence)
          .slice(0, Math.max(100, options.maxSlots * 5));
        for (const candidate of maxima) candidates.push(candidate);
      } finally {
        result?.delete();
        borderMask?.delete();
        deleteScaled(scaled);
      }
    }
  }

  const selected: SlotCandidate[] = [];
  for (const candidate of candidates.sort(
    (left, right) => right.localizationConfidence - left.localizationConfidence,
  )) {
    if (selected.some((slot) => intersectionOverUnion(slot, candidate) >= 0.45)) continue;
    selected.push(candidate);
    if (selected.length >= options.maxSlots) break;
  }

  return assignGridCoordinates(selected);
}

function toMatchAlternative(
  template: LoadedTemplate,
  match: { confidence: number; x: number; y: number },
): MatchAlternative {
  const metadata = template.metadata;
  return {
    itemId: metadata.id,
    name: metadata.name,
    confidence: match.confidence,
    offset: { x: match.x, y: match.y },
    classifier: "template",
    ...(metadata.spriteName !== undefined ? { spriteName: metadata.spriteName } : {}),
    ...(metadata.displayNames !== undefined ? { displayNames: metadata.displayNames } : {}),
    ...(metadata.itemIds !== undefined ? { itemIds: metadata.itemIds } : {}),
    ...(metadata.ambiguousIdentity !== undefined
      ? { ambiguousIdentity: metadata.ambiguousIdentity }
      : {}),
    ...(metadata.itemVariants !== undefined ? { itemVariants: metadata.itemVariants } : {}),
    ...(metadata.rotationDegrees !== undefined
      ? { rotationDegrees: metadata.rotationDegrees }
      : {}),
    ...(metadata.canonicalTemplateId !== undefined
      ? { canonicalTemplateId: metadata.canonicalTemplateId }
      : {}),
  };
}

export function matchItemsInSlot(
  cv: OpenCv,
  imageRgb: Mat,
  slot: SlotRect,
  templates: readonly LoadedTemplate[],
  nativeSlotSize: number,
  alternativeCount: number,
): MatchAlternative[] {
  const x = Math.max(0, Math.round(slot.x));
  const y = Math.max(0, Math.round(slot.y));
  const width = Math.min(Math.round(slot.width), imageRgb.cols - x);
  const height = Math.min(Math.round(slot.height), imageRgb.rows - y);
  if (width <= 0 || height <= 0) return [];

  const crop = imageRgb.roi(new cv.Rect(x, y, width, height));
  const scale = slot.scale ?? width / nativeSlotSize;
  const matches: MatchAlternative[] = [];

  try {
    for (const template of templates) {
      const scaled = scaleTemplate(cv, template, scale);
      try {
        const match = bestMatch(cv, crop, scaled);
        if (!match) continue;
        matches.push(toMatchAlternative(template, match));
      } finally {
        deleteScaled(scaled);
      }
    }
  } finally {
    crop.delete();
  }

  const bestByName = new Map<string, MatchAlternative>();
  for (const match of matches) {
    const previous = bestByName.get(match.name);
    if (!previous || match.confidence > previous.confidence) bestByName.set(match.name, match);
  }

  return [...bestByName.values()]
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, alternativeCount);
}

export function disposeTemplate(template: LoadedTemplate): void {
  template.rgb.delete();
  template.mask.delete();
}
