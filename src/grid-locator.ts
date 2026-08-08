import type { Mat } from "@techstark/opencv-js";

import type { OpenCv } from "./opencv.js";
import type { CatalogTemplate, SlotRect } from "./types.js";

interface GridTemplate {
  metadata: CatalogTemplate;
  rgb: Mat;
  mask: Mat;
  width: number;
  height: number;
}

export interface SlotCandidate extends SlotRect {
  localizationConfidence: number;
  templateId: string;
}

export interface LocatedGrid {
  slots: SlotCandidate[];
  support: number;
  confidence: number;
  origin: { x: number; y: number };
  pitch: number;
}

export interface GridLocationOptions {
  rows: number;
  columns: number;
  threshold: number;
  minSupport: number;
  tolerance?: number;
  preferredOriginY?: number;
}

interface GridEvaluation {
  originX: number;
  originY: number;
  support: number;
  supportScore: number;
  adjustedSupportScore: number;
  anchorBalance: number;
  totalScore: number;
  slotScores: number[];
}

interface GridScoreContext {
  correlation: Mat;
  difference: Mat;
  pitch: number;
  tolerance: number;
  differenceThreshold: number;
  options: GridLocationOptions;
}

/** @internal */
export function finiteScore(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
}

/** @internal */
export function collectLocalMaxima(
  result: Mat,
  threshold: number,
  width: number,
  height: number,
  scale: number,
  templateId: string,
): SlotCandidate[] {
  const candidates: SlotCandidate[] = [];
  const values = result.data32F;
  const columns = result.cols;
  const rows = result.rows;

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      const score = values[y * columns + x] ?? 0;
      if (!Number.isFinite(score) || score < threshold) continue;
      if (!isLocalMaximum(values, columns, rows, x, y, score)) continue;
      candidates.push({
        x,
        y,
        width,
        height,
        scale,
        localizationConfidence: finiteScore(score),
        templateId,
      });
    }
  }

  return candidates;
}

function isLocalMaximum(
  values: Float32Array,
  columns: number,
  rows: number,
  x: number,
  y: number,
  score: number,
): boolean {
  for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
    for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
      if (offsetX === 0 && offsetY === 0) continue;
      const neighborX = x + offsetX;
      const neighborY = y + offsetY;
      const outside = neighborX < 0 || neighborY < 0 || neighborX >= columns || neighborY >= rows;
      if (outside) continue;
      if ((values[neighborY * columns + neighborX] ?? 0) > score) return false;
    }
  }
  return true;
}

/** @internal */
export function scoreAt(result: Mat, x: number, y: number, tolerance: number): number {
  let best = 0;
  for (let offsetY = -tolerance; offsetY <= tolerance; offsetY += 1) {
    for (let offsetX = -tolerance; offsetX <= tolerance; offsetX += 1) {
      const sampleX = x + offsetX;
      const sampleY = y + offsetY;
      if (sampleX < 0 || sampleY < 0 || sampleX >= result.cols || sampleY >= result.rows) continue;
      const value = result.data32F[sampleY * result.cols + sampleX] ?? 0;
      if (Number.isFinite(value)) best = Math.max(best, value);
    }
  }
  return finiteScore(best);
}

/** @internal */
export function differenceConfidenceAt(
  result: Mat,
  x: number,
  y: number,
  tolerance: number,
): number {
  let best = 0;
  for (let offsetY = -tolerance; offsetY <= tolerance; offsetY += 1) {
    for (let offsetX = -tolerance; offsetX <= tolerance; offsetX += 1) {
      const sampleX = x + offsetX;
      const sampleY = y + offsetY;
      if (sampleX < 0 || sampleY < 0 || sampleX >= result.cols || sampleY >= result.rows) continue;
      const value = result.data32F[sampleY * result.cols + sampleX] ?? 1;
      if (Number.isFinite(value)) best = Math.max(best, 1 - value);
    }
  }
  return finiteScore(best);
}

function isNearCandidate(
  candidate: SlotCandidate,
  existing: readonly SlotCandidate[],
  pitch: number,
): boolean {
  const tolerance = Math.max(2, pitch * 0.2);
  return existing.some(
    (entry) =>
      Math.abs(entry.x - candidate.x) <= tolerance && Math.abs(entry.y - candidate.y) <= tolerance,
  );
}

function filterGridCandidates(
  differenceResult: Mat,
  rawCandidates: readonly SlotCandidate[],
  pitch: number,
  tolerance: number,
  differenceThreshold: number,
): SlotCandidate[] {
  const candidates: SlotCandidate[] = [];
  for (const candidate of rawCandidates) {
    const confidence = differenceConfidenceAt(
      differenceResult,
      candidate.x,
      candidate.y,
      tolerance,
    );
    if (confidence < differenceThreshold || isNearCandidate(candidate, candidates, pitch)) continue;
    candidates.push(candidate);
    if (candidates.length >= 200) break;
  }
  return candidates;
}

function isGridInBounds(
  originX: number,
  originY: number,
  imageRgb: Mat,
  context: GridScoreContext,
): boolean {
  return (
    originX >= 0 &&
    originY >= 0 &&
    originX + context.options.columns * context.pitch <= imageRgb.cols &&
    originY + context.options.rows * context.pitch <= imageRgb.rows
  );
}

function calculateAnchorBalance(
  support: number,
  bounds: { minRow: number; maxRow: number; minColumn: number; maxColumn: number },
  options: GridLocationOptions,
): number {
  if (support === 0) return Number.NEGATIVE_INFINITY;
  return -(
    Math.abs(bounds.minColumn - (options.columns - 1 - bounds.maxColumn)) +
    Math.abs(bounds.minRow - (options.rows - 1 - bounds.maxRow))
  );
}

function evaluateGridOrigin(
  context: GridScoreContext,
  originX: number,
  originY: number,
): GridEvaluation {
  const { correlation, difference, differenceThreshold, options, pitch, tolerance } = context;
  const slotScores: number[] = [];
  let support = 0;
  let supportScore = 0;
  let totalScore = 0;
  const bounds = {
    minRow: options.rows,
    maxRow: -1,
    minColumn: options.columns,
    maxColumn: -1,
  };
  for (let row = 0; row < options.rows; row += 1) {
    for (let column = 0; column < options.columns; column += 1) {
      const x = originX + column * pitch;
      const y = originY + row * pitch;
      const correlationScore = scoreAt(correlation, x, y, tolerance);
      const differenceScore = differenceConfidenceAt(difference, x, y, tolerance);
      const score = Math.min(correlationScore, differenceScore);
      slotScores.push(score);
      totalScore += (correlationScore + differenceScore) / 2;
      if (correlationScore < options.threshold || differenceScore < differenceThreshold) continue;
      support += 1;
      supportScore += score;
      bounds.minRow = Math.min(bounds.minRow, row);
      bounds.maxRow = Math.max(bounds.maxRow, row);
      bounds.minColumn = Math.min(bounds.minColumn, column);
      bounds.maxColumn = Math.max(bounds.maxColumn, column);
    }
  }
  return {
    originX,
    originY,
    support,
    supportScore,
    adjustedSupportScore:
      supportScore - Math.abs(originY - (options.preferredOriginY ?? originY)) * 0.005,
    anchorBalance: calculateAnchorBalance(support, bounds, options),
    totalScore,
    slotScores,
  };
}

function isBetterGrid(candidate: GridEvaluation, best: GridEvaluation | undefined): boolean {
  if (!best) return true;
  if (candidate.support !== best.support) return candidate.support > best.support;
  const adjustedDelta = candidate.adjustedSupportScore - best.adjustedSupportScore;
  if (Math.abs(adjustedDelta) > 1e-6) return adjustedDelta > 0;
  if (candidate.anchorBalance !== best.anchorBalance) {
    return candidate.anchorBalance > best.anchorBalance;
  }
  return candidate.totalScore > best.totalScore;
}

function findBestGrid(
  context: GridScoreContext,
  imageRgb: Mat,
  candidates: readonly SlotCandidate[],
): GridEvaluation | undefined {
  let best: GridEvaluation | undefined;
  for (const anchor of candidates) {
    for (let row = 0; row < context.options.rows; row += 1) {
      for (let column = 0; column < context.options.columns; column += 1) {
        const originX = anchor.x - column * context.pitch;
        const originY = anchor.y - row * context.pitch;
        if (!isGridInBounds(originX, originY, imageRgb, context)) continue;
        const evaluation = evaluateGridOrigin(context, originX, originY);
        if (isBetterGrid(evaluation, best)) best = evaluation;
      }
    }
  }
  return best;
}

function createLocatedSlots(
  best: GridEvaluation,
  pitch: number,
  templateId: string,
  options: GridLocationOptions,
): SlotCandidate[] {
  const slots: SlotCandidate[] = [];
  for (let row = 0; row < options.rows; row += 1) {
    for (let column = 0; column < options.columns; column += 1) {
      const index = row * options.columns + column;
      slots.push({
        x: best.originX + column * pitch,
        y: best.originY + row * pitch,
        width: pitch,
        height: pitch,
        row,
        column,
        scale: 1,
        localizationConfidence: best.slotScores[index] ?? 0,
        templateId,
      });
    }
  }
  return slots;
}

export function locateInventoryGrid(
  cv: OpenCv,
  imageRgb: Mat,
  template: GridTemplate,
  options: GridLocationOptions,
): LocatedGrid | null {
  if (template.width > imageRgb.cols || template.height > imageRgb.rows) return null;
  const pitch = template.width;
  const tolerance = options.tolerance ?? 1;
  const result = new cv.Mat();
  let differenceResult: Mat | undefined;
  const differenceThreshold = Math.max(0, options.threshold - 0.03);

  try {
    differenceResult = new cv.Mat();
    cv.matchTemplate(imageRgb, template.rgb, result, cv.TM_CCORR_NORMED, template.mask);
    cv.matchTemplate(imageRgb, template.rgb, differenceResult, cv.TM_SQDIFF_NORMED, template.mask);
    const rawCandidates = collectLocalMaxima(
      result,
      options.threshold,
      template.width,
      template.height,
      1,
      template.metadata.id,
    ).sort((left, right) => right.localizationConfidence - left.localizationConfidence);
    const candidates = filterGridCandidates(
      differenceResult,
      rawCandidates,
      pitch,
      tolerance,
      differenceThreshold,
    );
    const context: GridScoreContext = {
      correlation: result,
      difference: differenceResult,
      pitch,
      tolerance,
      differenceThreshold,
      options,
    };
    const best = findBestGrid(context, imageRgb, candidates);
    if (!best || best.support < options.minSupport) return null;
    return {
      slots: createLocatedSlots(best, pitch, template.metadata.id, options),
      support: best.support,
      confidence: best.totalScore / (options.rows * options.columns),
      origin: { x: best.originX, y: best.originY },
      pitch,
    };
  } finally {
    differenceResult?.delete();
    result.delete();
  }
}
