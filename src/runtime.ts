import { readFile } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";

import type { Mat } from "@techstark/opencv-js";

import { createGridSlots, validateGridOptions } from "./grid.js";
import { decodeImage, resizeRawImage, type RawImage } from "./image.js";
import {
  disposeTemplate,
  locateInventoryGrid,
  locateSlots,
  rgbaMatToTemplate,
  type LoadedTemplate,
} from "./matching.js";
import type { OpenCv } from "./opencv.js";
import type { CatalogTemplate, DetectOptions, ImageInput, SlotRect } from "./types.js";

export interface DecodedMat {
  mat: Mat;
  width: number;
  height: number;
  raw: RawImage;
}

export interface DetectionContext {
  imageRgb: Mat;
  slots: SlotRect[];
  coordinateScale: number;
  normalized: boolean;
  dispose(): void;
}

export function elapsedMilliseconds(startedAt: number): number {
  return Number((performance.now() - startedAt).toFixed(3));
}

export function assertUnitInterval(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${name} must be a finite number between 0 and 1, received ${value}`);
  }
}

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer, received ${value}`);
  }
}

function assertFiniteNonNegativePair(first: number, second: number, message: string): void {
  if (!Number.isFinite(first) || first < 0 || !Number.isFinite(second) || second < 0) {
    throw new Error(message);
  }
}

function assertFinitePositivePair(first: number, second: number, message: string): void {
  if (!Number.isFinite(first) || first <= 0 || !Number.isFinite(second) || second <= 0) {
    throw new Error(message);
  }
}

function validateOptionalNonNegativeInteger(value: number | undefined, name: string): void {
  if (value !== undefined && (!Number.isSafeInteger(value) || value < 0)) {
    throw new Error(`${name} must be a non-negative integer`);
  }
}

function validateSlot(slot: SlotRect, index: number): void {
  const prefix = `slots[${index}]`;
  assertFiniteNonNegativePair(
    slot.x,
    slot.y,
    `${prefix} coordinates must be finite and non-negative`,
  );
  assertFinitePositivePair(
    slot.width,
    slot.height,
    `${prefix} dimensions must be positive and finite`,
  );
  validateOptionalNonNegativeInteger(slot.row, `${prefix}.row`);
  validateOptionalNonNegativeInteger(slot.column, `${prefix}.column`);
  if (slot.scale !== undefined && (!Number.isFinite(slot.scale) || slot.scale <= 0)) {
    throw new Error(`${prefix}.scale must be positive and finite`);
  }
}

function validateScales(scales: readonly number[] | undefined): void {
  if (scales === undefined) return;
  if (scales.length === 0) throw new Error("scales must contain at least one value");
  for (const scale of scales) {
    if (!Number.isFinite(scale) || scale <= 0) {
      throw new Error(`scales must contain only positive finite values, received ${scale}`);
    }
  }
}

function validateGridDetectionOptions(options: DetectOptions): void {
  if (options.gridRows !== undefined) assertPositiveInteger(options.gridRows, "gridRows");
  if (options.gridColumns !== undefined) assertPositiveInteger(options.gridColumns, "gridColumns");
  if (options.minGridSupport === undefined) return;
  assertPositiveInteger(options.minGridSupport, "minGridSupport");
  const capacity = (options.gridRows ?? 4) * (options.gridColumns ?? 6);
  if (options.minGridSupport > capacity) {
    throw new Error(`minGridSupport cannot exceed grid capacity ${capacity}`);
  }
}

function validateThresholds(options: DetectOptions): void {
  if (options.slotThreshold !== undefined) {
    assertUnitInterval(options.slotThreshold, "slotThreshold");
  }
  if (options.itemThreshold !== undefined) {
    assertUnitInterval(options.itemThreshold, "itemThreshold");
  }
  if (options.emptySlotThreshold !== undefined) {
    assertUnitInterval(options.emptySlotThreshold, "emptySlotThreshold");
  }
}

export function validateDetectOptions(options: DetectOptions): void {
  if (options.slots && options.grid) {
    throw new Error("slots and grid are mutually exclusive detection inputs");
  }
  options.slots?.forEach(validateSlot);
  if (options.grid) validateGridOptions(options.grid);
  validateScales(options.scales);
  if (options.maxSlots !== undefined) assertPositiveInteger(options.maxSlots, "maxSlots");
  if (options.slotBorderWidth !== undefined) {
    assertPositiveInteger(options.slotBorderWidth, "slotBorderWidth");
  }
  validateGridDetectionOptions(options);
  validateThresholds(options);
  normalizeLogicalHeight(options.normalizationHeight);
}

export function positiveInteger(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  assertPositiveInteger(resolved, name);
  return resolved;
}

export function unitInterval(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  assertUnitInterval(resolved, name);
  return resolved;
}

export function normalizeLogicalHeight(value: number | false | undefined): number | false {
  const normalized = value ?? 288;
  if (normalized !== false && (!Number.isFinite(normalized) || normalized <= 0)) {
    throw new Error(`normalizationHeight must be positive or false, received ${normalized}`);
  }
  return normalized;
}

export async function imageToRgbaMat(cv: OpenCv, input: ImageInput): Promise<DecodedMat> {
  const image = await decodeImage(input);
  const mat = new cv.Mat(image.height, image.width, cv.CV_8UC4);
  try {
    mat.data.set(image.data);
    return { mat, width: image.width, height: image.height, raw: image };
  } catch (error) {
    mat.delete();
    throw error;
  }
}

type SlotPreparation = { kind: "ready"; slots: SlotRect[] } | { kind: "normalize"; height: number };

function prepareSlots(
  cv: OpenCv,
  imageRgb: Mat,
  slotTemplates: readonly LoadedTemplate[],
  options: DetectOptions,
): SlotPreparation {
  if (options.slots) return { kind: "ready", slots: options.slots };
  if (options.grid) return { kind: "ready", slots: createGridSlots(options.grid) };
  const normalizationHeight = normalizeLogicalHeight(options.normalizationHeight);
  if (normalizationHeight !== false) return { kind: "normalize", height: normalizationHeight };
  const slots = locateSlots(cv, imageRgb, slotTemplates, {
    scales: options.scales ?? [1],
    threshold: unitInterval(options.slotThreshold, 0.94, "slotThreshold"),
    borderWidth: options.slotBorderWidth ?? 4,
    maxSlots: options.maxSlots ?? 100,
  });
  return { kind: "ready", slots };
}

async function createNormalizedRgb(
  cv: OpenCv,
  raw: RawImage,
  normalizationHeight: number,
): Promise<Mat> {
  const normalizedImage = await resizeRawImage(raw, normalizationHeight);
  const normalizedRgba = new cv.Mat(normalizedImage.height, normalizedImage.width, cv.CV_8UC4);
  normalizedRgba.data.set(normalizedImage.data);
  const normalizedRgb = new cv.Mat();
  try {
    cv.cvtColor(normalizedRgba, normalizedRgb, cv.COLOR_RGBA2RGB);
    return normalizedRgb;
  } catch (error) {
    normalizedRgb.delete();
    throw error;
  } finally {
    normalizedRgba.delete();
  }
}

function locateNormalizedGrid(
  cv: OpenCv,
  imageRgb: Mat,
  slotTemplates: readonly LoadedTemplate[],
  options: DetectOptions,
): SlotRect[] {
  const emptySlotTemplate =
    slotTemplates.find((template) => template.metadata.name === "InventorySlot0") ??
    slotTemplates[0];
  if (!emptySlotTemplate) return [];
  const grid = locateInventoryGrid(cv, imageRgb, emptySlotTemplate, {
    rows: options.gridRows ?? 4,
    columns: options.gridColumns ?? 6,
    threshold: unitInterval(options.slotThreshold, 0.97, "slotThreshold"),
    minSupport: options.minGridSupport ?? 5,
    preferredOriginY: Math.round(imageRgb.rows * (99 / 288)),
  });
  return grid?.slots ?? [];
}

function consumeDecodedToRgb(cv: OpenCv, decoded: DecodedMat): Mat {
  const rgb = new cv.Mat();
  try {
    cv.cvtColor(decoded.mat, rgb, cv.COLOR_RGBA2RGB);
    return rgb;
  } catch (error) {
    rgb.delete();
    throw error;
  } finally {
    decoded.mat.delete();
  }
}

/**
 * Owns the RGB matrices and slot-localization state shared by every detector.
 * The decoded RGBA matrix is consumed; callers own only the returned context.
 */
export async function prepareDetectionContext(
  cv: OpenCv,
  decoded: DecodedMat,
  slotTemplates: readonly LoadedTemplate[],
  options: DetectOptions,
): Promise<DetectionContext> {
  const originalRgb = consumeDecodedToRgb(cv, decoded);

  let imageRgb = originalRgb;
  let coordinateScale = 1;
  let normalized = false;

  try {
    const preparation = prepareSlots(cv, imageRgb, slotTemplates, options);
    let slots: SlotRect[];
    if (preparation.kind === "normalize") {
      coordinateScale = decoded.height / preparation.height;
      imageRgb = await createNormalizedRgb(cv, decoded.raw, preparation.height);
      normalized = true;
      slots = locateNormalizedGrid(cv, imageRgb, slotTemplates, options);
    } else {
      slots = preparation.slots;
    }

    let disposed = false;
    return {
      imageRgb,
      slots,
      coordinateScale,
      normalized,
      dispose() {
        if (disposed) return;
        if (normalized) imageRgb.delete();
        originalRgb.delete();
        disposed = true;
      },
    };
  } catch (error) {
    if (normalized) imageRgb.delete();
    originalRgb.delete();
    throw error;
  }
}

export async function loadTemplate(
  cv: OpenCv,
  root: string,
  metadata: CatalogTemplate,
): Promise<LoadedTemplate> {
  const absoluteRoot = resolve(root);
  const absoluteFile = resolve(absoluteRoot, metadata.file);
  const relativeFile = relative(absoluteRoot, absoluteFile);
  if (relativeFile.startsWith("..") || isAbsolute(relativeFile)) {
    throw new Error(`Template file escapes the catalog directory: ${metadata.file}`);
  }
  const image = await decodeImage(await readFile(absoluteFile));
  const rgba = new cv.Mat(image.height, image.width, cv.CV_8UC4);
  try {
    rgba.data.set(image.data);
    return rgbaMatToTemplate(cv, rgba, metadata);
  } finally {
    rgba.delete();
  }
}

export async function loadTemplates(
  cv: OpenCv,
  root: string,
  metadata: readonly CatalogTemplate[],
): Promise<LoadedTemplate[]> {
  const templates: LoadedTemplate[] = [];
  try {
    for (const template of metadata) templates.push(await loadTemplate(cv, root, template));
    return templates;
  } catch (error) {
    templates.forEach(disposeTemplate);
    throw error;
  }
}
