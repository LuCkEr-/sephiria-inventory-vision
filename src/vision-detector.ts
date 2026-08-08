import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Mat } from "@techstark/opencv-js";

import { loadCatalog } from "./catalog.js";
import { ResourceLifecycle } from "./lifecycle.js";
import { disposeTemplate, type LoadedTemplate } from "./matching.js";
import { getOpenCv, type OpenCv } from "./opencv.js";
import {
  elapsedMilliseconds,
  imageToRgbaMat,
  loadTemplates,
  positiveInteger,
  prepareDetectionContext,
  unitInterval,
  validateDetectOptions,
} from "./runtime.js";
import type {
  ClassificationDiagnostics,
  DetectOptions,
  DetectionResult,
  ImageInput,
  InventoryCatalog,
  MatchAlternative,
  SlotRect,
  VisionDetectorOptions,
  VisionModel,
  VisionReference,
  VisionReferenceVariant,
} from "./types.js";
import {
  DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
  countVisionLabelSources,
  effectiveVisionAcceptanceThreshold,
  evaluateVisionAcceptance,
  visionConfidence,
} from "./vision-calibration.js";
import { extractVisionFeatures, VISION_FEATURE_SIZE } from "./vision-features.js";

function defaultVisionModelPath(): string {
  return fileURLToPath(new URL("../assets/vision/model.json", import.meta.url));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

const SUPPORTED_VARIANTS = new Set<VisionReferenceVariant>([
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
]);

function validateModelHeader(
  model: unknown,
  path: string,
): asserts model is Record<string, unknown> {
  if (
    !isRecord(model) ||
    model["schemaVersion"] !== 1 ||
    model["slotSize"] !== 32 ||
    !isNonEmptyString(model["method"]) ||
    !isNonEmptyString(model["inputPolicy"]) ||
    !isNonEmptyString(model["generatedAt"]) ||
    !Number.isFinite(Date.parse(model["generatedAt"]))
  ) {
    throw new Error(`Unsupported or malformed vision model: ${path}`);
  }
}

function validateLabels(value: unknown, path: string): string[] {
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.some((label) => !isNonEmptyString(label)) ||
    new Set(value).size !== value.length ||
    !value.includes("__empty__")
  ) {
    throw new Error(`Vision model labels are missing, duplicated, or lack __empty__: ${path}`);
  }
  return value as string[];
}

function validateSourceScreenshots(value: unknown, path: string): string[] {
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.some((source) => !isNonEmptyString(source)) ||
    new Set(value).size !== value.length
  ) {
    throw new Error(`Vision model sourceScreenshots are missing or duplicated: ${path}`);
  }
  return value as string[];
}

function validateAugmentationPolicy(
  value: unknown,
  path: string,
): readonly VisionReferenceVariant[] | undefined {
  if (
    value !== undefined &&
    (!Array.isArray(value) ||
      value.some((variant) => !SUPPORTED_VARIANTS.has(variant as VisionReferenceVariant)) ||
      new Set(value).size !== value.length)
  ) {
    throw new Error(`Vision model augmentationPolicy is malformed: ${path}`);
  }
  return value as readonly VisionReferenceVariant[] | undefined;
}

function validateAcceptanceThresholds(
  value: unknown,
  labels: readonly string[],
  path: string,
): asserts value is Readonly<Record<string, number>> | undefined {
  if (value === undefined) return;
  if (!isRecord(value) || Array.isArray(value)) {
    throw new Error(`Vision model acceptanceThresholds are malformed: ${path}`);
  }
  for (const [label, threshold] of Object.entries(value)) {
    if (
      label === "__empty__" ||
      !labels.includes(label) ||
      !Number.isFinite(threshold) ||
      (threshold as number) < 0 ||
      (threshold as number) > 1
    ) {
      throw new Error(`Vision model acceptance threshold for ${label} is malformed: ${path}`);
    }
  }
}

function validateReferenceFeatures(value: unknown, id: string): void {
  if (
    !Array.isArray(value) ||
    value.length !== VISION_FEATURE_SIZE ||
    value.some((feature) => !Number.isFinite(feature))
  ) {
    throw new Error(`Vision reference ${id} has an invalid feature vector`);
  }
  const magnitude = Math.sqrt(
    value.reduce<number>((sum, feature) => sum + (feature as number) ** 2, 0),
  );
  if (Math.abs(magnitude - 1) > 0.001) {
    throw new Error(`Vision reference ${id} feature vector must have unit length`);
  }
}

function validateReferenceItem(item: unknown, label: string, id: string): void {
  if (label === "__empty__") {
    if (item !== null) throw new Error(`Vision reference ${id} has inconsistent item identity`);
    return;
  }
  if (!isRecord(item)) {
    throw new Error(`Vision reference ${id} has inconsistent item identity`);
  }
  if (
    item["name"] !== label ||
    !isNonEmptyString(item["itemId"]) ||
    !Number.isFinite(item["confidence"]) ||
    (item["confidence"] as number) < 0 ||
    (item["confidence"] as number) > 1 ||
    !isRecord(item["offset"]) ||
    !Number.isFinite(item["offset"]["x"]) ||
    !Number.isFinite(item["offset"]["y"])
  ) {
    throw new Error(`Vision reference ${id} item does not match its label or schema`);
  }
}

function validateReferenceVariant(
  value: unknown,
  policy: readonly VisionReferenceVariant[] | undefined,
  id: string,
): void {
  if (value !== undefined && !SUPPORTED_VARIANTS.has(value as VisionReferenceVariant)) {
    throw new Error(`Vision reference ${id} uses unsupported variant ${JSON.stringify(value)}`);
  }
  const variant = (value ?? "base") as VisionReferenceVariant;
  if (policy && !policy.includes(variant)) {
    throw new Error(`Vision reference ${id} uses a variant outside augmentationPolicy`);
  }
}

function validateReference(
  reference: unknown,
  labels: readonly string[],
  sourceScreenshots: readonly string[],
  policy: readonly VisionReferenceVariant[] | undefined,
): { id: string; label: string } {
  if (!isRecord(reference) || !isNonEmptyString(reference["id"])) {
    throw new Error("Vision model contains a malformed reference");
  }
  const id = reference["id"];
  const sourceScreenshot = reference["sourceScreenshot"];
  if (!isNonEmptyString(sourceScreenshot) || !sourceScreenshots.includes(sourceScreenshot)) {
    throw new Error(`Vision reference ${id} uses an unknown source screenshot`);
  }
  if (
    !Number.isSafeInteger(reference["row"]) ||
    (reference["row"] as number) < 0 ||
    !Number.isSafeInteger(reference["column"]) ||
    (reference["column"] as number) < 0
  ) {
    throw new Error(`Vision reference ${id} has invalid grid coordinates`);
  }
  const label = reference["label"];
  if (!isNonEmptyString(label) || !labels.includes(label)) {
    throw new Error(`Vision reference ${id} uses unknown label ${String(label)}`);
  }
  validateReferenceFeatures(reference["features"], id);
  validateReferenceItem(reference["item"], label, id);
  validateReferenceVariant(reference["variant"], policy, id);
  return { id, label };
}

function validateModel(model: unknown, path: string): asserts model is VisionModel {
  validateModelHeader(model, path);
  const references = model["references"];
  if (!Array.isArray(references) || references.length === 0) {
    throw new Error(`Vision model must contain at least one reference: ${path}`);
  }
  const labels = validateLabels(model["labels"], path);
  const sourceScreenshots = validateSourceScreenshots(model["sourceScreenshots"], path);
  const augmentationPolicy = validateAugmentationPolicy(model["augmentationPolicy"], path);
  validateAcceptanceThresholds(model["acceptanceThresholds"], labels, path);
  const ids = new Set<string>();
  const representedLabels = new Set<string>();
  for (const reference of references) {
    const validated = validateReference(reference, labels, sourceScreenshots, augmentationPolicy);
    if (ids.has(validated.id)) {
      throw new Error(`Vision model contains a missing or duplicate reference id: ${validated.id}`);
    }
    ids.add(validated.id);
    representedLabels.add(validated.label);
  }
  if (labels.some((label) => !representedLabels.has(label))) {
    throw new Error(`Vision model contains labels without references: ${path}`);
  }
}

interface RankedReference {
  label: string;
  reference: VisionReference;
  score: number;
}

function rankReferences(
  features: readonly number[],
  references: readonly VisionReference[],
): RankedReference[] {
  const bestByLabel = new Map<string, Omit<RankedReference, "label">>();
  for (const reference of references) {
    const score = visionConfidence(features, reference.features);
    const previous = bestByLabel.get(reference.label);
    if (!previous || score > previous.score) bestByLabel.set(reference.label, { reference, score });
  }
  return [...bestByLabel.entries()]
    .map(([label, value]) => ({ label, ...value }))
    .sort((left, right) => right.score - left.score);
}

function rankedAlternatives(ranked: readonly RankedReference[], limit: number): MatchAlternative[] {
  const alternatives: MatchAlternative[] = [];
  for (const { label, reference, score } of ranked) {
    if (label === "__empty__" || !reference.item) continue;
    alternatives.push({
      ...reference.item,
      confidence: Math.max(0, Math.min(1, score)),
      offset: { x: 0, y: 0 },
      classifier: "vision-features",
      nearestReferenceId: reference.id,
      nearestReferenceScreenshot: reference.sourceScreenshot,
    });
    if (alternatives.length >= limit) break;
  }
  return alternatives;
}

interface ClassificationOptions {
  references: readonly VisionReference[];
  alternativeCount: number;
  acceptanceThreshold: number;
  acceptanceThresholds: Readonly<Record<string, number>>;
  sourceCounts: ReadonlyMap<string, number>;
}

function classifySlot(
  cv: OpenCv,
  imageRgb: Mat,
  slot: SlotRect,
  options: ClassificationOptions,
): {
  bestLabel: string;
  bestScore: number;
  alternatives: MatchAlternative[];
  classification: ClassificationDiagnostics;
} | null {
  const x = Math.max(0, Math.round(slot.x));
  const y = Math.max(0, Math.round(slot.y));
  const width = Math.min(Math.round(slot.width), imageRgb.cols - x);
  const height = Math.min(Math.round(slot.height), imageRgb.rows - y);
  if (width <= 0 || height <= 0) return null;

  const crop = imageRgb.roi(new cv.Rect(x, y, width, height));
  let normalized: Mat | undefined;
  try {
    normalized = new cv.Mat();
    cv.resize(crop, normalized, new cv.Size(32, 32), 0, 0, cv.INTER_NEAREST);
    const features = extractVisionFeatures(normalized.data, 32, 32, 3);
    const ranked = rankReferences(features, options.references);
    const best = ranked[0];
    if (!best) return null;
    const second = ranked[1];
    const margin = second ? best.score - second.score : best.score;
    const effectiveThreshold = effectiveVisionAcceptanceThreshold(
      options.acceptanceThreshold,
      best.label,
      options.acceptanceThresholds,
    );
    const acceptance = evaluateVisionAcceptance(
      best.label,
      best.score,
      margin,
      effectiveThreshold,
      options.sourceCounts.get(best.label) ?? 0,
    );
    return {
      bestLabel: best.label,
      bestScore: best.score,
      alternatives: rankedAlternatives(ranked, options.alternativeCount),
      classification: {
        bestLabel: best.label,
        bestScore: best.score,
        secondLabel: second?.label ?? null,
        secondScore: second?.score ?? null,
        margin,
        acceptanceThreshold: effectiveThreshold,
        minimumMargin: acceptance.minimumMargin,
        accepted: acceptance.accepted,
        acceptanceReason: acceptance.reason,
      },
    };
  } finally {
    normalized?.delete();
    crop.delete();
  }
}

export class InventoryVisionDetector {
  readonly catalog: InventoryCatalog;
  readonly model: VisionModel;
  readonly #cv: OpenCv;
  readonly #slotTemplates: LoadedTemplate[];
  readonly #sourceCounts: ReadonlyMap<string, number>;
  readonly #lifecycle: ResourceLifecycle;

  private constructor(
    cv: OpenCv,
    catalog: InventoryCatalog,
    model: VisionModel,
    slotTemplates: LoadedTemplate[],
  ) {
    this.#cv = cv;
    this.catalog = catalog;
    this.model = model;
    this.#slotTemplates = slotTemplates;
    this.#sourceCounts = countVisionLabelSources(model.references);
    this.#lifecycle = new ResourceLifecycle("InventoryVisionDetector", () => {
      this.#slotTemplates.forEach(disposeTemplate);
    });
  }

  static async create(options: VisionDetectorOptions = {}): Promise<InventoryVisionDetector> {
    const { cv } = await getOpenCv();
    const loaded = await loadCatalog(options.catalogPath);
    const modelPath = resolve(options.modelPath ?? defaultVisionModelPath());
    const model: unknown = JSON.parse(await readFile(modelPath, "utf8"));
    validateModel(model, modelPath);
    const slots = await loadTemplates(cv, loaded.root, loaded.catalog.slotTemplates);
    return new InventoryVisionDetector(cv, loaded.catalog, model, slots);
  }

  async detect(input: ImageInput, options: DetectOptions = {}): Promise<DetectionResult> {
    const leaveLifecycle = this.#lifecycle.enter();
    try {
      validateDetectOptions(options);
      const totalStarted = performance.now();
      const decodeStarted = performance.now();
      const decoded = await imageToRgbaMat(this.#cv, input);
      const decodeTime = elapsedMilliseconds(decodeStarted);
      const locateStarted = performance.now();
      const context = await prepareDetectionContext(
        this.#cv,
        decoded,
        this.#slotTemplates,
        options,
      );
      const locateTime = elapsedMilliseconds(locateStarted);
      const { coordinateScale, imageRgb, normalized, slots } = context;

      try {
        const matchStarted = performance.now();
        const threshold = unitInterval(
          options.itemThreshold,
          DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
          "itemThreshold",
        );
        const alternativeCount = positiveInteger(options.alternatives, 3, "alternatives");
        const workingSlots = slots.map((slot) => {
          const classified = classifySlot(this.#cv, imageRgb, slot, {
            references: this.model.references,
            alternativeCount,
            acceptanceThreshold: threshold,
            acceptanceThresholds: this.model.acceptanceThresholds ?? {},
            sourceCounts: this.#sourceCounts,
          });
          const bestItem = classified?.alternatives[0] ?? null;
          const item =
            classified &&
            classified.classification.accepted &&
            bestItem?.name === classified.bestLabel
              ? bestItem
              : null;
          return {
            ...slot,
            item,
            alternatives: classified?.alternatives ?? [],
            ...(classified ? { classification: classified.classification } : {}),
          };
        });

        const detectedSlots = normalized
          ? workingSlots.map((slot) => ({
              ...slot,
              x: Math.round(slot.x * coordinateScale),
              y: Math.round(slot.y * coordinateScale),
              width: Math.round(slot.width * coordinateScale),
              height: Math.round(slot.height * coordinateScale),
              scale: coordinateScale,
              item: slot.item ? { ...slot.item, offset: { x: 0, y: 0 } } : null,
              alternatives: slot.alternatives.map((alternative) => ({
                ...alternative,
                offset: { x: 0, y: 0 },
              })),
            }))
          : workingSlots;
        const matchTime = elapsedMilliseconds(matchStarted);
        return {
          image: { width: decoded.width, height: decoded.height },
          slots: detectedSlots,
          matchedItems: detectedSlots.filter((slot) => slot.item !== null),
          catalogSize: this.model.labels.filter((label) => label !== "__empty__").length,
          timingsMs: {
            decode: decodeTime,
            locateSlots: locateTime,
            matchItems: matchTime,
            total: elapsedMilliseconds(totalStarted),
          },
        };
      } finally {
        context.dispose();
      }
    } finally {
      leaveLifecycle();
    }
  }

  dispose(): void {
    this.#lifecycle.dispose();
  }
}

export async function createInventoryVisionDetector(
  options: VisionDetectorOptions = {},
): Promise<InventoryVisionDetector> {
  return InventoryVisionDetector.create(options);
}
