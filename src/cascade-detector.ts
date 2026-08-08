import { InventoryDetector } from "./detector.js";
import { ResourceLifecycle } from "./lifecycle.js";
import { assertUnitInterval, elapsedMilliseconds, positiveInteger } from "./runtime.js";
import { InventoryVisionDetector } from "./vision-detector.js";
import type {
  CascadeDetectOptions,
  CascadeDetectionResult,
  CascadeDetectorOptions,
  CascadeDetectedSlot,
  CascadeSlotDiagnostics,
  DetectOptions,
  DetectedSlot,
  ImageInput,
  InventoryCatalog,
  VisionModel,
} from "./types.js";

const DEFAULT_CASCADE_CONFIDENCE = 0.96;

function fallbackReason(
  slot: DetectedSlot,
  confidenceThreshold: number,
  marginThreshold: number,
): CascadeSlotDiagnostics["fallbackReason"] {
  if (!slot.classification) return "missing-classification";
  if (slot.classification.bestScore < confidenceThreshold) return "low-confidence";
  if (slot.classification.margin < marginThreshold) return "low-margin";
  if (slot.classification.bestLabel !== "__empty__" && !slot.item) return "vision-rejected";
  return null;
}

interface CascadeSettings {
  cascadeConfidence: number;
  cascadeMargin: number;
  templateItemThreshold: number;
  verifyVisionMatches: boolean;
  baseOptions: DetectOptions;
}

interface TemplateSelection {
  indexes: number[];
  candidateNames: (string | undefined)[];
  slots: DetectedSlot[];
}

interface TemplateFallback {
  slots: DetectedSlot[];
  totalMs: number;
  decodeMs: number;
  locateMs: number;
  matchMs: number;
}

function cascadeSettings(options: CascadeDetectOptions): CascadeSettings {
  const {
    cascadeConfidence = DEFAULT_CASCADE_CONFIDENCE,
    cascadeMargin = 0.03,
    templateItemThreshold = 0.85,
    verifyVisionMatches = true,
    ...baseOptions
  } = options;
  assertUnitInterval(cascadeConfidence, "cascadeConfidence");
  assertUnitInterval(cascadeMargin, "cascadeMargin");
  assertUnitInterval(templateItemThreshold, "templateItemThreshold");
  if (typeof verifyVisionMatches !== "boolean") {
    throw new Error(`verifyVisionMatches must be boolean, received ${typeof verifyVisionMatches}`);
  }
  return {
    cascadeConfidence,
    cascadeMargin,
    templateItemThreshold,
    verifyVisionMatches,
    baseOptions,
  };
}

function selectTemplateSlots(
  slots: readonly DetectedSlot[],
  reasons: readonly CascadeSlotDiagnostics["fallbackReason"][],
  verifyVisionMatches: boolean,
): TemplateSelection {
  const indexes: number[] = [];
  const candidateNames: (string | undefined)[] = [];
  slots.forEach((slot, index) => {
    if ((reasons[index] ?? null) !== null) {
      indexes.push(index);
      candidateNames.push(undefined);
    } else if (verifyVisionMatches && slot.item) {
      indexes.push(index);
      candidateNames.push(slot.item.name);
    }
  });
  return {
    indexes,
    candidateNames,
    slots: indexes.map((index) => {
      const slot = slots[index];
      if (!slot) throw new Error(`Vision result omitted requested slot ${index}`);
      return slot;
    }),
  };
}

async function runTemplateFallback(
  detector: InventoryDetector,
  input: ImageInput,
  selection: TemplateSelection,
  options: DetectOptions,
  itemThreshold: number,
): Promise<TemplateFallback> {
  if (selection.slots.length === 0) {
    return { slots: [], totalMs: 0, decodeMs: 0, locateMs: 0, matchMs: 0 };
  }
  const result = await detector.detectWithCandidateVerification(
    input,
    {
      ...options,
      slots: selection.slots,
      normalizationHeight: false,
      itemThreshold,
      alternatives: options.alternatives ?? 3,
    },
    selection.candidateNames,
  );
  return {
    slots: result.slots,
    totalMs: result.timingsMs.total,
    decodeMs: result.timingsMs.decode,
    locateMs: result.timingsMs.locateSlots,
    matchMs: result.timingsMs.matchItems,
  };
}

function verifyTemplateMatches(
  visionSlots: readonly DetectedSlot[],
  templateSlots: readonly DetectedSlot[],
  indexes: readonly number[],
  reasons: CascadeSlotDiagnostics["fallbackReason"][],
): number {
  let verifiedCount = 0;
  for (const [templateIndex, originalIndex] of indexes.entries()) {
    if ((reasons[originalIndex] ?? null) !== null) continue;
    const predicted = visionSlots[originalIndex]?.item;
    const verified = templateSlots[templateIndex]?.item;
    if (predicted && verified?.name === predicted.name) verifiedCount += 1;
    else reasons[originalIndex] = "asset-mismatch";
  }
  return verifiedCount;
}

function mergeCascadeSlots(
  visionSlots: readonly DetectedSlot[],
  templateSlots: readonly DetectedSlot[],
  indexes: readonly number[],
  reasons: readonly CascadeSlotDiagnostics["fallbackReason"][],
  verifyVisionMatches: boolean,
): CascadeDetectedSlot[] {
  const templatePositionByIndex = new Map(
    indexes.map((originalIndex, templateIndex) => [originalIndex, templateIndex]),
  );
  return visionSlots.map((slot, index) => {
    const reason = reasons[index] ?? null;
    if (reason === null) {
      return {
        ...slot,
        cascade: {
          backend: "vision-features",
          fallbackReason: null,
          assetVerified: verifyVisionMatches && slot.item ? true : null,
        },
      };
    }
    const templateIndex = templatePositionByIndex.get(index);
    const verified = templateIndex === undefined ? undefined : templateSlots[templateIndex];
    if (!verified) {
      throw new Error(
        `Template fallback did not return original slot ${index} of ${indexes.length}`,
      );
    }
    return {
      ...slot,
      ...verified,
      cascade: {
        backend: "template",
        fallbackReason: reason,
        assetVerified: reason === "asset-mismatch" ? false : null,
      },
    };
  });
}

/**
 * Fast structural/color classification with selective extracted-game-asset
 * verification. Uncertain cells use the complete catalog; otherwise accepted
 * non-empty identities are checked against only their predicted game asset.
 */
export class InventoryCascadeDetector {
  readonly catalog: InventoryCatalog;
  readonly model: VisionModel;
  readonly #vision: InventoryVisionDetector;
  readonly #template: InventoryDetector;
  readonly #lifecycle: ResourceLifecycle;

  private constructor(vision: InventoryVisionDetector, template: InventoryDetector) {
    this.#vision = vision;
    this.#template = template;
    this.catalog = template.catalog;
    this.model = vision.model;
    this.#lifecycle = new ResourceLifecycle("InventoryCascadeDetector", () => {
      this.#vision.dispose();
      this.#template.dispose();
    });
  }

  static async create(options: CascadeDetectorOptions = {}): Promise<InventoryCascadeDetector> {
    const vision = await InventoryVisionDetector.create(options);
    try {
      const template = await InventoryDetector.createForExplicitSlots(
        options.catalogPath ? { catalogPath: options.catalogPath } : {},
      );
      return new InventoryCascadeDetector(vision, template);
    } catch (error) {
      vision.dispose();
      throw error;
    }
  }

  async detect(
    input: ImageInput,
    options: CascadeDetectOptions = {},
  ): Promise<CascadeDetectionResult> {
    const leaveLifecycle = this.#lifecycle.enter();
    try {
      const totalStarted = performance.now();
      const settings = cascadeSettings(options);
      const visionOptions: DetectOptions = {
        ...settings.baseOptions,
        alternatives: Math.max(
          2,
          positiveInteger(settings.baseOptions.alternatives, 3, "alternatives"),
        ),
      };
      const visionResult = await this.#vision.detect(input, visionOptions);
      const reasons = visionResult.slots.map((slot) =>
        fallbackReason(slot, settings.cascadeConfidence, settings.cascadeMargin),
      );
      const selection = selectTemplateSlots(
        visionResult.slots,
        reasons,
        settings.verifyVisionMatches,
      );
      const fallback = await runTemplateFallback(
        this.#template,
        input,
        selection,
        settings.baseOptions,
        settings.templateItemThreshold,
      );
      const assetVerifiedSlots = verifyTemplateMatches(
        visionResult.slots,
        fallback.slots,
        selection.indexes,
        reasons,
      );
      const fallbackSlots = reasons.filter((reason) => reason !== null).length;
      const slots = mergeCascadeSlots(
        visionResult.slots,
        fallback.slots,
        selection.indexes,
        reasons,
        settings.verifyVisionMatches,
      );
      const total = elapsedMilliseconds(totalStarted);
      return {
        image: visionResult.image,
        slots,
        matchedItems: slots.filter((slot) => slot.item !== null),
        catalogSize: this.catalog.items.length,
        timingsMs: {
          decode: Number((visionResult.timingsMs.decode + fallback.decodeMs).toFixed(3)),
          locateSlots: Number((visionResult.timingsMs.locateSlots + fallback.locateMs).toFixed(3)),
          matchItems: Number((visionResult.timingsMs.matchItems + fallback.matchMs).toFixed(3)),
          total,
        },
        cascade: {
          fallbackSlots,
          visionSlots: slots.length - fallbackSlots,
          confidenceThreshold: settings.cascadeConfidence,
          marginThreshold: settings.cascadeMargin,
          assetVerificationEnabled: settings.verifyVisionMatches,
          assetVerifiedSlots,
          templateCheckedSlots: selection.slots.length,
          visionTotalMs: visionResult.timingsMs.total,
          templateFallbackTotalMs: fallback.totalMs,
        },
      };
    } finally {
      leaveLifecycle();
    }
  }

  dispose(): void {
    this.#lifecycle.dispose();
  }
}

export async function createInventoryCascadeDetector(
  options: CascadeDetectorOptions = {},
): Promise<InventoryCascadeDetector> {
  return InventoryCascadeDetector.create(options);
}
