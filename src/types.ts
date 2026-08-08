export type ImageInput = string | Buffer | Uint8Array;

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SlotRect extends Rect {
  row?: number;
  column?: number;
  scale?: number;
  localizationConfidence?: number;
  templateId?: string;
}

export interface GridOptions {
  x: number;
  y: number;
  rows: number;
  columns: number;
  slotSize?: number;
  gapX?: number;
  gapY?: number;
  scale?: number;
}

export interface PlacementRequirement {
  code: string;
  description: string;
  /** Unity component that supplied this rule. */
  sourceClass: string;
}

export interface ItemGameplayMetadata {
  itemId: number;
  displayName: string;
  itemType: string;
  itemTypeValue: number;
  rarity: number;
  /** Whether this item can be rotated in the inventory. Null means not present in the prefab data. */
  rotatable: boolean | null;
  /** Maximum charm effect/buff level. Null for non-charms or when unavailable. */
  maxBuffLevel: number | null;
  placementRequirements: PlacementRequirement[];
  /** Stone-tablet activation/condition pattern, when present. */
  conditionQuery: string | null;
  /** Stone-tablet affected-cell pattern, when present. */
  effectQuery: string | null;
  isUniqueEffect?: boolean;
  isWeaponRelated?: boolean;
  effectStringKeys?: string[];
  includeConditionInBounds?: boolean;
  componentClasses?: string[];
  sourcePrefab?: {
    asset: string;
    pathId: number;
    name: string;
  } | null;
}

export interface CatalogItemVariant {
  itemId: number;
  displayName: string;
  gameplay: ItemGameplayMetadata;
}

export interface CatalogTemplate {
  id: string;
  name: string;
  file: string;
  width: number;
  height: number;
  sourceFile: string;
  pathId: number;
  /** Unity sprite object name used for icon-only identification. */
  spriteName?: string;
  /** Real localized item names associated with this icon. More than one means the icon is not identity-unique. */
  displayNames?: string[];
  /** Game item IDs associated with this icon. */
  itemIds?: number[];
  /** True when multiple item definitions reuse the exact same icon. */
  ambiguousIdentity?: boolean;
  /** Per-item gameplay metadata. Multiple variants are retained when an icon is shared. */
  itemVariants?: CatalogItemVariant[];
  /** Clockwise rotation applied to the canonical extracted sprite. */
  rotationDegrees?: 0 | 90 | 180 | 270;
  /** Unrotated catalog template ID when this entry is a generated rotation. */
  canonicalTemplateId?: string;
}

export interface InventoryCatalog {
  schemaVersion: number;
  game: string;
  nativeSlotSize: number;
  generatedFrom?: string;
  items: CatalogTemplate[];
  slotTemplates: CatalogTemplate[];
}

export interface DetectorOptions {
  catalogPath?: string;
}

export interface VisionDetectorOptions extends DetectorOptions {
  modelPath?: string;
}

export type CascadeDetectorOptions = VisionDetectorOptions;

export type ClassifierBackend = "template" | "vision-features";

export interface DetectOptions {
  slots?: SlotRect[];
  grid?: GridOptions;
  scales?: number[];
  slotThreshold?: number;
  itemThreshold?: number;
  alternatives?: number;
  maxSlots?: number;
  slotBorderWidth?: number;
  /** Logical UI height used for fast, resolution-independent auto localization. Set false for legacy multi-scale scanning. */
  normalizationHeight?: number | false;
  gridRows?: number;
  gridColumns?: number;
  minGridSupport?: number;
  /** Empty-slot template confidence above which item matching is skipped. */
  emptySlotThreshold?: number;
}

export interface CascadeDetectOptions extends DetectOptions {
  /** Minimum nearest-reference similarity required to skip template verification. */
  cascadeConfidence?: number;
  /** Minimum difference between the two best labels required to skip template verification. */
  cascadeMargin?: number;
  /** Template acceptance threshold used only for fallback cells. */
  templateItemThreshold?: number;
  /** Verify otherwise accepted non-empty vision matches against their extracted game asset. */
  verifyVisionMatches?: boolean;
}

export interface MatchAlternative {
  itemId: string;
  name: string;
  confidence: number;
  offset: { x: number; y: number };
  spriteName?: string;
  displayNames?: string[];
  itemIds?: number[];
  ambiguousIdentity?: boolean;
  /** Gameplay metadata for every logical item represented by this matched icon. */
  itemVariants?: CatalogItemVariant[];
  /** Clockwise rotation of the matched icon template. */
  rotationDegrees?: 0 | 90 | 180 | 270;
  canonicalTemplateId?: string;
  /** Matcher backend that produced this result. */
  classifier?: ClassifierBackend;
  /** ID of the nearest learned inventory-cell reference. */
  nearestReferenceId?: string;
  /** Screenshot supplying the nearest learned reference. */
  nearestReferenceScreenshot?: string;
}

export interface VisionReference {
  id: string;
  sourceScreenshot: string;
  row: number;
  column: number;
  label: string;
  item: MatchAlternative | null;
  features: number[];
  /** Deterministic input transformation used to create this reference. */
  variant?: VisionReferenceVariant;
}

export type VisionReferenceVariant =
  | "base"
  | "brightness-85"
  | "jpeg-80"
  | "source-jpeg-80"
  | "nearest-150"
  | "roundtrip-nearest-150"
  | "roundtrip-cubic-150-y1"
  | "cubic-150"
  | "source-cubic-150"
  | "source-cubic-150-y1";

export interface VisionModel {
  schemaVersion: 1;
  method: string;
  inputPolicy: string;
  slotSize: 32;
  generatedAt: string;
  sourceScreenshots: string[];
  labels: string[];
  references: VisionReference[];
  augmentationPolicy?: readonly VisionReferenceVariant[];
  /**
   * Minimum raw cosine confidence for labels whose nearest known impostor is
   * stronger than the detector-wide threshold.
   */
  acceptanceThresholds?: Readonly<Record<string, number>>;
}

export interface DetectedSlot extends SlotRect {
  item: MatchAlternative | null;
  alternatives: MatchAlternative[];
  /** Nearest-label evidence retained for cascade decisions and diagnostics. */
  classification?: ClassificationDiagnostics;
}

export interface ClassificationDiagnostics {
  bestLabel: string;
  bestScore: number;
  secondLabel: string | null;
  secondScore: number | null;
  margin: number;
  /** Effective detector-wide plus class-specific acceptance threshold. */
  acceptanceThreshold: number;
  /** Effective margin floor, used only for non-exact singleton-class matches. */
  minimumMargin: number;
  /** Whether the winning non-empty label passed the complete acceptance policy. */
  accepted: boolean;
  /** Auditable reason for the final vision acceptance decision. */
  acceptanceReason: "accepted" | "empty" | "low-confidence" | "singleton-margin";
}

export interface CascadeSlotDiagnostics {
  backend: ClassifierBackend;
  fallbackReason: CascadeFallbackReason;
  /** Whether an accepted vision identity also matched its extracted game asset. */
  assetVerified: boolean | null;
}

export type CascadeFallbackReason =
  | "low-confidence"
  | "low-margin"
  | "vision-rejected"
  | "missing-classification"
  | "asset-mismatch"
  | null;

export interface CascadeDetectedSlot extends DetectedSlot {
  cascade: CascadeSlotDiagnostics;
}

export interface DetectionResult {
  image: { width: number; height: number };
  slots: DetectedSlot[];
  matchedItems: DetectedSlot[];
  catalogSize: number;
  timingsMs: DetectionTimings;
}

export interface DetectionTimings {
  decode: number;
  locateSlots: number;
  matchItems: number;
  total: number;
}

export interface CascadeDetectionResult extends Omit<DetectionResult, "slots" | "matchedItems"> {
  slots: CascadeDetectedSlot[];
  matchedItems: CascadeDetectedSlot[];
  cascade: {
    fallbackSlots: number;
    visionSlots: number;
    confidenceThreshold: number;
    marginThreshold: number;
    assetVerificationEnabled: boolean;
    assetVerifiedSlots: number;
    templateCheckedSlots: number;
    visionTotalMs: number;
    templateFallbackTotalMs: number;
  };
}
