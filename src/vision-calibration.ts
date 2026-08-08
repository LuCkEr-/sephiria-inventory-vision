import type { VisionReference } from "./types.js";
import { normalizedDotProduct } from "./vision-features.js";

export const DEFAULT_VISION_ACCEPTANCE_THRESHOLD = 0.936;
const CALIBRATION_EPSILON = 0.000_001;
const POSITIVE_GENERALIZATION_TOLERANCE = 0.007;
const HIGH_CONSISTENCY_THRESHOLD = 0.98;
const HIGH_CONSISTENCY_TOLERANCE = 0.022;
const SINGLETON_TRUSTED_SCORE = 0.99;
const SINGLETON_MARGIN_THRESHOLD = 0.04;

/** @internal */
export function countVisionLabelSources(
  references: readonly VisionReference[],
): ReadonlyMap<string, number> {
  const sources = new Map<string, Set<string>>();
  for (const reference of references) {
    const labelSources = sources.get(reference.label) ?? new Set<string>();
    labelSources.add(reference.sourceScreenshot);
    sources.set(reference.label, labelSources);
  }
  return new Map([...sources].map(([label, labelSources]) => [label, labelSources.size]));
}

/** @internal */
export function evaluateVisionAcceptance(
  label: string,
  score: number,
  margin: number,
  threshold: number,
  sourceCount: number,
): {
  accepted: boolean;
  minimumMargin: number;
  reason: "accepted" | "empty" | "low-confidence" | "singleton-margin";
} {
  const minimumMargin =
    sourceCount === 1 && score < SINGLETON_TRUSTED_SCORE ? SINGLETON_MARGIN_THRESHOLD : 0;
  if (label === "__empty__") return { accepted: false, minimumMargin, reason: "empty" };
  if (score < threshold) return { accepted: false, minimumMargin, reason: "low-confidence" };
  if (margin < minimumMargin) {
    return { accepted: false, minimumMargin, reason: "singleton-margin" };
  }
  return { accepted: true, minimumMargin, reason: "accepted" };
}

/** @internal Runtime confidence for model vectors already validated as unit length. */
export function visionConfidence(left: readonly number[], right: readonly number[]): number {
  return Math.max(0, Math.min(1, normalizedDotProduct(left, right)));
}

/** @internal */
export function effectiveVisionAcceptanceThreshold(
  detectorThreshold: number,
  label: string,
  classThresholds: Readonly<Record<string, number>> | undefined,
): number {
  return Math.max(detectorThreshold, classThresholds?.[label] ?? 0);
}

/** @internal */
export function calibrateVisionAcceptanceThresholds(
  samples: readonly VisionReference[],
  baseline = DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
): Readonly<Record<string, number>> {
  const baseQueries = samples.filter(
    (reference) => reference.label !== "__empty__" && (reference.variant ?? "base") === "base",
  );
  const itemLabels = [...new Set(baseQueries.map((reference) => reference.label))];
  return Object.fromEntries(
    itemLabels.flatMap((label) => {
      const labelReferences = samples.filter((reference) => reference.label === label);
      const labelBaseQueries = baseQueries.filter((query) => query.label === label);
      const impostorScores = baseQueries
        .filter((query) => query.label !== label)
        .flatMap((query) =>
          labelReferences.map((reference) => visionConfidence(query.features, reference.features)),
        );
      const strongestImpostor = Math.max(...impostorScores, 0);
      const crossSourcePositiveScores = labelBaseQueries.flatMap((query) => {
        const candidates = labelReferences.filter(
          (reference) => reference.sourceScreenshot !== query.sourceScreenshot,
        );
        return candidates.length === 0
          ? []
          : [
              Math.max(
                ...candidates.map((reference) =>
                  visionConfidence(query.features, reference.features),
                ),
              ),
            ];
      });
      const weakestPositive =
        crossSourcePositiveScores.length === 0 ? 0 : Math.min(...crossSourcePositiveScores);
      const positiveTolerance =
        weakestPositive >= HIGH_CONSISTENCY_THRESHOLD
          ? HIGH_CONSISTENCY_TOLERANCE
          : POSITIVE_GENERALIZATION_TOLERANCE;
      const positiveFloor =
        crossSourcePositiveScores.length === 0 ? 0 : weakestPositive - positiveTolerance;
      const separationFloor =
        strongestImpostor < baseline ? 0 : strongestImpostor + CALIBRATION_EPSILON;
      if (separationFloor === 0 && positiveFloor <= baseline) return [];
      const threshold = Math.min(1, Math.max(separationFloor, positiveFloor));
      return [[label, Number(threshold.toFixed(6))]];
    }),
  );
}
