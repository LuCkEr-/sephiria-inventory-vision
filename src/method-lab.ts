import { extractVisionFeatures, normalizedDotProduct } from "./vision-features.js";
import {
  cosineDescriptorSimilarity,
  extractEdgeShapeFeatures,
  extractOrbDescriptors,
  extractPerceptualHash,
  extractPixelVector,
  orbSimilarity,
  perceptualHashSimilarity,
  type OrbDescriptor,
  type PerceptualHash,
} from "./method-features.js";

export type ClassicalMethod =
  "vision-features" | "perceptual-hash" | "edge-shape" | "orb" | "hybrid-retrieval";

export const CLASSICAL_METHODS: readonly ClassicalMethod[] = [
  "vision-features",
  "perceptual-hash",
  "edge-shape",
  "orb",
  "hybrid-retrieval",
];

export interface MethodCell {
  id: string;
  label: string;
  pixels: ArrayLike<number>;
}

export interface PreparedMethodCell {
  id: string;
  label: string;
  pixels: number[];
  vision: number[];
  hash: PerceptualHash;
  edges: number[];
  orb: OrbDescriptor[];
  pixelVector: number[];
}

export interface MethodMatch {
  label: string;
  score: number;
  referenceId: string;
  method: ClassicalMethod;
}

export function prepareMethodCell(cell: MethodCell): PreparedMethodCell {
  if (typeof cell.id !== "string" || cell.id.length === 0) {
    throw new Error("Method cell id must be a non-empty string");
  }
  if (typeof cell.label !== "string" || cell.label.length === 0) {
    throw new Error(`Cell ${cell.id} label must be a non-empty string`);
  }
  if (cell.pixels.length < 32 * 32 * 3)
    throw new Error(`Cell ${cell.id} does not contain 32x32 RGB pixels`);
  return {
    id: cell.id,
    label: cell.label,
    pixels: Array.from(cell.pixels).slice(0, 32 * 32 * 3),
    vision: extractVisionFeatures(cell.pixels, 32, 32, 3),
    hash: extractPerceptualHash(cell.pixels, 32, 32, 3),
    edges: extractEdgeShapeFeatures(cell.pixels, 32, 32, 3),
    // Sixteen keypoints keeps the experimental ORB matcher practical in pure
    // TypeScript while retaining the strongest corners in a 32px icon.
    orb: extractOrbDescriptors(cell.pixels, 32, 32, 3, 16),
    pixelVector: extractPixelVector(cell.pixels, 32, 32, 3),
  };
}

function score(
  method: Exclude<ClassicalMethod, "hybrid-retrieval">,
  query: PreparedMethodCell,
  reference: PreparedMethodCell,
): number {
  switch (method) {
    case "vision-features":
      return normalizedDotProduct(query.vision, reference.vision);
    case "perceptual-hash":
      return perceptualHashSimilarity(query.hash, reference.hash);
    case "edge-shape":
      return cosineDescriptorSimilarity(query.edges, reference.edges);
    case "orb":
      return orbSimilarity(query.orb, reference.orb);
  }
}

function bestByLabel(candidates: readonly MethodMatch[]): MethodMatch[] {
  const labels = new Map<string, MethodMatch>();
  for (const candidate of candidates) {
    const previous = labels.get(candidate.label);
    if (!previous || candidate.score > previous.score) labels.set(candidate.label, candidate);
  }
  return [...labels.values()].sort((left, right) => right.score - left.score);
}

export function classifyPreparedCell(
  query: PreparedMethodCell,
  references: readonly PreparedMethodCell[],
  method: ClassicalMethod,
  alternatives = 3,
): MethodMatch[] {
  if (!CLASSICAL_METHODS.includes(method)) {
    throw new Error(`Unsupported classical method: ${method}`);
  }
  if (!Number.isSafeInteger(alternatives) || alternatives <= 0) {
    throw new Error(`alternatives must be a positive integer, received ${alternatives}`);
  }
  if (references.length === 0) return [];
  if (method !== "hybrid-retrieval") {
    return bestByLabel(
      references.map((reference) => ({
        label: reference.label,
        score: score(method, query, reference),
        referenceId: reference.id,
        method,
      })),
    ).slice(0, alternatives);
  }

  // Stage 1 is deliberately cheap: retain only the strongest pHash reference
  // for each label. Stage 2 verifies the top labels against normalized pixels.
  const shortlist = bestByLabel(
    references.map((reference) => ({
      label: reference.label,
      score: perceptualHashSimilarity(query.hash, reference.hash),
      referenceId: reference.id,
      method,
    })),
  ).slice(0, 8);
  const shortlistedLabels = new Set(shortlist.map((candidate) => candidate.label));
  return bestByLabel(
    references
      .filter((reference) => shortlistedLabels.has(reference.label))
      .map((reference) => ({
        label: reference.label,
        score: cosineDescriptorSimilarity(query.pixelVector, reference.pixelVector),
        referenceId: reference.id,
        method,
      })),
  ).slice(0, alternatives);
}

export function classifyMethodCell(
  query: MethodCell,
  references: readonly PreparedMethodCell[],
  method: ClassicalMethod,
  alternatives = 3,
): MethodMatch[] {
  return classifyPreparedCell(prepareMethodCell(query), references, method, alternatives);
}
