import { cosineSimilarity } from "./vision-features.js";

export interface PerceptualHash {
  words: [number, number];
}

export interface OrbDescriptor {
  x: number;
  y: number;
  angle: number;
  words: [number, number, number, number];
}

const DCT_COSINES = Array.from({ length: 8 }, (_, frequency) =>
  Float64Array.from({ length: 32 }, (_, position) =>
    Math.cos(((2 * position + 1) * frequency * Math.PI) / 64),
  ),
);

function assertCell(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels: number,
): void {
  if (width !== 32 || height !== 32 || !Number.isSafeInteger(channels) || channels < 3) {
    throw new Error(`Expected a 32x32 RGB cell, received ${width}x${height}x${channels}`);
  }
  const expectedLength = width * height * channels;
  if (pixels.length < expectedLength) {
    throw new Error(`Expected ${expectedLength} pixel values, received ${pixels.length}`);
  }
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * channels;
    for (let channel = 0; channel < 3; channel += 1) {
      const value = pixels[offset + channel];
      if (!Number.isInteger(value) || (value ?? -1) < 0 || (value ?? 256) > 255) {
        throw new Error(`Expected byte-valued RGB pixels; invalid pixel ${index}`);
      }
    }
  }
}

function grayscale(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels: number,
): Float32Array {
  assertCell(pixels, width, height, channels);
  const result = new Float32Array(width * height);
  for (let index = 0; index < result.length; index += 1) {
    const offset = index * channels;
    result[index] =
      (pixels[offset] as number) * 0.299 +
      (pixels[offset + 1] as number) * 0.587 +
      (pixels[offset + 2] as number) * 0.114;
  }
  return result;
}

function normalize(values: readonly number[]): number[] {
  const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
  const centered = values.map((value) => value - mean);
  const magnitude = Math.sqrt(centered.reduce((sum, value) => sum + value * value, 0)) || 1;
  return centered.map((value) => value / magnitude);
}

function popcount(value: number): number {
  let bits = value >>> 0;
  bits -= (bits >>> 1) & 0x55555555;
  bits = (bits & 0x33333333) + ((bits >>> 2) & 0x33333333);
  return (((bits + (bits >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

/** Rotation-sensitive 64-bit DCT perceptual hash. Rotated references make it rotation-aware. */
export function extractPerceptualHash(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels = 3,
): PerceptualHash {
  const gray = grayscale(pixels, width, height, channels);
  const coefficients: number[] = [];
  for (let v = 0; v < 8; v += 1) {
    const verticalCosines = DCT_COSINES[v] as Float64Array;
    for (let u = 0; u < 8; u += 1) {
      const horizontalCosines = DCT_COSINES[u] as Float64Array;
      let value = 0;
      for (let y = 0; y < 32; y += 1) {
        const cy = verticalCosines[y] as number;
        for (let x = 0; x < 32; x += 1) {
          value += (gray[y * 32 + x] as number) * (horizontalCosines[x] as number) * cy;
        }
      }
      coefficients.push(value);
    }
  }
  const sorted = coefficients.slice(1).sort((left, right) => left - right);
  const median = sorted[Math.floor(sorted.length / 2)] as number;
  const words: [number, number] = [0, 0];
  coefficients.forEach((value, index) => {
    if (value > median) words[index >>> 5] = (words[index >>> 5] as number) | (1 << (index & 31));
  });
  return { words };
}

export function perceptualHashSimilarity(left: PerceptualHash, right: PerceptualHash): number {
  const distance =
    popcount((left.words[0] ^ right.words[0]) >>> 0) +
    popcount((left.words[1] ^ right.words[1]) >>> 0);
  return 1 - distance / 64;
}

/** Sobel contour map plus spatial orientation histograms. */
export function extractEdgeShapeFeatures(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels = 3,
): number[] {
  const gray = grayscale(pixels, width, height, channels);
  const pooled = new Float32Array(64);
  const histograms = Array.from({ length: 16 }, () => new Float32Array(8));
  for (let y = 1; y < 31; y += 1) {
    for (let x = 1; x < 31; x += 1) {
      const gx = (gray[y * 32 + x + 1] as number) - (gray[y * 32 + x - 1] as number);
      const gy = (gray[(y + 1) * 32 + x] as number) - (gray[(y - 1) * 32 + x] as number);
      const magnitude = Math.hypot(gx, gy);
      const pooledIndex = Math.floor(y / 4) * 8 + Math.floor(x / 4);
      pooled[pooledIndex] = (pooled[pooledIndex] as number) + magnitude;
      let angle = Math.atan2(gy, gx) + Math.PI;
      if (angle >= Math.PI * 2) angle -= Math.PI * 2;
      const bin = Math.min(7, Math.floor((angle / (Math.PI * 2)) * 8));
      const histogram = histograms[Math.floor(y / 8) * 4 + Math.floor(x / 8)] as Float32Array;
      histogram[bin] = (histogram[bin] as number) + magnitude;
    }
  }
  return normalize([...pooled, ...histograms.flatMap((histogram) => [...histogram])]);
}

export function cosineDescriptorSimilarity(
  left: readonly number[],
  right: readonly number[],
): number {
  return cosineSimilarity(left, right);
}

type BriefPair = readonly [number, number, number, number];

function briefPairs(): BriefPair[] {
  let state = 0x5e71f1a9;
  const random = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0xffffffff;
  };
  return Array.from(
    { length: 128 },
    () =>
      [
        Math.round(random() * 12 - 6),
        Math.round(random() * 12 - 6),
        Math.round(random() * 12 - 6),
        Math.round(random() * 12 - 6),
      ] as const,
  );
}

const BRIEF_PAIRS = briefPairs();

function sample(gray: Float32Array, x: number, y: number): number {
  const sx = Math.max(0, Math.min(31, Math.round(x)));
  const sy = Math.max(0, Math.min(31, Math.round(y)));
  return gray[sy * 32 + sx] as number;
}

/** A compact ORB-style FAST/gradient keypoint + oriented BRIEF descriptor. */
export function extractOrbDescriptors(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels = 3,
  maximum = 32,
): OrbDescriptor[] {
  if (!Number.isSafeInteger(maximum) || maximum < 0) {
    throw new Error(`maximum must be a non-negative integer, received ${maximum}`);
  }
  if (maximum === 0) return [];
  const gray = grayscale(pixels, width, height, channels);
  const candidates: { x: number; y: number; score: number }[] = [];
  for (let y = 7; y < 25; y += 1) {
    for (let x = 7; x < 25; x += 1) {
      const gx = (gray[y * 32 + x + 1] as number) - (gray[y * 32 + x - 1] as number);
      const gy = (gray[(y + 1) * 32 + x] as number) - (gray[(y - 1) * 32 + x] as number);
      const diagonal = Math.abs(
        (gray[(y + 1) * 32 + x + 1] as number) - (gray[(y - 1) * 32 + x - 1] as number),
      );
      const score = Math.abs(gx) + Math.abs(gy) + diagonal * 0.5;
      if (score >= 24) candidates.push({ x, y, score });
    }
  }
  candidates.sort((left, right) => right.score - left.score);
  const selected: typeof candidates = [];
  for (const candidate of candidates) {
    if (
      selected.every(
        (keypoint) => Math.hypot(candidate.x - keypoint.x, candidate.y - keypoint.y) >= 3,
      )
    ) {
      selected.push(candidate);
      if (selected.length >= maximum) break;
    }
  }
  return selected.map(({ x, y }) => {
    let mx = 0;
    let my = 0;
    for (let dy = -4; dy <= 4; dy += 1) {
      for (let dx = -4; dx <= 4; dx += 1) {
        const intensity = sample(gray, x + dx, y + dy);
        mx += dx * intensity;
        my += dy * intensity;
      }
    }
    const angle = Math.atan2(my, mx);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const words: [number, number, number, number] = [0, 0, 0, 0];
    BRIEF_PAIRS.forEach(([x1, y1, x2, y2], index) => {
      const ax = x + x1 * cosine - y1 * sine;
      const ay = y + x1 * sine + y1 * cosine;
      const bx = x + x2 * cosine - y2 * sine;
      const by = y + x2 * sine + y2 * cosine;
      if (sample(gray, ax, ay) < sample(gray, bx, by)) {
        words[index >>> 5] = (words[index >>> 5] as number) | (1 << (index & 31));
      }
    });
    return { x, y, angle, words };
  });
}

function descriptorDistance(left: OrbDescriptor, right: OrbDescriptor): number {
  let distance = 0;
  for (let index = 0; index < 4; index += 1) {
    distance += popcount(((left.words[index] as number) ^ (right.words[index] as number)) >>> 0);
  }
  return distance;
}

export function orbSimilarity(
  left: readonly OrbDescriptor[],
  right: readonly OrbDescriptor[],
): number {
  if (left.length === 0 || right.length === 0) return 0;
  const smaller = left.length <= right.length ? left : right;
  const larger = left.length <= right.length ? right : left;
  const qualities = smaller
    .map((descriptor) => {
      let best = 128;
      for (const candidate of larger)
        best = Math.min(best, descriptorDistance(descriptor, candidate));
      return Math.max(0, 1 - best / 80);
    })
    .sort((a, b) => b - a);
  const keep = Math.max(1, Math.ceil(qualities.length * 0.6));
  return qualities.slice(0, keep).reduce((sum, value) => sum + value, 0) / keep;
}

/** Low-resolution RGB vector for exact pixel verification after retrieval. */
function averagePixelBlock(
  pixels: ArrayLike<number>,
  cellX: number,
  cellY: number,
  channel: number,
  channels: number,
): number {
  let value = 0;
  for (let offset = 0; offset < 4; offset += 1) {
    const x = cellX * 2 + (offset & 1);
    const y = cellY * 2 + (offset >>> 1);
    value += pixels[(y * 32 + x) * channels + channel] as number;
  }
  return value / (4 * 255);
}

export function extractPixelVector(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels = 3,
): number[] {
  assertCell(pixels, width, height, channels);
  const values: number[] = [];
  for (let cellY = 0; cellY < 16; cellY += 1) {
    for (let cellX = 0; cellX < 16; cellX += 1) {
      for (let channel = 0; channel < 3; channel += 1) {
        values.push(averagePixelBlock(pixels, cellX, cellY, channel, channels));
      }
    }
  }
  return normalize(values);
}
