const STRUCTURAL_FEATURE_SIZE = 272;
const PIXEL_FEATURE_SIZE = 16 * 16 * 3;
export const VISION_FEATURE_SIZE = STRUCTURAL_FEATURE_SIZE + PIXEL_FEATURE_SIZE;

function normalize(vector: number[]): number[] {
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => value / magnitude);
}

function extractPixelSignature(
  pixels: ArrayLike<number>,
  width: number,
  channels: number,
): number[] {
  const values: number[] = [];
  for (let outputIndex = 0; outputIndex < PIXEL_FEATURE_SIZE; outputIndex += 1) {
    const channel = outputIndex % 3;
    const cellIndex = Math.floor(outputIndex / 3);
    const cellX = cellIndex % 16;
    const cellY = Math.floor(cellIndex / 16);
    let value = 0;
    for (let dy = 0; dy < 2; dy += 1) {
      for (let dx = 0; dx < 2; dx += 1) {
        const x = cellX * 2 + dx;
        const y = cellY * 2 + dy;
        value += pixels[(y * width + x) * channels + channel] as number;
      }
    }
    values.push(value / (4 * 255));
  }

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return normalize(values.map((value) => value - mean));
}

export function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length) {
    throw new Error(
      `Cosine similarity requires equal vector lengths, received ${left.length} and ${right.length}`,
    );
  }
  let dot = 0;
  let leftMagnitudeSquared = 0;
  let rightMagnitudeSquared = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? Number.NaN;
    const rightValue = right[index] ?? Number.NaN;
    if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) {
      throw new Error(`Cosine similarity requires finite vectors; invalid value at index ${index}`);
    }
    dot += leftValue * rightValue;
    leftMagnitudeSquared += leftValue * leftValue;
    rightMagnitudeSquared += rightValue * rightValue;
  }
  const denominator = Math.sqrt(leftMagnitudeSquared * rightMagnitudeSquared);
  return denominator === 0 ? 0 : Math.max(-1, Math.min(1, dot / denominator));
}

/** Fast dot product for descriptor vectors already validated as unit length. */
export function normalizedDotProduct(left: readonly number[], right: readonly number[]): number {
  let score = 0;
  for (let index = 0; index < left.length; index += 1) {
    score += (left[index] as number) * (right[index] as number);
  }
  return Math.max(-1, Math.min(1, score));
}

interface VisionChannels {
  gray: Float32Array;
  red: Float32Array;
  green: Float32Array;
  blue: Float32Array;
  grayMean: number;
  grayStd: number;
}

function validateCellShape(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels: number,
): void {
  if (width !== 32 || height !== 32) {
    throw new Error(`Vision features require a 32x32 cell, received ${width}x${height}`);
  }
  if (!Number.isSafeInteger(channels) || channels < 3) {
    throw new Error(`Vision features require RGB pixels, received ${channels} channels`);
  }
  const expectedValues = width * height * channels;
  if (pixels.length < expectedValues) {
    throw new Error(
      `Vision features require ${expectedValues} pixel values, received ${pixels.length}`,
    );
  }
}

function byteValue(value: number | undefined, pixelIndex: number): number {
  if (!Number.isInteger(value) || (value ?? -1) < 0 || (value ?? 256) > 255) {
    throw new Error(`Vision features require byte-valued RGB pixels; invalid pixel ${pixelIndex}`);
  }
  return value as number;
}

function extractChannels(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels: number,
): VisionChannels {
  const pixelCount = width * height;
  const gray = new Float32Array(pixelCount);
  const red = new Float32Array(pixelCount);
  const green = new Float32Array(pixelCount);
  const blue = new Float32Array(pixelCount);
  let grayMean = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * channels;
    red[index] = byteValue(pixels[offset], index) / 255;
    green[index] = byteValue(pixels[offset + 1], index) / 255;
    blue[index] = byteValue(pixels[offset + 2], index) / 255;
    gray[index] =
      (red[index] as number) * 0.299 +
      (green[index] as number) * 0.587 +
      (blue[index] as number) * 0.114;
    grayMean += gray[index] as number;
  }
  grayMean /= pixelCount;
  let variance = 0;
  for (const value of gray) variance += (value - grayMean) ** 2;
  return {
    gray,
    red,
    green,
    blue,
    grayMean,
    grayStd: Math.sqrt(variance / pixelCount) || 1,
  };
}

function appendColorFeatures(
  features: number[],
  channels: Pick<VisionChannels, "red" | "green" | "blue">,
  width: number,
  height: number,
): void {
  const colorCells = 4;
  const cellWidth = width / colorCells;
  const cellHeight = height / colorCells;
  for (let cellY = 0; cellY < colorCells; cellY += 1) {
    for (let cellX = 0; cellX < colorCells; cellX += 1) {
      let red = 0;
      let green = 0;
      let blue = 0;
      let count = 0;
      for (let y = cellY * cellHeight; y < (cellY + 1) * cellHeight; y += 1) {
        for (let x = cellX * cellWidth; x < (cellX + 1) * cellWidth; x += 1) {
          const index = y * width + x;
          red += channels.red[index] as number;
          green += channels.green[index] as number;
          blue += channels.blue[index] as number;
          count += 1;
        }
      }
      const sum = red + green + blue || 1;
      features.push(
        (red / sum) * 0.7,
        (green / sum) * 0.7,
        (blue / sum) * 0.7,
        ((red + green + blue) / count / 3) * 0.25,
      );
    }
  }
}

function appendHogFeatures(features: number[], gray: Float32Array, width: number, height: number) {
  const cells = 4;
  const bins = 9;
  const histograms = Array.from({ length: cells * cells }, () => new Float32Array(bins));
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const gx = (gray[y * width + x + 1] as number) - (gray[y * width + x - 1] as number);
      const gy = (gray[(y + 1) * width + x] as number) - (gray[(y - 1) * width + x] as number);
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      let angle = Math.atan2(gy, gx);
      if (angle < 0) angle += Math.PI;
      if (angle >= Math.PI) angle -= Math.PI;
      const bin = Math.min(bins - 1, Math.floor((angle / Math.PI) * bins));
      const cellX = Math.min(cells - 1, Math.floor((x / width) * cells));
      const cellY = Math.min(cells - 1, Math.floor((y / height) * cells));
      const histogram = histograms[cellY * cells + cellX] as Float32Array;
      histogram[bin] = (histogram[bin] as number) + magnitude;
    }
  }
  for (const histogram of histograms) {
    for (const value of normalize([...histogram])) features.push(value * 1.5);
  }
}

function appendLuminanceFeatures(
  features: number[],
  gray: Float32Array,
  grayMean: number,
  grayStd: number,
  width: number,
): void {
  const cells = 8;
  const cellSize = 4;
  for (let cellY = 0; cellY < cells; cellY += 1) {
    for (let cellX = 0; cellX < cells; cellX += 1) {
      let value = 0;
      for (let y = cellY * cellSize; y < (cellY + 1) * cellSize; y += 1) {
        for (let x = cellX * cellSize; x < (cellX + 1) * cellSize; x += 1) {
          value += ((gray[y * width + x] as number) - grayMean) / grayStd;
        }
      }
      features.push((value / (cellSize * cellSize)) * 0.35);
    }
  }
}

/**
 * Extracts a compact classical-machine-vision descriptor from a normalized
 * 32x32 RGB inventory cell. It combines HOG, spatial chromaticity,
 * contrast-normalized luminance, and a centered 16x16 RGB signature. The
 * structural features preserve cross-capture recognition while the pixel
 * signature separates icons that share the same inventory-slot background.
 */
export function extractVisionFeatures(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  channels = 3,
): number[] {
  validateCellShape(pixels, width, height, channels);
  const extracted = extractChannels(pixels, width, height, channels);
  const features: number[] = [];
  appendColorFeatures(features, extracted, width, height);
  appendHogFeatures(features, extracted.gray, width, height);
  appendLuminanceFeatures(features, extracted.gray, extracted.grayMean, extracted.grayStd, width);

  if (features.length !== STRUCTURAL_FEATURE_SIZE) {
    throw new Error(`Unexpected structural feature length ${features.length}`);
  }
  const structural = normalize(features);
  const pixelSignature = extractPixelSignature(pixels, width, channels);
  return normalize([...structural, ...pixelSignature.map((value) => value * 0.5)]);
}
