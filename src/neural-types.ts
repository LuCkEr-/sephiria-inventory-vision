export interface NeuralCellSample {
  label: string;
  pixels: ArrayLike<number>;
}

export interface FeatureCellSample {
  label: string;
  features: readonly number[];
}

export interface NeuralPrediction {
  label: string;
  confidence: number;
  embedding: number[];
}

export interface TrainedTinyCnn {
  labels: string[];
  predict(pixels: ArrayLike<number>): NeuralPrediction;
  predictBatch(cells: readonly ArrayLike<number>[]): NeuralPrediction[];
  embed(pixels: ArrayLike<number>): number[];
  dispose(): void;
}

export interface TrainedSiameseEmbedding {
  embed(features: readonly number[]): number[];
  dispose(): void;
}

export interface TrainOptions {
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  seed?: number;
}

/** @internal */
export function validateTrainOptions(options: TrainOptions): void {
  for (const [name, value] of [
    ["epochs", options.epochs],
    ["batchSize", options.batchSize],
  ] as const) {
    if (value !== undefined && (!Number.isSafeInteger(value) || value <= 0)) {
      throw new Error(`${name} must be a positive integer, received ${value}`);
    }
  }
  if (
    options.learningRate !== undefined &&
    (!Number.isFinite(options.learningRate) || options.learningRate <= 0)
  ) {
    throw new Error(`learningRate must be positive and finite, received ${options.learningRate}`);
  }
  if (options.seed !== undefined && !Number.isSafeInteger(options.seed)) {
    throw new Error(`seed must be a safe integer, received ${options.seed}`);
  }
}
