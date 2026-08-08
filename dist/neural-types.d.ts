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
//# sourceMappingURL=neural-types.d.ts.map