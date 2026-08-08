import type { FeatureCellSample, NeuralCellSample, TrainedSiameseEmbedding, TrainedTinyCnn, TrainOptions } from "./neural-types.js";
export { nonMaximumSuppression } from "./non-maximum-suppression.js";
export type { ScoredBox } from "./non-maximum-suppression.js";
export type { FeatureCellSample, NeuralCellSample, NeuralPrediction, TrainedSiameseEmbedding, TrainedTinyCnn, TrainOptions, } from "./neural-types.js";
/** Trains the small real convolutional classifier used by the method lab. */
export declare function trainTinyCnn(samples: readonly NeuralCellSample[], options?: TrainOptions): Promise<TrainedTinyCnn>;
/** Trains a genuine shared-weight twin network with contrastive loss. */
export declare function trainSiameseEmbedding(samples: readonly FeatureCellSample[], options?: TrainOptions): Promise<TrainedSiameseEmbedding>;
//# sourceMappingURL=neural-methods.d.ts.map