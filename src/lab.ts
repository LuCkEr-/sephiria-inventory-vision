export {
  cosineDescriptorSimilarity,
  extractEdgeShapeFeatures,
  extractOrbDescriptors,
  extractPerceptualHash,
  extractPixelVector,
  orbSimilarity,
  perceptualHashSimilarity,
} from "./method-features.js";
export {
  CLASSICAL_METHODS,
  classifyMethodCell,
  classifyPreparedCell,
  prepareMethodCell,
} from "./method-lab.js";
export { nonMaximumSuppression, trainSiameseEmbedding, trainTinyCnn } from "./neural-methods.js";
export type { OrbDescriptor, PerceptualHash } from "./method-features.js";
export type { ClassicalMethod, MethodCell, MethodMatch, PreparedMethodCell } from "./method-lab.js";
export type {
  FeatureCellSample,
  NeuralCellSample,
  NeuralPrediction,
  ScoredBox,
  TrainedSiameseEmbedding,
  TrainedTinyCnn,
  TrainOptions,
} from "./neural-methods.js";
