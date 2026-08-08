export { cosineSimilarity, extractVisionFeatures, normalizedDotProduct, VISION_FEATURE_SIZE, } from "./vision-features.js";
export interface VisionFeatureReference<Label extends string = string> {
    label: Label;
    features: readonly number[];
}
export interface VisionFeatureMatch<Label extends string = string> {
    label: Label;
    score: number;
}
/**
 * Ranks unique labels by the best cosine similarity among their references.
 * Multiple rotations or render scales can therefore share the same label.
 */
export declare function rankVisionFeatures<Label extends string>(query: readonly number[], references: readonly VisionFeatureReference<Label>[], limit?: number): VisionFeatureMatch<Label>[];
//# sourceMappingURL=browser.d.ts.map