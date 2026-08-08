export declare const VISION_FEATURE_SIZE: number;
export declare function cosineSimilarity(left: readonly number[], right: readonly number[]): number;
/** Fast dot product for descriptor vectors already validated as unit length. */
export declare function normalizedDotProduct(left: readonly number[], right: readonly number[]): number;
/**
 * Extracts a compact classical-machine-vision descriptor from a normalized
 * 32x32 RGB inventory cell. It combines HOG, spatial chromaticity,
 * contrast-normalized luminance, and a centered 16x16 RGB signature. The
 * structural features preserve cross-capture recognition while the pixel
 * signature separates icons that share the same inventory-slot background.
 */
export declare function extractVisionFeatures(pixels: ArrayLike<number>, width: number, height: number, channels?: number): number[];
//# sourceMappingURL=vision-features.d.ts.map