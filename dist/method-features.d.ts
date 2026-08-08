export interface PerceptualHash {
    words: [number, number];
}
export interface OrbDescriptor {
    x: number;
    y: number;
    angle: number;
    words: [number, number, number, number];
}
/** Rotation-sensitive 64-bit DCT perceptual hash. Rotated references make it rotation-aware. */
export declare function extractPerceptualHash(pixels: ArrayLike<number>, width: number, height: number, channels?: number): PerceptualHash;
export declare function perceptualHashSimilarity(left: PerceptualHash, right: PerceptualHash): number;
/** Sobel contour map plus spatial orientation histograms. */
export declare function extractEdgeShapeFeatures(pixels: ArrayLike<number>, width: number, height: number, channels?: number): number[];
export declare function cosineDescriptorSimilarity(left: readonly number[], right: readonly number[]): number;
/** A compact ORB-style FAST/gradient keypoint + oriented BRIEF descriptor. */
export declare function extractOrbDescriptors(pixels: ArrayLike<number>, width: number, height: number, channels?: number, maximum?: number): OrbDescriptor[];
export declare function orbSimilarity(left: readonly OrbDescriptor[], right: readonly OrbDescriptor[]): number;
export declare function extractPixelVector(pixels: ArrayLike<number>, width: number, height: number, channels?: number): number[];
//# sourceMappingURL=method-features.d.ts.map