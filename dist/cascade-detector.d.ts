import type { CascadeDetectOptions, CascadeDetectionResult, CascadeDetectorOptions, ImageInput, InventoryCatalog, VisionModel } from "./types.js";
/**
 * Fast structural/color classification with selective extracted-game-asset
 * verification. Uncertain cells use the complete catalog; otherwise accepted
 * non-empty identities are checked against only their predicted game asset.
 */
export declare class InventoryCascadeDetector {
    #private;
    readonly catalog: InventoryCatalog;
    readonly model: VisionModel;
    private constructor();
    static create(options?: CascadeDetectorOptions): Promise<InventoryCascadeDetector>;
    detect(input: ImageInput, options?: CascadeDetectOptions): Promise<CascadeDetectionResult>;
    dispose(): void;
}
export declare function createInventoryCascadeDetector(options?: CascadeDetectorOptions): Promise<InventoryCascadeDetector>;
//# sourceMappingURL=cascade-detector.d.ts.map