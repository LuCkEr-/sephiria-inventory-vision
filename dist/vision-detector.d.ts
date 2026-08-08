import type { DetectOptions, DetectionResult, ImageInput, InventoryCatalog, VisionDetectorOptions, VisionModel } from "./types.js";
export declare class InventoryVisionDetector {
    #private;
    readonly catalog: InventoryCatalog;
    readonly model: VisionModel;
    private constructor();
    static create(options?: VisionDetectorOptions): Promise<InventoryVisionDetector>;
    detect(input: ImageInput, options?: DetectOptions): Promise<DetectionResult>;
    dispose(): void;
}
export declare function createInventoryVisionDetector(options?: VisionDetectorOptions): Promise<InventoryVisionDetector>;
//# sourceMappingURL=vision-detector.d.ts.map