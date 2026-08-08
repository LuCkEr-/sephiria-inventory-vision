import type { DetectOptions, DetectionResult, DetectorOptions, ImageInput, InventoryCatalog } from "./types.js";
export declare class InventoryDetector {
    #private;
    readonly catalog: InventoryCatalog;
    private constructor();
    static create(options?: DetectorOptions): Promise<InventoryDetector>;
    detect(input: ImageInput, options?: DetectOptions): Promise<DetectionResult>;
    dispose(): void;
}
export declare function createInventoryDetector(options?: DetectorOptions): Promise<InventoryDetector>;
//# sourceMappingURL=detector.d.ts.map