import type { Mat } from "@techstark/opencv-js";
import type { OpenCv } from "./opencv.js";
import type { CatalogTemplate, SlotRect } from "./types.js";
interface GridTemplate {
    metadata: CatalogTemplate;
    rgb: Mat;
    mask: Mat;
    width: number;
    height: number;
}
export interface SlotCandidate extends SlotRect {
    localizationConfidence: number;
    templateId: string;
}
export interface LocatedGrid {
    slots: SlotCandidate[];
    support: number;
    confidence: number;
    origin: {
        x: number;
        y: number;
    };
    pitch: number;
}
export interface GridLocationOptions {
    rows: number;
    columns: number;
    threshold: number;
    minSupport: number;
    tolerance?: number;
    preferredOriginY?: number;
}
export declare function locateInventoryGrid(cv: OpenCv, imageRgb: Mat, template: GridTemplate, options: GridLocationOptions): LocatedGrid | null;
export {};
//# sourceMappingURL=grid-locator.d.ts.map