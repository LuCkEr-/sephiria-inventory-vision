import type { Mat } from "@techstark/opencv-js";
import type { SlotCandidate } from "./grid-locator.js";
import type { OpenCv } from "./opencv.js";
import type { CatalogTemplate, MatchAlternative, SlotRect } from "./types.js";
export { locateInventoryGrid } from "./grid-locator.js";
export type { LocatedGrid, SlotCandidate } from "./grid-locator.js";
export interface LoadedTemplate {
    metadata: CatalogTemplate;
    rgb: Mat;
    mask: Mat;
    width: number;
    height: number;
}
interface ScaledTemplate {
    rgb: Mat;
    mask: Mat;
    width: number;
    height: number;
    owned: boolean;
}
export declare function rgbaMatToTemplate(cv: OpenCv, rgba: Mat, metadata: CatalogTemplate): LoadedTemplate;
export declare function scaleTemplate(cv: OpenCv, template: LoadedTemplate, scale: number): ScaledTemplate;
export declare function locateSlots(cv: OpenCv, imageRgb: Mat, templates: readonly LoadedTemplate[], options: {
    scales: number[];
    threshold: number;
    borderWidth: number;
    maxSlots: number;
}): SlotCandidate[];
export declare function matchItemsInSlot(cv: OpenCv, imageRgb: Mat, slot: SlotRect, templates: readonly LoadedTemplate[], nativeSlotSize: number, alternativeCount: number): MatchAlternative[];
export declare function disposeTemplate(template: LoadedTemplate): void;
//# sourceMappingURL=matching.d.ts.map