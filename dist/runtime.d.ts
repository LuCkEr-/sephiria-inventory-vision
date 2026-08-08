import type { Mat } from "@techstark/opencv-js";
import { type RawImage } from "./image.js";
import { type LoadedTemplate } from "./matching.js";
import type { OpenCv } from "./opencv.js";
import type { CatalogTemplate, DetectOptions, ImageInput, SlotRect } from "./types.js";
export interface DecodedMat {
    mat: Mat;
    width: number;
    height: number;
    raw: RawImage;
}
export interface DetectionContext {
    imageRgb: Mat;
    slots: SlotRect[];
    coordinateScale: number;
    normalized: boolean;
    dispose(): void;
}
export declare function elapsedMilliseconds(startedAt: number): number;
export declare function assertUnitInterval(value: number, name: string): void;
export declare function validateDetectOptions(options: DetectOptions): void;
export declare function positiveInteger(value: number | undefined, fallback: number, name: string): number;
export declare function unitInterval(value: number | undefined, fallback: number, name: string): number;
export declare function normalizeLogicalHeight(value: number | false | undefined): number | false;
export declare function imageToRgbaMat(cv: OpenCv, input: ImageInput): Promise<DecodedMat>;
/**
 * Owns the RGB matrices and slot-localization state shared by every detector.
 * The decoded RGBA matrix is consumed; callers own only the returned context.
 */
export declare function prepareDetectionContext(cv: OpenCv, decoded: DecodedMat, slotTemplates: readonly LoadedTemplate[], options: DetectOptions): Promise<DetectionContext>;
export declare function loadTemplate(cv: OpenCv, root: string, metadata: CatalogTemplate): Promise<LoadedTemplate>;
export declare function loadTemplates(cv: OpenCv, root: string, metadata: readonly CatalogTemplate[]): Promise<LoadedTemplate[]>;
//# sourceMappingURL=runtime.d.ts.map