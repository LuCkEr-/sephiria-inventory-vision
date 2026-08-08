import type * as OpenCvNamespace from "@techstark/opencv-js";
export type OpenCv = typeof OpenCvNamespace;
export interface OpenCvHandle {
    cv: OpenCv;
}
export declare function getOpenCv(): Promise<OpenCvHandle>;
//# sourceMappingURL=opencv.d.ts.map