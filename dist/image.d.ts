import type { ImageInput } from "./types.js";
export interface RawImage {
    data: Uint8Array;
    width: number;
    height: number;
    channels: 4;
}
export declare function decodeImage(input: ImageInput): Promise<RawImage>;
export declare function resizeRawImage(image: RawImage, height: number): Promise<RawImage>;
//# sourceMappingURL=image.d.ts.map