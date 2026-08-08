import { type OrbDescriptor, type PerceptualHash } from "./method-features.js";
export type ClassicalMethod = "vision-features" | "perceptual-hash" | "edge-shape" | "orb" | "hybrid-retrieval";
export declare const CLASSICAL_METHODS: readonly ClassicalMethod[];
export interface MethodCell {
    id: string;
    label: string;
    pixels: ArrayLike<number>;
}
export interface PreparedMethodCell {
    id: string;
    label: string;
    pixels: number[];
    vision: number[];
    hash: PerceptualHash;
    edges: number[];
    orb: OrbDescriptor[];
    pixelVector: number[];
}
export interface MethodMatch {
    label: string;
    score: number;
    referenceId: string;
    method: ClassicalMethod;
}
export declare function prepareMethodCell(cell: MethodCell): PreparedMethodCell;
export declare function classifyPreparedCell(query: PreparedMethodCell, references: readonly PreparedMethodCell[], method: ClassicalMethod, alternatives?: number): MethodMatch[];
export declare function classifyMethodCell(query: MethodCell, references: readonly PreparedMethodCell[], method: ClassicalMethod, alternatives?: number): MethodMatch[];
//# sourceMappingURL=method-lab.d.ts.map