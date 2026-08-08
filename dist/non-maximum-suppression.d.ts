export interface ScoredBox {
    x: number;
    y: number;
    width: number;
    height: number;
    label: string;
    confidence: number;
}
/** Class-agnostic non-maximum suppression for dense-grid object detection experiments. */
export declare function nonMaximumSuppression(boxes: readonly ScoredBox[], overlapThreshold?: number, maximum?: number): ScoredBox[];
//# sourceMappingURL=non-maximum-suppression.d.ts.map