function intersectionOverUnion(left, right) {
    const x1 = Math.max(left.x, right.x);
    const y1 = Math.max(left.y, right.y);
    const x2 = Math.min(left.x + left.width, right.x + right.width);
    const y2 = Math.min(left.y + left.height, right.y + right.height);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = left.width * left.height + right.width * right.height - intersection;
    return intersection / Math.max(1, union);
}
/** Class-agnostic non-maximum suppression for dense-grid object detection experiments. */
export function nonMaximumSuppression(boxes, overlapThreshold = 0.35, maximum = 100) {
    if (!Number.isFinite(overlapThreshold) || overlapThreshold < 0 || overlapThreshold > 1) {
        throw new Error(`overlapThreshold must be between 0 and 1, received ${overlapThreshold}`);
    }
    if (!Number.isSafeInteger(maximum) || maximum < 0) {
        throw new Error(`maximum must be a non-negative integer, received ${maximum}`);
    }
    boxes.forEach((box, index) => {
        if (!Number.isFinite(box.x) || !Number.isFinite(box.y)) {
            throw new Error(`boxes[${index}] coordinates must be finite`);
        }
        if (!Number.isFinite(box.width) ||
            box.width <= 0 ||
            !Number.isFinite(box.height) ||
            box.height <= 0) {
            throw new Error(`boxes[${index}] dimensions must be positive and finite`);
        }
        if (typeof box.label !== "string" || box.label.length === 0) {
            throw new Error(`boxes[${index}].label must be a non-empty string`);
        }
        if (!Number.isFinite(box.confidence) || box.confidence < 0 || box.confidence > 1) {
            throw new Error(`boxes[${index}].confidence must be between 0 and 1`);
        }
    });
    if (maximum === 0)
        return [];
    const kept = [];
    for (const box of [...boxes].sort((left, right) => right.confidence - left.confidence)) {
        if (kept.every((candidate) => intersectionOverUnion(box, candidate) < overlapThreshold)) {
            kept.push(box);
            if (kept.length >= maximum)
                break;
        }
    }
    return kept;
}
//# sourceMappingURL=non-maximum-suppression.js.map