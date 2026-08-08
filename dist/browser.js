import { cosineSimilarity } from "./vision-features.js";
export { cosineSimilarity, extractVisionFeatures, normalizedDotProduct, VISION_FEATURE_SIZE, } from "./vision-features.js";
/**
 * Ranks unique labels by the best cosine similarity among their references.
 * Multiple rotations or render scales can therefore share the same label.
 */
export function rankVisionFeatures(query, references, limit = 3) {
    if (!Number.isSafeInteger(limit) || limit < 0) {
        throw new Error(`Vision feature ranking requires a non-negative integer limit, received ${limit}`);
    }
    if (limit === 0 || references.length === 0)
        return [];
    const bestByLabel = new Map();
    for (const reference of references) {
        const score = cosineSimilarity(query, reference.features);
        const previous = bestByLabel.get(reference.label);
        if (previous === undefined || score > previous)
            bestByLabel.set(reference.label, score);
    }
    return [...bestByLabel]
        .map(([label, score]) => ({ label, score }))
        .sort((left, right) => right.score - left.score || left.label.localeCompare(right.label))
        .slice(0, limit);
}
//# sourceMappingURL=browser.js.map