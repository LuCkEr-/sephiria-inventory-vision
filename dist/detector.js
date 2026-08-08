import { loadCatalog } from "./catalog.js";
import { ResourceLifecycle } from "./lifecycle.js";
import { disposeTemplate, matchItemsInSlot } from "./matching.js";
import { getOpenCv } from "./opencv.js";
import { elapsedMilliseconds, imageToRgbaMat, loadTemplates, positiveInteger, prepareDetectionContext, unitInterval, validateDetectOptions, } from "./runtime.js";
export class InventoryDetector {
    catalog;
    #cv;
    #items;
    #itemsByName;
    #slotTemplates;
    #lifecycle;
    constructor(cv, catalog, items, slotTemplates) {
        this.#cv = cv;
        this.catalog = catalog;
        this.#items = items;
        const itemsByName = new Map();
        for (const item of items) {
            const namedItems = itemsByName.get(item.metadata.name) ?? [];
            namedItems.push(item);
            itemsByName.set(item.metadata.name, namedItems);
        }
        this.#itemsByName = itemsByName;
        this.#slotTemplates = slotTemplates;
        this.#lifecycle = new ResourceLifecycle("InventoryDetector", () => {
            this.#items.forEach(disposeTemplate);
            this.#slotTemplates.forEach(disposeTemplate);
        });
    }
    static async create(options = {}) {
        return InventoryDetector.#create(options, true);
    }
    /** @internal Cascade verification supplies explicit slots and does not need slot-frame templates. */
    static async createForExplicitSlots(options = {}) {
        return InventoryDetector.#create(options, false);
    }
    static async #create(options, includeSlotTemplates) {
        const { cv } = await getOpenCv();
        const loaded = await loadCatalog(options.catalogPath);
        let items = [];
        try {
            items = await loadTemplates(cv, loaded.root, loaded.catalog.items);
            const slots = includeSlotTemplates
                ? await loadTemplates(cv, loaded.root, loaded.catalog.slotTemplates)
                : [];
            return new InventoryDetector(cv, loaded.catalog, items, slots);
        }
        catch (error) {
            items.forEach(disposeTemplate);
            throw error;
        }
    }
    async detect(input, options = {}) {
        return this.#detect(input, options);
    }
    /** @internal Selectively verifies one expected item name per explicit slot. */
    async detectWithCandidateVerification(input, options, candidateNames) {
        if (options.slots?.length !== candidateNames.length) {
            throw new Error("Candidate verification requires one candidate entry per explicit slot");
        }
        return this.#detect(input, options, candidateNames);
    }
    async #detect(input, options, candidateNames) {
        const leaveLifecycle = this.#lifecycle.enter();
        try {
            validateDetectOptions(options);
            const totalStarted = performance.now();
            const decodeStarted = performance.now();
            const decoded = await imageToRgbaMat(this.#cv, input);
            const decodeTime = elapsedMilliseconds(decodeStarted);
            const locateStarted = performance.now();
            const context = await prepareDetectionContext(this.#cv, decoded, this.#slotTemplates, options);
            const locateTime = elapsedMilliseconds(locateStarted);
            const { coordinateScale, imageRgb, normalized, slots } = context;
            try {
                const matchStarted = performance.now();
                const itemThreshold = unitInterval(options.itemThreshold, 0.85, "itemThreshold");
                const emptySlotThreshold = unitInterval(options.emptySlotThreshold, 0.97, "emptySlotThreshold");
                const alternativeCount = positiveInteger(options.alternatives, 3, "alternatives");
                const workingSlots = slots.map((slot, index) => {
                    const candidateName = candidateNames?.[index];
                    let alternatives = candidateName
                        ? matchItemsInSlot(this.#cv, imageRgb, slot, this.#itemsByName.get(candidateName) ?? [], this.catalog.nativeSlotSize, alternativeCount)
                        : [];
                    if (!candidateName || !alternatives[0] || alternatives[0].confidence < itemThreshold) {
                        alternatives = matchItemsInSlot(this.#cv, imageRgb, slot, this.#items, this.catalog.nativeSlotSize, alternativeCount);
                    }
                    const best = alternatives[0];
                    const confidentlyEmpty = normalized &&
                        (slot.localizationConfidence ?? 0) >= emptySlotThreshold &&
                        (!best || best.confidence < itemThreshold);
                    return {
                        ...slot,
                        item: confidentlyEmpty ? null : best && best.confidence >= itemThreshold ? best : null,
                        alternatives: confidentlyEmpty ? [] : alternatives,
                    };
                });
                const detectedSlots = normalized
                    ? workingSlots.map((slot) => ({
                        ...slot,
                        x: Math.round(slot.x * coordinateScale),
                        y: Math.round(slot.y * coordinateScale),
                        width: Math.round(slot.width * coordinateScale),
                        height: Math.round(slot.height * coordinateScale),
                        scale: coordinateScale,
                        item: slot.item
                            ? {
                                ...slot.item,
                                offset: {
                                    x: Math.round(slot.item.offset.x * coordinateScale),
                                    y: Math.round(slot.item.offset.y * coordinateScale),
                                },
                            }
                            : null,
                        alternatives: slot.alternatives.map((alternative) => ({
                            ...alternative,
                            offset: {
                                x: Math.round(alternative.offset.x * coordinateScale),
                                y: Math.round(alternative.offset.y * coordinateScale),
                            },
                        })),
                    }))
                    : workingSlots;
                const matchTime = elapsedMilliseconds(matchStarted);
                return {
                    image: { width: decoded.width, height: decoded.height },
                    slots: detectedSlots,
                    matchedItems: detectedSlots.filter((slot) => slot.item !== null),
                    catalogSize: this.#items.length,
                    timingsMs: {
                        decode: decodeTime,
                        locateSlots: locateTime,
                        matchItems: matchTime,
                        total: elapsedMilliseconds(totalStarted),
                    },
                };
            }
            finally {
                context.dispose();
            }
        }
        finally {
            leaveLifecycle();
        }
    }
    dispose() {
        this.#lifecycle.dispose();
    }
}
export async function createInventoryDetector(options = {}) {
    return InventoryDetector.create(options);
}
//# sourceMappingURL=detector.js.map