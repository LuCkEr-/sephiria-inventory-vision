import { readFile } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";
import { createGridSlots, validateGridOptions } from "./grid.js";
import { decodeImage, resizeRawImage } from "./image.js";
import { disposeTemplate, locateInventoryGrid, locateSlots, rgbaMatToTemplate, } from "./matching.js";
export function elapsedMilliseconds(startedAt) {
    return Number((performance.now() - startedAt).toFixed(3));
}
export function assertUnitInterval(value, name) {
    if (!Number.isFinite(value) || value < 0 || value > 1) {
        throw new Error(`${name} must be a finite number between 0 and 1, received ${value}`);
    }
}
function assertPositiveInteger(value, name) {
    if (!Number.isSafeInteger(value) || value <= 0) {
        throw new Error(`${name} must be a positive integer, received ${value}`);
    }
}
function assertFiniteNonNegativePair(first, second, message) {
    if (!Number.isFinite(first) || first < 0 || !Number.isFinite(second) || second < 0) {
        throw new Error(message);
    }
}
function assertFinitePositivePair(first, second, message) {
    if (!Number.isFinite(first) || first <= 0 || !Number.isFinite(second) || second <= 0) {
        throw new Error(message);
    }
}
function validateOptionalNonNegativeInteger(value, name) {
    if (value !== undefined && (!Number.isSafeInteger(value) || value < 0)) {
        throw new Error(`${name} must be a non-negative integer`);
    }
}
function validateSlot(slot, index) {
    const prefix = `slots[${index}]`;
    assertFiniteNonNegativePair(slot.x, slot.y, `${prefix} coordinates must be finite and non-negative`);
    assertFinitePositivePair(slot.width, slot.height, `${prefix} dimensions must be positive and finite`);
    validateOptionalNonNegativeInteger(slot.row, `${prefix}.row`);
    validateOptionalNonNegativeInteger(slot.column, `${prefix}.column`);
    if (slot.scale !== undefined && (!Number.isFinite(slot.scale) || slot.scale <= 0)) {
        throw new Error(`${prefix}.scale must be positive and finite`);
    }
}
function validateScales(scales) {
    if (scales === undefined)
        return;
    if (scales.length === 0)
        throw new Error("scales must contain at least one value");
    for (const scale of scales) {
        if (!Number.isFinite(scale) || scale <= 0) {
            throw new Error(`scales must contain only positive finite values, received ${scale}`);
        }
    }
}
function validateGridDetectionOptions(options) {
    if (options.gridRows !== undefined)
        assertPositiveInteger(options.gridRows, "gridRows");
    if (options.gridColumns !== undefined)
        assertPositiveInteger(options.gridColumns, "gridColumns");
    if (options.minGridSupport === undefined)
        return;
    assertPositiveInteger(options.minGridSupport, "minGridSupport");
    const capacity = (options.gridRows ?? 4) * (options.gridColumns ?? 6);
    if (options.minGridSupport > capacity) {
        throw new Error(`minGridSupport cannot exceed grid capacity ${capacity}`);
    }
}
function validateThresholds(options) {
    if (options.slotThreshold !== undefined) {
        assertUnitInterval(options.slotThreshold, "slotThreshold");
    }
    if (options.itemThreshold !== undefined) {
        assertUnitInterval(options.itemThreshold, "itemThreshold");
    }
    if (options.emptySlotThreshold !== undefined) {
        assertUnitInterval(options.emptySlotThreshold, "emptySlotThreshold");
    }
}
export function validateDetectOptions(options) {
    if (options.slots && options.grid) {
        throw new Error("slots and grid are mutually exclusive detection inputs");
    }
    options.slots?.forEach(validateSlot);
    if (options.grid)
        validateGridOptions(options.grid);
    validateScales(options.scales);
    if (options.maxSlots !== undefined)
        assertPositiveInteger(options.maxSlots, "maxSlots");
    if (options.slotBorderWidth !== undefined) {
        assertPositiveInteger(options.slotBorderWidth, "slotBorderWidth");
    }
    validateGridDetectionOptions(options);
    validateThresholds(options);
    normalizeLogicalHeight(options.normalizationHeight);
}
export function positiveInteger(value, fallback, name) {
    const resolved = value ?? fallback;
    assertPositiveInteger(resolved, name);
    return resolved;
}
export function unitInterval(value, fallback, name) {
    const resolved = value ?? fallback;
    assertUnitInterval(resolved, name);
    return resolved;
}
export function normalizeLogicalHeight(value) {
    const normalized = value ?? 288;
    if (normalized !== false && (!Number.isFinite(normalized) || normalized <= 0)) {
        throw new Error(`normalizationHeight must be positive or false, received ${normalized}`);
    }
    return normalized;
}
export async function imageToRgbaMat(cv, input) {
    const image = await decodeImage(input);
    const mat = new cv.Mat(image.height, image.width, cv.CV_8UC4);
    try {
        mat.data.set(image.data);
        return { mat, width: image.width, height: image.height, raw: image };
    }
    catch (error) {
        mat.delete();
        throw error;
    }
}
function prepareSlots(cv, imageRgb, slotTemplates, options) {
    if (options.slots)
        return { kind: "ready", slots: options.slots };
    if (options.grid)
        return { kind: "ready", slots: createGridSlots(options.grid) };
    const normalizationHeight = normalizeLogicalHeight(options.normalizationHeight);
    if (normalizationHeight !== false)
        return { kind: "normalize", height: normalizationHeight };
    const slots = locateSlots(cv, imageRgb, slotTemplates, {
        scales: options.scales ?? [1],
        threshold: unitInterval(options.slotThreshold, 0.94, "slotThreshold"),
        borderWidth: options.slotBorderWidth ?? 4,
        maxSlots: options.maxSlots ?? 100,
    });
    return { kind: "ready", slots };
}
async function createNormalizedRgb(cv, raw, normalizationHeight) {
    const normalizedImage = await resizeRawImage(raw, normalizationHeight);
    const normalizedRgba = new cv.Mat(normalizedImage.height, normalizedImage.width, cv.CV_8UC4);
    normalizedRgba.data.set(normalizedImage.data);
    const normalizedRgb = new cv.Mat();
    try {
        cv.cvtColor(normalizedRgba, normalizedRgb, cv.COLOR_RGBA2RGB);
        return normalizedRgb;
    }
    catch (error) {
        normalizedRgb.delete();
        throw error;
    }
    finally {
        normalizedRgba.delete();
    }
}
function locateNormalizedGrid(cv, imageRgb, slotTemplates, options) {
    const emptySlotTemplate = slotTemplates.find((template) => template.metadata.name === "InventorySlot0") ??
        slotTemplates[0];
    if (!emptySlotTemplate)
        return [];
    const grid = locateInventoryGrid(cv, imageRgb, emptySlotTemplate, {
        rows: options.gridRows ?? 4,
        columns: options.gridColumns ?? 6,
        threshold: unitInterval(options.slotThreshold, 0.97, "slotThreshold"),
        minSupport: options.minGridSupport ?? 5,
        preferredOriginY: Math.round(imageRgb.rows * (99 / 288)),
    });
    return grid?.slots ?? [];
}
function consumeDecodedToRgb(cv, decoded) {
    const rgb = new cv.Mat();
    try {
        cv.cvtColor(decoded.mat, rgb, cv.COLOR_RGBA2RGB);
        return rgb;
    }
    catch (error) {
        rgb.delete();
        throw error;
    }
    finally {
        decoded.mat.delete();
    }
}
/**
 * Owns the RGB matrices and slot-localization state shared by every detector.
 * The decoded RGBA matrix is consumed; callers own only the returned context.
 */
export async function prepareDetectionContext(cv, decoded, slotTemplates, options) {
    const originalRgb = consumeDecodedToRgb(cv, decoded);
    let imageRgb = originalRgb;
    let coordinateScale = 1;
    let normalized = false;
    try {
        const preparation = prepareSlots(cv, imageRgb, slotTemplates, options);
        let slots;
        if (preparation.kind === "normalize") {
            coordinateScale = decoded.height / preparation.height;
            imageRgb = await createNormalizedRgb(cv, decoded.raw, preparation.height);
            normalized = true;
            slots = locateNormalizedGrid(cv, imageRgb, slotTemplates, options);
        }
        else {
            slots = preparation.slots;
        }
        let disposed = false;
        return {
            imageRgb,
            slots,
            coordinateScale,
            normalized,
            dispose() {
                if (disposed)
                    return;
                if (normalized)
                    imageRgb.delete();
                originalRgb.delete();
                disposed = true;
            },
        };
    }
    catch (error) {
        if (normalized)
            imageRgb.delete();
        originalRgb.delete();
        throw error;
    }
}
export async function loadTemplate(cv, root, metadata) {
    const absoluteRoot = resolve(root);
    const absoluteFile = resolve(absoluteRoot, metadata.file);
    const relativeFile = relative(absoluteRoot, absoluteFile);
    if (relativeFile.startsWith("..") || isAbsolute(relativeFile)) {
        throw new Error(`Template file escapes the catalog directory: ${metadata.file}`);
    }
    const image = await decodeImage(await readFile(absoluteFile));
    const rgba = new cv.Mat(image.height, image.width, cv.CV_8UC4);
    try {
        rgba.data.set(image.data);
        return rgbaMatToTemplate(cv, rgba, metadata);
    }
    finally {
        rgba.delete();
    }
}
export async function loadTemplates(cv, root, metadata) {
    const templates = [];
    try {
        for (const template of metadata)
            templates.push(await loadTemplate(cv, root, template));
        return templates;
    }
    catch (error) {
        templates.forEach(disposeTemplate);
        throw error;
    }
}
//# sourceMappingURL=runtime.js.map