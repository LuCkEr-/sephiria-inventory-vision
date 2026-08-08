import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
function isRecord(value) {
    return typeof value === "object" && value !== null;
}
function isNonEmptyString(value) {
    return typeof value === "string" && value.length > 0;
}
function validateStringArray(value, name) {
    if (!Array.isArray(value) || value.some((entry) => !isNonEmptyString(entry))) {
        throw new Error(`${name} must be an array of non-empty strings`);
    }
}
function validateNonNegativeIntegerFields(value, name, fields) {
    for (const field of fields) {
        if (!Number.isSafeInteger(value[field]) || value[field] < 0) {
            throw new Error(`${name}.${field} must be a non-negative integer`);
        }
    }
}
function validateNonEmptyFields(value, name, fields) {
    for (const field of fields) {
        if (!isNonEmptyString(value[field]))
            throw new Error(`${name}.${field} must be non-empty`);
    }
}
function validatePlacementRequirements(value, name) {
    if (!Array.isArray(value))
        throw new Error(`${name} must be an array`);
    value.forEach((requirement, index) => {
        if (!isRecord(requirement))
            throw new Error(`${name}[${index}] must be an object`);
        validateNonEmptyFields(requirement, `${name}[${index}]`, [
            "code",
            "description",
            "sourceClass",
        ]);
    });
}
function validateOptionalGameplayFields(value, name) {
    for (const field of ["isUniqueEffect", "isWeaponRelated", "includeConditionInBounds"]) {
        if (value[field] !== undefined && typeof value[field] !== "boolean") {
            throw new Error(`${name}.${field} must be boolean when present`);
        }
    }
    for (const field of ["effectStringKeys", "componentClasses"]) {
        if (value[field] !== undefined)
            validateStringArray(value[field], `${name}.${field}`);
    }
}
function validateSourcePrefab(value, name) {
    if (value === undefined || value === null)
        return;
    if (!isRecord(value) ||
        !isNonEmptyString(value["asset"]) ||
        !Number.isSafeInteger(value["pathId"]) ||
        value["pathId"] < 0 ||
        !isNonEmptyString(value["name"])) {
        throw new Error(`${name} is malformed`);
    }
}
function validateGameplay(value, name) {
    if (!isRecord(value))
        throw new Error(`${name} must be an object`);
    validateNonNegativeIntegerFields(value, name, ["itemId", "itemTypeValue", "rarity"]);
    validateNonEmptyFields(value, name, ["displayName", "itemType"]);
    if (value["rotatable"] !== null && typeof value["rotatable"] !== "boolean") {
        throw new Error(`${name}.rotatable must be boolean or null`);
    }
    if (value["maxBuffLevel"] !== null &&
        (!Number.isSafeInteger(value["maxBuffLevel"]) || value["maxBuffLevel"] < 0)) {
        throw new Error(`${name}.maxBuffLevel must be a non-negative integer or null`);
    }
    for (const field of ["conditionQuery", "effectQuery"]) {
        if (value[field] !== null && typeof value[field] !== "string") {
            throw new Error(`${name}.${field} must be a string or null`);
        }
    }
    validatePlacementRequirements(value["placementRequirements"], `${name}.placementRequirements`);
    validateOptionalGameplayFields(value, name);
    validateSourcePrefab(value["sourcePrefab"], `${name}.sourcePrefab`);
}
function validateTemplateIdentity(entry, index, name, ids) {
    for (const field of ["id", "name", "file"]) {
        if (!isNonEmptyString(entry[field])) {
            throw new Error(`${name}[${index}].${field} must be a non-empty string`);
        }
    }
    const id = entry["id"];
    if (ids.has(id))
        throw new Error(`${name} contains duplicate id ${id}`);
    ids.add(id);
}
function validateTemplateDimensions(entry, index, name) {
    for (const field of ["width", "height"]) {
        if (!Number.isSafeInteger(entry[field]) || entry[field] <= 0) {
            throw new Error(`${name}[${index}].${field} must be a positive integer`);
        }
    }
}
function validateTemplateProvenance(entry, index, name) {
    if (!isNonEmptyString(entry["sourceFile"])) {
        throw new Error(`${name}[${index}].sourceFile must be a non-empty string`);
    }
    if (!Number.isSafeInteger(entry["pathId"]) || entry["pathId"] < 0) {
        throw new Error(`${name}[${index}].pathId must be a non-negative integer`);
    }
}
function validateTemplateMetadata(entry, index, name) {
    if (entry["spriteName"] !== undefined && !isNonEmptyString(entry["spriteName"])) {
        throw new Error(`${name}[${index}].spriteName must be non-empty when present`);
    }
    if (entry["displayNames"] !== undefined) {
        validateStringArray(entry["displayNames"], `${name}[${index}].displayNames`);
    }
    const itemIds = entry["itemIds"];
    if (itemIds !== undefined &&
        (!Array.isArray(itemIds) ||
            itemIds.some((itemId) => !Number.isSafeInteger(itemId) || itemId < 0))) {
        throw new Error(`${name}[${index}].itemIds must contain non-negative integers`);
    }
    if (entry["ambiguousIdentity"] !== undefined && typeof entry["ambiguousIdentity"] !== "boolean") {
        throw new Error(`${name}[${index}].ambiguousIdentity must be boolean when present`);
    }
}
function validateTemplateRotation(entry, index, name) {
    if (entry["rotationDegrees"] !== undefined &&
        ![0, 90, 180, 270].includes(entry["rotationDegrees"])) {
        throw new Error(`${name}[${index}].rotationDegrees is unsupported`);
    }
    if (entry["canonicalTemplateId"] !== undefined &&
        !isNonEmptyString(entry["canonicalTemplateId"])) {
        throw new Error(`${name}[${index}].canonicalTemplateId must be non-empty when present`);
    }
}
function validateItemVariant(variant, variantName) {
    if (!isRecord(variant) ||
        !Number.isSafeInteger(variant["itemId"]) ||
        variant["itemId"] < 0 ||
        !isNonEmptyString(variant["displayName"])) {
        throw new Error(`${variantName} is malformed`);
    }
    validateGameplay(variant["gameplay"], `${variantName}.gameplay`);
    const gameplay = variant["gameplay"];
    if (!isRecord(gameplay) ||
        variant["itemId"] !== gameplay["itemId"] ||
        variant["displayName"] !== gameplay["displayName"]) {
        throw new Error(`${variantName} identity is inconsistent with its gameplay metadata`);
    }
}
function validateItemVariants(value, name) {
    if (value === undefined)
        return;
    if (!Array.isArray(value))
        throw new Error(`${name} must be an array`);
    value.forEach((variant, index) => {
        validateItemVariant(variant, `${name}[${index}]`);
    });
}
function validateTemplate(entry, index, name, ids) {
    if (!isRecord(entry))
        throw new Error(`${name}[${index}] must be an object`);
    validateTemplateIdentity(entry, index, name, ids);
    validateTemplateDimensions(entry, index, name);
    validateTemplateProvenance(entry, index, name);
    validateTemplateMetadata(entry, index, name);
    validateTemplateRotation(entry, index, name);
    validateItemVariants(entry["itemVariants"], `${name}[${index}].itemVariants`);
}
function validateTemplates(value, name) {
    if (!Array.isArray(value))
        throw new Error(`${name} must be an array`);
    const ids = new Set();
    value.forEach((entry, index) => {
        validateTemplate(entry, index, name, ids);
    });
}
function validateCatalog(value, path) {
    if (!isRecord(value) || ![1, 2, 3].includes(value["schemaVersion"])) {
        throw new Error(`Unsupported or malformed catalog: ${path}`);
    }
    if (typeof value["game"] !== "string" || value["game"].length === 0) {
        throw new Error(`Catalog game must be a non-empty string: ${path}`);
    }
    if (value["generatedFrom"] !== undefined && !isNonEmptyString(value["generatedFrom"])) {
        throw new Error(`Catalog generatedFrom must be non-empty when present: ${path}`);
    }
    if (!Number.isSafeInteger(value["nativeSlotSize"]) || value["nativeSlotSize"] <= 0) {
        throw new Error(`Catalog nativeSlotSize must be a positive integer: ${path}`);
    }
    const items = value["items"];
    const slotTemplates = value["slotTemplates"];
    validateTemplates(items, "catalog.items");
    validateTemplates(slotTemplates, "catalog.slotTemplates");
    if (slotTemplates.length === 0) {
        throw new Error(`Catalog must contain at least one slot template: ${path}`);
    }
    const itemIds = new Set(items.map((item) => item["id"]));
    for (const [index, item] of items.entries()) {
        const canonicalTemplateId = item["canonicalTemplateId"];
        if (canonicalTemplateId !== undefined && !itemIds.has(canonicalTemplateId)) {
            throw new Error(`catalog.items[${index}] references unknown canonical template`);
        }
    }
}
export function defaultCatalogPath() {
    return fileURLToPath(new URL("../assets/catalog/catalog.json", import.meta.url));
}
export async function loadCatalog(catalogPath = defaultCatalogPath()) {
    const absolutePath = resolve(catalogPath);
    const catalog = JSON.parse(await readFile(absolutePath, "utf8"));
    validateCatalog(catalog, absolutePath);
    return { catalog, root: dirname(absolutePath) };
}
//# sourceMappingURL=catalog.js.map