import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { InventoryCatalog } from "./types.js";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function validateStringArray(value: unknown, name: string): void {
  if (!Array.isArray(value) || value.some((entry) => !isNonEmptyString(entry))) {
    throw new Error(`${name} must be an array of non-empty strings`);
  }
}

function validateNonNegativeIntegerFields(
  value: Record<string, unknown>,
  name: string,
  fields: readonly string[],
): void {
  for (const field of fields) {
    if (!Number.isSafeInteger(value[field]) || (value[field] as number) < 0) {
      throw new Error(`${name}.${field} must be a non-negative integer`);
    }
  }
}

function validateNonEmptyFields(
  value: Record<string, unknown>,
  name: string,
  fields: readonly string[],
): void {
  for (const field of fields) {
    if (!isNonEmptyString(value[field])) throw new Error(`${name}.${field} must be non-empty`);
  }
}

function validatePlacementRequirements(value: unknown, name: string): void {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  value.forEach((requirement, index) => {
    if (!isRecord(requirement)) throw new Error(`${name}[${index}] must be an object`);
    validateNonEmptyFields(requirement, `${name}[${index}]`, [
      "code",
      "description",
      "sourceClass",
    ]);
  });
}

function validateOptionalGameplayFields(value: Record<string, unknown>, name: string): void {
  for (const field of ["isUniqueEffect", "isWeaponRelated", "includeConditionInBounds"] as const) {
    if (value[field] !== undefined && typeof value[field] !== "boolean") {
      throw new Error(`${name}.${field} must be boolean when present`);
    }
  }
  for (const field of ["effectStringKeys", "componentClasses"] as const) {
    if (value[field] !== undefined) validateStringArray(value[field], `${name}.${field}`);
  }
}

function validateSourcePrefab(value: unknown, name: string): void {
  if (value === undefined || value === null) return;
  if (
    !isRecord(value) ||
    !isNonEmptyString(value["asset"]) ||
    !Number.isSafeInteger(value["pathId"]) ||
    (value["pathId"] as number) < 0 ||
    !isNonEmptyString(value["name"])
  ) {
    throw new Error(`${name} is malformed`);
  }
}

function validateGameplay(value: unknown, name: string): void {
  if (!isRecord(value)) throw new Error(`${name} must be an object`);
  validateNonNegativeIntegerFields(value, name, ["itemId", "itemTypeValue", "rarity"]);
  validateNonEmptyFields(value, name, ["displayName", "itemType"]);
  if (value["rotatable"] !== null && typeof value["rotatable"] !== "boolean") {
    throw new Error(`${name}.rotatable must be boolean or null`);
  }
  if (
    value["maxBuffLevel"] !== null &&
    (!Number.isSafeInteger(value["maxBuffLevel"]) || (value["maxBuffLevel"] as number) < 0)
  ) {
    throw new Error(`${name}.maxBuffLevel must be a non-negative integer or null`);
  }
  for (const field of ["conditionQuery", "effectQuery"] as const) {
    if (value[field] !== null && typeof value[field] !== "string") {
      throw new Error(`${name}.${field} must be a string or null`);
    }
  }
  validatePlacementRequirements(value["placementRequirements"], `${name}.placementRequirements`);
  validateOptionalGameplayFields(value, name);
  validateSourcePrefab(value["sourcePrefab"], `${name}.sourcePrefab`);
}

function validateTemplateIdentity(
  entry: Record<string, unknown>,
  index: number,
  name: string,
  ids: Set<string>,
): void {
  for (const field of ["id", "name", "file"] as const) {
    if (!isNonEmptyString(entry[field])) {
      throw new Error(`${name}[${index}].${field} must be a non-empty string`);
    }
  }
  const id = entry["id"] as string;
  if (ids.has(id)) throw new Error(`${name} contains duplicate id ${id}`);
  ids.add(id);
}

function validateTemplateDimensions(
  entry: Record<string, unknown>,
  index: number,
  name: string,
): void {
  for (const field of ["width", "height"] as const) {
    if (!Number.isSafeInteger(entry[field]) || (entry[field] as number) <= 0) {
      throw new Error(`${name}[${index}].${field} must be a positive integer`);
    }
  }
}

function validateTemplateProvenance(
  entry: Record<string, unknown>,
  index: number,
  name: string,
): void {
  if (!isNonEmptyString(entry["sourceFile"])) {
    throw new Error(`${name}[${index}].sourceFile must be a non-empty string`);
  }
  if (!Number.isSafeInteger(entry["pathId"]) || (entry["pathId"] as number) < 0) {
    throw new Error(`${name}[${index}].pathId must be a non-negative integer`);
  }
}

function validateTemplateMetadata(
  entry: Record<string, unknown>,
  index: number,
  name: string,
): void {
  if (entry["spriteName"] !== undefined && !isNonEmptyString(entry["spriteName"])) {
    throw new Error(`${name}[${index}].spriteName must be non-empty when present`);
  }
  if (entry["displayNames"] !== undefined) {
    validateStringArray(entry["displayNames"], `${name}[${index}].displayNames`);
  }
  const itemIds = entry["itemIds"];
  if (
    itemIds !== undefined &&
    (!Array.isArray(itemIds) ||
      itemIds.some((itemId) => !Number.isSafeInteger(itemId) || itemId < 0))
  ) {
    throw new Error(`${name}[${index}].itemIds must contain non-negative integers`);
  }
  if (entry["ambiguousIdentity"] !== undefined && typeof entry["ambiguousIdentity"] !== "boolean") {
    throw new Error(`${name}[${index}].ambiguousIdentity must be boolean when present`);
  }
}

function validateTemplateRotation(
  entry: Record<string, unknown>,
  index: number,
  name: string,
): void {
  if (
    entry["rotationDegrees"] !== undefined &&
    ![0, 90, 180, 270].includes(entry["rotationDegrees"] as number)
  ) {
    throw new Error(`${name}[${index}].rotationDegrees is unsupported`);
  }
  if (
    entry["canonicalTemplateId"] !== undefined &&
    !isNonEmptyString(entry["canonicalTemplateId"])
  ) {
    throw new Error(`${name}[${index}].canonicalTemplateId must be non-empty when present`);
  }
}

function validateItemVariant(variant: unknown, variantName: string): void {
  if (
    !isRecord(variant) ||
    !Number.isSafeInteger(variant["itemId"]) ||
    (variant["itemId"] as number) < 0 ||
    !isNonEmptyString(variant["displayName"])
  ) {
    throw new Error(`${variantName} is malformed`);
  }
  validateGameplay(variant["gameplay"], `${variantName}.gameplay`);
  const gameplay = variant["gameplay"];
  if (
    !isRecord(gameplay) ||
    variant["itemId"] !== gameplay["itemId"] ||
    variant["displayName"] !== gameplay["displayName"]
  ) {
    throw new Error(`${variantName} identity is inconsistent with its gameplay metadata`);
  }
}

function validateItemVariants(value: unknown, name: string): void {
  if (value === undefined) return;
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  value.forEach((variant, index) => {
    validateItemVariant(variant, `${name}[${index}]`);
  });
}

function validateTemplate(entry: unknown, index: number, name: string, ids: Set<string>): void {
  if (!isRecord(entry)) throw new Error(`${name}[${index}] must be an object`);
  validateTemplateIdentity(entry, index, name, ids);
  validateTemplateDimensions(entry, index, name);
  validateTemplateProvenance(entry, index, name);
  validateTemplateMetadata(entry, index, name);
  validateTemplateRotation(entry, index, name);
  validateItemVariants(entry["itemVariants"], `${name}[${index}].itemVariants`);
}

function validateTemplates(
  value: unknown,
  name: string,
): asserts value is Record<string, unknown>[] {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  const ids = new Set<string>();
  value.forEach((entry, index) => {
    validateTemplate(entry, index, name, ids);
  });
}

function validateCatalog(value: unknown, path: string): asserts value is InventoryCatalog {
  if (!isRecord(value) || ![1, 2, 3].includes(value["schemaVersion"] as number)) {
    throw new Error(`Unsupported or malformed catalog: ${path}`);
  }
  if (typeof value["game"] !== "string" || value["game"].length === 0) {
    throw new Error(`Catalog game must be a non-empty string: ${path}`);
  }
  if (value["generatedFrom"] !== undefined && !isNonEmptyString(value["generatedFrom"])) {
    throw new Error(`Catalog generatedFrom must be non-empty when present: ${path}`);
  }
  if (!Number.isSafeInteger(value["nativeSlotSize"]) || (value["nativeSlotSize"] as number) <= 0) {
    throw new Error(`Catalog nativeSlotSize must be a positive integer: ${path}`);
  }
  const items = value["items"];
  const slotTemplates = value["slotTemplates"];
  validateTemplates(items, "catalog.items");
  validateTemplates(slotTemplates, "catalog.slotTemplates");
  if (slotTemplates.length === 0) {
    throw new Error(`Catalog must contain at least one slot template: ${path}`);
  }
  const itemIds = new Set(items.map((item) => item["id"] as string));
  for (const [index, item] of items.entries()) {
    const canonicalTemplateId = item["canonicalTemplateId"];
    if (canonicalTemplateId !== undefined && !itemIds.has(canonicalTemplateId as string)) {
      throw new Error(`catalog.items[${index}] references unknown canonical template`);
    }
  }
}

export function defaultCatalogPath(): string {
  return fileURLToPath(new URL("../assets/catalog/catalog.json", import.meta.url));
}

export async function loadCatalog(catalogPath = defaultCatalogPath()): Promise<{
  catalog: InventoryCatalog;
  root: string;
}> {
  const absolutePath = resolve(catalogPath);
  const catalog: unknown = JSON.parse(await readFile(absolutePath, "utf8"));
  validateCatalog(catalog, absolutePath);
  return { catalog, root: dirname(absolutePath) };
}
