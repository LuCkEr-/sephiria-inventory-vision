import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

interface ManifestRow {
  category: string;
  unity_name: string;
  width: string;
  height: string;
  collected_png: string;
  original_extracted_png: string;
  source_file: string;
  path_id: string;
}

interface ItemIconMapping {
  itemId: number;
  internalName: string;
  localizationKey: string;
  displayName: string;
  rarity: number;
  iconSourceFile: string;
  iconPathId: number;
  spriteName: string;
  spritePath: string;
  width: number;
  height: number;
}

interface PlacementRequirement {
  code: string;
  description: string;
  sourceClass: string;
}

interface ItemGameplayMapping {
  itemId: number;
  displayName: string;
  itemType: string;
  itemTypeValue: number;
  rarity: number;
  rotatable: boolean | null;
  maxBuffLevel: number | null;
  placementRequirements: PlacementRequirement[];
  conditionQuery: string | null;
  effectQuery: string | null;
  [key: string]: unknown;
}

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const defaultSource = String.raw`C:\Program Files (x86)\Steam\steamapps\common\Sephiria\Extracted_Image_Assets\Inventory_Related_Textures`;
const sourceRoot = resolve(process.argv[2] ?? defaultSource);
const outputRoot = resolve(projectRoot, "assets", "catalog");

function parseCsv<T>(content: string): T[] {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < content.length; index += 1) {
    const char = content.charAt(index);
    if (quoted) {
      if (char === '"' && content[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      record.push(field);
      field = "";
    } else if (char === "\n") {
      record.push(field.replace(/\r$/, ""));
      records.push(record);
      record = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (field.length > 0 || record.length > 0) {
    record.push(field.replace(/\r$/, ""));
    records.push(record);
  }

  const [headers, ...rows] = records;
  if (!headers) return [];
  return rows
    .filter((row) => row.some(Boolean))
    .map((row) =>
      Object.fromEntries(
        headers.map((header, index) => [header.replace(/^\uFEFF/, ""), row[index] ?? ""]),
      ),
    ) as T[];
}

function slugify(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
}

const slotPattern =
  /^InventorySlot(?:_New(?:0|1|1_1Normal|1_2Uncommon|1_2UncommonBond|1_3Rare|1_3RareBond|1_4Legendary|1_4LegendaryBond|2|3)|0|1)$/i;

async function main(): Promise<void> {
  const csv = await readFile(join(sourceRoot, "inventory_texture_manifest.csv"), "utf8");
  const rows = parseCsv<ManifestRow>(csv);
  const imageManifestRoot = dirname(sourceRoot);
  const itemMappings = JSON.parse(
    await readFile(join(projectRoot, "assets", "item-icon-map.json"), "utf8"),
  ) as ItemIconMapping[];
  const gameplayMappings = JSON.parse(
    await readFile(join(projectRoot, "assets", "item-gameplay-map.json"), "utf8"),
  ) as ItemGameplayMapping[];
  const gameplayByItemId = new Map(gameplayMappings.map((mapping) => [mapping.itemId, mapping]));
  const slotRows = rows.filter((row) => slotPattern.test(row.unity_name));

  await rm(outputRoot, { recursive: true, force: true });
  await mkdir(join(outputRoot, "items"), { recursive: true });
  await mkdir(join(outputRoot, "ui"), { recursive: true });

  const items = [];
  const mappingsBySprite = new Map<string, ItemIconMapping[]>();
  for (const mapping of itemMappings) {
    const key = `${mapping.iconSourceFile.toLowerCase()}:${mapping.iconPathId}`;
    const existing = mappingsBySprite.get(key) ?? [];
    existing.push(mapping);
    mappingsBySprite.set(key, existing);
  }
  for (const mappings of mappingsBySprite.values()) {
    const row = mappings[0];
    if (!row) continue;
    const filename = basename(row.spritePath);
    const sourceImagePath = join(imageManifestRoot, row.spritePath);
    await copyFile(sourceImagePath, join(outputRoot, "items", filename));
    const displayNames = [
      ...new Set(
        mappings.map((mapping) => mapping.displayName).filter((name) => name && name !== "..."),
      ),
    ];
    const canonicalTemplateId = `${slugify(row.spriteName)}-${row.iconPathId}`;
    const itemVariants = mappings
      .map((mapping) => ({
        itemId: mapping.itemId,
        displayName: mapping.displayName,
        gameplay: gameplayByItemId.get(mapping.itemId),
      }))
      .filter((variant) => variant.gameplay);
    const canonicalName =
      displayNames.length === 1 ? (displayNames[0] ?? row.spriteName) : row.spriteName;
    items.push({
      id: canonicalTemplateId,
      name: canonicalName,
      file: `items/${filename}`,
      width: row.width,
      height: row.height,
      sourceFile: row.iconSourceFile,
      pathId: row.iconPathId,
      spriteName: row.spriteName,
      displayNames,
      itemIds: [...new Set(mappings.map((mapping) => mapping.itemId))],
      ambiguousIdentity: displayNames.length !== 1,
      itemVariants,
      rotationDegrees: 0,
    });

    const rotatableVariants = itemVariants.filter(
      (variant) => variant.gameplay?.rotatable === true,
    );
    if (rotatableVariants.length > 0) {
      const rotatableDisplayNames = [
        ...new Set(
          rotatableVariants
            .map((variant) => variant.displayName)
            .filter((name) => name && name !== "..."),
        ),
      ];
      const stem = filename.replace(/\.png$/i, "");
      const rotatableName =
        rotatableDisplayNames.length === 1
          ? (rotatableDisplayNames[0] ?? row.spriteName)
          : row.spriteName;
      for (const rotationDegrees of [90, 180, 270] as const) {
        const rotatedFilename = `${stem}__rot${rotationDegrees}.png`;
        await sharp(sourceImagePath)
          .rotate(rotationDegrees)
          .png()
          .toFile(join(outputRoot, "items", rotatedFilename));
        const swapsDimensions = rotationDegrees === 90 || rotationDegrees === 270;
        items.push({
          id: `${canonicalTemplateId}-rot${rotationDegrees}`,
          name: rotatableName,
          file: `items/${rotatedFilename}`,
          width: swapsDimensions ? row.height : row.width,
          height: swapsDimensions ? row.width : row.height,
          sourceFile: row.iconSourceFile,
          pathId: row.iconPathId,
          spriteName: row.spriteName,
          displayNames: rotatableDisplayNames,
          itemIds: rotatableVariants.map((variant) => variant.itemId),
          ambiguousIdentity: rotatableDisplayNames.length !== 1,
          itemVariants: rotatableVariants,
          rotationDegrees,
          canonicalTemplateId,
        });
      }
    }
  }

  const curatedTemplates = JSON.parse(
    await readFile(
      join(projectRoot, "assets", "curated-items", "generated-templates.json"),
      "utf8",
    ),
  ) as {
    id: string;
    name: string;
    file: string;
    width: number;
    height: number;
    sourceFile: string;
    pathId: number;
    spriteName: string;
    displayNames: string[];
    itemIds: number[];
    ambiguousIdentity: boolean;
    rotationDegrees?: 0 | 90 | 180 | 270;
  }[];
  for (const template of curatedTemplates) {
    await copyFile(
      join(projectRoot, "assets", "curated-items", template.file),
      join(outputRoot, "items", template.file),
    );
    items.push({
      ...template,
      file: `items/${template.file}`,
      itemVariants: template.itemIds
        .map((itemId) => {
          const gameplay = gameplayByItemId.get(itemId);
          return gameplay ? { itemId, displayName: gameplay.displayName, gameplay } : null;
        })
        .filter((variant) => variant !== null),
    });
  }

  const slotTemplates = [];
  for (const row of slotRows) {
    const filename = basename(row.collected_png);
    await copyFile(join(sourceRoot, row.collected_png), join(outputRoot, "ui", filename));
    slotTemplates.push({
      id: `${slugify(row.unity_name)}-${row.path_id}`,
      name: row.unity_name,
      file: `ui/${filename}`,
      width: Number(row.width),
      height: Number(row.height),
      sourceFile: row.source_file,
      pathId: Number(row.path_id),
    });
  }

  const catalog = {
    schemaVersion: 3,
    game: "Sephiria",
    nativeSlotSize: 32,
    generatedFrom:
      "item-icon-map.json + item-gameplay-map.json + image_manifest.csv + inventory_texture_manifest.csv",
    items: items.sort((left, right) => left.name.localeCompare(right.name)),
    slotTemplates: slotTemplates.sort((left, right) => left.name.localeCompare(right.name)),
  };

  await writeFile(
    join(outputRoot, "catalog.json"),
    `${JSON.stringify(catalog, null, 2)}\n`,
    "utf8",
  );
  console.log(
    `Generated ${items.length} item templates and ${slotTemplates.length} slot templates in ${outputRoot}`,
  );
}

await main();
