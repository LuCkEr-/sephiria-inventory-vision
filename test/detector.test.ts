import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import test from "node:test";

import sharp from "sharp";

import { createInventoryDetector, type InventoryCatalog } from "../src/index.js";

const projectRoot = resolve(import.meta.dirname, "..");
const catalogRoot = join(projectRoot, "assets", "catalog");
const catalogPath = join(catalogRoot, "catalog.json");
const itemNames = [
  "Large Focus Potion",
  "Large Restorative Potion",
  "Item_Scroll00",
  "Throwing Knife",
];

test("catalog exposes prefab-authored buff, rotation, and placement metadata", async () => {
  const catalog = JSON.parse(await readFile(catalogPath, "utf8")) as InventoryCatalog;

  const gameplay = (name: string) => {
    const template = catalog.items.find((entry) => entry.displayNames?.includes(name));
    assert(template, `Expected catalog template for ${name}`);
    const variant = template.itemVariants?.find((entry) => entry.displayName === name);
    assert(variant, `Expected gameplay variant for ${name}`);
    return variant.gameplay;
  };

  assert.equal(gameplay("Charm of Skill").maxBuffLevel, 1);
  assert.equal(gameplay("Foundation").rotatable, false);
  assert.equal(gameplay("Advent").rotatable, true);
  assert.match(gameplay("Advent").effectQuery ?? "", /UPUP 1/);
  assert.deepEqual(
    gameplay("Silver Plate").placementRequirements.map((requirement) => requirement.code),
    ["bottom-row"],
  );
  assert.deepEqual(
    gameplay("Keel Fragment").placementRequirements.map((requirement) => requirement.code),
    ["outer-edge"],
  );

  const binaryStarRotations = catalog.items
    .filter((entry) => entry.displayNames?.includes("Binary Star"))
    .map((entry) => entry.rotationDegrees)
    .filter((rotation): rotation is 90 | 180 | 270 => rotation !== undefined && rotation !== 0);
  assert.deepEqual(
    [...new Set(binaryStarRotations)].sort((left, right) => left - right),
    [90, 180, 270],
  );
  assert.equal(
    catalog.items.some(
      (entry) => entry.displayNames?.includes("Good Will") && (entry.rotationDegrees ?? 0) !== 0,
    ),
    false,
  );
});

async function createSyntheticInventory(scale = 1): Promise<Buffer> {
  const catalog = JSON.parse(await readFile(catalogPath, "utf8")) as InventoryCatalog;
  const slot = catalog.slotTemplates.find((entry) => entry.name === "InventorySlot0");
  assert(slot, "Expected the empty inventory slot template in the generated catalog");

  const nativeSize = catalog.nativeSlotSize;
  const cells: Buffer[] = [];
  for (const name of [...itemNames, null, null]) {
    const composites: { input: string; left: number; top: number }[] = [
      { input: join(catalogRoot, slot.file), left: 0, top: 0 },
    ];
    if (name) {
      const item = catalog.items.find((entry) => entry.name === name);
      assert(item, `Expected item template ${name}`);
      composites.push({
        input: join(catalogRoot, item.file),
        left: Math.floor((nativeSize - item.width) / 2),
        top: Math.floor((nativeSize - item.height) / 2),
      });
    }
    cells.push(
      await sharp({
        create: { width: nativeSize, height: nativeSize, channels: 4, background: "#16131cff" },
      })
        .composite(composites)
        .png()
        .toBuffer(),
    );
  }

  const native = await sharp({
    create: { width: nativeSize * 3, height: nativeSize * 2, channels: 4, background: "#09070cff" },
  })
    .composite(
      cells.map((input, index) => ({
        input,
        left: (index % 3) * nativeSize,
        top: Math.floor(index / 3) * nativeSize,
      })),
    )
    .png()
    .toBuffer();

  if (scale === 1) return native;
  return sharp(native)
    .resize(nativeSize * 3 * scale, nativeSize * 2 * scale, { kernel: "nearest" })
    .png()
    .toBuffer();
}

test("matches known item templates in an explicit inventory grid", async () => {
  const detector = await createInventoryDetector({ catalogPath });
  try {
    const input = await createSyntheticInventory();
    await assert.rejects(
      detector.detectWithCandidateVerification(
        input,
        { slots: [{ x: 0, y: 0, width: 32, height: 32 }] },
        [],
      ),
      /one candidate entry per explicit slot/,
    );
    const recovered = await detector.detectWithCandidateVerification(
      input,
      { slots: [{ x: 0, y: 0, width: 32, height: 32 }], itemThreshold: 0.97 },
      ["Not in catalog"],
    );
    assert.equal(recovered.slots[0]?.item?.name, itemNames[0]);

    const result = await detector.detect(input, {
      grid: { x: 0, y: 0, rows: 2, columns: 3 },
      itemThreshold: 0.97,
      alternatives: 2,
    });

    assert.equal(result.slots.length, 6);
    assert.deepEqual(
      result.slots.slice(0, 4).map((slot) => slot.item?.name),
      itemNames,
    );
    assert.equal(result.slots[4]?.item, null);
    assert.equal(result.slots[5]?.item, null);
  } finally {
    detector.dispose();
  }
});

test("automatically locates slots and recognizes scaled item templates", async () => {
  const detector = await createInventoryDetector({ catalogPath });
  try {
    const result = await detector.detect(await createSyntheticInventory(2), {
      normalizationHeight: false,
      scales: [2],
      slotThreshold: 0.98,
      itemThreshold: 0.97,
      maxSlots: 6,
    });

    assert.equal(result.slots.length, 6);
    assert.deepEqual(
      result.matchedItems.map((slot) => slot.item?.name).sort(),
      [...itemNames].sort(),
    );
  } finally {
    detector.dispose();
  }
});

type ExpectedItem = readonly [row: number, column: number, name: string];
type ExpectedEmpty = readonly [row: number, column: number];

const realCases: readonly {
  file: string;
  origin: readonly [number, number];
  items: readonly ExpectedItem[];
  empty?: readonly ExpectedEmpty[];
}[] = [
  { file: "14-39-32.png", origin: [248, 98], items: [[0, 0, "Foundation"]] },
  { file: "15-05-25.png", origin: [248, 98], items: [[0, 0, "Foundation"]] },
  {
    file: "15-15-04.png",
    origin: [248, 98],
    items: [
      [0, 0, "Foundation"],
      [0, 1, "Cloak of Verdant Spirit"],
      [0, 2, "Lucky Medal"],
      [0, 3, "Zappy Custard Pastry"],
      [0, 5, "Charm of Skill"],
      [1, 2, "Future"],
      [1, 4, "Stack"],
    ],
  },
  {
    file: "15-16-10.png",
    origin: [248, 99],
    items: [
      [0, 0, "Foundation"],
      [0, 1, "Cloak of Verdant Spirit"],
      [0, 2, "Lucky Medal"],
      [0, 3, "Zappy Custard Pastry"],
      [0, 4, "Healing Stream"],
      [0, 5, "Charm of Skill"],
      [1, 2, "Future"],
      [1, 4, "Stack"],
    ],
  },
  {
    file: "15-20-26.png",
    origin: [248, 99],
    items: [
      [0, 0, "Foundation"],
      [0, 1, "Cloak of Verdant Spirit"],
      [0, 2, "Moment of Lambic"],
      [0, 3, "Zappy Custard Pastry"],
      [0, 4, "Lucky Medal"],
      [0, 5, "Charm of Skill"],
      [1, 1, "Future"],
      [1, 2, "Future"],
      [1, 3, "Healing Stream"],
      [1, 4, "Stack"],
    ],
  },
  {
    file: "15-35-41.png",
    origin: [248, 98],
    items: [
      [0, 0, "Golden Maple Leaf"],
      [1, 1, "Cloak of Verdant Spirit"],
      [1, 3, "Binary Star"],
      [1, 5, "Mouse Mage’s Pendant"],
      [3, 0, "Foundation"],
      [3, 1, "Silver Plate"],
      [3, 2, "Alfonso’s Horn"],
      [3, 3, "Charm of Skill"],
      [3, 4, "Crown of Humility"],
      [3, 5, "Keel Fragment"],
    ],
  },
  {
    file: "15-40-28.png",
    origin: [248, 98],
    items: [
      [0, 0, "Golden Maple Leaf"],
      [0, 1, "Blue Planet"],
      [1, 1, "Alfonso’s Horn"],
      [1, 3, "Binary Star"],
      [1, 5, "Cloak of Verdant Spirit"],
      [2, 1, "Harvest"],
      [3, 0, "Foundation"],
      [3, 1, "Eternal Winter"],
      [3, 2, "Silver Plate"],
      [3, 3, "Charm of Skill"],
      [3, 4, "Crown of Humility"],
      [3, 5, "Keel Fragment"],
    ],
  },
  {
    file: "15-48-31.png",
    origin: [248, 98],
    items: [
      [0, 0, "Golden Maple Leaf"],
      [0, 1, "Power"],
      [1, 1, "Alfonso’s Horn"],
      [1, 2, "Advent"],
      [1, 3, "Binary Star"],
      [1, 5, "Cloak of Verdant Spirit"],
      [2, 1, "Golden Handbell"],
      [2, 2, "Harvest"],
      [3, 0, "Foundation"],
      [3, 1, "Silver Plate"],
      [3, 2, "Broken Sapphire"],
      [3, 3, "Charm of Skill"],
      [3, 4, "Crown of Humility"],
      [3, 5, "Keel Fragment"],
    ],
    empty: [[2, 0]],
  },
  {
    file: "16-00-29.png",
    origin: [248, 98],
    items: [
      [0, 0, "Golden Maple Leaf"],
      [0, 1, "Power"],
      [0, 4, "Battle Bracelet of Lerid"],
      [1, 1, "Alfonso’s Horn"],
      [1, 2, "Advent"],
      [1, 3, "Binary Star"],
      [1, 4, "Magic Glasses"],
      [1, 5, "Cloak of Verdant Spirit"],
      [2, 1, "Golden Handbell"],
      [2, 2, "Harvest"],
      [2, 4, "Pointed Club"],
      [3, 0, "Foundation"],
      [3, 1, "Silver Plate"],
      [3, 2, "Broken Sapphire"],
      [3, 3, "Charm of Skill"],
      [3, 4, "Crown of Humility"],
      [3, 5, "Keel Fragment"],
    ],
  },
  {
    file: "16-08-37.png",
    origin: [248, 98],
    items: [
      [0, 0, "Leaf Bird Leather"],
      [0, 1, "Alfonso’s Horn"],
      [0, 2, "Keel Fragment"],
      [0, 3, "Exploitation"],
      [0, 4, "Foundation"],
      [0, 5, "Charm of Skill"],
      [1, 0, "Advent"],
      [1, 1, "Disconnection"],
      [1, 3, "Golden Maple Leaf"],
      [1, 4, "Peace"],
      [1, 5, "Magic Glasses"],
      [2, 0, "Harvest"],
      [2, 1, "Cloak of Verdant Spirit"],
      [2, 3, "Binary Star"],
      [2, 5, "Broken Sapphire"],
      [3, 0, "Silver Plate"],
      [3, 2, "Crown of Humility"],
      [3, 3, "Pointed Club"],
    ],
    empty: [[3, 5]],
  },
  {
    file: "image.png",
    origin: [242, 98],
    items: [
      [0, 0, "Foundation"],
      [0, 1, "Mouse Mage’s Pendant"],
      [0, 2, "Eternal Furnace"],
      [0, 3, "Solis Parvo"],
      [0, 4, "Windgrass Scarf"],
      [0, 5, "Frozen Bow"],
      [1, 0, "Past"],
      [1, 1, "Ahmad’s Spirit Powder"],
      [1, 2, "Begonia Sachet"],
      [1, 5, "Simultaneity"],
      [2, 1, "Thorns"],
      [3, 1, "Advance"],
    ],
    empty: [[3, 3]],
  },
  { file: "image2.png", origin: [160, 98], items: [[0, 0, "Foundation"]] },
] as const;

test("detects icon-only real item identities in all 12 screenshots", async (context) => {
  const detector = await createInventoryDetector({ catalogPath });
  try {
    for (const expected of realCases) {
      await context.test(expected.file, async () => {
        const result = await detector.detect(
          join(projectRoot, "test", "fixtures", "real", "all", expected.file),
        );
        assert.equal(result.slots.length, 24);
        assert.deepEqual([result.slots[0]?.x, result.slots[0]?.y], expected.origin);

        for (const [row, column, name] of expected.items) {
          const slot = result.slots.find(
            (candidate) => candidate.row === row && candidate.column === column,
          );
          assert.equal(slot?.item?.name, name);
          assert.ok(slot.item.confidence >= 0.85);
          if (name === "Binary Star") assert.equal(slot.item.rotationDegrees, 90);
        }

        if ("empty" in expected) {
          for (const [row, column] of expected.empty) {
            const slot = result.slots.find(
              (candidate) => candidate.row === row && candidate.column === column,
            );
            assert.equal(slot?.item, null);
          }
        }
      });
    }
  } finally {
    detector.dispose();
  }
});
