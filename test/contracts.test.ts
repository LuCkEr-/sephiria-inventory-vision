import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

import { loadCatalog } from "../src/catalog.js";
import { assertFourDecodedChannels } from "../src/image.js";
import { getOpenCv, initializeOpenCvModule, type OpenCv } from "../src/opencv.js";
import {
  imageToRgbaMat,
  loadTemplate,
  prepareDetectionContext,
  type DecodedMat,
} from "../src/runtime.js";
import {
  cosineSimilarity,
  createGridSlots,
  createInventoryVisionDetector,
  extractVisionFeatures,
  type CatalogTemplate,
  type VisionModel,
} from "../src/index.js";
import {
  DEFAULT_VISION_ACCEPTANCE_THRESHOLD,
  calibrateVisionAcceptanceThresholds,
  countVisionLabelSources,
  effectiveVisionAcceptanceThreshold,
  evaluateVisionAcceptance,
  visionConfidence,
} from "../src/vision-calibration.js";

const projectRoot = resolve(import.meta.dirname, "..");
const fixture = join(projectRoot, "test", "fixtures", "real", "all", "16-08-37.png");

test("native module and decoded-channel boundaries cover every supported runtime shape", async () => {
  assertFourDecodedChannels(4);
  assert.throws(() => {
    assertFourDecodedChannels(3);
  }, /Expected four decoded channels/);

  const ready = { Mat: Array } as unknown as OpenCv;
  assert.equal((await initializeOpenCvModule(ready)).cv, ready);

  const promised = Promise.resolve(ready);
  assert.equal((await initializeOpenCvModule(promised)).cv, ready);

  const delayed = {} as OpenCv;
  const initialization = initializeOpenCvModule(delayed);
  await Promise.resolve();
  assert.equal(typeof delayed.onRuntimeInitialized, "function");
  delayed.onRuntimeInitialized();
  assert.equal((await initialization).cv, delayed);
});

test("grid construction rejects ambiguous or unsafe geometry", () => {
  assert.throws(
    () => createGridSlots({ x: -1, y: 0, rows: 1, columns: 1 }),
    /grid\.x must be a finite non-negative number/,
  );
  assert.throws(
    () => createGridSlots({ x: 0, y: 0, rows: 0, columns: 1 }),
    /grid\.rows must be a positive integer/,
  );
  assert.throws(
    () => createGridSlots({ x: 0, y: 0, rows: 1, columns: 1, slotSize: 0 }),
    /grid\.slotSize must be positive and finite/,
  );
  assert.throws(
    () => createGridSlots({ x: 0, y: 0, rows: 1, columns: 1, gapX: -1 }),
    /grid\.gapX must be a finite non-negative number/,
  );
  assert.throws(
    () => createGridSlots({ x: 0, y: 0, rows: 1, columns: 1, scale: Number.NaN }),
    /grid\.scale must be positive and finite/,
  );

  assert.deepEqual(createGridSlots({ x: 2, y: 3, rows: 1, columns: 2, slotSize: 10 }), [
    { x: 2, y: 3, width: 10, height: 10, row: 0, column: 0, scale: 1 },
    { x: 12, y: 3, width: 10, height: 10, row: 0, column: 1, scale: 1 },
  ]);
});

test("detector options fail fast with actionable contract errors", async () => {
  const detector = await createInventoryVisionDetector();
  try {
    await assert.rejects(
      detector.detect(fixture, {
        slots: [{ x: 0, y: 0, width: 32, height: 32 }],
        grid: { x: 0, y: 0, rows: 1, columns: 1 },
      }),
      /slots and grid are mutually exclusive/,
    );
    await assert.rejects(detector.detect(fixture, { scales: [] }), /scales must contain/);
    await assert.rejects(
      detector.detect(fixture, { scales: [1, Number.POSITIVE_INFINITY] }),
      /positive finite values/,
    );
    await assert.rejects(detector.detect(fixture, { alternatives: 0 }), /positive integer/);
    await assert.rejects(detector.detect(fixture, { maxSlots: 1.5 }), /positive integer/);
    await assert.rejects(
      detector.detect(fixture, { slotBorderWidth: 0 }),
      /slotBorderWidth must be a positive integer/,
    );
    await assert.rejects(detector.detect(fixture, { itemThreshold: -0.1 }), /between 0 and 1/);
    await assert.rejects(
      detector.detect(fixture, { emptySlotThreshold: Number.NaN }),
      /emptySlotThreshold must be a finite number/,
    );
    await assert.rejects(
      detector.detect(fixture, { normalizationHeight: 0 }),
      /normalizationHeight must be positive/,
    );
    await assert.rejects(
      detector.detect(fixture, { gridRows: 2, gridColumns: 2, minGridSupport: 5 }),
      /cannot exceed grid capacity 4/,
    );
    await assert.rejects(detector.detect(fixture, { gridRows: 0 }), /gridRows.*positive integer/);
    await assert.rejects(
      detector.detect(fixture, { gridColumns: 0 }),
      /gridColumns.*positive integer/,
    );
    await assert.rejects(
      detector.detect(fixture, { minGridSupport: 0 }),
      /minGridSupport.*positive integer/,
    );
    await assert.rejects(
      detector.detect(fixture, { slotThreshold: 2 }),
      /slotThreshold must be a finite number/,
    );
    await assert.rejects(
      detector.detect(fixture, { slots: [{ x: -1, y: 0, width: 32, height: 32 }] }),
      /coordinates must be finite and non-negative/,
    );
    await assert.rejects(
      detector.detect(fixture, {
        slots: [{ x: 0, y: Number.NaN, width: 32, height: 32 }],
      }),
      /coordinates must be finite and non-negative/,
    );
    await assert.rejects(
      detector.detect(fixture, { slots: [{ x: 0, y: 0, width: 0, height: 32 }] }),
      /dimensions must be positive and finite/,
    );
    await assert.rejects(
      detector.detect(fixture, { slots: [{ x: 0, y: 0, width: 32, height: 32, row: -1 }] }),
      /row must be a non-negative integer/,
    );
    await assert.rejects(
      detector.detect(fixture, {
        slots: [{ x: 0, y: 0, width: 32, height: 32, column: 1.5 }],
      }),
      /column must be a non-negative integer/,
    );
    await assert.rejects(
      detector.detect(fixture, {
        slots: [{ x: 0, y: 0, width: 32, height: 32, scale: 0 }],
      }),
      /scale must be positive and finite/,
    );
  } finally {
    detector.dispose();
  }

  await assert.rejects(detector.detect(fixture), /has been disposed/);
  detector.dispose();
});

test("catalog loading validates schema and template records", async () => {
  const directory = await mkdtemp(join(tmpdir(), "sephiria-catalog-contract-"));
  try {
    const path = join(directory, "catalog.json");
    await writeFile(path, JSON.stringify({ schemaVersion: 99 }), "utf8");
    await assert.rejects(loadCatalog(path), /Unsupported or malformed catalog/);

    await writeFile(
      path,
      JSON.stringify({
        schemaVersion: 3,
        game: "Sephiria",
        nativeSlotSize: 32,
        items: [],
        slotTemplates: [],
      }),
      "utf8",
    );
    await assert.rejects(loadCatalog(path), /at least one slot template/);

    const validTopLevel = {
      schemaVersion: 3,
      game: "Sephiria",
      nativeSlotSize: 32,
      items: [],
      slotTemplates: [
        {
          id: "slot",
          name: "Slot",
          file: "slot.png",
          width: 1,
          height: 1,
          sourceFile: "test",
          pathId: 1,
        },
      ],
    };
    const rejectsTopLevel = async (catalog: unknown, message: RegExp): Promise<void> => {
      await writeFile(path, JSON.stringify(catalog), "utf8");
      await assert.rejects(loadCatalog(path), message);
    };
    await rejectsTopLevel({ ...validTopLevel, game: "" }, /game must be a non-empty string/);
    await rejectsTopLevel({ ...validTopLevel, nativeSlotSize: 0 }, /nativeSlotSize/);
    await rejectsTopLevel({ ...validTopLevel, items: {} }, /catalog\.items must be an array/);
    await rejectsTopLevel(
      { ...validTopLevel, slotTemplates: {} },
      /catalog\.slotTemplates must be an array/,
    );

    await writeFile(
      path,
      JSON.stringify({
        schemaVersion: 3,
        game: "Sephiria",
        generatedFrom: "",
        nativeSlotSize: 32,
        items: [],
        slotTemplates: [{}],
      }),
      "utf8",
    );
    await assert.rejects(loadCatalog(path), /generatedFrom must be non-empty/);

    await writeFile(
      path,
      JSON.stringify({
        schemaVersion: 3,
        game: "Sephiria",
        nativeSlotSize: 32,
        items: [
          {
            id: "duplicate",
            name: "One",
            file: "one.png",
            width: 1,
            height: 1,
            sourceFile: "test",
            pathId: 1,
          },
          {
            id: "duplicate",
            name: "Two",
            file: "two.png",
            width: 1,
            height: 1,
            sourceFile: "test",
            pathId: 2,
          },
        ],
        slotTemplates: [
          {
            id: "slot",
            name: "Slot",
            file: "slot.png",
            width: 1,
            height: 1,
            sourceFile: "test",
            pathId: 3,
          },
        ],
      }),
      "utf8",
    );
    await assert.rejects(loadCatalog(path), /duplicate id duplicate/);

    await writeFile(
      path,
      JSON.stringify({
        schemaVersion: 3,
        game: "Sephiria",
        nativeSlotSize: 32,
        items: [
          {
            id: "bad",
            name: "Bad",
            file: "bad.png",
            width: 0,
            height: 1,
            sourceFile: "test",
            pathId: 1,
          },
        ],
        slotTemplates: [
          {
            id: "slot",
            name: "Slot",
            file: "slot.png",
            width: 1,
            height: 1,
            sourceFile: "test",
            pathId: 2,
          },
        ],
      }),
      "utf8",
    );
    await assert.rejects(loadCatalog(path), /width must be a positive integer/);

    const slotTemplate = {
      id: "slot",
      name: "Slot",
      file: "slot.png",
      width: 1,
      height: 1,
      sourceFile: "test",
      pathId: 2,
    };
    const itemTemplate = {
      id: "item",
      name: "Item",
      file: "item.png",
      width: 1,
      height: 1,
      sourceFile: "test",
      pathId: 1,
    };
    const gameplay = {
      itemId: 1,
      displayName: "Item",
      itemType: "Charm",
      itemTypeValue: 1,
      rarity: 0,
      rotatable: false,
      maxBuffLevel: 0,
      placementRequirements: [],
      conditionQuery: null,
      effectQuery: null,
    };
    const catalogWithItem = (item: unknown) => ({
      schemaVersion: 3,
      game: "Sephiria",
      nativeSlotSize: 32,
      items: [item],
      slotTemplates: [slotTemplate],
    });
    const rejectsCatalog = async (item: unknown, message: RegExp): Promise<void> => {
      await writeFile(path, JSON.stringify(catalogWithItem(item)), "utf8");
      await assert.rejects(loadCatalog(path), message);
    };

    await rejectsCatalog(null, /catalog\.items\[0\] must be an object/);
    await rejectsCatalog({ ...itemTemplate, id: "" }, /id must be a non-empty string/);
    await rejectsCatalog({ ...itemTemplate, name: "" }, /name must be a non-empty string/);
    await rejectsCatalog({ ...itemTemplate, file: "" }, /file must be a non-empty string/);
    await rejectsCatalog({ ...itemTemplate, height: 0 }, /height must be a positive integer/);
    await rejectsCatalog({ ...itemTemplate, sourceFile: "" }, /sourceFile must be/);
    await rejectsCatalog({ ...itemTemplate, pathId: -1 }, /pathId must be/);
    await rejectsCatalog({ ...itemTemplate, spriteName: "" }, /spriteName must be/);
    await rejectsCatalog({ ...itemTemplate, displayNames: {} }, /displayNames must be/);
    await rejectsCatalog({ ...itemTemplate, displayNames: [""] }, /displayNames must be/);
    await rejectsCatalog({ ...itemTemplate, itemIds: {} }, /itemIds must contain/);
    await rejectsCatalog({ ...itemTemplate, itemIds: [-1] }, /itemIds must contain/);
    await rejectsCatalog({ ...itemTemplate, ambiguousIdentity: "yes" }, /ambiguousIdentity/);
    await rejectsCatalog({ ...itemTemplate, rotationDegrees: 45 }, /rotationDegrees/);
    await rejectsCatalog({ ...itemTemplate, canonicalTemplateId: "" }, /canonicalTemplateId/);
    await rejectsCatalog({ ...itemTemplate, itemVariants: {} }, /itemVariants must be an array/);
    await rejectsCatalog(
      { ...itemTemplate, itemVariants: [null] },
      /itemVariants\[0\] is malformed/,
    );

    const itemWithGameplay = (value: unknown) => ({
      ...itemTemplate,
      itemVariants: [{ itemId: 1, displayName: "Item", gameplay: value }],
    });
    await rejectsCatalog(itemWithGameplay(null), /gameplay must be an object/);
    await rejectsCatalog(itemWithGameplay({ ...gameplay, itemId: -1 }), /itemId/);
    await rejectsCatalog(itemWithGameplay({ ...gameplay, itemTypeValue: -1 }), /itemTypeValue/);
    await rejectsCatalog(itemWithGameplay({ ...gameplay, rarity: -1 }), /rarity/);
    await rejectsCatalog(itemWithGameplay({ ...gameplay, displayName: "" }), /displayName/);
    await rejectsCatalog(itemWithGameplay({ ...gameplay, itemType: "" }), /itemType/);
    await rejectsCatalog(itemWithGameplay({ ...gameplay, rotatable: "yes" }), /rotatable/);
    await rejectsCatalog(itemWithGameplay({ ...gameplay, maxBuffLevel: -1 }), /maxBuffLevel/);
    await rejectsCatalog(itemWithGameplay({ ...gameplay, conditionQuery: 1 }), /conditionQuery/);
    await rejectsCatalog(itemWithGameplay({ ...gameplay, effectQuery: 1 }), /effectQuery/);
    await rejectsCatalog(
      itemWithGameplay({ ...gameplay, placementRequirements: {} }),
      /placementRequirements must be an array/,
    );
    await rejectsCatalog(
      itemWithGameplay({ ...gameplay, placementRequirements: [null] }),
      /placementRequirements\[0\] must be an object/,
    );
    await rejectsCatalog(
      itemWithGameplay({
        ...gameplay,
        placementRequirements: [{ code: "", description: "Bad", sourceClass: "Test" }],
      }),
      /placementRequirements\[0\]\.code/,
    );
    await rejectsCatalog(
      itemWithGameplay({
        ...gameplay,
        placementRequirements: [{ code: "edge", description: "", sourceClass: "Test" }],
      }),
      /placementRequirements\[0\]\.description/,
    );
    await rejectsCatalog(
      itemWithGameplay({
        ...gameplay,
        placementRequirements: [{ code: "edge", description: "Bad", sourceClass: "" }],
      }),
      /placementRequirements\[0\]\.sourceClass/,
    );
    await rejectsCatalog(
      itemWithGameplay({ ...gameplay, isUniqueEffect: "yes" }),
      /isUniqueEffect/,
    );
    await rejectsCatalog(
      itemWithGameplay({ ...gameplay, isWeaponRelated: "yes" }),
      /isWeaponRelated/,
    );
    await rejectsCatalog(
      itemWithGameplay({ ...gameplay, includeConditionInBounds: "yes" }),
      /includeConditionInBounds/,
    );
    await rejectsCatalog(
      itemWithGameplay({ ...gameplay, effectStringKeys: [""] }),
      /effectStringKeys/,
    );
    await rejectsCatalog(
      itemWithGameplay({ ...gameplay, componentClasses: [""] }),
      /componentClasses/,
    );
    await rejectsCatalog(itemWithGameplay({ ...gameplay, sourcePrefab: {} }), /sourcePrefab/);
    await rejectsCatalog(
      itemWithGameplay({
        ...gameplay,
        sourcePrefab: { asset: "test", pathId: -1, name: "Prefab" },
      }),
      /sourcePrefab/,
    );
    await rejectsCatalog(
      itemWithGameplay({
        ...gameplay,
        sourcePrefab: { asset: "", pathId: 1, name: "Prefab" },
      }),
      /sourcePrefab/,
    );
    await rejectsCatalog(
      itemWithGameplay({
        ...gameplay,
        sourcePrefab: { asset: "test", pathId: 1, name: "" },
      }),
      /sourcePrefab/,
    );
    await rejectsCatalog(
      {
        ...itemTemplate,
        itemVariants: [{ itemId: 2, displayName: "Item", gameplay: { ...gameplay, itemId: 1 } }],
      },
      /identity is inconsistent/,
    );
    await rejectsCatalog(
      {
        ...itemTemplate,
        itemVariants: [
          { itemId: 1, displayName: "Wrong", gameplay: { ...gameplay, displayName: "Item" } },
        ],
      },
      /identity is inconsistent/,
    );
    await rejectsCatalog(
      { ...itemTemplate, canonicalTemplateId: "missing" },
      /unknown canonical template/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("template loading cannot escape the catalog directory", async () => {
  const { cv } = await getOpenCv();
  const metadata: CatalogTemplate = {
    id: "escape",
    name: "Escape",
    file: "../outside.png",
    width: 1,
    height: 1,
    sourceFile: "test",
    pathId: 1,
  };
  await assert.rejects(loadTemplate(cv, projectRoot, metadata), /escapes the catalog directory/);
});

test("normalized detection has a safe empty-catalog result", async () => {
  const { cv } = await getOpenCv();
  const decoded = await imageToRgbaMat(cv, fixture);
  const context = await prepareDetectionContext(cv, decoded, [], {});
  try {
    assert.deepEqual(context.slots, []);
    assert.equal(context.normalized, true);
    assert.ok(context.coordinateScale > 0);
  } finally {
    context.dispose();
    context.dispose();
  }
});

test("detection context releases every matrix when color conversion fails", async () => {
  class FakeMat {
    readonly data = new Uint8Array(4);
    deleteCalls = 0;

    delete(): void {
      this.deleteCalls += 1;
    }
  }

  const decodedMat = new FakeMat();
  const raw = { data: new Uint8Array(4), width: 1, height: 1, channels: 4 } as const;
  const decoded = { mat: decodedMat, width: 1, height: 1, raw } as unknown as DecodedMat;
  const allocations: FakeMat[] = [];
  const MatConstructor = class extends FakeMat {
    constructor(..._arguments: unknown[]) {
      super();
      allocations.push(this);
    }
  };

  let conversions = 0;
  const cv = {
    Mat: MatConstructor,
    COLOR_RGBA2RGB: 1,
    CV_8UC4: 2,
    cvtColor() {
      conversions += 1;
      throw new Error("injected conversion failure");
    },
  } as unknown as OpenCv;

  await assert.rejects(prepareDetectionContext(cv, decoded, [], {}), /injected conversion failure/);
  assert.equal(conversions, 1);
  assert.equal(decodedMat.deleteCalls, 1);
  assert.deepEqual(
    allocations.map((mat) => mat.deleteCalls),
    [1],
  );

  const normalizedDecodedMat = new FakeMat();
  const normalizedDecoded = {
    mat: normalizedDecodedMat,
    width: 1,
    height: 1,
    raw,
  } as unknown as DecodedMat;
  allocations.length = 0;
  conversions = 0;
  const normalizedCv = {
    ...cv,
    cvtColor() {
      conversions += 1;
      if (conversions === 2) throw new Error("injected normalization failure");
    },
  } as unknown as OpenCv;

  await assert.rejects(
    prepareDetectionContext(normalizedCv, normalizedDecoded, [], { normalizationHeight: 1 }),
    /injected normalization failure/,
  );
  assert.equal(conversions, 2);
  assert.equal(normalizedDecodedMat.deleteCalls, 1);
  assert.deepEqual(
    allocations.map((mat) => mat.deleteCalls),
    [1, 1, 1],
  );
});

test("vision feature contracts reject malformed cells and bound similarity", () => {
  assert.throws(() => extractVisionFeatures([], 31, 32), /require a 32x32 cell/);
  assert.throws(() => extractVisionFeatures([], 32, 32, 2), /require RGB pixels/);
  assert.throws(() => extractVisionFeatures([], 32, 32, 3), /require 3072 pixel values/);
  const black = new Uint8Array(32 * 32 * 3);
  assert.equal(extractVisionFeatures(black, 32, 32, 3).length, 1040);
  const invalid = new Float64Array(32 * 32 * 3);
  invalid[0] = Number.NaN;
  assert.throws(() => extractVisionFeatures(invalid, 32, 32, 3), /byte-valued RGB pixels/);
  invalid[0] = 0.5;
  assert.throws(() => extractVisionFeatures(invalid, 32, 32, 3), /byte-valued RGB pixels/);
  assert.equal(cosineSimilarity([2], [2]), 1);
  assert.equal(cosineSimilarity([-2], [2]), -1);
  assert.ok(Math.abs(cosineSimilarity([2, 2], [4, 0]) - Math.SQRT1_2) < 1e-12);
  assert.equal(cosineSimilarity([], []), 0);
  assert.throws(() => cosineSimilarity([1], [1, 2]), /equal vector lengths/);
  assert.throws(() => cosineSimilarity([Number.NaN], [1]), /finite vectors/);
});

test("vision confidence and class calibration share one bounded acceptance contract", () => {
  assert.equal(visionConfidence([-1], [1]), 0);
  assert.equal(visionConfidence([2], [2]), 1);
  assert.equal(effectiveVisionAcceptanceThreshold(0.8, "A", undefined), 0.8);
  assert.equal(effectiveVisionAcceptanceThreshold(0.8, "A", { A: 0.9 }), 0.9);
  const second = Math.sqrt(1 - 0.94 ** 2);
  const references: VisionModel["references"] = [
    {
      id: "a",
      sourceScreenshot: "test.png",
      row: 0,
      column: 0,
      label: "A",
      item: null,
      features: [1, 0],
      variant: "base",
    },
    {
      id: "b",
      sourceScreenshot: "test.png",
      row: 0,
      column: 1,
      label: "B",
      item: null,
      features: [0.94, second],
      variant: "base",
    },
  ];
  assert.deepEqual(calibrateVisionAcceptanceThresholds(references), {
    A: 0.940001,
    B: 0.940001,
  });
  assert.deepEqual(calibrateVisionAcceptanceThresholds(references, 0.95), {});
  assert.deepEqual(
    [...countVisionLabelSources(references)],
    [
      ["A", 1],
      ["B", 1],
    ],
  );
  assert.deepEqual(evaluateVisionAcceptance("__empty__", 1, 1, 0.9, 1), {
    accepted: false,
    minimumMargin: 0,
    reason: "empty",
  });
  assert.deepEqual(evaluateVisionAcceptance("A", 0.9, 1, 0.936, 1), {
    accepted: false,
    minimumMargin: 0.04,
    reason: "low-confidence",
  });
  assert.deepEqual(evaluateVisionAcceptance("A", 0.95, 0.03, 0.936, 1), {
    accepted: false,
    minimumMargin: 0.04,
    reason: "singleton-margin",
  });
  assert.deepEqual(evaluateVisionAcceptance("A", 0.99, 0, 0.936, 1), {
    accepted: true,
    minimumMargin: 0,
    reason: "accepted",
  });
  assert.equal(DEFAULT_VISION_ACCEPTANCE_THRESHOLD, 0.936);
});

test("vision model validation rejects inconsistent metadata and feature data", async () => {
  const directory = await mkdtemp(join(tmpdir(), "sephiria-model-contract-"));
  try {
    const path = join(directory, "model.json");
    const unitFeatures = Array.from({ length: 1040 }, (_, index) => (index === 0 ? 1 : 0));
    const emptyReference: VisionModel["references"][number] = {
      id: "empty",
      sourceScreenshot: "test.png",
      row: 0,
      column: 0,
      label: "__empty__",
      item: null,
      features: unitFeatures,
      variant: "base",
    };
    const valid: VisionModel = {
      schemaVersion: 1,
      method: "test-model",
      inputPolicy: "test",
      slotSize: 32,
      generatedAt: new Date(0).toISOString(),
      sourceScreenshots: ["test.png"],
      labels: ["__empty__"],
      references: [emptyReference],
      augmentationPolicy: ["base"],
    };
    const rejectsModel = async (model: unknown, message: RegExp): Promise<void> => {
      await writeFile(path, JSON.stringify(model), "utf8");
      await assert.rejects(createInventoryVisionDetector({ modelPath: path }), message);
    };

    await rejectsModel({ ...valid, inputPolicy: "" }, /Unsupported or malformed vision model/);
    await rejectsModel({ ...valid, schemaVersion: 2 }, /Unsupported or malformed vision model/);
    await rejectsModel({ ...valid, slotSize: 16 }, /Unsupported or malformed vision model/);
    await rejectsModel({ ...valid, method: "" }, /Unsupported or malformed vision model/);
    await rejectsModel(
      { ...valid, generatedAt: "invalid" },
      /Unsupported or malformed vision model/,
    );
    await rejectsModel({ ...valid, references: [] }, /at least one reference/);
    await rejectsModel({ ...valid, labels: {} }, /labels are missing/);
    await rejectsModel({ ...valid, labels: [] }, /labels are missing/);
    await rejectsModel({ ...valid, labels: [""] }, /labels are missing/);
    await rejectsModel({ ...valid, labels: ["Item"] }, /lack __empty__/);
    await rejectsModel(
      { ...valid, labels: ["__empty__", "__empty__"] },
      /labels are missing, duplicated, or lack __empty__/,
    );
    await rejectsModel(
      { ...valid, acceptanceThresholds: [] },
      /acceptanceThresholds are malformed/,
    );
    await rejectsModel(
      { ...valid, acceptanceThresholds: { __empty__: 0.94 } },
      /acceptance threshold for __empty__/,
    );
    await rejectsModel(
      { ...valid, acceptanceThresholds: { missing: 0.94 } },
      /acceptance threshold for missing/,
    );
    await rejectsModel(
      {
        ...valid,
        labels: ["__empty__", "Item"],
        acceptanceThresholds: { Item: -0.01 },
      },
      /acceptance threshold for Item/,
    );
    await rejectsModel(
      {
        ...valid,
        labels: ["__empty__", "Item"],
        acceptanceThresholds: { Item: 1.01 },
      },
      /acceptance threshold for Item/,
    );
    await rejectsModel(
      {
        ...valid,
        labels: ["__empty__", "Item"],
        acceptanceThresholds: { Item: Number.NaN },
      },
      /acceptance threshold for Item/,
    );
    await rejectsModel(
      { ...valid, sourceScreenshots: ["test.png", "test.png"] },
      /sourceScreenshots are missing or duplicated/,
    );
    await rejectsModel({ ...valid, sourceScreenshots: [] }, /sourceScreenshots are missing/);
    await rejectsModel({ ...valid, sourceScreenshots: [""] }, /sourceScreenshots are missing/);
    await rejectsModel(
      { ...valid, augmentationPolicy: ["unsupported"] },
      /augmentationPolicy is malformed/,
    );
    await rejectsModel({ ...valid, augmentationPolicy: {} }, /augmentationPolicy is malformed/);
    await rejectsModel(
      { ...valid, augmentationPolicy: ["base", "base"] },
      /augmentationPolicy is malformed/,
    );
    await rejectsModel({ ...valid, references: [null] }, /malformed reference/);
    await rejectsModel(
      { ...valid, references: [emptyReference, { ...emptyReference, row: 1 }] },
      /duplicate reference id/,
    );
    await rejectsModel(
      { ...valid, references: [{ ...emptyReference, sourceScreenshot: "unknown.png" }] },
      /unknown source screenshot/,
    );
    await rejectsModel(
      { ...valid, references: [{ ...emptyReference, row: -1 }] },
      /invalid grid coordinates/,
    );
    await rejectsModel(
      { ...valid, references: [{ ...emptyReference, label: "Unknown" }] },
      /uses unknown label/,
    );
    await rejectsModel(
      { ...valid, references: [{ ...emptyReference, features: [Number.NaN] }] },
      /invalid feature vector/,
    );
    await rejectsModel(
      { ...valid, references: [{ ...emptyReference, features: Array(1040).fill(0) }] },
      /must have unit length/,
    );
    await rejectsModel(
      { ...valid, references: [{ ...emptyReference, item: { name: "Wrong" } }] },
      /inconsistent item identity/,
    );
    const itemReference: VisionModel["references"][number] = {
      ...emptyReference,
      id: "item",
      column: 1,
      label: "Item",
      item: {
        itemId: "item",
        name: "Item",
        confidence: 1,
        offset: { x: 0, y: 0 },
      },
    };
    await rejectsModel(
      {
        ...valid,
        labels: ["__empty__", "Item"],
        references: [
          emptyReference,
          { ...itemReference, item: { ...itemReference.item, confidence: 2 } },
        ],
      },
      /item does not match its label or schema/,
    );
    await rejectsModel(
      {
        ...valid,
        labels: ["__empty__", "Item"],
        references: [
          emptyReference,
          { ...itemReference, item: { ...itemReference.item, itemId: "" } },
        ],
      },
      /item does not match its label or schema/,
    );
    await rejectsModel(
      {
        ...valid,
        labels: ["__empty__", "Item"],
        references: [
          emptyReference,
          { ...itemReference, item: { ...itemReference.item, offset: null } },
        ],
      },
      /item does not match its label or schema/,
    );
    await rejectsModel(
      { ...valid, references: [{ ...emptyReference, variant: "unsupported" }] },
      /unsupported variant/,
    );
    await rejectsModel(
      {
        ...valid,
        references: [{ ...emptyReference, variant: "jpeg-80" }],
        augmentationPolicy: ["base"],
      },
      /outside augmentationPolicy/,
    );
    await rejectsModel({ ...valid, labels: ["__empty__", "Unused"] }, /labels without references/);

    await writeFile(path, JSON.stringify(valid), "utf8");
    const detector = await createInventoryVisionDetector({ modelPath: path });
    detector.dispose();
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
