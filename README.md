# Sephiria Inventory Vision

A Node-first TypeScript library with a production augmented-vision-to-template cascade plus separately packaged comparison methods:

- `InventoryCascadeDetector`: recommended augmented vision primary with selective extracted-game-asset template verification.
- `InventoryVisionDetector`: learned structural and centered-RGB features with augmented nearest-neighbor classification.
- `InventoryDetector`: the original masked OpenCV template matcher.
- The `/lab` entry point contains DCT perceptual hash, Sobel edge/shape, oriented FAST/BRIEF, hybrid retrieval, CNN, and Siamese experiments.

The detector is tuned against 12 real 3440×1440 and 3840×2160 screenshots. Its bundled catalog contains 389 unique extracted item sprites mapped from 456 game item definitions, 141 metadata-gated rotation variants, 41 real-render calibration templates, and 13 inventory-slot frame variants.

Recognition is deliberately icon-only. The library does not read tooltips, labels, acquisition messages, stack-count text, or any text outside or inside the grid. Item names come from the game's extracted item metadata; screenshot matching samples only sprite-shaped icon pixels within each inventory cell.

## What it does

- Decodes PNG, JPEG, WebP, and other Sharp-supported screenshot formats.
- Normalizes screenshots to Sephiria's 288-pixel logical UI height for fast, resolution-independent matching.
- Locates the complete 6×4 grid using correlation, squared-difference evidence, and lattice scoring.
- Classifies complete 32×32 inventory cells with a 1,395-reference, ten-variant machine-vision model.
- Keeps the original alpha-masked normalized squared-difference backend for comparison.
- Generates 90°, 180°, and 270° template variants only for items whose Unity prefab declares `rotatable: true`.
- Uses raw cosine confidence plus model-generated per-class impostor thresholds to reject empty or unknown slots.
- Returns slot coordinates, item IDs, names, confidence, ranked alternatives, and game-authored item rules.
- Accepts an explicit slot grid when automatic localization is not appropriate.

## Setup

```powershell
npm install
npm run verify
```

Node.js 20 or newer is required.

## Recommended cascade

```ts
import { createInventoryCascadeDetector } from "sephiria-inventory-vision";

const detector = await createInventoryCascadeDetector();
try {
  const result = await detector.detect("screenshot.png");
  console.log(result.cascade); // fallbackSlots, visionSlots, thresholds and timing
  for (const slot of result.slots) {
    console.log(slot.item?.name, slot.cascade.backend, slot.cascade.fallbackReason);
  }
} finally {
  detector.dispose();
}
```

The default policy accepts calibrated vision results at confidence `>= 0.96` with a top-two label margin `>= 0.03`. It verifies each accepted non-empty prediction against that item's extracted game sprite, while uncertain cells and asset mismatches receive a full-catalog template search. Set `verifyVisionMatches: false` only when minimum latency matters more than this safeguard. Thresholds are configurable through `cascadeConfidence`, `cascadeMargin`, and `templateItemThreshold`.

On the held-out `16-08-37` challenge, where every reference from that screenshot was removed and class thresholds were recalibrated from the remaining data only, the cascade achieved 24/24 cells with a three-run median of 710 ms end to end, compared with 17/24 and 154 ms for vision alone or 24/24 and 1,739 ms for templates alone. That is a 2.45× end-to-end speedup over template-only matching. Run the reproducible measurement with `npm run benchmark:cascade`; detailed results are in `benchmarks/cascade.json`.

The stronger 12-fold cross-validation removes every learned reference from each evaluated screenshot in turn and recalibrates the fold without held-out evidence. It currently passes 288/288 grid cells and 111/111 occupied-item identities, averaging 360 ms per screenshot. This includes classes absent from the fold's vision model: candidate asset verification catches confident visual aliases and the full game-asset catalog recovers the correct identity. Reproduce it with `npm run benchmark:cascade-cv`; per-fold evidence is in `benchmarks/cascade-cross-validation.json`.

## Code quality

Detector matrix ownership, normalization, and grid localization live in one shared runtime. The template and vision detectors contain only their identity-specific matching logic, while the cascade owns fallback policy and result merging. OpenCV is loaded lazily, paths are containment-checked, and all disposable APIs are idempotent.

The repository uses Prettier, strict type-aware ESLint, maximally strict project TypeScript, built-in Node coverage thresholds, package-boundary checks, Publint, Are the Types Wrong, clean-install verification, leakage-free cascade cross-validation, extracted-asset open-set evaluation, and memory regression limits as required checks. Maintained source is limited to 500 lines per file, complexity 15, nesting depth 4, 120 lines per function, 6 parameters, and 30 statements per function. The 60-test suite enforces floors of 99.5% lines, 92.5% branches, and 100% functions; the current Node coverage report is 99.61%, 92.99%, and 100%, respectively.

```powershell
npm run quality       # formatting check + lint + strict type-check
npm run verify        # quality + coverage + build + package API + clean-install smoke test
npm run verify:release # verify + held-out and 12-fold cascade + memory/latency + production audit
npm run format        # apply the canonical formatting
npm run benchmark:memory
```

## Automatic slot localization

```ts
import { createInventoryVisionDetector } from "sephiria-inventory-vision";

const detector = await createInventoryVisionDetector();

try {
  const result = await detector.detect("screenshot.png", {
    // Defaults are tuned for the supplied real screenshots.
    slotThreshold: 0.97,
    minGridSupport: 5,
    itemThreshold: 0.936,
    alternatives: 3,
  });

  for (const slot of result.matchedItems) {
    console.log({
      row: slot.row,
      column: slot.column,
      item: slot.item?.name,
      confidence: slot.item?.confidence,
      rotationDegrees: slot.item?.rotationDegrees,
      variants: slot.item?.itemVariants?.map(({ itemId, gameplay }) => ({
        itemId,
        maxBuffLevel: gameplay.maxBuffLevel,
        rotatable: gameplay.rotatable,
        placementRequirements: gameplay.placementRequirements,
        effectQuery: gameplay.effectQuery,
      })),
      bounds: { x: slot.x, y: slot.y, width: slot.width, height: slot.height },
    });
  }
} finally {
  detector.dispose();
}
```

Always call `dispose()` when finished. OpenCV.js matrices use WebAssembly memory rather than JavaScript's normal garbage collector.

## Comparison results

`npm run benchmark` runs both backends over all 12 supplied screenshots and writes `benchmarks/comparison.json`.

| Metric                                 |                       Template backend | Vision backend |
| -------------------------------------- | -------------------------------------: | -------------: |
| Average total time                     |                               1,656 ms |         226 ms |
| Average identity step                  |                               1,517 ms |        74.7 ms |
| Same-set agreement                     |                              reference |  288/288 cells |
| Leave-one-screenshot-out item accuracy | not applicable to calibrated templates |  92/93 (98.9%) |

The same-set result is a regression check, not a generalization claim: the bundled vision references include those screenshots. The leave-one-screenshot-out result removes every reference from the screenshot being tested. Eighteen item instances belong to classes appearing in only one screenshot, so they are reported as unseen rather than counted as successes.

The vision model currently knows 40 item classes plus empty cells. Unknown classes need additional labeled inventory-grid examples before the learned backend can identify them. The template backend remains useful for broad catalog coverage.

### All-method experiment

`npm run benchmark:methods` trains the learned methods without the challenge screenshot, evaluates all methods against the same frozen cells, and writes `benchmarks/all-methods.json` plus a compact Markdown table.

On the held-out `16-08-37` capture, all five classical retrieval methods correctly identified all 14 occupied cells whose classes existed in the other screenshots. Across complete leave-one-screenshot-out evaluation, the occupied-item scores were:

| Method                            | Correct covered items | Accuracy | Challenge identity time / cell |
| --------------------------------- | --------------------: | -------: | -----------------------------: |
| Structural + centered RGB         |                 92/93 |    98.9% |                       0.212 ms |
| Perceptual hash                   |                 84/93 |    90.3% |                       0.017 ms |
| Edge/shape                        |                 91/93 |    97.8% |                       0.043 ms |
| ORB-style                         |                 92/93 |    98.9% |                       0.277 ms |
| Hash + normalized pixel retrieval |                 90/93 |    96.8% |                       0.186 ms |

The deep-learning experiment is deliberately reported even where it failed. With only 111 occupied cells, the closed-set CNN classified 0/14 covered challenge items; using its penultimate layer as an embedding recovered 11/14. The contrastive Siamese embedding reached 14/14. The CNN-based dense-window detector produced no accepted boxes, so its localization recall was 0/18. These results show that a CNN/object detector needs a substantially larger independently captured training set, while classical and Siamese retrieval are already viable.

Use the classical method API on an already cropped 32×32 RGB icon cell:

```ts
import {
  CLASSICAL_METHODS,
  classifyMethodCell,
  prepareMethodCell,
} from "sephiria-inventory-vision/lab";

const references = labeledCells.map((cell) => prepareMethodCell(cell));
for (const method of CLASSICAL_METHODS) {
  const alternatives = classifyMethodCell(queryCell, references, method);
  console.log(method, alternatives[0]);
}
```

`trainTinyCnn`, `trainSiameseEmbedding`, and `nonMaximumSuppression` are also exported from the `/lab` entry point for controlled experiments. Keeping the lab separate prevents TensorFlow.js from loading when an application imports the production detectors.

TensorFlow.js is an optional peer dependency. Install it only when using the neural training functions:

```powershell
npm install @tensorflow/tfjs
```

## Evaluation suite

Version 0.2 replaces detector-to-detector assertions with a frozen, detector-independent ground-truth manifest covering every cell:

- 12 screenshots and 288 explicitly labeled cells.
- 111 occupied cells, 177 empty cells, and 40 item classes.
- Complete row/column coverage and fixture-presence validation.
- Direct evaluation of both backends against ground truth.
- Leave-one-screenshot-out evaluation: 92/93 covered item cells correct (98.9%).
- Capture-family holdout, which removes adjacent/near-duplicate screenshots together: 28/30 covered item cells correct (93.3%).
- Rotation holdout for all five Binary Star observations.
- A boundary test that deletes every pixel outside the inventory grid and requires unchanged results.
- Brightness, JPEG, nearest-neighbor resize, and filtered-resize measurements.
- Leave-one-class-out open-set measurement using the exact runtime score and threshold implementation.
- An independent open-set corpus containing all 461 extracted templates and rotations absent from the vision model.

Run only the evaluation suite with:

```powershell
npm run test:evaluation
```

The robustness and open-set targets are enforced tests rather than TODOs. Brightness, JPEG-80, 150% nearest resizing, and 150% cubic resizing each retain 24/24 auto-localized cells on the reviewed robustness capture. At the raw cosine default `0.936`, 19/111 class-withheld screenshot queries exceed the global threshold. The deployed full-model class calibration suppresses all 19; the stricter experiment that rebuilds calibration without the query class also suppresses all 19. In a separate unknown-item corpus, all 461 extracted templates and rotations absent from the model are rejected after being rendered into inventory-slot backgrounds. The cascade still independently verifies game assets because finite regression corpora cannot prove performance on every future screenshot or item. Reproduce that audit with `npm run benchmark:open-set`; evidence is in `benchmarks/open-set.json`.

## Explicit grid

Supplying the grid is faster and more reliable when the inventory origin and dimensions are already known:

```ts
const result = await detector.detect("screenshot.png", {
  grid: {
    x: 412,
    y: 238,
    rows: 5,
    columns: 8,
    slotSize: 32,
    gapX: 0,
    gapY: 0,
    scale: 1,
  },
  itemThreshold: 0.9,
});
```

You can also pass arbitrary rectangles through `slots`:

```ts
const result = await detector.detect("screenshot.png", {
  slots: [{ x: 412, y: 238, width: 34, height: 34, row: 0, column: 0 }],
});
```

## Command-line example

```powershell
npm run example -- "C:\path\to\screenshot.png"
```

The example automatically normalizes and locates the grid, then prints JSON.

## Confidence tuning

- Increase `slotThreshold` if non-inventory UI elements are reported as slots.
- Decrease `minGridSupport` only when fewer than five empty slot anchors are visible.
- Set `normalizationHeight: false` and supply `scales` to use the legacy multi-scale locator.
- Increase `itemThreshold` to reject more empty or unknown slots. Model-generated class thresholds remain a non-bypassable lower bound for visually ambiguous labels.
- Decrease it when screenshots have color grading, compression, or non-integer scaling; the bundled default is `0.936`.
- Prefer nearest-neighbor screenshots. Bilinear filtering changes pixel colors and weakens deterministic matching.

The vision detector includes ranked, name-deduplicated alternatives and records `nearestReferenceId` plus `nearestReferenceScreenshot` for auditability. `classification` also reports the effective class threshold and whether it was accepted. Unknown icons remain `null` when the empty class wins or confidence is below the detector-wide or class-specific threshold.

To use the original matcher from the same package:

```ts
import { createInventoryDetector } from "sephiria-inventory-vision";

const templateDetector = await createInventoryDetector();
```

## Gameplay metadata

Every match returns `itemVariants`, one entry per logical game item represented by the matched icon. Each variant includes:

- `maxBuffLevel` for charms (the serialized `Charm_Basic.maxLevel`; `0` is a real value).
- `rotatable` for stone tablets and `false` for charms.
- `rotationDegrees` on each match reports the detected clockwise orientation of the icon.
- `placementRequirements`, including top row, bottom row, outer edge, interior, empty neighbors, or charms on both sides when the item prefab declares one.
- `effectQuery` and `conditionQuery`, the serialized stone-tablet grid patterns.
- item type, rarity, unique-effect/weapon-related flags, component classes, and source-prefab provenance.

These values are extracted from Unity item prefabs. They do not come from screenshot tooltips, labels, or OCR. Shared icons retain all candidate variants instead of attaching one item's rules to another.

## Catalog regeneration

The generated catalog is bundled under `assets/catalog`. To rebuild it from the extracted Sephiria collection:

```powershell
npm run generate:catalog -- "C:\Program Files (x86)\Steam\steamapps\common\Sephiria\Extracted_Image_Assets\Inventory_Related_Textures"
```

The catalog generator joins extracted sprites to `assets/item-icon-map.json` and `assets/item-gameplay-map.json`, produced from Sephiria's Unity item definitions, prefabs, and English localization data. Icons shared by multiple logical items are marked as ambiguous instead of being assigned a false unique identity.

Real-render templates can be regenerated from the icon-only calibration manifest:

```powershell
npm run generate:curated -- "C:\dev\experiments\sephiria-inventory-screenshots" "C:\Program Files (x86)\Steam\steamapps\common\Sephiria\Extracted_Image_Assets"
```

Rebuild the machine-vision reference model from labeled screenshots with:

```powershell
npm run generate:vision -- "C:\dev\experiments\sephiria-inventory-screenshots"
```

## Current limitations

- The vision model identifies only classes represented in its labeled reference set. It now has measured unknown-item rejection, but cannot name an unseen class; use the cascade for full-catalog recovery.
- Some logical items share one sprite. Those catalog entries expose their candidate display names and set `ambiguousIdentity: true` rather than claiming an exact identity.
- The 12 supplied screenshots all produce the correct 24-slot grid, but additional UI revisions may require threshold tuning.
- Stack counts and all other text are intentionally ignored.
- Item rarity and lock-state overlays are not returned separately.
- The package is currently Node-first. OpenCV.js itself supports browser use, but this implementation uses Sharp and Node filesystem APIs.

## Verification

The test suite covers all three production detectors, synthetic item recognition, legacy 2× matching, the frozen 288-cell corpus, leakage-resistant holdouts, input transformations, class-aware rejection, all 461 unseen extracted templates, native allocation failures, package boundaries, and all 12 committed logical-resolution fixtures. No OCR, tooltip crop, or label assertion participates in detection or expected identity generation.

```powershell
npm run verify
```

The extracted game artwork remains the property of its respective rights holders. Keep the bundled catalog for personal analysis or modding unless you have permission to redistribute it.
