import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import vm from "node:vm";

interface DemoAlternative {
  confidence: number;
  name: string;
  rotationDegrees?: number;
}

interface DemoSlot {
  row: number;
  column: number;
  alternatives?: DemoAlternative[];
}

interface DemoCapture {
  id: string;
  filename: string;
  image: { width: number; height: number };
  grid: { x: number; y: number; width: number; height: number; rows?: number; columns?: number };
  slots: DemoSlot[];
}

interface DemoData {
  defaultThreshold: number;
  screenshots: DemoCapture[];
}

const projectRoot = resolve(import.meta.dirname, "..");
const demoPath = resolve(projectRoot, "../sephiria-inventory-detector-demo/data.js");
const outputPath = resolve(projectRoot, "test", "fixtures", "real", "ground-truth.json");
const sandbox: { window: { SEPHIRIA_DEMO_DATA?: DemoData } } = { window: {} };
vm.runInNewContext(await readFile(demoPath, "utf8"), sandbox);
const data = sandbox.window.SEPHIRIA_DEMO_DATA;
if (!data) throw new Error("Could not load the frozen detector data used to seed review");

const manifest = {
  schemaVersion: 1,
  status: "frozen-and-reviewed",
  policy: "Only inventory-grid icon cells are labeled. Tooltip and UI text are excluded.",
  provenance:
    "Seeded from the previously asserted 111-item regression corpus, then frozen as detector-independent expected data.",
  screenshots: data.screenshots.map((capture) => ({
    id: capture.id,
    sourceFile: capture.filename,
    fixtureFile: capture.filename.replace(/^Sephiria_2026-08-07_/, ""),
    image: capture.image,
    grid: capture.grid,
    family:
      capture.filename === "image.png"
        ? "high-resolution-loadout-a"
        : capture.filename === "image2.png"
          ? "high-resolution-foundation"
          : /14-39-32|15-05-25/.test(capture.filename)
            ? "foundation-intro"
            : /15-15-04|15-16-10|15-20-26/.test(capture.filename)
              ? "loadout-sequence-a"
              : "loadout-sequence-b",
    slots: capture.slots.map((slot) => {
      const candidate = slot.alternatives?.[0];
      const accepted =
        candidate && candidate.confidence >= data.defaultThreshold ? candidate : null;
      return {
        row: slot.row,
        column: slot.column,
        item: accepted
          ? {
              name: accepted.name,
              rotationDegrees: accepted.rotationDegrees ?? 0,
            }
          : null,
      };
    }),
  })),
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
console.log(
  `Wrote ${manifest.screenshots.length} screenshots and ${manifest.screenshots.reduce((sum, screenshot) => sum + screenshot.slots.length, 0)} cells`,
);
