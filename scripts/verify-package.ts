import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = resolve(import.meta.dirname, "..");
const packageJson = JSON.parse(await readFile(join(root, "package.json"), "utf8")) as {
  name: string;
  version: string;
  sideEffects: boolean;
  exports: Record<string, { types: string; import: string }>;
};

assert.equal(packageJson.sideEffects, false, "package must remain tree-shaking safe");
assert.deepEqual(Object.keys(packageJson.exports).sort(), [".", "./browser", "./lab"]);
for (const [name, entry] of Object.entries(packageJson.exports)) {
  await readFile(resolve(root, entry.types));
  await readFile(resolve(root, entry.import));
  assert.ok(entry.types.endsWith(".d.ts"), `${name} must declare a type entry`);
  assert.ok(entry.import.endsWith(".js"), `${name} must declare an ESM entry`);
}

const production = (await import(pathToFileURL(join(root, "dist", "index.js")).href)) as Record<
  string,
  unknown
>;
const lab = (await import(pathToFileURL(join(root, "dist", "lab.js")).href)) as Record<
  string,
  unknown
>;
const browser = (await import(pathToFileURL(join(root, "dist", "browser.js")).href)) as Record<
  string,
  unknown
>;

for (const name of [
  "createInventoryCascadeDetector",
  "createInventoryDetector",
  "createInventoryVisionDetector",
  "createGridSlots",
]) {
  assert.equal(typeof production[name], "function", `Missing production export ${name}`);
}
for (const name of ["CLASSICAL_METHODS", "trainSiameseEmbedding", "trainTinyCnn"]) {
  assert.equal(production[name], undefined, `Experimental export leaked into root: ${name}`);
  assert.notEqual(lab[name], undefined, `Missing lab export ${name}`);
}
for (const name of ["extractVisionFeatures", "rankVisionFeatures"]) {
  assert.equal(typeof browser[name], "function", `Missing browser export ${name}`);
}
const browserEntry = await readFile(join(root, "dist", "browser.js"), "utf8");
assert.doesNotMatch(browserEntry, /node:|sharp|tensorflow|detector|opencv/);

const rootEntry = await readFile(join(root, "dist", "index.js"), "utf8");
assert.doesNotMatch(rootEntry, /tensorflow|neural-methods|method-lab/);
const detectorTypes = await readFile(join(root, "dist", "detector.d.ts"), "utf8");
assert.doesNotMatch(detectorTypes, /detectWithCandidateVerification/);
const imageTypes = await readFile(join(root, "dist", "image.d.ts"), "utf8");
const openCvTypes = await readFile(join(root, "dist", "opencv.d.ts"), "utf8");
assert.doesNotMatch(imageTypes, /assertFourDecodedChannels/);
assert.doesNotMatch(openCvTypes, /initializeOpenCvModule/);

console.log(
  JSON.stringify(
    {
      package: packageJson.name,
      version: packageJson.version,
      productionExports: Object.keys(production).length,
      labExports: Object.keys(lab).length,
      entrypoints: Object.keys(packageJson.exports),
    },
    null,
    2,
  ),
);
