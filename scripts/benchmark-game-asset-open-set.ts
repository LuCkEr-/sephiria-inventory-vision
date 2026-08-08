import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import { evaluateGameAssetOpenSet } from "./lib/game-asset-open-set.js";

const outputPath = resolve(
  process.argv[2] ?? resolve(import.meta.dirname, "..", "benchmarks", "open-set.json"),
);
const report = await evaluateGameAssetOpenSet();
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
if (report.accepted > 0) process.exitCode = 1;
