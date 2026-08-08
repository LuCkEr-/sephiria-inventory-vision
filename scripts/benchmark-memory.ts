import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

import { createInventoryVisionDetector } from "../src/index.js";

const root = resolve(import.meta.dirname, "..");
const input = join(root, "test", "fixtures", "real", "all", "16-08-37.png");
const outputPath = resolve(process.argv[2] ?? join(root, "benchmarks", "memory.json"));
const iterations = 50;

function collectGarbage(): void {
  if (typeof global.gc !== "function") {
    throw new Error("Memory benchmark requires Node.js --expose-gc");
  }
  global.gc();
}

function snapshot() {
  const usage = process.memoryUsage();
  return {
    heapUsed: usage.heapUsed,
    external: usage.external,
    arrayBuffers: usage.arrayBuffers,
    rss: usage.rss,
  };
}

function delta(after: ReturnType<typeof snapshot>, before: ReturnType<typeof snapshot>) {
  return Object.fromEntries(
    Object.keys(before).map((key) => {
      const metric = key as keyof typeof before;
      return [metric, after[metric] - before[metric]];
    }),
  ) as ReturnType<typeof snapshot>;
}

function percentile(values: readonly number[], fraction: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.floor(sorted.length * fraction));
  return sorted[index] ?? 0;
}

const detector = await createInventoryVisionDetector();
try {
  for (let index = 0; index < 3; index += 1) await detector.detect(input);
  collectGarbage();
  const before = snapshot();
  const timings: number[] = [];
  for (let index = 0; index < iterations; index += 1) {
    const result = await detector.detect(input);
    timings.push(result.timingsMs.total);
  }
  collectGarbage();
  const after = snapshot();
  const growth = delta(after, before);
  const report = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    detector: "vision-features",
    fixture: "16-08-37.png",
    warmupRuns: 3,
    measuredRuns: iterations,
    timingMs: {
      median: percentile(timings, 0.5),
      p95: percentile(timings, 0.95),
      minimum: Math.min(...timings),
      maximum: Math.max(...timings),
    },
    memoryBytes: { before, after, growth },
    limitsBytes: {
      heapUsedGrowth: 8 * 1024 * 1024,
      externalGrowth: 8 * 1024 * 1024,
      arrayBufferGrowth: 8 * 1024 * 1024,
    },
  };
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(report, null, 2));
  if (growth.heapUsed > report.limitsBytes.heapUsedGrowth) {
    throw new Error(`Heap growth exceeded limit: ${growth.heapUsed} bytes`);
  }
  if (growth.arrayBuffers > report.limitsBytes.arrayBufferGrowth) {
    throw new Error(`ArrayBuffer growth exceeded limit: ${growth.arrayBuffers} bytes`);
  }
  if (growth.external > report.limitsBytes.externalGrowth) {
    throw new Error(`External memory growth exceeded limit: ${growth.external} bytes`);
  }
  if (report.timingMs.p95 > 1_000) {
    throw new Error(`Vision p95 latency exceeded 1,000 ms: ${report.timingMs.p95} ms`);
  }
} finally {
  detector.dispose();
}
