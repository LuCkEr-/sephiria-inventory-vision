import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";

const execute = promisify(execFile);
const root = resolve(import.meta.dirname, "..");
const sourcePackage = JSON.parse(await readFile(join(root, "package.json"), "utf8")) as {
  name: string;
  version: string;
};
const temporaryRoot = await mkdtemp(join(tmpdir(), "sephiria-install-verification-"));

function requireNpmCli(value: string | undefined): string {
  if (!value) throw new Error("npm_execpath is required for clean-install verification");
  return value;
}

const npmCli = requireNpmCli(process.env["npm_execpath"]);

async function runNpm(arguments_: readonly string[], cwd: string): Promise<void> {
  await execute(process.execPath, [npmCli, ...arguments_], { cwd });
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

try {
  await runNpm(["pack", "--silent", "--pack-destination", temporaryRoot], root);
  const tarballs = (await readdir(temporaryRoot)).filter((file) => file.endsWith(".tgz"));
  assert.equal(tarballs.length, 1, "npm pack must produce exactly one tarball");
  const [tarballName] = tarballs;
  assert.ok(tarballName);
  const tarball = join(temporaryRoot, tarballName);
  await runNpm(
    ["install", "--silent", "--ignore-scripts", "--no-audit", "--no-fund", "--omit=dev", tarball],
    temporaryRoot,
  );

  const installedRoot = join(temporaryRoot, "node_modules", "sephiria-inventory-vision");
  const installedPackage = JSON.parse(
    await readFile(join(installedRoot, "package.json"), "utf8"),
  ) as { name: string; version: string };
  assert.equal(installedPackage.name, sourcePackage.name);
  assert.equal(installedPackage.version, sourcePackage.version);
  assert.equal(
    await pathExists(join(temporaryRoot, "node_modules", "@tensorflow", "tfjs")),
    false,
    "the optional TensorFlow peer must not be installed for production consumers",
  );
  for (const excluded of ["src", "scripts", "test", "tsconfig.json", "package-lock.json"]) {
    assert.equal(
      await pathExists(join(installedRoot, excluded)),
      false,
      `${excluded} must not leak into the production package`,
    );
  }

  const manifest = JSON.parse(
    await readFile(join(root, "test", "fixtures", "real", "ground-truth.json"), "utf8"),
  ) as {
    screenshots: {
      fixtureFile: string;
      slots: { item: { name: string } | null }[];
    }[];
  };
  const capture = manifest.screenshots.find(
    (candidate) => candidate.fixtureFile === "16-08-37.png",
  );
  assert.ok(capture);
  const input = join(root, "test", "fixtures", "real", "all", capture.fixtureFile);
  const expected = capture.slots.map((slot) => slot.item?.name ?? null);
  const smokePath = join(temporaryRoot, "smoke.mjs");
  await writeFile(
    smokePath,
    `import assert from "node:assert/strict";
import * as production from ${JSON.stringify(pathToFileURL(join(installedRoot, "dist", "index.js")).href)};
import * as lab from ${JSON.stringify(pathToFileURL(join(installedRoot, "dist", "lab.js")).href)};
assert.equal(lab.CLASSICAL_METHODS.length, 5);
await assert.rejects(
  lab.trainTinyCnn([{ label: "test", pixels: new Uint8Array(32 * 32 * 3) }], { epochs: 1 }),
  new RegExp("optional @tensorflow/tfjs peer dependency"),
);
const detector = await production.createInventoryCascadeDetector();
try {
  const result = await detector.detect(${JSON.stringify(input)});
  assert.deepEqual(result.slots.map((slot) => slot.item?.name ?? null), ${JSON.stringify(expected)});
} finally {
  detector.dispose();
}
console.log(JSON.stringify({ productionDetection: "24/24", labMethods: lab.CLASSICAL_METHODS.length, optionalPeerError: true }));
`,
    "utf8",
  );
  const smoke = await execute(process.execPath, [smokePath], { cwd: temporaryRoot });
  const smokeResult = JSON.parse(smoke.stdout.trim()) as {
    productionDetection: string;
    labMethods: number;
    optionalPeerError: boolean;
  };

  console.log(
    JSON.stringify(
      {
        package: `${installedPackage.name}@${installedPackage.version}`,
        optionalTensorFlowInstalled: false,
        productionDetection: smokeResult.productionDetection,
        labMethods: smokeResult.labMethods,
        optionalPeerError: smokeResult.optionalPeerError,
      },
      null,
      2,
    ),
  );
} finally {
  await rm(temporaryRoot, { recursive: true, force: true });
}
