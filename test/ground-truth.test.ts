import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import test from "node:test";

const projectRoot = resolve(import.meta.dirname, "..");
const fixtureRoot = join(projectRoot, "test", "fixtures", "real", "all");
const manifestPath = join(projectRoot, "test", "fixtures", "real", "ground-truth.json");

test("ground-truth corpus independently covers every inventory cell", async () => {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as {
    schemaVersion: number;
    status: string;
    screenshots: {
      fixtureFile: string;
      family: string;
      grid: { rows: number; columns: number };
      slots: { row: number; column: number; item: { name: string } | null }[];
    }[];
  };

  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.status, "frozen-and-reviewed");
  assert.equal(manifest.screenshots.length, 12);
  assert.equal(new Set(manifest.screenshots.map((screenshot) => screenshot.family)).size, 5);

  const labels: string[] = [];
  for (const screenshot of manifest.screenshots) {
    await access(join(fixtureRoot, screenshot.fixtureFile));
    assert.equal(screenshot.grid.rows, 4);
    assert.equal(screenshot.grid.columns, 6);
    assert.equal(screenshot.slots.length, 24, screenshot.fixtureFile);
    const coordinates = new Set(screenshot.slots.map((slot) => `${slot.row}:${slot.column}`));
    assert.equal(coordinates.size, 24, screenshot.fixtureFile);
    for (let row = 0; row < 4; row += 1) {
      for (let column = 0; column < 6; column += 1) {
        assert.ok(
          coordinates.has(`${row}:${column}`),
          `${screenshot.fixtureFile} R${row + 1} C${column + 1}`,
        );
      }
    }
    for (const slot of screenshot.slots) if (slot.item) labels.push(slot.item.name);
  }

  assert.equal(labels.length, 111);
  assert.equal(288 - labels.length, 177);
  assert.equal(new Set(labels).size, 40);
});
