import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import sharp from "sharp";

import { loadCatalog } from "../src/catalog.js";
import { getOpenCv } from "../src/opencv.js";

const screenshotPath = process.argv[2];
const scales = (process.argv[3] ?? "1")
  .split(",")
  .map(Number)
  .filter((value) => Number.isFinite(value) && value > 0);
const templateFilter = process.argv[4]?.toLowerCase();
const localMaxThreshold = process.argv[5] ? Number(process.argv[5]) : undefined;

function topLocalMaxima(
  result: InstanceType<typeof cv.Mat>,
  differenceResult: InstanceType<typeof cv.Mat>,
  threshold: number,
) {
  const matches: { score: number; differenceConfidence: number; x: number; y: number }[] = [];
  for (let y = 1; y < result.rows - 1; y += 1) {
    for (let x = 1; x < result.cols - 1; x += 1) {
      const index = y * result.cols + x;
      const score = result.data32F[index] ?? 0;
      if (!Number.isFinite(score) || score < threshold) continue;
      let maximum = true;
      for (let offsetY = -1; offsetY <= 1 && maximum; offsetY += 1) {
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          if (offsetX === 0 && offsetY === 0) continue;
          if ((result.data32F[(y + offsetY) * result.cols + x + offsetX] ?? 0) > score) {
            maximum = false;
            break;
          }
        }
      }
      if (maximum) {
        matches.push({
          score,
          differenceConfidence: Math.max(0, 1 - (differenceResult.data32F[index] ?? 1)),
          x,
          y,
        });
      }
    }
  }
  return matches.sort((left, right) => right.score - left.score).slice(0, 40);
}

if (!screenshotPath) {
  console.error("Usage: tsx scripts/diagnose-slot-matches.ts <screenshot-or-crop.png> [scales]");
  process.exit(1);
}

const { cv } = await getOpenCv();
const loadedCatalog = await loadCatalog();
const screenshotRaw = await sharp(resolve(screenshotPath))
  .ensureAlpha()
  .raw()
  .toBuffer({ resolveWithObject: true });
const screenshotRgba = new cv.Mat(screenshotRaw.info.height, screenshotRaw.info.width, cv.CV_8UC4);
screenshotRgba.data.set(screenshotRaw.data);
const screenshot = new cv.Mat();
cv.cvtColor(screenshotRgba, screenshot, cv.COLOR_RGBA2RGB);
screenshotRgba.delete();

try {
  for (const metadata of loadedCatalog.catalog.slotTemplates.filter(
    (entry) => !templateFilter || entry.name.toLowerCase() === templateFilter,
  )) {
    const templatePath = resolve(
      dirname(resolve(loadedCatalog.root, "catalog.json")),
      metadata.file,
    );
    const templateRaw = await sharp(await readFile(templatePath))
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    const rgba = new cv.Mat(templateRaw.info.height, templateRaw.info.width, cv.CV_8UC4);
    rgba.data.set(templateRaw.data);
    const rgb = new cv.Mat();
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    const channels = new cv.MatVector();
    cv.split(rgba, channels);
    const alpha = channels.get(3);
    const mask = new cv.Mat();
    cv.threshold(alpha, mask, 0, 255, cv.THRESH_BINARY);
    alpha.delete();
    channels.delete();
    rgba.delete();

    try {
      for (const scale of scales) {
        const width = Math.round(metadata.width * scale);
        const height = Math.round(metadata.height * scale);
        const scaledRgb = new cv.Mat();
        const scaledMask = new cv.Mat();
        cv.resize(rgb, scaledRgb, new cv.Size(width, height), 0, 0, cv.INTER_NEAREST);
        cv.resize(mask, scaledMask, new cv.Size(width, height), 0, 0, cv.INTER_NEAREST);
        const result = new cv.Mat();
        const differenceResult = new cv.Mat();
        const noMask = new cv.Mat();
        try {
          cv.matchTemplate(screenshot, scaledRgb, result, cv.TM_CCORR_NORMED, scaledMask);
          const extrema = cv.minMaxLoc(result, noMask);
          cv.matchTemplate(
            screenshot,
            scaledRgb,
            differenceResult,
            cv.TM_SQDIFF_NORMED,
            scaledMask,
          );
          const differenceExtrema = cv.minMaxLoc(differenceResult, noMask);
          console.log(
            JSON.stringify({
              name: metadata.name,
              scale,
              width,
              height,
              score: extrema.maxVal,
              x: extrema.maxLoc.x,
              y: extrema.maxLoc.y,
              differenceConfidence: Math.max(0, 1 - differenceExtrema.minVal),
              differenceX: differenceExtrema.minLoc.x,
              differenceY: differenceExtrema.minLoc.y,
              ...(localMaxThreshold === undefined
                ? {}
                : { localMaxima: topLocalMaxima(result, differenceResult, localMaxThreshold) }),
            }),
          );
        } finally {
          noMask.delete();
          differenceResult.delete();
          result.delete();
          scaledMask.delete();
          scaledRgb.delete();
        }
      }
    } finally {
      mask.delete();
      rgb.delete();
    }
  }
} finally {
  screenshot.delete();
}
