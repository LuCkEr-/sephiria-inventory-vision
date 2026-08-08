import type * as OpenCvNamespace from "@techstark/opencv-js";

import { RetryableLoader } from "./retryable-loader.js";

export type OpenCv = typeof OpenCvNamespace;

export interface OpenCvHandle {
  cv: OpenCv;
}

/** @internal Resolves every initialization shape published by OpenCV.js builds. */
export async function initializeOpenCvModule(cvModule: unknown): Promise<OpenCvHandle> {
  // The Emscripten module exposes a `.then` method but is not a real Promise;
  // awaiting that thenable can self-resolve forever. Follow the package's
  // documented runtime callback unless a genuine Promise build is supplied.
  if (cvModule instanceof Promise) {
    return { cv: (await cvModule) as OpenCv };
  }

  const cv = cvModule as OpenCv;
  if (typeof cv.Mat === "function") return { cv };

  await new Promise<void>((resolve) => {
    cv.onRuntimeInitialized = resolve;
  });
  // Wrap the module because Emscripten adds a `.then` method. Resolving a
  // native Promise with the module itself would trigger thenable assimilation.
  return { cv };
}

const openCvLoader = new RetryableLoader(async () => {
  const { default: cvModule } = await import("@techstark/opencv-js");
  return initializeOpenCvModule(cvModule);
});

export function getOpenCv(): Promise<OpenCvHandle> {
  return openCvLoader.get();
}
