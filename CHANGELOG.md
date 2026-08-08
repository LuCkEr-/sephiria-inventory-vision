# Changelog

## 0.8.0 — leakage-free rejection and source-order robustness

- Added one shared class-aware acceptance policy for runtime, evaluation, and benchmarks, including minimum-margin protection for single-source item classes.
- Calibrated positive generalization floors only from cross-source observations and eliminated every false accept in the 111-query leakage-free class holdout without weakening the 461-asset unknown-item rejection gate.
- Added source-order JPEG, filtered-resize crop jitter, and materialized nearest/cubic round-trip references so training reproduces the runtime image pipeline rather than relying on Sharp's resize fusion behavior.
- Converted resize robustness from an explicit-grid shortcut into an automatic-localization release gate; brightness, JPEG-80, nearest 150%, cubic 150%, and source-order cubic now each pass 24/24 cells.
- Expanded classification diagnostics with the effective minimum margin and a stable acceptance reason for accepted, empty, low-confidence, and singleton-margin outcomes.

## 0.7.0 — unified confidence, open-set calibration, and exhaustive release evidence

- Replaced the nonlinear vision-score transform with bounded raw cosine confidence so runtime, tests, model generation, and benchmarks use exactly one scoring contract.
- Added per-class impostor thresholds, effective-threshold and acceptance diagnostics, and strict model validation for calibration metadata.
- Reduced the vision default to the measured `0.936` cutoff, restoring 24/24 filtered-resize robustness while suppressing every measured in-model alias; the stricter recalibrated class holdout remains explicitly reported rather than hidden.
- Added an independent open-set regression that renders every one of the 461 extracted item templates and rotations absent from the vision model into an inventory-slot background; all 461 are rejected.
- Recalibrated every held-out cascade fold from training references only, preventing thresholds learned from the evaluated screenshot from leaking into cross-validation.
- Preserved 288/288 cross-validated cells and 111/111 item identities while restoring the cascade's raw-confidence-equivalent `0.96` fast-path threshold.
- Upgraded OpenCV.js to 5.0.0, removed duplicate cascade slot-template allocations, and hardened cleanup for partial color conversion, template scaling, border-mask creation, matching, grid localization, and image decoding failures.
- Split grid localization, neural contracts, and non-maximum suppression into focused modules; enforced 500 lines per file, 6 parameters, and 30 statements per function in addition to existing complexity and nesting limits.
- Added strict package linting with Publint and ESM type-resolution verification with Are the Types Wrong, and expanded the release gate to audit development as well as production dependencies.
- Raised enforced coverage to 99.5% lines, 92.5% branches, and 100% functions across 60 tests; the measured report is 99.63%, 92.96%, and 100%.

## 0.6.2 — structural hardening and retry-safe initialization

- Decomposed catalog validation, grid scoring, cascade fallback, runtime normalization, vision-model validation, feature extraction, and neural training into focused, independently testable units.
- Added permanent production-code limits of complexity 15, nesting depth 4, and 120 lines per function; every maintained source file passes them without exemptions.
- Replaced poisoned-promise initialization caches with a shared retryable loader that deduplicates concurrent work, caches successful initialization, and permits recovery after transient OpenCV or TensorFlow failures.
- Added failure-safe TensorFlow construction and training cleanup so partially initialized models and optimizers cannot leak after rejected training.
- Added direct real-OpenCV grid-matching regressions, retry/concurrency regressions, empty-catalog normalization coverage, complete option-validation cases, and single-class Siamese behavior.
- Raised the enforced branch-coverage floor from 90% to 91.5% while retaining the 99.4% line and 100% function gates across 52 tests.

## 0.6.1 — exhaustive boundaries and measurable cleanup

- Hardened classical and neural lab APIs against short, sparse, fractional, out-of-range, non-finite, and incorrectly shaped pixel buffers.
- Made descriptor cosine similarity reject unequal or non-finite vectors instead of silently truncating them.
- Added runtime validation for method names, alternative counts, neural labels and feature vectors, ORB limits, and every object-detector box field.
- Deepened catalog validation across every template field, gameplay identity consistency, optional metadata, placement rules, and prefab provenance.
- Extracted and tested all supported OpenCV.js initialization shapes plus Sharp's four-channel decoding invariant without leaking these helpers into public declarations.
- Removed bounds-safe `?? 0` reads and dead table checks from validated feature extraction paths, eliminating silent invariant masking and artificial branch debt.
- Raised enforced coverage floors from 99% lines / 84% branches to 99.4% lines / 90% branches while retaining 100% function coverage.

## 0.6.0 — contract and release integrity

- Corrected the exported cosine-similarity function to compute scale-invariant mathematical cosine similarity and reject mismatched or non-finite vectors.
- Added strict byte-buffer validation for vision descriptors while retaining a validated unit-vector fast path inside the detector.
- Made template, vision, and cascade disposal safe during concurrent asynchronous detection by deferring native-resource release until in-flight operations finish.
- Expanded vision-model validation to cover provenance, grid coordinates, normalized features, item schema, augmentation policy, and label/reference consistency.
- Expanded catalog validation through nested gameplay, placement, rotation, identity, and source-prefab metadata.
- Upgraded ESLint to strict type-aware and stylistic analysis, resolving unsafe indexing, dead branches, promise handling, and type-narrowing findings across source, scripts, tests, and examples.
- Added an isolated clean-install smoke test that packs the library, installs production dependencies only, verifies TensorFlow remains absent, imports both entry points, and performs a real 24/24 cascade detection.
- Added a release gate covering quality, coverage, build, package boundaries, clean installation, held-out cascade accuracy, relative latency, memory growth, and production dependency audit.
- Added default candidate game-asset verification for confident vision predictions, with automatic full-catalog recovery when the predicted sprite does not match.
- Added a 12-fold end-to-end cascade gate that removes every learned reference from each evaluated screenshot; it passes 288/288 cells and 111/111 occupied-item identities.
- Added a focused regression for a confident unseen-class alias (`Advance` misread as `Golden Handbell`) to prove the asset-mismatch recovery path.
- Raised enforced coverage minimums to 99% lines, 84% branches, and 100% functions.

## 0.5.0 — robustness and production boundaries

- Replaced the background-heavy vision descriptor with a combined structural and centered RGB signature, reducing leave-one-class-out false acceptance from 73.9% to 4.5%.
- Added deterministic brightness, JPEG, logical resize, and source-resolution resize augmentation. All enforced robustness variants now detect 24/24 cells.
- Compacted the augmented model from 36.6 MB to 9.14 MB by retaining every reviewed base cell, deduplicating redundant augmented empty cells, and quantizing stored features to six decimal places.
- Added strict grid, slot, threshold, model, catalog, training, and lifecycle validation with actionable errors.
- Prevented catalog template paths from escaping the catalog directory.
- Split experimental comparison methods into `sephiria-inventory-vision/lab` and made TensorFlow.js a lazily loaded optional peer, keeping it and its transitive install scripts out of production installs.
- Made OpenCV loading lazy so importing the production API does not initialize the WASM runtime.
- Added enforceable coverage gates, package-entry verification, and a 50-run memory regression benchmark.
- Tightened TypeScript with unused-code, implicit-return, fallthrough, index-signature, isolated-module, and unreachable-code checks.

## 0.4.1 — cleanup and hardening

- Consolidated RGBA decoding, RGB matrix ownership, normalization, coordinate scaling, and slot localization into one shared detection runtime.
- Made OpenCV context disposal idempotent and documented ownership at the boundary.
- Replaced duplicated template-loading loops with one failure-safe loader.
- Added finite range validation for slot, item, empty-slot, template, and cascade thresholds.
- Added an explicit `vision-rejected` cascade fallback and a hard invariant when template verification fails to return a requested slot.
- Added named public types for classifier backends, fallback reasons, and detection timings.
- Removed non-null assertions and untyped `any` values from maintained TypeScript sources, scripts, and tests.
- Cached the perceptual-hash DCT basis instead of recomputing cosine values for every pixel and candidate.
- Added canonical Prettier formatting, ESLint rules, and `quality`, `format`, and `format:check` commands.
- Changed the cascade benchmark to report the median of three runs.
- Re-ran the complete real-screenshot regression suite and every method benchmark without an accuracy regression.
