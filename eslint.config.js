import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "assets/**",
      "benchmarks/**",
      "dist/**",
      "node_modules/**",
      "test/fixtures/**",
      "scripts/**/*.mjs",
    ],
  },
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    files: ["src/**/*.ts", "examples/**/*.ts", "scripts/**/*.ts", "test/**/*.ts"],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "no-console": "off",
      "no-constant-binary-expression": "error",
      "no-duplicate-imports": ["error", { allowSeparateTypeImports: true }],
      "no-else-return": "error",
      "no-useless-rename": "error",
      "object-shorthand": "error",
      "prefer-const": "error",
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", fixStyle: "separate-type-imports" },
      ],
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-non-null-assertion": "error",
      // The stylistic alternative is `value!`, which the safety rule above
      // intentionally forbids. Bounds-proven indexed reads use `as T`.
      "@typescript-eslint/non-nullable-type-assertion-style": "off",
      "@typescript-eslint/restrict-template-expressions": ["error", { allowNumber: true }],
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" },
      ],
    },
  },
  {
    files: ["src/**/*.ts"],
    rules: {
      complexity: ["error", 15],
      "max-depth": ["error", 4],
      "max-lines": ["error", 500],
      "max-lines-per-function": ["error", 120],
      "max-params": ["error", 6],
      "max-statements": ["error", 30],
    },
  },
  {
    files: ["test/**/*.ts"],
    rules: {
      // node:test's registration function is typed as a thenable, but top-level
      // calls intentionally register tests with the runner rather than await.
      "@typescript-eslint/no-floating-promises": "off",
    },
  },
  {
    files: [
      "src/grid-locator.ts",
      "src/matching.ts",
      "src/runtime.ts",
      "src/vision-detector.ts",
      "scripts/diagnose-slot-matches.ts",
    ],
    rules: {
      // OpenCV.js publishes several numeric constants and matrix views as any.
      // Calls remain checked by our narrowed OpenCv boundary and integration tests.
      "@typescript-eslint/no-unsafe-argument": "off",
    },
  },
);
