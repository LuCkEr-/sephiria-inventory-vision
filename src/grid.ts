import type { GridOptions, SlotRect } from "./types.js";

function finiteNonNegative(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be a finite non-negative number, received ${value}`);
  }
}

function positiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer, received ${value}`);
  }
}

export function validateGridOptions(options: GridOptions): void {
  finiteNonNegative(options.x, "grid.x");
  finiteNonNegative(options.y, "grid.y");
  positiveInteger(options.rows, "grid.rows");
  positiveInteger(options.columns, "grid.columns");
  const slotSize = options.slotSize ?? 34;
  if (!Number.isFinite(slotSize) || slotSize <= 0) {
    throw new Error(`grid.slotSize must be positive and finite, received ${slotSize}`);
  }
  finiteNonNegative(options.gapX ?? 0, "grid.gapX");
  finiteNonNegative(options.gapY ?? 0, "grid.gapY");
  const scale = options.scale ?? 1;
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`grid.scale must be positive and finite, received ${scale}`);
  }
}

export function createGridSlots(options: GridOptions): SlotRect[] {
  validateGridOptions(options);
  const scale = options.scale ?? 1;
  const slotSize = Math.round((options.slotSize ?? 34) * scale);
  const gapX = Math.round((options.gapX ?? 0) * scale);
  const gapY = Math.round((options.gapY ?? 0) * scale);
  const slots: SlotRect[] = [];

  for (let row = 0; row < options.rows; row += 1) {
    for (let column = 0; column < options.columns; column += 1) {
      slots.push({
        x: Math.round(options.x + column * (slotSize + gapX)),
        y: Math.round(options.y + row * (slotSize + gapY)),
        width: slotSize,
        height: slotSize,
        row,
        column,
        scale,
      });
    }
  }

  return slots;
}
