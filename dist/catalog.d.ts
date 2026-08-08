import type { InventoryCatalog } from "./types.js";
export declare function defaultCatalogPath(): string;
export declare function loadCatalog(catalogPath?: string): Promise<{
    catalog: InventoryCatalog;
    root: string;
}>;
//# sourceMappingURL=catalog.d.ts.map