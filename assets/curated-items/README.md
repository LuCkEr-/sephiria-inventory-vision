# Curated real-render templates

The 41 generated templates cover 40 named item identities. Every template is built from exactly two icon-only inputs:

1. The alpha mask of the item's extracted Unity sprite.
2. Rendered RGB pixels sampled from that mask's position inside a known 32×32 inventory-grid cell.

The generator never crops or reads tooltips, labels, acquisition messages, quantity text, or other screen regions. Names and item IDs come from `assets/item-icon-map.json`, which maps game item metadata to extracted sprites. `manifest.json` contains only screenshot filenames, normalized grid origins, row/column coordinates, and the occasional sprite orientation or offset adjustment.

Some identities occur in multiple captures and therefore exercise cross-capture rendering differences. Single-capture calibrations still provide regression coverage for grid localization, normalization, mask placement, catalog identity wiring, and OpenCV matching; they should be supplemented with independent captures as those become available.
