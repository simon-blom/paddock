export interface RasterRect {
    x: number;
    y: number;
    width: number;
    height: number;
}
export interface RasterTile extends RasterRect {
    fullW: number;
    fullH: number;
    preview: boolean;
}
/** Original implementation. Viewport-only detail over a small full-page base is
 * also used by PDF.js detail views and EmbedPDF tiling; see docs/rendering.md.
 * Coordinates are integer device pixels, NOT capped full-page dimensions. */
export declare function planPageRaster(fullW: number, fullH: number, visible: RasterRect | null): RasterTile[];
//# sourceMappingURL=page-raster-plan.d.ts.map