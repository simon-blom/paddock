import type { LectorEngine } from '../engine/lector-engine.js';
import type { ViewportInstance } from '../plugins/viewport-plugin.js';
import type { PageOverlayManager } from './page-overlays.js';
import { rasterBudgetFor } from './raster-budget.js';
interface Options {
    readonly engine: LectorEngine;
    readonly viewport: ViewportInstance;
    readonly scrollArea: HTMLElement;
    readonly overlays: PageOverlayManager;
    readonly pageElements: Map<number, HTMLElement>;
}
/**
 * The single page/raster lifecycle for both the headless pane and full viewer.
 * Layout metadata covers the document; DOM/overlays cover a viewport window;
 * full-resolution backing covers visible pixels only. A shared byte LRU covers
 * every canvas (including pending ones), not just completed ImageBitmaps.
 */
export declare class PageRenderer {
    #private;
    constructor(options: Options);
    get stats(): {
        mountedPages: number;
        surfaces: number;
        residentBytes: number;
        sharedBudget: ReturnType<typeof rasterBudgetFor>['stats'];
    };
    /** True only after the current viewport generation has full-resolution detail.
     * An overscan preview or an old zoom's loading class is not readiness. */
    isPageReady(index: number): boolean;
    /** Batch reactive cascades and scroll events into at most one DOM pass/frame. */
    update(): void;
    /** Invalidation never leaves stale high-resolution detail above fresh content. */
    invalidate(pageIndex?: number): void;
    destroy(): void;
}
export {};
//# sourceMappingURL=page-renderer.d.ts.map