import type { DocumentHandle, LectorEngine } from '../engine/lector-engine.js';
/** Sidebar placeholders retain layout, not raster backing. Only intersecting
 * thumbnails plus a small overscan region render. Their backing shares the
 * engine's page budget and is always evictable in favour of visible detail. */
export declare class ThumbnailRenderer {
    #private;
    readonly engine: LectorEngine;
    readonly document: DocumentHandle;
    constructor(engine: LectorEngine, document: DocumentHandle, root: HTMLElement);
    observe(canvas: HTMLCanvasElement, page: number, width: number, height: number): void;
    invalidate(page: number): void;
    destroy(): void;
}
//# sourceMappingURL=thumbnail-renderer.d.ts.map