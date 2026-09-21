/** Byte-based LRU for retained AND pending canvas backing. Pinned visible tiles
 * cannot be evicted by prefetch. The owner zeroes canvases and aborts work when
 * evicted; deleting a map entry alone does not release browser raster storage. */
export declare class RasterBudget {
    #private;
    readonly limitBytes: number;
    constructor(limitBytes?: number);
    get stats(): {
        bytes: number;
        peakBytes: number;
        limitBytes: number;
        entries: number;
    };
    reserve(key: object, bytes: number, pinned: boolean, evict: () => void): boolean;
    touch(key: object, pinned: boolean): void;
    release(key: object): void;
}
export declare function rasterBudgetFor(engine: object): RasterBudget;
//# sourceMappingURL=raster-budget.d.ts.map