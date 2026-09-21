import type * as Comlink from 'comlink';
import type { DocumentId, TaskId } from '../types/handle-id.js';
import { type RenderOptions, type RenderPriority } from '../types/render.js';
import type { PdfiumWorkerApi } from '../types/worker-api.js';
import type { RenderPool } from './render-pool.js';
export interface RenderSchedulingOptions {
    readonly priority?: RenderPriority;
    readonly signal?: AbortSignal;
    /** Explicit consumer slot, e.g. viewport/page/tile. Never inferred from page number. */
    readonly consumerKey?: string;
    /** Content generation. A render from before an edit must never satisfy one after it. */
    readonly generation?: string;
}
export interface RenderRequest extends RenderSchedulingOptions {
    readonly docId: DocumentId;
    readonly pageIndex: number;
    readonly width: number;
    readonly height: number;
    readonly options?: RenderOptions;
    readonly tile?: {
        readonly x: number;
        readonly y: number;
        readonly fullW: number;
        readonly fullH: number;
    };
}
/** Nominal transient raster bytes; PDF parsing/font caches and browser overhead are separate. */
export interface RenderSchedulerOptions {
    readonly maxInFlightBytes?: number;
    readonly maxQueuedTasks?: number;
}
/**
 * Admission queue shared by full pages and detail tiles. Worker concurrency and
 * transient pixel reservations remain occupied until physical work completes,
 * even when all consumers abort. JS cancellation cannot interrupt PDFium.
 *
 * Identical work is shared, ownership is not: every consumer receives an
 * independently closeable bitmap. Coalescing requires an explicit consumer slot;
 * different panes never cancel each other implicitly.
 */
export declare class RenderScheduler implements Disposable {
    #private;
    constructor(proxy: Comlink.Remote<PdfiumWorkerApi>, pool?: RenderPool, options?: RenderSchedulerOptions);
    get stats(): {
        active: number;
        queued: number;
        reservedBytes: number;
        peakReservedBytes: number;
        budgetBytes: number;
    };
    enqueue(request: RenderRequest): Promise<ImageBitmap>;
    cancel(taskId: TaskId): void;
    cancelDocument(docId: DocumentId): void;
    reprioritize(docId: DocumentId, pageIndex: number, priority: RenderPriority): void;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=render-scheduler.d.ts.map