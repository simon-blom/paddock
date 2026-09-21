import type { InjectionKey, Ref } from 'vue'
import type { AudioPart, Conversation } from '@/types/chat'

/** Optional presentation port. The browser never provides it. It removes only
 * application chrome; ChatThread, document/graph/artifact panels and stream
 * orchestration stay the same components on both platforms. */
export interface NativeContentHost {
  preview: Ref<Conversation | null>
  audioPreview?: Ref<AudioPart | null>
  /** Renderer trial only. Unmount web message bodies instead of painting twice. */
  nativeMarkdown?: Readonly<Ref<boolean>>
  /** Only the document renderer is mounted when Swift owns the transcript. */
  document?: Readonly<Ref<Conversation | null>>
  graphVisible?: Readonly<Ref<boolean>>
  graphArtifact?: Readonly<Ref<{ id: string; title: string; body: string } | null>>
  documentActions?(actions: { info(): void; download(): void } | null): void
  mounted(): void
  /** Visible content-column bounds, not the containing pane's outer bounds. */
  viewport(left: number, width: number): void
}
export const NATIVE_CONTENT: InjectionKey<NativeContentHost> = Symbol('native-content')
