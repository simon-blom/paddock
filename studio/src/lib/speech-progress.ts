/** Item identities are shared by speech-start, commit and completion. Counting
 * events double-counts an utterance and can clear a newer speaking item. */
export class SpeechProgress {
  readonly pending = new Set<string>()
  private readonly completed = new Set<string>()
  speaking?: string
  draining = false
  get waiting(): boolean { return this.draining || this.pending.size > 0 }
  beginDrain(): void { this.draining = true }
  receive(event: Record<string, unknown>): boolean {
    const id = typeof event.item_id === 'string' ? event.item_id : 'legacy-item'
    switch (event.type) {
      case 'input_audio_buffer.speech_started':
        this.speaking = id
        if (!this.completed.has(id)) this.pending.add(id)
        break
      case 'input_audio_buffer.committed':
        if (!this.completed.has(id)) this.pending.add(id)
        break
      case 'input_audio_buffer.speech_stopped':
        if (this.speaking === id) this.speaking = undefined
        break
      case 'input_audio_buffer.drained':
        this.draining = false
        this.speaking = undefined
        for (const id of Array.isArray(event.pending_item_ids) ? event.pending_item_ids : []) {
          if (typeof id === 'string' && !this.completed.has(id)) this.pending.add(id)
        }
        break
      case 'conversation.item.input_audio_transcription.completed':
        this.pending.delete(id)
        if (this.speaking === id) this.speaking = undefined
        if (id === 'legacy-item') return true
        if (this.completed.has(id)) return false
        this.completed.add(id)
        break
    }
    return true
  }
}
