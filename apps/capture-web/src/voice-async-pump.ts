/** One active operation and a bounded FIFO; no per-input promise chain. */
export class BoundedVoiceAsyncPump<T> {
  private readonly queue: T[] = [];
  private running = false;
  private disposed = false;
  maximumQueued = 0;
  constructor(private readonly process: (item: T) => Promise<void> | void,
    private readonly discard: (item: T) => void,
    private readonly failed: (error: unknown) => void,
    readonly capacity = 3) {}
  get idle(): boolean { return !this.running && this.queue.length === 0; }
  submit(item: T): void {
    if (this.disposed || (this.running && this.queue.length >= this.capacity)) { this.discard(item); return; }
    if (this.running) {
      this.queue.push(item);
      this.maximumQueued = Math.max(this.maximumQueued, this.queue.length);
    } else { this.running = true; void this.drain(item); }
  }
  flush(): void { for (const item of this.queue.splice(0)) this.discard(item); }
  dispose(): void { this.disposed = true; this.flush(); }
  private async drain(first: T): Promise<void> {
    let item: T | undefined = first;
    try {
      while (item !== undefined) {
        try { await this.process(item); } catch (error) { this.failed(error); }
        item = this.disposed ? undefined : this.queue.shift();
      }
    } finally { this.running = false; }
  }
}
