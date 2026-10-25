/** Finite-source controls arrive on independent ports; none implies another. */
export class NaturalSourceCompletionFence {
  private readonly received = new Set<"origin" | "worker" | "credits">();
  private finish: (() => void) | null = null;
  private waiting = false;
  receive(kind: "origin" | "worker" | "credits"): void {
    this.received.add(kind);
    if (this.received.size === 3) this.finish?.();
  }
  async wait(timeoutMs: number): Promise<{ status: "complete" | "timeout"; missing: string[] }> {
    if (this.waiting || !Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("Invalid finite-source completion wait.");
    this.waiting = true;
    return new Promise(resolve => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      this.finish = () => {
        if (timer) clearTimeout(timer);
        this.finish = null;
        const missing = (["origin", "worker", "credits"] as const).filter(kind => !this.received.has(kind));
        resolve({ status: missing.length ? "timeout" : "complete", missing });
      };
      if (this.received.size === 3) this.finish();
      else timer = setTimeout(() => this.finish?.(), timeoutMs);
    });
  }
}
