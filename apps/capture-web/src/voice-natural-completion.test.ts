import { afterEach, describe, expect, it, vi } from "vitest";
import { NaturalSourceCompletionFence } from "./voice-natural-completion.js";
describe("finite source cross-port completion", () => {
  afterEach(() => vi.useRealTimers());
  it("keeps worker completion and returned credits pending until a late origin receipt arrives", async () => {
    vi.useFakeTimers();
    const fence = new NaturalSourceCompletionFence();
    let finished = false;
    const result = fence.wait(6000).then(value => { finished = true; return value; });
    fence.receive("worker"); fence.receive("credits");
    await vi.advanceTimersByTimeAsync(5999);
    expect(finished).toBe(false);
    fence.receive("origin");
    expect(await result).toEqual({ status: "complete", missing: [] });
    expect(vi.getTimerCount()).toBe(0);
  });
  it("retains a missing recycle acknowledgement at the frozen deadline", async () => {
    vi.useFakeTimers();
    const fence = new NaturalSourceCompletionFence();
    fence.receive("origin"); fence.receive("worker");
    const result = fence.wait(6000);
    await vi.advanceTimersByTimeAsync(6000);
    expect(await result).toEqual({ status: "timeout", missing: ["credits"] });
    fence.receive("credits"); // A late receipt cannot rewrite the timed-out outcome.
    expect(await result).toEqual({ status: "timeout", missing: ["credits"] });
  });
});
