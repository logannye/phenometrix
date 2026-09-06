import { describe, expect, it, vi } from "vitest";
import { CaptureRuntime, withTimeout } from "./capture-runtime.js";

describe("CaptureRuntime", () => {
  it("disposes in privacy-safe order, stops every track, and is idempotent", async () => {
    const order: string[] = [];
    const tracks = [
      { stop: vi.fn(() => order.push("track-a")) },
      { stop: vi.fn(() => order.push("track-b")) }
    ];
    const stream = { getTracks: () => tracks } as unknown as MediaStream;
    const video = {
      pause: vi.fn(() => order.push("video")),
      srcObject: stream
    } as unknown as HTMLVideoElement;
    const audioContext = {
      state: "running",
      close: vi.fn(async () => {
        order.push("audio-context");
      })
    } as unknown as AudioContext;
    const runtime = new CaptureRuntime();
    runtime.attach({
      cancelTimers: () => order.push("timers"),
      cancelPendingStartup: () => order.push("startup"),
      stopFacePump: () => order.push("pump"),
      stopVoicePipeline: async () => {
        order.push("voice");
      },
      disposeFaceWorker: async () => {
        order.push("face-worker");
      },
      streams: [stream],
      video,
      disconnectAudio: () => order.push("audio-nodes"),
      audioContext
    });

    const first = runtime.dispose(false);
    const second = runtime.dispose(false);
    expect(first).toBe(second);
    await first;
    expect(order).toEqual([
      "timers",
      "startup",
      "pump",
      "track-a",
      "track-b",
      "video",
      "voice",
      "face-worker",
      "audio-nodes",
      "audio-context"
    ]);
    expect(await runtime.dispose(false)).toBeNull();
  });

  it("freezes a derived-only snapshot and clears runtime ownership", async () => {
    const runtime = new CaptureRuntime();
    runtime.addVoiceFrame({ sequence: 1 } as never);
    runtime.addFaceFrame({ sequence: 2 } as never);
    const snapshot = await runtime.dispose(true);
    expect(snapshot?.voice).toHaveLength(1);
    expect(snapshot?.face).toHaveLength(1);
    expect(Object.isFrozen(snapshot)).toBe(true);
    expect(Object.isFrozen(snapshot?.voice)).toBe(true);
    expect(
      (runtime as unknown as { disposing: unknown }).disposing
    ).toBeNull();
  });

  it("lets destructive disposal downgrade an in-flight preserved snapshot", async () => {
    let releaseWorker!: () => void;
    const worker = new Promise<void>((resolve) => {
      releaseWorker = resolve;
    });
    const runtime = new CaptureRuntime();
    runtime.addVoiceFrame({ sequence: 1 } as never);
    runtime.addFaceFrame({ sequence: 2 } as never);
    runtime.attach({ disposeFaceWorker: () => worker });

    const finalization = runtime.dispose(true);
    const discard = runtime.dispose(false);
    expect(discard).toBe(finalization);

    releaseWorker();
    await expect(finalization).resolves.toBeNull();
    expect(await runtime.dispose(true)).toBeNull();
  });

  it("tracks and awaits handles attached after disposal starts", async () => {
    let releaseInitialWorker!: () => void;
    const initialWorker = new Promise<void>((resolve) => {
      releaseInitialWorker = resolve;
    });
    const runtime = new CaptureRuntime();
    runtime.attach({ disposeFaceWorker: () => initialWorker });
    const disposal = runtime.dispose(false);

    let releaseLateWorker!: () => void;
    const lateWorkerCleanup = new Promise<void>((resolve) => {
      releaseLateWorker = resolve;
    });
    const lateWorker = vi.fn(() => lateWorkerCleanup);
    const latePipeline = vi.fn(async () => undefined);
    const lateTrack = { stop: vi.fn() };
    runtime.attach({
      disposeFaceWorker: lateWorker,
      stopVoicePipeline: latePipeline,
      streams: [
        { getTracks: () => [lateTrack] } as unknown as MediaStream
      ]
    });
    await vi.waitFor(() => {
      expect(latePipeline).toHaveBeenCalledOnce();
    });

    let disposalCompleted = false;
    void disposal.then(() => {
      disposalCompleted = true;
    });
    releaseInitialWorker();
    await vi.waitFor(() => expect(lateWorker).toHaveBeenCalledOnce());
    await Promise.resolve();
    expect(disposalCompleted).toBe(false);
    expect(lateTrack.stop).toHaveBeenCalledOnce();

    releaseLateWorker();
    await disposal;
    expect(disposalCompleted).toBe(true);
  });

  it("uses a bounded worker-disposal fallback", async () => {
    vi.useFakeTimers();
    const resultPromise = withTimeout(
      new Promise<string>(() => undefined),
      500,
      () => "terminated"
    );
    await vi.advanceTimersByTimeAsync(500);
    await expect(resultPromise).resolves.toBe("terminated");
    vi.useRealTimers();
  });
});
