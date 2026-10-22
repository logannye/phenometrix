import type { VoiceRenderLease } from "./voice-render-freshness.js";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { describe, expect, it } from "vitest";

interface WorkletMessagePort {
  onmessage: ((event: { data: unknown }) => void) | null;
  postMessage(message: unknown, transfer?: unknown[]): void;
  start(): void;
  close(): void;
}
interface CapturedBlock {
  schemaVersion: string; type: string; captureEpoch: number; clockMappingId: number;
  renderLease: VoiceRenderLease | null;
  sequence: number; absoluteSampleIndex: number; endAudioFrameExclusive: number;
  acquisitionAudioTimeSeconds: number; transportId: number;
  droppedBlockCount: number; droppedSampleCount: number; inFlightBlockCount: number; buffer: ArrayBuffer;
}
interface Processor {
  port: WorkletMessagePort; process(inputs: Float32Array[][], outputs?: Float32Array[][]): boolean;
  pool: ArrayBuffer[]; inFlight: Set<number>; pending: Float32Array; maximumInFlightBlocks: number;
}
function fixture() {
  let registered: (new () => Processor) | null = null;
  class AudioWorkletProcessorFixture {
    port: WorkletMessagePort = { onmessage: null, postMessage() {}, start() {}, close() {} };
  }
  const context = { AudioWorkletProcessor: AudioWorkletProcessorFixture,
    registerProcessor: (name: string, processor: typeof registered) => {
      expect(name).toBe("phenometric-voice-capture"); registered = processor;
    }, sampleRate: 48_000, currentTime: 1, currentFrame: 48_000, ArrayBuffer, Float32Array, Math, Set };
  runInNewContext(readFileSync(new URL("../public/voice-capture-worklet.js", import.meta.url), "utf8"), context);
  if (!registered) throw new Error("Worklet processor was not registered.");
  const processor = new (registered as new () => Processor)();
  const blocks: CapturedBlock[] = [];
  const controlMessages: Record<string, unknown>[] = [];
  processor.port.postMessage = message => controlMessages.push(message as Record<string, unknown>);
  const dataPort: WorkletMessagePort = { onmessage: null,
    postMessage: message => blocks.push(message as CapturedBlock), start() {}, close() {} };
  processor.port.onmessage?.({ data: { type: "attach-port", port: dataPort } });
  processor.port.onmessage?.({ data: { type: "capture-epoch", captureEpoch: 7, clockMappingId: 0 } });
  const render = (length = 960, value = 0.25) => {
    context.currentTime = context.currentFrame / context.sampleRate;
    const alive = processor.process([[Float32Array.from({ length }, () => value)]]);
    context.currentFrame += length;
    return alive;
  };
  const recycle = (block: CapturedBlock) => dataPort.onmessage?.({ data: {
    type: "recycle", transportId: block.transportId, buffer: block.buffer
  } });
  return { processor, blocks, dataPort, render, recycle, context, controlMessages };
}

describe("voice capture AudioWorklet", () => {
  it("snapshots the lease before the first block sample and never refreshes partial support", () => {
    const f = fixture();
    const first: VoiceRenderLease = { schemaVersion: "phenometric.voice-render-freshness.v1", leaseId: 1,
      captureEpoch: 7, clockMappingId: 0, mainTimeOriginMs: 100000, issuedAtMainMs: 1000, expiresAtMainMs: 1100 };
    f.render(128); // This partial block started before any lease existed.
    f.processor.port.onmessage?.({ data: { type: "render-freshness-lease", lease: first } });
    f.render(832);
    expect(f.blocks[0].renderLease).toBeNull();
    f.render(128);
    const second = { ...first, leaseId: 2, issuedAtMainMs: 1025, expiresAtMainMs: 1125 };
    f.processor.port.onmessage?.({ data: { type: "render-freshness-lease", lease: second } });
    f.render(832);
    expect(f.blocks[1].renderLease).toEqual(first); // Both potential 10 ms endpoints keep this oldest lease.
    f.render();
    expect(f.blocks[2].renderLease).toEqual(second);
    expect(f.blocks.map(block => block.absoluteSampleIndex)).toEqual([0, 960, 1920]);
    expect(f.controlMessages.map(message => message.disposition)).toEqual(["accepted", "accepted"]);
  });

  it("clears leases at mapping and epoch changes and returns retired controls without minting PCM credit", () => {
    const f = fixture();
    const lease = { schemaVersion: "phenometric.voice-render-freshness.v1", leaseId: 1,
      captureEpoch: 7, clockMappingId: 0, mainTimeOriginMs: 100000, issuedAtMainMs: 1000, expiresAtMainMs: 1100 };
    f.processor.port.onmessage?.({ data: { type: "render-freshness-lease", lease } });
    f.render();
    f.processor.port.onmessage?.({ data: { type: "clock-mapping", clockMappingId: 1 } });
    f.processor.port.onmessage?.({ data: { type: "render-freshness-lease", lease: { ...lease, leaseId: 2 } } });
    f.render();
    expect(f.blocks[1].renderLease).toBeNull();
    expect(f.controlMessages.at(-1)).toMatchObject({ leaseId: 2, disposition: "retired" });
    f.processor.port.onmessage?.({ data: { type: "render-freshness-lease", lease: { ...lease, leaseId: 3, clockMappingId: 1 } } });
    f.processor.port.onmessage?.({ data: { type: "capture-epoch", captureEpoch: 8, clockMappingId: 2 } });
    f.render();
    expect(f.blocks[2].renderLease).toBeNull();
    expect(f.processor.pool.length + f.processor.inFlight.size).toBe(4);
    f.processor.port.onmessage?.({ data: { type: "dispose" } });
    const count = f.controlMessages.length;
    f.processor.port.onmessage?.({ data: { type: "render-freshness-lease", lease: { ...lease, leaseId: 4 } } });
    expect(f.controlMessages).toHaveLength(count);
  });

  it("observes the next render boundary even when a stalled worker exhausts all PCM credits", () => {
    const f = fixture();
    for (let i = 0; i < 4; i += 1) f.render(); // Worker has returned no buffers.
    const frame = f.context.currentFrame;
    f.processor.port.onmessage?.({ data: { schemaVersion: "phenometric.voice-clock-probe.v1", type: "clock-probe",
      captureEpoch: 7, clockMappingId: 0, probeId: 1 } });
    expect(f.controlMessages).toHaveLength(0); // Message handling itself is not the render observation.
    f.render(128);
    expect(f.controlMessages).toEqual([{ schemaVersion: "phenometric.voice-clock-probe.v1", type: "clock-probe-result",
      captureEpoch: 7, clockMappingId: 0, probeId: 1, renderStartFrame: frame, sampleRateHz: 48000 }]);
    expect(f.blocks).toHaveLength(4);
    expect(f.processor.inFlight.size).toBe(4);
  });

  it("bounds pending probes and invalidates old mapping/epoch requests", () => {
    const f = fixture();
    const request = { schemaVersion: "phenometric.voice-clock-probe.v1", type: "clock-probe",
      captureEpoch: 7, clockMappingId: 0, probeId: 1 };
    f.processor.port.onmessage?.({ data: request });
    f.processor.port.onmessage?.({ data: { ...request, probeId: 2 } });
    f.render(128);
    expect(f.controlMessages).toHaveLength(1);
    expect(f.controlMessages[0].probeId).toBe(1);
    f.processor.port.onmessage?.({ data: request }); // Completed duplicate.
    f.render(128); expect(f.controlMessages).toHaveLength(1);
    f.processor.port.onmessage?.({ data: { ...request, probeId: 2 } });
    f.processor.port.onmessage?.({ data: { type: "clock-mapping", clockMappingId: 1 } });
    f.render(128); expect(f.controlMessages).toHaveLength(1);
    f.processor.port.onmessage?.({ data: { ...request, probeId: 3 } }); // Retired mapping.
    f.render(128); expect(f.controlMessages).toHaveLength(1);
  });

  it("emits exact endpoints inside irregular render quanta and reuses returned credits", () => {
    const f = fixture();
    f.render(800); f.render(300);
    expect(f.blocks).toHaveLength(1);
    expect(f.blocks[0]).toMatchObject({ schemaVersion: "phenometric.voice-worklet-message.v2",
      captureEpoch: 7, sequence: 1, absoluteSampleIndex: 0, endAudioFrameExclusive: 48_960 });
    expect(f.blocks[0].acquisitionAudioTimeSeconds).toBe(48_800 / 48_000);
    expect(new Float32Array(f.blocks[0].buffer)).toHaveLength(960);
    const returnedBuffer = f.blocks[0].buffer;
    f.recycle(f.blocks[0]); f.render(820, -0.25);
    expect(f.blocks[1].buffer).toBe(returnedBuffer);
    expect(f.blocks[1]).toMatchObject({ sequence: 2, absoluteSampleIndex: 960, endAudioFrameExclusive: 49_920 });
  });

  it("bounds all transport buffers, advances through drops, and exposes lost sample intervals", () => {
    const f = fixture();
    for (let index = 0; index < 10; index += 1) f.render();
    expect(f.blocks).toHaveLength(4);
    expect(f.processor.inFlight.size).toBe(4);
    expect(f.processor.pool).toHaveLength(0);
    f.recycle(f.blocks[0]); f.render();
    expect(f.blocks).toHaveLength(5);
    expect(f.blocks[4]).toMatchObject({ sequence: 11, absoluteSampleIndex: 9600,
      droppedBlockCount: 6, droppedSampleCount: 5760, inFlightBlockCount: 4 });
    expect(f.processor.pool.length + f.processor.inFlight.size).toBe(4);
  });

  it("keeps outstanding credits bounded across epoch reset and safely accepts late returns", () => {
    const f = fixture();
    for (let index = 0; index < 4; index += 1) f.render();
    f.processor.port.onmessage?.({ data: { type: "capture-epoch", captureEpoch: 11, clockMappingId: 1 } });
    f.render(); // All four previous-epoch buffers are still outstanding.
    expect(f.blocks).toHaveLength(4);
    f.recycle(f.blocks[0]); f.recycle(f.blocks[0]); // Duplicate acknowledgement cannot mint a credit.
    expect(f.processor.pool).toHaveLength(1);
    f.render(960, 0.2);
    expect(f.blocks.at(-1)).toMatchObject({ captureEpoch: 11, clockMappingId: 1,
      sequence: 2, absoluteSampleIndex: 960, droppedBlockCount: 1 });
    expect(f.processor.pool.length + f.processor.inFlight.size).toBe(4);
  });

  it("rejects unrecognized and wrong-sized returned buffers", () => {
    const f = fixture();
    f.render();
    f.dataPort.onmessage?.({ data: { type: "recycle", transportId: f.blocks[0].transportId, buffer: new ArrayBuffer(4) } });
    f.dataPort.onmessage?.({ data: { type: "recycle", transportId: 999, buffer: new ArrayBuffer(3840) } });
    expect(f.processor.pool).toHaveLength(3);
    expect(f.processor.inFlight.size).toBe(1);
  });

  it("drops partial support on mapping change without rewinding sample time", () => {
    const f = fixture();
    f.render(128);
    f.processor.port.onmessage?.({ data: { type: "clock-mapping", clockMappingId: 4 } });
    f.render();
    expect(f.blocks[0]).toMatchObject({ absoluteSampleIndex: 128, clockMappingId: 4, droppedSampleCount: 128 });
  });

  it("exposes missing input quanta as a sample gap instead of joining or filling silence", () => {
    const f = fixture();
    f.render(128);
    f.context.currentFrame += 256; // Two render quanta with no input channel.
    f.render();
    expect(f.blocks[0]).toMatchObject({ absoluteSampleIndex: 384, droppedSampleCount: 384 });
  });

  it("accounts terminal absent-input quanta immediately without counting partial support twice", () => {
    const f = fixture();
    f.render(128);
    for (let i = 0; i < 3; i += 1) {
      f.processor.process([[]], [[new Float32Array(128)]]);
      f.context.currentFrame += 128;
    }
    f.processor.port.onmessage?.({ data: { type: "dispose" } });
    expect(f.controlMessages.at(-1)).toMatchObject({ sampleClockEndExclusive: 512,
      droppedSampleCount: 512, discardedPartialSampleCount: 0, droppedBlockCount: 0 });
  });

  it("does not charge an already counted absent quantum again when input resumes", () => {
    const f = fixture();
    f.render(128);
    f.processor.process([[]], [[new Float32Array(128)]]);
    f.context.currentFrame += 128;
    f.render(960);
    expect(f.blocks[0]).toMatchObject({ absoluteSampleIndex: 256, droppedSampleCount: 256 });
  });

  it("zeros retained PCM and rejects processing and late returns after disposal", () => {
    const f = fixture();
    f.render(); f.render(128);
    f.processor.port.onmessage?.({ data: { type: "dispose" } });
    f.recycle(f.blocks[0]);
    expect(f.render()).toBe(false);
    expect(f.processor.pending.every(value => value === 0)).toBe(true);
    expect(f.processor.pool).toHaveLength(0);
    expect(f.processor.inFlight.size).toBe(0);
    expect(f.blocks).toHaveLength(1);
    expect(f.controlMessages.at(-1)).toMatchObject({ type: "transport-diagnostics", final: true,
      captureEpoch: 7, sampleClockEndExclusive: 1088, droppedSampleCount: 128,
      discardedPartialSampleCount: 128, inFlightBlockCountAtDispose: 1 });
    const snapshot = f.controlMessages.at(-1);
    f.processor.port.onmessage?.({ data: { type: "dispose" } });
    expect(f.controlMessages.at(-1)).toEqual(snapshot);
  });

  it("reports terminal full-block and partial drops even when no later PCM can carry counters", () => {
    const f = fixture();
    for (let index = 0; index < 6; index += 1) f.render();
    f.render(128);
    f.processor.port.onmessage?.({ data: { type: "dispose" } });
    expect(f.controlMessages.at(-1)).toMatchObject({ droppedBlockCount: 2, droppedSampleCount: 2048,
      discardedPartialSampleCount: 128, sampleClockEndExclusive: 5888, inFlightBlockCountAtDispose: 4 });
  });
});
