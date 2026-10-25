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
  it("binds scheduled finite source zero exactly and fences a retained final partial without post-source silence", () => {
    const f = fixture();
    f.processor.port.onmessage?.({ data: { type: "research-natural-source", captureEpoch: 7,
      startAudioFrame: 49920, sampleCount: 2017 } });
    expect(f.controlMessages.at(-1)).toMatchObject({ status: "armed" });
    f.render(1920, 0); // Before the scheduled source; must not enter capture coordinates.
    expect(f.blocks.filter(block => block.type === "pcm-block")).toHaveLength(0);
    const waveform = Float32Array.from({ length: 2048 }, (_, i) => i < 2017 ? (i + 1) / 4096 : -1);
    f.processor.process([[waveform]]);
    const pcmBlocks = f.blocks.filter(block => block.type === "pcm-block");
    expect(pcmBlocks.map(block => block.absoluteSampleIndex)).toEqual([0, 960]);
    expect(pcmBlocks.map(block => block.endAudioFrameExclusive)).toEqual([50880, 51840]);
    expect(new Float32Array(pcmBlocks[0].buffer)).toEqual(waveform.slice(0, 960));
    expect(new Float32Array(pcmBlocks[1].buffer)).toEqual(waveform.slice(960, 1920));
    expect(f.blocks.at(-1)).toMatchObject({ type: "research-natural-end", fenceId: 1 });
    expect(f.controlMessages.at(-1)).toMatchObject({ type: "research-natural-ended", scheduledStartAudioFrame: 49920,
      observedStartAudioFrame: 49920, sourceOffset: 0, captureEndSampleExclusive: 2017,
      discardedPartialSampleCount: 97, droppedSampleCount: 97, firstSamples: Array.from(waveform.slice(0, 16)) });
    f.context.currentFrame += 2048;
    f.render(128, 0);
    f.processor.port.onmessage?.({ data: { type: "dispose" } });
    expect(f.controlMessages.at(-1)).toMatchObject({ sampleClockEndExclusive: 2017, discardedPartialSampleCount: 97, droppedSampleCount: 97 });
  });

  it("acknowledges finite PCM credit drain only after the final delayed recycle is observed", () => {
    const f = fixture();
    f.processor.port.onmessage?.({ data: { type: "research-natural-source", captureEpoch: 7, startAudioFrame: 49920, sampleCount: 1920 } });
    f.context.currentFrame = 49920;
    f.render(1920, .1);
    const pcm = f.blocks.filter(block => block.type === "pcm-block");
    expect(f.controlMessages.some(row => row.type === "research-natural-ended")).toBe(true);
    expect(f.controlMessages.some(row => row.type === "research-natural-credits-drained")).toBe(false);
    f.recycle(pcm[0]);
    expect(f.controlMessages.some(row => row.type === "research-natural-credits-drained")).toBe(false);
    f.recycle(pcm[1]);
    expect(f.controlMessages.at(-1)).toMatchObject({ type: "research-natural-credits-drained", captureEpoch: 7,
      sourceSampleCount: 1920, sequence: 2, inFlightBlockCount: 0, pendingSampleCount: 0 });
    f.recycle(pcm[1]);
    expect(f.controlMessages.filter(row => row.type === "research-natural-credits-drained")).toHaveLength(1);
    f.processor.port.onmessage?.({ data: { type: "dispose" } });
    expect(f.controlMessages.at(-1)).toMatchObject({ inFlightBlockCountAtDispose: 0 });
  });

  it("rejects late and off-grid finite-source starts, preserving the ordinary transport", () => {
    const f = fixture();
    for (const startAudioFrame of [48000, 49921]) {
      f.processor.port.onmessage?.({ data: { type: "research-natural-source", captureEpoch: 7, startAudioFrame, sampleCount: 1920 } });
      expect(f.controlMessages.at(-1)).toMatchObject({ status: "rejected" });
    }
    f.render(); expect(f.blocks[0].absoluteSampleIndex).toBe(0);
  });

  it("retains source-coordinate loss when initial input is absent instead of rebinding zero", () => {
    const f = fixture();
    f.processor.port.onmessage?.({ data: { type: "research-natural-source", captureEpoch: 7, startAudioFrame: 49920, sampleCount: 1920 } });
    f.context.currentFrame = 49920;
    f.processor.process([[]], [[new Float32Array(128)]]);
    f.context.currentFrame += 128; f.render(1792, .2);
    expect(f.blocks.find(block => block.type === "pcm-block")).toMatchObject({ absoluteSampleIndex: 960, droppedSampleCount: 960 });
    expect(f.controlMessages.at(-1)).toMatchObject({ observedStartAudioFrame: 50048,
      captureEndSampleExclusive: 1920, droppedSampleCount: 960, discardedPartialSampleCount: 0 });
  });

  it("retains a positive source gap and resumes at the original block grid with exact PCM and endcap", () => {
    const f = fixture();
    const start = 49920;
    f.processor.port.onmessage?.({ data: { type: "research-natural-source", captureEpoch: 7, startAudioFrame: start, sampleCount: 3969 } });
    const renderAt = (offset: number, length: number) => {
      f.context.currentFrame = start + offset;
      f.processor.process([[Float32Array.from({ length }, (_, i) => (offset + i) / 8192)]]);
    };
    renderAt(0, 128);
    renderAt(256, 128); // One missing quantum; first128 is discarded partial support.
    renderAt(384, 640); // Finish charging grid restart, then begin source960.
    renderAt(1024, 3072); // Exact3969 end falls within this callback.
    const blocks = f.blocks.filter(row => row.type === "pcm-block");
    expect(blocks.map(row => row.absoluteSampleIndex)).toEqual([960, 1920, 2880]);
    for (const block of blocks) {
      expect(block.endAudioFrameExclusive).toBe(start + block.absoluteSampleIndex + 960);
      expect(new Float32Array(block.buffer)).toEqual(Float32Array.from({ length: 960 }, (_, i) => (block.absoluteSampleIndex + i) / 8192));
    }
    const receipt = f.controlMessages.find(row => row.type === "research-natural-ended");
    expect(receipt).toMatchObject({ captureEndSampleExclusive: 3969, discardedPartialSampleCount: 129, droppedSampleCount: 1089,
      timeline: { status: "complete", overflow: 0, events: [
        { kind: "forward-gap", expectedAudioFrame: start + 128, observedStartAudioFrame: start + 256,
          missingSampleCount: 128, discardedPendingSampleCount: 128 },
        { kind: "grid-restart", startAudioFrame: start + 256, endAudioFrameExclusive: start + 960, skippedSampleCount: 704 }
      ] } });
  });

  it.each([128, 64])("rejects a %i-sample duplicate/overlap before consuming ambiguous PCM", rewind => {
    const f = fixture();
    const start = 49920;
    f.processor.port.onmessage?.({ data: { type: "research-natural-source", captureEpoch: 7, startAudioFrame: start, sampleCount: 3840 } });
    f.context.currentFrame = start;
    f.render(1792); // One emitted block and exactly832 pending samples, matching attempt1.
    f.context.currentFrame -= rewind;
    f.render(128, -.5);
    const receipt = f.controlMessages.find(row => row.type === "research-natural-ended");
    expect(receipt).toMatchObject({ captureEndSampleExclusive: 1792, discardedPartialSampleCount: 832, droppedSampleCount: 832,
      timeline: { status: "failed", error: "overlapping-render-quantum", overflow: 0,
        events: [{ kind: "overlap-rejected", expectedAudioFrame: start + 1792,
          observedStartAudioFrame: start + 1792 - rewind, overlapSampleCount: rewind, discardedPendingSampleCount: 832 }] } });
    expect(f.blocks.filter(row => row.type === "pcm-block")).toHaveLength(1);
    f.context.currentFrame = start + 3840; f.render();
    expect(f.controlMessages.filter(row => row.type === "research-natural-ended")).toHaveLength(1);
    expect(f.blocks.filter(row => row.type === "pcm-block")).toHaveLength(1);
  });

  it("rejects unknown finite input shapes and bounds retained discontinuity receipts", () => {
    const f = fixture();
    f.processor.port.onmessage?.({ data: { type: "research-natural-source", captureEpoch: 7, startAudioFrame: 49920, sampleCount: 48000 } });
    f.context.currentFrame = 49920;
    f.processor.process([[new Float32Array(64)]], [[new Float32Array(128)]]);
    expect(f.controlMessages.at(-1)).toMatchObject({ type: "research-natural-credits-drained" });
    expect(f.controlMessages.find(row => row.type === "research-natural-ended")).toMatchObject({
      timeline: { status: "failed", error: "unknown-input-quantum-length" } });
    const g = fixture();
    g.processor.port.onmessage?.({ data: { type: "research-natural-source", captureEpoch: 7, startAudioFrame: 49920, sampleCount: 48000 } });
    g.context.currentFrame = 49920;
    for (let i = 0; i < 65; i += 1) { g.processor.process([[]], [[new Float32Array(128)]]); g.context.currentFrame += 128; }
    const receipt = g.controlMessages.find(row => row.type === "research-natural-ended");
    expect(receipt).toMatchObject({ timeline: { status: "failed", error: "timeline-event-overflow", capacity: 64, overflow: 1 } });
    expect((receipt?.timeline as { events: unknown[] }).events).toHaveLength(64);
  });

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
