class PhenoMetrixVoiceCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.dataPort = null;
    this.disposed = false;
    this.finalDiagnostics = null;
    this.captureEpoch = 0;
    this.clockMappingId = 0;
    this.sequence = 0;
    this.absoluteSampleIndex = 0;
    this.pendingStartSampleIndex = 0;
    this.expectedAudioFrame = null;
    this.lastQuantumLength = 128;
    this.blockSamples = Math.max(1, Math.round(sampleRate * 0.02));
    this.pending = new Float32Array(this.blockSamples);
    this.pendingLength = 0;
    this.maximumInFlightBlocks = 4;
    this.pool = Array.from({ length: this.maximumInFlightBlocks }, () =>
      new ArrayBuffer(this.blockSamples * Float32Array.BYTES_PER_ELEMENT));
    this.inFlight = new Set();
    this.nextTransportId = 0;
    this.droppedBlockCount = 0;
    this.droppedSampleCount = 0;
    this.port.onmessage = (event) => {
      const message = event.data;
      if (this.disposed) {
        if (message?.type === "dispose" && this.finalDiagnostics) this.port.postMessage(this.finalDiagnostics);
        return;
      }
      if (message?.type === "attach-port" && message.port) {
        this.dataPort?.close();
        this.dataPort = message.port;
        this.dataPort.onmessage = (portEvent) => {
          const recycled = portEvent.data;
          if (!this.disposed && recycled?.type === "recycle" &&
              this.inFlight.has(recycled.transportId) &&
              recycled.buffer instanceof ArrayBuffer &&
              recycled.buffer.byteLength === this.blockSamples * Float32Array.BYTES_PER_ELEMENT) {
            this.inFlight.delete(recycled.transportId);
            this.pool.push(recycled.buffer);
          }
        };
        this.dataPort.start();
      } else if (message?.type === "capture-epoch") {
        this.pending.fill(0);
        this.captureEpoch = message.captureEpoch;
        this.clockMappingId = message.clockMappingId;
        this.sequence = 0;
        this.absoluteSampleIndex = 0;
        this.pendingLength = 0;
        this.expectedAudioFrame = null;
        this.droppedBlockCount = 0;
        this.droppedSampleCount = 0;
        // Outstanding credits survive epoch changes. Late returns cannot create
        // additional buffers or cause an old block to be read in the new epoch.
      } else if (message?.type === "clock-mapping") {
        this.clockMappingId = message.clockMappingId;
        this.droppedSampleCount += this.pendingLength;
        this.pending.fill(0);
        this.pendingLength = 0;
      } else if (message?.type === "dispose") {
        this.droppedSampleCount += this.pendingLength;
        this.finalDiagnostics = {
          schemaVersion: "phenometric.voice-worklet-message.v2", type: "transport-diagnostics",
          captureEpoch: this.captureEpoch, final: true, sequence: this.sequence,
          sampleClockEndExclusive: this.absoluteSampleIndex,
          droppedBlockCount: this.droppedBlockCount, droppedSampleCount: this.droppedSampleCount,
          discardedPartialSampleCount: this.pendingLength,
          inFlightBlockCountAtDispose: this.inFlight.size,
          maximumInFlightBlocks: this.maximumInFlightBlocks
        };
        this.port.postMessage(this.finalDiagnostics);
        this.disposed = true;
        this.dataPort?.close();
        this.dataPort = null;
        this.pending.fill(0);
        for (const buffer of this.pool) new Float32Array(buffer).fill(0);
        this.pool.length = 0;
        this.inFlight.clear();
        this.pendingLength = 0;
      }
    };
  }

  emitBlock(endAudioFrameExclusive) {
    this.sequence += 1;
    if (!this.dataPort || this.pool.length === 0) {
      this.droppedBlockCount += 1;
      this.droppedSampleCount += this.blockSamples;
      this.pending.fill(0);
      this.pendingLength = 0;
      return;
    }
    const buffer = this.pool.pop();
    new Float32Array(buffer).set(this.pending);
    const transportId = ++this.nextTransportId;
    this.inFlight.add(transportId);
    this.dataPort.postMessage(
      {
        schemaVersion: "phenometric.voice-worklet-message.v2",
        type: "pcm-block",
        captureEpoch: this.captureEpoch,
        clockMappingId: this.clockMappingId,
        sequence: this.sequence,
        absoluteSampleIndex: this.pendingStartSampleIndex,
        acquisitionAudioTimeSeconds: currentTime,
        endAudioFrameExclusive,
        transportId,
        droppedBlockCount: this.droppedBlockCount,
        droppedSampleCount: this.droppedSampleCount,
        inFlightBlockCount: this.inFlight.size,
        sampleRateHz: sampleRate,
        channelCount: 1,
        buffer
      },
      [buffer]
    );
    this.pending.fill(0);
    this.pendingLength = 0;
  }

  process(inputs, outputs) {
    if (this.disposed) return false;
    const channel = inputs[0]?.[0];
    const quantumLength = outputs?.[0]?.[0]?.length ?? channel?.length ?? this.lastQuantumLength;
    this.lastQuantumLength = quantumLength;
    if (!channel) {
      // Before the first observed input there is no capture sample origin yet.
      // Once established, charge absent quanta immediately, including at stop.
      if (this.expectedAudioFrame !== null) {
        const missingSamples = Math.max(0, currentFrame - this.expectedAudioFrame) + quantumLength;
        this.absoluteSampleIndex += missingSamples;
        this.droppedSampleCount += missingSamples + this.pendingLength;
        this.pending.fill(0);
        this.pendingLength = 0;
        this.expectedAudioFrame = currentFrame + quantumLength;
      }
      return true;
    }
    if (this.expectedAudioFrame !== null && currentFrame !== this.expectedAudioFrame) {
      // A missing input quantum is not observed silence. Preserve its sample gap.
      const missingSamples = Math.max(0, currentFrame - this.expectedAudioFrame);
      this.absoluteSampleIndex += missingSamples;
      this.droppedSampleCount += missingSamples + this.pendingLength;
      this.pending.fill(0);
      this.pendingLength = 0;
    }
    this.expectedAudioFrame = currentFrame + channel.length;
    let sourceIndex = 0;
    while (sourceIndex < channel.length) {
      if (this.pendingLength === 0) this.pendingStartSampleIndex = this.absoluteSampleIndex;
      const count = Math.min(channel.length - sourceIndex, this.blockSamples - this.pendingLength);
      this.pending.set(channel.subarray(sourceIndex, sourceIndex + count), this.pendingLength);
      this.pendingLength += count;
      this.absoluteSampleIndex += count;
      sourceIndex += count;
      if (this.pendingLength === this.blockSamples) this.emitBlock(currentFrame + sourceIndex);
    }
    return true;
  }
}

registerProcessor("phenometric-voice-capture", PhenoMetrixVoiceCaptureProcessor);
