class PhenoMetrixVoiceCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.dataPort = null;
    this.disposed = false;
    this.naturalSource = null;
    this.naturalEnded = false;
    this.naturalCreditDrainSent = false;
    this.naturalFirstSamples = [];
    this.naturalFirstObservedFrame = null;
    this.naturalDiscardedPartial = 0;
    this.naturalTimelineEvents = [];
    this.naturalTimelineCapacity = 64;
    this.naturalTimelineOverflow = 0;
    this.naturalTimelineError = null;
    this.naturalResumeAudioFrame = null;
    this.naturalResumeEvent = null;
    this.finalDiagnostics = null;
    this.activeRenderLease = null;
    this.blockRenderLease = null;
    this.lastRenderLeaseId = 0;
    this.pendingClockProbe = null;
    this.lastClockProbeId = 0;
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
            this.acknowledgeNaturalCredits();
          }
        };
        this.dataPort.start();
      } else if (message?.type === "research-natural-source") {
        // Opt-in research fixture only. The ordinary microphone path never sends this.
        const valid = this.sequence === 0 && this.absoluteSampleIndex === 0 && !this.naturalSource &&
          message.captureEpoch === this.captureEpoch && Number.isSafeInteger(message.startAudioFrame) &&
          message.startAudioFrame > currentFrame && message.startAudioFrame % 1920 === 0 &&
          Number.isSafeInteger(message.sampleCount) && message.sampleCount >= 1920 && message.sampleCount <= 48000 * 120;
        if (valid) {
          this.naturalSource = { startAudioFrame: message.startAudioFrame, sampleCount: message.sampleCount };
          this.dataPort?.postMessage({ type: "research-natural-start", captureEpoch: this.captureEpoch,
            startAudioFrame: message.startAudioFrame, sampleCount: message.sampleCount });
        }
        this.port.postMessage({ type: "research-natural-armed", status: valid ? "armed" : "rejected",
          captureEpoch: this.captureEpoch, startAudioFrame: message.startAudioFrame });
      } else if (message?.type === "render-freshness-lease") {
        const lease = message.lease;
        const valid = lease?.schemaVersion === "phenometric.voice-render-freshness.v1" &&
          Number.isSafeInteger(lease.leaseId) && lease.leaseId > this.lastRenderLeaseId &&
          Number.isFinite(lease.mainTimeOriginMs) && Number.isFinite(lease.issuedAtMainMs) && lease.issuedAtMainMs >= 0 &&
          lease.expiresAtMainMs === lease.issuedAtMainMs + 100;
        const accepted = valid && lease.captureEpoch === this.captureEpoch && lease.clockMappingId === this.clockMappingId;
        if (valid) this.lastRenderLeaseId = lease.leaseId;
        if (accepted) this.activeRenderLease = { ...lease };
        if (lease && Number.isSafeInteger(lease.leaseId)) this.port.postMessage({
          schemaVersion: "phenometric.voice-render-freshness.v1", type: "render-freshness-lease-ack",
          leaseId: lease.leaseId, captureEpoch: lease.captureEpoch, clockMappingId: lease.clockMappingId,
          disposition: accepted ? "accepted" : "retired" });
      } else if (message?.schemaVersion === "phenometric.voice-clock-probe.v1" && message.type === "clock-probe") {
        if (!this.pendingClockProbe && Number.isSafeInteger(message.probeId) && message.probeId > this.lastClockProbeId &&
            message.captureEpoch === this.captureEpoch && message.clockMappingId === this.clockMappingId) {
          this.pendingClockProbe = message;
          this.lastClockProbeId = message.probeId;
        }
      } else if (message?.type === "capture-epoch") {
        this.activeRenderLease = null; this.blockRenderLease = null;
        this.pendingClockProbe = null;
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
        this.activeRenderLease = null; this.blockRenderLease = null;
        this.pendingClockProbe = null;
        this.clockMappingId = message.clockMappingId;
        this.droppedSampleCount += this.pendingLength;
        this.pending.fill(0);
        this.pendingLength = 0;
      } else if (message?.type === "dispose") {
        this.activeRenderLease = null; this.blockRenderLease = null;
        this.pendingClockProbe = null;
        this.droppedSampleCount += this.pendingLength;
        this.finalDiagnostics = {
          schemaVersion: "phenometric.voice-worklet-message.v2", type: "transport-diagnostics",
          captureEpoch: this.captureEpoch, final: true, sequence: this.sequence,
          sampleClockEndExclusive: this.absoluteSampleIndex,
          droppedBlockCount: this.droppedBlockCount, droppedSampleCount: this.droppedSampleCount,
          discardedPartialSampleCount: this.pendingLength + this.naturalDiscardedPartial,
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
        renderLease: this.blockRenderLease,
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

  acknowledgeNaturalCredits() {
    if (!this.naturalEnded || this.naturalCreditDrainSent || this.inFlight.size !== 0) return;
    this.naturalCreditDrainSent = true;
    this.port.postMessage({ type: "research-natural-credits-drained", captureEpoch: this.captureEpoch,
      scheduledStartAudioFrame: this.naturalSource.startAudioFrame, sourceSampleCount: this.naturalSource.sampleCount,
      sequence: this.sequence, inFlightBlockCount: 0, pendingSampleCount: this.pendingLength });
  }

  finishNaturalSource() {
    if (this.naturalEnded) return;
    this.naturalEnded = true;
    this.naturalDiscardedPartial = this.pendingLength;
    this.droppedSampleCount += this.pendingLength;
    this.pending.fill(0); this.pendingLength = 0;
    const receipt = { type: "research-natural-ended", captureEpoch: this.captureEpoch,
      scheduledStartAudioFrame: this.naturalSource.startAudioFrame,
      observedStartAudioFrame: this.naturalFirstObservedFrame,
      sourceSampleCount: this.naturalSource.sampleCount, sourceOffset: 0,
      captureEndSampleExclusive: this.absoluteSampleIndex, firstSamples: this.naturalFirstSamples,
      sequence: this.sequence, droppedBlockCount: this.droppedBlockCount,
      droppedSampleCount: this.droppedSampleCount, discardedPartialSampleCount: this.naturalDiscardedPartial,
      timeline: { schemaVersion: "phenometric.voice-natural-timeline.v2", status: this.naturalTimelineError ? "failed" : "complete",
        error: this.naturalTimelineError, events: this.naturalTimelineEvents, capacity: this.naturalTimelineCapacity,
        overflow: this.naturalTimelineOverflow } };
    this.port.postMessage(receipt);
    this.dataPort?.postMessage({ type: "research-natural-end", captureEpoch: this.captureEpoch, fenceId: 1 });
    this.acknowledgeNaturalCredits();
  }

  recordNaturalTimeline(event) {
    if (this.naturalTimelineEvents.length === this.naturalTimelineCapacity) {
      this.naturalTimelineOverflow += 1;
      this.naturalTimelineError = "timeline-event-overflow";
      this.finishNaturalSource();
      return false;
    }
    this.naturalTimelineEvents.push(event);
    return true;
  }

  processNaturalSource(channel, quantumLength) {
    if (this.naturalEnded) return true;
    const start = this.naturalSource.startAudioFrame;
    const end = start + this.naturalSource.sampleCount;
    if (currentFrame + quantumLength <= start) return true;
    const first = Math.max(0, start - currentFrame);
    const renderStart = Math.min(end, currentFrame + first);
    const renderEnd = Math.min(end, currentFrame + quantumLength);
    const expected = this.expectedAudioFrame ?? start;
    const event = { expectedAudioFrame: expected, observedStartAudioFrame: renderStart,
      observedEndAudioFrameExclusive: renderEnd, discardedPendingSampleCount: this.pendingLength };
    // A backward callback cannot establish what samples the source supplied.
    // Retain the fault and stop accepting PCM; never reuse, clamp, or relabel it.
    if (renderStart < expected) {
      this.naturalTimelineError = "overlapping-render-quantum";
      this.recordNaturalTimeline({ ...event, kind: "overlap-rejected", overlapSampleCount: expected - renderStart });
      this.finishNaturalSource();
      return true;
    }
    if (channel && channel.length !== quantumLength) {
      this.naturalTimelineError = "unknown-input-quantum-length";
      this.recordNaturalTimeline({ ...event, kind: "input-shape-rejected", inputSampleCount: channel.length, quantumLength });
      this.finishNaturalSource();
      return true;
    }
    if (!channel || renderStart > expected) {
      const missingEnd = !channel ? renderEnd : renderStart;
      const missing = missingEnd - expected;
      if (!this.recordNaturalTimeline({ ...event, kind: channel ? "forward-gap" : "absent-input", missingSampleCount: missing })) return true;
      this.droppedSampleCount += missing + this.pendingLength;
      this.pending.fill(0); this.pendingLength = 0;
      this.absoluteSampleIndex = missingEnd - start;
      this.naturalResumeAudioFrame = null;
      this.naturalResumeEvent = null;
    }
    this.expectedAudioFrame = renderEnd;
    if (!channel || renderStart === end) {
      if (renderEnd === end) this.finishNaturalSource();
      return true;
    }
    channel = channel.subarray(first, first + renderEnd - renderStart);
    if (this.naturalFirstObservedFrame === null) {
      this.naturalFirstObservedFrame = renderStart;
      this.naturalFirstSamples = Array.from(channel.subarray(0, 16));
    }
    let sourceIndex = 0;
    // A gap discards partial support. Resume on the original transport grid,
    // charging every skipped source sample; do not shift the reference grid.
    if (this.pendingLength === 0 && this.naturalResumeAudioFrame === null && (renderStart - start) % this.blockSamples !== 0) {
      this.naturalResumeAudioFrame = Math.min(end, start + Math.ceil((renderStart - start) / this.blockSamples) * this.blockSamples);
      this.naturalResumeEvent = { kind: "grid-restart", startAudioFrame: renderStart,
        endAudioFrameExclusive: renderStart, targetAudioFrame: this.naturalResumeAudioFrame, skippedSampleCount: 0 };
      if (!this.recordNaturalTimeline(this.naturalResumeEvent)) return true;
    }
    if (this.naturalResumeAudioFrame !== null) {
      sourceIndex = Math.min(channel.length, this.naturalResumeAudioFrame - renderStart);
      this.droppedSampleCount += sourceIndex;
      this.naturalResumeEvent.endAudioFrameExclusive += sourceIndex;
      this.naturalResumeEvent.skippedSampleCount += sourceIndex;
      this.absoluteSampleIndex = renderStart + sourceIndex - start;
      if (renderStart + sourceIndex === this.naturalResumeAudioFrame) this.naturalResumeAudioFrame = null;
    }
    while (sourceIndex < channel.length) {
      this.absoluteSampleIndex = renderStart + sourceIndex - start;
      if (this.pendingLength === 0) {
        this.pendingStartSampleIndex = this.absoluteSampleIndex;
        this.blockRenderLease = this.activeRenderLease;
      }
      const count = Math.min(channel.length - sourceIndex, this.blockSamples - this.pendingLength);
      this.pending.set(channel.subarray(sourceIndex, sourceIndex + count), this.pendingLength);
      this.pendingLength += count;
      sourceIndex += count;
      this.absoluteSampleIndex = renderStart + sourceIndex - start;
      if (this.pendingLength === this.blockSamples) this.emitBlock(renderStart + sourceIndex);
    }
    if (renderEnd === end) this.finishNaturalSource();
    return true;
  }

  process(inputs, outputs) {
    if (this.disposed) return false;
    if (this.pendingClockProbe) {
      const probe = this.pendingClockProbe;
      this.pendingClockProbe = null;
      this.port.postMessage({ schemaVersion: "phenometric.voice-clock-probe.v1", type: "clock-probe-result",
        probeId: probe.probeId, captureEpoch: probe.captureEpoch, clockMappingId: probe.clockMappingId,
        renderStartFrame: currentFrame, sampleRateHz: sampleRate });
    }
    let channel = inputs[0]?.[0];
    const quantumLength = outputs?.[0]?.[0]?.length ?? channel?.length ?? this.lastQuantumLength;
    this.lastQuantumLength = quantumLength;
    if (this.naturalSource) return this.processNaturalSource(channel, quantumLength);
    const renderStartFrame = currentFrame;
    if (!channel) {
      // Before the first observed input there is no capture sample origin yet.
      // Once established, charge absent quanta immediately, including at stop.
      if (this.expectedAudioFrame !== null) {
        const missingSamples = Math.max(0, renderStartFrame - this.expectedAudioFrame) + quantumLength;
        this.absoluteSampleIndex += missingSamples;
        this.droppedSampleCount += missingSamples + this.pendingLength;
        this.pending.fill(0);
        this.pendingLength = 0;
        this.expectedAudioFrame = renderStartFrame + quantumLength;
      }
      return true;
    }
    if (this.expectedAudioFrame !== null && renderStartFrame !== this.expectedAudioFrame) {
      // A missing input quantum is not observed silence. Preserve its sample gap.
      const missingSamples = Math.max(0, renderStartFrame - this.expectedAudioFrame);
      this.absoluteSampleIndex += missingSamples;
      this.droppedSampleCount += missingSamples + this.pendingLength;
      this.pending.fill(0);
      this.pendingLength = 0;
    }
    this.expectedAudioFrame = renderStartFrame + channel.length;
    let sourceIndex = 0;
    while (sourceIndex < channel.length) {
      if (this.pendingLength === 0) {
        this.pendingStartSampleIndex = this.absoluteSampleIndex;
        this.blockRenderLease = this.activeRenderLease;
      }
      const count = Math.min(channel.length - sourceIndex, this.blockSamples - this.pendingLength);
      this.pending.set(channel.subarray(sourceIndex, sourceIndex + count), this.pendingLength);
      this.pendingLength += count;
      this.absoluteSampleIndex += count;
      sourceIndex += count;
      if (this.pendingLength === this.blockSamples) this.emitBlock(renderStartFrame + sourceIndex);
    }
    return true;
  }
}

registerProcessor("phenometric-voice-capture", PhenoMetrixVoiceCaptureProcessor);
