import type {
  FacialKinematicsFrameV1,
  VoiceSignalFrameV1
} from "@phenometrix/ambient-core";

export interface CaptureResourceHandles {
  cancelPendingStartup?(): void;
  stopFacePump?(): void;
  stopVoicePipeline?(): Promise<void>;
  disposeFaceWorker?(): Promise<void>;
  streams?: readonly MediaStream[];
  video?: HTMLVideoElement;
  disconnectAudio?(): void;
  audioContext?: AudioContext;
  cancelTimers?(): void;
}

export interface DerivedCaptureSnapshot {
  readonly voice: readonly VoiceSignalFrameV1[];
  readonly face: readonly FacialKinematicsFrameV1[];
}

function freezeSnapshot(
  voice: readonly VoiceSignalFrameV1[],
  face: readonly FacialKinematicsFrameV1[]
): DerivedCaptureSnapshot {
  return Object.freeze({
    voice: Object.freeze(voice.map((frame) => Object.freeze({ ...frame }))),
    face: Object.freeze(face.map((frame) => Object.freeze({ ...frame })))
  });
}

async function ignoreFailure(action: (() => void | Promise<void>) | undefined): Promise<void> {
  if (!action) return;
  try {
    await action();
  } catch {
    // Teardown is best-effort per resource and must continue to later resources.
  }
}

async function cleanupHandles(handles: CaptureResourceHandles): Promise<void> {
  await ignoreFailure(handles.cancelTimers);
  await ignoreFailure(handles.cancelPendingStartup);
  await ignoreFailure(handles.stopFacePump);
  handles.streams?.forEach((stream) =>
    stream.getTracks().forEach((track) => track.stop())
  );
  if (handles.video) {
    handles.video.pause();
    handles.video.srcObject = null;
  }
  await ignoreFailure(handles.stopVoicePipeline);
  await ignoreFailure(handles.disposeFaceWorker);
  await ignoreFailure(handles.disconnectAudio);
  if (handles.audioContext && handles.audioContext.state !== "closed") {
    await ignoreFailure(() => handles.audioContext!.close());
  }
}

export class CaptureRuntime {
  private voiceFrames: VoiceSignalFrameV1[] = [];
  private faceFrames: FacialKinematicsFrameV1[] = [];
  private handles: CaptureResourceHandles = {};
  private disposing: Promise<DerivedCaptureSnapshot | null> | null = null;
  private preserveDerivedOnDispose = false;
  private disposed = false;
  private pendingLateCleanups = new Set<Promise<void>>();

  attach(handles: CaptureResourceHandles): void {
    if (this.disposed || this.disposing) {
      const cleanup = cleanupHandles(handles);
      this.pendingLateCleanups.add(cleanup);
      void cleanup.finally(() => this.pendingLateCleanups.delete(cleanup));
      return;
    }
    this.handles = handles;
  }

  addVoiceFrame(frame: VoiceSignalFrameV1): void {
    if (!this.disposed && !this.disposing) this.voiceFrames.push(frame);
  }

  addFaceFrame(frame: FacialKinematicsFrameV1): void {
    if (!this.disposed && !this.disposing) this.faceFrames.push(frame);
  }

  dispose(preserveDerived: boolean): Promise<DerivedCaptureSnapshot | null> {
    if (this.disposing) {
      // A discard/withdrawal racing a report finalization always wins. The
      // in-flight cleanup must not manufacture a derived snapshot after a
      // caller has requested destructive disposal.
      if (!preserveDerived) this.preserveDerivedOnDispose = false;
      return this.disposing;
    }
    if (this.disposed) return Promise.resolve(null);
    this.preserveDerivedOnDispose = preserveDerived;
    const disposal = this.disposeOnce();
    this.disposing = disposal;
    void disposal.then(
      () => {
        if (this.disposing === disposal) this.disposing = null;
      },
      () => {
        if (this.disposing === disposal) this.disposing = null;
      }
    );
    return disposal;
  }

  private async disposeOnce(): Promise<DerivedCaptureSnapshot | null> {
    const handles = this.handles;
    await cleanupHandles(handles);
    while (this.pendingLateCleanups.size > 0) {
      await Promise.all([...this.pendingLateCleanups]);
    }

    const snapshot = this.preserveDerivedOnDispose
      ? freezeSnapshot(this.voiceFrames, this.faceFrames)
      : null;
    this.voiceFrames.length = 0;
    this.faceFrames.length = 0;
    this.handles = {};
    this.preserveDerivedOnDispose = false;
    this.disposed = true;
    return snapshot;
  }
}

export async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  fallback: () => T
): Promise<T> {
  let handle: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<T>((resolve) => {
    handle = setTimeout(() => resolve(fallback()), timeoutMs);
  });
  const result = await Promise.race([promise, timeout]);
  if (handle !== undefined) clearTimeout(handle);
  return result;
}
