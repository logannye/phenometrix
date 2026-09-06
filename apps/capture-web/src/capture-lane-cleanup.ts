export interface AudioLaneCleanupResources {
  cancelPendingStartup?: () => void;
  stream?: Pick<MediaStream, "getTracks"> | null;
  pipeline?: { stop(): Promise<void> } | null;
  audioContext?: Pick<AudioContext, "state" | "close"> | null;
  deactivate?: () => void;
}

export interface FaceLaneCleanupResources {
  stream?: Pick<MediaStream, "getTracks"> | null;
  pump?: { stop(): void } | null;
  scheduler?: { stop(): void } | null;
  video?: Pick<HTMLVideoElement, "pause" | "srcObject"> | null;
  disposeWorker?: () => Promise<void>;
  deactivate?: () => void;
}

function ignoreSync(action: (() => void) | undefined): void {
  if (!action) return;
  try {
    action();
  } catch {
    // Teardown continues independently for every resource in the lane.
  }
}

async function ignoreAsync(
  action: (() => void | Promise<void>) | undefined
): Promise<void> {
  if (!action) return;
  try {
    await action();
  } catch {
    // Teardown continues independently for every resource in the lane.
  }
}

function stopTracks(
  stream: Pick<MediaStream, "getTracks"> | null | undefined
): void {
  if (!stream) return;
  let tracks: MediaStreamTrack[];
  try {
    tracks = stream.getTracks();
  } catch {
    return;
  }
  for (const track of tracks) {
    ignoreSync(() => track.stop());
  }
}

/**
 * Releases only microphone-lane resources. Track shutdown is synchronous so
 * UI state can accurately stop claiming that the device is live immediately.
 */
export async function cleanupAudioLaneResources(
  resources: AudioLaneCleanupResources
): Promise<void> {
  ignoreSync(resources.cancelPendingStartup);
  ignoreSync(resources.deactivate);
  stopTracks(resources.stream);
  await ignoreAsync(
    resources.pipeline ? () => resources.pipeline!.stop() : undefined
  );
  if (
    resources.audioContext &&
    resources.audioContext.state !== "closed"
  ) {
    await ignoreAsync(() => resources.audioContext!.close());
  }
}

/** Releases only camera-lane resources, including retained worker state. */
export async function cleanupFaceLaneResources(
  resources: FaceLaneCleanupResources
): Promise<void> {
  ignoreSync(() => resources.pump?.stop());
  ignoreSync(() => resources.scheduler?.stop());
  ignoreSync(resources.deactivate);
  stopTracks(resources.stream);
  ignoreSync(() => resources.video?.pause());
  ignoreSync(() => {
    if (resources.video) resources.video.srcObject = null;
  });
  await ignoreAsync(resources.disposeWorker);
}
