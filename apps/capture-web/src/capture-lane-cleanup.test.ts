import { describe, expect, it, vi } from "vitest";
import {
  cleanupAudioLaneResources,
  cleanupFaceLaneResources
} from "./capture-lane-cleanup.js";

function streamWith(stop: () => void): MediaStream {
  return {
    getTracks: () => [{ stop } as MediaStreamTrack]
  } as MediaStream;
}

describe("lane-specific capture cleanup", () => {
  it("fully releases the audio lane without touching face resources", async () => {
    const abortAudio = vi.fn();
    const stopAudioTrack = vi.fn();
    const stopVoicePipeline = vi.fn(async () => undefined);
    const closeAudioContext = vi.fn(async () => undefined);
    const deactivateAudio = vi.fn();
    const stopFaceTrack = vi.fn();
    const stopFacePump = vi.fn();
    const disposeFaceWorker = vi.fn(async () => undefined);

    await cleanupAudioLaneResources({
      cancelPendingStartup: abortAudio,
      stream: streamWith(stopAudioTrack),
      pipeline: { stop: stopVoicePipeline },
      audioContext: {
        state: "running",
        close: closeAudioContext
      } as Pick<AudioContext, "state" | "close">,
      deactivate: deactivateAudio
    });

    expect(abortAudio).toHaveBeenCalledOnce();
    expect(stopAudioTrack).toHaveBeenCalledOnce();
    expect(stopVoicePipeline).toHaveBeenCalledOnce();
    expect(closeAudioContext).toHaveBeenCalledOnce();
    expect(deactivateAudio).toHaveBeenCalledOnce();
    expect(stopFaceTrack).not.toHaveBeenCalled();
    expect(stopFacePump).not.toHaveBeenCalled();
    expect(disposeFaceWorker).not.toHaveBeenCalled();
  });

  it("fully releases the face lane without touching audio resources", async () => {
    const stopFaceTrack = vi.fn();
    const stopFacePump = vi.fn();
    const stopFaceScheduler = vi.fn();
    const pauseVideo = vi.fn();
    const disposeFaceWorker = vi.fn(async () => undefined);
    const deactivateFace = vi.fn();
    const video = {
      pause: pauseVideo,
      srcObject: {} as MediaProvider
    };
    const stopAudioTrack = vi.fn();
    const stopVoicePipeline = vi.fn();
    const closeAudioContext = vi.fn();

    await cleanupFaceLaneResources({
      stream: streamWith(stopFaceTrack),
      pump: { stop: stopFacePump },
      scheduler: { stop: stopFaceScheduler },
      video,
      disposeWorker: disposeFaceWorker,
      deactivate: deactivateFace
    });

    expect(stopFaceTrack).toHaveBeenCalledOnce();
    expect(stopFacePump).toHaveBeenCalledOnce();
    expect(stopFaceScheduler).toHaveBeenCalledOnce();
    expect(pauseVideo).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
    expect(disposeFaceWorker).toHaveBeenCalledOnce();
    expect(deactivateFace).toHaveBeenCalledOnce();
    expect(stopAudioTrack).not.toHaveBeenCalled();
    expect(stopVoicePipeline).not.toHaveBeenCalled();
    expect(closeAudioContext).not.toHaveBeenCalled();
  });

  it("continues releasing later audio resources after an earlier failure", async () => {
    const stopTrack = vi.fn(() => {
      throw new Error("track already unavailable");
    });
    const stopPipeline = vi.fn(async () => {
      throw new Error("worker unavailable");
    });
    const closeContext = vi.fn(async () => undefined);

    await expect(cleanupAudioLaneResources({
      stream: streamWith(stopTrack),
      pipeline: { stop: stopPipeline },
      audioContext: {
        state: "running",
        close: closeContext
      } as Pick<AudioContext, "state" | "close">
    })).resolves.toBeUndefined();

    expect(stopTrack).toHaveBeenCalledOnce();
    expect(stopPipeline).toHaveBeenCalledOnce();
    expect(closeContext).toHaveBeenCalledOnce();
  });
});
