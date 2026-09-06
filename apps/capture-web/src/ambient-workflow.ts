export const AMBIENT_SETUP_TIMEOUT_MS = 15_000;
export const AMBIENT_CAPTURE_LIMIT_MS = 300_000;

export type AmbientPhase =
  | "idle"
  | "requesting-permission"
  | "calibrating"
  | "observing"
  | "finalizing"
  | "report"
  | "discarded"
  | "error";

export type AmbientLaneState =
  | "off"
  | "requesting"
  | "calibrating"
  | "measurable"
  | "not-measurable";

export interface AmbientWorkflowState {
  readonly phase: AmbientPhase;
  readonly generation: number;
  readonly consented: boolean;
  readonly audioLane: AmbientLaneState;
  readonly faceLane: AmbientLaneState;
  readonly setupStartedAtMs: number | null;
  readonly captureStartedAtMs: number | null;
  readonly terminalReason: string | null;
}

export type AmbientWorkflowEvent =
  | { type: "consent-changed"; consented: boolean }
  | { type: "start-requested"; atMs: number }
  | {
      type: "permission-resolved";
      generation: number;
      lane: "audio" | "face";
      available: boolean;
      atMs: number;
    }
  | {
      type: "calibration-resolved";
      generation: number;
      lane: "audio" | "face";
      measurable: boolean;
      atMs: number;
    }
  | {
      type: "lane-failed";
      generation: number;
      lane: "audio" | "face";
      remainingLaneActive: boolean;
      atMs: number;
    }
  | { type: "setup-timeout"; generation: number; atMs: number }
  | { type: "finish-requested"; generation: number }
  | { type: "capture-limit-reached"; generation: number }
  | { type: "finalization-completed"; generation: number }
  | { type: "finalization-failed"; generation: number; reason: string }
  | { type: "visibility-lost" }
  | { type: "discard-requested"; reason: string }
  | { type: "reset" };

export type AmbientWorkflowEffect =
  | { type: "request-media"; generation: number }
  | { type: "begin-calibration"; generation: number }
  | { type: "begin-observation"; generation: number }
  | { type: "finalize"; generation: number }
  | { type: "dispose"; generation: number; discard: boolean };

export interface AmbientWorkflowTransition {
  readonly state: AmbientWorkflowState;
  readonly effects: readonly AmbientWorkflowEffect[];
}

export function createAmbientWorkflowState(): AmbientWorkflowState {
  return {
    phase: "idle",
    generation: 0,
    consented: false,
    audioLane: "off",
    faceLane: "off",
    setupStartedAtMs: null,
    captureStartedAtMs: null,
    terminalReason: null
  };
}

function transition(
  state: AmbientWorkflowState,
  effects: readonly AmbientWorkflowEffect[] = []
): AmbientWorkflowTransition {
  return { state: Object.freeze(state), effects: Object.freeze([...effects]) };
}

function isCurrent(state: AmbientWorkflowState, generation: number): boolean {
  return generation === state.generation;
}

function laneTerminal(lane: AmbientLaneState): boolean {
  return lane === "measurable" || lane === "not-measurable";
}

function atLeastOneLaneMeasurable(state: AmbientWorkflowState): boolean {
  return state.audioLane === "measurable" || state.faceLane === "measurable";
}

function maybeBeginCalibration(
  state: AmbientWorkflowState,
  atMs: number
): AmbientWorkflowTransition {
  if (
    state.phase !== "requesting-permission" ||
    state.audioLane === "requesting" ||
    state.faceLane === "requesting"
  ) {
    return transition(state);
  }

  if (!atLeastOneLaneMeasurable({
    ...state,
    audioLane:
      state.audioLane === "calibrating" ? "measurable" : state.audioLane,
    faceLane: state.faceLane === "calibrating" ? "measurable" : state.faceLane
  })) {
    return transition(
      {
        ...state,
        phase: "error",
        terminalReason: "no-capture-device-available"
      },
      [{ type: "dispose", generation: state.generation, discard: true }]
    );
  }

  return transition(
    { ...state, phase: "calibrating", setupStartedAtMs: atMs },
    [{ type: "begin-calibration", generation: state.generation }]
  );
}

function maybeBeginObservation(
  state: AmbientWorkflowState,
  atMs: number,
  allowCaptureWithIncompleteCalibration = false
): AmbientWorkflowTransition {
  if (
    state.phase !== "calibrating" ||
    !laneTerminal(state.audioLane) ||
    !laneTerminal(state.faceLane)
  ) {
    return transition(state);
  }
  if (
    !atLeastOneLaneMeasurable(state) &&
    !allowCaptureWithIncompleteCalibration
  ) {
    return transition(
      {
        ...state,
        phase: "error",
        terminalReason: "no-measurable-modality"
      },
      [{ type: "dispose", generation: state.generation, discard: true }]
    );
  }
  return transition(
    {
      ...state,
      phase: "observing",
      captureStartedAtMs: atMs,
      terminalReason: null
    },
    [{ type: "begin-observation", generation: state.generation }]
  );
}

function discard(
  state: AmbientWorkflowState,
  reason: string
): AmbientWorkflowTransition {
  if (state.phase === "idle" || state.phase === "report") {
    return transition(state);
  }
  const generation = state.generation + 1;
  return transition(
    {
      ...state,
      phase: "discarded",
      generation,
      consented: false,
      audioLane: "off",
      faceLane: "off",
      setupStartedAtMs: null,
      captureStartedAtMs: null,
      terminalReason: reason
    },
    [{ type: "dispose", generation, discard: true }]
  );
}

export function reduceAmbientWorkflow(
  state: AmbientWorkflowState,
  event: AmbientWorkflowEvent
): AmbientWorkflowTransition {
  if (event.type === "reset") {
    return transition({
      ...createAmbientWorkflowState(),
      generation: state.generation + 1
    });
  }
  if (event.type === "visibility-lost") {
    return discard(state, "page-hidden");
  }
  if (event.type === "discard-requested") {
    return discard(state, event.reason);
  }
  if (event.type === "consent-changed") {
    if (!event.consented && state.phase !== "idle") {
      return discard(state, "consent-withdrawn");
    }
    if (state.phase !== "idle") return transition(state);
    return transition({ ...state, consented: event.consented });
  }
  if (event.type === "start-requested") {
    if (state.phase !== "idle" || !state.consented) return transition(state);
    const generation = state.generation + 1;
    return transition(
      {
        ...state,
        phase: "requesting-permission",
        generation,
        audioLane: "requesting",
        faceLane: "requesting",
        terminalReason: null
      },
      [{ type: "request-media", generation }]
    );
  }
  if ("generation" in event && !isCurrent(state, event.generation)) {
    return transition(state);
  }
  if (event.type === "permission-resolved") {
    if (state.phase !== "requesting-permission") return transition(state);
    const next = {
      ...state,
      [event.lane === "audio" ? "audioLane" : "faceLane"]:
        event.available ? "calibrating" : "not-measurable"
    } as AmbientWorkflowState;
    return maybeBeginCalibration(next, event.atMs);
  }
  if (event.type === "calibration-resolved") {
    if (state.phase === "observing" && event.measurable) {
      return transition({
        ...state,
        [event.lane === "audio" ? "audioLane" : "faceLane"]: "measurable"
      } as AmbientWorkflowState);
    }
    if (state.phase !== "calibrating") return transition(state);
    const next = {
      ...state,
      [event.lane === "audio" ? "audioLane" : "faceLane"]:
        event.measurable ? "measurable" : "not-measurable"
    } as AmbientWorkflowState;
    return maybeBeginObservation(next, event.atMs);
  }
  if (event.type === "lane-failed") {
    if (
      state.phase !== "requesting-permission" &&
      state.phase !== "calibrating" &&
      state.phase !== "observing"
    ) {
      return transition(state);
    }
    const next = {
      ...state,
      [event.lane === "audio" ? "audioLane" : "faceLane"]:
        "not-measurable"
    } as AmbientWorkflowState;
    if (state.phase === "requesting-permission") {
      return maybeBeginCalibration(next, event.atMs);
    }
    if (state.phase === "calibrating") {
      return maybeBeginObservation(next, event.atMs);
    }
    if (!event.remainingLaneActive) {
      return transition(
        {
          ...next,
          phase: "finalizing",
          terminalReason: "all-active-lanes-failed"
        },
        [{ type: "finalize", generation: state.generation }]
      );
    }
    return transition(next);
  }
  if (event.type === "setup-timeout") {
    if (state.phase !== "calibrating") return transition(state);
    const hadCaptureCapableLane =
      state.audioLane === "calibrating" ||
      state.faceLane === "calibrating";
    const next = {
      ...state,
      audioLane:
        state.audioLane === "calibrating"
          ? "not-measurable"
          : state.audioLane,
      faceLane:
        state.faceLane === "calibrating"
          ? "not-measurable"
          : state.faceLane
    } satisfies AmbientWorkflowState;
    return maybeBeginObservation(
      next,
      event.atMs,
      hadCaptureCapableLane
    );
  }
  if (
    event.type === "finish-requested" ||
    event.type === "capture-limit-reached"
  ) {
    if (state.phase !== "observing") return transition(state);
    return transition(
      { ...state, phase: "finalizing", terminalReason: event.type },
      [{ type: "finalize", generation: state.generation }]
    );
  }
  if (event.type === "finalization-completed") {
    if (state.phase !== "finalizing") return transition(state);
    return transition({ ...state, phase: "report" });
  }
  if (event.type === "finalization-failed") {
    if (state.phase !== "finalizing") return transition(state);
    return transition({
      ...state,
      phase: "error",
      terminalReason: event.reason
    });
  }
  return transition(state);
}
