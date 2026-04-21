import type { Dispatch, ReactNode } from "react";
import { createContext, useContext, useMemo, useReducer } from "react";

import type { ReplayFrame, ReplayRunDataset } from "@/entities/replay/model/types";
import type { RunSummary } from "@/entities/run/model/types";

export type MissionMode = "replay" | "live";
export type ConnectionStatus = "idle" | "loading" | "live" | "degraded" | "disconnected";
export type CameraMode = "earth_lock" | "spacecraft_follow" | "mission_replay" | "free_inspect";

interface MissionUiState {
  mode: MissionMode;
  cameraMode: CameraMode;
  isPlaying: boolean;
  speed: number;
  frameIndex: number;
}

interface MissionDataState {
  runs: RunSummary[];
  selectedRunId: string | null;
  dataset: ReplayRunDataset | null;
  liveFrameBuffer: ReplayFrame[];
  streamStatus: ConnectionStatus;
  streamError: string | null;
  source: "backend" | "mock";
}

export interface MissionState extends MissionUiState, MissionDataState {}

const initialState: MissionState = {
  mode: "replay",
  cameraMode: "mission_replay",
  isPlaying: false,
  speed: 1,
  frameIndex: 0,
  runs: [],
  selectedRunId: null,
  dataset: null,
  liveFrameBuffer: [],
  streamStatus: "idle",
  streamError: null,
  source: "mock",
};

type MissionAction =
  | { type: "set_runs"; runs: RunSummary[]; source: "backend" | "mock" }
  | { type: "set_selected_run"; runId: string }
  | { type: "set_dataset"; dataset: ReplayRunDataset | null }
  | { type: "set_mode"; mode: MissionMode }
  | { type: "set_camera_mode"; cameraMode: CameraMode }
  | { type: "set_playing"; value: boolean }
  | { type: "toggle_playing" }
  | { type: "set_speed"; speed: number }
  | { type: "set_frame"; frameIndex: number }
  | { type: "step_frame"; delta: number }
  | { type: "reset_replay" }
  | { type: "set_stream_status"; status: ConnectionStatus; error?: string | null }
  | { type: "append_live_frames"; frames: ReplayFrame[]; maxFrames: number }
  | { type: "clear_live_frames" };

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function reducer(state: MissionState, action: MissionAction): MissionState {
  switch (action.type) {
    case "set_runs":
      return { ...state, runs: action.runs, source: action.source };
    case "set_selected_run":
      return {
        ...state,
        selectedRunId: action.runId,
        frameIndex: 0,
        isPlaying: false,
        liveFrameBuffer: [],
        streamError: null,
      };
    case "set_dataset":
      return { ...state, dataset: action.dataset, frameIndex: 0 };
    case "set_mode":
      return {
        ...state,
        mode: action.mode,
        frameIndex: 0,
        isPlaying: action.mode === "live",
        liveFrameBuffer: action.mode === "live" ? state.liveFrameBuffer : [],
      };
    case "set_camera_mode":
      return { ...state, cameraMode: action.cameraMode };
    case "set_playing":
      return { ...state, isPlaying: action.value };
    case "toggle_playing":
      return { ...state, isPlaying: !state.isPlaying };
    case "set_speed":
      return { ...state, speed: action.speed > 0 ? action.speed : state.speed };
    case "set_frame": {
      const frameCount = Math.max(
        1,
        state.mode === "live" && state.liveFrameBuffer.length > 0
          ? state.liveFrameBuffer.length
          : (state.dataset?.frames.length ?? 1),
      );
      return { ...state, frameIndex: clamp(action.frameIndex, 0, frameCount - 1) };
    }
    case "step_frame": {
      const frameCount = Math.max(
        1,
        state.mode === "live" && state.liveFrameBuffer.length > 0
          ? state.liveFrameBuffer.length
          : (state.dataset?.frames.length ?? 1),
      );
      const next = clamp(state.frameIndex + action.delta, 0, frameCount - 1);
      return { ...state, frameIndex: next };
    }
    case "reset_replay":
      return { ...state, frameIndex: 0, isPlaying: false };
    case "set_stream_status":
      return {
        ...state,
        streamStatus: action.status,
        streamError: action.error ?? null,
      };
    case "append_live_frames": {
      if (action.frames.length === 0) {
        return state;
      }
      const merged = [...state.liveFrameBuffer, ...action.frames];
      const start = Math.max(0, merged.length - action.maxFrames);
      const nextBuffer = merged.slice(start);
      const nextFrameIndex = state.isPlaying ? Math.max(0, nextBuffer.length - 1) : state.frameIndex;
      return {
        ...state,
        liveFrameBuffer: nextBuffer,
        frameIndex: clamp(nextFrameIndex, 0, Math.max(0, nextBuffer.length - 1)),
      };
    }
    case "clear_live_frames":
      return { ...state, liveFrameBuffer: [], frameIndex: 0 };
    default:
      return state;
  }
}

interface MissionStoreValue {
  state: MissionState;
  dispatch: Dispatch<MissionAction>;
}

const MissionStoreContext = createContext<MissionStoreValue | null>(null);

export function MissionStoreProvider({ children }: { children: ReactNode }): JSX.Element {
  const [state, dispatch] = useReducer(reducer, initialState);
  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <MissionStoreContext.Provider value={value}>{children}</MissionStoreContext.Provider>;
}

export function useMissionStore(): MissionStoreValue {
  const context = useContext(MissionStoreContext);
  if (!context) {
    throw new Error("useMissionStore must be used inside MissionStoreProvider");
  }
  return context;
}
