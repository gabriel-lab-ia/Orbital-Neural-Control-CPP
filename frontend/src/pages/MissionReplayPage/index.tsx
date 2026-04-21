"use client";

import { useEffect, useMemo } from "react";

import type { RunSummary } from "@/entities/run/model/types";
import { useMissionDataset, useMissionRuntime } from "@/api/mission-runtime";
import { MISSION_REPLAY_DATASETS } from "@/shared/mock/mission-replay.mock";
import { MissionTopbar } from "@/widgets/MissionTopbar/ui/mission-topbar";
import { OrbitalCanvasWidget } from "@/widgets/OrbitalCanvas/ui/orbital-canvas-widget";
import { ReplayTimelineWidget } from "@/widgets/ReplayTimeline/ui/replay-timeline-widget";
import { TelemetrySidebarWidget } from "@/widgets/TelemetrySidebar/ui/telemetry-sidebar-widget";
import { useMissionStore } from "@/store/mission-store";

export function MissionReplayPage(): JSX.Element {
  const fallbackDataset = MISSION_REPLAY_DATASETS[0];
  if (!fallbackDataset) {
    throw new Error("Mission replay datasets are not configured.");
  }

  useMissionRuntime(1400);
  const { state, dispatch } = useMissionStore();
  const dataset = useMissionDataset() ?? fallbackDataset;

  const frameCount = Math.max(1, dataset.frames.length);
  const frameIndex = Math.min(state.frameIndex, frameCount - 1);
  const currentFrame = dataset.frames[frameIndex] ?? dataset.frames[0];

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.code === "Space") {
        event.preventDefault();
        dispatch({ type: "toggle_playing" });
        return;
      }
      if (event.code === "ArrowRight") {
        dispatch({ type: "step_frame", delta: 1 });
        return;
      }
      if (event.code === "ArrowLeft") {
        dispatch({ type: "step_frame", delta: -1 });
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [dispatch]);

  const runsForTopbar = useMemo<RunSummary[]>(() => {
    if (state.runs.length === 0) {
      return MISSION_REPLAY_DATASETS.map((item) => item.run);
    }
    return state.runs.map((run) => ({
      ...run,
      totalTimesteps: state.selectedRunId === run.runId ? dataset.frames.length : run.totalTimesteps,
    }));
  }, [dataset.frames.length, state.runs, state.selectedRunId]);

  const selectedRun = runsForTopbar.find((run) => run.runId === state.selectedRunId) ?? dataset.run;

  if (!currentFrame) {
    return (
      <main className="mission-shell">
        <div className="mission-shell__gradient" />
        <div className="mission-shell__content">
          <p className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-200">
            Replay data unavailable for the selected run.
          </p>
        </div>
      </main>
    );
  }

  const dataSource = state.source === "backend" ? "backend api" : "mock fallback";

  return (
    <main className="mission-shell">
      <div className="mission-shell__gradient" />

      <div className="mission-shell__content">
        <div className="hud-panel glow-border mb-3 flex flex-wrap items-center gap-2 rounded-xl border border-cyan-200/15 px-3 py-2 text-[11px] uppercase tracking-[0.16em] text-slate-300">
          <span>data source: {dataSource}</span>
          <span className="text-slate-500">|</span>
          <span>mission mode:</span>

          <button
            type="button"
            onClick={() => dispatch({ type: "set_mode", mode: "replay" })}
            className={`rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ${
              state.mode === "replay"
                ? "border border-cyan-300/45 bg-cyan-300/14 text-cyan-100"
                : "border border-slate-600/80 bg-black/50 text-slate-300"
            }`}
          >
            replay
          </button>

          <button
            type="button"
            onClick={() => dispatch({ type: "set_mode", mode: "live" })}
            className={`rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ${
              state.mode === "live"
                ? "border border-cyan-300/45 bg-cyan-300/14 text-cyan-100"
                : "border border-slate-600/80 bg-black/50 text-slate-300"
            }`}
          >
            live
          </button>

          <span className="text-slate-500">|</span>
          <span className={state.streamStatus === "live" ? "text-emerald-200" : "text-amber-200"}>
            stream: {state.streamStatus}
          </span>
          <span className="text-slate-500">|</span>
          <span>playback: Space play/pause, ←/→ step</span>
          {state.streamError ? <span className="text-amber-300">{state.streamError}</span> : null}
        </div>

        <MissionTopbar
          runs={runsForTopbar}
          selectedRun={selectedRun}
          onSelectRun={(runId) => dispatch({ type: "set_selected_run", runId })}
        />

        <section className="mt-5 grid gap-4 2xl:grid-cols-[minmax(0,1fr)_380px]">
          <OrbitalCanvasWidget
            frame={currentFrame}
            orbitPath={dataset.orbitPath}
            cameraMode={state.cameraMode}
            onCameraModeChange={(cameraMode) => dispatch({ type: "set_camera_mode", cameraMode })}
          />
          <TelemetrySidebarWidget frame={currentFrame} run={selectedRun} benchmark={dataset.benchmark} events={dataset.events} />
        </section>

        <section className="mt-4">
          <ReplayTimelineWidget
            frame={currentFrame}
            frameIndex={frameIndex}
            frameCount={frameCount}
            isPlaying={state.isPlaying}
            speed={state.speed}
            events={dataset.events}
            onTogglePlay={() => dispatch({ type: "toggle_playing" })}
            onStepBackward={() => dispatch({ type: "step_frame", delta: -1 })}
            onStepForward={() => dispatch({ type: "step_frame", delta: 1 })}
            onReset={() => dispatch({ type: "reset_replay" })}
            onFrameChange={(nextFrame) => dispatch({ type: "set_frame", frameIndex: nextFrame })}
            onSpeedChange={(nextSpeed) => dispatch({ type: "set_speed", speed: nextSpeed })}
          />
        </section>
      </div>
    </main>
  );
}
