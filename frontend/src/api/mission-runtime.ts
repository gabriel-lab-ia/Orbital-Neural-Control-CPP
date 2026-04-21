import { useEffect, useMemo } from "react";

import type { ReplayFrame, ReplayRunDataset } from "@/entities/replay/model/types";
import type { RunSummary } from "@/entities/run/model/types";
import { replayPayloadToDataset } from "@/entities/replay/model/adapters";
import { MISSION_REPLAY_DATASET_BY_ID, MISSION_REPLAY_DATASETS } from "@/shared/mock/mission-replay.mock";
import { DEFAULT_FRAME_RATE } from "@/shared/config/replay";
import { backendWsBaseUrl, orbitalApi } from "@/shared/api";
import type { RunDto, TelemetrySampleDto, WsMessage, WsReplayChunk, WsTelemetrySample } from "@/shared/api/generated/orbital-api";
import { useMissionStore } from "@/store/mission-store";

function toRunSummaryFromApi(run: RunDto): RunSummary {
  return {
    runId: run.run_id,
    label: run.run_id,
    mode: run.mode,
    environment: run.environment,
    backend: "libtorch_cpu",
    deterministic: true,
    totalTimesteps: 0,
    status: run.status === "completed" ? "ok" : run.status === "running" ? "running" : "warning",
    artifactStatus: run.status === "completed" ? "complete" : run.status === "running" ? "partial" : "unknown",
    startedAtIso: run.started_at,
  };
}

function toLiveFrame(run: RunSummary, sample: TelemetrySampleDto, index: number): ReplayFrame {
  return {
    frameIndex: index,
    timestampIso: sample.timestamp,
    telemetry: {
      runId: run.runId,
      environment: run.environment,
      timestep: sample.step,
      missionTimeS: sample.mission_time_s,
      reward: sample.reward,
      controlMagnitude: sample.control_magnitude,
      orbitalErrorKm: sample.orbital_error_km,
      velocityMagnitudeKmS: sample.velocity_magnitude_kmps,
      policyStd: sample.policy_std,
      backend: run.backend,
      deterministic: run.deterministic,
      positionKm: sample.position_km,
      velocityKmS: sample.velocity_kmps,
      controlVector: sample.control_vector,
      terminated: sample.terminated,
      truncated: sample.truncated,
      timestampIso: sample.timestamp,
    },
    orbit: {
      timestep: sample.step,
      missionTimeS: sample.mission_time_s,
      positionKm: sample.position_km,
      velocityKmS: sample.velocity_kmps,
      orbitalErrorKm: sample.orbital_error_km,
      controlMagnitude: sample.control_magnitude,
      reward: sample.reward,
    },
  };
}

function isTelemetryMessage(message: WsMessage): message is WsTelemetrySample {
  return message.type === "telemetry.sample";
}

function isReplayChunkMessage(message: WsMessage): message is WsReplayChunk {
  return message.type === "replay.chunk";
}

export function useMissionRuntime(maxReplaySamples = 1400): void {
  const {
    state: { mode, selectedRunId, speed, isPlaying, frameIndex, dataset, liveFrameBuffer },
    dispatch,
  } = useMissionStore();

  const frameCount = mode === "live" && liveFrameBuffer.length > 0 ? liveFrameBuffer.length : (dataset?.frames.length ?? 0);

  useEffect(() => {
    let cancelled = false;
    orbitalApi
      .listRuns({ limit: 200, offset: 0 })
      .then((result) => {
        if (cancelled) {
          return;
        }
        const runs = result.data.items.map(toRunSummaryFromApi);
        dispatch({ type: "set_runs", runs, source: "backend" });
        if (runs.length > 0 && !selectedRunId) {
          dispatch({ type: "set_selected_run", runId: runs[0].runId });
        }
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        const runs = MISSION_REPLAY_DATASETS.map((item) => item.run);
        dispatch({ type: "set_runs", runs, source: "mock" });
        if (runs.length > 0 && !selectedRunId) {
          dispatch({ type: "set_selected_run", runId: runs[0].runId });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [dispatch, selectedRunId]);

  useEffect(() => {
    if (!selectedRunId) {
      return;
    }
    let cancelled = false;
    dispatch({ type: "set_stream_status", status: "loading" });

    orbitalApi
      .getRunReplay(selectedRunId, { downsample: maxReplaySamples })
      .then((result) => {
        if (cancelled) {
          return;
        }
        const next = replayPayloadToDataset(result.data);
        dispatch({ type: "set_dataset", dataset: next });
        dispatch({ type: "set_stream_status", status: "idle" });
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        const fallback = MISSION_REPLAY_DATASET_BY_ID.get(selectedRunId) ?? MISSION_REPLAY_DATASETS[0] ?? null;
        dispatch({ type: "set_dataset", dataset: fallback });
        dispatch({ type: "set_stream_status", status: "degraded", error: "Replay API unavailable, using local sample data." });
      });

    return () => {
      cancelled = true;
    };
  }, [dispatch, maxReplaySamples, selectedRunId]);

  useEffect(() => {
    if (mode !== "live" || !selectedRunId) {
      dispatch({ type: "clear_live_frames" });
      return;
    }

    const selectedRun =
      (dataset?.run.runId === selectedRunId ? dataset.run : null) ??
      MISSION_REPLAY_DATASET_BY_ID.get(selectedRunId)?.run ??
      MISSION_REPLAY_DATASETS[0]?.run;

    if (!selectedRun) {
      return;
    }

    const wsUrl = `${backendWsBaseUrl}/ws/runs/${encodeURIComponent(selectedRunId)}/stream`;
    const seen = new Set<string>();
    let reconnectTimer: number | null = null;
    let retries = 0;
    let socket: WebSocket | null = null;
    let active = true;

    const connectAgain = (): void => {
      if (reconnectTimer !== null || !active) {
        return;
      }
      const delay = Math.min(4000, 500 + retries * 350);
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        retries += 1;
        dispatch({ type: "set_stream_status", status: "degraded", error: `Stream reconnect attempt ${retries}` });
        connect();
      }, delay);
    };

    const appendSamples = (samples: TelemetrySampleDto[]): void => {
      const frames = samples
        .filter((sample) => {
          const id = `${sample.step}-${sample.timestamp}`;
          if (seen.has(id)) {
            return false;
          }
          seen.add(id);
          return true;
        })
        .map((sample, index) => toLiveFrame(selectedRun, sample, index));

      dispatch({ type: "append_live_frames", frames, maxFrames: 2500 });
    };

    const connect = (): void => {
      if (!active) {
        return;
      }

      dispatch({ type: "set_stream_status", status: "loading", error: null });
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        retries = 0;
        dispatch({ type: "set_stream_status", status: "live", error: null });
      };

      socket.onerror = () => {
        dispatch({ type: "set_stream_status", status: "degraded", error: "WebSocket stream error" });
      };

      socket.onclose = () => {
        dispatch({ type: "set_stream_status", status: "disconnected", error: "Stream disconnected" });
        connectAgain();
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(String(event.data)) as WsMessage;
          if (isTelemetryMessage(message)) {
            appendSamples([message.payload]);
            return;
          }
          if (isReplayChunkMessage(message)) {
            appendSamples(message.payload.samples);
          }
        } catch {
          dispatch({ type: "set_stream_status", status: "degraded", error: "Invalid stream payload" });
        }
      };
    };

    connect();

    return () => {
      active = false;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      socket?.close();
    };
  }, [dataset?.run, dispatch, mode, selectedRunId]);

  useEffect(() => {
    if (!isPlaying || frameCount <= 1) {
      return;
    }
    const intervalMs = Math.max(12, Math.round(1000 / (DEFAULT_FRAME_RATE * speed)));
    const timer = window.setInterval(() => {
      dispatch({ type: "set_frame", frameIndex: frameIndex >= frameCount - 1 ? 0 : frameIndex + 1 });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [dispatch, frameCount, frameIndex, isPlaying, speed]);
}

export function useMissionDataset(): ReplayRunDataset | null {
  const {
    state: { dataset, liveFrameBuffer, mode, selectedRunId },
  } = useMissionStore();

  return useMemo(() => {
    if (!dataset) {
      return null;
    }
    if (mode !== "live" || liveFrameBuffer.length === 0) {
      return dataset;
    }
    const run = {
      ...dataset.run,
      runId: selectedRunId ?? dataset.run.runId,
      label: `${selectedRunId ?? dataset.run.runId} (live)`,
      totalTimesteps: liveFrameBuffer.length,
    };
    return {
      ...dataset,
      run,
      frames: liveFrameBuffer.map((frame, index) => ({ ...frame, frameIndex: index })),
      orbitPath: liveFrameBuffer.map((frame) => frame.orbit),
      benchmark: {
        ...dataset.benchmark,
        totalTimesteps: liveFrameBuffer.length,
      },
    };
  }, [dataset, liveFrameBuffer, mode, selectedRunId]);
}
