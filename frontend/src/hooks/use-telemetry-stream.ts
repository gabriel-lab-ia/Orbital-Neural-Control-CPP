"use client";

import { useEffect, useMemo, useState } from "react";

import type { StreamState, TelemetrySample } from "@/types/telemetry";

function parseSample(payload: string): TelemetrySample | null {
  try {
    const data = JSON.parse(payload) as Partial<TelemetrySample>;
    if (
      typeof data.step !== "number" ||
      typeof data.mission_time_s !== "number" ||
      !Array.isArray(data.position_m) ||
      data.position_m.length !== 3 ||
      typeof data.policy_std !== "number" ||
      typeof data.reward !== "number"
    ) {
      return null;
    }

    return {
      step: data.step,
      mission_time_s: data.mission_time_s,
      position_m: [data.position_m[0], data.position_m[1], data.position_m[2]],
      policy_std: data.policy_std,
      reward: data.reward,
    };
  } catch {
    return null;
  }
}

export function useTelemetryStream(maxSamples = 320): StreamState {
  const [connected, setConnected] = useState(false);
  const [samples, setSamples] = useState<TelemetrySample[]>([]);
  const [lastMessageAt, setLastMessageAt] = useState<number | undefined>(undefined);

  useEffect(() => {
    const wsBase = process.env.NEXT_PUBLIC_BACKEND_WS ?? "ws://localhost:8080";
    const httpBase = process.env.NEXT_PUBLIC_BACKEND_HTTP ?? "http://localhost:8080";
    const ws = new WebSocket(`${wsBase}/ws/telemetry`);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (event) => {
      const sample = parseSample(String(event.data));
      if (!sample) {
        return;
      }

      setLastMessageAt(Date.now());
      setSamples((previous) => {
        const next = [...previous, sample];
        return next.slice(Math.max(0, next.length - maxSamples));
      });
    };

    const bootstrap = async (): Promise<void> => {
      try {
        const response = await fetch(`${httpBase}/api/telemetry/snapshot`, { cache: "no-store" });
        const text = await response.text();
        const snapshot = parseSample(text);
        if (!snapshot) {
          return;
        }
        setSamples((previous) => (previous.length > 0 ? previous : [snapshot]));
      } catch {
        // keep running in websocket-only mode
      }
    };

    void bootstrap();

    return () => {
      ws.close();
    };
  }, [maxSamples]);

  return useMemo(
    () => ({
      connected,
      lastMessageAt,
      samples,
    }),
    [connected, lastMessageAt, samples],
  );
}
