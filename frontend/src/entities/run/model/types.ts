export interface RunSummary {
  runId: string;
  label: string;
  mode: string;
  environment: string;
  backend: string;
  torchDevice: string;
  computeBackendRequested: string;
  computeBackendResolved: string;
  cudaFallbackUsed: boolean;
  avgInferenceLatencyMs: number | null;
  p50InferenceLatencyMs: number | null;
  p95InferenceLatencyMs: number | null;
  deterministic: boolean;
  totalTimesteps: number;
  status: "ok" | "warning" | "running" | "failed";
  artifactStatus: "complete" | "partial" | "unknown";
  startedAtIso: string;
}

export interface BenchmarkSummary {
  totalTimesteps: number;
  meanReward: number;
  deterministic: boolean;
  backend: string;
  artifactStatus: "complete" | "partial" | "unknown";
}

export interface MissionEventSummary {
  id: number;
  eventType: string;
  level: string;
  message: string;
  timestampIso: string;
}
