"use client";

import { Activity, Database, Radar, Satellite } from "lucide-react";

import { OrbitScene } from "@/components/mission/orbit-scene";
import { TelemetryCharts } from "@/components/mission/telemetry-charts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useTelemetryStream } from "@/hooks/use-telemetry-stream";

function formatTimestamp(timestamp?: number): string {
  if (!timestamp) {
    return "waiting";
  }
  return new Date(timestamp).toLocaleTimeString();
}

export function MissionDashboard(): JSX.Element {
  const stream = useTelemetryStream();
  const latest = stream.samples[stream.samples.length - 1];

  return (
    <main className="min-h-screen bg-background bg-grid [background-size:20px_20px] text-foreground">
      <section className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
        <header className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Badge>Orbital Simulation</Badge>
            <Badge>PPO Telemetry</Badge>
            <Badge>WebSocket Stream</Badge>
            <Badge variant="neutral">CPU-First Baseline</Badge>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Orbital Mission Control Dashboard</h1>
          <p className="max-w-3xl text-sm text-muted-foreground sm:text-base">
            Live telemetry stream for orbit-state residuals, policy uncertainty, and reward evolution.
            This UI is designed for real-time mission playback and reproducible evaluation diagnostics.
          </p>
        </header>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Activity className="h-4 w-4 text-cyan-300" /> Stream Status</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg font-semibold text-cyan-200">{stream.connected ? "Connected" : "Disconnected"}</p>
              <p className="text-xs text-muted-foreground">last frame: {formatTimestamp(stream.lastMessageAt)}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Radar className="h-4 w-4 text-cyan-300" /> Policy Std</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg font-semibold text-cyan-200">{latest ? latest.policy_std.toFixed(4) : "-"}</p>
              <p className="text-xs text-muted-foreground">stochastic envelope</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Satellite className="h-4 w-4 text-cyan-300" /> Reward</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg font-semibold text-cyan-200">{latest ? latest.reward.toFixed(4) : "-"}</p>
              <p className="text-xs text-muted-foreground">latest control cycle</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Database className="h-4 w-4 text-cyan-300" /> Samples</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg font-semibold text-cyan-200">{stream.samples.length}</p>
              <p className="text-xs text-muted-foreground">rolling mission window</p>
            </CardContent>
          </Card>
        </div>

        <OrbitScene samples={stream.samples} />
        <TelemetryCharts samples={stream.samples} />
      </section>
    </main>
  );
}
