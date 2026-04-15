"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TelemetrySample } from "@/types/telemetry";

interface TelemetryChartsProps {
  samples: TelemetrySample[];
}

function missionRadius(position: [number, number, number]): number {
  const [x, y, z] = position;
  return Math.sqrt(x * x + y * y + z * z);
}

export function TelemetryCharts({ samples }: TelemetryChartsProps): JSX.Element {
  const chartData = samples.map((sample) => ({
    t: Number(sample.mission_time_s.toFixed(1)),
    reward: sample.reward,
    policy_std: sample.policy_std,
    orbit_residual_km: (missionRadius(sample.position_m) - 6_800_000) / 1000,
  }));

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle>Reward Trajectory</CardTitle>
        </CardHeader>
        <CardContent className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f3246" />
              <XAxis dataKey="t" stroke="#93a8bf" />
              <YAxis stroke="#93a8bf" />
              <Tooltip contentStyle={{ backgroundColor: "#0f1f31", border: "1px solid #1f3246" }} />
              <Line type="monotone" dataKey="reward" stroke="#67e8f9" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Policy Std Envelope</CardTitle>
        </CardHeader>
        <CardContent className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f3246" />
              <XAxis dataKey="t" stroke="#93a8bf" />
              <YAxis stroke="#93a8bf" />
              <Tooltip contentStyle={{ backgroundColor: "#0f1f31", border: "1px solid #1f3246" }} />
              <Line type="monotone" dataKey="policy_std" stroke="#a5f3fc" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Orbital Residual (km)</CardTitle>
        </CardHeader>
        <CardContent className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f3246" />
              <XAxis dataKey="t" stroke="#93a8bf" />
              <YAxis stroke="#93a8bf" />
              <Tooltip contentStyle={{ backgroundColor: "#0f1f31", border: "1px solid #1f3246" }} />
              <Line type="monotone" dataKey="orbit_residual_km" stroke="#22d3ee" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
