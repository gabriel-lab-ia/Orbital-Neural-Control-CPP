export interface TelemetrySample {
  step: number;
  mission_time_s: number;
  position_m: [number, number, number];
  policy_std: number;
  reward: number;
}

export interface StreamState {
  connected: boolean;
  lastMessageAt?: number;
  samples: TelemetrySample[];
}
