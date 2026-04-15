"use client";

import { Line, OrbitControls, Stars } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import type { TelemetrySample } from "@/types/telemetry";

interface OrbitSceneProps {
  samples: TelemetrySample[];
}

function scalePosition(sample: TelemetrySample): [number, number, number] {
  const orbitalScale = 1.0 / 1_700_000.0;
  return [
    sample.position_m[0] * orbitalScale,
    sample.position_m[2] * orbitalScale,
    sample.position_m[1] * orbitalScale,
  ];
}

export function OrbitScene({ samples }: OrbitSceneProps): JSX.Element {
  const trail = useMemo(() => samples.map(scalePosition), [samples]);
  const last = trail[trail.length - 1] ?? [4.1, 0.2, 0.0];

  return (
    <div className="h-[360px] w-full animate-fade-up rounded-xl border border-border/70 bg-gradient-to-b from-[#09131f] via-[#0a1a2f] to-[#080d16]">
      <Canvas camera={{ position: [0, 3.2, 6.6], fov: 44 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 6, 4]} intensity={1.1} color="#8cd2ff" />
        <pointLight position={[-4, -2, -3]} intensity={0.5} color="#5eead4" />

        <mesh>
          <sphereGeometry args={[1.65, 64, 64]} />
          <meshStandardMaterial color="#0f3a6a" roughness={0.66} metalness={0.12} />
        </mesh>

        <mesh position={last}>
          <sphereGeometry args={[0.07, 20, 20]} />
          <meshStandardMaterial emissive="#4cf8ff" color="#7dd3fc" emissiveIntensity={1.8} />
        </mesh>

        {trail.length > 1 ? <Line points={trail} color="#67e8f9" lineWidth={1.3} /> : null}

        <Stars radius={95} depth={45} count={1600} factor={5} fade speed={1.1} />
        <OrbitControls enablePan={false} minDistance={3.5} maxDistance={12} autoRotate autoRotateSpeed={0.2} />
      </Canvas>
    </div>
  );
}
