import { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere, Html } from "@react-three/drei";
import * as THREE from "three";
import { type FrequencyBands, ELECTRODE_POSITIONS } from "@/lib/eeg-types";
import { useStore } from "@/store/useStore";

interface Brain3DViewerProps {
  frequencyData: FrequencyBands;
  stressLevel: number;
  isProcessing: boolean;
  activeElectrode?: string | null;
  onElectrodeClick?: (electrodeId: string) => void;
}

function getStressColor(stressLevel: number): THREE.Color {
  if (stressLevel < 20) return new THREE.Color(0x4caf50);
  if (stressLevel < 40) return new THREE.Color(0x8bc34a);
  if (stressLevel < 60) return new THREE.Color(0xffeb3b);
  if (stressLevel < 80) return new THREE.Color(0xff9800);
  return new THREE.Color(0xf44336);
}

function getBandColor(band: keyof FrequencyBands): THREE.Color {
  const colors: Record<keyof FrequencyBands, number> = {
    delta: 0x3b82f6,
    theta: 0x06b6d4,
    alpha: 0x10b981,
    beta: 0xf59e0b,
    gamma: 0xa855f7,
  };
  return new THREE.Color(colors[band] ?? 0x10b981);
}

function getDominantBand(data: FrequencyBands): keyof FrequencyBands {
  const entries = Object.entries(data) as [keyof FrequencyBands, number][];
  if (!entries.length || entries.every(([, v]) => v === 0)) return "alpha";
  return entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
}

function BrainMesh({
  frequencyData,
  stressLevel,
  isProcessing,
}: {
  frequencyData: FrequencyBands;
  stressLevel: number;
  isProcessing: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const color = useMemo(() => getStressColor(stressLevel), [stressLevel]);
  const dominantBandColor = useMemo(() => getBandColor(getDominantBand(frequencyData)), [frequencyData]);

  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.1;
      if (isProcessing) {
        const pulse = Math.sin(state.clock.elapsedTime * 3) * 0.05;
        meshRef.current.scale.setScalar(1 + pulse);
      } else {
        meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
      }
    }
    if (glowRef.current) {
      glowRef.current.rotation.y += delta * 0.05;
      const glowPulse = Math.sin(state.clock.elapsedTime * 2) * 0.1 + 1.15;
      glowRef.current.scale.setScalar(glowPulse);
    }
  });

  return (
    <group>
      <mesh ref={glowRef}>
        <sphereGeometry args={[1.2, 32, 32]} />
        <meshBasicMaterial color={dominantBandColor} transparent opacity={0.1} side={THREE.BackSide} />
      </mesh>

      <mesh ref={meshRef} onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshPhongMaterial
          color={color}
          emissive={dominantBandColor}
          emissiveIntensity={isProcessing ? 0.4 : 0.2}
          shininess={50}
          transparent
          opacity={0.85}
        />
      </mesh>

      <mesh>
        <sphereGeometry args={[0.6, 32, 32]} />
        <meshPhongMaterial
          color={dominantBandColor}
          emissive={dominantBandColor}
          emissiveIntensity={0.5}
          transparent
          opacity={0.4}
        />
      </mesh>

      {Array.from({ length: 8 }).map((_, i) => (
        <mesh key={i} rotation={[0, (i * Math.PI) / 4, Math.PI / 6]}>
          <torusGeometry args={[0.95, 0.02, 8, 32, Math.PI]} />
          <meshBasicMaterial color={0x1a237e} transparent opacity={0.5} />
        </mesh>
      ))}

      {hovered && (
        <Html distanceFactor={3}>
          <div className="rounded-lg bg-slate-900/90 px-3 py-2 text-xs backdrop-blur border border-slate-800 shadow-xl">
            <p className="font-semibold text-white">Activité Cérébrale</p>
            <p className="text-slate-400">Stress: {Math.round(stressLevel)}%</p>
            <p className="text-slate-400">Bande dominante: {getDominantBand(frequencyData).toUpperCase()}</p>
          </div>
        </Html>
      )}
    </group>
  );
}

function ElectrodeNode({
  position,
  electrode,
  value,
  isActive,
  onClick,
}: {
  position: [number, number, number];
  electrode: (typeof ELECTRODE_POSITIONS)[0];
  value: number;
  isActive: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const color = useMemo(() => {
    const intensity = Math.min(value / 100, 1);
    return new THREE.Color().lerpColors(
      new THREE.Color(0x4caf50),
      new THREE.Color(0xf44336),
      intensity
    );
  }, [value]);

  useFrame((state) => {
    if (meshRef.current) {
      const pulse = isActive ? Math.sin(state.clock.elapsedTime * 5) * 0.3 + 1 : 1;
      meshRef.current.scale.setScalar(pulse);
    }
  });

  return (
    <group position={position}>
      <Sphere
        ref={meshRef}
        args={[0.08, 16, 16]}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshPhongMaterial color={color} emissive={color} emissiveIntensity={isActive ? 0.8 : 0.3} />
      </Sphere>

      {(hovered || isActive) && (
        <Html distanceFactor={3}>
          <div className="whitespace-nowrap rounded bg-slate-900/95 px-2 py-1 text-xs backdrop-blur border border-slate-800 shadow-lg">
            <p className="font-mono font-semibold text-white">{electrode.name}</p>
            <p className="text-slate-400">{electrode.region}</p>
            <p className="text-accent-teal">{value.toFixed(1)}%</p>
          </div>
        </Html>
      )}
    </group>
  );
}

function ElectrodeNetwork({
  activeElectrode,
  onElectrodeClick,
  stressLevel,
}: {
  activeElectrode?: string | null;
  onElectrodeClick?: (id: string) => void;
  stressLevel: number;
}) {
  return (
    <group>
      {ELECTRODE_POSITIONS.map((electrode) => {
        const value = (Math.abs(electrode.x + electrode.y) * 30 + stressLevel * 0.5) % 100;
        return (
          <ElectrodeNode
            key={electrode.id}
            position={[electrode.x * 1.1, electrode.z * 1.1, electrode.y * 1.1]}
            electrode={electrode}
            value={value}
            isActive={activeElectrode === electrode.id}
            onClick={() => onElectrodeClick?.(electrode.id)}
          />
        );
      })}
    </group>
  );
}

function NeuralParticles({ stressLevel }: { stressLevel: number }) {
  const particlesRef = useRef<THREE.Points>(null);
  const particleCount = 200;

  const [positions, velocities] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const vel = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const r = 0.8 + Math.random() * 0.4;
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
      vel[i * 3] = (Math.random() - 0.5) * 0.02;
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.02;
    }
    return [pos, vel];
  }, []);

  useFrame((_, delta) => {
    if (!particlesRef.current) return;
    const posAttr = particlesRef.current.geometry.attributes.position;
    const speed = 0.5 + (stressLevel / 100) * 1.5;
    for (let i = 0; i < particleCount; i++) {
      const idx = i * 3;
      posAttr.array[idx] += velocities[idx] * speed * delta * 30;
      posAttr.array[idx + 1] += velocities[idx + 1] * speed * delta * 30;
      posAttr.array[idx + 2] += velocities[idx + 2] * speed * delta * 30;
      const dist = Math.sqrt(
        (posAttr.array[idx] ** 2) + (posAttr.array[idx + 1] ** 2) + (posAttr.array[idx + 2] ** 2)
      );
      if (dist > 1.5 || dist < 0.5) {
        velocities[idx] *= -1;
        velocities[idx + 1] *= -1;
        velocities[idx + 2] *= -1;
      }
    }
    posAttr.needsUpdate = true;
  });

  const color = useMemo(() => getStressColor(stressLevel), [stressLevel]);

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={particleCount} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial size={0.03} color={color} transparent opacity={0.6} sizeAttenuation />
    </points>
  );
}

function Scene({ frequencyData, stressLevel, isProcessing, activeElectrode, onElectrodeClick }: Brain3DViewerProps) {
  const normalizedStress = stressLevel / 100;
  const dynamicColor = useMemo(() => {
    const cool = new THREE.Color("#0d9488");
    const hot = new THREE.Color("#ef4444");
    return cool.clone().lerp(hot, normalizedStress);
  }, [normalizedStress]);

  return (
    <>
      <ambientLight intensity={0.2 + normalizedStress * 0.4} color={dynamicColor} />
      <pointLight position={[10, 10, 10]} intensity={0.5 + normalizedStress * 1.5} color={dynamicColor} />
      <pointLight position={[-10, -10, -10]} intensity={0.2} color="#4fc3f7" />
      <spotLight position={[0, 5, 0]} angle={0.3} penumbra={1} intensity={0.5 + normalizedStress} color={dynamicColor} />
      <BrainMesh frequencyData={frequencyData} stressLevel={stressLevel} isProcessing={isProcessing} />
      <ElectrodeNetwork activeElectrode={activeElectrode} onElectrodeClick={onElectrodeClick} stressLevel={stressLevel} />
      <NeuralParticles stressLevel={stressLevel} />
      <OrbitControls enablePan={false} minDistance={2} maxDistance={6} autoRotate={!isProcessing} autoRotateSpeed={0.5} />
    </>
  );
}

export function Brain3DViewer({ frequencyData, stressLevel, isProcessing, activeElectrode, onElectrodeClick }: Brain3DViewerProps) {
  return (
    <div className="relative h-full w-full min-h-[500px] rounded-xl overflow-hidden bg-gradient-to-b from-slate-900 to-background border border-slate-800">
      <Canvas camera={{ position: [0, 0, 3.5], fov: 50 }} gl={{ antialias: true, alpha: true }}>
        <Scene
          frequencyData={frequencyData}
          stressLevel={stressLevel}
          isProcessing={isProcessing}
          activeElectrode={activeElectrode}
          onElectrodeClick={onElectrodeClick}
        />
      </Canvas>

      <div className="absolute bottom-4 left-4 flex flex-wrap gap-2 text-[10px]">
        {[
          { label: "Delta", color: "bg-blue-500" },
          { label: "Theta", color: "bg-cyan-500" },
          { label: "Alpha", color: "bg-green-500" },
          { label: "Beta", color: "bg-orange-500" },
          { label: "Gamma", color: "bg-purple-500" },
        ].map(({ label, color }) => (
          <div key={label} className="flex items-center gap-1.5 rounded-full bg-slate-900/80 px-2 py-1 backdrop-blur border border-slate-800">
            <div className={`h-1.5 w-1.5 rounded-full ${color}`} />
            <span className="text-slate-300">{label}</span>
          </div>
        ))}
      </div>

      {isProcessing && (
        <div className="absolute right-4 top-4 flex items-center gap-2 rounded-full bg-accent-violet/20 px-3 py-1.5 text-xs backdrop-blur border border-accent-violet/30">
          <div className="h-2 w-2 animate-pulse rounded-full bg-accent-violet" />
          <span className="text-accent-violet font-medium">Traitement...</span>
        </div>
      )}
    </div>
  );
}

// ── Main Brain3D page component ──────────────────────────────────────────────
const Brain3D: React.FC = () => {
  const { prediction, isProcessing } = useStore();
  const [activeElectrode, setActiveElectrode] = useState<string | null>(null);

  const stressLevel = (prediction?.probabilities?.stress ?? 0) * 100;

  /**
   * bandPowers from backend: Array<{band_name: number}>
   * e.g. [{"delta": 1.2}, {"theta": 0.8}, ...]
   * Convert to FrequencyBands object for Three.js
   */
  const frequencyData = useMemo<FrequencyBands>(() => {
    const defaultData: FrequencyBands = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 };
    if (!prediction?.bandPowers || !Array.isArray(prediction.bandPowers)) return defaultData;

    return prediction.bandPowers.reduce<FrequencyBands>((acc, item) => {
      if (typeof item !== "object" || item === null) return acc;
      const entries = Object.entries(item);
      if (entries.length === 0) return acc;
      const [band, value] = entries[0];
      if (band in acc && typeof value === "number") {
        acc[band as keyof FrequencyBands] = value;
      }
      return acc;
    }, { ...defaultData });
  }, [prediction]);

  const selectedElectrodeData = useMemo(() => {
    if (!activeElectrode) return null;
    const pos = ELECTRODE_POSITIONS.find(e => e.id === activeElectrode);
    if (!pos) return null;
    const value = (Math.abs(pos.x + pos.y) * 30 + stressLevel * 0.5) % 100;
    return { ...pos, value };
  }, [activeElectrode, stressLevel]);

  // Band power display values for the sidebar
  const bandDisplayValues = useMemo(() => {
    return Object.entries(frequencyData).map(([band, value]) => ({
      band,
      value: value as number,
      displayPct: Math.min(Math.max(((value as number) / 10) * 100, 0), 100),
    }));
  }, [frequencyData]);

  return (
    <div className="space-y-8 h-full flex flex-col">
      <header>
        <h2 className="text-3xl font-bold tracking-tight mb-2">Carte d'Activité Neurale 3D</h2>
        <p className="text-slate-400">Cartographie spatiale des patrons neuronaux liés au stress.</p>
      </header>

      {!prediction && (
        <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl text-amber-400 text-sm">
          ⚠ Aucune prédiction disponible. Lancez d'abord une classification dans l'onglet "Classification".
        </div>
      )}

      <div className="flex gap-6 flex-1 min-h-[500px]">
        {/* 3D Canvas */}
        <div className="flex-1 relative">
          <Brain3DViewer
            frequencyData={frequencyData}
            stressLevel={stressLevel}
            isProcessing={isProcessing}
            activeElectrode={activeElectrode}
            onElectrodeClick={setActiveElectrode}
          />

          {selectedElectrodeData && (
            <div className="absolute top-6 left-6 w-64 bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-2xl">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <div className="text-accent-teal font-mono text-xs font-bold uppercase tracking-widest mb-1">Électrode</div>
                  <h3 className="text-2xl font-black text-white leading-none">{selectedElectrodeData.name}</h3>
                </div>
                <button
                  onClick={() => setActiveElectrode(null)}
                  className="p-1 hover:bg-slate-800 rounded-lg transition-colors text-slate-500 hover:text-white"
                >
                  ✕
                </button>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between py-2 border-b border-slate-800/50">
                  <span className="text-slate-500 text-xs uppercase font-bold">Région</span>
                  <span className="text-white text-sm">{selectedElectrodeData.region}</span>
                </div>
                <div className="flex justify-between py-2 border-b border-slate-800/50">
                  <span className="text-slate-500 text-xs uppercase font-bold">Activité</span>
                  <span className="text-accent-teal text-sm font-mono font-bold">{selectedElectrodeData.value.toFixed(1)}%</span>
                </div>
                <div>
                  <div className="text-slate-500 text-[10px] uppercase font-bold tracking-widest mb-2">Force du signal</div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent-teal transition-all duration-500"
                      style={{ width: `${Math.round(selectedElectrodeData.value)}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Band power sidebar */}
        <div className="w-52 space-y-4">
          <div className="card p-4">
            <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Niveau de Stress</h4>
            <div className="text-3xl font-black font-mono" style={{
              color: stressLevel > 60 ? "#ef4444" : stressLevel > 40 ? "#f59e0b" : "#10b981"
            }}>
              {Math.round(stressLevel)}%
            </div>
            <div className="mt-2 h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full transition-all duration-1000"
                style={{
                  width: `${Math.round(stressLevel)}%`,
                  backgroundColor: stressLevel > 60 ? "#ef4444" : stressLevel > 40 ? "#f59e0b" : "#10b981",
                }}
              />
            </div>
          </div>

          <div className="card p-4">
            <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Bandes EEG</h4>
            <div className="space-y-3">
              {bandDisplayValues.map(({ band, value, displayPct }) => {
                const hexColor = {
                  delta: "#3b82f6",
                  theta: "#06b6d4",
                  alpha: "#10b981",
                  beta: "#f59e0b",
                  gamma: "#a855f7",
                }[band] ?? "#64748b";

                return (
                  <div key={band}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="font-bold uppercase" style={{ color: hexColor }}>{band}</span>
                      <span className="font-mono text-slate-400">{value.toFixed(2)}</span>
                    </div>
                    <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full transition-all duration-1000"
                        style={{ width: `${Math.round(displayPct)}%`, backgroundColor: hexColor }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Brain3D;