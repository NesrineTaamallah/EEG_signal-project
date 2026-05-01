import { useRef, useMemo, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere, Html } from "@react-three/drei";
import * as THREE from "three";
import { type FrequencyBands, ELECTRODE_POSITIONS } from "@/lib/eeg-types";
import { useStore } from "@/store/useStore";
import { cn } from "@/lib/utils";
import { Brain, Zap, Activity } from "lucide-react";

// ── Color helpers ──────────────────────────────────────────────────────────────

function getStressColor(stressLevel: number): THREE.Color {
  const t = Math.min(Math.max(stressLevel / 100, 0), 1);
  if (t < 0.33) {
    return new THREE.Color().lerpColors(
      new THREE.Color(0x10b981), new THREE.Color(0xf59e0b), t / 0.33
    );
  } else if (t < 0.66) {
    return new THREE.Color().lerpColors(
      new THREE.Color(0xf59e0b), new THREE.Color(0xef4444), (t - 0.33) / 0.33
    );
  }
  return new THREE.Color().lerpColors(
    new THREE.Color(0xef4444), new THREE.Color(0x7f1d1d), (t - 0.66) / 0.34
  );
}

function getBandColor(band: keyof FrequencyBands): THREE.Color {
  const colors: Record<keyof FrequencyBands, number> = {
    delta: 0x3b82f6,
    theta: 0x06b6d4,
    alpha: 0x10b981,
    beta:  0xf59e0b,
    gamma: 0xa855f7,
  };
  return new THREE.Color(colors[band] ?? 0x10b981);
}

function getDominantBand(data: FrequencyBands): keyof FrequencyBands {
  const entries = Object.entries(data) as [keyof FrequencyBands, number][];
  if (!entries.length || entries.every(([, v]) => v === 0)) return "alpha";
  return entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
}

// ── Brain mesh ────────────────────────────────────────────────────────────────

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

  const color            = useMemo(() => getStressColor(stressLevel), [stressLevel]);
  const dominantBandColor = useMemo(() => getBandColor(getDominantBand(frequencyData)), [frequencyData]);
  const dominantBand      = useMemo(() => getDominantBand(frequencyData), [frequencyData]);

  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.12;
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
      {/* Outer glow */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[1.2, 32, 32]} />
        <meshBasicMaterial
          color={dominantBandColor}
          transparent
          opacity={0.08 + (stressLevel / 100) * 0.12}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Main brain sphere */}
      <mesh
        ref={meshRef}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[1, 64, 64]} />
        <meshPhongMaterial
          color={color}
          emissive={dominantBandColor}
          emissiveIntensity={isProcessing ? 0.5 : 0.15 + (stressLevel / 100) * 0.35}
          shininess={60}
          transparent
          opacity={0.88}
        />
      </mesh>

      {/* Inner activity sphere — pulses with stress level */}
      <mesh>
        <sphereGeometry args={[0.55, 32, 32]} />
        <meshPhongMaterial
          color={dominantBandColor}
          emissive={dominantBandColor}
          emissiveIntensity={0.3 + (stressLevel / 100) * 0.7}
          transparent
          opacity={0.35 + (stressLevel / 100) * 0.25}
        />
      </mesh>

      {/* Cortex fold lines */}
      {Array.from({ length: 8 }).map((_, i) => (
        <mesh key={i} rotation={[0, (i * Math.PI) / 4, Math.PI / 6]}>
          <torusGeometry args={[0.95, 0.015, 8, 32, Math.PI]} />
          <meshBasicMaterial color={0x1a237e} transparent opacity={0.4} />
        </mesh>
      ))}

      {/* Hover tooltip */}
      {hovered && (
        <Html distanceFactor={3}>
          <div className="rounded-lg bg-slate-900/95 px-4 py-2.5 text-xs backdrop-blur border border-slate-700 shadow-xl min-w-[160px]">
            <p className="font-semibold text-white mb-1">Activité Cérébrale</p>
            <p className="text-slate-400">
              Stress: <span className={cn(
                "font-mono font-bold",
                stressLevel > 60 ? "text-red-400" : stressLevel > 40 ? "text-amber-400" : "text-green-400"
              )}>{Math.round(stressLevel)}%</span>
            </p>
            <p className="text-slate-400">
              Bande dominante:{" "}
              <span className="text-white font-medium">{dominantBand.toUpperCase()}</span>
            </p>
          </div>
        </Html>
      )}
    </group>
  );
}

// ── Electrode node ─────────────────────────────────────────────────────────────

function ElectrodeNode({
  position,
  electrode,
  activityPct,
  bandColor,
  isActive,
  onClick,
}: {
  position: [number, number, number];
  electrode: (typeof ELECTRODE_POSITIONS)[0];
  activityPct: number;
  bandColor: string;
  isActive: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const color = useMemo(() => {
    const base = new THREE.Color(bandColor);
    return base;
  }, [bandColor]);

  useFrame((state) => {
    if (meshRef.current) {
      const pulse = isActive
        ? Math.sin(state.clock.elapsedTime * 5) * 0.4 + 1
        : 1 + Math.sin(state.clock.elapsedTime * 2 + activityPct) * 0.05;
      meshRef.current.scale.setScalar(pulse);
    }
  });

  return (
    <group position={position}>
      <Sphere
        ref={meshRef}
        args={[0.065, 16, 16]}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshPhongMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isActive ? 0.9 : 0.2 + activityPct * 0.006}
        />
      </Sphere>

      {/* Activity ring around active electrode */}
      {isActive && (
        <mesh>
          <torusGeometry args={[0.12, 0.01, 8, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.6} />
        </mesh>
      )}

      {(hovered || isActive) && (
        <Html distanceFactor={3}>
          <div className="whitespace-nowrap rounded-lg bg-slate-900/95 px-3 py-2 text-xs backdrop-blur border border-slate-700 shadow-lg">
            <p className="font-mono font-semibold text-white">{electrode.name}</p>
            <p className="text-slate-400">{electrode.region}</p>
            <p style={{ color: bandColor }} className="font-mono font-bold">
              {activityPct.toFixed(1)}%
            </p>
          </div>
        </Html>
      )}
    </group>
  );
}

// ── Electrode network driven by real per-channel band powers ───────────────────

function ElectrodeNetwork({
  activeElectrode,
  onElectrodeClick,
  stressLevel,
  dominantBand,
  channelBandPowers,
}: {
  activeElectrode?: string | null;
  onElectrodeClick?: (id: string) => void;
  stressLevel: number;
  dominantBand: string;
  channelBandPowers: Record<string, Record<string, number>>;
}) {
  const bandColorHex: Record<string, string> = {
    delta: "#3b82f6",
    theta: "#06b6d4",
    alpha: "#10b981",
    beta:  "#f59e0b",
    gamma: "#a855f7",
  };

  return (
    <group>
      {ELECTRODE_POSITIONS.map((electrode, idx) => {
        // Match channel data by index if available
        const chKey = `EEG${idx + 1}`;
        const chData = channelBandPowers[chKey];

        let activityPct: number;
        let bandCol: string;

        if (chData) {
          // Use real beta/alpha ratio as activity indicator
          const alpha = (chData["alpha"] ?? 1) + 1e-10;
          const beta  = chData["beta"]  ?? 0;
          const dominantVal = chData[dominantBand] ?? 0;
          activityPct = Math.min((beta / alpha) * 20 + stressLevel * 0.3, 100);
          bandCol = bandColorHex[dominantBand] ?? "#10b981";
          // Blend toward stress color when stress is high
          if (stressLevel > 60) bandCol = "#ef4444";
        } else {
          // Fallback: spatial + stress heuristic
          activityPct = Math.min(
            (Math.abs(electrode.x + electrode.y) * 25 + stressLevel * 0.4) % 100,
            100
          );
          bandCol = bandColorHex[dominantBand] ?? "#10b981";
        }

        return (
          <ElectrodeNode
            key={electrode.id}
            position={[electrode.x * 1.1, electrode.z * 1.1, electrode.y * 1.1]}
            electrode={electrode}
            activityPct={activityPct}
            bandColor={bandCol}
            isActive={activeElectrode === electrode.id}
            onClick={() => onElectrodeClick?.(electrode.id)}
          />
        );
      })}
    </group>
  );
}

// ── Neural particles — speed driven by stress ─────────────────────────────────

function NeuralParticles({
  stressLevel,
  dominantBand,
}: {
  stressLevel: number;
  dominantBand: string;
}) {
  const particlesRef = useRef<THREE.Points>(null);
  const particleCount = 300;

  const [positions, velocities] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const vel = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(Math.random() * 2 - 1);
      const r     = 0.9 + Math.random() * 0.5;
      pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
      vel[i * 3]     = (Math.random() - 0.5) * 0.02;
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.02;
    }
    return [pos, vel];
  }, []);

  const color = useMemo(() => getStressColor(stressLevel), [stressLevel]);

  useFrame((_, delta) => {
    if (!particlesRef.current) return;
    const posAttr = particlesRef.current.geometry.attributes.position;
    const speed   = 0.4 + (stressLevel / 100) * 2.0;
    for (let i = 0; i < particleCount; i++) {
      const idx = i * 3;
      posAttr.array[idx]     += (velocities[idx]     as number) * speed * delta * 30;
      posAttr.array[idx + 1] += (velocities[idx + 1] as number) * speed * delta * 30;
      posAttr.array[idx + 2] += (velocities[idx + 2] as number) * speed * delta * 30;
      const dist = Math.sqrt(
        (posAttr.array[idx] as number)     ** 2 +
        (posAttr.array[idx + 1] as number) ** 2 +
        (posAttr.array[idx + 2] as number) ** 2
      );
      if (dist > 1.6 || dist < 0.6) {
        velocities[idx]     *= -1;
        velocities[idx + 1] *= -1;
        velocities[idx + 2] *= -1;
      }
    }
    posAttr.needsUpdate = true;
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.025}
        color={color}
        transparent
        opacity={0.5 + (stressLevel / 100) * 0.3}
        sizeAttenuation
      />
    </points>
  );
}

// ── Scene ─────────────────────────────────────────────────────────────────────

function Scene({
  frequencyData,
  stressLevel,
  isProcessing,
  activeElectrode,
  onElectrodeClick,
  dominantBand,
  channelBandPowers,
}: {
  frequencyData: FrequencyBands;
  stressLevel: number;
  isProcessing: boolean;
  activeElectrode?: string | null;
  onElectrodeClick?: (id: string) => void;
  dominantBand: string;
  channelBandPowers: Record<string, Record<string, number>>;
}) {
  const norm     = stressLevel / 100;
  const coolCol  = new THREE.Color("#0d9488");
  const hotCol   = new THREE.Color("#ef4444");
  const dynColor = coolCol.clone().lerp(hotCol, norm);

  return (
    <>
      <ambientLight intensity={0.2 + norm * 0.5} color={dynColor} />
      <pointLight position={[10, 10, 10]} intensity={0.5 + norm * 2.0} color={dynColor} />
      <pointLight position={[-10, -10, -10]} intensity={0.2} color="#4fc3f7" />
      <spotLight
        position={[0, 5, 0]}
        angle={0.3}
        penumbra={1}
        intensity={0.5 + norm * 1.5}
        color={dynColor}
      />
      <BrainMesh
        frequencyData={frequencyData}
        stressLevel={stressLevel}
        isProcessing={isProcessing}
      />
      <ElectrodeNetwork
        activeElectrode={activeElectrode}
        onElectrodeClick={onElectrodeClick}
        stressLevel={stressLevel}
        dominantBand={dominantBand}
        channelBandPowers={channelBandPowers}
      />
      <NeuralParticles stressLevel={stressLevel} dominantBand={dominantBand} />
      <OrbitControls
        enablePan={false}
        minDistance={2}
        maxDistance={6}
        autoRotate={!isProcessing}
        autoRotateSpeed={0.4 + norm * 0.8}
      />
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Brain3D page
// ═══════════════════════════════════════════════════════════════════════════════

const Brain3D: React.FC = () => {
  const { prediction, isProcessing } = useStore();
  const [activeElectrode, setActiveElectrode] = useState<string | null>(null);

  const stressLevel = (prediction?.probabilities?.stress ?? 0) * 100;

  // Average band powers → FrequencyBands for mesh coloring
  const frequencyData = useMemo<FrequencyBands>(() => {
    const defaults: FrequencyBands = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 };
    if (!prediction?.bandPowers) return defaults;
    return prediction.bandPowers.reduce<FrequencyBands>((acc, item) => {
      const [band, val] = Object.entries(item)[0] ?? [];
      if (band && band in acc) acc[band as keyof FrequencyBands] = Number(val);
      return acc;
    }, { ...defaults });
  }, [prediction]);

  const dominantBand = useMemo(() => {
    const entries = Object.entries(frequencyData) as [keyof FrequencyBands, number][];
    if (entries.every(([, v]) => v === 0)) return "alpha";
    return entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0] as string;
  }, [frequencyData]);

  // Per-channel band powers map: { "EEG1": {delta, theta, alpha, beta, gamma}, ... }
  const channelBandPowers = useMemo<Record<string, Record<string, number>>>(() => {
    if (!prediction?.bandPowersPerCh) return {};
    const map: Record<string, Record<string, number>> = {};
    prediction.bandPowersPerCh.forEach((ch) => {
      const { channel, ...bands } = ch as any;
      map[channel] = bands;
    });
    return map;
  }, [prediction]);

  // Selected electrode info
  const selectedElectrodeData = useMemo(() => {
    if (!activeElectrode) return null;
    const pos = ELECTRODE_POSITIONS.find((e) => e.id === activeElectrode);
    if (!pos) return null;
    const chKey  = `EEG${ELECTRODE_POSITIONS.indexOf(pos) + 1}`;
    const chData = channelBandPowers[chKey];
    const alpha  = (chData?.["alpha"] ?? 1) + 1e-10;
    const beta   = chData?.["beta"] ?? 0;
    const value  = Math.min((beta / alpha) * 20 + stressLevel * 0.3, 100);
    return { ...pos, value, chData };
  }, [activeElectrode, channelBandPowers, stressLevel]);

  // Band display for sidebar
  const bandDisplayValues = useMemo(() =>
    Object.entries(frequencyData).map(([band, value]) => ({
      band,
      value: value as number,
      displayPct: Math.min(Math.max(((value as number) / 10) * 100, 0), 100),
    })),
    [frequencyData]
  );

  const BAND_HEX: Record<string, string> = {
    delta: "#3b82f6", theta: "#06b6d4",
    alpha: "#10b981", beta: "#f59e0b", gamma: "#a855f7",
  };

  return (
    <div className="space-y-6 h-full flex flex-col">
      {/* Header */}
      <header>
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-accent-violet/20 rounded-xl">
            <Brain className="w-6 h-6 text-accent-violet" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">Carte Neurale 3D</h2>
        </div>
        <p className="text-slate-400">
          Cartographie dynamique des activités EEG — liée en temps réel aux résultats de classification.
        </p>
      </header>

      {!prediction && (
        <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl text-amber-400 text-sm flex items-center gap-2">
          <Zap className="w-4 h-4 shrink-0" />
          Aucune prédiction disponible. Lancez une classification dans l'onglet{" "}
          <strong>Classification</strong> pour animer le cerveau avec des données réelles.
        </div>
      )}

      {prediction && (
        <div className="flex flex-wrap gap-4 p-3 bg-slate-800/40 border border-slate-700/50 rounded-xl text-xs">
          <span className="text-slate-400">
            <span className="text-slate-300 font-medium">Modèle:</span>{" "}
            {prediction.model_source === "trained_model"
              ? <span className="text-accent-teal">✓ .joblib actif</span>
              : <span className="text-amber-400">⚠ Heuristique</span>}
          </span>
          <span className="text-slate-400">
            <span className="text-slate-300 font-medium">Bande dominante:</span>{" "}
            <span style={{ color: BAND_HEX[dominantBand] ?? "#fff" }} className="font-bold">
              {dominantBand.toUpperCase()}
            </span>
          </span>
          <span className="text-slate-400">
            <span className="text-slate-300 font-medium">Canaux avec données réelles:</span>{" "}
            {Object.keys(channelBandPowers).length}
          </span>
        </div>
      )}

      <div className="flex gap-6 flex-1 min-h-[520px]">
        {/* 3D Canvas */}
        <div className="flex-1 relative">
          <div className="relative h-full w-full min-h-[520px] rounded-xl overflow-hidden bg-gradient-to-b from-slate-900 to-background border border-slate-800">
            <Canvas
              camera={{ position: [0, 0, 3.5], fov: 50 }}
              gl={{ antialias: true, alpha: true }}
            >
              <Scene
                frequencyData={frequencyData}
                stressLevel={stressLevel}
                isProcessing={isProcessing}
                activeElectrode={activeElectrode}
                onElectrodeClick={setActiveElectrode}
                dominantBand={dominantBand}
                channelBandPowers={channelBandPowers}
              />
            </Canvas>

            {/* Legend */}
            <div className="absolute bottom-4 left-4 flex flex-wrap gap-2 text-[10px]">
              {Object.entries(BAND_HEX).map(([band, color]) => (
                <div
                  key={band}
                  className={cn(
                    "flex items-center gap-1.5 rounded-full px-2 py-1 backdrop-blur border transition-all",
                    dominantBand === band
                      ? "border-white/30 bg-slate-900/90"
                      : "border-slate-800 bg-slate-900/70"
                  )}
                >
                  <div
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  <span className={dominantBand === band ? "text-white font-bold" : "text-slate-400"}>
                    {band.charAt(0).toUpperCase() + band.slice(1)}
                  </span>
                  {dominantBand === band && (
                    <span className="text-white/60">★</span>
                  )}
                </div>
              ))}
            </div>

            {/* Processing indicator */}
            {isProcessing && (
              <div className="absolute right-4 top-4 flex items-center gap-2 rounded-full bg-accent-violet/20 px-3 py-1.5 text-xs backdrop-blur border border-accent-violet/30">
                <div className="h-2 w-2 animate-pulse rounded-full bg-accent-violet" />
                <span className="text-accent-violet font-medium">Traitement...</span>
              </div>
            )}

            {/* Stress level overlay */}
            {prediction && (
              <div className="absolute top-4 left-4 bg-slate-900/80 backdrop-blur rounded-xl border border-slate-700 px-4 py-2">
                <div className="flex items-center gap-2">
                  <Activity className="w-3 h-3 text-slate-400" />
                  <span className="text-xs text-slate-400">Niveau de stress</span>
                </div>
                <div
                  className="text-2xl font-black font-mono mt-0.5"
                  style={{
                    color: stressLevel > 60
                      ? "#ef4444"
                      : stressLevel > 40 ? "#f59e0b" : "#10b981"
                  }}
                >
                  {Math.round(stressLevel)}%
                </div>
              </div>
            )}
          </div>

          {/* Selected electrode panel */}
          {selectedElectrodeData && (
            <div className="absolute top-20 left-4 w-60 bg-slate-900/95 backdrop-blur-md border border-slate-700 rounded-2xl p-5 shadow-2xl">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <div className="text-accent-teal font-mono text-xs font-bold uppercase tracking-widest mb-1">
                    Électrode
                  </div>
                  <h3 className="text-2xl font-black text-white leading-none">
                    {selectedElectrodeData.name}
                  </h3>
                </div>
                <button
                  onClick={() => setActiveElectrode(null)}
                  className="p-1 hover:bg-slate-800 rounded-lg transition-colors text-slate-500 hover:text-white text-lg leading-none"
                >
                  ×
                </button>
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between py-1.5 border-b border-slate-800/50">
                  <span className="text-slate-500 font-bold uppercase text-[10px]">Région</span>
                  <span className="text-white">{selectedElectrodeData.region}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800/50">
                  <span className="text-slate-500 font-bold uppercase text-[10px]">Activité</span>
                  <span className="text-accent-teal font-mono font-bold">
                    {selectedElectrodeData.value.toFixed(1)}%
                  </span>
                </div>
                {selectedElectrodeData.chData && (
                  <>
                    {Object.entries(selectedElectrodeData.chData).map(([band, val]) => (
                      <div key={band} className="flex justify-between py-1">
                        <span
                          className="font-bold uppercase text-[10px]"
                          style={{ color: BAND_HEX[band] ?? "#fff" }}
                        >
                          {band}
                        </span>
                        <span className="font-mono text-slate-300">{Number(val).toFixed(3)}</span>
                      </div>
                    ))}
                  </>
                )}
                <div className="mt-2">
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

        {/* Sidebar — band powers */}
        <div className="w-52 space-y-4 shrink-0">
          {/* Stress gauge */}
          <div className="card p-4">
            <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">
              Niveau de Stress
            </h4>
            <div
              className="text-3xl font-black font-mono"
              style={{
                color: stressLevel > 60
                  ? "#ef4444"
                  : stressLevel > 40 ? "#f59e0b" : "#10b981"
              }}
            >
              {Math.round(stressLevel)}%
            </div>
            <div className="mt-2 h-2 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full transition-all duration-1000 rounded-full"
                style={{
                  width: `${Math.round(stressLevel)}%`,
                  backgroundColor: stressLevel > 60
                    ? "#ef4444"
                    : stressLevel > 40 ? "#f59e0b" : "#10b981",
                }}
              />
            </div>
            <p className="text-[10px] text-slate-600 mt-2">
              {prediction?.model_source === "trained_model"
                ? "Source: modèle .joblib"
                : "Source: heuristique EEG"}
            </p>
          </div>

          {/* Band EEG */}
          <div className="card p-4">
            <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">
              Bandes EEG
            </h4>
            <div className="space-y-3">
              {bandDisplayValues.map(({ band, value, displayPct }) => {
                const hexColor = BAND_HEX[band] ?? "#64748b";
                const isDominant = band === dominantBand;
                return (
                  <div key={band}>
                    <div className="flex justify-between text-xs mb-1">
                      <span
                        className={cn("font-bold uppercase text-[10px]", isDominant && "text-white")}
                        style={{ color: hexColor }}
                      >
                        {band} {isDominant && "★"}
                      </span>
                      <span className="font-mono text-slate-400">{value.toFixed(2)}</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={cn(
                          "h-full transition-all duration-1000",
                          isDominant && "shadow-sm"
                        )}
                        style={{
                          width: `${Math.round(displayPct)}%`,
                          backgroundColor: hexColor,
                          boxShadow: isDominant ? `0 0 6px ${hexColor}` : "none",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Classification verdict */}
          {prediction && (
            <div className={cn(
              "card p-4 border",
              prediction.prediction === 1
                ? "border-red-500/30 bg-red-500/5"
                : "border-green-500/30 bg-green-500/5"
            )}>
              <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">
                Verdict
              </h4>
              <div className={cn(
                "text-sm font-black",
                prediction.prediction === 1 ? "text-red-400" : "text-green-400"
              )}>
                {prediction.prediction === 1 ? "⚠ STRESS" : "✓ NORMAL"}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Confiance: <span className="text-white font-mono">
                  {Math.round(prediction.confidence * 100)}%
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Brain3D;