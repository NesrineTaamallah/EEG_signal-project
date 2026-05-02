import React, { useState, useMemo, useCallback } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, LineChart, Line, ScatterChart, Scatter, ZAxis,
  ReferenceLine
} from "recharts";
import { useStore } from "@/store/useStore";
import { extractFeatures } from "@/services/api";
import {
  Cpu, TrendingUp, Activity, Zap,
  Loader2, AlertCircle, ChevronDown, ChevronRight,
  Grid, Waves
} from "lucide-react";
import { cn } from "@/lib/utils";

// ── Types ─────────────────────────────────────────────────────────────────────

interface FeatureGroup {
  name: string;
  features: Record<string, number>;
  color: string;
}

interface ExtractedFeatures {
  groups: FeatureGroup[];
  bandPowers: Record<string, number>[];
  statistics: {
    totalFeatures: number;
    channels: number;
    windows: number;
    samplingRate: number;
  };
  channelProfiles: Array<{
    channel: string;
    variance: number;
    rms: number;
    mobility: number;
    complexity: number;
    betaAlpha: number;
    thetaAlpha: number;
    entropy: number;
  }>;
  temporalEvolution: Array<{
    window: number;
    delta: number;
    theta: number;
    alpha: number;
    beta: number;
    gamma: number;
    arousal: number;
  }>;
}

// ── Sub-components ─────────────────────────────────────────────────────────────

const StatCard: React.FC<{ label: string; value: string | number; sub?: string; color?: string }> = ({
  label, value, sub, color = "text-accent-teal"
}) => (
  <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4">
    <div className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-1">{label}</div>
    <div className={cn("text-2xl font-bold font-mono", color)}>{value}</div>
    {sub && <div className="text-xs text-slate-600 mt-1">{sub}</div>}
  </div>
);

const SectionHeader: React.FC<{
  icon: React.ReactNode;
  title: string;
  expanded: boolean;
  onToggle: () => void;
}> = ({ icon, title, expanded, onToggle }) => (
  <button onClick={onToggle} className="flex items-center gap-3 w-full text-left group">
    <div className="p-2 bg-slate-800 rounded-lg text-accent-violet group-hover:bg-slate-700 transition-colors">
      {icon}
    </div>
    <span className="font-bold text-lg text-slate-200 group-hover:text-white transition-colors flex-1">
      {title}
    </span>
    {expanded
      ? <ChevronDown className="w-4 h-4 text-slate-500" />
      : <ChevronRight className="w-4 h-4 text-slate-500" />}
  </button>
);

const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-slate-400 mb-1 font-medium">{label}</p>
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
          <span className="text-slate-300">{p.name}:</span>
          <span className="text-white font-mono">
            {typeof p.value === "number" ? p.value.toFixed(4) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
};

const BAND_COLORS: Record<string, string> = {
  delta: "#3b82f6",
  theta: "#06b6d4",
  alpha: "#10b981",
  beta:  "#f59e0b",
  gamma: "#a855f7",
  arousal: "#ef4444",
};

// ── Main Component ────────────────────────────────────────────────────────────

const FeatureVisualizer: React.FC = () => {
  const { activeSignal } = useStore();
  const [features, setFeatures] = useState<ExtractedFeatures | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedChannel, setSelectedChannel] = useState(0);
  const [activeView, setActiveView] = useState<"radar" | "bar" | "scatter">("radar");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    stats: true, bands: true, temporal: true, channels: true
  });

  const toggle = useCallback((key: string) => {
    setExpanded(prev => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleExtract = useCallback(async () => {
    if (!activeSignal) return;
    setError(null);
    setIsExtracting(true);
    try {
      const signal = activeSignal.cleaned?.length > 0 ? activeSignal.cleaned : activeSignal.raw;
      const result = await extractFeatures(signal, activeSignal.sfreq);
      setFeatures(result);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "Feature extraction failed.");
    } finally {
      setIsExtracting(false);
    }
  }, [activeSignal]);

  const radarData = useMemo(() => {
    if (!features?.channelProfiles?.[selectedChannel]) return [];
    const ch = features.channelProfiles[selectedChannel];
    return [
      { feature: "Variance",   value: Math.min(Math.abs(ch.variance) * 100, 100) },
      { feature: "RMS",        value: Math.min(ch.rms * 100, 100) },
      { feature: "Mobility",   value: Math.min(ch.mobility * 50, 100) },
      { feature: "Complexity", value: Math.min(ch.complexity * 30, 100) },
      { feature: "β/α Ratio",  value: Math.min(ch.betaAlpha * 20, 100) },
      { feature: "θ/α Ratio",  value: Math.min(ch.thetaAlpha * 20, 100) },
      { feature: "Entropy",    value: Math.min(ch.entropy * 15, 100) },
    ];
  }, [features, selectedChannel]);

  const bandBarData = useMemo(() => {
    if (!features?.channelProfiles) return [];
    return features.channelProfiles.map((ch, i) => ({
      channel: ch.channel || `EEG${i + 1}`,
      beta:  ch.betaAlpha,
      theta: ch.thetaAlpha,
    }));
  }, [features]);

  const scatterData = useMemo(() => {
    if (!features?.channelProfiles) return [];
    return features.channelProfiles.map((ch, i) => ({
      x: ch.complexity,
      y: ch.entropy,
      z: ch.rms * 1000,
      name: ch.channel || `EEG${i + 1}`,
    }));
  }, [features]);

  if (!activeSignal) {
    return (
      <div className="card h-96 flex flex-col items-center justify-center border-dashed border-2 border-slate-800">
        <Cpu className="w-12 h-12 text-slate-600 mb-4" />
        <p className="text-slate-500 font-medium">No signal loaded.</p>
        <p className="text-sm text-slate-600 mt-1">
          Import an EEG signal in the <strong className="text-slate-400">Analyzer</strong> tab first.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <header className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-accent-violet/20 rounded-xl">
              <Cpu className="w-6 h-6 text-accent-violet" />
            </div>
            <h2 className="text-3xl font-bold tracking-tight">Feature Explorer</h2>
          </div>
          <p className="text-slate-400">
            Interactive visualization of extracted EEG features.
          </p>
        </div>

        <button
          onClick={handleExtract}
          disabled={isExtracting}
          className="btn-primary flex items-center gap-2"
        >
          {isExtracting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
          <span>{isExtracting ? "Extracting..." : "Extract Features"}</span>
        </button>
      </header>

      {/* Signal info banner */}
      <div className="flex flex-wrap gap-4 p-3 bg-slate-800/40 border border-slate-700/50 rounded-xl text-xs">
        <span className="text-slate-400">
          <span className="text-slate-300 font-medium">Signal:</span>{" "}
          {activeSignal.cleaned?.length > 0 ? "✓ Cleaned" : "⚠ Raw"}
        </span>
        <span className="text-slate-400">
          <span className="text-slate-300 font-medium">Channels:</span>{" "}
          {activeSignal.channelNames.length}
        </span>
        <span className="text-slate-400">
          <span className="text-slate-300 font-medium">Sample rate:</span>{" "}
          {activeSignal.sfreq} Hz
        </span>
        <span className="text-slate-400">
          <span className="text-slate-300 font-medium">Duration:</span>{" "}
          {activeSignal.raw[0]
            ? (activeSignal.raw[0].length / activeSignal.sfreq).toFixed(1)
            : "?"} s
        </span>
      </div>

      {error && (
        <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {!features && !isExtracting && (
        <div className="card h-64 flex flex-col items-center justify-center border-dashed border-slate-700">
          <Grid className="w-10 h-10 text-slate-700 mb-3" />
          <p className="text-slate-500 text-sm">
            Click "Extract Features" to analyze the EEG signal.
          </p>
          <p className="text-slate-600 text-xs mt-2">
            Extracts time-domain, frequency, Hjorth, fractal, and entropy features.
          </p>
        </div>
      )}

      {isExtracting && (
        <div className="card h-64 flex flex-col items-center justify-center gap-4">
          <Loader2 className="w-10 h-10 text-accent-violet animate-spin" />
          <div className="text-center">
            <p className="text-white font-medium">Extracting features...</p>
            <p className="text-slate-400 text-sm mt-1">
              Computing time-domain, spectral, Hjorth, fractal and entropy features
            </p>
          </div>
        </div>
      )}

      {features && (
        <div className="space-y-8">

          {/* ── Stats Overview ────────────────────────────────────────── */}
          <section>
            <SectionHeader
              icon={<Activity className="w-4 h-4" />}
              title="Overview"
              expanded={expanded.stats}
              onToggle={() => toggle("stats")}
            />
            {expanded.stats && (
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                <StatCard
                  label="Channels"
                  value={features.statistics.channels}
                  sub={activeSignal.channelNames[0] + " ... " + activeSignal.channelNames.at(-1)}
                  color="text-accent-teal"
                />
                <StatCard
                  label="Windows"
                  value={features.statistics.windows}
                  sub="50% overlap"
                  color="text-blue-400"
                />
                <StatCard
                  label="Sample Rate"
                  value={features.statistics.samplingRate + " Hz"}
                  sub="1 s window"
                  color="text-amber-400"
                />
              </div>
            )}
          </section>

          {/* ── Band Power Temporal Evolution ─────────────────────────── */}
          <section>
            <SectionHeader
              icon={<Waves className="w-4 h-4" />}
              title="Temporal Evolution of Frequency Bands"
              expanded={expanded.temporal}
              onToggle={() => toggle("temporal")}
            />
            {expanded.temporal && features.temporalEvolution?.length > 0 && (
              <div className="mt-4 card p-4">
                <p className="text-xs text-slate-500 mb-4">
                  Band power (µV²/Hz) per analysis window — averaged across all channels
                </p>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={features.temporalEvolution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="window"
                      stroke="#64748b"
                      fontSize={10}
                      label={{ value: "Window", position: "insideBottom", offset: -4, fill: "#64748b", fontSize: 11 }}
                    />
                    <YAxis stroke="#64748b" fontSize={10} />
                    <Tooltip content={<CustomTooltip />} />
                    {Object.entries(BAND_COLORS)
                      .filter(([k]) => k !== "arousal")
                      .map(([band, color]) => (
                        <Line
                          key={band}
                          type="monotone"
                          dataKey={band}
                          stroke={color}
                          strokeWidth={1.5}
                          dot={false}
                          name={band.charAt(0).toUpperCase() + band.slice(1)}
                        />
                      ))}
                  </LineChart>
                </ResponsiveContainer>

                {/* Arousal index */}
                <div className="mt-4 border-t border-slate-800 pt-4">
                  <p className="text-xs text-slate-500 mb-3">
                    Arousal index (β+γ)/(δ+θ+α) — EEG stress marker
                  </p>
                  <ResponsiveContainer width="100%" height={100}>
                    <LineChart data={features.temporalEvolution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="window" stroke="#64748b" fontSize={9} />
                      <YAxis stroke="#64748b" fontSize={9} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine y={1} stroke="#f59e0b" strokeDasharray="4 2" strokeWidth={1} />
                      <Line
                        type="monotone"
                        dataKey="arousal"
                        stroke={BAND_COLORS.arousal}
                        strokeWidth={2}
                        dot={false}
                        name="Arousal"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-xs text-slate-600 mt-1">
                    Yellow line = threshold 1.0 (above = higher potential stress)
                  </p>
                </div>
              </div>
            )}
          </section>

          {/* ── Per-channel Analysis ───────────────────────────────────── */}
          <section>
            <SectionHeader
              icon={<Activity className="w-4 h-4" />}
              title="Channel Profile"
              expanded={expanded.channels}
              onToggle={() => toggle("channels")}
            />
            {expanded.channels && (
              <div className="mt-4 space-y-4">
                {/* Channel selector */}
                <div className="flex flex-wrap gap-1.5">
                  {(features.channelProfiles || []).map((ch, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedChannel(i)}
                      className={cn(
                        "px-2.5 py-1 rounded-lg text-xs font-mono transition-all",
                        selectedChannel === i
                          ? "bg-accent-violet text-white"
                          : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                      )}
                    >
                      {ch.channel || `EEG${i + 1}`}
                    </button>
                  ))}
                </div>

                {/* View toggle */}
                <div className="flex items-center gap-2">
                  {(["radar", "bar", "scatter"] as const).map(v => (
                    <button
                      key={v}
                      onClick={() => setActiveView(v)}
                      className={cn(
                        "px-3 py-1.5 rounded-lg text-xs font-medium transition-all capitalize",
                        activeView === v
                          ? "bg-accent-teal text-white"
                          : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                      )}
                    >
                      {v === "radar" ? "Radar" : v === "bar" ? "Spectral Ratios" : "Complexity/Entropy"}
                    </button>
                  ))}
                </div>

                <div className="card p-4">
                  {activeView === "radar" && (
                    <>
                      <p className="text-xs text-slate-500 mb-2">
                        Selected channel:{" "}
                        <span className="text-accent-violet font-medium">
                          {features.channelProfiles[selectedChannel]?.channel || `EEG${selectedChannel + 1}`}
                        </span>{" "}
                        — normalized profile 0–100
                      </p>
                      <ResponsiveContainer width="100%" height={300}>
                        <RadarChart data={radarData}>
                          <PolarGrid stroke="#1e293b" />
                          <PolarAngleAxis dataKey="feature" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 9 }} />
                          <Radar
                            name="Feature"
                            dataKey="value"
                            stroke="#7c3aed"
                            fill="#7c3aed"
                            fillOpacity={0.3}
                            strokeWidth={2}
                          />
                          <Tooltip content={<CustomTooltip />} />
                        </RadarChart>
                      </ResponsiveContainer>
                      <div className="grid grid-cols-4 gap-2 mt-4 border-t border-slate-800 pt-4">
                        {features.channelProfiles[selectedChannel] && (
                          <>
                            {[
                              ["Variance",   features.channelProfiles[selectedChannel].variance.toExponential(2)],
                              ["RMS",        features.channelProfiles[selectedChannel].rms.toExponential(2)],
                              ["Mobility",   features.channelProfiles[selectedChannel].mobility.toFixed(4)],
                              ["Complexity", features.channelProfiles[selectedChannel].complexity.toFixed(4)],
                              ["β/α",        features.channelProfiles[selectedChannel].betaAlpha.toFixed(3)],
                              ["θ/α",        features.channelProfiles[selectedChannel].thetaAlpha.toFixed(3)],
                              ["Entropy",    features.channelProfiles[selectedChannel].entropy.toFixed(4)],
                            ].map(([label, val]) => (
                              <div key={label} className="bg-slate-800/50 rounded-lg p-2 text-center">
                                <div className="text-xs text-slate-500">{label}</div>
                                <div className="text-xs font-mono text-accent-teal mt-0.5">{val}</div>
                              </div>
                            ))}
                          </>
                        )}
                      </div>
                    </>
                  )}

                  {activeView === "bar" && (
                    <>
                      <p className="text-xs text-slate-500 mb-4">
                        β/α (cognitive stress) and θ/α (engagement/fatigue) ratios per channel
                      </p>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={bandBarData} layout="vertical" margin={{ left: 40, right: 20, top: 8, bottom: 8 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                          <XAxis type="number" stroke="#64748b" fontSize={10} />
                          <YAxis dataKey="channel" type="category" stroke="#64748b" fontSize={9} width={50} />
                          <Tooltip content={<CustomTooltip />} />
                          <Bar dataKey="beta"  fill={BAND_COLORS.beta}  name="β/α" radius={[0, 3, 3, 0]} barSize={8} />
                          <Bar dataKey="theta" fill={BAND_COLORS.theta} name="θ/α" radius={[0, 3, 3, 0]} barSize={8} />
                        </BarChart>
                      </ResponsiveContainer>
                      <p className="text-xs text-slate-600 mt-2">
                        High β/α → potential cognitive stress · High θ/α → engagement or fatigue
                      </p>
                    </>
                  )}

                  {activeView === "scatter" && (
                    <>
                      <p className="text-xs text-slate-500 mb-4">
                        Hjorth Complexity vs. Spectral Entropy — size = RMS amplitude
                      </p>
                      <ResponsiveContainer width="100%" height={300}>
                        <ScatterChart margin={{ left: 20, right: 20, top: 8, bottom: 30 }}>
                          <CartesianGrid stroke="#1e293b" />
                          <XAxis
                            dataKey="x"
                            name="Complexity"
                            stroke="#64748b"
                            fontSize={10}
                            label={{ value: "Hjorth Complexity", position: "insideBottom", offset: -10, fill: "#64748b", fontSize: 11 }}
                          />
                          <YAxis
                            dataKey="y"
                            name="Entropy"
                            stroke="#64748b"
                            fontSize={10}
                            label={{ value: "Entropy", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
                          />
                          <ZAxis dataKey="z" range={[40, 400]} name="RMS" />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (!active || !payload?.[0]) return null;
                              const d = payload[0].payload;
                              return (
                                <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs">
                                  <p className="text-white font-medium">{d.name}</p>
                                  <p className="text-slate-400">Complexity: {d.x?.toFixed(4)}</p>
                                  <p className="text-slate-400">Entropy: {d.y?.toFixed(4)}</p>
                                </div>
                              );
                            }}
                          />
                          <Scatter data={scatterData} fill="#7c3aed" fillOpacity={0.7} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </>
                  )}
                </div>
              </div>
            )}
          </section>

          {/* ── Band Power Heatmap ───────────────────────────────────────── */}
          <section>
            <SectionHeader
              icon={<TrendingUp className="w-4 h-4" />}
              title="Spectral Power Map by Channel"
              expanded={expanded.bands}
              onToggle={() => toggle("bands")}
            />
            {expanded.bands && (
              <div className="mt-4 card p-4 overflow-x-auto">
                <p className="text-xs text-slate-500 mb-4">
                  Normalized band power per frequency band for each EEG channel
                </p>
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="text-left text-slate-500 font-medium pb-2 pr-4">Channel</th>
                      {["δ Delta", "θ Theta", "α Alpha", "β Beta", "γ Gamma"].map(b => (
                        <th key={b} className="text-slate-500 font-medium pb-2 px-2">{b}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(features.channelProfiles || []).map((ch, i) => {
                      const bp    = features.bandPowers?.[i] || {};
                      const bands = ["delta", "theta", "alpha", "beta", "gamma"];
                      const vals  = bands.map(b => Number(bp[b] ?? 0));
                      const maxV  = Math.max(...vals, 1e-10);
                      return (
                        <tr
                          key={i}
                          className={cn(
                            "border-t border-slate-800/50 cursor-pointer transition-colors",
                            selectedChannel === i ? "bg-accent-violet/10" : "hover:bg-slate-800/30"
                          )}
                          onClick={() => setSelectedChannel(i)}
                        >
                          <td className="py-1.5 pr-4 font-mono text-slate-300">
                            {ch.channel || `EEG${i + 1}`}
                          </td>
                          {vals.map((v, bi) => {
                            const pct = (v / maxV) * 100;
                            const col = Object.values(BAND_COLORS).filter(c => c !== BAND_COLORS.arousal)[bi];
                            return (
                              <td key={bi} className="py-1.5 px-2 text-center">
                                <div className="relative h-5 flex items-center justify-center">
                                  <div
                                    className="absolute inset-0 rounded"
                                    style={{ backgroundColor: col + "33", width: `${pct.toFixed(0)}%`, margin: "0 auto" }}
                                  />
                                  <span className="relative font-mono text-slate-300 text-[10px]">
                                    {v.toFixed(2)}
                                  </span>
                                </div>
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="text-xs text-slate-600 mt-3">
                  Click a row to update the radar and detail views above.
                </p>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
};

export default FeatureVisualizer;