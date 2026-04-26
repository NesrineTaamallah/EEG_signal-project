import React, { useState, useMemo } from "react";
import Plot from "react-plotly.js";
import { ArrowLeftRight } from "lucide-react";
import { useStore } from "@/store/useStore";
import { cn } from "@/lib/utils";

const ComparativeAnalysis: React.FC = () => {
  const { activeSignal } = useStore();
  const [selectedChannel, setSelectedChannel] = useState(0);

  const hasCleanedSignal = useMemo(() => (
    activeSignal?.cleaned &&
    Array.isArray(activeSignal.cleaned) &&
    activeSignal.cleaned.length > 0 &&
    activeSignal.cleaned[0]?.length > 0
  ), [activeSignal]);

  if (!activeSignal) {
    return (
      <div className="flex flex-col items-center justify-center h-[500px] border-2 border-dashed border-slate-800 rounded-2xl">
        <ArrowLeftRight className="w-16 h-16 text-slate-700 mb-4" />
        <p className="text-slate-500 font-medium">Aucun signal chargé pour la comparaison.</p>
        <p className="text-sm text-slate-600 mt-1">Importez un signal dans l'onglet Analyseur.</p>
      </div>
    );
  }

  const time = Array.from(
    { length: activeSignal.raw[selectedChannel]?.length ?? 0 },
    (_, i) => i / activeSignal.sfreq
  );

  const rawData = activeSignal.raw[selectedChannel] ?? [];
  const cleanedData = hasCleanedSignal ? (activeSignal.cleaned[selectedChannel] ?? []) : [];

  // Compute simple stats for display
  const rawStd = rawData.length > 0
    ? Math.sqrt(rawData.reduce((s, v) => s + v * v, 0) / rawData.length) * 1e6
    : 0;
  const cleanStd = cleanedData.length > 0
    ? Math.sqrt(cleanedData.reduce((s, v) => s + v * v, 0) / cleanedData.length) * 1e6
    : 0;
  const noiseReduction = rawStd > 0 ? ((rawStd - cleanStd) / rawStd * 100) : 0;

  const plotData: any[] = [
    {
      x: time,
      y: rawData,
      type: "scattergl",
      mode: "lines",
      name: "Brut",
      line: { color: "#64748b", width: 1 },
    },
  ];

  if (hasCleanedSignal && cleanedData.length > 0) {
    plotData.push({
      x: time,
      y: cleanedData,
      type: "scattergl",
      mode: "lines",
      name: "Nettoyé (ASR + Filtre)",
      line: { color: "#0d9488", width: 1.5 },
    });
  }

  return (
    <div className="space-y-8">
      <header>
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-accent-teal/20 rounded-xl">
            <ArrowLeftRight className="w-6 h-6 text-accent-teal" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">Analyse Comparative</h2>
        </div>
        <p className="text-slate-400">Comparaison côte à côte du signal brut vs traité.</p>
      </header>

      {!hasCleanedSignal && (
        <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl text-amber-400 text-sm">
          ⚠ Signal nettoyé non disponible. Seul le signal brut sera affiché.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="card lg:col-span-1 space-y-6">
          <div>
            <label className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3 block">
              Canal analysé
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {activeSignal.channelNames.slice(0, 12).map((name, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedChannel(i)}
                  className={cn(
                    "px-2 py-1.5 rounded-lg text-xs font-mono transition-all",
                    selectedChannel === i
                      ? "bg-accent-teal text-white"
                      : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                  )}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>

          <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Métriques</h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">RMS Brut</span>
                <span className="text-xs font-mono text-slate-300">{rawStd.toFixed(1)} µV</span>
              </div>
              {hasCleanedSignal && (
                <>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-500">RMS Nettoyé</span>
                    <span className="text-xs font-mono text-accent-teal">{cleanStd.toFixed(1)} µV</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-500">Réduction bruit</span>
                    <span className={cn(
                      "text-xs font-mono",
                      noiseReduction > 0 ? "text-accent-teal" : "text-red-400"
                    )}>
                      {noiseReduction > 0 ? "+" : ""}{noiseReduction.toFixed(1)}%
                    </span>
                  </div>
                </>
              )}
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500">Durée</span>
                <span className="text-xs font-mono text-slate-300">
                  {(rawData.length / activeSignal.sfreq).toFixed(1)}s
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="card lg:col-span-3 p-2 overflow-hidden">
          <Plot
            data={plotData}
            layout={{
              autosize: true,
              height: 500,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { t: 40, b: 40, l: 60, r: 20 },
              xaxis: {
                title: "Temps (s)",
                color: "#64748b",
                gridcolor: "#1e293b",
                rangeslider: { visible: true },
              },
              yaxis: {
                title: "Amplitude (V)",
                color: "#64748b",
                gridcolor: "#1e293b",
              },
              legend: {
                font: { color: "#94a3b8" },
                bgcolor: "rgba(0,0,0,0)",
                orientation: "h",
                y: 1.1,
              },
              title: {
                text: `Comparaison Signal — ${activeSignal.channelNames[selectedChannel] ?? `EEG${selectedChannel + 1}`}`,
                font: { color: "#f1f5f9", size: 14 },
              },
            }}
            config={{ responsive: true, displaylogo: false }}
            className="w-full"
          />
        </div>
      </div>
    </div>
  );
};

export default ComparativeAnalysis;