import React, { useState } from "react";
import Plot from "react-plotly.js";
import { useStore } from "@/store/useStore";
import { preprocessSignal } from "@/services/api";
import { Upload, Activity, CheckCircle, Loader2, AlertCircle, Waves } from "lucide-react";
import { cn } from "@/lib/utils";

const SignalAnalyzer: React.FC = () => {
  const { activeSignal, setActiveSignal, setPipelineStep, isProcessing, setIsProcessing } = useStore();
  const [selectedChannel, setSelectedChannel] = useState(0);
  const [showCleaned, setShowCleaned] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError(null);
    setFileName(file.name);
    setIsProcessing(true);
    setSelectedChannel(0);
    setShowCleaned(false);

    try {
      const data = await preprocessSignal(file);

      // Validate the response structure
      if (!data.raw_signal || !Array.isArray(data.raw_signal) || data.raw_signal.length === 0) {
        throw new Error("Le backend n'a pas retourné de signal valide");
      }

      if (!data.cleaned_signal || !Array.isArray(data.cleaned_signal) || data.cleaned_signal.length === 0) {
        throw new Error("Le preprocessing n'a pas produit de signal nettoyé");
      }

      setActiveSignal({
        raw: data.raw_signal,
        cleaned: data.cleaned_signal,
        channelNames: data.channel_names || data.raw_signal.map((_: any, i: number) => `EEG${i + 1}`),
        sfreq: data.sfreq || 256,
      });
      setPipelineStep(1);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || "Erreur inconnue lors du preprocessing";
      setError(msg);
      console.error("Preprocessing failed:", err);
    } finally {
      setIsProcessing(false);
      // Reset input so same file can be re-uploaded
      e.target.value = "";
    }
  };

  const getPlotData = () => {
    if (!activeSignal) return [];

    const signal = showCleaned && activeSignal.cleaned?.length > 0
      ? activeSignal.cleaned
      : activeSignal.raw;

    if (!signal[selectedChannel]) return [];

    const time = Array.from(
      { length: signal[selectedChannel].length },
      (_, i) => i / activeSignal.sfreq
    );

    const traces: any[] = [];

    // Always show raw as reference
    if (activeSignal.raw[selectedChannel]) {
      traces.push({
        x: time,
        y: activeSignal.raw[selectedChannel],
        type: "scattergl",
        mode: "lines",
        name: "Signal Brut",
        line: { color: "#64748b", width: 1 },
        opacity: showCleaned ? 0.4 : 1,
      });
    }

    // Show cleaned on top when toggled
    if (showCleaned && activeSignal.cleaned?.[selectedChannel]) {
      traces.push({
        x: time,
        y: activeSignal.cleaned[selectedChannel],
        type: "scattergl",
        mode: "lines",
        name: "Signal Nettoyé",
        line: { color: "#0d9488", width: 1.5 },
      });
    }

    return traces;
  };

  const hasCleanedSignal = activeSignal?.cleaned && activeSignal.cleaned.length > 0 &&
    activeSignal.cleaned[0] && activeSignal.cleaned[0].length > 0;

  return (
    <div className="space-y-8">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight mb-2">Analyseur de Signal</h2>
          <p className="text-slate-400">Importez et inspectez les signaux EEG avec prétraitement avancé.</p>
        </div>

        <label className={cn(
          "btn-primary flex items-center gap-2 cursor-pointer",
          isProcessing && "opacity-50 pointer-events-none"
        )}>
          {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
          <span>{isProcessing ? "Traitement en cours..." : "Importer fichier .MAT"}</span>
          <input
            type="file"
            className="hidden"
            accept=".mat"
            onChange={handleFileUpload}
            disabled={isProcessing}
          />
        </label>
      </header>

      {/* Error display */}
      {error && (
        <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-red-400 font-medium text-sm">Erreur de prétraitement</p>
            <p className="text-red-400/70 text-xs mt-1">{error}</p>
          </div>
        </div>
      )}

      {!activeSignal && !isProcessing ? (
        <div className="card h-96 flex flex-col items-center justify-center border-dashed border-2 border-slate-800 bg-slate-900/30">
          <div className="p-4 bg-slate-800/50 rounded-full mb-4">
            <Waves className="w-12 h-12 text-slate-600" />
          </div>
          <p className="text-slate-500 font-medium">Aucun signal chargé.</p>
          <p className="text-sm text-slate-600 mt-1">Importez un fichier .MAT pour commencer l'analyse.</p>
          <p className="text-xs text-slate-700 mt-4">Formats supportés: MATLAB .mat (données EEG 2D)</p>
        </div>
      ) : isProcessing ? (
        <div className="card h-96 flex flex-col items-center justify-center gap-4">
          <Loader2 className="w-12 h-12 text-accent-violet animate-spin" />
          <div className="text-center">
            <p className="text-white font-medium">Prétraitement en cours...</p>
            <p className="text-slate-400 text-sm mt-1">{fileName}</p>
            <p className="text-slate-500 text-xs mt-2">Filtrage notch 50Hz, passe-bande 1-40Hz, re-référencement</p>
          </div>
        </div>
      ) : activeSignal && (
        <div className="space-y-6">
          {/* Status banner */}
          <div className="flex items-center gap-3 p-3 bg-accent-teal/10 border border-accent-teal/20 rounded-xl">
            <CheckCircle className="w-5 h-5 text-accent-teal shrink-0" />
            <div className="flex-1 min-w-0">
              <span className="text-accent-teal font-medium text-sm">Signal chargé avec succès</span>
              <span className="text-slate-400 text-xs ml-3">
                {activeSignal.channelNames.length} canaux · {activeSignal.sfreq}Hz ·{" "}
                {(activeSignal.raw[0]?.length / activeSignal.sfreq).toFixed(1)}s
                {hasCleanedSignal && " · Signal nettoyé disponible"}
              </span>
            </div>
            {fileName && <span className="text-slate-500 text-xs truncate max-w-48">{fileName}</span>}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Controls */}
            <div className="card lg:col-span-1 space-y-6 flex flex-col" style={{ maxHeight: "600px" }}>
              <div className="flex-1 overflow-hidden flex flex-col">
                <label className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3 block">
                  Sélection du canal
                </label>
                <div className="grid grid-cols-2 gap-1.5 overflow-y-auto pr-1 flex-1">
                  {activeSignal.channelNames.map((name, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedChannel(i)}
                      className={cn(
                        "px-2 py-1.5 rounded-lg text-xs font-mono transition-all",
                        selectedChannel === i
                          ? "bg-accent-violet text-white"
                          : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                      )}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="pt-4 border-t border-slate-800 space-y-3">
                <div
                  className={cn(
                    "flex items-center justify-between cursor-pointer group",
                    !hasCleanedSignal && "opacity-40 pointer-events-none"
                  )}
                  onClick={() => hasCleanedSignal && setShowCleaned(!showCleaned)}
                >
                  <div>
                    <span className="text-sm font-medium text-slate-300 group-hover:text-white transition-colors block">
                      Signal Nettoyé
                    </span>
                    {!hasCleanedSignal && (
                      <span className="text-xs text-slate-600">Non disponible</span>
                    )}
                  </div>
                  <div className={cn(
                    "w-10 h-5 rounded-full p-0.5 transition-colors",
                    showCleaned && hasCleanedSignal ? "bg-accent-teal" : "bg-slate-700"
                  )}>
                    <div className={cn(
                      "w-4 h-4 bg-white rounded-full transition-transform",
                      showCleaned && hasCleanedSignal ? "translate-x-5" : "translate-x-0"
                    )} />
                  </div>
                </div>

                <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                  <p className="text-xs text-slate-400 leading-relaxed">
                    <span className="text-accent-teal font-medium">Pipeline actif:</span> Notch 50Hz,
                    Passe-bande 1–40Hz, Référence moyenne
                  </p>
                </div>
              </div>
            </div>

            {/* Plot */}
            <div className="card lg:col-span-3 p-2 overflow-hidden">
              <Plot
                data={getPlotData() as any}
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
                    zerolinecolor: "#1e293b",
                    rangeslider: { visible: true },
                  },
                  yaxis: {
                    title: "Amplitude (V)",
                    color: "#64748b",
                    gridcolor: "#1e293b",
                    zerolinecolor: "#1e293b",
                    fixedrange: false,
                  },
                  legend: {
                    font: { color: "#94a3b8" },
                    bgcolor: "rgba(0,0,0,0)",
                    orientation: "h",
                    y: 1.1,
                  },
                  title: {
                    text: `Canal: ${activeSignal.channelNames[selectedChannel] ?? `EEG${selectedChannel + 1}`}`,
                    font: { color: "#f1f5f9", size: 14 },
                  },
                }}
                config={{
                  responsive: true,
                  displayModeBar: true,
                  modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"],
                  displaylogo: false,
                }}
                className="w-full"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SignalAnalyzer;