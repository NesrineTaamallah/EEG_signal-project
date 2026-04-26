import React, { useState } from "react";
import { useStore } from "@/store/useStore";
import { predictStress } from "@/services/api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Zap, ShieldCheck, AlertTriangle, Loader2, Info, FileText, Brain } from "lucide-react";
import { cn } from "@/lib/utils";

const BAND_COLORS: Record<string, string> = {
  delta: "#3b82f6",
  theta: "#06b6d4",
  alpha: "#10b981",
  beta: "#f59e0b",
  gamma: "#a855f7",
};

const ClassificationDashboard: React.FC = () => {
  const { activeSignal, prediction, setPrediction, isProcessing, setIsProcessing, setPipelineStep } = useStore();
  const [classifyError, setClassifyError] = useState<string | null>(null);

  const handleGenerateReport = () => {
    if (!prediction) return;

    const report = `
NEURO-STRESS ANALYSIS REPORT
============================
Status: ${prediction.prediction === 1 ? "STRESS DETECTED" : "NORMAL STATE"}
Confidence: ${(prediction.confidence * 100).toFixed(1)}%
Stress Probability: ${(prediction.probabilities.stress * 100).toFixed(1)}%

NEURAL BAND ANALYSIS
-------------------
${prediction.bandPowers?.map((b) => {
  const [name, val] = Object.entries(b)[0];
  return `${name.toUpperCase()}: ${Number(val).toFixed(4)} (normalized)`;
}).join("\n") || "N/A"}

TOP NEURAL FEATURES
------------------
${prediction.topFeatures?.map((f) => `- ${f.name}: ${(f.importance * 100).toFixed(1)}% importance`).join("\n") || "N/A"}

SYSTEM METADATA
--------------
Signal Source: SAM Dataset
Channels: ${activeSignal?.channelNames.length || "N/A"}
Sampling Rate: ${activeSignal?.sfreq || "N/A"} Hz
Timestamp: ${new Date().toLocaleString()}
    `.trim();

    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `neuro_report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClassify = async () => {
    if (!activeSignal) return;
    setClassifyError(null);

    // Use cleaned signal if available, fall back to raw
    const signalToUse = activeSignal.cleaned?.length > 0 && activeSignal.cleaned[0]?.length > 0
      ? activeSignal.cleaned
      : activeSignal.raw;

    if (!signalToUse || signalToUse.length === 0) {
      setClassifyError("Aucun signal disponible pour la classification.");
      return;
    }

    setIsProcessing(true);
    try {
      const data = await predictStress(signalToUse, activeSignal.sfreq);

      setPrediction({
        prediction: data.prediction ?? 0,
        probabilities: {
          stress: data.probabilities?.stress ?? 0.5,
          non_stress: data.probabilities?.non_stress ?? 0.5,
        },
        confidence: data.confidence ?? 0.5,
        topFeatures: data.topFeatures ?? [],
        bandPowers: data.bandPowers ?? [],
        model_source: data.model_source ?? "heuristic",
      });
      setPipelineStep(3);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || "Erreur lors de la classification";
      setClassifyError(msg);
      console.error("Classification failed:", err);
    } finally {
      setIsProcessing(false);
    }
  };

  const stressPercent = prediction ? (prediction.probabilities.stress * 100) : 0;
  const isStress = prediction?.prediction === 1;

  if (!activeSignal) {
    return (
      <div className="card h-96 flex flex-col items-center justify-center border-dashed border-2 border-slate-800">
        <Brain className="w-12 h-12 text-slate-600 mb-4" />
        <p className="text-slate-500 font-medium">Aucun signal disponible pour la classification.</p>
        <p className="text-sm text-slate-600 mt-1">Importez et traitez un signal dans l'onglet Analyseur.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight mb-2">Moteur de Classification</h2>
          <p className="text-slate-400">Détection de stress pilotée par ML et analyse d'importance des features.</p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleGenerateReport}
            disabled={!prediction}
            className="btn-secondary flex items-center gap-2"
          >
            <FileText className="w-4 h-4" />
            <span>Rapport</span>
          </button>

          <button
            onClick={handleClassify}
            disabled={isProcessing}
            className="btn-primary flex items-center gap-2"
          >
            {isProcessing
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Zap className="w-4 h-4" />}
            <span>{isProcessing ? "Analyse..." : "Lancer l'inférence"}</span>
          </button>
        </div>
      </header>

      {classifyError && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
          {classifyError}
        </div>
      )}

      {/* Signal info */}
      <div className="p-3 bg-slate-800/40 rounded-xl border border-slate-700 flex flex-wrap gap-4 text-xs">
        <span className="text-slate-400">
          <span className="text-slate-300 font-medium">Signal:</span>{" "}
          {activeSignal.cleaned?.length > 0 ? "✓ Nettoyé" : "⚠ Brut (nettoyage non effectué)"}
        </span>
        <span className="text-slate-400">
          <span className="text-slate-300 font-medium">Canaux:</span> {activeSignal.channelNames.length}
        </span>
        <span className="text-slate-400">
          <span className="text-slate-300 font-medium">Fréq:</span> {activeSignal.sfreq}Hz
        </span>
        <span className="text-slate-400">
          <span className="text-slate-300 font-medium">Durée:</span>{" "}
          {activeSignal.raw[0] ? (activeSignal.raw[0].length / activeSignal.sfreq).toFixed(1) : "?"}s
        </span>
      </div>

      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Result */}
          <div className="lg:col-span-1 space-y-6">
            <div className={cn(
              "card flex flex-col items-center text-center p-8",
              isStress ? "border-red-500/30 bg-red-500/5" : "border-green-500/30 bg-green-500/5"
            )}>
              <div className={cn(
                "p-4 rounded-full mb-6",
                isStress ? "bg-red-500/20 text-red-500" : "bg-green-500/20 text-green-500"
              )}>
                {isStress
                  ? <AlertTriangle className="w-12 h-12" />
                  : <ShieldCheck className="w-12 h-12" />}
              </div>

              <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">État Détecté</h3>
              <div className={cn(
                "text-3xl font-black tracking-tighter mb-4",
                isStress ? "text-red-500" : "text-green-500"
              )}>
                {isStress ? "STRESS DÉTECTÉ" : "ÉTAT NORMAL"}
              </div>

              <div className="w-full space-y-2">
                <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden">
                  <div
                    className={cn(
                      "h-full transition-all duration-1000 rounded-full",
                      isStress ? "bg-red-500" : "bg-green-500"
                    )}
                    style={{ width: `${Math.round(stressPercent)}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs font-mono text-slate-500">
                  <span>NORMAL {Math.round((1 - prediction.probabilities.stress) * 100)}%</span>
                  <span>STRESS {Math.round(stressPercent)}%</span>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center gap-2 mb-3">
                <Info className="w-4 h-4 text-accent-teal" />
                <h4 className="font-bold text-sm">Score de confiance</h4>
              </div>
              <div className="text-4xl font-bold font-mono text-accent-teal">
                {Math.round(prediction.confidence * 100)}%
              </div>
              <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                Basé sur l'ensemble VotingClassifier (XGBoost + LightGBM + CatBoost).
              </p>
            </div>

            {/* Model source indicator */}
            <div className={cn(
              "card p-4 border",
              prediction.model_source === "trained_model"
                ? "border-accent-violet/40 bg-accent-violet/5"
                : "border-amber-500/40 bg-amber-500/5"
            )}>
              <div className="flex items-center gap-2 mb-1">
                <div className={cn(
                  "w-2 h-2 rounded-full animate-pulse",
                  prediction.model_source === "trained_model" ? "bg-accent-violet" : "bg-amber-500"
                )} />
                <span className={cn(
                  "text-xs font-bold uppercase tracking-wider",
                  prediction.model_source === "trained_model" ? "text-accent-violet" : "text-amber-500"
                )}>
                  {prediction.model_source === "trained_model" ? "Modèle ML entraîné" : "Mode heuristique"}
                </span>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                {prediction.model_source === "trained_model"
                  ? "Prédiction via votre modèle XGB+LGBM+CatBoost entraîné."
                  : "Modèle non trouvé. Résultats basés sur les ratios Beta/Alpha, Theta/Alpha et l'indice d'éveil — marqueurs EEG validés. Pour activer le ML, définissez NEUROSTRESS_MODEL_PATH dans votre .env."}
              </p>
            </div>
          </div>

          {/* Features & Bands */}
          <div className="lg:col-span-2 space-y-8">
            {/* Feature Importances */}
            {prediction.topFeatures && prediction.topFeatures.length > 0 && (
              <div className="card">
                <h3 className="text-lg font-bold mb-6">Top Features Importances</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={prediction.topFeatures.slice(0, 8)} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                      <XAxis
                        type="number"
                        stroke="#64748b"
                        tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                        fontSize={10}
                      />
                      <YAxis
                        dataKey="name"
                        type="category"
                        stroke="#64748b"
                        width={140}
                        fontSize={10}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#111827", border: "1px solid #1e293b" }}
                        formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "Importance"]}
                      />
                      <Bar dataKey="importance" fill="#7c3aed" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Band Powers */}
            {prediction.bandPowers && prediction.bandPowers.length > 0 && (
              <div className="card">
                <h3 className="text-lg font-bold mb-6">Puissances par Bande de Fréquence</h3>
                <div className="grid grid-cols-5 gap-4">
                  {prediction.bandPowers.map((bp, i) => {
                    const entries = Object.entries(bp);
                    if (entries.length === 0) return null;
                    const [band, rawValue] = entries[0];
                    const value = typeof rawValue === "number" ? rawValue : 0;
                    // value is normalized 0-10 from backend, convert to 0-100% for display
                    const displayPct = Math.min(Math.max((value / 10) * 100, 2), 100);
                    const color = BAND_COLORS[band] ?? "#64748b";

                    return (
                      <div key={i} className="flex flex-col items-center gap-2">
                        <div className="w-full bg-slate-800 h-32 rounded-lg relative overflow-hidden">
                          <div
                            className="absolute bottom-0 w-full transition-all duration-1000 rounded-b-lg"
                            style={{
                              height: `${Math.round(displayPct)}%`,
                              backgroundColor: color + "66",
                              borderTop: `2px solid ${color}`,
                            }}
                          />
                          <div className="absolute bottom-2 w-full text-center">
                            <span className="text-xs font-mono font-bold" style={{ color }}>
                              {value.toFixed(2)}
                            </span>
                          </div>
                        </div>
                        <span className="text-xs font-bold uppercase text-slate-400">{band}</span>
                        <span className="text-xs text-slate-600">{Math.round(displayPct)}%</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {!prediction && !isProcessing && (
        <div className="card h-64 flex flex-col items-center justify-center border-dashed border-slate-700">
          <Zap className="w-10 h-10 text-slate-700 mb-3" />
          <p className="text-slate-500">Cliquez sur "Lancer l'inférence" pour classifier le signal EEG.</p>
        </div>
      )}
    </div>
  );
};

export default ClassificationDashboard;