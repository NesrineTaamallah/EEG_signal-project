import React, { useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getMetrics } from "@/services/api";
import { Activity, Target, TrendingUp, AlertCircle, Users, Clock, Brain, Info } from "lucide-react";
import { cn } from "@/lib/utils";

// ── Image imports (place files in src/assets/) ────────────────────────────────
import stroopImg  from "@/assets/stroop.jpg";
import mathImg    from "@/assets/math.png";
import miroirImg  from "@/assets/miroir.png";

interface MetricsData {
  balanced_accuracy: { mean: number; std: number };
  roc_auc: { mean: number; std: number };
  confusion_matrix: number[][];
  fold_scores: number[];
}

// ── Illustration components ───────────────────────────────────────────────────
const StroopIllustration: React.FC = () => (
  <img
    src={stroopImg}
    alt="Stroop Test illustration"
    className="w-full h-full object-cover rounded-lg"
  />
);

const ArithmeticIllustration: React.FC = () => (
  <img
    src={mathImg}
    alt="Arithmetic Task illustration"
    className="w-full h-full object-cover rounded-lg"
  />
);

const MirrorIllustration: React.FC = () => (
  <img
    src={miroirImg}
    alt="Mirror Image Test illustration"
    className="w-full h-full object-cover rounded-lg"
  />
);

const Dashboard: React.FC = () => {
  const { data, isLoading, error } = useQuery<MetricsData>({
    queryKey: ["metrics"],
    queryFn: getMetrics,
  });

  if (isLoading) return <div className="animate-pulse space-y-4">
    <div className="h-32 bg-slate-800 rounded-xl" />
    <div className="grid grid-cols-2 gap-4">
      <div className="h-64 bg-slate-800 rounded-xl" />
      <div className="h-64 bg-slate-800 rounded-xl" />
    </div>
  </div>;

  if (error || !data) return (
    <div className="flex flex-col items-center justify-center h-[500px] border-2 border-dashed border-red-500/20 rounded-2xl bg-red-500/5 p-8 text-center">
      <div className="p-4 bg-red-500/20 rounded-full mb-4">
        <AlertCircle className="w-12 h-12 text-red-500" />
      </div>
      <h3 className="text-xl font-bold text-white mb-2">Backend Connection Failed</h3>
      <p className="text-slate-400 max-w-md mb-6">
        We couldn't connect to the neural processing engine. This might be due to the server starting up or a network issue.
      </p>
      <button
        onClick={() => window.location.reload()}
        className="btn-primary flex items-center gap-2"
      >
        <Activity className="w-4 h-4" />
        <span>Retry Connection</span>
      </button>
    </div>
  );

  return (
    <div className="space-y-8">
      <header>
        <h2 className="text-3xl font-bold tracking-tight mb-2">System Performance</h2>
        <p className="text-slate-400">Cross-validation metrics and model accuracy reports.</p>
      </header>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card flex items-center gap-4">
          <div className="p-3 bg-accent-violet/10 rounded-lg">
            <Target className="w-6 h-6 text-accent-violet" />
          </div>
          <div>
            <div className="text-sm text-slate-500 font-medium">Balanced Accuracy</div>
            <div className="text-2xl font-bold monospace">
              {(data.balanced_accuracy.mean * 100).toFixed(2)}%
              <span className="text-sm text-slate-500 ml-2">±{(data.balanced_accuracy.std * 100).toFixed(1)}</span>
            </div>
          </div>
        </div>

        <div className="card flex items-center gap-4">
          <div className="p-3 bg-accent-teal/10 rounded-lg">
            <TrendingUp className="w-6 h-6 text-accent-teal" />
          </div>
          <div>
            <div className="text-sm text-slate-500 font-medium">ROC-AUC Score</div>
            <div className="text-2xl font-bold monospace">
              {(data.roc_auc.mean * 100).toFixed(2)}
            </div>
          </div>
        </div>

        <div className="card flex items-center gap-4">
          <div className="p-3 bg-blue-500/10 rounded-lg">
            <Activity className="w-6 h-6 text-blue-500" />
          </div>
          <div>
            <div className="text-sm text-slate-500 font-medium">Model Status</div>
            <div className="text-2xl font-bold text-green-500 uppercase tracking-tighter">Optimized</div>
          </div>
        </div>
      </div>

      {/* SAM Dataset Overview */}
      <section className="space-y-6">
        <div className="flex items-center gap-2">
          <Info className="w-5 h-5 text-accent-teal" />
          <h3 className="text-xl font-bold">SAM Dataset Overview</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl">
            <div className="flex items-center gap-3 mb-2">
              <Users className="w-4 h-4 text-slate-400" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Participants</span>
            </div>
            <div className="text-xl font-bold">40 Subjects</div>
            <div className="text-xs text-slate-500 mt-1">14 Females | 26 Males</div>
          </div>

          <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl">
            <div className="flex items-center gap-3 mb-2">
              <Clock className="w-4 h-4 text-slate-400" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Age Range</span>
            </div>
            <div className="text-xl font-bold">Mean: 21.5 yrs</div>
            <div className="text-xs text-slate-500 mt-1">Standardized demographic</div>
          </div>

          <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl">
            <div className="flex items-center gap-3 mb-2">
              <Brain className="w-4 h-4 text-slate-400" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">EEG Config</span>
            </div>
            <div className="text-xl font-bold">32 Channels</div>
            <div className="text-xs text-slate-500 mt-1">128 Hz Sampling Rate</div>
          </div>

          <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl">
            <div className="flex items-center gap-3 mb-2">
              <TrendingUp className="w-4 h-4 text-slate-400" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Trial Length</span>
            </div>
            <div className="text-xl font-bold">25 Seconds</div>
            <div className="text-xs text-slate-500 mt-1">3 trials per condition</div>
          </div>
        </div>

        {/* Test Cards with Image Illustrations */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Stroop Test */}
          <div className={cn("rounded-2xl border overflow-hidden", "border-accent-violet/30 bg-accent-violet/5")}>
            <div className="p-4 pb-0">
              <h4 className="font-bold text-white mb-1">Stroop Test (SCWT)</h4>
              <p className="text-sm text-slate-400 leading-relaxed mb-3">
                Cognitive interference task where subjects identify ink colors of words. Induces mental stress through incongruent conditions.
              </p>
            </div>
            <div className="px-4 pb-2 h-44">
              <StroopIllustration />
            </div>
            <div className="px-4 pb-4 flex gap-2 flex-wrap">
              <span className="text-xs bg-accent-violet/20 text-accent-violet px-2 py-0.5 rounded-full border border-accent-violet/30">Frontal β↑</span>
              <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">Interference</span>
              <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">3 trials</span>
            </div>
          </div>

          {/* Arithmetic Test */}
          <div className={cn("rounded-2xl border overflow-hidden", "border-accent-teal/30 bg-accent-teal/5")}>
            <div className="p-4 pb-0">
              <h4 className="font-bold text-white mb-1">Arithmetic Task</h4>
              <p className="text-sm text-slate-400 leading-relaxed mb-3">
                Mental calculation under pressure. Subjects solve arithmetic problems and indicate correctness within strict time limits.
              </p>
            </div>
            <div className="px-4 pb-2 h-44">
              <ArithmeticIllustration />
            </div>
            <div className="px-4 pb-4 flex gap-2 flex-wrap">
              <span className="text-xs bg-accent-teal/20 text-accent-teal px-2 py-0.5 rounded-full border border-accent-teal/30">WM Load↑</span>
              <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">Timed</span>
              <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">3 trials</span>
            </div>
          </div>

          {/* Mirror Image Test */}
          <div className={cn("rounded-2xl border overflow-hidden", "border-blue-500/30 bg-blue-500/5")}>
            <div className="p-4 pb-0">
              <h4 className="font-bold text-white mb-1">Mirror Image</h4>
              <p className="text-sm text-slate-400 leading-relaxed mb-3">
                Visual symmetry recognition. Subjects decide if pairs of mirror images are symmetric or asymmetric under pressure.
              </p>
            </div>
            <div className="px-4 pb-2 h-44">
              <MirrorIllustration />
            </div>
            <div className="px-4 pb-4 flex gap-2 flex-wrap">
              <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded-full border border-blue-500/30">Parietal θ↑</span>
              <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">Spatial</span>
              <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">3 trials</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Dashboard;