import React, { useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getMetrics } from "@/services/api";
import { Activity, Target, TrendingUp, AlertCircle, Users, Clock, Brain, Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricsData {
  balanced_accuracy: { mean: number; std: number };
  roc_auc: { mean: number; std: number };
  confusion_matrix: number[][];
  fold_scores: number[];
}

// ── Stroop Test SVG Illustration ─────────────────────────────────────────────
const StroopIllustration: React.FC = () => (
  <svg viewBox="0 0 280 160" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
    <defs>
      <filter id="glow-stroop">
        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
        <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    {/* Background grid */}
    {[...Array(6)].map((_, i) => (
      <line key={i} x1={i * 47} y1="0" x2={i * 47} y2="160" stroke="#1e293b" strokeWidth="0.5"/>
    ))}
    {[...Array(4)].map((_, i) => (
      <line key={i} x1="0" y1={i * 40} x2="280" y2={i * 40} stroke="#1e293b" strokeWidth="0.5"/>
    ))}

    {/* Word cards with mismatched colors - the core Stroop effect */}
    <rect x="10" y="20" width="80" height="36" rx="8" fill="#1e293b" stroke="#ef4444" strokeWidth="1.5"/>
    <text x="50" y="43" textAnchor="middle" fill="#3b82f6" fontSize="18" fontWeight="bold" fontFamily="monospace">ROUGE</text>
    <text x="50" y="66" textAnchor="middle" fill="#ef4444" fontSize="9" fontFamily="monospace" opacity="0.7">ink: RED</text>

    <rect x="100" y="20" width="80" height="36" rx="8" fill="#1e293b" stroke="#3b82f6" strokeWidth="1.5"/>
    <text x="140" y="43" textAnchor="middle" fill="#10b981" fontSize="18" fontWeight="bold" fontFamily="monospace">BLEU</text>
    <text x="140" y="66" textAnchor="middle" fill="#3b82f6" fontSize="9" fontFamily="monospace" opacity="0.7">ink: BLUE</text>

    <rect x="190" y="20" width="80" height="36" rx="8" fill="#1e293b" stroke="#10b981" strokeWidth="1.5"/>
    <text x="230" y="43" textAnchor="middle" fill="#f59e0b" fontSize="18" fontWeight="bold" fontFamily="monospace">VERT</text>
    <text x="230" y="66" textAnchor="middle" fill="#10b981" fontSize="9" fontFamily="monospace" opacity="0.7">ink: GREEN</text>

    {/* Interference arrows */}
    <path d="M50 72 L50 90" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="3,2" opacity="0.6"/>
    <path d="M140 72 L140 90" stroke="#3b82f6" strokeWidth="1.5" strokeDasharray="3,2" opacity="0.6"/>
    <path d="M230 72 L230 90" stroke="#10b981" strokeWidth="1.5" strokeDasharray="3,2" opacity="0.6"/>

    {/* Brain response indicators */}
    <circle cx="50" cy="105" r="14" fill="none" stroke="#ef4444" strokeWidth="2" filter="url(#glow-stroop)"/>
    <text x="50" y="110" textAnchor="middle" fill="#ef4444" fontSize="10" fontFamily="monospace">✗</text>
    <circle cx="140" cy="105" r="14" fill="none" stroke="#3b82f6" strokeWidth="2" filter="url(#glow-stroop)"/>
    <text x="140" y="110" textAnchor="middle" fill="#3b82f6" fontSize="10" fontFamily="monospace">?</text>
    <circle cx="230" cy="105" r="14" fill="none" stroke="#10b981" strokeWidth="2" filter="url(#glow-stroop)"/>
    <text x="230" y="110" textAnchor="middle" fill="#10b981" fontSize="10" fontFamily="monospace">✓</text>

    {/* EEG wave representation */}
    <path d="M10,140 Q30,125 50,140 Q70,155 90,140 Q110,125 130,140 Q150,155 170,140 Q190,125 210,140 Q230,155 250,140 Q265,130 270,140"
          stroke="#7c3aed" strokeWidth="1.5" fill="none" opacity="0.8" filter="url(#glow-stroop)"/>
    <text x="140" y="157" textAnchor="middle" fill="#64748b" fontSize="8" fontFamily="monospace">COGNITIVE INTERFERENCE → EEG STRESS MARKER</text>
  </svg>
);

// ── Arithmetic Test SVG Illustration ─────────────────────────────────────────
const ArithmeticIllustration: React.FC = () => (
  <svg viewBox="0 0 280 160" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
    <defs>
      <filter id="glow-arith">
        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
        <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>

    {/* Math problem display */}
    <rect x="10" y="10" width="170" height="60" rx="10" fill="#0f172a" stroke="#f59e0b" strokeWidth="1.5"/>
    <text x="95" y="38" textAnchor="middle" fill="#f59e0b" fontSize="22" fontWeight="bold" fontFamily="monospace">847 × 23</text>
    <text x="95" y="58" textAnchor="middle" fill="#64748b" fontSize="10" fontFamily="monospace">= ???</text>

    {/* Timer countdown */}
    <circle cx="238" cy="40" r="28" fill="none" stroke="#ef4444" strokeWidth="2.5"/>
    <circle cx="238" cy="40" r="28" fill="none" stroke="#ef4444" strokeWidth="2.5"
            strokeDasharray="88 88" strokeDashoffset="55" opacity="0.3"/>
    <text x="238" y="36" textAnchor="middle" fill="#ef4444" fontSize="14" fontWeight="bold" fontFamily="monospace">5.2</text>
    <text x="238" y="50" textAnchor="middle" fill="#ef4444" fontSize="8" fontFamily="monospace">SEC</text>

    {/* Computation steps visualization */}
    <rect x="10" y="82" width="80" height="26" rx="5" fill="#1e293b" stroke="#0d9488" strokeWidth="1"/>
    <text x="50" y="99" textAnchor="middle" fill="#0d9488" fontSize="11" fontFamily="monospace">Step 1: 7×23</text>

    <rect x="100" y="82" width="80" height="26" rx="5" fill="#1e293b" stroke="#3b82f6" strokeWidth="1"/>
    <text x="140" y="99" textAnchor="middle" fill="#3b82f6" fontSize="11" fontFamily="monospace">Step 2: 40×23</text>

    <rect x="190" y="82" width="80" height="26" rx="5" fill="#1e293b" stroke="#7c3aed" strokeWidth="1"/>
    <text x="230" y="99" textAnchor="middle" fill="#7c3aed" fontSize="11" fontFamily="monospace">Step 3: Σ = ?</text>

    {/* Working memory load indicator */}
    <text x="10" y="128" fill="#64748b" fontSize="8" fontFamily="monospace">WORKING MEMORY LOAD</text>
    <rect x="10" y="132" width="200" height="8" rx="4" fill="#1e293b"/>
    <rect x="10" y="132" width="170" height="8" rx="4" fill="#f59e0b" opacity="0.8" filter="url(#glow-arith)"/>
    <text x="215" y="141" fill="#f59e0b" fontSize="8" fontFamily="monospace">85%</text>

    {/* Beta wave spike */}
    <path d="M10,155 L50,155 L55,145 L60,165 L65,140 L70,158 L75,152 L120,152 L125,145 L130,160 L135,148 L140,155 L270,155"
          stroke="#f59e0b" strokeWidth="1.5" fill="none" filter="url(#glow-arith)"/>
    <text x="140" y="159" textAnchor="middle" fill="#f59e0b" fontSize="7" fontFamily="monospace">β-BAND ELEVATION</text>
  </svg>
);

// ── Mirror Image Test SVG Illustration ───────────────────────────────────────
const MirrorIllustration: React.FC = () => (
  <svg viewBox="0 0 280 160" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
    <defs>
      <filter id="glow-mirror">
        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
        <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>

    {/* Mirror axis */}
    <line x1="140" y1="10" x2="140" y2="150" stroke="#7c3aed" strokeWidth="2" strokeDasharray="5,3" filter="url(#glow-mirror)"/>
    <text x="140" y="8" textAnchor="middle" fill="#7c3aed" fontSize="8" fontFamily="monospace">MIRROR AXIS</text>

    {/* Left shape - complex polygon */}
    <polygon points="30,30 70,25 85,55 75,80 45,85 20,65" fill="none" stroke="#0d9488" strokeWidth="2"/>
    <polygon points="40,40 60,37 68,58 60,70 42,73 30,58" fill="#0d9488" fillOpacity="0.15"/>
    <circle cx="45" cy="48" r="4" fill="#0d9488" opacity="0.7"/>
    <circle cx="65" cy="45" r="3" fill="#0d9488" opacity="0.7"/>
    <rect x="38" y="60" width="20" height="8" rx="2" fill="none" stroke="#0d9488" strokeWidth="1.5"/>

    {/* Right shape - mirrored (correct) */}
    <polygon points="250,30 210,25 195,55 205,80 235,85 260,65" fill="none" stroke="#3b82f6" strokeWidth="2"/>
    <polygon points="240,40 220,37 212,58 220,70 238,73 250,58" fill="#3b82f6" fillOpacity="0.15"/>
    <circle cx="235" cy="48" r="4" fill="#3b82f6" opacity="0.7"/>
    <circle cx="215" cy="45" r="3" fill="#3b82f6" opacity="0.7"/>
    <rect x="222" y="60" width="20" height="8" rx="2" fill="none" stroke="#3b82f6" strokeWidth="1.5"/>

    {/* Symmetric verification lines */}
    <line x1="85" y1="55" x2="195" y2="55" stroke="#64748b" strokeWidth="0.5" strokeDasharray="2,3" opacity="0.4"/>
    <line x1="70" y1="25" x2="210" y2="25" stroke="#64748b" strokeWidth="0.5" strokeDasharray="2,3" opacity="0.4"/>

    {/* Result badge */}
    <rect x="110" y="95" width="60" height="24" rx="12" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="1.5"/>
    <text x="140" y="111" textAnchor="middle" fill="#10b981" fontSize="11" fontFamily="monospace" fontWeight="bold">SYMMETRIC ✓</text>

    {/* Parieto-occipital activity */}
    <text x="10" y="140" fill="#64748b" fontSize="8" fontFamily="monospace">PARIETO-OCCIPITAL ACTIVATION</text>
    <path d="M10,148 Q35,138 60,148 Q85,158 110,148 Q135,138 160,148 Q185,158 210,148 Q235,138 260,148 L270,148"
          stroke="#a855f7" strokeWidth="1.5" fill="none" filter="url(#glow-mirror)"/>
    <text x="140" y="159" textAnchor="middle" fill="#a855f7" fontSize="7" fontFamily="monospace">VISUAL CORTEX THETA RESPONSE</text>
  </svg>
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

        {/* Test Cards with SVG Illustrations */}
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