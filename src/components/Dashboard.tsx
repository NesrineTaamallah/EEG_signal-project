import React from "react";
import { useQuery } from "@tanstack/react-query";
import { getMetrics } from "@/services/api";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from "recharts";
import { Activity, Target, TrendingUp, AlertCircle, Users, Clock, Brain, Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricsData {
  balanced_accuracy: { mean: number; std: number };
  roc_auc: { mean: number; std: number };
  confusion_matrix: number[][];
  fold_scores: number[];
}

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

  const foldData = (data?.fold_scores || []).map((score: number, i: number) => ({
    name: `Fold ${i + 1}`,
    score: score * 100,
  }));

  const COLORS = ["#7c3aed", "#0d9488", "#3b82f6", "#f59e0b"];

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

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              title: "Stroop Test (SCWT)",
              desc: "Cognitive interference task where subjects identify ink colors of words. Induces mental stress through incongruent conditions.",
              color: "border-accent-violet/30 bg-accent-violet/5"
            },
            {
              title: "Arithmetic Task",
              desc: "Mental calculation under pressure. Subjects solve arithmetic problems and indicate correctness within strict time limits.",
              color: "border-accent-teal/30 bg-accent-teal/5"
            },
            {
              title: "Mirror Image",
              desc: "Visual symmetry recognition. Subjects decide if pairs of mirror images are symmetric or asymmetric under pressure.",
              color: "border-blue-500/30 bg-blue-500/5"
            }
          ].map((task, i) => (
            <div key={i} className={cn("p-6 rounded-2xl border", task.color)}>
              <h4 className="font-bold text-white mb-2">{task.title}</h4>
              <p className="text-sm text-slate-400 leading-relaxed">{task.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Fold Scores */}
        <div className="card">
          <h3 className="text-lg font-bold mb-6">Accuracy per Fold</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={foldData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="name" stroke="#64748b" />
                <YAxis stroke="#64748b" domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#111827", border: "1px solid #1e293b" }}
                  itemStyle={{ color: "#7c3aed" }}
                />
                <Bar dataKey="score" fill="#7c3aed" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix Heatmap (Simulated with Grid) */}
        <div className="card">
          <h3 className="text-lg font-bold mb-6">Confusion Matrix</h3>
          <div className="grid grid-cols-2 gap-4 h-80">
            {data.confusion_matrix.flat().map((val: number, i: number) => (
              <div 
                key={i}
                className={cn(
                  "flex flex-col items-center justify-center rounded-xl border border-slate-800",
                  i === 0 || i === 3 ? "bg-accent-teal/20 border-accent-teal/30" : "bg-red-500/10 border-red-500/20"
                )}
              >
                <div className="text-3xl font-bold monospace">{val}</div>
                <div className="text-xs text-slate-500 uppercase tracking-widest mt-2">
                  {i === 0 ? "True Negative" : i === 1 ? "False Positive" : i === 2 ? "False Negative" : "True Positive"}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
