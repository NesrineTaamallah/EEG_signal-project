import React from "react";
import { useStore } from "@/store/useStore";
import { Check, Loader2, FileUp, Filter, Zap, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";

const PipelineStepper: React.FC = () => {
  const { pipelineStep, isProcessing } = useStore();

  const steps = [
    { id: 0, label: "Signal Acquisition",   icon: FileUp,   desc: "Upload .MAT EEG data" },
    { id: 1, label: "Preprocessing",        icon: Filter,   desc: "ASR, ICA & Filtering" },
    { id: 2, label: "Feature Extraction",   icon: BarChart3, desc: "Time/Freq Domain Features" },
    { id: 3, label: "Classification",       icon: Zap,      desc: "Stress Level Prediction" },
  ];

  return (
    <div className="space-y-12">
      <header>
        <h2 className="text-3xl font-bold tracking-tight mb-2">Processing Pipeline</h2>
        <p className="text-slate-400">End-to-end neural signal processing workflow.</p>
      </header>

      <div className="card p-12">
        <div className="relative flex justify-between">
          {/* Progress Line */}
          <div className="absolute top-6 left-0 w-full h-0.5 bg-slate-800 -z-10" />
          <div
            className="absolute top-6 left-0 h-0.5 bg-accent-violet transition-all duration-500 -z-10"
            style={{ width: `${(pipelineStep / (steps.length - 1)) * 100}%` }}
          />

          {steps.map((step) => {
            const isActive    = pipelineStep === step.id;
            const isCompleted = pipelineStep > step.id;
            const Icon        = step.icon;

            return (
              <div key={step.id} className="flex flex-col items-center text-center w-48">
                <div
                  className={cn(
                    "w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all duration-300",
                    isCompleted
                      ? "bg-accent-violet border-accent-violet text-white"
                      : isActive
                      ? "bg-background border-accent-violet text-accent-violet shadow-[0_0_15px_rgba(124,58,237,0.3)]"
                      : "bg-background border-slate-800 text-slate-600"
                  )}
                >
                  {isCompleted ? (
                    <Check className="w-6 h-6" />
                  ) : isActive && isProcessing ? (
                    <Loader2 className="w-6 h-6 animate-spin" />
                  ) : (
                    <Icon className="w-6 h-6" />
                  )}
                </div>

                <div className="mt-4">
                  <div className={cn("font-bold text-sm mb-1", isActive ? "text-white" : "text-slate-500")}>
                    {step.label}
                  </div>
                  <p className="text-xs text-slate-600 max-w-[120px] mx-auto">{step.desc}</p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-24 grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="p-6 bg-slate-800/30 rounded-xl border border-slate-800">
            <h4 className="font-bold mb-4 flex items-center gap-2">
              <div className="w-2 h-2 bg-accent-violet rounded-full" />
              Pipeline Logic
            </h4>
            <ul className="space-y-3 text-sm text-slate-400">
              <li className="flex gap-2">
                <span className="text-accent-violet">•</span>
                MNE-based artifact removal (ASR)
              </li>
              <li className="flex gap-2">
                <span className="text-accent-violet">•</span>
                Multi-domain feature extraction (Hjorth, Entropy)
              </li>
              <li className="flex gap-2">
                <span className="text-accent-violet">•</span>
                Ensemble Voting Classifier (XGB + LGBM + Cat)
              </li>
            </ul>
          </div>

          <div className="p-6 bg-slate-800/30 rounded-xl border border-slate-800">
            <h4 className="font-bold mb-4 flex items-center gap-2">
              <div className="w-2 h-2 bg-accent-teal rounded-full" />
              Real-time Metrics
            </h4>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500 uppercase">Latency</span>
                <span className="text-sm font-mono text-accent-teal">12.4 ms</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500 uppercase">Throughput</span>
                <span className="text-sm font-mono text-accent-teal">256 Hz</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-500 uppercase">Buffer Status</span>
                <span className="text-sm font-mono text-accent-teal">OPTIMAL</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PipelineStepper;