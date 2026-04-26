import { create } from "zustand";

export interface SignalData {
  raw: number[][];
  cleaned: number[][];
  channelNames: string[];
  sfreq: number;
}

export interface BandPower {
  [band: string]: number;
}

export interface PredictionData {
  prediction: number;
  probabilities: { stress: number; non_stress: number };
  confidence: number;
  topFeatures: { name: string; importance: number }[];
  bandPowers: BandPower[];
  model_source?: "trained_model" | "heuristic";
}

interface AppState {
  activeSignal: SignalData | null;
  prediction: PredictionData | null;
  pipelineStep: number;
  isProcessing: boolean;
  setActiveSignal: (signal: SignalData | null) => void;
  setPrediction: (prediction: PredictionData | null) => void;
  setPipelineStep: (step: number) => void;
  setIsProcessing: (isProcessing: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  activeSignal: null,
  prediction: null,
  pipelineStep: 0,
  isProcessing: false,
  setActiveSignal: (signal) => set({ activeSignal: signal }),
  setPrediction: (prediction) => set({ prediction }),
  setPipelineStep: (step) => set({ pipelineStep: step }),
  setIsProcessing: (isProcessing) => set({ isProcessing }),
}));