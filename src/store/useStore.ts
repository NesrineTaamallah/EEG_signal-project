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

export interface ChannelBandPower {
  channel: string;
  delta: number;
  theta: number;
  alpha: number;
  beta: number;
  gamma: number;
}

export interface PredictionData {
  prediction: number;
  probabilities: { stress: number; non_stress: number };
  confidence: number;
  topFeatures: { name: string; importance: number }[];
  bandPowers: BandPower[];
  bandPowersPerCh?: ChannelBandPower[];
  model_source?: "trained_model" | "heuristic";
}

export interface ChannelProfile {
  channel: string;
  variance: number;
  rms: number;
  mobility: number;
  complexity: number;
  betaAlpha: number;
  thetaAlpha: number;
  entropy: number;
}

export interface TemporalPoint {
  window: number;
  delta: number;
  theta: number;
  alpha: number;
  beta: number;
  gamma: number;
  arousal: number;
}

export interface FeatureGroup {
  name: string;
  features: Record<string, number>;
  color: string;
}

export interface ExtractedFeaturesData {
  groups: FeatureGroup[];
  bandPowers: Record<string, number>[];
  statistics: {
    totalFeatures: number;
    channels: number;
    windows: number;
    samplingRate: number;
  };
  channelProfiles: ChannelProfile[];
  temporalEvolution: TemporalPoint[];
}

interface AppState {
  activeSignal: SignalData | null;
  prediction: PredictionData | null;
  extractedFeatures: ExtractedFeaturesData | null;
  pipelineStep: number;
  isProcessing: boolean;
  setActiveSignal: (signal: SignalData | null) => void;
  setPrediction: (prediction: PredictionData | null) => void;
  setExtractedFeatures: (features: ExtractedFeaturesData | null) => void;
  setPipelineStep: (step: number) => void;
  setIsProcessing: (isProcessing: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  activeSignal: null,
  prediction: null,
  extractedFeatures: null,
  pipelineStep: 0,
  isProcessing: false,
  setActiveSignal: (signal) => set({ activeSignal: signal }),
  setPrediction: (prediction) => set({ prediction }),
  setExtractedFeatures: (features) => set({ extractedFeatures: features }),
  setPipelineStep: (step) => set({ pipelineStep: step }),
  setIsProcessing: (isProcessing) => set({ isProcessing }),
}));