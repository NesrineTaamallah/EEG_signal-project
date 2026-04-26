export interface FrequencyBands {
  delta: number;
  theta: number;
  alpha: number;
  beta: number;
  gamma: number;
}

export const ELECTRODE_POSITIONS = [
  { id: "Fp1", name: "Fp1", region: "Frontal", x: -0.3, y: 0.9, z: 0.2 },
  { id: "Fp2", name: "Fp2", region: "Frontal", x: 0.3, y: 0.9, z: 0.2 },
  { id: "F7", name: "F7", region: "Frontal", x: -0.8, y: 0.5, z: 0.1 },
  { id: "F3", name: "F3", region: "Frontal", x: -0.4, y: 0.5, z: 0.6 },
  { id: "Fz", name: "Fz", region: "Frontal", x: 0, y: 0.5, z: 0.8 },
  { id: "F4", name: "F4", region: "Frontal", x: 0.4, y: 0.5, z: 0.6 },
  { id: "F8", name: "F8", region: "Frontal", x: 0.8, y: 0.5, z: 0.1 },
  { id: "FT9", name: "FT9", region: "Temporal", x: -0.9, y: 0.2, z: -0.2 },
  { id: "FC5", name: "FC5", region: "Frontal", x: -0.6, y: 0.3, z: 0.4 },
  { id: "FC1", name: "FC1", region: "Frontal", x: -0.2, y: 0.3, z: 0.8 },
  { id: "FC2", name: "FC2", region: "Frontal", x: 0.2, y: 0.3, z: 0.8 },
  { id: "FC6", name: "FC6", region: "Frontal", x: 0.6, y: 0.3, z: 0.4 },
  { id: "FT10", name: "FT10", region: "Temporal", x: 0.9, y: 0.2, z: -0.2 },
  { id: "T7", name: "T7", region: "Temporal", x: -1.0, y: 0, z: 0 },
  { id: "C3", name: "C3", region: "Central", x: -0.6, y: 0, z: 0.7 },
  { id: "Cz", name: "Cz", region: "Central", x: 0, y: 0, z: 1.0 },
  { id: "C4", name: "C4", region: "Central", x: 0.6, y: 0, z: 0.7 },
  { id: "T8", name: "T8", region: "Temporal", x: 1.0, y: 0, z: 0 },
  { id: "CP5", name: "CP5", region: "Central", x: -0.6, y: -0.3, z: 0.4 },
  { id: "CP1", name: "CP1", region: "Central", x: -0.2, y: -0.3, z: 0.8 },
  { id: "CP2", name: "CP2", region: "Central", x: 0.2, y: -0.3, z: 0.8 },
  { id: "CP6", name: "CP6", region: "Central", x: 0.6, y: -0.3, z: 0.4 },
  { id: "P7", name: "P7", region: "Parietal", x: -0.8, y: -0.5, z: 0.1 },
  { id: "P3", name: "P3", region: "Parietal", x: -0.4, y: -0.5, z: 0.6 },
  { id: "Pz", name: "Pz", region: "Parietal", x: 0, y: -0.5, z: 0.8 },
  { id: "P4", name: "P4", region: "Parietal", x: 0.4, y: -0.5, z: 0.6 },
  { id: "P8", name: "P8", region: "Parietal", x: 0.8, y: -0.5, z: 0.1 },
  { id: "PO9", name: "PO9", region: "Occipital", x: -0.5, y: -0.8, z: -0.1 },
  { id: "O1", name: "O1", region: "Occipital", x: -0.3, y: -0.9, z: 0.2 },
  { id: "Oz", name: "Oz", region: "Occipital", x: 0, y: -1.0, z: 0.1 },
  { id: "O2", name: "O2", region: "Occipital", x: 0.3, y: -0.9, z: 0.2 },
  { id: "PO10", name: "PO10", region: "Occipital", x: 0.5, y: -0.8, z: -0.1 },
];
