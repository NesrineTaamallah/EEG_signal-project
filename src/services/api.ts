import axios from "axios";

const api = axios.create({
  baseURL: "/api",
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API Error Details:", {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message,
      url: error.config?.url,
    });
    return Promise.reject(error);
  }
);

export const preprocessSignal = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post("/preprocess", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const predictStress = async (signal: number[][], sfreq: number) => {
  if (!signal || !Array.isArray(signal) || signal.length === 0) {
    throw new Error("Signal data is empty or invalid");
  }
  const response = await api.post("/predict", { signal, sfreq });
  return response.data;
};

export const extractFeatures = async (signal: number[][], sfreq: number) => {
  if (!signal || !Array.isArray(signal) || signal.length === 0) {
    throw new Error("Signal data is empty or invalid");
  }
  const response = await api.post("/extract-features", { signal, sfreq });
  return response.data;
};

export const getMetrics = async () => {
  const response = await api.get("/metrics");
  return response.data;
};

export default api;