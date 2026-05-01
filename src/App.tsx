/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Layout from "./components/Layout";
import Dashboard from "./components/Dashboard";
import SignalAnalyzer from "./components/SignalAnalyzer";
import ClassificationDashboard from "./components/ClassificationDashboard";
import Brain3D from "./components/Brain3D";
import PipelineStepper from "./components/PipelineStepper";
import NeuroAssistant from "./components/NeuroAssistant";
import ComparativeAnalysis from "./components/ComparativeAnalysis";
import FeatureVisualizer from "./components/FeatureVisualizer";
import { motion, AnimatePresence } from "motion/react";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
});

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");

  const renderContent = () => {
    switch (activeTab) {
      case "dashboard":  return <Dashboard />;
      case "signal":     return <SignalAnalyzer />;
      case "compare":    return <ComparativeAnalysis />;
      case "features":   return <FeatureVisualizer />;
      case "classify":   return <ClassificationDashboard />;
      case "brain":      return <Brain3D />;
      case "pipeline":   return <PipelineStepper />;
      case "assistant":  return <NeuroAssistant />;
      default:           return <Dashboard />;
    }
  };

  return (
    <QueryClientProvider client={queryClient}>
      <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="h-full"
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </Layout>
    </QueryClientProvider>
  );
}