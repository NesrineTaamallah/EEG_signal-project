import React from "react";
import {
  Brain, Activity, BarChart3, Zap, Layers,
  ArrowLeftRight, Cpu
} from "lucide-react";
import { cn } from "@/lib/utils";

interface LayoutProps {
  children: React.ReactNode;
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const Layout: React.FC<LayoutProps> = ({ children, activeTab, setActiveTab }) => {
  const menuItems = [
    { id: "dashboard",  label: "Dashboard",        icon: BarChart3 },
    { id: "signal",     label: "Signal Analyzer",  icon: Activity },
    { id: "compare",    label: "Compare",           icon: ArrowLeftRight },
    { id: "features",   label: "Feature Explorer",  icon: Cpu },
    { id: "classify",   label: "Classification",    icon: Zap },
    { id: "brain",      label: "3D Brain Map",      icon: Brain },
    { id: "pipeline",   label: "Pipeline",          icon: Layers },
  ];

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 border-r border-slate-800 bg-surface flex flex-col">
        <div className="p-6 flex items-center gap-3 border-b border-slate-800">
          <div className="p-2 bg-accent-violet rounded-lg">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <h1 className="font-bold text-lg tracking-tight">
            NEURO<span className="text-accent-teal">CTRL</span>
          </h1>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={cn(
                "w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200 text-sm",
                activeTab === item.id
                  ? "bg-accent-violet/10 text-accent-violet border border-accent-violet/20"
                  : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              )}
            >
              <item.icon className="w-4 h-4 shrink-0" />
              <span className="font-medium">{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-800">
          <div className="p-4 bg-slate-800/50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-500 font-medium uppercase tracking-wider">
                System Status
              </span>
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            </div>
            <div className="text-sm font-mono text-accent-teal">ONLINE</div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-8">{children}</main>
    </div>
  );
};

export default Layout;