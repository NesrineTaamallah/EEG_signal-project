import React, { useState, useRef, useEffect } from "react";
import { GoogleGenAI } from "@google/genai";
import { Send, Bot, User, Loader2, Sparkles } from "lucide-react";
import { useStore } from "@/store/useStore";
import { cn } from "@/lib/utils";

const NeuroAssistant: React.FC = () => {
  const { prediction, activeSignal } = useStore();
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([
    {
      role: "assistant",
      content: "Hello! I am your Neuro-Assistant. I can help you interpret your EEG data, explain stress patterns, or answer questions about the SAM dataset. How can I assist you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setInput("");
    setIsLoading(true);

    try {
      const apiKey = (process.env.GEMINI_API_KEY || (import.meta as any).env?.VITE_GEMINI_API_KEY) as string;
      
      if (!apiKey || apiKey === "undefined") {
        throw new Error("API_KEY_MISSING");
      }

      const ai = new GoogleGenAI({ apiKey });
      
      // Prepare context based on current data
      let context = "Context: The user is currently analyzing EEG data for stress classification.\n";
      if (prediction) {
        context += `Current Prediction: ${prediction.prediction === 1 ? "Stress Detected" : "No Stress Detected"}\n`;
        context += `Stress Probability: ${(prediction.probabilities.stress * 100).toFixed(1)}%\n`;
        context += `Dominant Band Powers: ${JSON.stringify(prediction.bandPowers)}\n`;
      }
      if (activeSignal) {
        context += `Signal Info: ${activeSignal.channelNames.length} channels, ${activeSignal.sfreq}Hz sampling rate.\n`;
      }

      const prompt = `${context}\nUser Question: ${userMessage}\n\nPlease provide a helpful, neuroscientific explanation or answer. Keep it professional but accessible.`;

      const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: prompt,
        config: {
          systemInstruction: "You are a specialized Neuro-Assistant for an EEG stress classification platform. You help users understand complex brain signal data and the SAM dataset. Use technical terms correctly but explain them if they are obscure.",
        },
      });

      setMessages((prev) => [...prev, { role: "assistant", content: response.text || "I'm sorry, I couldn't generate a response." }]);
    } catch (error: any) {
      console.error("Gemini API error:", error);
      let errorMsg = "Error: Failed to connect to the neural engine. Please check your API configuration.";
      if (error.message === "API_KEY_MISSING") {
        errorMsg = "Error: Gemini API Key is missing. Please provide it in the settings menu to enable the Neuro-Assistant.";
      } else if (error.message?.includes("401")) {
        errorMsg = "Error: Unauthorized access (401). Your API key might be invalid or expired.";
      }
      setMessages((prev) => [...prev, { role: "assistant", content: errorMsg }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      <header className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-accent-violet/20 rounded-xl">
            <Sparkles className="w-6 h-6 text-accent-violet" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">Neuro-Assistant</h2>
        </div>
        <p className="text-slate-400">AI-powered insights and explanations for your neural data.</p>
      </header>

      <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden flex flex-col min-h-[500px]">
        {/* Messages Area */}
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-slate-800"
        >
          {(messages || []).map((msg, i) => (
            <div 
              key={i} 
              className={cn(
                "flex gap-4 max-w-[85%]",
                msg.role === "user" ? "ml-auto flex-row-reverse" : ""
              )}
            >
              <div className={cn(
                "w-10 h-10 rounded-xl flex items-center justify-center shrink-0",
                msg.role === "assistant" ? "bg-accent-violet/20 text-accent-violet" : "bg-slate-800 text-slate-400"
              )}>
                {msg.role === "assistant" ? <Bot className="w-6 h-6" /> : <User className="w-6 h-6" />}
              </div>
              <div className={cn(
                "p-4 rounded-2xl text-sm leading-relaxed",
                msg.role === "assistant" ? "bg-slate-800/50 text-slate-200" : "bg-accent-violet text-white"
              )}>
                {msg.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-4 max-w-[85%]">
              <div className="w-10 h-10 rounded-xl bg-accent-violet/20 text-accent-violet flex items-center justify-center animate-pulse">
                <Bot className="w-6 h-6" />
              </div>
              <div className="bg-slate-800/50 p-4 rounded-2xl flex items-center gap-3">
                <Loader2 className="w-4 h-4 animate-spin text-accent-violet" />
                <span className="text-sm text-slate-400 italic">Analyzing neural patterns...</span>
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 bg-slate-900 border-t border-slate-800">
          <div className="relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Ask about your EEG data or the SAM dataset..."
              className="w-full bg-slate-800 border-none rounded-xl py-4 pl-6 pr-14 text-white placeholder:text-slate-500 focus:ring-2 focus:ring-accent-violet transition-all"
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              className="absolute right-2 top-2 bottom-2 px-4 bg-accent-violet hover:bg-accent-violet/80 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-all text-white"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <p className="text-[10px] text-slate-600 mt-2 text-center uppercase tracking-widest font-bold">
            Powered by Gemini 2.0 Flash Neural Engine
          </p>
        </div>
      </div>
    </div>
  );
};

export default NeuroAssistant;
