import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { spawn } from "child_process";
import { createProxyMiddleware } from "http-proxy-middleware";
import fs from "fs";

const PYTHON_PORT = 8000;
const PORT = 3000;

// ── Read .env.local and inject into process.env ───────────────────────────
function loadEnvLocal(): Record<string, string> {
  const envPath = path.join(process.cwd(), ".env.local");
  const envVars: Record<string, string> = {};

  if (!fs.existsSync(envPath)) {
    console.log("[Server] No .env.local found — using system env only.");
    return envVars;
  }

  const lines = fs.readFileSync(envPath, "utf-8").split("\n");
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;          // skip blanks & comments
    const eqIdx = line.indexOf("=");
    if (eqIdx === -1) continue;
    const key = line.slice(0, eqIdx).trim();
    const val = line.slice(eqIdx + 1).trim()
      .replace(/^["']|["']$/g, "");                      // strip optional quotes
    if (key) {
      envVars[key] = val;
      process.env[key] = val;                            // also set in Node process
    }
  }

  const loaded = Object.keys(envVars);
  if (loaded.length > 0) {
    console.log(`[Server] Loaded from .env.local: ${loaded.join(", ")}`);
  }

  // Log model path specifically so user can confirm it is picked up
  if (envVars["NEUROSTRESS_MODEL_PATH"]) {
    const mp = envVars["NEUROSTRESS_MODEL_PATH"];
    const exists = fs.existsSync(mp);
    console.log(
      `[Server] NEUROSTRESS_MODEL_PATH → ${mp} ${exists ? "✓ file found" : "✗ FILE NOT FOUND — check path"}`
    );
  }

  return envVars;
}

// ── poll until the Python FastAPI server is ready ──────────────────────────
function waitForBackend(
  url: string,
  maxAttempts = 40,
  intervalMs = 500
): Promise<void> {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const check = () => {
      fetch(url)
        .then((r) => {
          if (r.ok) {
            console.log("[Server] Python backend is ready ✓");
            resolve();
          } else {
            retry();
          }
        })
        .catch(retry);
    };
    const retry = () => {
      attempts++;
      if (attempts >= maxAttempts) {
        reject(
          new Error(
            `Python backend did not start after ${maxAttempts * intervalMs}ms`
          )
        );
      } else {
        setTimeout(check, intervalMs);
      }
    };
    check();
  });
}

async function startServer() {
  const app = express();

  // ── Load .env.local → inject into Node + pass to Python ─────────────────
  const extraEnv = loadEnvLocal();

  // ── Start Python backend ─────────────────────────────────────────────────
  // main.py lives in  <project-root>/backend/main.py
  // uvicorn module:   backend.main:app   (requires backend/__init__.py)
  // cwd must be the project root so Python can resolve the module path.
  const cwd = process.cwd();
  console.log(`[Server] Starting Python backend (cwd: ${cwd}) …`);

  const pythonProcess = spawn(
      "C:\\Users\\nesri\\OneDrive\\Desktop\\signal\\env311\\Scripts\\python.exe",
      [
        "-m", "uvicorn", "backend.main:app",
        "--host", "127.0.0.1",
        "--port", String(PYTHON_PORT),
        "--log-level", "info",
      ],
    {
      cwd,
      env: { ...process.env, ...extraEnv },   // ← .env.local vars injected here
    }
  );

  pythonProcess.stdout.on("data", (data: Buffer) => {
    process.stdout.write(`[Python] ${data}`);
  });

  pythonProcess.stderr.on("data", (data: Buffer) => {
    const msg = data.toString();
    process.stderr.write(`[Python] ${msg}`);

    if (msg.includes("No module named uvicorn")) {
      console.error(
        "\n[Server] CRITICAL: uvicorn not found.\n" +
          "  Run: pip install -r requirements.txt\n"
      );
    }
    if (msg.includes("No module named backend")) {
      console.error(
        "\n[Server] CRITICAL: backend package not found.\n" +
          "  Make sure backend/__init__.py and backend/main.py exist in the project root.\n"
      );
    }
  });

  pythonProcess.on("close", (code: number | null) => {
    console.log(`[Python] Process exited with code ${code}`);
  });

  // ── Wait for FastAPI to be healthy ───────────────────────────────────────
  try {
    await waitForBackend(`http://127.0.0.1:${PYTHON_PORT}/health`);
  } catch (e) {
    console.error("[Server] Warning: Python backend health check timed out.");
    console.error("[Server] API calls will be proxied anyway; check Python logs.");
  }

  // ── Proxy  /api/*  →  http://127.0.0.1:8000/* ───────────────────────────
  app.use(
    "/api",
    createProxyMiddleware({
      target: `http://127.0.0.1:${PYTHON_PORT}`,
      changeOrigin: true,
      pathRewrite: { "^/api": "" },   // /api/preprocess → /preprocess
      on: {
        proxyReq: (_proxyReq: any, req: any) => {
          console.log(`[Proxy] ${req.method} ${req.url}`);
        },
        error: (err: Error, req: any, res: any) => {
          console.error("[Proxy Error]:", err.message);

          // Graceful fallback only for /metrics
          if (
            String(req.url) === "/metrics" ||
            String(req.url) === "/api/metrics"
          ) {
            res.json({
              balanced_accuracy: { mean: 0.6347, std: 0.0304 },
              roc_auc:           { mean: 0.6721, std: 0.0412 },
              confusion_matrix:  [[45, 15], [12, 48]],
              fold_scores:       [0.61, 0.65, 0.62, 0.66, 0.63],
              is_mock:           true,
            });
            return;
          }

          res.status(502).json({
            error: "Backend unavailable",
            message:
              "The Python neural engine is not responding. " +
              "Check that uvicorn started correctly (see server logs).",
            details: err.message,
          });
        },
      },
    })
  );

  // ── Vite dev middleware ──────────────────────────────────────────────────
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req: any, res: any) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`\n[Server] Frontend + proxy running → http://localhost:${PORT}`);
    console.log(`[Server] Python API              → http://127.0.0.1:${PYTHON_PORT}`);
  });
}

startServer().catch((err) => {
  console.error("[Server] Fatal error:", err);
  process.exit(1);
});