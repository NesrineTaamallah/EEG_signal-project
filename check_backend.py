#!/usr/bin/env python3
"""
check_backend.py  — run from the project root to diagnose backend issues.

Usage:
    python3 check_backend.py
"""
import sys, os, subprocess, importlib

ROOT = os.path.dirname(os.path.abspath(__file__))
print("=" * 60)
print("NeuroStress Backend Diagnostic")
print("=" * 60)
print(f"Project root : {ROOT}")
print(f"Python       : {sys.executable}  ({sys.version.split()[0]})")

errors = []

# 1. Check backend package
backend_init = os.path.join(ROOT, "backend", "__init__.py")
backend_main = os.path.join(ROOT, "backend", "main.py")

print("\n── File structure ──")
for label, fpath in [
    ("backend/__init__.py", backend_init),
    ("backend/main.py",     backend_main),
]:
    exists = os.path.isfile(fpath)
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status}  {label}")
    if not exists:
        errors.append(f"Missing file: {fpath}")

# 2. Check Python packages
print("\n── Python packages ──")
packages = {
    "fastapi":            "fastapi",
    "uvicorn":            "uvicorn",
    "numpy":              "numpy",
    "scipy":              "scipy",
    "python-multipart":   "multipart",
}
for display, module in packages.items():
    try:
        importlib.import_module(module)
        print(f"  ✓  {display}")
    except ImportError:
        print(f"  ✗  {display}  ← NOT INSTALLED")
        errors.append(f"Missing package: {display}")

# optional but recommended
print("\n── Optional packages ──")
for display, module in [("mne", "mne"), ("joblib", "joblib")]:
    try:
        importlib.import_module(module)
        print(f"  ✓  {display}")
    except ImportError:
        print(f"  ⚠  {display}  (not installed — scipy fallback will be used)")

# 3. Try importing the app
print("\n── Import backend.main ──")
sys.path.insert(0, ROOT)
try:
    from backend.main import app  # noqa: F401
    print("  ✓  from backend.main import app  — OK")
except Exception as exc:
    print(f"  ✗  Import failed: {exc}")
    errors.append(f"Import error: {exc}")

# 4. Summary
print("\n" + "=" * 60)
if errors:
    print("PROBLEMS FOUND:")
    for e in errors:
        print(f"  • {e}")
    print("\nFix:")
    print("  1. Copy output/backend/ into your project root.")
    print("  2. pip install -r requirements.txt")
else:
    print("All checks passed — backend should start correctly.")
    print("\nStart command (from project root):")
    print("  npm run dev")
    print("  — or manually —")
    print("  python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
print("=" * 60)