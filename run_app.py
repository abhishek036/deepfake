"""
run_app.py
Unified launch script for realitycheck.fr Deepfake Detection Platform.

Starts the FastAPI backend service (serving the unified UI on port 8000)
and optionally launches the Vite dev server on port 5173 with proxying.

Usage:
    python run_app.py          # Starts FastAPI backend (serving built UI on http://localhost:8000)
    python run_app.py --dev    # Starts FastAPI backend + Vite dev server (http://localhost:5173)
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CORNERSTONE_DIR = ROOT_DIR / "cornerstone"
CORNERSTONE_DIST = CORNERSTONE_DIR / "dist"


def build_frontend():
    """Ensure the unified frontend in cornerstone is built."""
    if not (CORNERSTONE_DIST / "index.html").exists():
        print("[*] Building unified frontend in cornerstone...")
        subprocess.run(["npm", "run", "build"], cwd=str(CORNERSTONE_DIR), shell=True, check=True)
        print("[+] Frontend build completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Launch realitycheck.fr Deepfake Detection Suite")
    parser.add_argument("--dev", action="store_true", help="Launch Vite dev server alongside FastAPI")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    # Make sure frontend is built for production serving
    if not args.dev:
        build_frontend()

    processes = []
    try:
        if args.dev:
            print("[*] Starting Vite dev server on port 5173...")
            vite_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(CORNERSTONE_DIR),
                shell=True,
            )
            processes.append(vite_proc)
            time.sleep(2)

        target_url = "http://localhost:5173" if args.dev else f"http://localhost:{args.port}"
        print(f"\n==================================================================")
        print(f"  realitycheck.fr Deepfake Detection Platform Active!")
        print(f"  Unified UI (Landing + Detection + Camera + Results): {target_url}")
        print(f"  API Docs (Swagger): http://localhost:{args.port}/api/docs")
        print(f"==================================================================\n")

        if not args.no_browser:
            try:
                webbrowser.open(target_url)
            except Exception:
                pass

        # Run FastAPI
        import uvicorn
        from service.api import app

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    except KeyboardInterrupt:
        print("\n[*] Shutting down realitycheck.fr Platform...")
    finally:
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
