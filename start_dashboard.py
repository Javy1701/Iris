#!/usr/bin/env python3
"""
Startup script for Iris Dashboard System

This script helps start the FastAPI backend and Streamlit dashboard
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def check_port(port):
    """Check if a port is in use."""
    try:
        response = requests.get(f"http://localhost:{port}/docs", timeout=2)
        return True
    except:
        return False

def start_backend():
    """Start the FastAPI backend."""
    print("🚀 Starting FastAPI backend...")
    
    if check_port(8000):
        print("✅ FastAPI backend is already running on port 8000")
        return None
    
    try:
        # Start FastAPI in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for backend to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_port(8000):
                print("✅ FastAPI backend started successfully!")
                return process
            time.sleep(1)
        
        print("❌ Backend failed to start within 30 seconds")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_dashboard():
    """Start the Streamlit dashboard."""
    print("🎨 Starting Streamlit dashboard...")
    
    if check_port(8502):
        print("✅ Dashboard is already running on port 8502")
        return
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/dashboard_app.py", 
            "--server.port", "8502",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

def main():
    """Main startup function."""
    print("🎨 Iris Dashboard System Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("app/main.py").exists():
        print("❌ Error: Please run this script from the Iris_backend directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Start backend
    backend_process = start_backend()
    
    if backend_process is None and not check_port(8000):
        print("❌ Cannot start dashboard without backend")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("✅ System ready!")
    print("📊 Dashboard: http://localhost:8502")
    print("🔧 API Docs: http://localhost:8000/docs")
    print("💬 Chat Interface: http://localhost:8501")
    print("=" * 40)
    print("\nPress Ctrl+C to stop the dashboard")
    
    try:
        # Start dashboard (this will block)
        start_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        if backend_process:
            print("🛑 Stopping backend...")
            backend_process.terminate()
            backend_process.wait()

if __name__ == "__main__":
    main()
