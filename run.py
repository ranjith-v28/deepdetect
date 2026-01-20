"""
Quick Installation and Run Script
Execute this to get started quickly.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("🚀 AI Deepfake Detection System - Quick Start")
    print("=" * 70)
    print()
    
    # Step 1: Check location
    print("📍 Step 1: Verifying location...")
    cwd = Path.cwd()
    if cwd.name != 'deepfake_detection':
        print(f"⚠️  Current directory: {cwd}")
        print("   Please run this from the deepfake_detection directory")
        return
    print(f"✓ Location: {cwd}")
    print()
    
    # Step 2: Install dependencies
    print("📦 Step 2: Installing dependencies...")
    response = input("   Install required packages? (y/n): ").lower().strip()
    if response == 'y':
        print("   Installing... (this may take a few minutes)")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'
            ])
            print("✓ Dependencies installed")
        except:
            print("⚠️  Installation encountered issues. Try manually:")
            print("   pip install -r requirements.txt")
    print()
    
    # Step 3: Create demo models
    print("🎯 Step 3: Setting up demo models...")
    response = input("   Create dummy models for testing? (y/n): ").lower().strip()
    if response == 'y':
        try:
            subprocess.check_call([sys.executable, 'demo.py'])
        except:
            print("⚠️  Run manually: python demo.py")
    print()
    
    # Step 4: Launch application
    print("🌐 Step 4: Launching web application...")
    print()
    print("=" * 70)
    print("To start the application, run:")
    print()
    print("    streamlit run app.py")
    print()
    print("Then open your browser to: http://localhost:8501")
    print("=" * 70)
    print()
    
    response = input("Launch now? (y/n): ").lower().strip()
    if response == 'y':
        print("\n🚀 Starting Streamlit server...\n")
        try:
            subprocess.check_call([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
        except KeyboardInterrupt:
            print("\n\n👋 Application stopped.")
        except:
            print("\n⚠️  Could not start. Run manually: streamlit run app.py")


if __name__ == '__main__':
    main()
