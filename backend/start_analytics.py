#!/usr/bin/env python3
"""
Oreja Speaker Analytics Dashboard Starter
Easy launcher for the speaker analytics GUI with dependency checking.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_and_install_dependencies():
    """Check and install required dependencies for the analytics dashboard"""
    required_packages = [
        'matplotlib',
        'numpy', 
        'pandas',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please install manually using: pip install matplotlib numpy pandas requests")
            return False
    
    return True

def check_tkinter():
    """Check if tkinter is available"""
    try:
        import tkinter
        print("✅ tkinter is available")
        return True
    except ImportError:
        print("❌ tkinter is not available")
        print("Please install tkinter for your Python distribution:")
        print("  Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  CentOS/RHEL: sudo yum install tkinter")
        print("  Windows/Mac: tkinter should be included with Python")
        return False

def main():
    """Main startup function"""
    print("🎤 Oreja Speaker Analytics Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("speaker_analytics_gui.py").exists():
        print("❌ speaker_analytics_gui.py not found!")
        print("Please run this script from the backend directory.")
        return 1
    
    # Check tkinter first (most common issue)
    if not check_tkinter():
        return 1
    
    # Check and install other dependencies
    if not check_and_install_dependencies():
        return 1
    
    # Launch the analytics dashboard
    print("\n🚀 Launching Speaker Analytics Dashboard...")
    try:
        import speaker_analytics_gui
        speaker_analytics_gui.main()
    except Exception as e:
        print(f"❌ Failed to start analytics dashboard: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the backend server is running (python server.py)")
        print("2. Check that the speaker database exists")
        print("3. Verify all dependencies are installed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 