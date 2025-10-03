#!/usr/bin/env python3
"""
FrameSift Lite GUI Launcher
Simple launcher script that checks dependencies and starts the GUI
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'tkinter',
        'PIL',
        'matplotlib',
        'numpy',
        'cv2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages and try again.")
        return False
    
    return True

def check_modules():
    """Check if FrameSift modules are available"""
    try:
        from modules.video_processor import VideoProcessor
        from modules.results_generator import ResultsGenerator
        print("✓ FrameSift modules")
        return True
    except ImportError as e:
        print(f"✗ FrameSift modules - {e}")
        return False

def main():
    """Main launcher function"""
    print("FrameSift Lite GUI Launcher")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version}")
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check FrameSift modules
    print("\nChecking FrameSift modules...")
    if not check_modules():
        sys.exit(1)
    
    # Create results directory
    results_dir = Path("gui_results")
    results_dir.mkdir(exist_ok=True)
    print(f"✓ Results directory: {results_dir.absolute()}")
    
    print("\n🚀 Starting FrameSift Lite GUI...")
    
    # Import and start the GUI
    try:
        from framesift_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()