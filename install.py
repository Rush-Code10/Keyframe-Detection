#!/usr/bin/env python3
"""
FrameSift Lite Installation Script
Automated setup for the desktop application
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   FrameSift Lite requires Python 3.7 or higher")
        return False

def main():
    """Main installation function"""
    print("🚀 FrameSift Lite Desktop Application Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not os.path.exists('framesift_gui.py'):
        print("❌ Please run this script from the FrameSift Lite directory")
        sys.exit(1)
    
    print("✅ Found FrameSift Lite files")
    
    # Create virtual environment (optional but recommended)
    create_venv = input("\n🔧 Create virtual environment? (recommended) [Y/n]: ").strip().lower()
    if create_venv in ['', 'y', 'yes']:
        if run_command("python -m venv venv", "Creating virtual environment"):
            # Activate virtual environment
            if os.name == 'nt':  # Windows
                activate_cmd = "venv\\Scripts\\activate && "
            else:  # Unix/Linux/Mac
                activate_cmd = "source venv/bin/activate && "
            print("📝 Virtual environment created. To activate it manually:")
            if os.name == 'nt':
                print("   venv\\Scripts\\activate")
            else:
                print("   source venv/bin/activate")
        else:
            print("⚠️  Continuing without virtual environment")
            activate_cmd = ""
    else:
        activate_cmd = ""
    
    # Install dependencies
    install_cmd = f"{activate_cmd}pip install -r requirements.txt"
    if not run_command(install_cmd, "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create results directory
    results_dir = Path("gui_results")
    results_dir.mkdir(exist_ok=True)
    print("✅ Created results directory")
    
    # Test import of key modules
    print("\n🧪 Testing module imports...")
    test_imports = [
        "tkinter",
        "cv2",
        "numpy",
        "matplotlib",
        "PIL"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("   Please check the installation and try again")
        sys.exit(1)
    
    # Test FrameSift modules
    try:
        from modules.video_processor import VideoProcessor
        from modules.results_generator import ResultsGenerator
        from modules.ivp_algorithms import FrameDifferenceCalculator
        print("✅ FrameSift modules")
    except ImportError as e:
        print(f"❌ FrameSift modules: {e}")
        sys.exit(1)
    
    print("\n🎉 Installation completed successfully!")
    print("\n🚀 To launch FrameSift Lite:")
    print("   python launch_gui.py")
    print("   or")
    print("   python framesift_gui.py")
    
    if os.name == 'nt':  # Windows
        print("   or double-click: launch_gui.bat")
    
    # Ask if user wants to launch now
    launch_now = input("\n🎬 Launch FrameSift Lite now? [Y/n]: ").strip().lower()
    if launch_now in ['', 'y', 'yes']:
        print("\n🚀 Launching FrameSift Lite...")
        try:
            if activate_cmd:
                os.system(f"{activate_cmd}python launch_gui.py")
            else:
                os.system("python launch_gui.py")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()