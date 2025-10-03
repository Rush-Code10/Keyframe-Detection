# FrameSift Lite - Desktop Application

A powerful desktop keyframe extraction tool built with Python and Tkinter. Automatically extracts representative keyframes from videos while achieving **70-80% frame reduction** using three core Image and Video Processing (IVP) algorithms.

![Python](https://img.shields.io/badge/Python-3.7+-blue) ![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green) ![OpenCV](https://img.shields.io/badge/CV-OpenCV-red) ![License](https://img.shields.io/badge/License-Educational-orange)

## ✨ Key Features

- **🖥️ Desktop GUI**: Modern Tkinter interface with intuitive controls
- **🎯 Smart Keyframe Extraction**: Combines three IVP algorithms for optimal frame selection
- ** Visual Analytics**: Real-time processing plots and comprehensive statistics
- **🎬 Live Preview**: Keyframe gallery with thumbnail previews
- **⚡ High Performance**: Achieves 70-80% reduction while preserving key visual information
- **� Export Options**: Save results, export keyframes, and copy statistics
- **🔧 Parameter Presets**: Conservative, Balanced, and Aggressive processing modes

## 🧠 IVP Algorithms

| Algorithm | Purpose | Method |
|-----------|---------|--------|
| **Frame Difference** | Redundancy removal | Mean Absolute Difference (MAD) |
| **Histogram Comparison** | Scene boundary detection | Chi-Square distance |
| **Blur Detection** | Quality filtering | Variance of Laplacian |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd framesift-lite

# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Application

**Option 1: Using the launcher (Recommended)**
```bash
python launch_gui.py
```

**Option 2: Direct launch**
```bash
python framesift_gui.py
```

**Option 3: Windows batch file**
```bash
launch_gui.bat
```

### Basic Usage

1. **🎬 Upload Tab**: Select your video file using browse button or drag & drop
2. **⚙️ Parameters Tab**: Fine-tune algorithm settings or use presets
3. **📊 Results Tab**: View keyframes, statistics, and performance plots  
4. **📝 Logs Tab**: Monitor processing logs and debug information

## 📊 Example Results

Try the included example videos to see FrameSift Lite in action:

```bash
# Generate example videos (optional)
python create_example_videos.py
```

| Video Type | Demonstrates | Expected Reduction |
|------------|--------------|-------------------|
| `scene_transitions.mp4` | Shot boundary detection | ~75% |
| `motion_demo.mp4` | Motion analysis | ~70% |
| `quality_variation.mp4` | Blur filtering | ~80% |
| `mixed_content.mp4` | All techniques combined | ~75% |

## 🧪 Testing

```bash
# Unit tests for core algorithms
python -m pytest tests/ -v

# Test video processing with demo script
python demo_video_processor.py examples/motion_demo.mp4
```

## 📁 Project Structure

```
framesift-lite/
├── framesift_gui.py          # Main Tkinter GUI application
├── launch_gui.py             # Application launcher with dependency checks
├── launch_gui.bat            # Windows batch launcher
├── modules/                  # Core processing modules
│   ├── ivp_algorithms.py     # IVP algorithm implementations
│   ├── video_processor.py    # Video processing pipeline
│   └── results_generator.py  # Results formatting and export
├── tests/                    # Unit test suite
├── examples/                 # Example video files
├── gui_results/              # Generated results and thumbnails
├── demo_video_processor.py   # Command-line processing demo
├── create_example_videos.py  # Example video generator
└── requirements.txt          # Python dependencies
```

## 🔧 Parameter Tuning Guide

### Frame Difference Threshold (0.1-1.0)
- **Lower (0.1-0.3)**: More sensitive, keeps more frames with subtle changes
- **Higher (0.6-1.0)**: Less sensitive, removes more similar frames

### Histogram Threshold (0.1-1.0)
- **Lower (0.1-0.4)**: Detects only major scene changes
- **Higher (0.6-0.9)**: Detects subtle lighting/color changes

### Blur Threshold (10-200)
- **Lower (10-40)**: Accepts slightly blurry frames
- **Higher (70-200)**: Only keeps very sharp frames

### Presets Available
- **🎯 Conservative**: Higher thresholds, keeps more frames
- **⚖️ Balanced**: Default settings for most use cases
- **🚀 Aggressive**: Lower thresholds, maximum reduction

## 💾 Export Options

- **Save Results**: Export processing statistics as JSON
- **Export Keyframes**: Save keyframe images to folder
- **Copy Statistics**: Copy detailed stats to clipboard

## 🎨 GUI Features

- **Maximized Window**: Automatically opens in full-screen mode
- **150% Scaling**: All elements sized 50% larger for better visibility
- **Tabbed Interface**: Organized workflow across multiple tabs
- **Real-time Progress**: Live progress bar and status updates
- **Thumbnail Gallery**: Large preview images of selected keyframes
- **Interactive Plots**: Matplotlib-based algorithm performance visualization

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot open video file" | Check format support (MP4, AVI, MOV), try video conversion |
| "Processing failed" | Verify file size, disk space, try different parameters |
| Slow processing | Reduce video resolution, check system resources |
| No keyframes extracted | Lower blur threshold, adjust other parameters |
| GUI doesn't start | Check Python version (3.7+), install missing dependencies |
| Blank thumbnails | Verify video codec support, try different video file |

**Debug Information**: Check the Logs tab in the GUI for detailed error information and processing status.

## 📋 System Requirements

- **Python**: 3.7+ (Tkinter included)
- **OS**: Windows, macOS, Linux
- **RAM**: 4GB minimum, 8GB recommended for large videos
- **Storage**: 1GB free space for results and temporary files
- **Display**: 1920x1080 recommended for optimal GUI experience

### Dependencies
- **OpenCV**: Video processing and computer vision
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Plotting and data visualization
- **Pillow**: Image processing and thumbnail generation
- **Tkinter**: GUI framework (included with Python)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes focusing on the desktop application
4. Test with the included test suite
5. Submit a pull request
