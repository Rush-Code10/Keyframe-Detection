# FrameSift Lite - Desktop Keyframe Extraction Tool

A powerful desktop application for automatic video keyframe extraction using advanced Image and Video Processing (IVP) algorithms. Built with Python and Tkinter for a seamless user experience.

![Python](https://img.shields.io/badge/Python-3.7+-blue) ![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green) ![OpenCV](https://img.shields.io/badge/CV-OpenCV-red) ![License](https://img.shields.io/badge/License-Educational-orange)

## Overview

FrameSift Lite automatically identifies and extracts the most important frames from videos while achieving **70-80% frame reduction**. The application uses three complementary computer vision algorithms to intelligently select representative keyframes that capture essential visual information while discarding redundant frames.

## Key Features

### Core Functionality
- **Desktop GUI**: Modern Tkinter interface with intuitive controls
- **Smart Keyframe Extraction**: Combines three IVP algorithms for optimal frame selection
- **Visual Analytics**: Real-time processing plots and comprehensive statistics
- **Live Preview**: Keyframe gallery with thumbnail previews
- **High Performance**: Achieves 70-80% reduction while preserving key visual information
- **Export Options**: Save results, export keyframes, and copy statistics
- **Parameter Presets**: Conservative, Balanced, and Aggressive processing modes

### User Interface
- **Maximized Window**: Automatically opens in full-screen mode
- **150% Scaling**: All elements sized 50% larger for better visibility
- **Tabbed Interface**: Organized workflow across multiple tabs
- **Real-time Progress**: Live progress bar and status updates
- **Thumbnail Gallery**: Large preview images of selected keyframes
- **Interactive Plots**: Matplotlib-based algorithm performance visualization

## IVP Algorithms

| Algorithm | Purpose | Method |
|-----------|---------|--------|
| **Frame Difference** | Redundancy removal | Mean Absolute Difference (MAD) |
| **Histogram Comparison** | Scene boundary detection | Chi-Square distance |
| **Blur Detection** | Quality filtering | Variance of Laplacian |

### Processing Pipeline
1. **Video Analysis** - Extract frames and compute algorithm scores
2. **Multi-Criteria Scoring** - Combine difference, histogram, and blur metrics
3. **Intelligent Selection** - Apply thresholds and selection strategies
4. **Quality Filtering** - Remove blurry or low-quality frames
5. **Result Generation** - Create thumbnails, statistics, and visualizations

## Installation

### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd "Keyframe Detection"

# Create virtual environment (recommended)
python -m venv venv

# Windows activation:
venv\Scripts\activate

# Linux/Mac activation:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Automated Installation
```bash
# Run automated installation script
python install.py
```

### System Requirements
- **Python**: 3.7+ (Tkinter included)
- **Operating System**: Windows, macOS, Linux
- **Memory**: 4GB minimum, 8GB recommended for large videos
- **Storage**: 1GB free space for results and temporary files
- **Display**: 1920x1080 recommended for optimal GUI experience

## Usage

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

### Basic Workflow

1. **Upload Tab**: Select your video file using browse button or drag & drop
2. **Parameters Tab**: Fine-tune algorithm settings or use presets
3. **Results Tab**: View keyframes, statistics, and performance plots
4. **Logs Tab**: Monitor processing logs and debug information

### Parameter Configuration

#### Algorithm Thresholds

**Frame Difference Threshold (0.1-1.0)**
- Lower (0.1-0.3): More sensitive, keeps more frames with subtle changes
- Higher (0.6-1.0): Less sensitive, removes more similar frames

**Histogram Threshold (0.1-1.0)**
- Lower (0.1-0.4): Detects only major scene changes
- Higher (0.6-0.9): Detects subtle lighting/color changes

**Blur Threshold (10-200)**
- Lower (10-40): Accepts slightly blurry frames
- Higher (70-200): Only keeps very sharp frames

#### Available Presets
- **Conservative**: Higher thresholds, keeps more frames
- **Balanced**: Default settings for most use cases
- **Aggressive**: Lower thresholds, maximum reduction

## Example Results

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

## Export Options

- **Save Results**: Export processing statistics as JSON
- **Export Keyframes**: Save keyframe images to folder
- **Copy Statistics**: Copy detailed stats to clipboard

## Project Structure

```
framesift-lite/
├── framesift_gui.py          # Main Tkinter GUI application
├── launch_gui.py             # Application launcher with dependency checks
├── launch_gui.bat            # Windows batch launcher
├── install.py                # Automated installation script
├── demo_video_processor.py   # Command-line processing demo
├── create_example_videos.py  # Example video generator
│
├── modules/                  # Core processing modules
│   ├── ivp_algorithms.py     # IVP algorithm implementations
│   ├── video_processor.py    # Video processing pipeline
│   └── results_generator.py  # Results formatting and export
│
├── tests/                    # Unit test suite
│   ├── test_ivp_algorithms.py      # Algorithm unit tests
│   ├── test_video_processor.py     # Processing pipeline tests
│   └── test_results_generator.py   # Results generation tests
│
├── examples/                 # Example video files
├── gui_results/             # Generated results and thumbnails
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── ALGORITHMS.md           # Detailed algorithm documentation
└── .gitignore             # Git ignore rules
```

## Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_ivp_algorithms.py -v
python -m pytest tests/test_video_processor.py -v
python -m pytest tests/test_results_generator.py -v

# Run with coverage report
python -m pytest tests/ --cov=modules --cov-report=html
```

### Command Line Demo
```bash
# Test video processing with demo script
python demo_video_processor.py examples/motion_demo.mp4
```

## Dependencies

### Core Libraries
- **OpenCV**: Video processing and computer vision
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Plotting and data visualization
- **Pillow**: Image processing and thumbnail generation
- **Tkinter**: GUI framework (included with Python)

### Development Tools
- **pytest**: Unit testing framework
- **threading**: Multi-threaded processing support

## Performance

### Processing Speed
- **Small videos** (< 100MB): Real-time processing
- **Medium videos** (100MB - 1GB): 2-5x faster than real-time
- **Large videos** (> 1GB): Processing time varies based on resolution and length

### Memory Usage
- **Base application**: ~50MB RAM
- **During processing**: Depends on video resolution and length
- **Results storage**: Thumbnails and metadata require minimal space

### Supported Formats
- **Input**: MP4, AVI, MOV, MKV, WMV, FLV
- **Output**: JPEG thumbnails, JSON statistics, PNG plots

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot open video file" | Check format support (MP4, AVI, MOV), try video conversion |
| "Processing failed" | Verify file size, disk space, try different parameters |
| Slow processing | Reduce video resolution, check system resources |
| No keyframes extracted | Lower blur threshold, adjust other parameters |
| GUI doesn't start | Check Python version (3.7+), install missing dependencies |
| Blank thumbnails | Verify video codec support, try different video file |

**Debug Information**: Check the Logs tab in the GUI for detailed error information and processing status.

## Applications

### Professional Use Cases
- **Video Summarization**: Create representative frame collections
- **Content Analysis**: Extract frames for machine learning datasets
- **Storage Optimization**: Reduce video storage requirements
- **Video Indexing**: Build searchable frame databases
- **Quality Control**: Identify and filter low-quality segments

### Educational Applications
- **Computer Vision Learning**: Understand IVP algorithm implementations
- **Algorithm Comparison**: Visualize different processing approaches
- **Parameter Tuning**: Experiment with threshold values and their effects
- **Research Projects**: Use as baseline for keyframe extraction studies

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes focusing on the desktop application
4. Test with the included test suite
5. Submit a pull request

### Code Style
- Follow PEP 8 Python style guidelines
- Use descriptive variable and function names
- Include docstrings for all functions and classes
- Maintain consistent indentation and formatting