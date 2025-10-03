# FrameSift Lite - IVP Algorithms Documentation

This document provides a comprehensive explanation of the Image and Video Processing (IVP) algorithms used in FrameSift Lite for intelligent keyframe extraction.

## 📊 Algorithm Overview

FrameSift Lite combines three fundamental computer vision algorithms to achieve optimal keyframe selection:

| Algorithm | Purpose | Method | Output Range |
|-----------|---------|--------|--------------|
| **Frame Difference** | Redundancy removal | Mean Absolute Difference (MAD) | 0.0 - 1.0 |
| **Histogram Comparison** | Scene boundary detection | Chi-Square distance | 0.0 - ∞ |
| **Blur Detection** | Quality filtering | Variance of Laplacian | 0.0 - ∞ |

---

## 🔄 Algorithm 1: Frame Difference (MAD)

### Purpose
Removes redundant frames by detecting significant visual changes between consecutive frames.

### Method: Mean Absolute Difference (MAD)
The Mean Absolute Difference calculates the average pixel-wise difference between two consecutive frames.

### Mathematical Formula
```
MAD(f1, f2) = (1/N) × Σ|f1(i,j) - f2(i,j)|
```

Where:
- `f1, f2` = consecutive frames (grayscale)
- `N` = total number of pixels
- `i, j` = pixel coordinates

### Algorithm Steps
1. **Preprocessing**: Convert frames to grayscale for consistent comparison
2. **Pixel Difference**: Calculate absolute difference for each pixel pair
3. **Normalization**: Divide by total pixels and normalize to [0,1] range
4. **Thresholding**: Compare against user-defined threshold

### Implementation Details
```python
def calculate_mad(frame1, frame2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Calculate mean and normalize
    mad_score = np.mean(diff) / 255.0
    
    return mad_score
```

### Threshold Guidelines
- **Low (0.1-0.3)**: More sensitive, keeps frames with subtle changes
- **Medium (0.3-0.6)**: Balanced approach, default range
- **High (0.6-1.0)**: Less sensitive, removes more similar frames

### Use Cases
- **Action videos**: Lower thresholds to capture motion
- **Static scenes**: Higher thresholds to remove near-duplicates
- **Talking heads**: Medium thresholds for facial expression changes

---

## 📈 Algorithm 2: Histogram Comparison (Chi-Square)

### Purpose
Detects scene boundaries and shot transitions by analyzing color distribution changes.

### Method: Chi-Square Distance
Compares color histograms between frames using statistical chi-square distance measure.

### Mathematical Formula
```
χ²(h1, h2) = Σ[(h1(i) - h2(i))² / (h1(i) + h2(i))]
```

Where:
- `h1, h2` = color histograms of consecutive frames
- `i` = histogram bin index
- Division by zero is handled with small epsilon value

### Algorithm Steps
1. **Histogram Calculation**: Generate RGB color histograms (typically 64 bins per channel)
2. **Normalization**: Normalize histograms to probability distributions
3. **Chi-Square Computation**: Calculate statistical distance between distributions
4. **Boundary Detection**: Identify peaks indicating scene changes

### Implementation Details
```python
def calculate_chi_square_distance(frame1, frame2):
    # Calculate histograms for each RGB channel
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    hist1 = hist1.flatten() / np.sum(hist1)
    hist2 = hist2.flatten() / np.sum(hist2)
    
    # Calculate chi-square distance
    chi_square = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
    
    return chi_square
```

### Threshold Guidelines
- **Low (0.1-0.4)**: Detects only major scene changes (cuts, fades)
- **Medium (0.4-0.7)**: Balanced detection of lighting and content changes
- **High (0.7-1.0)**: Sensitive to subtle color variations

### Advantages
- **Robust to motion**: Focuses on color content rather than spatial changes
- **Scene detection**: Excellent for identifying shot boundaries
- **Lighting changes**: Detects illumination variations effectively

### Limitations
- **Color-dependent**: Less effective with monochrome content
- **Gradual transitions**: May miss slow fades or dissolves
- **Similar scenes**: Different scenes with similar colors may be missed

---

## 🔍 Algorithm 3: Blur Detection (Variance of Laplacian)

### Purpose
Filters out blurry or low-quality frames to maintain visual quality in keyframe selection.

### Method: Variance of Laplacian
Measures image sharpness by calculating the variance of the Laplacian operator response.

### Mathematical Formula
```
Laplacian(I) = ∇²I = ∂²I/∂x² + ∂²I/∂y²
Blur_Score = Var(Laplacian(I))
```

Where:
- `I` = input image (grayscale)
- `∇²` = Laplacian operator
- `Var()` = variance function

### Algorithm Steps
1. **Grayscale Conversion**: Convert frame to grayscale for consistent analysis
2. **Laplacian Calculation**: Apply Laplacian kernel to detect edges
3. **Variance Computation**: Calculate variance of Laplacian response
4. **Quality Assessment**: Higher variance indicates sharper edges (less blur)

### Implementation Details
```python
def calculate_blur_score(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calculate variance
    blur_score = laplacian.var()
    
    return blur_score
```

### Laplacian Kernel
The standard 3x3 Laplacian kernel used:
```
[ 0, -1,  0]
[-1,  4, -1]
[ 0, -1,  0]
```

### Threshold Guidelines
- **Low (10-50)**: Accepts slightly blurry frames, good for low-resolution videos
- **Medium (50-100)**: Balanced quality filtering, default range
- **High (100-300)**: Strict quality requirements, only very sharp frames

### Score Interpretation
- **High scores (>100)**: Sharp, well-focused frames
- **Medium scores (50-100)**: Acceptable quality
- **Low scores (<50)**: Blurry or out-of-focus frames

### Advantages
- **Fast computation**: Simple convolution operation
- **Objective measure**: Quantitative sharpness assessment
- **Resolution independent**: Scales well across different video sizes

### Limitations
- **Noise sensitivity**: High noise can increase variance artificially
- **Edge dependency**: Requires sufficient edge content for accuracy
- **Context ignorance**: Doesn't consider semantic importance of blur

---

## 🔗 Algorithm Integration

### Combined Decision Making
FrameSift Lite combines all three algorithms using a multi-criteria approach:

1. **Blur Filtering**: First eliminate frames below blur threshold
2. **Change Detection**: Apply frame difference for redundancy removal
3. **Scene Analysis**: Use histogram comparison for boundary detection
4. **Final Selection**: Rank remaining frames and select top candidates

### Scoring Strategy
```python
def select_keyframes(frames, metrics, reduction_target):
    # Step 1: Filter by blur quality
    sharp_frames = [f for f in frames if f.blur_score > blur_threshold]
    
    # Step 2: Detect significant changes
    change_frames = []
    for i in range(1, len(sharp_frames)):
        if (sharp_frames[i].difference_score > diff_threshold or 
            sharp_frames[i].histogram_score > hist_threshold):
            change_frames.append(sharp_frames[i])
    
    # Step 3: Select based on reduction target
    target_count = int(len(frames) * (1 - reduction_target))
    keyframes = select_top_frames(change_frames, target_count)
    
    return keyframes
```

### Parameter Tuning Strategy
1. **Conservative**: Higher thresholds, preserves more frames
2. **Balanced**: Default settings for general use
3. **Aggressive**: Lower thresholds, maximum reduction

---

## 📊 Performance Characteristics

### Computational Complexity
| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Frame Difference | O(n×m) | O(1) | n×m = frame dimensions |
| Histogram Comparison | O(n×m + b³) | O(b³) | b = bins per channel |
| Blur Detection | O(n×m) | O(n×m) | Temporary Laplacian storage |

### Typical Processing Times
- **HD Video (1920×1080)**: ~50ms per frame
- **SD Video (720×480)**: ~15ms per frame
- **Mobile Video (640×360)**: ~8ms per frame

### Memory Requirements
- **Frame storage**: 3×width×height bytes per frame
- **Histogram storage**: 64³×4 = 1MB per frame pair
- **Working memory**: ~10-20MB for typical processing

---

## 🎯 Algorithm Strengths and Applications

### Frame Difference (MAD)
**Best for:**
- Action sequences with significant motion
- Removing duplicate or near-duplicate frames
- Fast processing requirements

**Limitations:**
- Camera movement can cause false positives
- Subtle changes may be missed
- Background noise can affect accuracy

### Histogram Comparison (Chi-Square)
**Best for:**
- Scene transition detection
- Color-based content analysis
- Robust motion handling

**Limitations:**
- Computationally more expensive
- Less effective with monochrome content
- May miss spatial rearrangements

### Blur Detection (Variance of Laplacian)
**Best for:**
- Quality assurance
- Focus-based selection
- Simple implementation

**Limitations:**
- Sensitive to noise
- Requires edge content
- May favor high-contrast artifacts

---

## 🔧 Optimization Techniques

### Frame Sampling
- **Temporal sampling**: Process every nth frame for speed
- **Spatial downsampling**: Reduce resolution for histogram computation
- **Region of interest**: Analyze specific image regions

### Adaptive Thresholding
- **Content-aware**: Adjust thresholds based on video characteristics
- **Temporal adaptation**: Modify parameters over time
- **Statistical feedback**: Use processing results to refine parameters

### Memory Optimization
- **Streaming processing**: Process frames without storing entire video
- **Buffer management**: Maintain sliding window of recent frames
- **Garbage collection**: Explicit memory cleanup for large videos

---

## 📚 References and Further Reading

### Academic Papers
1. **Frame Difference**: "Video Summarization Using Shot Boundary Detection" - IEEE Transactions on Multimedia
2. **Histogram Comparison**: "Color-Based Shot Boundary Detection" - Computer Vision and Image Understanding
3. **Blur Detection**: "Analysis of Focus Measure Operators for Shape-From-Focus" - Pattern Recognition

### OpenCV Documentation
- [Histogram Calculation](https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html)
- [Image Filtering](https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html)
- [Statistical Functions](https://docs.opencv.org/master/d2/de8/group__core__array.html)

### Implementation Resources
- **NumPy Statistical Functions**: For efficient array operations
- **Matplotlib Visualization**: For algorithm performance analysis
- **SciPy Signal Processing**: For advanced filtering techniques

---

**This documentation provides the theoretical foundation and practical implementation details for understanding and extending the FrameSift Lite keyframe extraction algorithms.** 🎬📊