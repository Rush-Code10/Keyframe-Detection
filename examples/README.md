# FrameSift Lite Example Videos

This directory contains example videos designed to demonstrate the different IVP (Image and Video Processing) techniques implemented in FrameSift Lite.

## Example Videos

### 1. scene_transitions.mp4
- **Duration**: 10 seconds, 24 fps
- **Purpose**: Demonstrates histogram comparison and shot boundary detection
- **Content**: Four distinct scenes (Sunset, Forest, Desert, Ocean) with clear visual transitions
- **Expected Results**: Should detect scene boundaries and extract keyframes at transitions
- **Best Parameters**: Higher histogram threshold (0.6-0.8)

### 2. motion_demo.mp4
- **Duration**: 8 seconds, 30 fps
- **Purpose**: Demonstrates frame difference detection for motion analysis
- **Content**: Moving objects (ball, rotating rectangle, pulsing circle) with gradual background changes
- **Expected Results**: Should identify frames with significant motion changes
- **Best Parameters**: Lower frame difference threshold (0.1-0.3)

### 3. quality_variation.mp4
- **Duration**: 6 seconds, 25 fps
- **Purpose**: Demonstrates blur detection and quality filtering
- **Content**: Detailed patterns with alternating sharp and blurry sections
- **Expected Results**: Should filter out blurry frames and keep sharp ones
- **Best Parameters**: Higher blur threshold (70-100)

### 4. mixed_content.mp4
- **Duration**: 12 seconds, 20 fps
- **Purpose**: Comprehensive test of all three IVP techniques
- **Content**: Four sections testing different aspects:
  - Section 1: Detail patterns with blur variations
  - Section 2: Gradual motion
  - Section 3: Scene changes with different colors/shapes
  - Section 4: Combined motion, scene changes, and blur
- **Expected Results**: Should demonstrate all three algorithms working together
- **Best Parameters**: Balanced settings (default values work well)

## Usage Instructions

1. **Upload any example video** to FrameSift Lite
2. **Adjust parameters** based on the video type (see recommendations above)
3. **Process the video** and observe the results
4. **Compare results** with different parameter settings to understand their effects

## Expected Performance

All example videos are designed to achieve the target 70-80% frame reduction while preserving key visual information:

- **scene_transitions.mp4**: ~75% reduction (should extract 1-2 keyframes per scene)
- **motion_demo.mp4**: ~70% reduction (should capture key motion states)
- **quality_variation.mp4**: ~80% reduction (should keep only sharp frames)
- **mixed_content.mp4**: ~75% reduction (balanced extraction across all sections)

## Educational Value

These examples help understand:

1. **Frame Difference (MAD)**: How motion and content changes affect frame similarity
2. **Histogram Comparison**: How color and lighting changes indicate scene boundaries
3. **Blur Detection**: How image sharpness affects keyframe selection
4. **Combined Processing**: How all three techniques work together for optimal results

## Creating Custom Examples

You can create your own example videos using the `create_example_videos.py` script:

```bash
python create_example_videos.py
```

This will generate all example videos in the `examples/` directory.
