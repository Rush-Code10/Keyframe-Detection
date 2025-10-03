#!/usr/bin/env python3
"""
Create example videos for FrameSift Lite demonstration

This script generates sample videos with different characteristics
to demonstrate the keyframe extraction capabilities.
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_directory():
    """Create examples directory if it doesn't exist."""
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)
    return examples_dir


def create_scene_transition_video(output_path, duration=10, fps=24):
    """
    Create a video with distinct scene transitions.
    Good for demonstrating histogram comparison and shot boundary detection.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    scene_duration = total_frames // 4  # 4 scenes
    
    scenes = [
        {'color': (100, 50, 200), 'name': 'Sunset', 'pattern': 'circles'},
        {'color': (50, 200, 100), 'name': 'Forest', 'pattern': 'trees'},
        {'color': (200, 100, 50), 'name': 'Desert', 'pattern': 'dunes'},
        {'color': (50, 100, 200), 'name': 'Ocean', 'pattern': 'waves'}
    ]
    
    for i in range(total_frames):
        scene_index = min(i // scene_duration, len(scenes) - 1)
        scene = scenes[scene_index]
        
        # Create base frame with scene color
        frame = np.full((480, 640, 3), scene['color'], dtype=np.uint8)
        
        # Add scene-specific patterns
        progress_in_scene = (i % scene_duration) / scene_duration
        
        if scene['pattern'] == 'circles':
            # Sunset with sun and clouds
            sun_y = int(100 + 50 * progress_in_scene)
            cv2.circle(frame, (500, sun_y), 40, (255, 255, 100), -1)
            # Clouds
            for x in range(100, 600, 150):
                cv2.ellipse(frame, (x, 80), (60, 30), 0, 0, 360, (255, 255, 255), -1)
                
        elif scene['pattern'] == 'trees':
            # Forest with trees
            for x in range(50, 600, 80):
                tree_height = int(200 + 50 * np.sin(progress_in_scene * np.pi + x/100))
                cv2.rectangle(frame, (x, 480-tree_height), (x+20, 480), (139, 69, 19), -1)
                cv2.circle(frame, (x+10, 480-tree_height), 30, (34, 139, 34), -1)
                
        elif scene['pattern'] == 'dunes':
            # Desert with dunes
            for x in range(0, 640, 100):
                dune_height = int(100 + 80 * np.sin(progress_in_scene * 2 * np.pi + x/200))
                points = np.array([[x, 480], [x+50, 480-dune_height], [x+100, 480]], np.int32)
                cv2.fillPoly(frame, [points], (255, 215, 0))
                
        elif scene['pattern'] == 'waves':
            # Ocean with waves
            for y in range(200, 480, 30):
                wave_offset = int(20 * np.sin(progress_in_scene * 4 * np.pi + y/50))
                cv2.line(frame, (0, y + wave_offset), (640, y + wave_offset), (255, 255, 255), 3)
        
        # Add scene title
        cv2.putText(frame, f"Scene: {scene['name']}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (20, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Created scene transition video: {output_path}")


def create_motion_video(output_path, duration=8, fps=30):
    """
    Create a video with moving objects and gradual changes.
    Good for demonstrating frame difference detection.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        progress = i / total_frames
        
        # Create gradient background that changes over time
        bg_color = (
            int(50 + 100 * progress),
            int(100 + 50 * np.sin(progress * 2 * np.pi)),
            int(150 - 50 * progress)
        )
        frame = np.full((480, 640, 3), bg_color, dtype=np.uint8)
        
        # Moving ball
        ball_x = int(50 + 540 * (progress % 1))
        ball_y = int(240 + 150 * np.sin(progress * 4 * np.pi))
        cv2.circle(frame, (ball_x, ball_y), 25, (255, 255, 255), -1)
        cv2.circle(frame, (ball_x, ball_y), 15, (0, 0, 0), -1)
        
        # Rotating rectangle
        center = (320, 240)
        angle = progress * 360 * 2
        size = (100, 60)
        
        # Calculate rectangle points
        rect_points = cv2.boxPoints(((center[0], center[1]), size, angle))
        rect_points = np.int32(rect_points)
        cv2.drawContours(frame, [rect_points], 0, (255, 255, 0), -1)
        cv2.drawContours(frame, [rect_points], 0, (0, 0, 0), 3)
        
        # Pulsing circle
        pulse_radius = int(30 + 20 * np.sin(progress * 8 * np.pi))
        cv2.circle(frame, (160, 120), pulse_radius, (255, 0, 255), 3)
        
        # Add motion info
        cv2.putText(frame, "Motion Demo", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (20, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Created motion video: {output_path}")


def create_quality_variation_video(output_path, duration=6, fps=25):
    """
    Create a video with varying image quality (sharp vs blurry).
    Good for demonstrating blur detection.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        progress = i / total_frames
        
        # Create detailed base frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add checkerboard pattern for detail
        checker_size = 20
        for y in range(0, 480, checker_size):
            for x in range(0, 640, checker_size):
                if (x//checker_size + y//checker_size) % 2 == 0:
                    frame[y:y+checker_size, x:x+checker_size] = (200, 200, 200)
                else:
                    frame[y:y+checker_size, x:x+checker_size] = (100, 100, 100)
        
        # Add text and shapes for sharpness testing
        cv2.putText(frame, "SHARPNESS TEST", (180, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, "Fine Details: 123456789", (150, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        
        # Add geometric shapes
        cv2.rectangle(frame, (100, 200), (300, 350), (255, 0, 0), 2)
        cv2.circle(frame, (450, 275), 60, (0, 0, 255), 2)
        
        # Add fine lines
        for j in range(10):
            y_pos = 380 + j * 3
            cv2.line(frame, (50, y_pos), (590, y_pos), (255, 255, 0), 1)
        
        # Apply blur based on position in video
        blur_cycle = np.sin(progress * 6 * np.pi)  # 3 complete cycles
        
        quality_label = "SHARP"
        if blur_cycle > 0.3:  # High blur periods
            kernel_size = int(15 + 10 * blur_cycle)
            if kernel_size % 2 == 0:
                kernel_size += 1
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            quality_label = "VERY BLURRY"
        elif blur_cycle > -0.3:  # Medium blur periods
            kernel_size = int(7 + 4 * abs(blur_cycle))
            if kernel_size % 2 == 0:
                kernel_size += 1
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            quality_label = "SLIGHTLY BLURRY"
        # else: sharp frames (no blur)
        
        # Add quality indicator (after blur to remain readable)
        cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Quality: {quality_label}", (15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Created quality variation video: {output_path}")


def create_mixed_content_video(output_path, duration=12, fps=20):
    """
    Create a video that combines all three characteristics.
    Good for comprehensive testing of all IVP algorithms.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        progress = i / total_frames
        
        # Determine current section (4 sections with different characteristics)
        section = int(progress * 4)
        section_progress = (progress * 4) % 1
        
        if section == 0:
            # Section 1: Static scene with fine details (good for blur detection)
            frame = np.full((480, 640, 3), (80, 120, 160), dtype=np.uint8)
            
            # Add detailed patterns
            for x in range(0, 640, 40):
                for y in range(0, 480, 40):
                    cv2.rectangle(frame, (x, y), (x+20, y+20), (255, 255, 255), 1)
            
            cv2.putText(frame, "Section 1: Detail Test", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Apply varying blur
            if section_progress > 0.5:
                kernel_size = int(5 + 10 * (section_progress - 0.5) * 2)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        elif section == 1:
            # Section 2: Gradual motion (good for frame difference)
            frame = np.full((480, 640, 3), (120, 80, 160), dtype=np.uint8)
            
            # Moving elements
            obj_x = int(50 + 540 * section_progress)
            cv2.circle(frame, (obj_x, 200), 30, (255, 255, 0), -1)
            cv2.circle(frame, (obj_x, 280), 20, (0, 255, 255), -1)
            
            cv2.putText(frame, "Section 2: Motion Test", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        elif section == 2:
            # Section 3: Scene changes (good for histogram comparison)
            colors = [
                (200, 100, 100),  # Red-ish
                (100, 200, 100),  # Green-ish
                (100, 100, 200),  # Blue-ish
                (200, 200, 100)   # Yellow-ish
            ]
            
            color_index = int(section_progress * len(colors))
            color_index = min(color_index, len(colors) - 1)
            
            frame = np.full((480, 640, 3), colors[color_index], dtype=np.uint8)
            
            # Add scene-specific elements
            if color_index == 0:  # Red scene
                cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), 3)
            elif color_index == 1:  # Green scene
                cv2.circle(frame, (320, 240), 80, (255, 255, 255), 3)
            elif color_index == 2:  # Blue scene
                points = np.array([[320, 160], [250, 320], [390, 320]], np.int32)
                cv2.drawContours(frame, [points], 0, (255, 255, 255), 3)
            else:  # Yellow scene
                cv2.ellipse(frame, (320, 240), (100, 60), 0, 0, 360, (255, 255, 255), 3)
            
            cv2.putText(frame, "Section 3: Scene Changes", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        else:
            # Section 4: Combined challenges
            frame = np.full((480, 640, 3), (160, 120, 80), dtype=np.uint8)
            
            # Moving object
            obj_x = int(100 + 440 * section_progress)
            obj_y = int(240 + 100 * np.sin(section_progress * 4 * np.pi))
            cv2.circle(frame, (obj_x, obj_y), 25, (255, 255, 255), -1)
            
            # Changing background
            bg_intensity = int(50 + 100 * section_progress)
            overlay = np.full((480, 640, 3), (bg_intensity, bg_intensity//2, bg_intensity//3), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            cv2.putText(frame, "Section 4: Combined Test", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Apply occasional blur
            if section_progress > 0.7:
                kernel_size = 7
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        # Add global frame counter
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (20, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Created mixed content video: {output_path}")


def create_readme_for_examples(examples_dir):
    """Create README file explaining the example videos."""
    readme_content = """# FrameSift Lite Example Videos

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
"""
    
    readme_path = examples_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Created examples README: {readme_path}")


def main():
    """Create all example videos."""
    logger.info("Creating example videos for FrameSift Lite...")
    
    examples_dir = create_example_directory()
    
    # Create example videos
    create_scene_transition_video(examples_dir / 'scene_transitions.mp4')
    create_motion_video(examples_dir / 'motion_demo.mp4')
    create_quality_variation_video(examples_dir / 'quality_variation.mp4')
    create_mixed_content_video(examples_dir / 'mixed_content.mp4')
    
    # Create documentation
    create_readme_for_examples(examples_dir)
    
    logger.info("All example videos created successfully!")
    logger.info(f"Examples are available in: {examples_dir.absolute()}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXAMPLE VIDEOS CREATED")
    print("="*60)
    
    for video_file in examples_dir.glob('*.mp4'):
        file_size = video_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ {video_file.name:<25} ({file_size:.1f} MB)")
    
    print(f"\nLocation: {examples_dir.absolute()}")
    print("See examples/README.md for detailed descriptions and usage instructions.")
    print("="*60)


if __name__ == '__main__':
    main()