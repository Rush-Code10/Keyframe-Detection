#!/usr/bin/env python3
"""
Demo script for VideoProcessor class.

This script demonstrates how to use the VideoProcessor class
to extract keyframes from a video using IVP algorithms.
"""

import numpy as np
import cv2
import tempfile
import os
from modules.video_processor import VideoProcessor


def create_test_video(filename: str, duration_seconds: int = 5, fps: int = 30):
    """
    Create a simple test video with changing content.
    
    Args:
        filename: Output video filename
        duration_seconds: Video duration in seconds
        fps: Frames per second
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))
    
    total_frames = duration_seconds * fps
    
    for i in range(total_frames):
        # Create frame with changing content
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some patterns that change over time
        if i < total_frames // 3:
            # First third: red background with moving circle
            frame[:, :, 2] = 100  # Red background
            center_x = int(320 + 200 * np.sin(i * 0.1))
            center_y = 240
            cv2.circle(frame, (center_x, center_y), 50, (255, 255, 255), -1)
        elif i < 2 * total_frames // 3:
            # Second third: green background with moving rectangle
            frame[:, :, 1] = 100  # Green background
            rect_x = int(200 + 200 * np.cos(i * 0.1))
            cv2.rectangle(frame, (rect_x, 200), (rect_x + 100, 300), (255, 255, 255), -1)
        else:
            # Last third: blue background with text
            frame[:, :, 0] = 100  # Blue background
            text = f"Frame {i}"
            cv2.putText(frame, text, (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add some noise for blur variation
        if i % 10 < 3:  # Every 10 frames, make 3 frames blurry
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {filename}")


def demo_video_processing():
    """Demonstrate video processing pipeline."""
    
    print("=== VideoProcessor Demo ===\n")
    
    # Create a temporary test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        test_video_path = tmp_file.name
    
    try:
        print("1. Creating test video...")
        create_test_video(test_video_path, duration_seconds=3, fps=10)  # Short video for demo
        
        print("\n2. Initializing VideoProcessor...")
        processor = VideoProcessor(
            frame_diff_threshold=0.1,
            histogram_threshold=0.3,
            blur_threshold=50.0
        )
        
        print("\n3. Processing video...")
        results = processor.process_video_complete(
            video_path=test_video_path,
            target_reduction=0.7,  # 70% reduction
            sample_rate=1
        )
        
        if results['success']:
            print("✓ Video processing completed successfully!")
            
            # Display results
            summary = results['summary']
            print(f"\n=== Processing Results ===")
            print(f"Total frames: {summary['total_frames']}")
            print(f"Keyframes selected: {summary['keyframes_selected']}")
            print(f"Reduction achieved: {summary['reduction_percentage']:.1f}%")
            print(f"Keyframe indices: {summary['keyframe_indices']}")
            
            # Algorithm statistics
            alg_summary = results['algorithm_summary']
            print(f"\n=== Algorithm Statistics ===")
            for alg_name, stats in alg_summary.items():
                print(f"{alg_name.replace('_', ' ').title()}:")
                print(f"  Mean score: {stats['mean']:.3f}")
                print(f"  Score range: {stats['min']:.3f} - {stats['max']:.3f}")
                print(f"  Threshold: {stats['threshold']:.3f}")
            
            # Temporal distribution
            temp_dist = summary['temporal_distribution']
            if temp_dist['gaps']:
                print(f"\n=== Temporal Distribution ===")
                print(f"Average gap between keyframes: {temp_dist['mean_gap']:.1f} frames")
                print(f"Gap range: {temp_dist['min_gap']} - {temp_dist['max_gap']} frames")
            
            print(f"\n=== Individual Keyframe Details ===")
            keyframes = results['keyframes']
            for i, kf in enumerate(keyframes):
                print(f"Keyframe {i+1}: Frame #{kf.frame_number} at {kf.timestamp:.2f}s")
                print(f"  Scores - Diff: {kf.difference_score:.3f}, Hist: {kf.histogram_score:.3f}, Blur: {kf.blur_score:.1f}")
        
        else:
            print(f"✗ Video processing failed: {results['error']}")
        
        print("\n4. Testing parameter updates...")
        processor.update_algorithm_parameters(
            frame_diff_threshold=0.2,
            blur_threshold=75.0
        )
        print("✓ Parameters updated successfully")
        
        print("\n5. Testing video validation...")
        validation = processor.validate_video_file(test_video_path)
        if validation['valid']:
            props = validation['properties']
            print("✓ Video validation passed")
            print(f"  Resolution: {props['width']}x{props['height']}")
            print(f"  Duration: {props['duration']:.2f} seconds")
            print(f"  Frame rate: {props['fps']:.1f} fps")
            print(f"  File size: {props['file_size'] / 1024:.1f} KB")
        else:
            print(f"✗ Video validation failed: {validation['error']}")
    
    finally:
        # Clean up
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)
            print(f"\n6. Cleaned up test video file")
    
    print("\n=== Demo completed ===")


if __name__ == "__main__":
    demo_video_processing()