#!/usr/bin/env python3
"""
News Detection Demo Script
Demonstrates the news content detection functionality with example parameters.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.news_detector import NewsContentDetector, NewsKeyframeExtractor

def demo_news_detection(video_path: str, output_dir: str = "demo_news_results"):
    """
    Demo the news detection functionality on a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save results
    """
    print("🎬 News Content Detection Demo")
    print("=" * 50)
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector with demo parameters
    print("\n🔧 Initializing News Content Detector...")
    detector = NewsContentDetector(
        headline_threshold=0.4,    # Slightly lower for demo
        person_threshold=0.3,      # Lower to catch more person changes
        scene_threshold=0.25,      # Lower to catch more scene changes
        min_gap=1.5               # 1.5 seconds minimum gap
    )
    
    print("✅ NewsContentDetector initialized")
    
    # Initialize extractor
    extractor = NewsKeyframeExtractor(detector)
    print("✅ NewsKeyframeExtractor initialized")
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    print(f"\n📹 Video Information:")
    print(f"   • File: {os.path.basename(video_path)}")
    print(f"   • Duration: {duration:.1f} seconds")
    print(f"   • Frames: {frame_count}")
    print(f"   • FPS: {fps:.1f}")
    
    # Process video
    print(f"\n🔍 Analyzing video for news transitions...")
    
    try:
        def progress_callback(progress):
            print(f"   Progress: {progress:.1f}%", end='\r')
        
        keyframe_paths = extractor.extract_keyframes(
            video_path=video_path,
            output_dir=output_dir,
            progress_callback=progress_callback
        )
        
        print(f"\n✅ Processing Complete!")
        print(f"\n📊 Results Summary:")
        print(f"   • Total transitions detected: {len(keyframe_paths)}")
        print(f"   • Output directory: {output_dir}")
        
        if keyframe_paths:
            print(f"\n🎬 Detected Transitions:")
            for i, path in enumerate(keyframe_paths, 1):
                filename = os.path.basename(path)
                # Parse filename to get transition info
                parts = filename.replace('.jpg', '').split('_')
                if len(parts) >= 7:  # news_keyframe_XXXXXX_type_change_conf_frac
                    frame_num = parts[2]
                    transition_type = f"{parts[3]}_{parts[4]}"  # e.g., "scene_change"
                    confidence_str = f"{parts[5]}.{parts[6]}"
                    print(f"   {i:2d}. Frame {frame_num}: {transition_type.replace('_', ' ').title()} (confidence: {confidence_str})")
            
            print(f"\n📁 Keyframes saved to: {output_dir}")
            print(f"   You can view the extracted keyframes in the results folder.")
        else:
            print(f"\nℹ️  No significant transitions detected.")
            print(f"   Try using a video with more scene changes or adjust the thresholds.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        return False

def main():
    """Main demo function"""
    print("News Content Detection Demo")
    print("This demo will analyze a video for news content transitions.\n")
    
    # Check for example videos
    example_videos = [
        "examples/mixed_content.mp4",
        "examples/scene_transitions.mp4",
        "examples/motion_demo.mp4"
    ]
    
    # Find available video
    test_video = None
    for video in example_videos:
        if os.path.exists(video):
            test_video = video
            break
    
    if test_video:
        print(f"📹 Using example video: {test_video}")
        success = demo_news_detection(test_video)
    else:
        print("ℹ️  No example videos found.")
        print("To run this demo:")
        print("1. Ensure you have video files in the 'examples' folder, or")
        print("2. Run: python create_example_videos.py")
        print("3. Then run this demo again")
        return
    
    if success:
        print(f"\n🎉 Demo completed successfully!")
        print(f"You can now use the News Detection tab in the GUI:")
        print(f"   python framesift_gui.py")
    else:
        print(f"\n❌ Demo failed. Check the error messages above.")

if __name__ == "__main__":
    main()