"""
Demo script for News Detection with Text Extraction
Demonstrates the IVP-based OCR functionality integrated with news keyframe detection.
"""

import cv2
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.news_detector import NewsContentDetector, NewsKeyframeExtractor
from modules.text_extractor import create_text_extractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_text_extraction():
    """Demo the text extraction functionality."""
    
    print("🎬 FrameSift News Detection with Text Extraction Demo")
    print("=" * 60)
    
    # Check if demo video exists
    demo_videos = [
        "examples/mixed_content.mp4",
        "examples/scene_transitions.mp4", 
        "examples/motion_demo.mp4"
    ]
    
    video_path = None
    for video in demo_videos:
        if os.path.exists(video):
            video_path = video
            break
    
    if not video_path:
        print("❌ No demo video found. Please ensure demo videos exist in examples/ folder.")
        return
    
    print(f"📹 Using demo video: {video_path}")
    
    # Create output directory
    output_dir = "demo_news_text_results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize news detector with text extraction
        print("\n🔧 Initializing news content detector with text extraction...")
        detector = NewsContentDetector(
            headline_threshold=0.6,
            person_threshold=0.5,
            scene_threshold=0.4,
            min_gap=1.0
        )
        
        # Create extractor
        extractor = NewsKeyframeExtractor(detector)
        
        print("🔍 Detecting news transitions and extracting text...")
        
        # Progress callback
        def progress_callback(progress):
            print(f"Progress: {progress:.1f}%")
        
        # Extract keyframes with text
        keyframe_paths, text_results = extractor.extract_keyframes(
            video_path=video_path,
            output_dir=output_dir,
            progress_callback=progress_callback
        )
        
        # Display results
        print(f"\n✅ Processing Complete!")
        print(f"📊 Extracted {len(keyframe_paths)} keyframes")
        print(f"📝 Found text in {len(text_results)} keyframes")
        
        if text_results:
            print("\n📄 Text Extraction Results:")
            print("-" * 40)
            
            for i, result in enumerate(text_results):
                print(f"\nKeyframe {i+1}: Frame {result['frame_number']} (t={result['timestamp']:.2f}s)")
                print(f"  Transition: {result['transition_type']}")
                print(f"  OCR Method: {result['ocr_method']}")
                print(f"  OCR Confidence: {result['total_confidence']:.2f}")
                
                if result['text_regions']:
                    print(f"  Text Regions ({len(result['text_regions'])}):")
                    for j, region in enumerate(result['text_regions']):
                        print(f"    {j+1}. [{region['region_type']}] \"{region['text']}\" (conf: {region['confidence']:.2f})")
                else:
                    print("  No text detected")
            
            # Show JSON file location
            json_file = os.path.join(output_dir, "extracted_text.json")
            if os.path.exists(json_file):
                print(f"\n💾 Detailed results saved to: {json_file}")
        else:
            print("\n📝 No text was extracted from keyframes")
        
        print(f"\n📁 All results saved to: {output_dir}")
        
        # Display IVP techniques used
        print("\n🔬 IVP Techniques Applied:")
        print("  • Contrast stretching (linear transformation)")
        print("  • Histogram equalization")
        print("  • Gaussian filtering (noise reduction)")
        print("  • Otsu thresholding (automatic binarization)")
        print("  • Morphological operations (opening/closing)")
        print("  • Canny edge detection")
        print("  • Component labeling and segmentation")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

def test_individual_image():
    """Test text extraction on a single image."""
    print("\n🖼️ Testing individual image text extraction...")
    
    # Check if any keyframes exist from previous runs
    test_dirs = ["demo_news_results", "gui_results"]
    test_image = None
    
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            for file in os.listdir(dir_name):
                if file.endswith('.jpg') and 'keyframe' in file:
                    test_image = os.path.join(dir_name, file)
                    break
            if test_image:
                break
    
    if not test_image:
        print("  No test images found. Run news detection first to generate keyframes.")
        return
    
    print(f"  📸 Testing with: {test_image}")
    
    try:
        # Load image
        img = cv2.imread(test_image)
        if img is None:
            print("  ❌ Could not load image")
            return
        
        # Create text extractor
        extractor = create_text_extractor()
        
        # Extract text
        result = extractor.extract_text_from_keyframe(img, 0, 0.0)
        
        print(f"  📊 Results: {len(result.regions)} text regions found")
        print(f"  🔧 Method: {result.processing_method}")
        print(f"  📈 Confidence: {result.total_confidence:.2f}")
        
        if result.regions:
            print("  📝 Extracted Text:")
            for i, region in enumerate(result.regions):
                print(f"    {i+1}. [{region.region_type}] \"{region.text}\" (conf: {region.confidence:.2f})")
        else:
            print("  No text detected in this image")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    # Run the demo
    demo_text_extraction()
    
    # Test individual image
    test_individual_image()
    
    print("\n🎉 Demo completed! Check the output directories for results.")