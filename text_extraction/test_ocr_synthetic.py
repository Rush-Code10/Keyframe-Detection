"""
Create a test news image with text for OCR testing
"""

import cv2
import numpy as np
import os

def create_test_news_image():
    """Create a synthetic news image with text for testing OCR."""
    
    # Create a blank image (news-like background)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # Dark background
    
    # Add a red banner at top (typical news layout)
    cv2.rectangle(img, (0, 0), (640, 80), (0, 0, 180), -1)
    
    # Add headline text
    cv2.putText(img, "BREAKING NEWS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Add main content area
    cv2.rectangle(img, (20, 100), (620, 300), (50, 50, 50), -1)
    
    # Add main headline
    cv2.putText(img, "Major Technology Breakthrough", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "Scientists develop new AI system", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "Research shows 95% improvement", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Add ticker at bottom
    cv2.rectangle(img, (0, 400), (640, 480), (0, 100, 0), -1)
    cv2.putText(img, "STOCK MARKET: DOW +2.5% | NASDAQ +1.8% | S&P500 +2.1%", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add timestamp
    cv2.putText(img, "12:30 PM EST", (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def test_text_extraction_on_synthetic_image():
    """Test text extraction on a synthetic news image."""
    
    print("🎨 Creating synthetic news image for OCR testing...")
    
    # Create test image
    test_img = create_test_news_image()
    
    # Save the test image
    test_path = "test_news_image.jpg"
    cv2.imwrite(test_path, test_img)
    print(f"📸 Test image saved: {test_path}")
    
    # Test OCR
    print("🔍 Testing OCR on synthetic image...")
    
    try:
        import sys
        sys.path.append('.')
        from modules.text_extractor import create_text_extractor
        
        # Create text extractor
        extractor = create_text_extractor()
        
        # Extract text
        result = extractor.extract_text_from_keyframe(test_img, 0, 0.0)
        
        print(f"📊 OCR Results:")
        print(f"  Method: {result.processing_method}")
        print(f"  Confidence: {result.total_confidence:.2f}")
        print(f"  Regions found: {len(result.regions)}")
        
        if result.regions:
            print(f"📝 Extracted Text:")
            for i, region in enumerate(result.regions):
                print(f"  {i+1}. [{region.region_type}] \"{region.text}\" (conf: {region.confidence:.2f})")
                print(f"      BBox: {region.bbox}")
        else:
            print("  No text detected")
            
        # Create visualization
        if result.regions:
            viz_img = extractor.visualize_text_regions(test_img, result.regions)
            cv2.imwrite("test_news_image_annotated.jpg", viz_img)
            print(f"📸 Annotated image saved: test_news_image_annotated.jpg")
            
    except Exception as e:
        print(f"❌ Error during OCR test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_text_extraction_on_synthetic_image()