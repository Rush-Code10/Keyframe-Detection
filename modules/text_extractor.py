"""
Text Extraction Module for News Keyframes
Uses IVP (Image and Video Processing) techniques for OCR and text analysis.
Implements concepts from the IVP syllabus including:
- Image enhancement techniques
- Morphological operations
- Edge detection
- Image segmentation
- Histogram processing
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import re

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TextRegion:
    """Represents a detected text region in an image."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    region_type: str  # 'headline', 'ticker', 'caption', 'other'

@dataclass
class ExtractedText:
    """Container for all extracted text from a keyframe."""
    frame_number: int
    timestamp: float
    regions: List[TextRegion]
    processing_method: str
    total_confidence: float

class IVPTextExtractor:
    """
    Text extraction using IVP techniques from the syllabus.
    Implements image enhancement, morphological operations, and segmentation.
    """
    
    def __init__(self):
        """Initialize the IVP-based text extractor."""
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                
        # Tesseract configuration for better text detection
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?:;-+()'
        
    def enhance_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply IVP enhancement techniques to improve OCR accuracy.
        Implements Unit 2: Image Enhancement techniques from syllabus.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 1. Linear transformation - Contrast stretching (Unit 2)
        # Piecewise linear contrast stretching with safe calculations
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        if max_val > min_val and (max_val - min_val) > 1:
            # Safe contrast stretching to avoid overflow
            contrast_stretched = np.clip((gray - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)
        else:
            contrast_stretched = gray
            
        # 2. Histogram equalization (Unit 2)
        hist_eq = cv2.equalizeHist(contrast_stretched)
        
        # 3. Gaussian filtering for noise reduction (Unit 2)
        gaussian_filtered = cv2.GaussianBlur(hist_eq, (3, 3), 0)
        
        # 4. Adaptive thresholding for binarization (Unit 6 - Segmentation)
        # Otsu method implementation (Unit 6)
        _, otsu_thresh = cv2.threshold(gaussian_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Morphological operations (Unit 4)
        # Opening operation to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
        
        # Closing operation to fill gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def detect_text_regions_morphological(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using morphological operations (Unit 4).
        Implements dilation, erosion, opening, closing from syllabus.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply enhancement
        enhanced = self.enhance_image_for_ocr(gray)
        
        # Morphological operations to detect text regions
        # Use rectangular kernel for text detection
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        
        # Dilation to connect text characters
        dilated = cv2.dilate(enhanced, kernel_rect, iterations=2)
        
        # Erosion to separate text lines
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        eroded = cv2.erode(dilated, kernel_line, iterations=1)
        
        # Find contours for text regions
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter based on aspect ratio and size (typical for text)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 1.5 and w > 50 and h > 10:  # Text-like regions
                text_regions.append((x, y, w, h))
                
        return text_regions
    
    def detect_edges_for_text(self, image: np.ndarray) -> np.ndarray:
        """
        Apply edge detection for text enhancement (Unit 5).
        Implements Canny edge detector from syllabus.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny edge detection (Unit 5)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        return edges
    
    def classify_text_region_type(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> str:
        """
        Classify text region type based on position and size.
        Common news layout analysis.
        """
        x, y, w, h = bbox
        img_height, img_width = image_shape[:2]
        
        # Relative position analysis
        y_ratio = y / img_height
        x_ratio = x / img_width
        w_ratio = w / img_width
        h_ratio = h / img_height
        
        # Headlines typically in upper portion and wide
        if y_ratio < 0.3 and w_ratio > 0.4:
            return 'headline'
        
        # Tickers typically at bottom and wide
        elif y_ratio > 0.8 and w_ratio > 0.5:
            return 'ticker'
            
        # Captions typically small and in lower half
        elif y_ratio > 0.5 and h_ratio < 0.1:
            return 'caption'
            
        else:
            return 'other'
    
    def extract_text_pytesseract(self, image: np.ndarray) -> List[TextRegion]:
        """Extract text using Tesseract OCR with IVP preprocessing."""
        if not PYTESSERACT_AVAILABLE:
            logger.warning("pytesseract not available")
            return []
            
        try:
            # Apply IVP enhancement
            enhanced = self.enhance_image_for_ocr(image)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(enhanced, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
            
            regions = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    text = data['text'][i].strip()
                    if text and len(text) > 2:  # Filter out single characters
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        region_type = self.classify_text_region_type((x, y, w, h), image.shape)
                        
                        regions.append(TextRegion(
                            text=text,
                            confidence=float(data['conf'][i]) / 100.0,
                            bbox=(x, y, w, h),
                            region_type=region_type
                        ))
                        
            return regions
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return []
    
    def extract_text_easyocr(self, image: np.ndarray) -> List[TextRegion]:
        """Extract text using EasyOCR with IVP preprocessing."""
        if not self.easyocr_reader:
            logger.warning("EasyOCR not available")
            return []
            
        try:
            regions = []
            
            # Try with multiple preprocessing approaches
            images_to_try = [
                image,  # Original
                self.enhance_image_for_ocr(image),  # Enhanced
            ]
            
            # Also try with different EasyOCR parameters
            for img_variant in images_to_try:
                try:
                    # EasyOCR detection with lower confidence threshold and different settings
                    results = self.easyocr_reader.readtext(
                        img_variant, 
                        width_ths=0.7,  # Lower threshold for text width
                        height_ths=0.7,  # Lower threshold for text height
                        detail=1,
                        paragraph=False
                    )
                    
                    for detection in results:
                        bbox_points, text, confidence = detection
                        if confidence > 0.2 and len(text.strip()) > 1:  # Lower confidence threshold
                            # Convert bbox points to rectangle
                            x_coords = [point[0] for point in bbox_points]
                            y_coords = [point[1] for point in bbox_points]
                            x, y = int(min(x_coords)), int(min(y_coords))
                            w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                            
                            # Check if this text is already detected (avoid duplicates)
                            text_clean = text.strip()
                            is_duplicate = any(
                                self._texts_similar(text_clean, existing.text) 
                                for existing in regions
                            )
                            
                            if not is_duplicate:
                                region_type = self.classify_text_region_type((x, y, w, h), image.shape)
                                
                                regions.append(TextRegion(
                                    text=text_clean,
                                    confidence=confidence,
                                    bbox=(x, y, w, h),
                                    region_type=region_type
                                ))
                                
                except Exception as inner_e:
                    logger.warning(f"EasyOCR variant failed: {inner_e}")
                    continue
                    
            return regions
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return []
    
    def extract_text_from_keyframe(self, image: np.ndarray, frame_number: int, timestamp: float) -> ExtractedText:
        """
        Extract text from a keyframe using multiple OCR methods.
        Combines IVP preprocessing with OCR engines.
        """
        all_regions = []
        methods_used = []
        
        # Try EasyOCR first (often more accurate)
        if self.easyocr_reader:
            easyocr_regions = self.extract_text_easyocr(image)
            all_regions.extend(easyocr_regions)
            methods_used.append("EasyOCR")
            
        # Try Tesseract as backup/supplement
        if PYTESSERACT_AVAILABLE:
            tesseract_regions = self.extract_text_pytesseract(image)
            # Merge with EasyOCR results, avoiding duplicates
            for t_region in tesseract_regions:
                # Check if this text is already found by EasyOCR
                is_duplicate = False
                for e_region in all_regions:
                    if self._texts_similar(t_region.text, e_region.text):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    all_regions.append(t_region)
            methods_used.append("Tesseract")
        
        # Calculate overall confidence
        if all_regions:
            total_confidence = sum(region.confidence for region in all_regions) / len(all_regions)
        else:
            total_confidence = 0.0
            
        return ExtractedText(
            frame_number=frame_number,
            timestamp=timestamp,
            regions=all_regions,
            processing_method=" + ".join(methods_used) if methods_used else "None",
            total_confidence=total_confidence
        )
    
    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar (for duplicate detection)."""
        if not text1 or not text2:
            return False
            
        # Simple similarity check
        text1_clean = re.sub(r'[^\w\s]', '', text1.lower())
        text2_clean = re.sub(r'[^\w\s]', '', text2.lower())
        
        if text1_clean == text2_clean:
            return True
            
        # Check if one is contained in the other
        if text1_clean in text2_clean or text2_clean in text1_clean:
            return True
            
        return False
    
    def visualize_text_regions(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """
        Visualize detected text regions on the image.
        Different colors for different region types.
        """
        result = image.copy()
        
        color_map = {
            'headline': (0, 255, 0),    # Green
            'ticker': (255, 0, 0),      # Blue  
            'caption': (0, 255, 255),   # Yellow
            'other': (255, 0, 255)      # Magenta
        }
        
        for region in regions:
            x, y, w, h = region.bbox
            color = color_map.get(region.region_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Add text label
            label = f"{region.region_type}: {region.confidence:.2f}"
            cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return result

# Factory function to create text extractor
def create_text_extractor() -> IVPTextExtractor:
    """Create and return a text extractor instance."""
    return IVPTextExtractor()