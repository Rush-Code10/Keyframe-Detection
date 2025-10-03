"""
News Content Detection Module
Specialized detector for news content transitions in video clips with text extraction.
"""

import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

# Import text extraction module
try:
    from .text_extractor import create_text_extractor, ExtractedText, TextRegion
    TEXT_EXTRACTION_AVAILABLE = True
except ImportError:
    TEXT_EXTRACTION_AVAILABLE = False
    ExtractedText = None
    TextRegion = None

logger = logging.getLogger(__name__)

@dataclass
class NewsTransition:
    """Represents a detected news content transition."""
    frame_number: int
    timestamp: float
    transition_type: str  # 'headline_change', 'person_change', 'scene_change'
    confidence: float
    roi_changes: Dict[str, float]  # Region-specific change scores
    extracted_text: Optional[object] = None  # OCR results for this keyframe

class NewsContentDetector:
    """Specialized detector for news content transitions."""
    
    def __init__(self, headline_threshold: float = 0.5, person_threshold: float = 0.4, 
                 scene_threshold: float = 0.3, min_gap: float = 2.0):
        """
        Initialize the news content detector.
        
        Args:
            headline_threshold: Minimum threshold for headline changes
            person_threshold: Minimum threshold for person changes  
            scene_threshold: Minimum threshold for scene changes
            min_gap: Minimum gap in seconds between transitions
        """
        self.headline_threshold = headline_threshold
        self.person_threshold = person_threshold
        self.scene_threshold = scene_threshold
        self.min_gap = min_gap
        
        # Initialize face detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            logger.warning(f"Could not load face cascade: {e}")
            self.face_cascade = None
            
        # Initialize text extractor
        self.text_extractor = None
        if TEXT_EXTRACTION_AVAILABLE:
            try:
                self.text_extractor = create_text_extractor()
                logger.info("Text extraction initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize text extractor: {e}")
            
        self.prev_frame = None
        self.prev_faces = []
        
    def detect_transitions(self, video_path: str, progress_callback=None) -> List[NewsTransition]:
        """
        Detect content transitions in news clips.
        
        Args:
            video_path: Path to the video file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of detected news transitions
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        transitions = []
        frame_count = 0
        
        # Define regions of interest for news content
        rois = self._define_news_rois()
        
        logger.info(f"Analyzing {total_frames} frames for news transitions...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_count / fps
            
            if self.prev_frame is not None:
                transition = self._analyze_frame_transition(frame, frame_count, timestamp, rois)
                if transition:
                    transitions.append(transition)
            
            self.prev_frame = frame.copy()
            frame_count += 1
            
            # Update progress
            if progress_callback and frame_count % 30 == 0:  # Update every 30 frames
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
            
        cap.release()
        
        # Filter transitions that are too close together
        filtered_transitions = self._filter_transitions(transitions)
        
        logger.info(f"Detected {len(filtered_transitions)} significant news transitions")
        return filtered_transitions
    
    def _define_news_rois(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Define regions of interest for news content."""
        return {
            'headline_lower': (0.0, 0.7, 1.0, 1.0),      # Bottom 30% for lower thirds/headlines
            'headline_upper': (0.0, 0.0, 1.0, 0.2),        # Top 20% for breaking news banners
            'main_content': (0.0, 0.2, 1.0, 0.7),        # Middle 50% for main content/faces
            'sidebar': (0.7, 0.2, 1.0, 0.7),           # Right side for graphics/info
        }
    
    def _detect_headline_changes(self, frame: np.ndarray, prev_frame: np.ndarray, 
                               rois: Dict) -> float:
        """Detect changes in headline/text regions."""
        h, w = frame.shape[:2]
        max_change = 0.0
        
        for roi_name in ['headline_lower', 'headline_upper', 'sidebar']:
            if roi_name in rois:
                x1, y1, x2, y2 = rois[roi_name]
                x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                
                # Extract ROI
                roi_current = frame[y1:y2, x1:x2]
                roi_prev = prev_frame[y1:y2, x1:x2]
                
                if roi_current.size == 0 or roi_prev.size == 0:
                    continue
                
                # Convert to grayscale for text analysis
                gray_current = cv2.cvtColor(roi_current, cv2.COLOR_BGR2GRAY)
                gray_prev = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to enhance text
                _, thresh_current = cv2.threshold(gray_current, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, thresh_prev = cv2.threshold(gray_prev, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Calculate structural similarity for text regions
                diff = cv2.absdiff(thresh_current, thresh_prev)
                change_ratio = np.sum(diff > 50) / diff.size
                
                max_change = max(max_change, change_ratio)
        
        return max_change
    
    def _detect_person_changes(self, frame: np.ndarray, prev_frame: np.ndarray) -> Tuple[float, List]:
        """Detect changes in people (faces) in the frame."""
        if self.face_cascade is None:
            return 0.0, []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Calculate face change score
        face_change_score = 0.0
        
        if len(faces) != len(self.prev_faces):
            # Different number of faces
            face_change_score = 0.8
        elif len(faces) > 0 and len(self.prev_faces) > 0:
            # Compare face positions and sizes
            face_distances = []
            for face in faces:
                min_dist = float('inf')
                for prev_face in self.prev_faces:
                    # Calculate distance between face centers
                    center1 = (face[0] + face[2]//2, face[1] + face[3]//2)
                    center2 = (prev_face[0] + prev_face[2]//2, prev_face[1] + prev_face[3]//2)
                    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    min_dist = min(min_dist, dist)
                face_distances.append(min_dist)
            
            if face_distances:
                avg_movement = np.mean(face_distances)
                face_change_score = min(avg_movement / 100.0, 1.0)  # Normalize
        
        self.prev_faces = faces.copy() if len(faces) > 0 else []
        return face_change_score, faces
    
    def _detect_scene_changes(self, frame: np.ndarray, prev_frame: np.ndarray) -> float:
        """Detect general scene changes using histogram comparison."""
        # Convert to HSV for better color analysis
        hsv_current = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist_current = cv2.calcHist([hsv_current], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist_prev = cv2.calcHist([hsv_prev], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # Compare histograms
        correlation = cv2.compareHist(hist_current, hist_prev, cv2.HISTCMP_CORREL)
        scene_change_score = max(0.0, 1.0 - correlation)
        
        return scene_change_score
    
    def _analyze_frame_transition(self, frame: np.ndarray, frame_num: int, 
                                timestamp: float, rois: Dict) -> Optional[NewsTransition]:
        """Analyze if current frame represents a significant content transition."""
        
        # Detect headline changes
        headline_score = self._detect_headline_changes(frame, self.prev_frame, rois)
        
        # Detect person changes
        person_score, faces = self._detect_person_changes(frame, self.prev_frame)
        
        # Detect general scene changes
        scene_score = self._detect_scene_changes(frame, self.prev_frame)
        
        # Calculate overall transition confidence
        roi_changes = {
            'headline': headline_score,
            'person': person_score,
            'scene': scene_score
        }
        
        # Determine transition type and confidence
        max_change = max(headline_score, person_score, scene_score)
        
        if max_change < self.scene_threshold:  # No significant change
            return None
            
        # Classify transition type
        if headline_score >= self.headline_threshold:
            transition_type = 'headline_change'
            confidence = headline_score
        elif person_score >= self.person_threshold:
            transition_type = 'person_change'
            confidence = person_score
        else:
            transition_type = 'scene_change'
            confidence = scene_score
        
        return NewsTransition(
            frame_number=frame_num,
            timestamp=timestamp,
            transition_type=transition_type,
            confidence=confidence,
            roi_changes=roi_changes,
            extracted_text=None  # Text extraction will be done during export
        )
    
    def _filter_transitions(self, transitions: List[NewsTransition]) -> List[NewsTransition]:
        """Remove transitions that are too close together."""
        if not transitions:
            return []
            
        # Sort by timestamp
        sorted_transitions = sorted(transitions, key=lambda x: x.timestamp)
        filtered = [sorted_transitions[0]]
        
        for transition in sorted_transitions[1:]:
            if transition.timestamp - filtered[-1].timestamp >= self.min_gap:
                filtered.append(transition)
            elif transition.confidence > filtered[-1].confidence:
                # Replace with higher confidence transition
                filtered[-1] = transition
                
        return filtered

class NewsKeyframeExtractor:
    """Extract keyframes from news content transitions."""
    
    def __init__(self, detector: NewsContentDetector):
        self.detector = detector
    
    def extract_keyframes(self, video_path: str, output_dir: str, 
                         progress_callback=None) -> List[str]:
        """
        Extract keyframes from news clips focusing on content changes.
        Text extraction will be done separately during export.
        """
        
        # Detect transitions
        transitions = self.detector.detect_transitions(video_path, progress_callback)
        
        if not transitions:
            logger.warning("No news transitions detected")
            return []
        
        # Extract frames only (no text processing during detection)
        keyframe_paths = []
        cap = cv2.VideoCapture(video_path)
        
        for i, transition in enumerate(transitions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, transition.frame_number)
            ret, frame = cap.read()
            
            if ret:
                confidence_str = f"{transition.confidence:.2f}".replace('.', '_')
                filename = f"news_keyframe_{transition.frame_number:06d}_{transition.transition_type}_{confidence_str}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Only save if file doesn't already exist
                if not os.path.exists(filepath):
                    cv2.imwrite(filepath, frame)
                    keyframe_paths.append(filepath)
                    
                    logger.info(f"Extracted {transition.transition_type} at {transition.timestamp:.2f}s "
                              f"(confidence: {transition.confidence:.2f})")
                else:
                    keyframe_paths.append(filepath)  # Still add to list even if exists
            
            # Update progress for keyframe extraction
            if progress_callback:
                progress = 50 + ((i + 1) / len(transitions)) * 50  # 50-100% range
                progress_callback(progress)
        
        cap.release()
        logger.info(f"Extracted {len(keyframe_paths)} news keyframes")
        
        return keyframe_paths
