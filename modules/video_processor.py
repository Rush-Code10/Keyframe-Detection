"""
Video Processing Pipeline Module

Core video processing logic that coordinates IVP algorithms
for keyframe extraction.
"""

import cv2
import numpy as np
import os
import logging
import traceback
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from .ivp_algorithms import FrameDifferenceCalculator, HistogramComparator, BlurDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Custom exception for video processing errors."""
    def __init__(self, message: str, error_code: str = "VIDEO_PROCESSING_ERROR", details: str = None):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)


class FrameExtractionError(VideoProcessingError):
    """Exception for frame extraction failures."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message, "FRAME_EXTRACTION_ERROR", details)


class AlgorithmProcessingError(VideoProcessingError):
    """Exception for IVP algorithm processing failures."""
    def __init__(self, message: str, algorithm: str = None, details: str = None):
        error_code = f"{algorithm.upper()}_ALGORITHM_ERROR" if algorithm else "ALGORITHM_ERROR"
        super().__init__(message, error_code, details)


@dataclass
class FrameMetrics:
    """Data structure to store all algorithm results for a frame."""
    frame_number: int
    timestamp: float
    difference_score: float
    histogram_score: float
    blur_score: float
    is_keyframe: bool = False


class VideoProcessor:
    """
    Core video processing class that handles video file loading,
    frame extraction, and coordinates IVP algorithms for keyframe extraction.
    """
    
    def __init__(self, 
                 frame_diff_threshold: float = 0.1,
                 histogram_threshold: float = 0.5,
                 blur_threshold: float = 100.0):
        """
        Initialize VideoProcessor with algorithm parameters.
        
        Args:
            frame_diff_threshold: Threshold for frame difference detection
            histogram_threshold: Threshold for histogram comparison
            blur_threshold: Threshold for blur detection
        """
        self.frame_diff_threshold = frame_diff_threshold
        self.histogram_threshold = histogram_threshold
        self.blur_threshold = blur_threshold
        
        # Initialize IVP algorithm instances
        self.frame_diff_calc = FrameDifferenceCalculator(frame_diff_threshold)
        self.histogram_comp = HistogramComparator(histogram_threshold)
        self.blur_detector = BlurDetector(blur_threshold)
        
        # Processing state
        self.video_path = None
        self.frames = []
        self.frame_metrics = []
        self.fps = 0
        self.total_frames = 0
    
    def load_video(self, video_path: str) -> bool:
        """
        Load video file and validate it can be processed.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if video loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            VideoProcessingError: If video file is corrupted or unsupported
        """
        try:
            if not video_path:
                raise VideoProcessingError("Video path cannot be empty", "INVALID_PATH")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Check file permissions
            if not os.access(video_path, os.R_OK):
                raise VideoProcessingError(f"Cannot read video file: {video_path}", "PERMISSION_ERROR")
            
            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                raise VideoProcessingError("Video file is empty", "EMPTY_FILE")
            
            logger.info(f"Loading video file: {video_path} (size: {file_size / (1024*1024):.1f}MB)")
            
            # Open video capture with error handling
            cap = None
            try:
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    raise VideoProcessingError(f"Cannot open video file: {video_path}", "UNSUPPORTED_FORMAT", 
                                             "The video format may not be supported by OpenCV")
                
                # Get video properties with validation
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Validate video properties
                if self.fps <= 0 or self.fps > 120:
                    raise VideoProcessingError(f"Invalid frame rate: {self.fps} fps", "INVALID_FPS")
                
                if self.total_frames <= 0 or self.total_frames > 100000:
                    raise VideoProcessingError(f"Invalid frame count: {self.total_frames}", "INVALID_FRAME_COUNT")
                
                if width <= 0 or height <= 0 or width > 4096 or height > 4096:
                    raise VideoProcessingError(f"Invalid resolution: {width}x{height}", "INVALID_RESOLUTION")
                
                # Test reading first frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise VideoProcessingError("Cannot read frames from video file", "UNREADABLE_FRAMES",
                                             "The video file may be corrupted or use an unsupported codec")
                
                # Validate frame data
                if frame.size == 0:
                    raise VideoProcessingError("Video frames are empty", "EMPTY_FRAMES")
                
                self.video_path = video_path
                logger.info(f"Video loaded successfully: {video_path}")
                logger.info(f"Video properties: {self.total_frames} frames, {self.fps:.2f} fps, {width}x{height}")
                
                return True
                
            finally:
                if cap is not None:
                    cap.release()
            
        except FileNotFoundError:
            logger.error(f"Video file not found: {video_path}")
            raise
        except VideoProcessingError:
            logger.error(f"Video processing error for {video_path}: {traceback.format_exc()}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading video {video_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise VideoProcessingError(f"Failed to load video: {str(e)}", "UNEXPECTED_ERROR", str(e))
    
    def extract_frames(self, max_frames: Optional[int] = None, 
                      sample_rate: int = 1) -> List[np.ndarray]:
        """
        Extract frames from the loaded video with memory management.
        
        Args:
            max_frames: Maximum number of frames to extract (None for all)
            sample_rate: Extract every Nth frame (1 for all frames)
            
        Returns:
            List of frame arrays
            
        Raises:
            FrameExtractionError: If no video is loaded or extraction fails
        """
        try:
            if not self.video_path:
                raise FrameExtractionError("No video loaded. Call load_video() first.")
            
            # Validate parameters
            if sample_rate < 1:
                raise FrameExtractionError("Sample rate must be at least 1", f"Invalid sample_rate: {sample_rate}")
            
            if max_frames is not None and max_frames < 1:
                raise FrameExtractionError("Max frames must be at least 1", f"Invalid max_frames: {max_frames}")
            
            cap = None
            try:
                cap = cv2.VideoCapture(self.video_path)
                
                if not cap.isOpened():
                    raise FrameExtractionError("Failed to reopen video file", 
                                             f"Could not open {self.video_path} for frame extraction")
                
                frames = []
                frame_count = 0
                extracted_count = 0
                failed_reads = 0
                max_failed_reads = 10  # Allow some failed reads before giving up
                
                logger.info(f"Starting frame extraction (sample_rate={sample_rate}, max_frames={max_frames})")
                
                while True:
                    try:
                        ret, frame = cap.read()
                        
                        if not ret:
                            if frame_count == 0:
                                raise FrameExtractionError("Cannot read any frames from video", 
                                                          "Video file may be corrupted or empty")
                            break  # End of video
                        
                        # Reset failed read counter on successful read
                        failed_reads = 0
                        
                        # Apply sampling rate
                        if frame_count % sample_rate == 0:
                            # Validate frame data
                            if frame is None or frame.size == 0:
                                logger.warning(f"Empty frame at position {frame_count}, skipping")
                                frame_count += 1
                                continue
                            
                            try:
                                # Resize frame if too large to manage memory
                                height, width = frame.shape[:2]
                                if width > 1920 or height > 1080:
                                    # Resize to max 1920x1080 while maintaining aspect ratio
                                    scale = min(1920/width, 1080/height)
                                    new_width = int(width * scale)
                                    new_height = int(height * scale)
                                    frame = cv2.resize(frame, (new_width, new_height))
                                
                                # Create a copy to avoid memory issues
                                frame_copy = frame.copy()
                                frames.append(frame_copy)
                                extracted_count += 1
                                
                                # Check max frames limit
                                if max_frames and extracted_count >= max_frames:
                                    logger.info(f"Reached max frames limit: {max_frames}")
                                    break
                                
                                # Log progress for large videos
                                if extracted_count % 100 == 0:
                                    logger.info(f"Extracted {extracted_count} frames...")
                                
                            except Exception as frame_error:
                                logger.warning(f"Error processing frame {frame_count}: {str(frame_error)}")
                                # Continue with next frame instead of failing completely
                        
                        frame_count += 1
                        
                    except Exception as read_error:
                        failed_reads += 1
                        logger.warning(f"Failed to read frame {frame_count}: {str(read_error)}")
                        
                        if failed_reads >= max_failed_reads:
                            raise FrameExtractionError(f"Too many failed frame reads ({failed_reads})", 
                                                     "Video file may be corrupted")
                        
                        frame_count += 1
                        continue
                
                if not frames:
                    raise FrameExtractionError("No frames could be extracted from video", 
                                             f"Processed {frame_count} frames but none were extractable")
                
                self.frames = frames
                logger.info(f"Frame extraction completed: {len(frames)} frames extracted from {frame_count} total frames")
                
                return frames
                
            finally:
                if cap is not None:
                    cap.release()
            
        except FrameExtractionError:
            logger.error(f"Frame extraction error: {traceback.format_exc()}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during frame extraction: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise FrameExtractionError(f"Frame extraction failed: {str(e)}", str(e))
    
    def get_frame_timestamp(self, frame_number: int, sample_rate: int = 1) -> float:
        """
        Calculate timestamp for a frame number.
        
        Args:
            frame_number: Frame index in extracted frames
            sample_rate: Sampling rate used during extraction
            
        Returns:
            Timestamp in seconds
        """
        actual_frame_number = frame_number * sample_rate
        return actual_frame_number / self.fps if self.fps > 0 else 0.0
    
    def validate_video_file(self, video_path: str) -> Dict[str, Any]:
        """
        Validate video file and return properties without loading all frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties and validation status
        """
        validation_result = {
            'valid': False,
            'error': None,
            'properties': {}
        }
        
        try:
            if not os.path.exists(video_path):
                validation_result['error'] = "File does not exist"
                return validation_result
            
            # Check file size (limit to reasonable size for processing)
            file_size = os.path.getsize(video_path)
            max_size = 500 * 1024 * 1024  # 500MB limit
            
            if file_size > max_size:
                validation_result['error'] = f"File too large: {file_size / (1024*1024):.1f}MB (max: {max_size / (1024*1024)}MB)"
                return validation_result
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                validation_result['error'] = "Cannot open video file - unsupported format"
                return validation_result
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Test reading first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                validation_result['error'] = "Cannot read frames from video"
                return validation_result
            
            # Check reasonable video properties
            if fps <= 0 or fps > 120:
                validation_result['error'] = f"Invalid frame rate: {fps}"
                return validation_result
            
            if frame_count <= 0 or frame_count > 100000:
                validation_result['error'] = f"Invalid frame count: {frame_count}"
                return validation_result
            
            if width <= 0 or height <= 0 or width > 4096 or height > 4096:
                validation_result['error'] = f"Invalid resolution: {width}x{height}"
                return validation_result
            
            # Success
            validation_result['valid'] = True
            validation_result['properties'] = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'file_size': file_size
            }
            
        except Exception as e:
            validation_result['error'] = f"Validation error: {str(e)}"
        
        return validation_result
    
    def process_with_ivp_algorithms(self, frames: Optional[List[np.ndarray]] = None) -> List[FrameMetrics]:
        """
        Process frames through the IVP algorithm pipeline:
        frame difference → histogram comparison → blur detection
        
        Args:
            frames: List of frames to process (uses self.frames if None)
            
        Returns:
            List of FrameMetrics with all algorithm results
            
        Raises:
            AlgorithmProcessingError: If no frames available or algorithm processing fails
        """
        try:
            if frames is None:
                frames = self.frames
            
            if not frames:
                raise AlgorithmProcessingError("No frames available for processing")
            
            if len(frames) < 2:
                raise AlgorithmProcessingError("At least 2 frames required for processing", 
                                             details=f"Only {len(frames)} frames provided")
            
            logger.info(f"Starting IVP algorithm processing on {len(frames)} frames")
            
            # Initialize metrics list
            frame_metrics = []
            
            # Step 1: Calculate frame differences (MAD) with error handling
            logger.info("Step 1: Calculating frame differences using MAD...")
            try:
                frame_diff_scores = self.frame_diff_calc.process_frame_sequence(frames)
                if len(frame_diff_scores) != len(frames) - 1:
                    logger.warning(f"Frame difference scores length mismatch: expected {len(frames)-1}, got {len(frame_diff_scores)}")
            except Exception as e:
                logger.error(f"Frame difference calculation failed: {str(e)}")
                raise AlgorithmProcessingError("Frame difference calculation failed", "frame_difference", str(e))
            
            # Step 2: Calculate histogram comparisons (Chi-Square) with error handling
            logger.info("Step 2: Calculating histogram comparisons using Chi-Square...")
            try:
                histogram_scores = self.histogram_comp.process_frame_sequence(frames)
                if len(histogram_scores) != len(frames) - 1:
                    logger.warning(f"Histogram scores length mismatch: expected {len(frames)-1}, got {len(histogram_scores)}")
            except Exception as e:
                logger.error(f"Histogram comparison failed: {str(e)}")
                raise AlgorithmProcessingError("Histogram comparison failed", "histogram", str(e))
            
            # Step 3: Calculate blur scores (Variance of Laplacian) with error handling
            logger.info("Step 3: Calculating blur scores using Variance of Laplacian...")
            try:
                blur_scores = self.blur_detector.process_frame_sequence(frames)
                if len(blur_scores) != len(frames):
                    logger.warning(f"Blur scores length mismatch: expected {len(frames)}, got {len(blur_scores)}")
            except Exception as e:
                logger.error(f"Blur detection failed: {str(e)}")
                raise AlgorithmProcessingError("Blur detection failed", "blur", str(e))
            
            # Create FrameMetrics for each frame with error handling
            sample_rate = 1  # Default, can be parameterized later
            
            for i, frame in enumerate(frames):
                try:
                    timestamp = self.get_frame_timestamp(i, sample_rate)
                    
                    # Get scores for this frame with bounds checking
                    # Frame difference and histogram scores are between consecutive frames
                    # So frame i has score comparing it to frame i-1
                    diff_score = frame_diff_scores[i-1] if i > 0 and i-1 < len(frame_diff_scores) else 0.0
                    hist_score = histogram_scores[i-1] if i > 0 and i-1 < len(histogram_scores) else 0.0
                    blur_score = blur_scores[i] if i < len(blur_scores) else 0.0
                    
                    # Validate scores
                    if not isinstance(diff_score, (int, float)) or not np.isfinite(diff_score):
                        logger.warning(f"Invalid frame difference score for frame {i}: {diff_score}, using 0.0")
                        diff_score = 0.0
                    
                    if not isinstance(hist_score, (int, float)) or not np.isfinite(hist_score):
                        logger.warning(f"Invalid histogram score for frame {i}: {hist_score}, using 0.0")
                        hist_score = 0.0
                    
                    if not isinstance(blur_score, (int, float)) or not np.isfinite(blur_score):
                        logger.warning(f"Invalid blur score for frame {i}: {blur_score}, using 0.0")
                        blur_score = 0.0
                    
                    metrics = FrameMetrics(
                        frame_number=i,
                        timestamp=timestamp,
                        difference_score=float(diff_score),
                        histogram_score=float(hist_score),
                        blur_score=float(blur_score),
                        is_keyframe=False  # Will be set in keyframe selection
                    )
                    
                    frame_metrics.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Error creating metrics for frame {i}: {str(e)}")
                    # Create default metrics to avoid breaking the pipeline
                    metrics = FrameMetrics(
                        frame_number=i,
                        timestamp=self.get_frame_timestamp(i, sample_rate),
                        difference_score=0.0,
                        histogram_score=0.0,
                        blur_score=0.0,
                        is_keyframe=False
                    )
                    frame_metrics.append(metrics)
            
            if not frame_metrics:
                raise AlgorithmProcessingError("No frame metrics could be created")
            
            self.frame_metrics = frame_metrics
            logger.info(f"IVP algorithm processing completed for {len(frame_metrics)} frames")
            
            return frame_metrics
            
        except AlgorithmProcessingError:
            logger.error(f"Algorithm processing error: {traceback.format_exc()}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in IVP algorithm processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise AlgorithmProcessingError(f"IVP algorithm processing failed: {str(e)}", details=str(e))
    
    def update_algorithm_parameters(self, 
                                  frame_diff_threshold: Optional[float] = None,
                                  histogram_threshold: Optional[float] = None,
                                  blur_threshold: Optional[float] = None):
        """
        Update algorithm thresholds and reinitialize algorithm instances.
        
        Args:
            frame_diff_threshold: New threshold for frame difference detection
            histogram_threshold: New threshold for histogram comparison
            blur_threshold: New threshold for blur detection
        """
        if frame_diff_threshold is not None:
            self.frame_diff_threshold = frame_diff_threshold
            self.frame_diff_calc = FrameDifferenceCalculator(frame_diff_threshold)
            logger.info(f"Updated frame difference threshold to {frame_diff_threshold}")
        
        if histogram_threshold is not None:
            self.histogram_threshold = histogram_threshold
            self.histogram_comp = HistogramComparator(histogram_threshold)
            logger.info(f"Updated histogram threshold to {histogram_threshold}")
        
        if blur_threshold is not None:
            self.blur_threshold = blur_threshold
            self.blur_detector = BlurDetector(blur_threshold)
            logger.info(f"Updated blur threshold to {blur_threshold}")
    
    def get_algorithm_scores_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of algorithm scores across all processed frames.
        
        Returns:
            Dictionary with score statistics for each algorithm
        """
        if not self.frame_metrics:
            return {}
        
        diff_scores = [m.difference_score for m in self.frame_metrics]
        hist_scores = [m.histogram_score for m in self.frame_metrics]
        blur_scores = [m.blur_score for m in self.frame_metrics]
        
        summary = {
            'frame_difference': {
                'min': min(diff_scores) if diff_scores else 0,
                'max': max(diff_scores) if diff_scores else 0,
                'mean': np.mean(diff_scores) if diff_scores else 0,
                'std': np.std(diff_scores) if diff_scores else 0,
                'threshold': self.frame_diff_threshold
            },
            'histogram_comparison': {
                'min': min(hist_scores) if hist_scores else 0,
                'max': max(hist_scores) if hist_scores else 0,
                'mean': np.mean(hist_scores) if hist_scores else 0,
                'std': np.std(hist_scores) if hist_scores else 0,
                'threshold': self.histogram_threshold
            },
            'blur_detection': {
                'min': min(blur_scores) if blur_scores else 0,
                'max': max(blur_scores) if blur_scores else 0,
                'mean': np.mean(blur_scores) if blur_scores else 0,
                'std': np.std(blur_scores) if blur_scores else 0,
                'threshold': self.blur_threshold
            }
        }
        
        return summary
    
    def filter_frames_by_algorithm(self, algorithm: str, above_threshold: bool = True) -> List[int]:
        """
        Filter frame indices based on a specific algorithm's results.
        
        Args:
            algorithm: Algorithm name ('frame_diff', 'histogram', 'blur')
            above_threshold: If True, return frames above threshold; if False, below threshold
            
        Returns:
            List of frame indices that meet the criteria
        """
        if not self.frame_metrics:
            return []
        
        filtered_indices = []
        
        for metrics in self.frame_metrics:
            if algorithm == 'frame_diff':
                score = metrics.difference_score
                threshold = self.frame_diff_threshold
            elif algorithm == 'histogram':
                score = metrics.histogram_score
                threshold = self.histogram_threshold
            elif algorithm == 'blur':
                score = metrics.blur_score
                threshold = self.blur_threshold
            else:
                continue
            
            if (above_threshold and score > threshold) or (not above_threshold and score <= threshold):
                filtered_indices.append(metrics.frame_number)
        
        return filtered_indices
    
    def select_keyframes(self, target_reduction: float = 0.75) -> List[FrameMetrics]:
        """
        Select keyframes based on combined IVP metrics to achieve target frame reduction.
        
        Args:
            target_reduction: Target reduction percentage (0.7-0.8 for 70-80% reduction)
            
        Returns:
            List of FrameMetrics for selected keyframes
            
        Raises:
            RuntimeError: If no frame metrics available
        """
        if not self.frame_metrics:
            raise RuntimeError("No frame metrics available. Run process_with_ivp_algorithms() first.")
        
        logger.info(f"Starting keyframe selection with target reduction: {target_reduction:.1%}")
        
        total_frames = len(self.frame_metrics)
        target_keyframes = max(1, int(total_frames * (1 - target_reduction)))
        
        logger.info(f"Target keyframes: {target_keyframes} out of {total_frames} frames")
        
        # Step 1: Filter out blurry frames (below blur threshold)
        sharp_candidates = []
        for metrics in self.frame_metrics:
            if metrics.blur_score > self.blur_threshold:
                sharp_candidates.append(metrics)
        
        logger.info(f"Sharp frame candidates after blur filtering: {len(sharp_candidates)}")
        
        # If we have too few sharp frames, lower the blur threshold
        if len(sharp_candidates) < target_keyframes:
            logger.warning(f"Not enough sharp frames ({len(sharp_candidates)}), including some blurry frames")
            # Sort all frames by blur score and take the sharpest ones
            all_sorted_by_blur = sorted(self.frame_metrics, key=lambda x: x.blur_score, reverse=True)
            sharp_candidates = all_sorted_by_blur[:min(target_keyframes * 2, total_frames)]
        
        # Step 2: Apply frame difference and histogram filtering
        significant_change_candidates = []
        
        for metrics in sharp_candidates:
            # Include first frame as it's always a potential keyframe
            if metrics.frame_number == 0:
                significant_change_candidates.append(metrics)
                continue
            
            # Include frames with significant changes (either frame diff or histogram)
            has_frame_diff = metrics.difference_score > self.frame_diff_threshold
            has_histogram_change = metrics.histogram_score > self.histogram_threshold
            
            if has_frame_diff or has_histogram_change:
                significant_change_candidates.append(metrics)
        
        logger.info(f"Candidates after change detection: {len(significant_change_candidates)}")
        
        # Step 3: If we still have too many candidates, rank and select the best ones
        if len(significant_change_candidates) > target_keyframes:
            keyframes = self._rank_and_select_keyframes(significant_change_candidates, target_keyframes)
        else:
            # If we have too few candidates, add more based on ranking
            if len(significant_change_candidates) < target_keyframes:
                additional_needed = target_keyframes - len(significant_change_candidates)
                remaining_candidates = [m for m in sharp_candidates if m not in significant_change_candidates]
                
                if remaining_candidates:
                    additional_keyframes = self._rank_and_select_keyframes(remaining_candidates, additional_needed)
                    keyframes = significant_change_candidates + additional_keyframes
                else:
                    keyframes = significant_change_candidates
            else:
                keyframes = significant_change_candidates
        
        # Step 4: Ensure temporal distribution
        keyframes = self._ensure_temporal_distribution(keyframes, target_keyframes)
        
        # Step 5: Mark selected frames as keyframes
        keyframe_numbers = {kf.frame_number for kf in keyframes}
        for metrics in self.frame_metrics:
            metrics.is_keyframe = metrics.frame_number in keyframe_numbers
        
        # Sort keyframes by frame number
        keyframes.sort(key=lambda x: x.frame_number)
        
        reduction_achieved = 1 - (len(keyframes) / total_frames)
        logger.info(f"Keyframe selection completed: {len(keyframes)} keyframes selected")
        logger.info(f"Reduction achieved: {reduction_achieved:.1%}")
        
        return keyframes
    
    def _rank_and_select_keyframes(self, candidates: List[FrameMetrics], target_count: int) -> List[FrameMetrics]:
        """
        Rank candidates by combined IVP scores and select the best ones.
        
        Args:
            candidates: List of candidate FrameMetrics
            target_count: Number of keyframes to select
            
        Returns:
            List of selected FrameMetrics
        """
        if not candidates:
            return []
        
        # Calculate combined scores with weights
        # Higher weights for more important criteria
        frame_diff_weight = 0.3
        histogram_weight = 0.4  # Shot boundaries are important
        blur_weight = 0.3       # Sharpness is important
        
        scored_candidates = []
        
        for metrics in candidates:
            # Normalize scores to 0-1 range for fair weighting
            # Frame difference and histogram scores are already roughly 0-1
            # Blur scores need normalization
            max_blur = max(c.blur_score for c in candidates)
            min_blur = min(c.blur_score for c in candidates)
            blur_range = max_blur - min_blur if max_blur > min_blur else 1
            
            normalized_blur = (metrics.blur_score - min_blur) / blur_range
            
            # Calculate combined score
            combined_score = (
                frame_diff_weight * metrics.difference_score +
                histogram_weight * metrics.histogram_score +
                blur_weight * normalized_blur
            )
            
            scored_candidates.append((metrics, combined_score))
        
        # Sort by combined score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top candidates
        selected = [candidate[0] for candidate in scored_candidates[:target_count]]
        
        return selected
    
    def _ensure_temporal_distribution(self, keyframes: List[FrameMetrics], target_count: int) -> List[FrameMetrics]:
        """
        Ensure keyframes are well distributed temporally across the video.
        
        Args:
            keyframes: List of selected keyframes
            target_count: Target number of keyframes
            
        Returns:
            List of keyframes with better temporal distribution
        """
        if len(keyframes) <= 1 or not self.frame_metrics:
            return keyframes
        
        total_frames = len(self.frame_metrics)
        
        # Sort keyframes by frame number
        keyframes.sort(key=lambda x: x.frame_number)
        
        # Check for temporal gaps
        ideal_gap = total_frames / target_count
        min_gap = ideal_gap * 0.3  # Minimum gap between keyframes
        
        # Remove keyframes that are too close together
        filtered_keyframes = [keyframes[0]]  # Always keep first keyframe
        
        for kf in keyframes[1:]:
            last_frame_num = filtered_keyframes[-1].frame_number
            if kf.frame_number - last_frame_num >= min_gap:
                filtered_keyframes.append(kf)
        
        # If we removed too many, add back the highest scoring ones
        if len(filtered_keyframes) < target_count * 0.8:  # Allow some flexibility
            removed_keyframes = [kf for kf in keyframes if kf not in filtered_keyframes]
            if removed_keyframes:
                # Sort removed keyframes by their combined importance
                additional_needed = min(len(removed_keyframes), 
                                      int(target_count * 0.8) - len(filtered_keyframes))
                
                # Add back the most important ones that don't violate minimum gap too severely
                for kf in removed_keyframes[:additional_needed]:
                    # Check if adding this keyframe creates reasonable distribution
                    insert_pos = 0
                    for i, existing_kf in enumerate(filtered_keyframes):
                        if kf.frame_number < existing_kf.frame_number:
                            insert_pos = i
                            break
                        insert_pos = i + 1
                    
                    filtered_keyframes.insert(insert_pos, kf)
        
        return filtered_keyframes
    
    def get_keyframe_summary(self) -> Dict[str, Any]:
        """
        Get summary information about selected keyframes.
        
        Returns:
            Dictionary with keyframe statistics and information
        """
        if not self.frame_metrics:
            return {}
        
        keyframes = [m for m in self.frame_metrics if m.is_keyframe]
        total_frames = len(self.frame_metrics)
        
        if not keyframes:
            return {
                'total_frames': total_frames,
                'keyframes_selected': 0,
                'reduction_percentage': 0.0,
                'keyframe_indices': [],
                'temporal_distribution': []
            }
        
        keyframe_indices = [kf.frame_number for kf in keyframes]
        reduction_percentage = (1 - len(keyframes) / total_frames) * 100
        
        # Calculate temporal distribution
        temporal_gaps = []
        for i in range(1, len(keyframes)):
            gap = keyframes[i].frame_number - keyframes[i-1].frame_number
            temporal_gaps.append(gap)
        
        summary = {
            'total_frames': total_frames,
            'keyframes_selected': len(keyframes),
            'reduction_percentage': reduction_percentage,
            'keyframe_indices': keyframe_indices,
            'temporal_distribution': {
                'gaps': temporal_gaps,
                'mean_gap': np.mean(temporal_gaps) if temporal_gaps else 0,
                'std_gap': np.std(temporal_gaps) if temporal_gaps else 0,
                'min_gap': min(temporal_gaps) if temporal_gaps else 0,
                'max_gap': max(temporal_gaps) if temporal_gaps else 0
            },
            'algorithm_stats': {
                'avg_frame_diff_score': np.mean([kf.difference_score for kf in keyframes]),
                'avg_histogram_score': np.mean([kf.histogram_score for kf in keyframes]),
                'avg_blur_score': np.mean([kf.blur_score for kf in keyframes])
            }
        }
        
        return summary
    
    def process_video_complete(self, video_path: str, 
                             target_reduction: float = 0.75,
                             max_frames: Optional[int] = None,
                             sample_rate: int = 1) -> Dict[str, Any]:
        """
        Complete video processing pipeline from loading to keyframe selection.
        
        Args:
            video_path: Path to video file
            target_reduction: Target frame reduction percentage
            max_frames: Maximum frames to process (None for all)
            sample_rate: Frame sampling rate
            
        Returns:
            Dictionary with complete processing results
        """
        try:
            # Step 1: Load and validate video
            logger.info("Step 1: Loading video...")
            self.load_video(video_path)
            
            # Step 2: Extract frames
            logger.info("Step 2: Extracting frames...")
            frames = self.extract_frames(max_frames=max_frames, sample_rate=sample_rate)
            
            # Step 3: Process with IVP algorithms
            logger.info("Step 3: Processing with IVP algorithms...")
            self.process_with_ivp_algorithms(frames)
            
            # Step 4: Select keyframes
            logger.info("Step 4: Selecting keyframes...")
            keyframes = self.select_keyframes(target_reduction=target_reduction)
            
            # Step 5: Generate results
            results = {
                'success': True,
                'video_properties': {
                    'fps': self.fps,
                    'total_frames': self.total_frames,
                    'processed_frames': len(frames)
                },
                'keyframes': keyframes,
                'frame_metrics': self.frame_metrics,
                'summary': self.get_keyframe_summary(),
                'algorithm_summary': self.get_algorithm_scores_summary()
            }
            
            logger.info("Video processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'keyframes': [],
                'frame_metrics': [],
                'summary': {},
                'algorithm_summary': {}
            }
    
    def process_video_complete(self, video_path: str, target_reduction: float = 0.75, 
                              max_frames: Optional[int] = None, sample_rate: int = 1) -> Dict[str, Any]:
        """
        Complete video processing pipeline from loading to keyframe selection.
        
        Args:
            video_path: Path to video file
            target_reduction: Target frame reduction percentage (0.7-0.8)
            max_frames: Maximum frames to process (None for all)
            sample_rate: Frame sampling rate
            
        Returns:
            Dictionary containing all processing results
            
        Raises:
            VideoProcessingError: If any step in the pipeline fails
        """
        try:
            logger.info(f"Starting complete video processing pipeline for: {video_path}")
            
            # Step 1: Load video
            success = self.load_video(video_path)
            if not success:
                raise VideoProcessingError("Failed to load video file")
            
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            video_properties = {
                'fps': self.fps,
                'total_frames': self.total_frames,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': self.total_frames / self.fps if self.fps > 0 else 0
            }
            cap.release()
            
            # Step 2: Extract frames
            frames = self.extract_frames(max_frames=max_frames, sample_rate=sample_rate)
            if not frames:
                raise VideoProcessingError("No frames could be extracted from video")
            
            logger.info(f"Extracted {len(frames)} frames for processing")
            
            # Step 3: Process with IVP algorithms
            all_metrics = self.process_with_ivp_algorithms(frames)
            if not all_metrics:
                raise VideoProcessingError("IVP algorithm processing failed")
            
            logger.info(f"Processed {len(all_metrics)} frames with IVP algorithms")
            
            # Step 4: Select keyframes
            keyframe_metrics = self.select_keyframes(target_reduction=target_reduction)
            if not keyframe_metrics:
                raise VideoProcessingError("Keyframe selection failed")
            
            logger.info(f"Selected {len(keyframe_metrics)} keyframes")
            
            # Return complete results
            results = {
                'keyframes': keyframe_metrics,
                'all_metrics': all_metrics,
                'frames': frames,
                'video_properties': video_properties,
                'processing_summary': {
                    'total_frames_processed': len(frames),
                    'keyframes_selected': len(keyframe_metrics),
                    'reduction_achieved': 1 - (len(keyframe_metrics) / len(frames)),
                    'sample_rate_used': sample_rate
                }
            }
            
            logger.info("Complete video processing pipeline finished successfully")
            return results
            
        except VideoProcessingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in complete video processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise VideoProcessingError(f"Complete video processing failed: {str(e)}", details=str(e))