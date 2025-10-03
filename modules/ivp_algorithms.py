"""
IVP Algorithms Module

Implementation of core Image and Video Processing techniques:
- Frame Difference using MAD (Mean Absolute Difference)
- Histogram Comparison using Chi-Square distance
- Blur Detection using Variance of Laplacian
"""

import cv2
import numpy as np
import logging
import traceback
from typing import Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)


class IVPAlgorithmError(Exception):
    """Base exception for IVP algorithm errors."""
    def __init__(self, message: str, algorithm: str = None, details: str = None):
        self.message = message
        self.algorithm = algorithm
        self.details = details
        super().__init__(self.message)


class FrameDifferenceError(IVPAlgorithmError):
    """Exception for frame difference calculation errors."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message, "FrameDifference", details)


class HistogramError(IVPAlgorithmError):
    """Exception for histogram comparison errors."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message, "Histogram", details)


class BlurDetectionError(IVPAlgorithmError):
    """Exception for blur detection errors."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message, "BlurDetection", details)


class FrameDifferenceCalculator:
    """
    Calculates Mean Absolute Difference (MAD) between consecutive frames
    for redundancy removal in video processing.
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize the FrameDifferenceCalculator.
        
        Args:
            threshold: Minimum MAD score to consider frames as different (0.0-1.0)
        """
        self.threshold = threshold
    
    def calculate_mad(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate Mean Absolute Difference between two frames.
        
        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array
            
        Returns:
            MAD score normalized to 0-1 range
            
        Raises:
            FrameDifferenceError: If frames are invalid or calculation fails
        """
        try:
            # Validate input frames
            if frame1 is None or frame2 is None:
                raise FrameDifferenceError("Input frames cannot be None")
            
            if not isinstance(frame1, np.ndarray) or not isinstance(frame2, np.ndarray):
                raise FrameDifferenceError("Input frames must be numpy arrays")
            
            if frame1.size == 0 or frame2.size == 0:
                raise FrameDifferenceError("Input frames cannot be empty")
            
            if frame1.shape != frame2.shape:
                raise FrameDifferenceError(f"Frames must have the same dimensions: {frame1.shape} vs {frame2.shape}")
            
            # Convert to grayscale if color frames
            try:
                if len(frame1.shape) == 3:
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                else:
                    gray1, gray2 = frame1, frame2
            except Exception as e:
                raise FrameDifferenceError(f"Failed to convert frames to grayscale: {str(e)}")
            
            # Validate grayscale conversion
            if gray1 is None or gray2 is None or gray1.size == 0 or gray2.size == 0:
                raise FrameDifferenceError("Grayscale conversion resulted in empty frames")
            
            # Calculate absolute difference
            try:
                diff = cv2.absdiff(gray1, gray2)
                if diff is None or diff.size == 0:
                    raise FrameDifferenceError("Frame difference calculation resulted in empty array")
            except Exception as e:
                raise FrameDifferenceError(f"Failed to calculate frame difference: {str(e)}")
            
            # Calculate mean absolute difference and normalize to 0-1
            try:
                mad_score = np.mean(diff) / 255.0
                
                # Validate result
                if not np.isfinite(mad_score):
                    logger.warning(f"Non-finite MAD score: {mad_score}, returning 0.0")
                    mad_score = 0.0
                elif mad_score < 0:
                    logger.warning(f"Negative MAD score: {mad_score}, clamping to 0.0")
                    mad_score = 0.0
                elif mad_score > 1:
                    logger.warning(f"MAD score > 1: {mad_score}, clamping to 1.0")
                    mad_score = 1.0
                
                return float(mad_score)
                
            except Exception as e:
                raise FrameDifferenceError(f"Failed to calculate mean difference: {str(e)}")
            
        except FrameDifferenceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in MAD calculation: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise FrameDifferenceError(f"MAD calculation failed: {str(e)}", str(e))
    
    def is_significant_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        Determine if there's a significant change between two frames.
        
        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array
            
        Returns:
            True if change is significant (above threshold)
        """
        mad_score = self.calculate_mad(frame1, frame2)
        return mad_score > self.threshold
    
    def process_frame_sequence(self, frames: list) -> list:
        """
        Process a sequence of frames and return MAD scores for consecutive pairs.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            List of MAD scores (length = len(frames) - 1)
            
        Raises:
            FrameDifferenceError: If frame sequence processing fails
        """
        try:
            if not frames:
                raise FrameDifferenceError("Frame list cannot be empty")
            
            if len(frames) < 2:
                logger.warning("Less than 2 frames provided, returning empty score list")
                return []
            
            mad_scores = []
            failed_calculations = 0
            max_failures = len(frames) // 4  # Allow up to 25% failures
            
            for i in range(len(frames) - 1):
                try:
                    score = self.calculate_mad(frames[i], frames[i + 1])
                    mad_scores.append(score)
                except Exception as e:
                    failed_calculations += 1
                    logger.warning(f"Failed to calculate MAD for frames {i}-{i+1}: {str(e)}")
                    
                    # Use default score for failed calculations
                    mad_scores.append(0.0)
                    
                    if failed_calculations > max_failures:
                        raise FrameDifferenceError(f"Too many failed MAD calculations: {failed_calculations}/{len(frames)-1}")
            
            if failed_calculations > 0:
                logger.warning(f"Frame difference processing completed with {failed_calculations} failures")
            
            return mad_scores
            
        except FrameDifferenceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in frame sequence processing: {str(e)}")
            raise FrameDifferenceError(f"Frame sequence processing failed: {str(e)}", str(e))


class HistogramComparator:
    """
    Computes histograms and Chi-Square distance between frames
    for shot boundary detection in video processing.
    """
    
    def __init__(self, threshold: float = 0.5, bins: int = 256):
        """
        Initialize the HistogramComparator.
        
        Args:
            threshold: Minimum Chi-Square distance to consider frames as different shots
            bins: Number of histogram bins (default 256 for 8-bit images)
        """
        self.threshold = threshold
        self.bins = bins
    
    def calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Calculate histogram for a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Normalized histogram as numpy array
            
        Raises:
            HistogramError: If histogram calculation fails
        """
        try:
            # Validate input frame
            if frame is None:
                raise HistogramError("Input frame cannot be None")
            
            if not isinstance(frame, np.ndarray):
                raise HistogramError("Input frame must be a numpy array")
            
            if frame.size == 0:
                raise HistogramError("Input frame cannot be empty")
            
            # Convert to grayscale if color frame
            try:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
            except Exception as e:
                raise HistogramError(f"Failed to convert frame to grayscale: {str(e)}")
            
            # Validate grayscale conversion
            if gray is None or gray.size == 0:
                raise HistogramError("Grayscale conversion resulted in empty frame")
            
            # Calculate histogram
            try:
                hist = cv2.calcHist([gray], [0], None, [self.bins], [0, 256])
                if hist is None or hist.size == 0:
                    raise HistogramError("Histogram calculation resulted in empty array")
            except Exception as e:
                raise HistogramError(f"Failed to calculate histogram: {str(e)}")
            
            # Normalize histogram
            try:
                hist = hist.flatten()
                hist_sum = np.sum(hist)
                
                if hist_sum == 0:
                    logger.warning("Histogram sum is zero, using uniform distribution")
                    hist = np.ones(self.bins) / self.bins
                else:
                    hist = hist / (hist_sum + 1e-10)  # Add small epsilon to avoid division by zero
                
                # Validate result
                if not np.all(np.isfinite(hist)):
                    logger.warning("Non-finite values in histogram, using uniform distribution")
                    hist = np.ones(self.bins) / self.bins
                
                return hist
                
            except Exception as e:
                raise HistogramError(f"Failed to normalize histogram: {str(e)}")
            
        except HistogramError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in histogram calculation: {str(e)}")
            raise HistogramError(f"Histogram calculation failed: {str(e)}", str(e))
    
    def calculate_chi_square_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Calculate Chi-Square distance between two histograms.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
            
        Returns:
            Chi-Square distance (0 = identical, higher = more different)
            
        Raises:
            HistogramError: If chi-square calculation fails
        """
        try:
            # Validate input histograms
            if hist1 is None or hist2 is None:
                raise HistogramError("Input histograms cannot be None")
            
            if not isinstance(hist1, np.ndarray) or not isinstance(hist2, np.ndarray):
                raise HistogramError("Input histograms must be numpy arrays")
            
            if hist1.size == 0 or hist2.size == 0:
                raise HistogramError("Input histograms cannot be empty")
            
            if hist1.shape != hist2.shape:
                raise HistogramError(f"Histograms must have the same shape: {hist1.shape} vs {hist2.shape}")
            
            # Validate histogram values
            if not np.all(np.isfinite(hist1)) or not np.all(np.isfinite(hist2)):
                raise HistogramError("Histograms contain non-finite values")
            
            if np.any(hist1 < 0) or np.any(hist2 < 0):
                raise HistogramError("Histograms contain negative values")
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            hist1_safe = hist1 + epsilon
            hist2_safe = hist2 + epsilon
            
            # Calculate Chi-Square distance
            try:
                numerator = (hist1_safe - hist2_safe) ** 2
                denominator = hist1_safe + hist2_safe
                
                # Check for division by zero (shouldn't happen with epsilon, but be safe)
                if np.any(denominator == 0):
                    logger.warning("Zero denominator in chi-square calculation, using fallback")
                    return 0.0
                
                chi_square = 0.5 * np.sum(numerator / denominator)
                
                # Validate result
                if not np.isfinite(chi_square):
                    logger.warning(f"Non-finite chi-square distance: {chi_square}, returning 0.0")
                    chi_square = 0.0
                elif chi_square < 0:
                    logger.warning(f"Negative chi-square distance: {chi_square}, returning 0.0")
                    chi_square = 0.0
                
                return float(chi_square)
                
            except Exception as e:
                raise HistogramError(f"Failed to calculate chi-square distance: {str(e)}")
            
        except HistogramError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chi-square calculation: {str(e)}")
            raise HistogramError(f"Chi-square calculation failed: {str(e)}", str(e))
    
    def compare_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compare two frames using histogram Chi-Square distance.
        
        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array
            
        Returns:
            Chi-Square distance between frame histograms
        """
        hist1 = self.calculate_histogram(frame1)
        hist2 = self.calculate_histogram(frame2)
        
        return self.calculate_chi_square_distance(hist1, hist2)
    
    def is_shot_boundary(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        Determine if there's a shot boundary between two frames.
        
        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array
            
        Returns:
            True if shot boundary detected (distance above threshold)
        """
        distance = self.compare_frames(frame1, frame2)
        return distance > self.threshold
    
    def process_frame_sequence(self, frames: list) -> list:
        """
        Process a sequence of frames and return Chi-Square distances for consecutive pairs.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            List of Chi-Square distances (length = len(frames) - 1)
        """
        if len(frames) < 2:
            return []
        
        distances = []
        for i in range(len(frames) - 1):
            distance = self.compare_frames(frames[i], frames[i + 1])
            distances.append(distance)
        
        return distances


class BlurDetector:
    """
    Calculates blur scores using Variance of Laplacian method
    for quality filtering to identify sharp vs blurry frames.
    """
    
    def __init__(self, threshold: float = 100.0):
        """
        Initialize the BlurDetector.
        
        Args:
            threshold: Minimum variance of Laplacian to consider frame as sharp
                      (typical range: 10-500, higher = sharper required)
        """
        self.threshold = threshold
    
    def calculate_blur_score(self, frame: np.ndarray) -> float:
        """
        Calculate blur score using Variance of Laplacian.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Blur score (higher = sharper, lower = blurrier)
            
        Raises:
            BlurDetectionError: If blur score calculation fails
        """
        try:
            # Validate input frame
            if frame is None:
                raise BlurDetectionError("Input frame cannot be None")
            
            if not isinstance(frame, np.ndarray):
                raise BlurDetectionError("Input frame must be a numpy array")
            
            if frame.size == 0:
                raise BlurDetectionError("Input frame cannot be empty")
            
            # Convert to grayscale if color frame
            try:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
            except Exception as e:
                raise BlurDetectionError(f"Failed to convert frame to grayscale: {str(e)}")
            
            # Validate grayscale conversion
            if gray is None or gray.size == 0:
                raise BlurDetectionError("Grayscale conversion resulted in empty frame")
            
            # Apply Laplacian operator
            try:
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                if laplacian is None or laplacian.size == 0:
                    raise BlurDetectionError("Laplacian operation resulted in empty array")
            except Exception as e:
                raise BlurDetectionError(f"Failed to apply Laplacian operator: {str(e)}")
            
            # Calculate variance of Laplacian
            try:
                variance = laplacian.var()
                
                # Validate result
                if not np.isfinite(variance):
                    logger.warning(f"Non-finite blur score: {variance}, returning 0.0")
                    variance = 0.0
                elif variance < 0:
                    logger.warning(f"Negative blur score: {variance}, returning 0.0")
                    variance = 0.0
                
                return float(variance)
                
            except Exception as e:
                raise BlurDetectionError(f"Failed to calculate variance: {str(e)}")
            
        except BlurDetectionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in blur score calculation: {str(e)}")
            raise BlurDetectionError(f"Blur score calculation failed: {str(e)}", str(e))
    
    def is_sharp(self, frame: np.ndarray) -> bool:
        """
        Determine if a frame is sharp (not blurry).
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            True if frame is sharp (blur score above threshold)
        """
        blur_score = self.calculate_blur_score(frame)
        return blur_score > self.threshold
    
    def is_blurry(self, frame: np.ndarray) -> bool:
        """
        Determine if a frame is blurry.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            True if frame is blurry (blur score below threshold)
        """
        return not self.is_sharp(frame)
    
    def filter_sharp_frames(self, frames: list) -> list:
        """
        Filter a list of frames to keep only sharp ones.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            List of tuples (frame_index, frame) for sharp frames only
        """
        sharp_frames = []
        for i, frame in enumerate(frames):
            if self.is_sharp(frame):
                sharp_frames.append((i, frame))
        
        return sharp_frames
    
    def process_frame_sequence(self, frames: list) -> list:
        """
        Process a sequence of frames and return blur scores for each frame.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            List of blur scores (same length as input frames)
            
        Raises:
            BlurDetectionError: If frame sequence processing fails
        """
        try:
            if not frames:
                logger.warning("Empty frame list provided, returning empty score list")
                return []
            
            blur_scores = []
            failed_calculations = 0
            max_failures = len(frames) // 4  # Allow up to 25% failures
            
            for i, frame in enumerate(frames):
                try:
                    score = self.calculate_blur_score(frame)
                    blur_scores.append(score)
                except Exception as e:
                    failed_calculations += 1
                    logger.warning(f"Failed to calculate blur score for frame {i}: {str(e)}")
                    
                    # Use default score for failed calculations
                    blur_scores.append(0.0)
                    
                    if failed_calculations > max_failures:
                        raise BlurDetectionError(f"Too many failed blur calculations: {failed_calculations}/{len(frames)}")
            
            if failed_calculations > 0:
                logger.warning(f"Blur detection processing completed with {failed_calculations} failures")
            
            return blur_scores
            
        except BlurDetectionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in blur frame sequence processing: {str(e)}")
            raise BlurDetectionError(f"Blur frame sequence processing failed: {str(e)}", str(e))
    
    def get_sharpness_ranking(self, frames: list) -> list:
        """
        Rank frames by sharpness (blur score).
        
        Args:
            frames: List of frame arrays
            
        Returns:
            List of tuples (frame_index, blur_score) sorted by sharpness (descending)
        """
        if not frames:
            return []
        
        scores_with_indices = []
        for i, frame in enumerate(frames):
            score = self.calculate_blur_score(frame)
            scores_with_indices.append((i, score))
        
        # Sort by blur score in descending order (sharpest first)
        scores_with_indices.sort(key=lambda x: x[1], reverse=True)
        
        return scores_with_indices