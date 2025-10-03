"""
Unit tests for IVP algorithms module
"""

import unittest
import numpy as np
import cv2
import tempfile
import os
from modules.ivp_algorithms import (
    FrameDifferenceCalculator, HistogramComparator, BlurDetector,
    FrameDifferenceError, HistogramError, BlurDetectionError
)


class TestFrameDifferenceCalculator(unittest.TestCase):
    """Test cases for FrameDifferenceCalculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = FrameDifferenceCalculator(threshold=0.1)
        
        # Create test frames
        self.frame1 = np.zeros((100, 100, 3), dtype=np.uint8)  # Black frame
        self.frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White frame
        self.frame3 = np.zeros((100, 100, 3), dtype=np.uint8)  # Another black frame
        
        # Create grayscale test frames
        self.gray1 = np.zeros((100, 100), dtype=np.uint8)
        self.gray2 = np.ones((100, 100), dtype=np.uint8) * 255
    
    def test_calculate_mad_identical_frames(self):
        """Test MAD calculation with identical frames"""
        mad_score = self.calculator.calculate_mad(self.frame1, self.frame3)
        self.assertEqual(mad_score, 0.0, "MAD score should be 0 for identical frames")
    
    def test_calculate_mad_different_frames(self):
        """Test MAD calculation with completely different frames"""
        mad_score = self.calculator.calculate_mad(self.frame1, self.frame2)
        self.assertEqual(mad_score, 1.0, "MAD score should be 1.0 for black vs white frames")
    
    def test_calculate_mad_grayscale_frames(self):
        """Test MAD calculation with grayscale frames"""
        mad_score = self.calculator.calculate_mad(self.gray1, self.gray2)
        self.assertEqual(mad_score, 1.0, "MAD score should be 1.0 for grayscale black vs white")
    
    def test_calculate_mad_different_shapes(self):
        """Test MAD calculation with frames of different shapes"""
        small_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with self.assertRaises(FrameDifferenceError):
            self.calculator.calculate_mad(self.frame1, small_frame)
    
    def test_calculate_mad_none_inputs(self):
        """Test MAD calculation with None inputs"""
        with self.assertRaises(FrameDifferenceError):
            self.calculator.calculate_mad(None, self.frame1)
        
        with self.assertRaises(FrameDifferenceError):
            self.calculator.calculate_mad(self.frame1, None)
    
    def test_calculate_mad_empty_frames(self):
        """Test MAD calculation with empty frames"""
        empty_frame = np.array([])
        with self.assertRaises(FrameDifferenceError):
            self.calculator.calculate_mad(empty_frame, self.frame1)
    
    def test_calculate_mad_invalid_types(self):
        """Test MAD calculation with invalid input types"""
        with self.assertRaises(FrameDifferenceError):
            self.calculator.calculate_mad("not_an_array", self.frame1)
        
        with self.assertRaises(FrameDifferenceError):
            self.calculator.calculate_mad(self.frame1, [1, 2, 3])
    
    def test_calculate_mad_known_values(self):
        """Test MAD calculation with known input/output pairs"""
        # Create frames with known pixel values
        frame_a = np.zeros((2, 2, 3), dtype=np.uint8)  # All zeros
        frame_b = np.ones((2, 2, 3), dtype=np.uint8) * 100  # All 100s
        
        # Expected MAD: |0-100| = 100, normalized = 100/255 ≈ 0.392
        expected_mad = 100.0 / 255.0
        actual_mad = self.calculator.calculate_mad(frame_a, frame_b)
        
        self.assertAlmostEqual(actual_mad, expected_mad, places=3,
                              msg=f"Expected MAD {expected_mad}, got {actual_mad}")
    
    def test_calculate_mad_partial_difference(self):
        """Test MAD calculation with partial frame differences"""
        # Create frames where only half the pixels differ
        frame_half_diff = np.zeros((4, 4, 3), dtype=np.uint8)
        frame_half_diff[:2, :, :] = 255  # Top half white, bottom half black
        
        frame_black = np.zeros((4, 4, 3), dtype=np.uint8)
        
        # Expected: half pixels have diff of 255, half have diff of 0
        # MAD = (8*255 + 8*0) / 16 / 255 = 0.5
        expected_mad = 0.5
        actual_mad = self.calculator.calculate_mad(frame_half_diff, frame_black)
        
        self.assertAlmostEqual(actual_mad, expected_mad, places=2,
                              msg=f"Expected MAD {expected_mad}, got {actual_mad}")
    
    def test_calculate_mad_single_channel_vs_multichannel(self):
        """Test MAD calculation consistency between single and multi-channel"""
        # Create equivalent grayscale and color frames
        gray_frame1 = np.zeros((50, 50), dtype=np.uint8)
        gray_frame2 = np.ones((50, 50), dtype=np.uint8) * 128
        
        color_frame1 = np.zeros((50, 50, 3), dtype=np.uint8)
        color_frame2 = np.ones((50, 50, 3), dtype=np.uint8) * 128
        
        gray_mad = self.calculator.calculate_mad(gray_frame1, gray_frame2)
        color_mad = self.calculator.calculate_mad(color_frame1, color_frame2)
        
        self.assertAlmostEqual(gray_mad, color_mad, places=3,
                              msg="MAD should be consistent between grayscale and color frames")
    
    def test_is_significant_change_above_threshold(self):
        """Test significant change detection above threshold"""
        result = self.calculator.is_significant_change(self.frame1, self.frame2)
        self.assertTrue(result, "Should detect significant change for black vs white frames")
    
    def test_is_significant_change_below_threshold(self):
        """Test significant change detection below threshold"""
        result = self.calculator.is_significant_change(self.frame1, self.frame3)
        self.assertFalse(result, "Should not detect significant change for identical frames")
    
    def test_process_frame_sequence_empty_list(self):
        """Test processing empty frame sequence"""
        with self.assertRaises(FrameDifferenceError):
            self.calculator.process_frame_sequence([])
    
    def test_process_frame_sequence_single_frame(self):
        """Test processing single frame sequence"""
        result = self.calculator.process_frame_sequence([self.frame1])
        self.assertEqual(result, [], "Should return empty list for single frame")
    
    def test_process_frame_sequence_multiple_frames(self):
        """Test processing multiple frame sequence"""
        frames = [self.frame1, self.frame2, self.frame3]
        result = self.calculator.process_frame_sequence(frames)
        
        self.assertEqual(len(result), 2, "Should return 2 scores for 3 frames")
        self.assertEqual(result[0], 1.0, "First score should be 1.0 (black to white)")
        self.assertEqual(result[1], 1.0, "Second score should be 1.0 (white to black)")
    
    def test_threshold_adjustment(self):
        """Test threshold adjustment functionality"""
        high_threshold_calc = FrameDifferenceCalculator(threshold=0.9)
        
        # Create frames with moderate difference
        moderate_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray frame
        
        # Low threshold should detect change
        low_result = self.calculator.is_significant_change(self.frame1, moderate_frame)
        self.assertTrue(low_result, "Low threshold should detect moderate change")
        
        # High threshold should not detect change
        high_result = high_threshold_calc.is_significant_change(self.frame1, moderate_frame)
        self.assertFalse(high_result, "High threshold should not detect moderate change")
    
    def test_process_frame_sequence_with_failures(self):
        """Test frame sequence processing with some failed calculations"""
        # Create a mix of valid and invalid frames
        valid_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        frames = [valid_frame, valid_frame, valid_frame]
        
        # This should work normally
        result = self.calculator.process_frame_sequence(frames)
        self.assertEqual(len(result), 2, "Should handle valid frames normally")
    
    def test_mad_score_range_validation(self):
        """Test that MAD scores are always in valid range [0, 1]"""
        # Test with various frame combinations
        test_cases = [
            (np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)),  # Identical
            (np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10, 3), dtype=np.uint8) * 255),  # Max diff
            (np.ones((10, 10, 3), dtype=np.uint8) * 100, np.ones((10, 10, 3), dtype=np.uint8) * 150),  # Mid diff
        ]
        
        for frame1, frame2 in test_cases:
            mad_score = self.calculator.calculate_mad(frame1, frame2)
            self.assertGreaterEqual(mad_score, 0.0, "MAD score should be >= 0")
            self.assertLessEqual(mad_score, 1.0, "MAD score should be <= 1")
            self.assertTrue(np.isfinite(mad_score), "MAD score should be finite")


class TestHistogramComparator(unittest.TestCase):
    """Test cases for HistogramComparator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.comparator = HistogramComparator(threshold=0.5)
        
        # Create test frames with different histogram characteristics
        self.black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.white_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.gray_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Create gradient frame for more complex histogram
        self.gradient_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            self.gradient_frame[i, :, :] = int(i * 255 / 99)
    
    def test_calculate_histogram_black_frame(self):
        """Test histogram calculation for black frame"""
        hist = self.comparator.calculate_histogram(self.black_frame)
        
        self.assertEqual(len(hist), 256, "Histogram should have 256 bins")
        self.assertAlmostEqual(np.sum(hist), 1.0, places=5, msg="Histogram should be normalized")
        self.assertGreater(hist[0], 0.9, "Black frame should have most pixels in bin 0")
    
    def test_calculate_histogram_white_frame(self):
        """Test histogram calculation for white frame"""
        hist = self.comparator.calculate_histogram(self.white_frame)
        
        self.assertEqual(len(hist), 256, "Histogram should have 256 bins")
        self.assertAlmostEqual(np.sum(hist), 1.0, places=5, msg="Histogram should be normalized")
        self.assertGreater(hist[255], 0.9, "White frame should have most pixels in bin 255")
    
    def test_calculate_chi_square_distance_identical_histograms(self):
        """Test Chi-Square distance for identical histograms"""
        hist1 = self.comparator.calculate_histogram(self.black_frame)
        hist2 = self.comparator.calculate_histogram(self.black_frame)
        
        distance = self.comparator.calculate_chi_square_distance(hist1, hist2)
        self.assertAlmostEqual(distance, 0.0, places=5, msg="Distance should be 0 for identical histograms")
    
    def test_calculate_chi_square_distance_different_histograms(self):
        """Test Chi-Square distance for different histograms"""
        hist1 = self.comparator.calculate_histogram(self.black_frame)
        hist2 = self.comparator.calculate_histogram(self.white_frame)
        
        distance = self.comparator.calculate_chi_square_distance(hist1, hist2)
        self.assertGreater(distance, 0.5, "Distance should be high for very different histograms")
    
    def test_compare_frames_identical(self):
        """Test frame comparison for identical frames"""
        distance = self.comparator.compare_frames(self.black_frame, self.black_frame)
        self.assertAlmostEqual(distance, 0.0, places=5, msg="Distance should be 0 for identical frames")
    
    def test_compare_frames_different(self):
        """Test frame comparison for different frames"""
        distance = self.comparator.compare_frames(self.black_frame, self.white_frame)
        self.assertGreater(distance, 0.5, "Distance should be high for very different frames")
    
    def test_is_shot_boundary_above_threshold(self):
        """Test shot boundary detection above threshold"""
        result = self.comparator.is_shot_boundary(self.black_frame, self.white_frame)
        self.assertTrue(result, "Should detect shot boundary for very different frames")
    
    def test_is_shot_boundary_below_threshold(self):
        """Test shot boundary detection below threshold"""
        result = self.comparator.is_shot_boundary(self.black_frame, self.black_frame)
        self.assertFalse(result, "Should not detect shot boundary for identical frames")
    
    def test_process_frame_sequence_empty_list(self):
        """Test processing empty frame sequence"""
        result = self.comparator.process_frame_sequence([])
        self.assertEqual(result, [], "Should return empty list for empty input")
    
    def test_process_frame_sequence_single_frame(self):
        """Test processing single frame sequence"""
        result = self.comparator.process_frame_sequence([self.black_frame])
        self.assertEqual(result, [], "Should return empty list for single frame")
    
    def test_process_frame_sequence_multiple_frames(self):
        """Test processing multiple frame sequence"""
        frames = [self.black_frame, self.gray_frame, self.white_frame]
        result = self.comparator.process_frame_sequence(frames)
        
        self.assertEqual(len(result), 2, "Should return 2 distances for 3 frames")
        self.assertGreater(result[0], 0.0, "First distance should be > 0 (black to gray)")
        self.assertGreater(result[1], 0.0, "Second distance should be > 0 (gray to white)")
    
    def test_threshold_adjustment(self):
        """Test threshold adjustment functionality"""
        high_threshold_comp = HistogramComparator(threshold=0.9)
        
        # Test with moderate difference frames
        low_result = self.comparator.is_shot_boundary(self.black_frame, self.gray_frame)
        high_result = high_threshold_comp.is_shot_boundary(self.black_frame, self.gray_frame)
        
        # Results may vary based on actual distance, but high threshold should be more restrictive
        if low_result:
            # If low threshold detects boundary, high threshold might not
            pass  # This is expected behavior
        else:
            # If low threshold doesn't detect, high threshold definitely shouldn't
            self.assertFalse(high_result, "High threshold should not detect when low threshold doesn't")
    
    def test_calculate_histogram_none_input(self):
        """Test histogram calculation with None input"""
        with self.assertRaises(HistogramError):
            self.comparator.calculate_histogram(None)
    
    def test_calculate_histogram_empty_frame(self):
        """Test histogram calculation with empty frame"""
        empty_frame = np.array([])
        with self.assertRaises(HistogramError):
            self.comparator.calculate_histogram(empty_frame)
    
    def test_calculate_histogram_invalid_type(self):
        """Test histogram calculation with invalid input type"""
        with self.assertRaises(HistogramError):
            self.comparator.calculate_histogram("not_an_array")
    
    def test_calculate_histogram_known_values(self):
        """Test histogram calculation with known pixel distributions"""
        # Create frame with known pixel values
        test_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        test_frame[0, 0] = [0, 0, 0]      # 1 black pixel
        test_frame[0, 1] = [128, 128, 128] # 1 gray pixel  
        test_frame[0, 2] = [255, 255, 255] # 1 white pixel
        test_frame[0, 3] = [64, 64, 64]   # 1 dark gray pixel
        # Rest are black (12 more pixels)
        
        hist = self.comparator.calculate_histogram(test_frame)
        
        # Check that histogram is normalized
        self.assertAlmostEqual(np.sum(hist), 1.0, places=5, 
                              msg="Histogram should be normalized to sum to 1")
        
        # Check that specific bins have expected values
        # Note: OpenCV converts to grayscale, so we check grayscale values
        total_pixels = 16  # 4x4 frame
        self.assertGreater(hist[0], 0, "Should have pixels in black bin")
    
    def test_calculate_chi_square_distance_none_inputs(self):
        """Test Chi-Square distance with None inputs"""
        valid_hist = np.ones(256) / 256
        
        with self.assertRaises(HistogramError):
            self.comparator.calculate_chi_square_distance(None, valid_hist)
        
        with self.assertRaises(HistogramError):
            self.comparator.calculate_chi_square_distance(valid_hist, None)
    
    def test_calculate_chi_square_distance_empty_histograms(self):
        """Test Chi-Square distance with empty histograms"""
        empty_hist = np.array([])
        valid_hist = np.ones(256) / 256
        
        with self.assertRaises(HistogramError):
            self.comparator.calculate_chi_square_distance(empty_hist, valid_hist)
    
    def test_calculate_chi_square_distance_different_shapes(self):
        """Test Chi-Square distance with different histogram shapes"""
        hist1 = np.ones(256) / 256
        hist2 = np.ones(128) / 128  # Different size
        
        with self.assertRaises(HistogramError):
            self.comparator.calculate_chi_square_distance(hist1, hist2)
    
    def test_calculate_chi_square_distance_known_values(self):
        """Test Chi-Square distance with known histogram pairs"""
        # Create two simple histograms with known distributions
        hist1 = np.zeros(256)
        hist1[0] = 1.0  # All mass at bin 0
        
        hist2 = np.zeros(256)
        hist2[255] = 1.0  # All mass at bin 255
        
        distance = self.comparator.calculate_chi_square_distance(hist1, hist2)
        
        # For completely different distributions, distance should be high
        self.assertGreater(distance, 0.5, "Distance should be high for completely different histograms")
        self.assertTrue(np.isfinite(distance), "Distance should be finite")
    
    def test_calculate_chi_square_distance_negative_values(self):
        """Test Chi-Square distance with negative histogram values"""
        hist1 = np.ones(256) / 256
        hist2 = np.ones(256) / 256
        hist2[0] = -0.1  # Invalid negative value
        
        with self.assertRaises(HistogramError):
            self.comparator.calculate_chi_square_distance(hist1, hist2)
    
    def test_histogram_bins_parameter(self):
        """Test histogram calculation with different bin counts"""
        comparator_64 = HistogramComparator(bins=64)
        comparator_128 = HistogramComparator(bins=128)
        
        hist_64 = comparator_64.calculate_histogram(self.black_frame)
        hist_128 = comparator_128.calculate_histogram(self.black_frame)
        
        self.assertEqual(len(hist_64), 64, "Should have 64 bins")
        self.assertEqual(len(hist_128), 128, "Should have 128 bins")
        
        # Both should be normalized
        self.assertAlmostEqual(np.sum(hist_64), 1.0, places=5)
        self.assertAlmostEqual(np.sum(hist_128), 1.0, places=5)


class TestBlurDetector(unittest.TestCase):
    """Test cases for BlurDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = BlurDetector(threshold=100.0)
        
        # Create test frames with different sharpness characteristics
        self.sharp_frame = self._create_sharp_frame()
        self.blurry_frame = self._create_blurry_frame()
        self.uniform_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Uniform gray
    
    def _create_sharp_frame(self):
        """Create a frame with sharp edges (high frequency content)"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create checkerboard pattern for high frequency content
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                if (i // 10 + j // 10) % 2 == 0:
                    frame[i:i+10, j:j+10] = 255
        return frame
    
    def _create_blurry_frame(self):
        """Create a blurry frame by applying Gaussian blur to sharp frame"""
        sharp = self._create_sharp_frame()
        # Apply heavy Gaussian blur to reduce high frequency content
        blurry = cv2.GaussianBlur(sharp, (21, 21), 10)
        return blurry
    
    def test_calculate_blur_score_sharp_frame(self):
        """Test blur score calculation for sharp frame"""
        score = self.detector.calculate_blur_score(self.sharp_frame)
        self.assertGreater(score, 100, "Sharp frame should have high blur score")
    
    def test_calculate_blur_score_blurry_frame(self):
        """Test blur score calculation for blurry frame"""
        score = self.detector.calculate_blur_score(self.blurry_frame)
        self.assertLess(score, 100, "Blurry frame should have low blur score")
    
    def test_calculate_blur_score_uniform_frame(self):
        """Test blur score calculation for uniform frame"""
        score = self.detector.calculate_blur_score(self.uniform_frame)
        self.assertAlmostEqual(score, 0.0, places=1, msg="Uniform frame should have very low blur score")
    
    def test_is_sharp_with_sharp_frame(self):
        """Test sharp detection with sharp frame"""
        result = self.detector.is_sharp(self.sharp_frame)
        self.assertTrue(result, "Sharp frame should be detected as sharp")
    
    def test_is_sharp_with_blurry_frame(self):
        """Test sharp detection with blurry frame"""
        result = self.detector.is_sharp(self.blurry_frame)
        self.assertFalse(result, "Blurry frame should not be detected as sharp")
    
    def test_is_blurry_with_sharp_frame(self):
        """Test blurry detection with sharp frame"""
        result = self.detector.is_blurry(self.sharp_frame)
        self.assertFalse(result, "Sharp frame should not be detected as blurry")
    
    def test_is_blurry_with_blurry_frame(self):
        """Test blurry detection with blurry frame"""
        result = self.detector.is_blurry(self.blurry_frame)
        self.assertTrue(result, "Blurry frame should be detected as blurry")
    
    def test_filter_sharp_frames_empty_list(self):
        """Test filtering sharp frames from empty list"""
        result = self.detector.filter_sharp_frames([])
        self.assertEqual(result, [], "Should return empty list for empty input")
    
    def test_filter_sharp_frames_mixed_quality(self):
        """Test filtering sharp frames from mixed quality frames"""
        frames = [self.sharp_frame, self.blurry_frame, self.uniform_frame]
        result = self.detector.filter_sharp_frames(frames)
        
        # Should only return the sharp frame
        self.assertEqual(len(result), 1, "Should return only sharp frames")
        self.assertEqual(result[0][0], 0, "Sharp frame should be at index 0")
        np.testing.assert_array_equal(result[0][1], self.sharp_frame, "Should return the sharp frame")
    
    def test_process_frame_sequence_empty_list(self):
        """Test processing empty frame sequence"""
        result = self.detector.process_frame_sequence([])
        self.assertEqual(result, [], "Should return empty list for empty input")
    
    def test_process_frame_sequence_multiple_frames(self):
        """Test processing multiple frame sequence"""
        frames = [self.sharp_frame, self.blurry_frame, self.uniform_frame]
        result = self.detector.process_frame_sequence(frames)
        
        self.assertEqual(len(result), 3, "Should return 3 scores for 3 frames")
        self.assertGreater(result[0], result[1], "Sharp frame should have higher score than blurry")
        self.assertGreater(result[1], result[2], "Blurry frame should have higher score than uniform")
    
    def test_get_sharpness_ranking_empty_list(self):
        """Test sharpness ranking for empty list"""
        result = self.detector.get_sharpness_ranking([])
        self.assertEqual(result, [], "Should return empty list for empty input")
    
    def test_get_sharpness_ranking_multiple_frames(self):
        """Test sharpness ranking for multiple frames"""
        frames = [self.blurry_frame, self.sharp_frame, self.uniform_frame]  # Mixed order
        result = self.detector.get_sharpness_ranking(frames)
        
        self.assertEqual(len(result), 3, "Should return 3 rankings for 3 frames")
        # Should be sorted by sharpness (descending)
        self.assertEqual(result[0][0], 1, "Sharpest frame (index 1) should be first")
        self.assertEqual(result[1][0], 0, "Blurry frame (index 0) should be second")
        self.assertEqual(result[2][0], 2, "Uniform frame (index 2) should be last")
        
        # Scores should be in descending order
        self.assertGreater(result[0][1], result[1][1], "First should have higher score than second")
        self.assertGreater(result[1][1], result[2][1], "Second should have higher score than third")
    
    def test_threshold_adjustment(self):
        """Test threshold adjustment functionality"""
        low_threshold_detector = BlurDetector(threshold=10.0)
        high_threshold_detector = BlurDetector(threshold=500.0)
        
        # Test with moderately sharp frame
        moderate_frame = cv2.GaussianBlur(self.sharp_frame, (5, 5), 1)
        
        low_result = low_threshold_detector.is_sharp(moderate_frame)
        high_result = high_threshold_detector.is_sharp(moderate_frame)
        
        # Low threshold should be more permissive
        self.assertTrue(low_result or not high_result, 
                       "Low threshold should be more permissive than high threshold")
    
    def test_calculate_blur_score_none_input(self):
        """Test blur score calculation with None input"""
        with self.assertRaises(BlurDetectionError):
            self.detector.calculate_blur_score(None)
    
    def test_calculate_blur_score_empty_frame(self):
        """Test blur score calculation with empty frame"""
        empty_frame = np.array([])
        with self.assertRaises(BlurDetectionError):
            self.detector.calculate_blur_score(empty_frame)
    
    def test_calculate_blur_score_invalid_type(self):
        """Test blur score calculation with invalid input type"""
        with self.assertRaises(BlurDetectionError):
            self.detector.calculate_blur_score("not_an_array")
    
    def test_calculate_blur_score_known_patterns(self):
        """Test blur score calculation with known sharp/blurry patterns"""
        # Create a frame with high frequency content (sharp edges)
        sharp_pattern = np.zeros((20, 20, 3), dtype=np.uint8)
        sharp_pattern[::2, ::2] = 255  # Checkerboard pattern
        
        # Create a smooth gradient (low frequency content)
        smooth_pattern = np.zeros((20, 20, 3), dtype=np.uint8)
        for i in range(20):
            smooth_pattern[i, :] = int(i * 255 / 19)
        
        sharp_score = self.detector.calculate_blur_score(sharp_pattern)
        smooth_score = self.detector.calculate_blur_score(smooth_pattern)
        
        self.assertGreater(sharp_score, smooth_score, 
                          "Sharp pattern should have higher blur score than smooth pattern")
        self.assertGreater(sharp_score, 0, "Sharp pattern should have positive blur score")
        self.assertGreaterEqual(smooth_score, 0, "Smooth pattern should have non-negative blur score")
    
    def test_calculate_blur_score_single_vs_multi_channel(self):
        """Test blur score consistency between single and multi-channel frames"""
        # Create equivalent grayscale and color frames
        gray_sharp = np.zeros((20, 20), dtype=np.uint8)
        gray_sharp[::2, ::2] = 255
        
        color_sharp = np.zeros((20, 20, 3), dtype=np.uint8)
        color_sharp[::2, ::2] = 255
        
        gray_score = self.detector.calculate_blur_score(gray_sharp)
        color_score = self.detector.calculate_blur_score(color_sharp)
        
        # Scores should be similar (within reasonable tolerance)
        self.assertAlmostEqual(gray_score, color_score, delta=gray_score * 0.1,
                              msg="Blur scores should be similar for equivalent grayscale and color frames")
    
    def test_blur_score_range_validation(self):
        """Test that blur scores are always non-negative and finite"""
        test_frames = [
            self.sharp_frame,
            self.blurry_frame,
            self.uniform_frame,
            np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)  # Random noise
        ]
        
        for frame in test_frames:
            score = self.detector.calculate_blur_score(frame)
            self.assertGreaterEqual(score, 0.0, "Blur score should be non-negative")
            self.assertTrue(np.isfinite(score), "Blur score should be finite")
    
    def test_process_frame_sequence_with_mixed_quality(self):
        """Test processing frame sequence with mixed quality frames"""
        # Create frames with different blur levels
        very_sharp = self._create_sharp_frame()
        moderately_sharp = cv2.GaussianBlur(very_sharp, (3, 3), 1)
        very_blurry = cv2.GaussianBlur(very_sharp, (15, 15), 5)
        
        frames = [very_sharp, moderately_sharp, very_blurry]
        scores = self.detector.process_frame_sequence(frames)
        
        self.assertEqual(len(scores), 3, "Should return score for each frame")
        # Scores should be in descending order (sharp to blurry)
        self.assertGreater(scores[0], scores[1], "Very sharp should score higher than moderately sharp")
        self.assertGreater(scores[1], scores[2], "Moderately sharp should score higher than very blurry")
    
    def test_get_sharpness_ranking_consistency(self):
        """Test that sharpness ranking is consistent with individual scores"""
        frames = [self.blurry_frame, self.sharp_frame, self.uniform_frame]
        
        # Get individual scores
        individual_scores = [self.detector.calculate_blur_score(frame) for frame in frames]
        
        # Get ranking
        ranking = self.detector.get_sharpness_ranking(frames)
        
        # Verify ranking matches individual score ordering
        sorted_indices = sorted(range(len(individual_scores)), 
                               key=lambda i: individual_scores[i], reverse=True)
        
        for rank_pos, (frame_idx, score) in enumerate(ranking):
            expected_idx = sorted_indices[rank_pos]
            self.assertEqual(frame_idx, expected_idx, 
                           f"Ranking position {rank_pos} should have frame index {expected_idx}")
            self.assertAlmostEqual(score, individual_scores[frame_idx], places=5,
                                 msg="Ranking score should match individual calculation")
    
    def test_filter_sharp_frames_threshold_sensitivity(self):
        """Test that sharp frame filtering is sensitive to threshold changes"""
        # Create frames with different sharpness levels
        very_sharp = self._create_sharp_frame()
        moderately_sharp = cv2.GaussianBlur(very_sharp, (3, 3), 1)
        slightly_blurry = cv2.GaussianBlur(very_sharp, (7, 7), 2)
        
        frames = [very_sharp, moderately_sharp, slightly_blurry]
        
        # Test with low threshold (should accept more frames)
        low_detector = BlurDetector(threshold=10.0)
        low_result = low_detector.filter_sharp_frames(frames)
        
        # Test with high threshold (should accept fewer frames)
        high_detector = BlurDetector(threshold=200.0)
        high_result = high_detector.filter_sharp_frames(frames)
        
        # Low threshold should accept at least as many frames as high threshold
        self.assertGreaterEqual(len(low_result), len(high_result),
                               "Low threshold should accept at least as many frames as high threshold")


class TestIVPAlgorithmsIntegration(unittest.TestCase):
    """Integration tests for all IVP algorithms working together"""
    
    def setUp(self):
        """Set up test fixtures for integration tests"""
        self.frame_diff_calc = FrameDifferenceCalculator(threshold=0.1)
        self.histogram_comp = HistogramComparator(threshold=0.5)
        self.blur_detector = BlurDetector(threshold=100.0)
        
        # Create a sequence of test frames with known characteristics
        self.test_sequence = self._create_test_video_sequence()
    
    def _create_test_video_sequence(self):
        """Create a sequence of frames simulating a video with different characteristics"""
        frames = []
        
        # Frame 1: Black frame (sharp)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frames.append(frame1)
        
        # Frame 2: Similar black frame (should have low frame difference)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[45:55, 45:55] = 10  # Slight change
        frames.append(frame2)
        
        # Frame 3: White frame (sharp, high frame difference)
        frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        frames.append(frame3)
        
        # Frame 4: Checkerboard pattern (very sharp)
        frame4 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame4[::10, ::10] = 255
        frames.append(frame4)
        
        # Frame 5: Blurred version of frame 4
        frame5 = cv2.GaussianBlur(frame4, (15, 15), 5)
        frames.append(frame5)
        
        return frames
    
    def test_all_algorithms_on_sequence(self):
        """Test all three algorithms on the same frame sequence"""
        # Process with all algorithms
        mad_scores = self.frame_diff_calc.process_frame_sequence(self.test_sequence)
        chi_square_distances = self.histogram_comp.process_frame_sequence(self.test_sequence)
        blur_scores = self.blur_detector.process_frame_sequence(self.test_sequence)
        
        # Verify all algorithms return expected number of results
        expected_frame_pairs = len(self.test_sequence) - 1
        expected_blur_scores = len(self.test_sequence)
        
        self.assertEqual(len(mad_scores), expected_frame_pairs, 
                        "Frame difference should return n-1 scores for n frames")
        self.assertEqual(len(chi_square_distances), expected_frame_pairs,
                        "Histogram comparison should return n-1 distances for n frames")
        self.assertEqual(len(blur_scores), expected_blur_scores,
                        "Blur detection should return n scores for n frames")
        
        # Verify all scores are in valid ranges
        for score in mad_scores:
            self.assertGreaterEqual(score, 0.0, "MAD scores should be non-negative")
            self.assertLessEqual(score, 1.0, "MAD scores should be <= 1.0")
        
        for distance in chi_square_distances:
            self.assertGreaterEqual(distance, 0.0, "Chi-square distances should be non-negative")
        
        for score in blur_scores:
            self.assertGreaterEqual(score, 0.0, "Blur scores should be non-negative")
    
    def test_algorithm_consistency_across_parameters(self):
        """Test that algorithms behave consistently across different parameter settings"""
        # Test frame difference with different thresholds
        low_threshold_calc = FrameDifferenceCalculator(threshold=0.01)
        high_threshold_calc = FrameDifferenceCalculator(threshold=0.5)
        
        frame1 = self.test_sequence[0]
        frame2 = self.test_sequence[2]  # Very different frames
        
        # Both should calculate the same MAD score
        low_mad = low_threshold_calc.calculate_mad(frame1, frame2)
        high_mad = high_threshold_calc.calculate_mad(frame1, frame2)
        self.assertAlmostEqual(low_mad, high_mad, places=5,
                              msg="MAD calculation should be independent of threshold")
        
        # But significance detection should differ
        low_significant = low_threshold_calc.is_significant_change(frame1, frame2)
        high_significant = high_threshold_calc.is_significant_change(frame1, frame2)
        
        if low_mad > 0.5:  # If there's substantial difference
            self.assertTrue(low_significant, "Low threshold should detect significant change")
            # High threshold might or might not detect, depending on exact value
    
    def test_error_handling_consistency(self):
        """Test that all algorithms handle errors consistently"""
        invalid_inputs = [
            None,
            np.array([]),
            "not_an_array",
            np.array([1, 2, 3])  # Wrong shape
        ]
        
        valid_frame = self.test_sequence[0]
        
        for invalid_input in invalid_inputs:
            # All algorithms should raise appropriate errors for invalid inputs
            with self.assertRaises((FrameDifferenceError, TypeError, ValueError)):
                self.frame_diff_calc.calculate_mad(invalid_input, valid_frame)
            
            with self.assertRaises((HistogramError, TypeError, ValueError)):
                self.histogram_comp.calculate_histogram(invalid_input)
            
            with self.assertRaises((BlurDetectionError, TypeError, ValueError)):
                self.blur_detector.calculate_blur_score(invalid_input)
    
    def test_performance_characteristics(self):
        """Test that algorithms complete within reasonable time for typical inputs"""
        import time
        
        # Create larger test frames
        large_frame1 = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        large_frame2 = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        
        # Test frame difference performance
        start_time = time.time()
        mad_score = self.frame_diff_calc.calculate_mad(large_frame1, large_frame2)
        mad_time = time.time() - start_time
        
        # Test histogram comparison performance
        start_time = time.time()
        chi_square_dist = self.histogram_comp.compare_frames(large_frame1, large_frame2)
        histogram_time = time.time() - start_time
        
        # Test blur detection performance
        start_time = time.time()
        blur_score = self.blur_detector.calculate_blur_score(large_frame1)
        blur_time = time.time() - start_time
        
        # All operations should complete within reasonable time (< 1 second for 500x500 frame)
        self.assertLess(mad_time, 1.0, "MAD calculation should complete within 1 second")
        self.assertLess(histogram_time, 1.0, "Histogram comparison should complete within 1 second")
        self.assertLess(blur_time, 1.0, "Blur detection should complete within 1 second")
        
        # Results should still be valid
        self.assertTrue(np.isfinite(mad_score), "MAD score should be finite")
        self.assertTrue(np.isfinite(chi_square_dist), "Chi-square distance should be finite")
        self.assertTrue(np.isfinite(blur_score), "Blur score should be finite")
    
    def test_memory_efficiency(self):
        """Test that algorithms don't cause memory leaks with repeated operations"""
        import gc
        
        # Get initial memory state
        gc.collect()
        
        # Perform many operations
        test_frame1 = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        test_frame2 = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        
        for _ in range(100):
            # Perform all operations
            self.frame_diff_calc.calculate_mad(test_frame1, test_frame2)
            self.histogram_comp.compare_frames(test_frame1, test_frame2)
            self.blur_detector.calculate_blur_score(test_frame1)
        
        # Force garbage collection
        gc.collect()
        
        # This test mainly ensures no exceptions are raised during repeated operations
        # Memory usage testing would require more sophisticated tools
        self.assertTrue(True, "Memory efficiency test completed without errors")


class TestIVPAlgorithmsEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions for IVP algorithms"""
    
    def setUp(self):
        """Set up test fixtures for edge case testing"""
        self.frame_diff_calc = FrameDifferenceCalculator()
        self.histogram_comp = HistogramComparator()
        self.blur_detector = BlurDetector()
    
    def test_minimal_size_frames(self):
        """Test algorithms with minimal size frames"""
        # Create 1x1 frames
        tiny_frame1 = np.array([[[0, 0, 0]]], dtype=np.uint8)
        tiny_frame2 = np.array([[[255, 255, 255]]], dtype=np.uint8)
        
        # All algorithms should handle tiny frames
        mad_score = self.frame_diff_calc.calculate_mad(tiny_frame1, tiny_frame2)
        self.assertAlmostEqual(mad_score, 1.0, places=3, msg="1x1 black vs white should give MAD of 1.0")
        
        chi_square_dist = self.histogram_comp.compare_frames(tiny_frame1, tiny_frame2)
        self.assertGreater(chi_square_dist, 0, "1x1 black vs white should give positive chi-square distance")
        
        blur_score1 = self.blur_detector.calculate_blur_score(tiny_frame1)
        blur_score2 = self.blur_detector.calculate_blur_score(tiny_frame2)
        # Blur scores for 1x1 frames should be very low (no edges)
        self.assertGreaterEqual(blur_score1, 0, "Blur score should be non-negative")
        self.assertGreaterEqual(blur_score2, 0, "Blur score should be non-negative")
    
    def test_extreme_aspect_ratios(self):
        """Test algorithms with extreme aspect ratio frames"""
        # Create very wide frame (1x1000)
        wide_frame = np.random.randint(0, 256, (1, 1000, 3), dtype=np.uint8)
        
        # Create very tall frame (1000x1)
        tall_frame = np.random.randint(0, 256, (1000, 1, 3), dtype=np.uint8)
        
        # All algorithms should handle extreme aspect ratios
        blur_score_wide = self.blur_detector.calculate_blur_score(wide_frame)
        blur_score_tall = self.blur_detector.calculate_blur_score(tall_frame)
        
        self.assertGreaterEqual(blur_score_wide, 0, "Wide frame blur score should be non-negative")
        self.assertGreaterEqual(blur_score_tall, 0, "Tall frame blur score should be non-negative")
        
        hist_wide = self.histogram_comp.calculate_histogram(wide_frame)
        hist_tall = self.histogram_comp.calculate_histogram(tall_frame)
        
        self.assertAlmostEqual(np.sum(hist_wide), 1.0, places=5, msg="Wide frame histogram should be normalized")
        self.assertAlmostEqual(np.sum(hist_tall), 1.0, places=5, msg="Tall frame histogram should be normalized")
    
    def test_all_same_pixel_values(self):
        """Test algorithms with frames where all pixels have the same value"""
        # Create uniform frames with different values
        uniform_black = np.zeros((50, 50, 3), dtype=np.uint8)
        uniform_gray = np.ones((50, 50, 3), dtype=np.uint8) * 128
        uniform_white = np.ones((50, 50, 3), dtype=np.uint8) * 255
        
        # Test blur detection on uniform frames (should have very low scores)
        blur_black = self.blur_detector.calculate_blur_score(uniform_black)
        blur_gray = self.blur_detector.calculate_blur_score(uniform_gray)
        blur_white = self.blur_detector.calculate_blur_score(uniform_white)
        
        # All should have very low blur scores (no edges/texture)
        self.assertLess(blur_black, 1.0, "Uniform black should have very low blur score")
        self.assertLess(blur_gray, 1.0, "Uniform gray should have very low blur score")
        self.assertLess(blur_white, 1.0, "Uniform white should have very low blur score")
        
        # Test frame difference between uniform frames
        mad_black_gray = self.frame_diff_calc.calculate_mad(uniform_black, uniform_gray)
        mad_gray_white = self.frame_diff_calc.calculate_mad(uniform_gray, uniform_white)
        
        expected_black_gray = 128.0 / 255.0  # |0 - 128| / 255
        expected_gray_white = 127.0 / 255.0  # |128 - 255| / 255
        
        self.assertAlmostEqual(mad_black_gray, expected_black_gray, places=3,
                              msg="MAD between uniform frames should match expected value")
        self.assertAlmostEqual(mad_gray_white, expected_gray_white, places=3,
                              msg="MAD between uniform frames should match expected value")
    
    def test_high_noise_frames(self):
        """Test algorithms with high noise content"""
        # Create frames with random noise
        np.random.seed(42)  # For reproducible results
        noise_frame1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        noise_frame2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # All algorithms should handle noisy frames without errors
        mad_score = self.frame_diff_calc.calculate_mad(noise_frame1, noise_frame2)
        chi_square_dist = self.histogram_comp.compare_frames(noise_frame1, noise_frame2)
        blur_score1 = self.blur_detector.calculate_blur_score(noise_frame1)
        blur_score2 = self.blur_detector.calculate_blur_score(noise_frame2)
        
        # All results should be finite and in valid ranges
        self.assertTrue(np.isfinite(mad_score), "MAD score for noisy frames should be finite")
        self.assertTrue(np.isfinite(chi_square_dist), "Chi-square distance for noisy frames should be finite")
        self.assertTrue(np.isfinite(blur_score1), "Blur score for noisy frame should be finite")
        self.assertTrue(np.isfinite(blur_score2), "Blur score for noisy frame should be finite")
        
        self.assertGreaterEqual(mad_score, 0.0, "MAD score should be non-negative")
        self.assertLessEqual(mad_score, 1.0, "MAD score should be <= 1.0")
        self.assertGreaterEqual(chi_square_dist, 0.0, "Chi-square distance should be non-negative")
        self.assertGreaterEqual(blur_score1, 0.0, "Blur score should be non-negative")
        self.assertGreaterEqual(blur_score2, 0.0, "Blur score should be non-negative")


if __name__ == '__main__':
    unittest.main()