"""
Tests for VideoProcessor class and video processing pipeline.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock

from modules.video_processor import (
    VideoProcessor, FrameMetrics, VideoProcessingError, 
    FrameExtractionError, AlgorithmProcessingError
)


class TestVideoProcessor:
    """Test cases for VideoProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor(
            frame_diff_threshold=0.1,
            histogram_threshold=0.5,
            blur_threshold=100.0
        )
    
    def test_initialization(self):
        """Test VideoProcessor initialization."""
        assert self.processor.frame_diff_threshold == 0.1
        assert self.processor.histogram_threshold == 0.5
        assert self.processor.blur_threshold == 100.0
        assert self.processor.video_path is None
        assert self.processor.frames == []
        assert self.processor.frame_metrics == []
    
    def test_update_algorithm_parameters(self):
        """Test updating algorithm parameters."""
        self.processor.update_algorithm_parameters(
            frame_diff_threshold=0.2,
            histogram_threshold=0.6,
            blur_threshold=150.0
        )
        
        assert self.processor.frame_diff_threshold == 0.2
        assert self.processor.histogram_threshold == 0.6
        assert self.processor.blur_threshold == 150.0
    
    def test_get_frame_timestamp(self):
        """Test frame timestamp calculation."""
        self.processor.fps = 30.0
        
        timestamp = self.processor.get_frame_timestamp(0, sample_rate=1)
        assert timestamp == 0.0
        
        timestamp = self.processor.get_frame_timestamp(30, sample_rate=1)
        assert timestamp == 1.0
        
        timestamp = self.processor.get_frame_timestamp(15, sample_rate=2)
        assert timestamp == 1.0
    
    @patch('cv2.VideoCapture')
    def test_load_video_success(self, mock_video_capture):
        """Test successful video loading."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # Create temporary video file with some content
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b'fake video content')  # Add some content to avoid empty file error
            video_path = tmp_file.name
        
        try:
            result = self.processor.load_video(video_path)
            assert result is True
            assert self.processor.video_path == video_path
            assert self.processor.fps == 30.0
            assert self.processor.total_frames == 100
        finally:
            os.unlink(video_path)
    
    def test_load_video_file_not_found(self):
        """Test loading non-existent video file."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_video("nonexistent_video.mp4")
    
    @patch('cv2.VideoCapture')
    def test_load_video_cannot_open(self, mock_video_capture):
        """Test loading video that cannot be opened."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b'fake video content')  # Add content but mock will still fail to open
            video_path = tmp_file.name
        
        try:
            with pytest.raises(VideoProcessingError, match="Cannot open video file"):
                self.processor.load_video(video_path)
        finally:
            os.unlink(video_path)
    
    def test_extract_frames_no_video_loaded(self):
        """Test frame extraction without loading video first."""
        with pytest.raises(FrameExtractionError, match="No video loaded"):
            self.processor.extract_frames()
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_success(self, mock_video_capture):
        """Test successful frame extraction."""
        # Setup mock video capture for loading
        mock_cap_load = Mock()
        mock_cap_load.isOpened.return_value = True
        mock_cap_load.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 5,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }.get(prop, 0)
        mock_cap_load.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Setup mock video capture for extraction
        mock_cap_extract = Mock()
        mock_cap_extract.isOpened.return_value = True
        
        # Mock frame reading sequence
        test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]
        read_results = [(True, frame) for frame in test_frames] + [(False, None)]
        mock_cap_extract.read.side_effect = read_results
        
        mock_video_capture.side_effect = [mock_cap_load, mock_cap_extract]
        
        # Create temporary video file and load it
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b'fake video content')  # Add some content
            video_path = tmp_file.name
        
        try:
            self.processor.load_video(video_path)
            frames = self.processor.extract_frames()
            
            assert len(frames) == 5
            assert len(self.processor.frames) == 5
            assert all(isinstance(frame, np.ndarray) for frame in frames)
        finally:
            os.unlink(video_path)
    
    def test_process_with_ivp_algorithms_no_frames(self):
        """Test IVP processing without frames."""
        with pytest.raises(AlgorithmProcessingError, match="No frames available"):
            self.processor.process_with_ivp_algorithms()
    
    def test_process_with_ivp_algorithms_success(self):
        """Test successful IVP algorithm processing."""
        # Create test frames
        test_frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]
        
        self.processor.fps = 30.0
        
        metrics = self.processor.process_with_ivp_algorithms(test_frames)
        
        assert len(metrics) == 3
        assert all(isinstance(m, FrameMetrics) for m in metrics)
        assert all(m.frame_number == i for i, m in enumerate(metrics))
        assert all(hasattr(m, 'difference_score') for m in metrics)
        assert all(hasattr(m, 'histogram_score') for m in metrics)
        assert all(hasattr(m, 'blur_score') for m in metrics)
    
    def test_select_keyframes_no_metrics(self):
        """Test keyframe selection without frame metrics."""
        with pytest.raises(RuntimeError, match="No frame metrics available"):
            self.processor.select_keyframes()
    
    def test_select_keyframes_success(self):
        """Test successful keyframe selection."""
        # Create mock frame metrics
        self.processor.frame_metrics = [
            FrameMetrics(
                frame_number=i,
                timestamp=i/30.0,
                difference_score=0.2 if i % 2 == 0 else 0.05,
                histogram_score=0.6 if i % 3 == 0 else 0.3,
                blur_score=150.0 if i % 2 == 0 else 50.0,
                is_keyframe=False
            )
            for i in range(10)
        ]
        
        keyframes = self.processor.select_keyframes(target_reduction=0.7)
        
        assert len(keyframes) <= 10
        assert len(keyframes) >= 1
        assert all(isinstance(kf, FrameMetrics) for kf in keyframes)
        assert all(kf.is_keyframe for kf in keyframes)
        
        # Check that keyframes are marked in frame_metrics
        keyframe_numbers = {kf.frame_number for kf in keyframes}
        for metrics in self.processor.frame_metrics:
            if metrics.frame_number in keyframe_numbers:
                assert metrics.is_keyframe
            else:
                assert not metrics.is_keyframe
    
    def test_get_keyframe_summary_no_metrics(self):
        """Test keyframe summary with no metrics."""
        summary = self.processor.get_keyframe_summary()
        assert summary == {}
    
    def test_get_keyframe_summary_with_keyframes(self):
        """Test keyframe summary with selected keyframes."""
        # Create mock frame metrics with some keyframes
        self.processor.frame_metrics = [
            FrameMetrics(
                frame_number=i,
                timestamp=i/30.0,
                difference_score=0.2,
                histogram_score=0.5,
                blur_score=120.0,
                is_keyframe=(i % 3 == 0)  # Every 3rd frame is keyframe
            )
            for i in range(9)
        ]
        
        summary = self.processor.get_keyframe_summary()
        
        assert summary['total_frames'] == 9
        assert summary['keyframes_selected'] == 3
        assert summary['reduction_percentage'] > 0
        assert len(summary['keyframe_indices']) == 3
        assert 'temporal_distribution' in summary
        assert 'algorithm_stats' in summary
    
    def test_get_algorithm_scores_summary_no_metrics(self):
        """Test algorithm scores summary with no metrics."""
        summary = self.processor.get_algorithm_scores_summary()
        assert summary == {}
    
    def test_get_algorithm_scores_summary_with_metrics(self):
        """Test algorithm scores summary with metrics."""
        self.processor.frame_metrics = [
            FrameMetrics(
                frame_number=i,
                timestamp=i/30.0,
                difference_score=0.1 + i * 0.1,
                histogram_score=0.2 + i * 0.1,
                blur_score=100.0 + i * 10,
                is_keyframe=False
            )
            for i in range(3)
        ]
        
        summary = self.processor.get_algorithm_scores_summary()
        
        assert 'frame_difference' in summary
        assert 'histogram_comparison' in summary
        assert 'blur_detection' in summary
        
        for alg_summary in summary.values():
            assert 'min' in alg_summary
            assert 'max' in alg_summary
            assert 'mean' in alg_summary
            assert 'std' in alg_summary
            assert 'threshold' in alg_summary
    
    def test_filter_frames_by_algorithm(self):
        """Test filtering frames by algorithm results."""
        self.processor.frame_metrics = [
            FrameMetrics(
                frame_number=i,
                timestamp=i/30.0,
                difference_score=0.2 if i % 2 == 0 else 0.05,
                histogram_score=0.6 if i % 3 == 0 else 0.3,
                blur_score=150.0 if i % 2 == 0 else 50.0,
                is_keyframe=False
            )
            for i in range(6)
        ]
        
        # Test frame difference filtering
        above_threshold = self.processor.filter_frames_by_algorithm('frame_diff', above_threshold=True)
        assert len(above_threshold) == 3  # Frames 0, 2, 4
        
        # Test blur filtering
        sharp_frames = self.processor.filter_frames_by_algorithm('blur', above_threshold=True)
        assert len(sharp_frames) == 3  # Frames 0, 2, 4
        
        # Test histogram filtering
        hist_frames = self.processor.filter_frames_by_algorithm('histogram', above_threshold=True)
        assert len(hist_frames) == 2  # Frames 0, 3


class TestFrameMetrics:
    """Test cases for FrameMetrics dataclass."""
    
    def test_frame_metrics_creation(self):
        """Test FrameMetrics creation and attributes."""
        metrics = FrameMetrics(
            frame_number=5,
            timestamp=0.167,
            difference_score=0.25,
            histogram_score=0.45,
            blur_score=125.0,
            is_keyframe=True
        )
        
        assert metrics.frame_number == 5
        assert metrics.timestamp == 0.167
        assert metrics.difference_score == 0.25
        assert metrics.histogram_score == 0.45
        assert metrics.blur_score == 125.0
        assert metrics.is_keyframe is True
    
    def test_frame_metrics_default_keyframe(self):
        """Test FrameMetrics default is_keyframe value."""
        metrics = FrameMetrics(
            frame_number=0,
            timestamp=0.0,
            difference_score=0.0,
            histogram_score=0.0,
            blur_score=0.0
        )
        
        assert metrics.is_keyframe is False


class TestVideoProcessingIntegration:
    """Integration tests for the complete video processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.processor = VideoProcessor(
            frame_diff_threshold=0.1,
            histogram_threshold=0.5,
            blur_threshold=100.0
        )
    
    def _create_test_video_file(self, num_frames=10, fps=30, width=100, height=100):
        """Create a temporary test video file with known characteristics."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
        
        try:
            for i in range(num_frames):
                # Create frames with different characteristics
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                if i % 4 == 0:  # Every 4th frame: black
                    pass  # Already black
                elif i % 4 == 1:  # White frame
                    frame[:] = 255
                elif i % 4 == 2:  # Gray frame
                    frame[:] = 128
                else:  # Checkerboard pattern (sharp)
                    frame[::10, ::10] = 255
                
                writer.write(frame)
            
            writer.release()
            return temp_file.name
            
        except Exception as e:
            writer.release()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e
    
    def test_end_to_end_video_processing_small_video(self):
        """Test complete end-to-end processing with a small test video."""
        video_path = None
        try:
            # Create test video
            video_path = self._create_test_video_file(num_frames=8, fps=30)
            
            # Process video completely
            results = self.processor.process_video_complete(
                video_path=video_path,
                target_reduction=0.5,  # 50% reduction
                max_frames=None,
                sample_rate=1
            )
            
            # Verify results structure
            assert results['success'] is True
            assert 'video_properties' in results
            assert 'keyframes' in results
            assert 'frame_metrics' in results
            assert 'summary' in results
            assert 'algorithm_summary' in results
            
            # Verify video properties
            props = results['video_properties']
            assert props['fps'] == 30.0
            assert props['processed_frames'] == 8
            
            # Verify frame metrics
            metrics = results['frame_metrics']
            assert len(metrics) == 8
            assert all(isinstance(m, FrameMetrics) for m in metrics)
            
            # Verify keyframes were selected
            keyframes = results['keyframes']
            assert len(keyframes) > 0
            assert len(keyframes) < 8  # Should be reduced
            assert all(kf.is_keyframe for kf in keyframes)
            
            # Verify summary
            summary = results['summary']
            assert summary['total_frames'] == 8
            assert summary['keyframes_selected'] == len(keyframes)
            assert 0 < summary['reduction_percentage'] < 100
            
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_end_to_end_processing_with_parameter_adjustment(self):
        """Test end-to-end processing with different parameter settings."""
        video_path = None
        try:
            # Create test video with varied content
            video_path = self._create_test_video_file(num_frames=12, fps=30)
            
            # Test with strict parameters (high thresholds)
            strict_processor = VideoProcessor(
                frame_diff_threshold=0.8,
                histogram_threshold=0.8,
                blur_threshold=200.0
            )
            
            strict_results = strict_processor.process_video_complete(
                video_path=video_path,
                target_reduction=0.7
            )
            
            # Test with lenient parameters (low thresholds)
            lenient_processor = VideoProcessor(
                frame_diff_threshold=0.05,
                histogram_threshold=0.1,
                blur_threshold=50.0
            )
            
            lenient_results = lenient_processor.process_video_complete(
                video_path=video_path,
                target_reduction=0.7
            )
            
            # Both should succeed
            assert strict_results['success'] is True
            assert lenient_results['success'] is True
            
            # Compare results - lenient should potentially select more diverse frames
            strict_keyframes = len(strict_results['keyframes'])
            lenient_keyframes = len(lenient_results['keyframes'])
            
            # Both should have selected some keyframes
            assert strict_keyframes > 0
            assert lenient_keyframes > 0
            
            # Verify algorithm summaries are different
            strict_summary = strict_results['algorithm_summary']
            lenient_summary = lenient_results['algorithm_summary']
            
            assert strict_summary['frame_difference']['threshold'] == 0.8
            assert lenient_summary['frame_difference']['threshold'] == 0.05
            
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_keyframe_selection_logic_with_known_content(self):
        """Test keyframe selection logic with video content designed to test specific scenarios."""
        video_path = None
        try:
            # Create test video with specific pattern
            video_path = self._create_test_video_file(num_frames=16, fps=30)
            
            # Load and process video
            self.processor.load_video(video_path)
            frames = self.processor.extract_frames()
            
            # Process with IVP algorithms
            metrics = self.processor.process_with_ivp_algorithms(frames)
            
            # Verify metrics structure
            assert len(metrics) == 16
            for i, m in enumerate(metrics):
                assert m.frame_number == i
                assert m.timestamp == i / 30.0
                assert isinstance(m.difference_score, float)
                assert isinstance(m.histogram_score, float)
                assert isinstance(m.blur_score, float)
                assert m.is_keyframe is False  # Not set until keyframe selection
            
            # Select keyframes with different reduction targets
            keyframes_50 = self.processor.select_keyframes(target_reduction=0.5)
            
            # Reset keyframe flags
            for m in self.processor.frame_metrics:
                m.is_keyframe = False
            
            keyframes_75 = self.processor.select_keyframes(target_reduction=0.75)
            
            # Verify reduction targets are approximately met
            assert len(keyframes_50) <= 8  # 50% of 16
            assert len(keyframes_75) <= 4  # 25% of 16
            assert len(keyframes_50) >= len(keyframes_75)  # Less aggressive reduction should keep more
            
            # Verify keyframes are properly marked (check the last selection - 75% reduction)
            keyframe_75_numbers = {kf.frame_number for kf in keyframes_75}
            for m in self.processor.frame_metrics:
                if m.frame_number in keyframe_75_numbers:
                    assert m.is_keyframe
                else:
                    assert not m.is_keyframe
            
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_processing_with_sampling_and_frame_limits(self):
        """Test processing with frame sampling and limits."""
        video_path = None
        try:
            # Create larger test video
            video_path = self._create_test_video_file(num_frames=20, fps=30)
            
            # Test with frame limit
            results_limited = self.processor.process_video_complete(
                video_path=video_path,
                max_frames=10,
                sample_rate=1
            )
            
            # Test with sampling
            processor_sampled = VideoProcessor()
            results_sampled = processor_sampled.process_video_complete(
                video_path=video_path,
                max_frames=None,
                sample_rate=2  # Every 2nd frame
            )
            
            # Verify frame limits work
            assert results_limited['success'] is True
            assert results_limited['video_properties']['processed_frames'] == 10
            
            # Verify sampling works
            assert results_sampled['success'] is True
            expected_sampled_frames = 20 // 2  # Every 2nd frame
            assert results_sampled['video_properties']['processed_frames'] == expected_sampled_frames
            
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_error_handling_during_processing(self):
        """Test error handling during various processing stages."""
        # Test with non-existent file
        results_no_file = self.processor.process_video_complete("nonexistent.mp4")
        assert results_no_file['success'] is False
        assert 'error' in results_no_file
        
        # Test with empty file
        empty_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        empty_file.close()
        
        try:
            results_empty = self.processor.process_video_complete(empty_file.name)
            assert results_empty['success'] is False
            assert 'error' in results_empty
        finally:
            os.unlink(empty_file.name)
    
    def test_algorithm_consistency_across_runs(self):
        """Test that algorithms produce consistent results across multiple runs."""
        video_path = None
        try:
            # Create test video
            video_path = self._create_test_video_file(num_frames=8, fps=30)
            
            # Process same video multiple times
            results1 = self.processor.process_video_complete(video_path)
            
            processor2 = VideoProcessor(
                frame_diff_threshold=self.processor.frame_diff_threshold,
                histogram_threshold=self.processor.histogram_threshold,
                blur_threshold=self.processor.blur_threshold
            )
            results2 = processor2.process_video_complete(video_path)
            
            # Results should be identical
            assert results1['success'] == results2['success']
            assert len(results1['frame_metrics']) == len(results2['frame_metrics'])
            
            # Compare frame metrics (should be identical)
            for m1, m2 in zip(results1['frame_metrics'], results2['frame_metrics']):
                assert m1.frame_number == m2.frame_number
                assert abs(m1.difference_score - m2.difference_score) < 1e-6
                assert abs(m1.histogram_score - m2.histogram_score) < 1e-6
                assert abs(m1.blur_score - m2.blur_score) < 1e-6
            
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_performance_with_larger_video(self):
        """Test processing performance with a larger video."""
        video_path = None
        try:
            # Create larger test video
            video_path = self._create_test_video_file(num_frames=50, fps=30, width=200, height=200)
            
            # Measure processing time
            start_time = time.time()
            results = self.processor.process_video_complete(video_path)
            processing_time = time.time() - start_time
            
            # Verify processing completed successfully
            assert results['success'] is True
            assert results['video_properties']['processed_frames'] == 50
            
            # Processing should complete in reasonable time (< 10 seconds for test video)
            assert processing_time < 10.0, f"Processing took too long: {processing_time:.2f}s"
            
            # Verify keyframes were selected
            assert len(results['keyframes']) > 0
            assert len(results['keyframes']) < 50
            
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_edge_case_very_short_video(self):
        """Test processing with very short video (2 frames)."""
        video_path = None
        try:
            # Create minimal video
            video_path = self._create_test_video_file(num_frames=2, fps=30)
            
            results = self.processor.process_video_complete(video_path)
            
            # Should succeed even with minimal frames
            assert results['success'] is True
            assert results['video_properties']['processed_frames'] == 2
            assert len(results['frame_metrics']) == 2
            
            # Should select at least 1 keyframe
            assert len(results['keyframes']) >= 1
            
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_algorithm_score_ranges_and_validity(self):
        """Test that all algorithm scores are within expected ranges and valid."""
        video_path = None
        try:
            # Create test video with varied content
            video_path = self._create_test_video_file(num_frames=12, fps=30)
            
            results = self.processor.process_video_complete(video_path)
            assert results['success'] is True
            
            # Check all frame metrics have valid scores
            for metrics in results['frame_metrics']:
                # Frame difference scores should be 0-1
                assert 0.0 <= metrics.difference_score <= 1.0
                assert np.isfinite(metrics.difference_score)
                
                # Histogram scores should be non-negative
                assert metrics.histogram_score >= 0.0
                assert np.isfinite(metrics.histogram_score)
                
                # Blur scores should be non-negative
                assert metrics.blur_score >= 0.0
                assert np.isfinite(metrics.blur_score)
                
                # Timestamps should be reasonable
                assert metrics.timestamp >= 0.0
                assert np.isfinite(metrics.timestamp)
            
            # Check algorithm summary statistics
            alg_summary = results['algorithm_summary']
            
            for alg_name, stats in alg_summary.items():
                assert stats['min'] <= stats['max']
                assert np.isfinite(stats['min'])
                assert np.isfinite(stats['max'])
                assert np.isfinite(stats['mean'])
                assert np.isfinite(stats['std'])
                assert stats['std'] >= 0.0
            
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)


class TestVideoProcessingErrorConditions:
    """Test error conditions and edge cases in video processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
    
    def test_invalid_video_file_formats(self):
        """Test handling of invalid video file formats."""
        # Create text file with video extension
        text_file = tempfile.NamedTemporaryFile(suffix='.mp4', mode='w', delete=False)
        text_file.write("This is not a video file")
        text_file.close()
        
        try:
            results = self.processor.process_video_complete(text_file.name)
            assert results['success'] is False
            assert 'error' in results
        finally:
            os.unlink(text_file.name)
    
    def test_corrupted_video_handling(self):
        """Test handling of corrupted video files."""
        # Create file with video header but corrupted content
        corrupted_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        corrupted_file.write(b'\x00\x00\x00\x20ftypmp42')  # Partial MP4 header
        corrupted_file.write(b'\x00' * 100)  # Garbage data
        corrupted_file.close()
        
        try:
            results = self.processor.process_video_complete(corrupted_file.name)
            assert results['success'] is False
            assert 'error' in results
        finally:
            os.unlink(corrupted_file.name)
    
    def test_processing_with_invalid_parameters(self):
        """Test processing with invalid parameter values."""
        video_path = None
        try:
            # Create valid test video
            video_path = self._create_simple_test_video()
            
            # Test with invalid reduction target
            results = self.processor.process_video_complete(
                video_path=video_path,
                target_reduction=1.5  # Invalid: > 1.0
            )
            # Should handle gracefully or raise appropriate error
            # Implementation may clamp values or raise error
            
            # Test with invalid sample rate
            results = self.processor.process_video_complete(
                video_path=video_path,
                sample_rate=0  # Invalid: must be >= 1
            )
            assert results['success'] is False
            
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_memory_handling_with_large_frames(self):
        """Test memory handling when processing large frame dimensions."""
        # This test would ideally create a video with very large frames
        # For testing purposes, we'll simulate the scenario
        
        # Create frames that would be memory-intensive
        large_frames = [
            np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        # Processing should handle large frames without crashing
        try:
            metrics = self.processor.process_with_ivp_algorithms(large_frames)
            assert len(metrics) == 3
            assert all(isinstance(m, FrameMetrics) for m in metrics)
        except MemoryError:
            # If memory error occurs, it should be handled gracefully
            pytest.skip("Insufficient memory for large frame test")
    
    def test_algorithm_failure_recovery(self):
        """Test recovery from individual algorithm failures."""
        # Create frames that might cause algorithm issues
        problematic_frames = [
            np.zeros((10, 10, 3), dtype=np.uint8),  # All black
            np.ones((10, 10, 3), dtype=np.uint8) * 255,  # All white
            np.full((10, 10, 3), 128, dtype=np.uint8),  # All gray
        ]
        
        # Processing should handle edge cases gracefully
        metrics = self.processor.process_with_ivp_algorithms(problematic_frames)
        
        assert len(metrics) == 3
        for m in metrics:
            # All scores should be finite even for edge case frames
            assert np.isfinite(m.difference_score)
            assert np.isfinite(m.histogram_score)
            assert np.isfinite(m.blur_score)
    
    def _create_simple_test_video(self, num_frames=5):
        """Helper to create a simple test video."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_file.name, fourcc, 30, (100, 100))
        
        try:
            for i in range(num_frames):
                frame = np.full((100, 100, 3), i * 50, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            return temp_file.name
        except Exception as e:
            writer.release()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e