"""
Unit tests for the ResultsGenerator module.
"""

import unittest
import tempfile
import shutil
import os
import json
import time
import numpy as np
from unittest.mock import patch, MagicMock

from modules.results_generator import ResultsGenerator
from modules.video_processor import FrameMetrics


class TestResultsGenerator(unittest.TestCase):
    """Test cases for ResultsGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.results_gen = ResultsGenerator(output_dir=self.test_dir)
        
        # Create sample frame metrics
        self.sample_metrics = [
            FrameMetrics(
                frame_number=0,
                timestamp=0.0,
                difference_score=0.1,
                histogram_score=0.2,
                blur_score=150.0,
                is_keyframe=True
            ),
            FrameMetrics(
                frame_number=1,
                timestamp=0.33,
                difference_score=0.05,
                histogram_score=0.1,
                blur_score=80.0,
                is_keyframe=False
            ),
            FrameMetrics(
                frame_number=2,
                timestamp=0.67,
                difference_score=0.3,
                histogram_score=0.6,
                blur_score=200.0,
                is_keyframe=True
            )
        ]
        
        # Create sample frames (mock numpy arrays)
        self.sample_frames = [
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ResultsGenerator initialization."""
        self.assertEqual(self.results_gen.output_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertEqual(self.results_gen.thumbnail_size, (150, 100))
        self.assertEqual(self.results_gen.thumbnail_quality, 85)
    
    def test_create_keyframe_thumbnails(self):
        """Test keyframe thumbnail creation."""
        keyframes = [m for m in self.sample_metrics if m.is_keyframe]
        
        with patch('cv2.cvtColor') as mock_cvtColor, \
             patch('PIL.Image.fromarray') as mock_fromarray:
            
            # Mock PIL Image operations
            mock_image = MagicMock()
            mock_thumbnail = MagicMock()
            mock_fromarray.return_value = mock_image
            mock_image.copy.return_value = mock_thumbnail
            
            # Mock cv2 color conversion
            mock_cvtColor.return_value = self.sample_frames[0]
            
            result = self.results_gen.create_keyframe_thumbnails(
                self.sample_frames, keyframes, "test_session"
            )
            
            # Verify results
            self.assertEqual(len(result), 2)  # Two keyframes
            self.assertEqual(result[0]["frame_number"], 0)
            self.assertEqual(result[1]["frame_number"], 2)
            
            # Check that thumbnails were "saved"
            mock_thumbnail.save.assert_called()
    
    def test_calculate_processing_statistics(self):
        """Test processing statistics calculation."""
        start_time = time.time()
        end_time = start_time + 5.0  # 5 second processing time
        
        stats = self.results_gen.calculate_processing_statistics(
            self.sample_metrics, start_time, end_time
        )
        
        # Verify basic statistics
        self.assertEqual(stats["total_frames"], 3)
        self.assertEqual(stats["keyframes_extracted"], 2)
        self.assertEqual(stats["reduction_percentage"], 33.33)  # (3-2)/3 * 100
        self.assertEqual(stats["processing_time"], 5.0)
        
        # Verify algorithm scores are included
        self.assertIn("algorithm_scores", stats)
        self.assertIn("frame_difference", stats["algorithm_scores"])
        self.assertIn("histogram_comparison", stats["algorithm_scores"])
        self.assertIn("blur_detection", stats["algorithm_scores"])
    
    def test_build_json_response(self):
        """Test JSON response building."""
        keyframe_data = [
            {
                "frame_number": 0,
                "timestamp": 0.0,
                "thumbnail_path": "/test/path.jpg",
                "metrics": {"frame_diff_score": 0.1}
            }
        ]
        
        processing_stats = {
            "total_frames": 3,
            "keyframes_extracted": 1,
            "processing_time": 2.5
        }
        
        response = self.results_gen.build_json_response(
            keyframe_data, processing_stats, success=True
        )
        
        # Verify response structure
        self.assertTrue(response["success"])
        self.assertEqual(len(response["keyframes"]), 1)
        self.assertEqual(response["processing_stats"]["total_frames"], 3)
        self.assertIn("timestamp", response)
    
    def test_build_json_response_with_error(self):
        """Test JSON response building with error."""
        error_info = {
            "code": "PROCESSING_ERROR",
            "message": "Test error message"
        }
        
        response = self.results_gen.build_json_response(
            [], {}, success=False, error_info=error_info
        )
        
        # Verify error response
        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["code"], "PROCESSING_ERROR")
        self.assertEqual(response["error"]["message"], "Test error message")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_processing_plots(self, mock_close, mock_savefig):
        """Test processing plot generation."""
        plot_path, plot_data = self.results_gen.generate_processing_plots(
            self.sample_metrics, "test_session"
        )
        
        # Verify plot was "saved"
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Verify plot data structure
        self.assertIn("frame_numbers", plot_data)
        self.assertIn("difference_scores", plot_data)
        self.assertIn("histogram_scores", plot_data)
        self.assertIn("blur_scores", plot_data)
        self.assertIn("keyframe_indices", plot_data)
        
        # Verify data content
        self.assertEqual(len(plot_data["frame_numbers"]), 3)
        self.assertEqual(len(plot_data["keyframe_indices"]), 2)  # Two keyframes
    
    def test_save_results_to_file(self):
        """Test saving results to JSON file."""
        test_data = {
            "success": True,
            "keyframes": [],
            "processing_stats": {"total_frames": 10}
        }
        
        file_path = self.results_gen.save_results_to_file(test_data, "test_session")
        
        # Verify file was created
        self.assertTrue(os.path.exists(file_path))
        self.assertIn("test_session", file_path)
        
        # Verify file content
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data["success"], True)
        self.assertEqual(loaded_data["processing_stats"]["total_frames"], 10)
    
    def test_cleanup_session_files(self):
        """Test session file cleanup."""
        # Create some test files
        test_files = [
            "keyframe_test_session_001.jpg",
            "results_test_session.json",
            "plot_test_session.png",
            "other_file.txt"  # Should not be removed
        ]
        
        for filename in test_files:
            file_path = os.path.join(self.test_dir, filename)
            with open(file_path, 'w') as f:
                f.write("test content")
        
        # Cleanup session files
        self.results_gen.cleanup_session_files("test_session")
        
        # Verify only session files were removed
        remaining_files = os.listdir(self.test_dir)
        self.assertIn("other_file.txt", remaining_files)
        self.assertNotIn("keyframe_test_session_001.jpg", remaining_files)
        self.assertNotIn("results_test_session.json", remaining_files)
        self.assertNotIn("plot_test_session.png", remaining_files)
    
    def test_thumbnail_dimensions_management(self):
        """Test thumbnail dimension getter/setter."""
        # Test getter
        width, height = self.results_gen.get_thumbnail_dimensions()
        self.assertEqual((width, height), (150, 100))
        
        # Test setter
        self.results_gen.set_thumbnail_dimensions(200, 150)
        width, height = self.results_gen.get_thumbnail_dimensions()
        self.assertEqual((width, height), (200, 150))
    
    def test_thumbnail_quality_management(self):
        """Test thumbnail quality setter."""
        # Test valid quality
        self.results_gen.set_thumbnail_quality(95)
        self.assertEqual(self.results_gen.thumbnail_quality, 95)
        
        # Test invalid quality
        with self.assertRaises(ValueError):
            self.results_gen.set_thumbnail_quality(150)  # Too high
        
        with self.assertRaises(ValueError):
            self.results_gen.set_thumbnail_quality(0)  # Too low
    
    def test_generate_processing_summary(self):
        """Test processing summary generation."""
        response_data = {
            "success": True,
            "keyframes": [{"frame_number": 0}, {"frame_number": 10}],
            "processing_stats": {
                "total_frames": 100,
                "keyframes_extracted": 2,
                "reduction_percentage": 98.0,
                "processing_time": 3.5,
                "frames_per_second_processed": 28.6
            }
        }
        
        summary = self.results_gen.generate_processing_summary(response_data)
        
        # Verify summary content
        self.assertIn("Total frames processed: 100", summary)
        self.assertIn("Keyframes extracted: 2", summary)
        self.assertIn("Frame reduction: 98.0%", summary)
        self.assertIn("Processing time: 3.50 seconds", summary)
    
    def test_generate_processing_summary_with_error(self):
        """Test processing summary generation with error."""
        response_data = {
            "success": False,
            "error": {
                "message": "Test processing error"
            }
        }
        
        summary = self.results_gen.generate_processing_summary(response_data)
        
        # Verify error summary
        self.assertIn("Processing failed", summary)
        self.assertIn("Test processing error", summary)


if __name__ == '__main__':
    unittest.main()