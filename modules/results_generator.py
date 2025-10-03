"""
Results Generator Module

Handles keyframe output generation, thumbnail creation, and JSON response building
for the FrameSift Lite video processing pipeline.
"""

import os
import json
import time
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from .video_processor import FrameMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsGenerationError(Exception):
    """Custom exception for results generation errors."""
    def __init__(self, message: str, error_code: str = "RESULTS_ERROR", details: str = None):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)


class ThumbnailCreationError(ResultsGenerationError):
    """Exception for thumbnail creation failures."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message, "THUMBNAIL_ERROR", details)


class PlotGenerationError(ResultsGenerationError):
    """Exception for plot generation failures."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message, "PLOT_ERROR", details)


class ResultsGenerator:
    """
    Generates processing results including keyframe thumbnails, JSON responses,
    and processing statistics for the FrameSift Lite application.
    """
    
    def __init__(self, output_dir: str = "static/results"):
        """
        Initialize the ResultsGenerator.
        
        Args:
            output_dir: Directory to save generated thumbnails and results
        """
        self.output_dir = output_dir
        self.thumbnail_size = (150, 100)  # Width x Height for thumbnails
        self.thumbnail_quality = 85  # JPEG quality for thumbnails
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ResultsGenerator initialized with output directory: {output_dir}")
    
    def create_keyframe_thumbnails(self, 
                                 frames: List[np.ndarray], 
                                 keyframe_metrics: List[FrameMetrics],
                                 session_id: str = None) -> List[Dict[str, Any]]:
        """
        Create thumbnail images for keyframes using Pillow.
        
        Args:
            frames: List of all video frames
            keyframe_metrics: List of FrameMetrics for selected keyframes
            session_id: Unique session identifier for file naming
            
        Returns:
            List of dictionaries with keyframe data and thumbnail paths
            
        Raises:
            ValueError: If frames and metrics don't match
            IOError: If thumbnail creation fails
        """
        if not keyframe_metrics:
            logger.warning("No keyframes provided for thumbnail creation")
            return []
        
        if session_id is None:
            session_id = str(int(time.time()))
        
        logger.info(f"Creating thumbnails for {len(keyframe_metrics)} keyframes")
        
        keyframe_data = []
        
        for i, metrics in enumerate(keyframe_metrics):
            try:
                # Get the frame for this keyframe
                frame_idx = metrics.frame_number
                
                if frame_idx >= len(frames):
                    logger.error(f"Frame index {frame_idx} out of range (total frames: {len(frames)})")
                    continue
                
                frame = frames[frame_idx]
                
                # Convert OpenCV frame (BGR) to PIL Image (RGB)
                if len(frame.shape) == 3:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # Grayscale frame
                    frame_rgb = frame
                
                # Create PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Create thumbnail
                thumbnail = pil_image.copy()
                thumbnail.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                
                # Generate thumbnail filename
                thumbnail_filename = f"keyframe_{session_id}_{frame_idx:06d}.jpg"
                thumbnail_path = os.path.join(self.output_dir, thumbnail_filename)
                
                # Save thumbnail
                thumbnail.save(thumbnail_path, "JPEG", quality=self.thumbnail_quality)
                
                # Create keyframe data entry
                keyframe_info = {
                    "frame_number": metrics.frame_number,
                    "timestamp": round(metrics.timestamp, 3),
                    "thumbnail_path": thumbnail_path,
                    "thumbnail_url": f"/static/results/{thumbnail_filename}",  # Web-accessible URL
                    "metrics": {
                        "frame_diff_score": round(metrics.difference_score, 4),
                        "histogram_score": round(metrics.histogram_score, 4),
                        "blur_score": round(metrics.blur_score, 2)
                    },
                    "is_keyframe": metrics.is_keyframe
                }
                
                keyframe_data.append(keyframe_info)
                
                logger.debug(f"Created thumbnail for frame {frame_idx}: {thumbnail_filename}")
                
            except Exception as e:
                logger.error(f"Failed to create thumbnail for frame {frame_idx}: {str(e)}")
                continue
        
        logger.info(f"Successfully created {len(keyframe_data)} keyframe thumbnails")
        return keyframe_data
    
    def calculate_processing_statistics(self, 
                                      all_metrics: List[FrameMetrics],
                                      processing_start_time: float,
                                      processing_end_time: float,
                                      video_properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive processing statistics.
        
        Args:
            all_metrics: List of all frame metrics from processing
            processing_start_time: Start time of processing (timestamp)
            processing_end_time: End time of processing (timestamp)
            video_properties: Optional video properties dictionary
            
        Returns:
            Dictionary with processing statistics
        """
        if not all_metrics:
            return {
                "total_frames": 0,
                "keyframes_extracted": 0,
                "reduction_percentage": 0.0,
                "processing_time": processing_end_time - processing_start_time
            }
        
        # Basic counts
        total_frames = len(all_metrics)
        keyframes = [m for m in all_metrics if m.is_keyframe]
        keyframes_extracted = len(keyframes)
        
        # Calculate reduction percentage
        reduction_percentage = ((total_frames - keyframes_extracted) / total_frames * 100) if total_frames > 0 else 0.0
        
        # Processing time
        processing_time = processing_end_time - processing_start_time
        
        # Algorithm score statistics
        frame_diff_scores = [m.difference_score for m in all_metrics]
        histogram_scores = [m.histogram_score for m in all_metrics]
        blur_scores = [m.blur_score for m in all_metrics]
        
        # Keyframe-specific statistics
        keyframe_stats = {}
        if keyframes:
            keyframe_frame_diff = [kf.difference_score for kf in keyframes]
            keyframe_histogram = [kf.histogram_score for kf in keyframes]
            keyframe_blur = [kf.blur_score for kf in keyframes]
            
            keyframe_stats = {
                "avg_frame_diff_score": round(np.mean(keyframe_frame_diff), 4),
                "avg_histogram_score": round(np.mean(keyframe_histogram), 4),
                "avg_blur_score": round(np.mean(keyframe_blur), 2),
                "temporal_distribution": self._calculate_temporal_distribution(keyframes)
            }
        
        # Compile statistics
        stats = {
            "total_frames": total_frames,
            "keyframes_extracted": keyframes_extracted,
            "reduction_percentage": round(reduction_percentage, 2),
            "processing_time": round(processing_time, 3),
            "frames_per_second_processed": round(total_frames / processing_time, 2) if processing_time > 0 else 0,
            
            # Algorithm score summaries for all frames
            "algorithm_scores": {
                "frame_difference": {
                    "min": round(min(frame_diff_scores), 4) if frame_diff_scores else 0,
                    "max": round(max(frame_diff_scores), 4) if frame_diff_scores else 0,
                    "mean": round(np.mean(frame_diff_scores), 4) if frame_diff_scores else 0,
                    "std": round(np.std(frame_diff_scores), 4) if frame_diff_scores else 0
                },
                "histogram_comparison": {
                    "min": round(min(histogram_scores), 4) if histogram_scores else 0,
                    "max": round(max(histogram_scores), 4) if histogram_scores else 0,
                    "mean": round(np.mean(histogram_scores), 4) if histogram_scores else 0,
                    "std": round(np.std(histogram_scores), 4) if histogram_scores else 0
                },
                "blur_detection": {
                    "min": round(min(blur_scores), 2) if blur_scores else 0,
                    "max": round(max(blur_scores), 2) if blur_scores else 0,
                    "mean": round(np.mean(blur_scores), 2) if blur_scores else 0,
                    "std": round(np.std(blur_scores), 2) if blur_scores else 0
                }
            },
            
            # Keyframe-specific statistics
            "keyframe_statistics": keyframe_stats
        }
        
        # Add video properties if provided
        if video_properties:
            stats["video_properties"] = video_properties
        
        logger.info(f"Processing statistics calculated: {keyframes_extracted}/{total_frames} keyframes ({reduction_percentage:.1f}% reduction)")
        
        return stats
    
    def _calculate_temporal_distribution(self, keyframes: List[FrameMetrics]) -> Dict[str, Any]:
        """
        Calculate temporal distribution statistics for keyframes.
        
        Args:
            keyframes: List of keyframe metrics
            
        Returns:
            Dictionary with temporal distribution statistics
        """
        if len(keyframes) < 2:
            return {
                "gaps": [],
                "mean_gap": 0,
                "std_gap": 0,
                "min_gap": 0,
                "max_gap": 0
            }
        
        # Sort keyframes by frame number
        sorted_keyframes = sorted(keyframes, key=lambda x: x.frame_number)
        
        # Calculate gaps between consecutive keyframes
        gaps = []
        for i in range(1, len(sorted_keyframes)):
            gap = sorted_keyframes[i].frame_number - sorted_keyframes[i-1].frame_number
            gaps.append(gap)
        
        return {
            "gaps": gaps,
            "mean_gap": round(np.mean(gaps), 2) if gaps else 0,
            "std_gap": round(np.std(gaps), 2) if gaps else 0,
            "min_gap": min(gaps) if gaps else 0,
            "max_gap": max(gaps) if gaps else 0
        }
    
    def build_json_response(self, 
                          keyframe_data: List[Dict[str, Any]],
                          processing_stats: Dict[str, Any],
                          plot_data: Dict[str, Any] = None,
                          plot_image_path: str = None,
                          success: bool = True,
                          error_info: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Build structured JSON response with keyframe data and metrics.
        
        Args:
            keyframe_data: List of keyframe information dictionaries
            processing_stats: Processing statistics dictionary
            plot_data: Optional plot data for visualization
            plot_image_path: Optional path to generated plot image
            success: Whether processing was successful
            error_info: Error information if processing failed
            
        Returns:
            Structured JSON response dictionary
        """
        response = {
            "success": success,
            "timestamp": time.time(),
            "keyframes": keyframe_data,
            "processing_stats": processing_stats
        }
        
        # Add plot information if available
        if plot_data:
            response["plot_data"] = plot_data
        
        if plot_image_path:
            # Convert absolute path to web-accessible URL
            if plot_image_path.startswith(self.output_dir):
                plot_filename = os.path.basename(plot_image_path)
                response["plot_image_url"] = f"/static/results/{plot_filename}"
            else:
                response["plot_image_url"] = plot_image_path
        
        # Add error information if processing failed
        if not success and error_info:
            response["error"] = error_info
        
        logger.info(f"JSON response built: {len(keyframe_data)} keyframes, success={success}")
        
        return response
    
    def save_results_to_file(self, 
                           response_data: Dict[str, Any], 
                           session_id: str = None) -> str:
        """
        Save processing results to a JSON file.
        
        Args:
            response_data: Complete response data dictionary
            session_id: Unique session identifier for file naming
            
        Returns:
            Path to saved results file
        """
        if session_id is None:
            session_id = str(int(time.time()))
        
        results_filename = f"results_{session_id}.json"
        results_path = os.path.join(self.output_dir, results_filename)
        
        try:
            with open(results_path, 'w') as f:
                json.dump(response_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to file: {results_path}")
            return results_path
            
        except Exception as e:
            logger.error(f"Failed to save results to file: {str(e)}")
            raise IOError(f"Could not save results: {str(e)}")
    
    def cleanup_session_files(self, session_id: str):
        """
        Clean up temporary files for a processing session.
        
        Args:
            session_id: Session identifier to clean up
        """
        try:
            # Find and remove files with this session ID
            files_removed = 0
            
            for filename in os.listdir(self.output_dir):
                if session_id in filename:
                    file_path = os.path.join(self.output_dir, filename)
                    try:
                        os.remove(file_path)
                        files_removed += 1
                        logger.debug(f"Removed session file: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not remove file {filename}: {str(e)}")
            
            logger.info(f"Cleaned up {files_removed} files for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {str(e)}")
    
    def get_thumbnail_dimensions(self) -> Tuple[int, int]:
        """
        Get the current thumbnail dimensions.
        
        Returns:
            Tuple of (width, height) for thumbnails
        """
        return self.thumbnail_size
    
    def set_thumbnail_dimensions(self, width: int, height: int):
        """
        Set new thumbnail dimensions.
        
        Args:
            width: Thumbnail width in pixels
            height: Thumbnail height in pixels
        """
        self.thumbnail_size = (width, height)
        logger.info(f"Thumbnail dimensions updated to {width}x{height}")
    
    def set_thumbnail_quality(self, quality: int):
        """
        Set JPEG quality for thumbnails.
        
        Args:
            quality: JPEG quality (1-100, higher = better quality)
        """
        if 1 <= quality <= 100:
            self.thumbnail_quality = quality
            logger.info(f"Thumbnail quality updated to {quality}")
        else:
            raise ValueError("Quality must be between 1 and 100")
    
    def generate_processing_summary(self, response_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of processing results.
        
        Args:
            response_data: Complete response data dictionary
            
        Returns:
            Formatted summary string
        """
        if not response_data.get("success", False):
            return "Processing failed: " + response_data.get("error", {}).get("message", "Unknown error")
        
        stats = response_data.get("processing_stats", {})
        keyframes = response_data.get("keyframes", [])
        
        summary_lines = [
            "=== FrameSift Lite Processing Summary ===",
            f"Total frames processed: {stats.get('total_frames', 0)}",
            f"Keyframes extracted: {stats.get('keyframes_extracted', 0)}",
            f"Frame reduction: {stats.get('reduction_percentage', 0):.1f}%",
            f"Processing time: {stats.get('processing_time', 0):.2f} seconds",
            f"Processing speed: {stats.get('frames_per_second_processed', 0):.1f} fps",
            ""
        ]
        
        # Add algorithm statistics
        algo_stats = stats.get("algorithm_scores", {})
        if algo_stats:
            summary_lines.append("Algorithm Score Ranges:")
            
            for algo_name, scores in algo_stats.items():
                summary_lines.append(f"  {algo_name.replace('_', ' ').title()}:")
                summary_lines.append(f"    Range: {scores.get('min', 0):.3f} - {scores.get('max', 0):.3f}")
                summary_lines.append(f"    Average: {scores.get('mean', 0):.3f} ± {scores.get('std', 0):.3f}")
            
            summary_lines.append("")
        
        # Add keyframe statistics
        kf_stats = stats.get("keyframe_statistics", {})
        if kf_stats:
            summary_lines.append("Keyframe Quality Metrics:")
            summary_lines.append(f"  Average frame difference: {kf_stats.get('avg_frame_diff_score', 0):.3f}")
            summary_lines.append(f"  Average histogram score: {kf_stats.get('avg_histogram_score', 0):.3f}")
            summary_lines.append(f"  Average blur score: {kf_stats.get('avg_blur_score', 0):.1f}")
            
            temporal = kf_stats.get("temporal_distribution", {})
            if temporal:
                summary_lines.append(f"  Frame gap range: {temporal.get('min_gap', 0)} - {temporal.get('max_gap', 0)}")
                summary_lines.append(f"  Average gap: {temporal.get('mean_gap', 0):.1f} frames")
        
        return "\n".join(summary_lines)
    
    def generate_processing_plots(self, 
                                all_metrics: List[FrameMetrics],
                                session_id: str = None,
                                plot_title: str = "IVP Algorithm Scores Over Time") -> Tuple[str, Dict[str, Any]]:
        """
        Generate matplotlib plots showing all three IVP technique scores over time.
        
        Args:
            all_metrics: List of all frame metrics from processing
            session_id: Unique session identifier for file naming
            plot_title: Title for the plot
            
        Returns:
            Tuple of (plot_image_path, plot_data_dict)
            
        Raises:
            ValueError: If no metrics provided
            IOError: If plot generation fails
        """
        if not all_metrics:
            raise ValueError("No frame metrics provided for plotting")
        
        if session_id is None:
            session_id = str(int(time.time()))
        
        logger.info(f"Generating processing plots for {len(all_metrics)} frames")
        
        try:
            # Set matplotlib style for clean plots
            mplstyle.use('default')
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            
            # Extract data for plotting
            frame_numbers = [m.frame_number for m in all_metrics]
            timestamps = [m.timestamp for m in all_metrics]
            frame_diff_scores = [m.difference_score for m in all_metrics]
            histogram_scores = [m.histogram_score for m in all_metrics]
            blur_scores = [m.blur_score for m in all_metrics]
            keyframe_indices = [i for i, m in enumerate(all_metrics) if m.is_keyframe]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(plot_title, fontsize=14, fontweight='bold')
            
            # Plot 1: Frame Difference and Histogram Scores (0-1 range)
            ax1.plot(frame_numbers, frame_diff_scores, 
                    label='Frame Difference (MAD)', 
                    color='#2E86AB', linewidth=1.5, alpha=0.8)
            ax1.plot(frame_numbers, histogram_scores, 
                    label='Histogram Comparison (Chi-Square)', 
                    color='#A23B72', linewidth=1.5, alpha=0.8)
            
            # Mark keyframes on first plot
            if keyframe_indices:
                keyframe_frame_nums = [frame_numbers[i] for i in keyframe_indices]
                keyframe_diff_scores = [frame_diff_scores[i] for i in keyframe_indices]
                ax1.scatter(keyframe_frame_nums, keyframe_diff_scores, 
                           color='#F18F01', s=30, alpha=0.8, zorder=5, 
                           label='Selected Keyframes')
            
            ax1.set_xlabel('Frame Number')
            ax1.set_ylabel('Score (0-1)')
            ax1.set_title('Frame Difference and Histogram Comparison Scores')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(1.1, max(max(frame_diff_scores), max(histogram_scores)) * 1.1))
            
            # Plot 2: Blur Detection Scores (separate scale)
            ax2.plot(frame_numbers, blur_scores, 
                    label='Blur Detection (Variance of Laplacian)', 
                    color='#C73E1D', linewidth=1.5, alpha=0.8)
            
            # Mark keyframes on second plot
            if keyframe_indices:
                keyframe_blur_scores = [blur_scores[i] for i in keyframe_indices]
                ax2.scatter(keyframe_frame_nums, keyframe_blur_scores, 
                           color='#F18F01', s=30, alpha=0.8, zorder=5, 
                           label='Selected Keyframes')
            
            ax2.set_xlabel('Frame Number')
            ax2.set_ylabel('Blur Score (Higher = Sharper)')
            ax2.set_title('Blur Detection Scores')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            # Add threshold lines if they make sense
            if blur_scores:
                blur_threshold = np.mean(blur_scores)  # Use mean as reference line
                ax2.axhline(y=blur_threshold, color='gray', linestyle='--', alpha=0.5, 
                           label=f'Mean Blur Score ({blur_threshold:.1f})')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"processing_plot_{session_id}.png"
            plot_path = os.path.join(self.output_dir, plot_filename)
            
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()  # Close to free memory
            
            # Prepare plot data for JSON response
            plot_data = {
                "frame_numbers": frame_numbers,
                "timestamps": [round(t, 3) for t in timestamps],
                "difference_scores": [round(s, 4) for s in frame_diff_scores],
                "histogram_scores": [round(s, 4) for s in histogram_scores],
                "blur_scores": [round(s, 2) for s in blur_scores],
                "keyframe_indices": keyframe_indices,
                "plot_statistics": {
                    "total_frames": len(all_metrics),
                    "keyframes_count": len(keyframe_indices),
                    "score_ranges": {
                        "frame_difference": {
                            "min": round(min(frame_diff_scores), 4),
                            "max": round(max(frame_diff_scores), 4)
                        },
                        "histogram_comparison": {
                            "min": round(min(histogram_scores), 4),
                            "max": round(max(histogram_scores), 4)
                        },
                        "blur_detection": {
                            "min": round(min(blur_scores), 2),
                            "max": round(max(blur_scores), 2)
                        }
                    }
                }
            }
            
            logger.info(f"Processing plot generated successfully: {plot_path}")
            return plot_path, plot_data
            
        except Exception as e:
            logger.error(f"Failed to generate processing plots: {str(e)}")
            raise IOError(f"Plot generation failed: {str(e)}")
    
    def create_keyframe_timeline_plot(self, 
                                    keyframe_metrics: List[FrameMetrics],
                                    total_frames: int,
                                    session_id: str = None) -> str:
        """
        Create a simple timeline plot showing keyframe distribution.
        
        Args:
            keyframe_metrics: List of keyframe metrics
            total_frames: Total number of frames in video
            session_id: Unique session identifier for file naming
            
        Returns:
            Path to generated timeline plot image
        """
        if not keyframe_metrics:
            raise ValueError("No keyframe metrics provided")
        
        if session_id is None:
            session_id = str(int(time.time()))
        
        try:
            # Create timeline plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 3))
            
            # Extract keyframe positions
            keyframe_positions = [kf.frame_number for kf in keyframe_metrics]
            keyframe_timestamps = [kf.timestamp for kf in keyframe_metrics]
            
            # Create timeline
            ax.scatter(keyframe_positions, [1] * len(keyframe_positions), 
                      s=50, color='#F18F01', alpha=0.8, zorder=5)
            
            # Add timeline line
            ax.plot([0, total_frames], [1, 1], color='gray', linewidth=2, alpha=0.5)
            
            # Styling
            ax.set_xlim(0, total_frames)
            ax.set_ylim(0.5, 1.5)
            ax.set_xlabel('Frame Number')
            ax.set_title(f'Keyframe Distribution Timeline ({len(keyframe_positions)} keyframes)')
            ax.set_yticks([])
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add frame count annotations
            ax.text(0, 0.7, '0', ha='center', va='top', fontsize=8)
            ax.text(total_frames, 0.7, str(total_frames), ha='center', va='top', fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            timeline_filename = f"keyframe_timeline_{session_id}.png"
            timeline_path = os.path.join(self.output_dir, timeline_filename)
            
            plt.savefig(timeline_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Keyframe timeline plot generated: {timeline_path}")
            return timeline_path
            
        except Exception as e:
            logger.error(f"Failed to generate timeline plot: {str(e)}")
            raise IOError(f"Timeline plot generation failed: {str(e)}")
    
    def create_algorithm_comparison_plot(self, 
                                       all_metrics: List[FrameMetrics],
                                       session_id: str = None) -> str:
        """
        Create a comparison plot showing algorithm score distributions.
        
        Args:
            all_metrics: List of all frame metrics
            session_id: Unique session identifier for file naming
            
        Returns:
            Path to generated comparison plot image
        """
        if not all_metrics:
            raise ValueError("No frame metrics provided")
        
        if session_id is None:
            session_id = str(int(time.time()))
        
        try:
            # Extract scores
            frame_diff_scores = [m.difference_score for m in all_metrics]
            histogram_scores = [m.histogram_score for m in all_metrics]
            blur_scores = [m.blur_score for m in all_metrics]
            
            # Create comparison plot with histograms
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Algorithm Score Distributions', fontsize=14, fontweight='bold')
            
            # Frame Difference histogram
            axes[0].hist(frame_diff_scores, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
            axes[0].set_title('Frame Difference (MAD)')
            axes[0].set_xlabel('Score')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Histogram Comparison histogram
            axes[1].hist(histogram_scores, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
            axes[1].set_title('Histogram Comparison')
            axes[1].set_xlabel('Score')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True, alpha=0.3)
            
            # Blur Detection histogram
            axes[2].hist(blur_scores, bins=30, alpha=0.7, color='#C73E1D', edgecolor='black')
            axes[2].set_title('Blur Detection')
            axes[2].set_xlabel('Score')
            axes[2].set_ylabel('Frequency')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            comparison_filename = f"algorithm_comparison_{session_id}.png"
            comparison_path = os.path.join(self.output_dir, comparison_filename)
            
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Algorithm comparison plot generated: {comparison_path}")
            return comparison_path
            
        except Exception as e:
            logger.error(f"Failed to generate comparison plot: {str(e)}")
            raise IOError(f"Comparison plot generation failed: {str(e)}")