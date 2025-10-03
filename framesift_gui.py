#!/usr/bin/env python3
"""
FrameSift Lite - Desktop GUI Application
A Tkinter-based keyframe extraction tool with all Flask app features
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tkinter.font as font
from PIL import Image, ImageTk
import os
import sys
import threading
import time
import json
import uuid
import shutil
from pathlib import Path
import logging
import traceback
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import numpy as np

# Import the video processing modules
from modules.video_processor import VideoProcessor
from modules.results_generator import ResultsGenerator
from modules.news_detector import NewsContentDetector, NewsKeyframeExtractor
from modules.simple_fan_control import start_processing_fans, stop_processing_fans, is_fan_control_active

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('framesift_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FrameSiftGUI:
    """Main GUI application class for FrameSift Lite"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_variables()
        self.setup_styles()
        self.create_widgets()
        self.setup_logging_display()
        
        # Processing state
        self.current_video_path = None
        self.processing_results = None
        self.session_id = None
        self.processing_thread = None
        
        # News processing state
        self.current_news_video_path = None
        self.news_session_id = None
        self.news_processing_thread = None
        self.news_output_dir = None
        
        # Simple fan control
        self.update_fan_status()  # Start fan status updates
        
        # Setup cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        logger.info("FrameSift GUI initialized successfully")
    
    def on_closing(self):
        """Handle application closing"""
        try:
            # Stop fan control
            stop_processing_fans()
            logger.info("Fan control stopped on application close")
        except Exception as e:
            logger.error(f"Error stopping fan control: {e}")
        finally:
            self.root.destroy()
    
    def setup_window(self):
        """Configure the main window"""
        self.root.title("FrameSift Lite - Keyframe Extraction Tool")
        self.root.geometry("1800x1200")  # 150% larger
        self.root.minsize(1500, 900)     # 150% larger minimum
        
        # Set window state to maximized for better experience
        self.root.state('zoomed')  # Windows maximize
        
        # Configure window icon if available
        try:
            # You can add an icon file if you have one
            pass
        except:
            pass
    
    def setup_variables(self):
        """Initialize GUI variables"""
        # Parameter variables
        self.frame_diff_var = tk.DoubleVar(value=0.3)
        self.histogram_var = tk.DoubleVar(value=0.5)
        self.blur_var = tk.DoubleVar(value=50.0)
        self.target_reduction_var = tk.DoubleVar(value=0.75)
        
        # Status variables
        self.status_var = tk.StringVar(value="Ready to process video")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.video_info_var = tk.StringVar(value="No video loaded")
        
        # Results variables
        self.results_text = tk.StringVar(value="")
        
        # News detection variables
        self.news_headline_threshold_var = tk.DoubleVar(value=0.5)
        self.news_person_threshold_var = tk.DoubleVar(value=0.4)
        self.news_scene_threshold_var = tk.DoubleVar(value=0.3)
        self.news_min_gap_var = tk.DoubleVar(value=2.0)
        self.news_video_path_var = tk.StringVar()
        self.news_video_info_var = tk.StringVar(value="No video loaded")
        self.news_results_var = tk.StringVar(value="No news analysis results yet")
    
    def setup_styles(self):
        """Configure ttk styles with 150% scaling"""
        self.style = ttk.Style()
        
        # Configure custom styles with larger fonts (150% scaling)
        self.style.configure('Title.TLabel', font=('Arial', 24, 'bold'))      # 16 -> 24
        self.style.configure('Heading.TLabel', font=('Arial', 18, 'bold'))    # 12 -> 18
        self.style.configure('Info.TLabel', font=('Arial', 15))               # 10 -> 15
        self.style.configure('Success.TLabel', foreground='green', font=('Arial', 15, 'bold'))
        self.style.configure('Error.TLabel', foreground='red', font=('Arial', 15, 'bold'))
        
        # Configure button styles with larger fonts and padding
        self.style.configure('Large.TButton', font=('Arial', 14, 'bold'), padding=(15, 10))
        self.style.configure('Process.TButton', font=('Arial', 16, 'bold'), padding=(20, 15))
        
        # Configure notebook styles
        self.style.configure('TNotebook.Tab', font=('Arial', 14, 'bold'), padding=(20, 10))
        
        # Configure frame styles with more padding
        self.style.configure('TLabelframe.Label', font=('Arial', 16, 'bold'))
        self.style.configure('TLabelframe', borderwidth=2, relief='groove')
        
        # Configure progress bar style
        self.style.configure('Large.Horizontal.TProgressbar', troughcolor='lightgray',
                           borderwidth=2, lightcolor='blue', darkcolor='blue')
    
    def create_widgets(self):
        """Create and layout all GUI widgets"""
        # Create main container with larger padding
        main_frame = ttk.Frame(self.root, padding="20")  # Increased from 10 to 20
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew", pady=(0, 15))  # Increased padding
        main_frame.rowconfigure(0, weight=1)
        
        # Create tabs with larger spacing
        self.create_upload_tab()
        self.create_parameters_tab()
        self.create_news_tab()
        self.create_results_tab()
        self.create_logs_tab()
        
        # Create status bar
        self.create_status_bar(main_frame)
    
    def create_upload_tab(self):
        """Create the video upload and processing tab"""
        upload_frame = ttk.Frame(self.notebook, padding="30")  # Increased padding
        self.notebook.add(upload_frame, text="📹 Video Upload")
        
        # Title
        title_label = ttk.Label(upload_frame, text="FrameSift Lite - Keyframe Extraction", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 30))  # Increased spacing
        
        # Video selection section
        video_section = ttk.LabelFrame(upload_frame, text="Video Selection", padding="25")  # Increased padding
        video_section.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 30))  # Increased spacing
        upload_frame.columnconfigure(0, weight=1)
        video_section.columnconfigure(1, weight=1)
        
        ttk.Label(video_section, text="Select Video File:", font=('Arial', 16)).grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        self.video_path_var = tk.StringVar()
        video_entry = ttk.Entry(video_section, textvariable=self.video_path_var, width=60, font=('Arial', 14))  # Larger entry
        video_entry.grid(row=0, column=1, sticky="ew", padx=(15, 15))
        
        browse_btn = ttk.Button(video_section, text="📁 Browse...", command=self.browse_video, style='Large.TButton')
        browse_btn.grid(row=0, column=2, padx=(0, 0))
        
        # Video info display
        self.video_info_label = ttk.Label(video_section, textvariable=self.video_info_var, 
                                         style='Info.TLabel')
        self.video_info_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(15, 0))
        
        # Quick parameters section
        params_section = ttk.LabelFrame(upload_frame, text="Quick Parameters", padding="25")  # Increased padding
        params_section.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 30))  # Increased spacing
        params_section.columnconfigure(1, weight=1)
        
        # Frame difference threshold
        ttk.Label(params_section, text="Frame Difference:", font=('Arial', 16)).grid(row=0, column=0, sticky="w", pady=(5, 5))
        frame_diff_scale = ttk.Scale(params_section, from_=0.1, to=1.0, 
                                    variable=self.frame_diff_var, orient="horizontal", length=400)  # Longer scale
        frame_diff_scale.grid(row=0, column=1, sticky="ew", padx=(15, 15))
        self.frame_diff_label = ttk.Label(params_section, text="0.3", font=('Arial', 16, 'bold'))
        self.frame_diff_label.grid(row=0, column=2)
        
        # Histogram threshold
        ttk.Label(params_section, text="Histogram:", font=('Arial', 16)).grid(row=1, column=0, sticky="w", pady=(5, 5))
        histogram_scale = ttk.Scale(params_section, from_=0.1, to=1.0, 
                                   variable=self.histogram_var, orient="horizontal", length=400)
        histogram_scale.grid(row=1, column=1, sticky="ew", padx=(15, 15))
        self.histogram_label = ttk.Label(params_section, text="0.5", font=('Arial', 16, 'bold'))
        self.histogram_label.grid(row=1, column=2)
        
        # Blur threshold
        ttk.Label(params_section, text="Blur Threshold:", font=('Arial', 16)).grid(row=2, column=0, sticky="w", pady=(5, 5))
        blur_scale = ttk.Scale(params_section, from_=10, to=200, 
                              variable=self.blur_var, orient="horizontal", length=400)
        blur_scale.grid(row=2, column=1, sticky="ew", padx=(15, 15))
        self.blur_label = ttk.Label(params_section, text="50", font=('Arial', 16, 'bold'))
        self.blur_label.grid(row=2, column=2)
        
        # Target reduction
        ttk.Label(params_section, text="Target Reduction:", font=('Arial', 16)).grid(row=3, column=0, sticky="w", pady=(5, 5))
        reduction_scale = ttk.Scale(params_section, from_=0.5, to=0.9, 
                                   variable=self.target_reduction_var, orient="horizontal", length=400)
        reduction_scale.grid(row=3, column=1, sticky="ew", padx=(15, 15))
        self.reduction_label = ttk.Label(params_section, text="75%", font=('Arial', 16, 'bold'))
        self.reduction_label.grid(row=3, column=2)
        
        # Bind scale events to update labels
        frame_diff_scale.configure(command=self.update_frame_diff_label)
        histogram_scale.configure(command=self.update_histogram_label)
        blur_scale.configure(command=self.update_blur_label)
        reduction_scale.configure(command=self.update_reduction_label)
        
        # Processing section
        process_section = ttk.LabelFrame(upload_frame, text="Processing", padding="25")  # Increased padding
        process_section.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 30))
        process_section.columnconfigure(0, weight=1)
        
        # Progress bar - much larger
        self.progress_bar = ttk.Progressbar(process_section, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.configure(style='Large.Horizontal.TProgressbar')
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 20), ipady=8)  # Taller progress bar
        
        # Control buttons - much larger
        button_frame = ttk.Frame(process_section)
        button_frame.grid(row=1, column=0, pady=(0, 0))
        
        self.process_btn = ttk.Button(button_frame, text="🎬 Process Video", 
                                     command=self.start_processing, state="disabled", style='Process.TButton')
        self.process_btn.pack(side="left", padx=(0, 20))  # More spacing
        
        self.cancel_btn = ttk.Button(button_frame, text="❌ Cancel", 
                                    command=self.cancel_processing, state="disabled", style='Large.TButton')
        self.cancel_btn.pack(side="left", padx=(0, 20))
        
        self.reset_btn = ttk.Button(button_frame, text="🔄 Reset", 
                                   command=self.reset_session, style='Large.TButton')
        self.reset_btn.pack(side="left")
    
    def create_parameters_tab(self):
        """Create the detailed parameters configuration tab"""
        params_frame = ttk.Frame(self.notebook, padding="30")  # Increased padding
        self.notebook.add(params_frame, text="⚙️ Parameters")
        
        # Title
        ttk.Label(params_frame, text="Advanced Processing Parameters", 
                 style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 30))
        
        # Create scrollable frame
        canvas = tk.Canvas(params_frame)
        scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        
        params_frame.columnconfigure(0, weight=1)
        params_frame.rowconfigure(1, weight=1)
        
        # Algorithm explanations and detailed controls
        self.create_detailed_parameters(scrollable_frame)
    
    def create_detailed_parameters(self, parent):
        """Create detailed parameter controls with explanations"""
        
        # Frame Difference Algorithm
        fd_frame = ttk.LabelFrame(parent, text="Frame Difference Algorithm (MAD)", padding="15")
        fd_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        parent.columnconfigure(0, weight=1)
        fd_frame.columnconfigure(1, weight=1)
        
        ttk.Label(fd_frame, text="Purpose:", style='Heading.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Label(fd_frame, text="Removes redundant frames by detecting significant changes", 
                 style='Info.TLabel').grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(fd_frame, text="Method:", style='Heading.TLabel').grid(row=1, column=0, sticky="w")
        ttk.Label(fd_frame, text="Mean Absolute Difference between consecutive frames", 
                 style='Info.TLabel').grid(row=1, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(fd_frame, text="Threshold:", style='Heading.TLabel').grid(row=2, column=0, sticky="w")
        fd_detailed_scale = ttk.Scale(fd_frame, from_=0.01, to=1.0, 
                                     variable=self.frame_diff_var, orient="horizontal")
        fd_detailed_scale.grid(row=2, column=1, sticky="ew", padx=(10, 0))
        
        ttk.Label(fd_frame, text="Lower values = more sensitive to changes (fewer frames removed)", 
                 style='Info.TLabel').grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        # Histogram Comparison Algorithm
        hc_frame = ttk.LabelFrame(parent, text="Histogram Comparison Algorithm", padding="15")
        hc_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        hc_frame.columnconfigure(1, weight=1)
        
        ttk.Label(hc_frame, text="Purpose:", style='Heading.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Label(hc_frame, text="Detects scene boundaries and shot transitions", 
                 style='Info.TLabel').grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(hc_frame, text="Method:", style='Heading.TLabel').grid(row=1, column=0, sticky="w")
        ttk.Label(hc_frame, text="Chi-Square distance between color histograms", 
                 style='Info.TLabel').grid(row=1, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(hc_frame, text="Threshold:", style='Heading.TLabel').grid(row=2, column=0, sticky="w")
        hc_detailed_scale = ttk.Scale(hc_frame, from_=0.01, to=1.0, 
                                     variable=self.histogram_var, orient="horizontal")
        hc_detailed_scale.grid(row=2, column=1, sticky="ew", padx=(10, 0))
        
        ttk.Label(hc_frame, text="Higher values = more sensitive to scene changes", 
                 style='Info.TLabel').grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        # Blur Detection Algorithm
        bd_frame = ttk.LabelFrame(parent, text="Blur Detection Algorithm", padding="15")
        bd_frame.grid(row=2, column=0, sticky="ew", pady=(0, 20))
        bd_frame.columnconfigure(1, weight=1)
        
        ttk.Label(bd_frame, text="Purpose:", style='Heading.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Label(bd_frame, text="Filters out blurry or low-quality frames", 
                 style='Info.TLabel').grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(bd_frame, text="Method:", style='Heading.TLabel').grid(row=1, column=0, sticky="w")
        ttk.Label(bd_frame, text="Variance of Laplacian for sharpness measurement", 
                 style='Info.TLabel').grid(row=1, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(bd_frame, text="Threshold:", style='Heading.TLabel').grid(row=2, column=0, sticky="w")
        bd_detailed_scale = ttk.Scale(bd_frame, from_=1, to=300, 
                                     variable=self.blur_var, orient="horizontal")
        bd_detailed_scale.grid(row=2, column=1, sticky="ew", padx=(10, 0))
        
        ttk.Label(bd_frame, text="Higher values = stricter quality requirements (more frames filtered)", 
                 style='Info.TLabel').grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        # Processing Settings
        ps_frame = ttk.LabelFrame(parent, text="Processing Settings", padding="15")
        ps_frame.grid(row=3, column=0, sticky="ew", pady=(0, 20))
        ps_frame.columnconfigure(1, weight=1)
        
        ttk.Label(ps_frame, text="Target Reduction:", style='Heading.TLabel').grid(row=0, column=0, sticky="w")
        tr_detailed_scale = ttk.Scale(ps_frame, from_=0.3, to=0.95, 
                                     variable=self.target_reduction_var, orient="horizontal")
        tr_detailed_scale.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        
        ttk.Label(ps_frame, text="Percentage of frames to remove (0.75 = 75% reduction)", 
                 style='Info.TLabel').grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        # Preset buttons - larger
        preset_frame = ttk.LabelFrame(parent, text="Parameter Presets", padding="20")
        preset_frame.grid(row=4, column=0, sticky="ew", pady=(0, 20))
        
        ttk.Button(preset_frame, text="🎯 Conservative", 
                  command=self.load_conservative_preset, style='Large.TButton').pack(side="left", padx=(0, 15))
        ttk.Button(preset_frame, text="⚖️ Balanced", 
                  command=self.load_balanced_preset, style='Large.TButton').pack(side="left", padx=(0, 15))
        ttk.Button(preset_frame, text="🚀 Aggressive", 
                  command=self.load_aggressive_preset, style='Large.TButton').pack(side="left", padx=(0, 15))
    
    def create_news_tab(self):
        """Create the news content detection tab"""
        news_frame = ttk.Frame(self.notebook, padding="30")
        self.notebook.add(news_frame, text="📺 News Detection")
        
        # Title
        title_label = ttk.Label(news_frame, text="News Content Detection", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 30))
        
        # Configure grid weights
        news_frame.columnconfigure(0, weight=1)
        
        # Video selection section for news
        news_video_section = ttk.LabelFrame(news_frame, text="News Video Selection", padding="25")
        news_video_section.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 30))
        news_video_section.columnconfigure(1, weight=1)
        
        ttk.Label(news_video_section, text="Select News Video:", font=('Arial', 16)).grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        news_video_entry = ttk.Entry(news_video_section, textvariable=self.news_video_path_var, width=60, font=('Arial', 14))
        news_video_entry.grid(row=0, column=1, sticky="ew", padx=(15, 15))
        
        news_browse_btn = ttk.Button(news_video_section, text="📁 Browse...", command=self.browse_news_video, style='Large.TButton')
        news_browse_btn.grid(row=0, column=2, padx=(0, 0))
        
        # News video info display
        self.news_video_info_label = ttk.Label(news_video_section, textvariable=self.news_video_info_var, style='Info.TLabel')
        self.news_video_info_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(15, 0))
        
        # News detection parameters
        news_params_section = ttk.LabelFrame(news_frame, text="Detection Parameters", padding="25")
        news_params_section.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 30))
        news_params_section.columnconfigure(1, weight=1)
        
        # Headline change threshold
        ttk.Label(news_params_section, text="Headline Change:", font=('Arial', 16)).grid(row=0, column=0, sticky="w", pady=(5, 5))
        headline_scale = ttk.Scale(news_params_section, from_=0.1, to=1.0, 
                                  variable=self.news_headline_threshold_var, orient="horizontal", length=400)
        headline_scale.grid(row=0, column=1, sticky="ew", padx=(15, 15))
        self.news_headline_label = ttk.Label(news_params_section, text="0.5", font=('Arial', 16, 'bold'))
        self.news_headline_label.grid(row=0, column=2)
        
        # Person change threshold
        ttk.Label(news_params_section, text="Person Change:", font=('Arial', 16)).grid(row=1, column=0, sticky="w", pady=(5, 5))
        person_scale = ttk.Scale(news_params_section, from_=0.1, to=1.0, 
                                variable=self.news_person_threshold_var, orient="horizontal", length=400)
        person_scale.grid(row=1, column=1, sticky="ew", padx=(15, 15))
        self.news_person_label = ttk.Label(news_params_section, text="0.4", font=('Arial', 16, 'bold'))
        self.news_person_label.grid(row=1, column=2)
        
        # Scene change threshold
        ttk.Label(news_params_section, text="Scene Change:", font=('Arial', 16)).grid(row=2, column=0, sticky="w", pady=(5, 5))
        scene_scale = ttk.Scale(news_params_section, from_=0.1, to=1.0, 
                               variable=self.news_scene_threshold_var, orient="horizontal", length=400)
        scene_scale.grid(row=2, column=1, sticky="ew", padx=(15, 15))
        self.news_scene_label = ttk.Label(news_params_section, text="0.3", font=('Arial', 16, 'bold'))
        self.news_scene_label.grid(row=2, column=2)
        
        # Minimum gap between transitions
        ttk.Label(news_params_section, text="Min Gap (seconds):", font=('Arial', 16)).grid(row=3, column=0, sticky="w", pady=(5, 5))
        gap_scale = ttk.Scale(news_params_section, from_=0.5, to=10.0, 
                             variable=self.news_min_gap_var, orient="horizontal", length=400)
        gap_scale.grid(row=3, column=1, sticky="ew", padx=(15, 15))
        self.news_gap_label = ttk.Label(news_params_section, text="2.0", font=('Arial', 16, 'bold'))
        self.news_gap_label.grid(row=3, column=2)
        
        # Bind scale events to update labels
        headline_scale.configure(command=lambda v: self.news_headline_label.configure(text=f"{float(v):.2f}"))
        person_scale.configure(command=lambda v: self.news_person_label.configure(text=f"{float(v):.2f}"))
        scene_scale.configure(command=lambda v: self.news_scene_label.configure(text=f"{float(v):.2f}"))
        gap_scale.configure(command=lambda v: self.news_gap_label.configure(text=f"{float(v):.1f}"))
        
        # Processing section
        news_process_section = ttk.LabelFrame(news_frame, text="Process News Video", padding="25")
        news_process_section.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 30))
        
        # Process button
        self.news_process_btn = ttk.Button(news_process_section, text="🎬 Detect News Transitions", 
                                          command=self.start_news_processing, style='Process.TButton')
        self.news_process_btn.pack(pady=(10, 10))
        
        # Progress bar for news processing
        self.news_progress_var = tk.DoubleVar()
        self.news_progress_bar = ttk.Progressbar(news_process_section, variable=self.news_progress_var, 
                                                style='Large.Horizontal.TProgressbar', length=600)
        self.news_progress_bar.pack(pady=(10, 10), fill="x")
        
        # News results section
        news_results_section = ttk.LabelFrame(news_frame, text="Detection Results", padding="25")
        news_results_section.grid(row=4, column=0, columnspan=2, sticky="ew")
        news_results_section.columnconfigure(0, weight=1)
        
        # Results text area
        self.news_results_text = scrolledtext.ScrolledText(news_results_section, height=10, width=80, 
                                                          font=('Courier', 12), wrap=tk.WORD)
        self.news_results_text.grid(row=0, column=0, sticky="nsew", pady=(10, 10))
        news_results_section.rowconfigure(0, weight=1)
        
        # Button frame for news results
        news_button_frame = ttk.Frame(news_results_section)
        news_button_frame.grid(row=1, column=0, pady=(0, 10))
        
        # Open results folder button
        self.news_open_folder_btn = ttk.Button(news_button_frame, text="📁 Open Results Folder", 
                                              command=self.open_news_results_folder, style='Large.TButton', state='disabled')
        self.news_open_folder_btn.pack(side="left", padx=(0, 15))
        
        # Export frames button
        self.news_export_btn = ttk.Button(news_button_frame, text="💾 Export Frames", 
                                         command=self.export_news_frames, style='Large.TButton', state='disabled')
        self.news_export_btn.pack(side="left")
    
    def create_results_tab(self):
        """Create the results display tab"""
        results_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(results_frame, text="📊 Results")
        
        # Title
        ttk.Label(results_frame, text="Processing Results", 
                 style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Create paned window for results
        paned = ttk.PanedWindow(results_frame, orient="horizontal")
        paned.grid(row=1, column=0, columnspan=2, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Left panel - Statistics and info
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)
        
        # Statistics section
        stats_frame = ttk.LabelFrame(left_panel, text="Processing Statistics", padding="20")  # Increased padding
        stats_frame.pack(fill="both", expand=False, pady=(0, 15))
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=12, width=50, font=('Consolas', 12))  # Larger text area
        self.stats_text.pack(fill="both", expand=True)
        
        # Plot section
        plot_frame = ttk.LabelFrame(left_panel, text="Algorithm Performance", padding="20")
        plot_frame.pack(fill="both", expand=True)
        
        # Matplotlib figure - larger
        self.fig, self.ax = plt.subplots(figsize=(9, 6))  # Increased from (6, 4)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Right panel - Keyframes gallery
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=2)
        
        gallery_frame = ttk.LabelFrame(right_panel, text="Keyframes Gallery", padding="20")  # Increased padding
        gallery_frame.pack(fill="both", expand=True)
        
        # Create scrollable gallery
        gallery_canvas = tk.Canvas(gallery_frame)
        gallery_scrollbar = ttk.Scrollbar(gallery_frame, orient="vertical", command=gallery_canvas.yview)
        self.gallery_frame = ttk.Frame(gallery_canvas)
        
        self.gallery_frame.bind(
            "<Configure>",
            lambda e: gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all"))
        )
        
        gallery_canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        gallery_canvas.configure(yscrollcommand=gallery_scrollbar.set)
        
        gallery_canvas.pack(side="left", fill="both", expand=True)
        gallery_scrollbar.pack(side="right", fill="y")
        
        # Control buttons for results - larger buttons
        control_frame = ttk.Frame(results_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=(15, 0))
        
        ttk.Button(control_frame, text="💾 Save Results", 
                  command=self.save_results, style='Large.TButton').pack(side="left", padx=(0, 20))
        ttk.Button(control_frame, text="📁 Export Keyframes", 
                  command=self.export_keyframes, style='Large.TButton').pack(side="left", padx=(0, 20))
        ttk.Button(control_frame, text="📋 Copy Statistics", 
                  command=self.copy_statistics, style='Large.TButton').pack(side="left")
    
    def create_logs_tab(self):
        """Create the logs display tab"""
        logs_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(logs_frame, text="📝 Logs")
        
        # Title
        ttk.Label(logs_frame, text="Processing Logs", 
                 style='Title.TLabel').grid(row=0, column=0, pady=(0, 20))
        
        # Log display - larger text area
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=35, width=120, font=('Consolas', 12))
        self.log_text.grid(row=1, column=0, sticky="nsew")
        logs_frame.columnconfigure(0, weight=1)
        logs_frame.rowconfigure(1, weight=1)
        
        # Log controls - larger buttons
        log_controls = ttk.Frame(logs_frame)
        log_controls.grid(row=2, column=0, pady=(15, 0))
        
        ttk.Button(log_controls, text="🗑️ Clear Logs", 
                  command=self.clear_logs, style='Large.TButton').pack(side="left", padx=(0, 20))
        ttk.Button(log_controls, text="💾 Save Logs", 
                  command=self.save_logs, style='Large.TButton').pack(side="left")
    
    def create_status_bar(self, parent):
        """Create the status bar at the bottom"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=1, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        
        # Status label - larger font
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief="sunken", 
                                font=('Arial', 14), padding=(10, 8))  # Larger font and padding
        status_label.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        # Fan status - shows if fan control is active
        self.fan_status_var = tk.StringVar(value="🌀 Fan: Ready")
        self.fan_label = ttk.Label(status_frame, textvariable=self.fan_status_var, relief="sunken",
                                  font=('Arial', 14), padding=(10, 8))
        self.fan_label.grid(row=0, column=1, padx=(0, 10))
        
        # Version info - larger font
        version_label = ttk.Label(status_frame, text="FrameSift Lite v1.0", relief="sunken",
                                 font=('Arial', 14), padding=(10, 8))
        version_label.grid(row=0, column=2)
    
    def setup_logging_display(self):
        """Set up logging to display in the GUI"""
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                msg = self.format(record)
                def append():
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.see(tk.END)
                self.text_widget.after(0, append)
        
        # Add GUI handler to logger
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(gui_handler)
    
    # Event handlers and utility methods
    def update_frame_diff_label(self, value):
        """Update frame difference label"""
        self.frame_diff_label.config(text=f"{float(value):.2f}")
    
    def update_histogram_label(self, value):
        """Update histogram label"""
        self.histogram_label.config(text=f"{float(value):.2f}")
    
    def update_blur_label(self, value):
        """Update blur label"""
        self.blur_label.config(text=f"{float(value):.0f}")
    
    def update_reduction_label(self, value):
        """Update reduction label"""
        self.reduction_label.config(text=f"{float(value)*100:.0f}%")
    
    def browse_video(self):
        """Open file dialog to select video"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if filename:
            self.video_path_var.set(filename)
            self.current_video_path = filename
            self.load_video_info(filename)
            self.process_btn.config(state="normal")
            logger.info(f"Selected video: {filename}")
    
    def load_video_info(self, video_path):
        """Load and display video information"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.video_info_var.set("❌ Cannot read video file")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            
            info_text = (f"✅ {width}x{height}, {fps:.1f} FPS, "
                        f"{frame_count} frames, {duration:.1f}s, {file_size:.1f}MB")
            
            self.video_info_var.set(info_text)
            cap.release()
            
        except Exception as e:
            self.video_info_var.set(f"❌ Error reading video: {str(e)}")
            logger.error(f"Error loading video info: {e}")
    
    def load_conservative_preset(self):
        """Load conservative processing parameters"""
        self.frame_diff_var.set(0.5)
        self.histogram_var.set(0.7)
        self.blur_var.set(30.0)
        self.target_reduction_var.set(0.6)
        logger.info("Loaded conservative preset")
    
    def load_balanced_preset(self):
        """Load balanced processing parameters"""
        self.frame_diff_var.set(0.3)
        self.histogram_var.set(0.5)
        self.blur_var.set(50.0)
        self.target_reduction_var.set(0.75)
        logger.info("Loaded balanced preset")
    
    def load_aggressive_preset(self):
        """Load aggressive processing parameters"""
        self.frame_diff_var.set(0.1)
        self.histogram_var.set(0.3)
        self.blur_var.set(80.0)
        self.target_reduction_var.set(0.85)
        logger.info("Loaded aggressive preset")
    
    # News Detection Methods
    def browse_news_video(self):
        """Open file dialog to select news video"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select News Video File",
            filetypes=filetypes
        )
        
        if filename:
            self.news_video_path_var.set(filename)
            self.current_news_video_path = filename
            self.load_news_video_info(filename)
            self.news_process_btn.config(state="normal")
            logger.info(f"Selected news video: {filename}")
    
    def load_news_video_info(self, video_path):
        """Load and display news video information"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.news_video_info_var.set("❌ Cannot read video file")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            
            info_text = (f"✅ {width}x{height}, {fps:.1f} FPS, "
                        f"{frame_count} frames, {duration:.1f}s, {file_size:.1f}MB")
            
            self.news_video_info_var.set(info_text)
            cap.release()
            
        except Exception as e:
            self.news_video_info_var.set(f"❌ Error reading video: {str(e)}")
            logger.error(f"Error loading news video info: {e}")
    
    def start_news_processing(self):
        """Start news video processing in a separate thread"""
        if not hasattr(self, 'current_news_video_path') or not self.current_news_video_path:
            messagebox.showerror("Error", "Please select a news video file first")
            return
        
        # Disable controls
        self.news_process_btn.config(state="disabled")
        self.news_progress_var.set(0)
        
        # Clear previous results
        self.news_results_text.delete(1.0, tk.END)
        self.news_results_text.insert(tk.END, "Starting news content detection...\n")
        
        # Generate session ID for news processing
        self.news_session_id = str(uuid.uuid4())
        
        # Start processing thread
        self.news_processing_thread = threading.Thread(target=self.process_news_video_thread)
        self.news_processing_thread.daemon = True
        self.news_processing_thread.start()
        
        logger.info(f"Started news processing thread for session: {self.news_session_id}")
    
    def process_news_video_thread(self):
        """Process news video in background thread"""
        try:
            # Start fan control for news processing
            start_processing_fans()
            
            self.update_news_status("� Starting fan control and news content detector...")
            self.update_news_progress(5)
            
            # Initialize news detector with current parameters
            detector = NewsContentDetector(
                headline_threshold=self.news_headline_threshold_var.get(),
                person_threshold=self.news_person_threshold_var.get(),
                scene_threshold=self.news_scene_threshold_var.get(),
                min_gap=self.news_min_gap_var.get()
            )
            
            # Initialize keyframe extractor
            extractor = NewsKeyframeExtractor(detector)
            
            self.update_news_status("Analyzing video for news transitions...")
            self.update_news_progress(10)
            
            # Create output directory
            output_dir = os.path.join("gui_results", f"news_{self.news_session_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Process news video with progress callback
            def progress_callback(progress):
                self.update_news_progress(10 + (progress * 0.9))  # Scale to 10-100%
            
            keyframe_paths, text_results = extractor.extract_keyframes(
                video_path=self.current_news_video_path,
                output_dir=output_dir,
                progress_callback=progress_callback
            )
            
            self.news_output_dir = output_dir
            
            # Update results display
            self.root.after(0, lambda: self.news_processing_complete(keyframe_paths, text_results))
            
        except Exception as e:
            error_msg = f"News processing error: {str(e)}"
            logger.error(f"News processing error: {e}")
            logger.error(traceback.format_exc())
            self.root.after(0, lambda: self.news_processing_error(error_msg))
        
        finally:
            # Always stop fan control when news processing ends
            stop_processing_fans()
    
    def update_news_status(self, message):
        """Update news processing status"""
        self.root.after(0, lambda: self.news_results_text.insert(tk.END, f"{message}\n"))
        self.root.after(0, lambda: self.news_results_text.see(tk.END))
    
    def update_news_progress(self, value):
        """Update news processing progress bar"""
        self.root.after(0, lambda: self.news_progress_var.set(value))
    
    def news_processing_complete(self, keyframe_paths, text_results=None):
        """Handle completion of news processing with text extraction"""
        self.news_process_btn.config(state="normal")
        self.news_open_folder_btn.config(state="normal")
        self.news_export_btn.config(state="normal")
        self.news_progress_var.set(100)
        
        # Store keyframe paths and text results for export
        self.news_keyframe_paths = keyframe_paths
        self.news_text_results = text_results or []
        
        # Display results
        results_text = f"\n✅ News Content Detection Complete!\n"
        results_text += f"📊 Extracted {len(keyframe_paths)} transition keyframes\n"
        
        if text_results:
            total_text_regions = sum(len(result['text_regions']) for result in text_results)
            results_text += f"📝 Extracted text from {len(text_results)} keyframes ({total_text_regions} text regions)\n"
        
        results_text += "\n"
        
        if keyframe_paths:
            results_text += "🎬 Detected Transitions:\n"
            for i, path in enumerate(keyframe_paths):
                filename = os.path.basename(path)
                # Parse filename to get transition info
                parts = filename.replace('.jpg', '').split('_')
                if len(parts) >= 7:
                    frame_num = parts[2]
                    transition_type = f"{parts[3]}_{parts[4]}"  # e.g., "scene_change"
                    confidence = f"{parts[5]}.{parts[6]}"
                    results_text += f"  • Frame {frame_num}: {transition_type.replace('_', ' ').title()} (confidence: {confidence})"
                    
                    # Add text preview if available
                    if text_results and i < len(text_results):
                        text_result = text_results[i]
                        if text_result['text_regions']:
                            main_texts = [r['text'] for r in text_result['text_regions'] if len(r['text']) > 3]
                            if main_texts:
                                preview = ', '.join(main_texts[:2])  # Show first 2 text regions
                                if len(preview) > 50:
                                    preview = preview[:47] + "..."
                                results_text += f" | Text: {preview}"
                    results_text += "\n"
            
            results_text += f"\n📁 Results saved to: {self.news_output_dir}\n"
            if text_results:
                results_text += f"📄 Text extraction results saved to: extracted_text.json\n"
        else:
            results_text += "ℹ️ No significant news transitions detected.\n"
            results_text += "Try adjusting the threshold parameters for more sensitive detection.\n"
        
        self.news_results_text.insert(tk.END, results_text)
        self.news_results_text.see(tk.END)
        
        logger.info(f"News processing completed successfully. {len(keyframe_paths)} keyframes extracted with text from {len(text_results or [])} frames.")
    
    def news_processing_error(self, error_msg):
        """Handle news processing error"""
        self.news_process_btn.config(state="normal")
        self.news_export_btn.config(state="disabled")
        self.news_progress_var.set(0)
        
        error_text = f"\n❌ News Processing Failed!\n{error_msg}\n"
        self.news_results_text.insert(tk.END, error_text)
        self.news_results_text.see(tk.END)
        
        messagebox.showerror("Processing Error", error_msg)
    
    def open_news_results_folder(self):
        """Open the news results folder in file explorer"""
        if hasattr(self, 'news_output_dir') and os.path.exists(self.news_output_dir):
            try:
                if sys.platform == "win32":
                    os.startfile(self.news_output_dir)
                elif sys.platform == "darwin":
                    os.system(f"open '{self.news_output_dir}'")
                else:
                    os.system(f"xdg-open '{self.news_output_dir}'")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open folder: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No results folder found. Please run news detection first.")
    
    def export_news_frames(self):
        """Export news transition frames to a user-selected folder"""
        if not hasattr(self, 'news_keyframe_paths') or not self.news_keyframe_paths:
            messagebox.showwarning("Warning", "No news keyframes to export. Please run news detection first.")
            return
        
        # Ask user to select export directory
        export_dir = filedialog.askdirectory(
            title="Select Folder to Export News Transition Frames",
            initialdir=os.path.expanduser("~")
        )
        
        if not export_dir:
            return  # User cancelled
        
        try:
            # Create subdirectory with timestamp for this export
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_subdir = os.path.join(export_dir, f"news_transitions_{timestamp}")
            os.makedirs(export_subdir, exist_ok=True)
            
            # Copy files with progress tracking
            total_files = len(self.news_keyframe_paths)
            copied_files = []
            
            for i, source_path in enumerate(self.news_keyframe_paths):
                if os.path.exists(source_path):
                    filename = os.path.basename(source_path)
                    dest_path = os.path.join(export_subdir, filename)
                    
                    # Copy file
                    shutil.copy2(source_path, dest_path)
                    copied_files.append(dest_path)
                    
                    # Update progress in status
                    progress = ((i + 1) / total_files) * 100
                    self.news_results_text.insert(tk.END, f"Exporting... {progress:.0f}%\r")
                    self.news_results_text.see(tk.END)
                    self.root.update_idletasks()
            
            # Clear the progress line
            self.news_results_text.insert(tk.END, "\n")
            
            # Create export summary file
            summary_path = os.path.join(export_subdir, "export_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("News Transition Frames Export Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source Video: {os.path.basename(self.current_news_video_path) if hasattr(self, 'current_news_video_path') else 'Unknown'}\n")
                f.write(f"Total Frames Exported: {len(copied_files)}\n\n")
                
                f.write("Detection Parameters Used:\n")
                f.write(f"  • Headline Threshold: {self.news_headline_threshold_var.get():.2f}\n")
                f.write(f"  • Person Threshold: {self.news_person_threshold_var.get():.2f}\n")
                f.write(f"  • Scene Threshold: {self.news_scene_threshold_var.get():.2f}\n")
                f.write(f"  • Minimum Gap: {self.news_min_gap_var.get():.1f} seconds\n\n")
                
                f.write("Exported Files:\n")
                for i, file_path in enumerate(copied_files, 1):
                    filename = os.path.basename(file_path)
                    # Parse filename for transition info
                    parts = filename.replace('.jpg', '').split('_')
                    if len(parts) >= 7:
                        frame_num = parts[2]
                        transition_type = f"{parts[3]}_{parts[4]}"
                        confidence = f"{parts[5]}.{parts[6]}"
                        f.write(f"  {i:2d}. {filename}\n")
                        f.write(f"      Frame: {frame_num}, Type: {transition_type.replace('_', ' ').title()}, Confidence: {confidence}\n")
                    else:
                        f.write(f"  {i:2d}. {filename}\n")
            
            # Show success message
            success_msg = f"✅ Export Complete!\n\n"
            success_msg += f"📊 Exported {len(copied_files)} transition frames\n"
            success_msg += f"📁 Location: {export_subdir}\n"
            success_msg += f"📄 Summary: export_summary.txt\n\n"
            
            self.news_results_text.insert(tk.END, success_msg)
            self.news_results_text.see(tk.END)
            
            # Ask if user wants to open the export folder
            if messagebox.askyesno("Export Complete", 
                                 f"Successfully exported {len(copied_files)} frames to:\n{export_subdir}\n\nWould you like to open the export folder?"):
                try:
                    if sys.platform == "win32":
                        os.startfile(export_subdir)
                    elif sys.platform == "darwin":
                        os.system(f"open '{export_subdir}'")
                    else:
                        os.system(f"xdg-open '{export_subdir}'")
                except Exception as e:
                    messagebox.showwarning("Warning", f"Could not open folder: {str(e)}")
            
            logger.info(f"Successfully exported {len(copied_files)} news frames to {export_subdir}")
            
        except Exception as e:
            error_msg = f"Failed to export frames: {str(e)}"
            logger.error(f"Export error: {e}")
            messagebox.showerror("Export Error", error_msg)
            self.news_results_text.insert(tk.END, f"❌ Export Failed: {error_msg}\n")
            self.news_results_text.see(tk.END)
    

    
    def update_fan_status(self):
        """Update fan status display in status bar"""
        try:
            if is_fan_control_active():
                status_text = "🌀 Fan: HIGH SPEED"
            else:
                status_text = "� Fan: Normal"
            
            self.fan_status_var.set(status_text)
            
        except Exception as e:
            logger.debug(f"Error updating fan status: {e}")
            self.fan_status_var.set("� Fan: N/A")
        
        # Schedule next update
        self.root.after(2000, self.update_fan_status)  # Update every 2 seconds
    
    def start_processing(self):
        """Start video processing in a separate thread"""
        if not self.current_video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return
        
        # Disable controls
        self.process_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress_var.set(0)
        
        # Generate session ID
        self.session_id = str(uuid.uuid4())
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_video_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info(f"Started processing thread for session: {self.session_id}")
    
    def process_video_thread(self):
        """Process video in background thread"""
        try:
            # Start fan control
            start_processing_fans()
            
            self.update_status("� Starting fan control and video processor...")
            self.update_progress(5)
            
            # Initialize processor with current parameters
            processor = VideoProcessor(
                frame_diff_threshold=self.frame_diff_var.get(),
                histogram_threshold=self.histogram_var.get(),
                blur_threshold=self.blur_var.get()
            )
            
            # Initialize results generator
            results_generator = ResultsGenerator(output_dir="gui_results")
            
            self.update_status("Processing video...")
            self.update_progress(10)
            
            # Process video
            processing_results = processor.process_video_complete(
                video_path=self.current_video_path,
                target_reduction=self.target_reduction_var.get(),
                max_frames=None,
                sample_rate=1
            )
            
            self.update_progress(60)
            self.update_status("Generating thumbnails...")
            
            # Generate thumbnails
            keyframe_data = results_generator.create_keyframe_thumbnails(
                frames=processing_results['frames'],
                keyframe_metrics=processing_results['keyframes'],
                session_id=self.session_id
            )
            
            self.update_progress(80)
            self.update_status("Calculating statistics...")
            
            # Calculate statistics
            processing_stats = results_generator.calculate_processing_statistics(
                all_metrics=processing_results['all_metrics'],
                processing_start_time=time.time() - 30,  # Approximate
                processing_end_time=time.time(),
                video_properties=processing_results['video_properties']
            )
            
            self.update_progress(90)
            self.update_status("Generating plots...")
            
            # Generate plots
            plot_path, plot_data = results_generator.generate_processing_plots(
                all_metrics=processing_results['all_metrics'],
                session_id=self.session_id,
                plot_title=f"Processing Results - {os.path.basename(self.current_video_path)}"
            )
            
            self.update_progress(100)
            self.update_status("Processing completed successfully!")
            
            # Store results
            self.processing_results = {
                'keyframes': keyframe_data,
                'statistics': processing_stats,
                'plot_path': plot_path,
                'plot_data': plot_data,
                'raw_results': processing_results
            }
            
            # Update GUI with results
            self.root.after(0, self.display_results)
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"Processing error: {e}")
            logger.error(traceback.format_exc())
            
            self.root.after(0, lambda: self.processing_error(error_msg))
        
        finally:
            # Always stop fan control when processing ends
            stop_processing_fans()
    
    def update_status(self, message):
        """Update status message thread-safely"""
        self.root.after(0, lambda: self.status_var.set(message))
    
    def update_progress(self, value):
        """Update progress bar thread-safely"""
        self.root.after(0, lambda: self.progress_var.set(value))
    
    def processing_error(self, error_msg):
        """Handle processing error"""
        messagebox.showerror("Processing Error", error_msg)
        self.process_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Processing failed")
    
    def cancel_processing(self):
        """Cancel processing (if possible)"""
        if self.processing_thread and self.processing_thread.is_alive():
            # Note: Python threads can't be forcefully stopped
            # This is more of a UI state reset
            messagebox.showinfo("Cancel", "Processing cannot be stopped immediately. Please wait for current operation to complete.")
        
        self.process_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.status_var.set("Processing cancelled")
    
    def reset_session(self):
        """Reset the current session"""
        self.current_video_path = None
        self.processing_results = None
        self.session_id = None
        self.video_path_var.set("")
        self.video_info_var.set("No video loaded")
        self.progress_var.set(0)
        self.status_var.set("Ready to process video")
        self.process_btn.config(state="disabled")
        
        # Clear results
        self.stats_text.delete(1.0, tk.END)
        self.ax.clear()
        self.canvas.draw()
        
        # Clear gallery
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        
        logger.info("Session reset")
    
    def display_results(self):
        """Display processing results in the GUI"""
        if not self.processing_results:
            return
        
        # Switch to results tab
        self.notebook.select(2)
        
        # Display statistics
        self.display_statistics()
        
        # Display plot
        self.display_plot()
        
        # Display keyframes gallery
        self.display_keyframes_gallery()
        
        # Re-enable controls
        self.process_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        
        logger.info("Results displayed successfully")
    
    def display_statistics(self):
        """Display processing statistics"""
        stats = self.processing_results['statistics']
        
        stats_text = f"""PROCESSING STATISTICS
{'='*50}

Video Information:
• File: {os.path.basename(self.current_video_path)}
• Duration: {stats.get('duration', 'N/A')} seconds
• Original Frames: {stats.get('total_frames', 'N/A')}
• Resolution: {stats.get('width', 'N/A')}x{stats.get('height', 'N/A')}
• Frame Rate: {stats.get('fps', 'N/A')} FPS

Processing Results:
• Keyframes Selected: {stats.get('keyframes_selected', 'N/A')}
• Reduction Achieved: {stats.get('reduction_percentage', 'N/A')}%
• Processing Time: {stats.get('processing_time', 'N/A')} seconds

Algorithm Performance:
• Frame Difference Threshold: {self.frame_diff_var.get():.2f}
• Histogram Threshold: {self.histogram_var.get():.2f}
• Blur Threshold: {self.blur_var.get():.1f}
• Target Reduction: {self.target_reduction_var.get()*100:.0f}%

Quality Metrics:
• Sharp Frames: {stats.get('sharp_frames', 'N/A')}
• Blurry Frames Filtered: {stats.get('blurry_frames', 'N/A')}
• Scene Changes Detected: {stats.get('scene_changes', 'N/A')}
"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def display_plot(self):
        """Display algorithm performance plot"""
        if not self.processing_results.get('raw_results'):
            return
        
        all_metrics = self.processing_results['raw_results']['all_metrics']
        
        # Clear previous plot
        self.ax.clear()
        
        # Extract data for plotting
        frame_numbers = [m.frame_number for m in all_metrics]
        diff_scores = [m.difference_score for m in all_metrics]
        hist_scores = [m.histogram_score for m in all_metrics]
        blur_scores = [m.blur_score for m in all_metrics]
        keyframes = [i for i, m in enumerate(all_metrics) if m.is_keyframe]
        
        # Plot algorithm scores
        self.ax.plot(frame_numbers, diff_scores, 'b-', alpha=0.7, label='Frame Difference')
        self.ax.plot(frame_numbers, hist_scores, 'g-', alpha=0.7, label='Histogram')
        
        # Normalize blur scores for display
        if blur_scores:
            max_blur = max(blur_scores)
            normalized_blur = [score / max_blur for score in blur_scores]
            self.ax.plot(frame_numbers, normalized_blur, 'r-', alpha=0.7, label='Blur (normalized)')
        
        # Mark keyframes
        if keyframes:
            keyframe_numbers = [frame_numbers[i] for i in keyframes]
            keyframe_diff = [diff_scores[i] for i in keyframes]
            self.ax.plot(keyframe_numbers, keyframe_diff, 'ro', markersize=8, 
                        label='Selected Keyframes')
        
        # Add thresholds
        self.ax.axhline(y=self.frame_diff_var.get(), color='b', linestyle='--', alpha=0.5)
        self.ax.axhline(y=self.histogram_var.get(), color='g', linestyle='--', alpha=0.5)
        
        self.ax.set_xlabel('Frame Number')
        self.ax.set_ylabel('Algorithm Score')
        self.ax.set_title('IVP Algorithm Performance')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def display_keyframes_gallery(self):
        """Display keyframes in the gallery"""
        # Clear existing gallery
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        
        keyframes = self.processing_results.get('keyframes', [])
        
        if not keyframes:
            ttk.Label(self.gallery_frame, text="No keyframes to display", 
                     style='Info.TLabel').pack(pady=20)
            return
        
        # Display keyframes in a grid - larger thumbnails
        cols = 2  # Reduced columns for larger display
        for i, keyframe in enumerate(keyframes):
            row = i // cols
            col = i % cols
            
            # Create frame for keyframe
            kf_frame = ttk.Frame(self.gallery_frame, relief="raised", borderwidth=2)
            kf_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")  # Increased spacing
            
            # Load and display thumbnail - much larger
            try:
                thumbnail_path = keyframe.get('thumbnail_path')
                if thumbnail_path and os.path.exists(thumbnail_path):
                    # Load image - larger thumbnails
                    img = Image.open(thumbnail_path)
                    img.thumbnail((300, 225), Image.Resampling.LANCZOS)  # 150% larger (200x150 -> 300x225)
                    photo = ImageTk.PhotoImage(img)
                    
                    # Display image
                    img_label = ttk.Label(kf_frame, image=photo)
                    img_label.image = photo  # Keep reference
                    img_label.pack(pady=(10, 5))
                    
                    # Add frame info - larger text
                    info_text = f"Frame {keyframe.get('frame_number', 'N/A')}\nTime: {keyframe.get('timestamp', 0.0):.2f}s"
                    ttk.Label(kf_frame, text=info_text, style='Info.TLabel').pack(pady=(0, 10))
                
                else:
                    ttk.Label(kf_frame, text=f"Frame {keyframe.get('frame_number', 'N/A')}\n(No preview)", 
                             style='Info.TLabel').pack(pady=20)
            
            except Exception as e:
                logger.error(f"Error displaying keyframe {i}: {e}")
                ttk.Label(kf_frame, text=f"Frame {i}\n(Error loading)", 
                         style='Error.TLabel').pack(pady=20)
    
    def save_results(self):
        """Save processing results to file"""
        if not self.processing_results:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Prepare results data
                results_data = {
                    'session_id': self.session_id,
                    'video_path': self.current_video_path,
                    'parameters': {
                        'frame_diff_threshold': self.frame_diff_var.get(),
                        'histogram_threshold': self.histogram_var.get(),
                        'blur_threshold': self.blur_var.get(),
                        'target_reduction': self.target_reduction_var.get()
                    },
                    'statistics': self.processing_results['statistics'],
                    'keyframes_count': len(self.processing_results.get('keyframes', [])),
                    'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(filename, 'w') as f:
                    json.dump(results_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to {filename}")
                logger.info(f"Results saved to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
                logger.error(f"Error saving results: {e}")
    
    def export_keyframes(self):
        """Export keyframe images to a folder"""
        if not self.processing_results:
            messagebox.showwarning("Warning", "No keyframes to export")
            return
        
        folder = filedialog.askdirectory(title="Select Export Folder")
        if not folder:
            return
        
        try:
            keyframes = self.processing_results.get('keyframes', [])
            exported_count = 0
            
            for i, keyframe in enumerate(keyframes):
                thumbnail_path = keyframe.get('thumbnail_path')
                if thumbnail_path and os.path.exists(thumbnail_path):
                    # Copy thumbnail to export folder
                    export_filename = f"keyframe_{i+1:03d}_frame_{keyframe.get('frame_number', 0):06d}.jpg"
                    export_path = os.path.join(folder, export_filename)
                    shutil.copy2(thumbnail_path, export_path)
                    exported_count += 1
            
            messagebox.showinfo("Success", f"Exported {exported_count} keyframes to {folder}")
            logger.info(f"Exported {exported_count} keyframes to {folder}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export keyframes: {str(e)}")
            logger.error(f"Error exporting keyframes: {e}")
    
    def copy_statistics(self):
        """Copy statistics to clipboard"""
        stats_text = self.stats_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(stats_text)
        messagebox.showinfo("Success", "Statistics copied to clipboard")
    
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)
    
    def save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                logs_content = self.log_text.get(1.0, tk.END)
                with open(filename, 'w') as f:
                    f.write(logs_content)
                messagebox.showinfo("Success", f"Logs saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {str(e)}")


def main():
    """Main application entry point"""
    try:
        # Create main window
        root = tk.Tk()
        
        # Create application
        app = FrameSiftGUI(root)
        
        # Configure window close handler
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit FrameSift Lite?"):
                logger.info("Application closing")
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start main loop
        logger.info("Starting FrameSift Lite GUI application")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error(traceback.format_exc())
        messagebox.showerror("Startup Error", f"Failed to start FrameSift Lite:\n{str(e)}")


if __name__ == "__main__":
    main()