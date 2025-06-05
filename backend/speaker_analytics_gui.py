#!/usr/bin/env python3
"""
Oreja Speaker Analytics Dashboard
A comprehensive GUI for exploring and visualizing the speaker database and learning progress.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import requests
from pathlib import Path
import pandas as pd

# Import enhanced transcription processor for conversation analysis
from enhanced_transcription_processor import EnhancedTranscriptionProcessor

# Import enhanced speaker database
from speaker_database_v2 import EnhancedSpeakerDatabase

class SpeakerAnalyticsDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Oreja Speaker Analytics Dashboard")
        self.root.geometry("1400x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize enhanced speaker database
        self.speaker_db = EnhancedSpeakerDatabase()
        
        # Legacy database path for compatibility
        self.db_path = Path("speaker_data_v2/speaker_records.json")
        
        # Backend URL
        self.backend_url = "http://127.0.0.1:8000"
        
        # Auto-refresh settings
        self.auto_refresh = tk.BooleanVar(value=True)
        self.refresh_interval = 5  # seconds
        
        # Data storage
        self.speaker_data = {}
        self.selected_speaker = None
        self.selected_speakers = set()  # For management tab multi-select
        
        # Initialize enhanced transcription processor
        try:
            self.enhanced_processor = EnhancedTranscriptionProcessor(sentiment_model="vader")
        except Exception as e:
            print(f"Warning: Could not initialize enhanced processor: {e}")
            self.enhanced_processor = None
        
        self.setup_ui()
        self.setup_auto_refresh()
        self.refresh_data()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Overview Dashboard
        self.setup_overview_tab()
        
        # Tab 2: Speaker Details
        self.setup_details_tab()
        
        # Tab 3: Learning Analytics
        self.setup_analytics_tab()
        
        # Tab 4: Conversation Analysis (NEW)
        self.setup_conversation_analysis_tab()
        
        # Tab 5: Batch Processing
        self.setup_batch_tab()
        
        # Tab 6: Database Management
        self.setup_management_tab()
        
        # Tab 7: User Training (NEW)
        self.setup_user_training_tab()
        
        # Status bar
        self.setup_status_bar()
        
    def setup_overview_tab(self):
        """Setup the overview dashboard tab"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="📊 Overview Dashboard")
        
        # Top controls
        controls_frame = ttk.Frame(overview_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="🎤 Speaker Database Overview", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT)
        
        # Auto-refresh checkbox
        ttk.Checkbutton(controls_frame, text="Auto-refresh", 
                       variable=self.auto_refresh).pack(side=tk.RIGHT, padx=10)
        
        # Refresh button
        ttk.Button(controls_frame, text="🔄 Refresh", 
                  command=self.refresh_data).pack(side=tk.RIGHT, padx=5)
        
        # Main content area with paned window
        paned = ttk.PanedWindow(overview_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Speaker list
        left_frame = ttk.LabelFrame(paned, text="Speakers", padding=10)
        paned.add(left_frame, weight=1)
        
        # Speaker list with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for speaker list
        self.speaker_tree = ttk.Treeview(list_frame, columns=('type', 'samples', 'confidence', 'last_seen'), 
                                        show='tree headings', height=15)
        
        # Configure columns
        self.speaker_tree.heading('#0', text='Speaker Name')
        self.speaker_tree.heading('type', text='Type')
        self.speaker_tree.heading('samples', text='Samples')
        self.speaker_tree.heading('confidence', text='Confidence')
        self.speaker_tree.heading('last_seen', text='Last Seen')
        
        self.speaker_tree.column('#0', width=200)
        self.speaker_tree.column('type', width=100)
        self.speaker_tree.column('samples', width=80)
        self.speaker_tree.column('confidence', width=100)
        self.speaker_tree.column('last_seen', width=120)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.speaker_tree.yview)
        self.speaker_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.speaker_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.speaker_tree.bind('<<TreeviewSelect>>', self.on_speaker_select)
        
        # Right panel: Statistics
        right_frame = ttk.LabelFrame(paned, text="Statistics", padding=10)
        paned.add(right_frame, weight=1)
        
        # Statistics display
        self.stats_text = tk.Text(right_frame, height=20, width=40, font=('Consolas', 10))
        stats_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_details_tab(self):
        """Setup the speaker details tab"""
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="🔍 Speaker Details")
        
        # Top info panel
        info_frame = ttk.LabelFrame(details_frame, text="Speaker Information", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.details_info = tk.Text(info_frame, height=8, font=('Consolas', 10))
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.details_info.yview)
        self.details_info.configure(yscrollcommand=info_scroll.set)
        
        self.details_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Charts frame
        charts_frame = ttk.LabelFrame(details_frame, text="Learning Progress Charts", padding=10)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure for charts
        self.fig = Figure(figsize=(12, 6), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_analytics_tab(self):
        """Setup the learning analytics tab"""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="📈 Learning Analytics")
        
        # Controls
        controls_frame = ttk.Frame(analytics_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Analytics Time Range:").pack(side=tk.LEFT, padx=5)
        
        self.time_range = tk.StringVar(value="24h")
        time_combo = ttk.Combobox(controls_frame, textvariable=self.time_range, 
                                 values=["1h", "6h", "24h", "7d", "30d", "All"])
        time_combo.pack(side=tk.LEFT, padx=5)
        time_combo.bind('<<ComboboxSelected>>', lambda e: self.update_analytics())
        
        ttk.Button(controls_frame, text="Export Data", 
                  command=self.export_analytics).pack(side=tk.RIGHT, padx=5)
        
        # Analytics charts
        analytics_charts = ttk.LabelFrame(analytics_frame, text="System Learning Analytics", padding=10)
        analytics_charts.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create analytics figure
        self.analytics_fig = Figure(figsize=(14, 8), facecolor='white')
        self.analytics_canvas = FigureCanvasTkAgg(self.analytics_fig, analytics_charts)
        self.analytics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_conversation_analysis_tab(self):
        """Setup the conversation analysis tab with summarization and annotation features"""
        conv_frame = ttk.Frame(self.notebook)
        self.notebook.add(conv_frame, text="🗣️ Conversation Analysis")
        
        # Top section: File selection and controls
        controls_frame = ttk.LabelFrame(conv_frame, text="Transcription Analysis", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File selection
        file_frame = ttk.Frame(controls_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Transcription File:").pack(side=tk.LEFT, padx=(0, 5))
        self.conv_file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.conv_file_var, width=60)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.select_transcription_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Analyze", command=self.analyze_transcription).pack(side=tk.LEFT)
        
        # Privacy mode section
        privacy_frame = ttk.Frame(controls_frame)
        privacy_frame.pack(fill=tk.X, pady=5)
        
        # Privacy mode toggle with prominent styling
        self.privacy_mode_var = tk.BooleanVar(value=False)
        privacy_checkbox = ttk.Checkbutton(
            privacy_frame, 
            text="🔒 Privacy Mode - Only save summaries and analyses (transcription text will not be saved)", 
            variable=self.privacy_mode_var,
            command=self.on_privacy_mode_changed
        )
        privacy_checkbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # Privacy status indicator
        self.privacy_status_label = ttk.Label(privacy_frame, text="", foreground="green")
        self.privacy_status_label.pack(side=tk.LEFT)
        
        # Analysis options
        options_frame = ttk.Frame(controls_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        # Summary options
        summary_frame = ttk.LabelFrame(options_frame, text="Summary Options")
        summary_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.summary_overall = tk.BooleanVar(value=True)
        self.summary_speaker = tk.BooleanVar(value=True)
        self.summary_time = tk.BooleanVar(value=True)
        self.summary_keypoints = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(summary_frame, text="Overall Summary", variable=self.summary_overall).pack(anchor=tk.W)
        ttk.Checkbutton(summary_frame, text="By Speaker", variable=self.summary_speaker).pack(anchor=tk.W)
        ttk.Checkbutton(summary_frame, text="Time-based", variable=self.summary_time).pack(anchor=tk.W)
        ttk.Checkbutton(summary_frame, text="Key Points", variable=self.summary_keypoints).pack(anchor=tk.W)
        
        # Annotation options
        annotation_frame = ttk.LabelFrame(options_frame, text="Annotation Options")
        annotation_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.ann_topics = tk.BooleanVar(value=True)
        self.ann_actions = tk.BooleanVar(value=True)
        self.ann_qa = tk.BooleanVar(value=True)
        self.ann_decisions = tk.BooleanVar(value=True)
        self.ann_emotions = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(annotation_frame, text="Topics", variable=self.ann_topics).pack(anchor=tk.W)
        ttk.Checkbutton(annotation_frame, text="Action Items", variable=self.ann_actions).pack(anchor=tk.W)
        ttk.Checkbutton(annotation_frame, text="Q&A Pairs", variable=self.ann_qa).pack(anchor=tk.W)
        ttk.Checkbutton(annotation_frame, text="Decisions", variable=self.ann_decisions).pack(anchor=tk.W)
        ttk.Checkbutton(annotation_frame, text="Emotional Moments", variable=self.ann_emotions).pack(anchor=tk.W)
        
        # Results display area with notebook
        results_notebook = ttk.Notebook(conv_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary Results Tab
        self.setup_summary_results_tab(results_notebook)
        
        # Annotation Results Tab
        self.setup_annotation_results_tab(results_notebook)
        
        # Export Tab
        self.setup_export_tab(results_notebook)
    
    def on_privacy_mode_changed(self):
        """Handle privacy mode toggle for conversation analysis"""
        privacy_enabled = self.privacy_mode_var.get()
        
        if privacy_enabled:
            self.privacy_info.config(text="Privacy mode: Speaker names will be anonymized in analysis")
        else:
            self.privacy_info.config(text="Privacy mode disabled: Speaker names will be preserved")
    
    def on_batch_privacy_mode_changed(self):
        """Handle privacy mode toggle for batch processing"""
        privacy_mode = self.batch_privacy_mode_var.get()
        
        if privacy_mode:
            self.batch_privacy_info.config(text="Legal-Safe Mode: Only analysis will be saved (no verbatim transcription)")
        else:
            self.batch_privacy_info.config(text="Legal-Safe Mode disabled: Full transcription will be saved")
        
        # Optionally disable speaker mapping when legal-safe mode is enabled
        # Legal-safe mode and speaker mapping can coexist but mapping is less relevant
    
    def analyze_transcription(self):
        """Analyze the selected transcription file"""
        if not self.enhanced_processor:
            messagebox.showerror("Error", "Enhanced transcription processor not available!")
            return
        
        file_path = self.conv_file_var.get().strip()
        if not file_path or not Path(file_path).exists():
            messagebox.showerror("Error", "Please select a valid transcription file!")
            return
        
        privacy_mode = self.privacy_mode_var.get()
        
        try:
            # Load transcription file
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    transcription_data = json.load(f)
                else:
                    # Simple text file - create basic structure
                    text_content = f.read()
                    transcription_data = {
                        "segments": [{"text": text_content, "speaker": "Unknown", "start": 0, "end": 0}],
                        "full_text": text_content
                    }
            
            # Get selected options
            summary_types = []
            if self.summary_overall.get():
                summary_types.append("overall")
            if self.summary_speaker.get():
                summary_types.append("by_speaker")
            if self.summary_time.get():
                summary_types.append("by_time")
            if self.summary_keypoints.get():
                summary_types.append("key_points")
            
            annotation_types = []
            if self.ann_topics.get():
                annotation_types.append("topics")
            if self.ann_actions.get():
                annotation_types.append("action_items")
            if self.ann_qa.get():
                annotation_types.append("questions_answers")
            if self.ann_decisions.get():
                annotation_types.append("decisions")
            if self.ann_emotions.get():
                annotation_types.append("emotional_moments")
            
            # Show progress
            status_msg = "Analyzing transcription (Privacy Mode)..." if privacy_mode else "Analyzing transcription..."
            self.update_status(status_msg)
            self.root.update()
            
            # Generate summaries
            summaries = {}
            if summary_types:
                summaries = self.enhanced_processor.generate_conversation_summary(
                    transcription_data, summary_types
                )
                self.display_summaries(summaries)
            
            # Generate annotations
            annotations = {}
            if annotation_types:
                annotations = self.enhanced_processor.generate_conversation_annotations(
                    transcription_data, annotation_types
                )
                self.display_annotations(annotations)
            
            # Store for export - handling privacy mode
            if privacy_mode:
                # In privacy mode, clear transcription text but keep metadata
                privacy_safe_transcription = {
                    "segments": [
                        {
                            "speaker": seg.get("speaker", "Unknown"),
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "text": "[REDACTED FOR PRIVACY]",
                            # Keep any analysis results but remove original text
                            **{k: v for k, v in seg.items() if k not in ["text", "full_text"]}
                        }
                        for seg in transcription_data.get("segments", [])
                    ],
                    "full_text": "[REDACTED FOR PRIVACY]",
                    "privacy_mode": True,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "original_file": Path(file_path).name
                }
                
                self.current_analysis = {
                    "summaries": summaries,
                    "annotations": annotations,
                    "transcription": privacy_safe_transcription,
                    "privacy_mode": True
                }
                
                # Clear original transcription data from memory for extra security
                del transcription_data
                
                self.update_status("Analysis completed (Privacy Mode - transcription text cleared)")
                messagebox.showinfo(
                    "Analysis Complete (Privacy Mode)", 
                    "✅ Analysis completed successfully!\n\n" +
                    "🔒 Privacy Mode was active:\n" +
                    "• Original transcription text has been cleared from memory\n" +
                    "• Only summaries and analyses are retained\n" +
                    "• Export options will be limited to privacy-safe content"
                )
            else:
                self.current_analysis = {
                    "summaries": summaries,
                    "annotations": annotations,
                    "transcription": transcription_data,
                    "privacy_mode": False
                }
                
                self.update_status("Analysis completed successfully!")
                messagebox.showinfo("Success", "Transcription analysis completed!")
            
        except Exception as e:
            error_msg = f"Error analyzing transcription: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def display_summaries(self, summaries):
        """Display generated summaries in the UI"""
        # Update export tab privacy indicator
        if hasattr(self, 'current_analysis') and self.current_analysis.get('privacy_mode', False):
            self.export_privacy_label.config(
                text="🔒 PRIVACY MODE ACTIVE - Only summaries and analyses available for export", 
                foreground="red"
            )
            # Update radio button text for privacy mode
            self.json_radio.config(text="JSON (Analytics Only)")
        else:
            self.export_privacy_label.config(
                text="🔓 Standard Mode - Full transcription available for export", 
                foreground="green"
            )
            # Update radio button text for standard mode
            self.json_radio.config(text="JSON (Complete)")
        
        # Overall Summary
        if "overall" in summaries:
            overall = summaries["overall"]
            summary_text = f"""📋 OVERALL SUMMARY
{'='*50}

Summary: {overall.get('summary', 'N/A')}

Key Metrics:
• Duration: {overall.get('key_metrics', {}).get('duration_minutes', 0)} minutes
• Speakers: {overall.get('key_metrics', {}).get('speaker_count', 0)}
• Segments: {overall.get('key_metrics', {}).get('segment_count', 0)}
• Dominant Sentiment: {overall.get('key_metrics', {}).get('dominant_sentiment', 'N/A')}

Main Topics: {', '.join(overall.get('main_topics', []))}
"""
            self.overall_summary_text.delete(1.0, tk.END)
            self.overall_summary_text.insert(tk.END, summary_text)
        
        # Speaker Summaries
        if "by_speaker" in summaries:
            speaker_text = "👥 SPEAKER SUMMARIES\n" + "="*50 + "\n\n"
            for speaker, data in summaries["by_speaker"].items():
                stats = data.get("participation_stats", {})
                speaker_text += f"{speaker}:\n"
                speaker_text += f"  Summary: {data.get('summary', 'N/A')}\n"
                speaker_text += f"  Participation: {stats.get('participation_percentage', 0)}%\n"
                speaker_text += f"  Total Time: {stats.get('total_time_seconds', 0)}s\n"
                speaker_text += f"  Segments: {stats.get('segment_count', 0)}\n"
                speaker_text += f"  Dominant Sentiment: {stats.get('dominant_sentiment', 'N/A')}\n\n"
            
            self.speaker_summary_text.delete(1.0, tk.END)
            self.speaker_summary_text.insert(tk.END, speaker_text)
        
        # Key Points
        if "key_points" in summaries:
            keypoints_text = "🎯 KEY POINTS\n" + "="*50 + "\n\n"
            for i, point in enumerate(summaries["key_points"], 1):
                keypoints_text += f"{i}. [{point.get('type', 'general')}] {point.get('text', 'N/A')}\n"
                keypoints_text += f"   Speaker: {point.get('speaker', 'N/A')} at {point.get('timestamp', 0):.1f}s\n"
                keypoints_text += f"   Sentiment: {point.get('sentiment', 'N/A')}\n\n"
            
            self.keypoints_text.delete(1.0, tk.END)
            self.keypoints_text.insert(tk.END, keypoints_text)
    
    def display_annotations(self, annotations):
        """Display generated annotations in the UI"""
        # Action Items
        if "action_items" in annotations:
            # Clear existing items
            for item in self.actions_tree.get_children():
                self.actions_tree.delete(item)
            
            for action in annotations["action_items"]:
                self.actions_tree.insert('', tk.END, 
                                       text=action.get('action', 'N/A'),
                                       values=(
                                           action.get('speaker', 'N/A'),
                                           f"{action.get('timestamp', 0):.1f}s",
                                           f"{action.get('confidence', 0):.2f}"
                                       ))
        
        # Q&A Pairs
        if "questions_answers" in annotations:
            qa_text = "❓ QUESTION & ANSWER PAIRS\n" + "="*50 + "\n\n"
            for i, pair in enumerate(annotations["questions_answers"], 1):
                qa_text += f"{i}. Q: {pair.get('question', 'N/A')}\n"
                qa_text += f"   Speaker: {pair.get('question_speaker', 'N/A')} at {pair.get('question_timestamp', 0):.1f}s\n\n"
                qa_text += f"   A: {pair.get('answer', 'N/A')}\n"
                qa_text += f"   Speaker: {pair.get('answer_speaker', 'N/A')} at {pair.get('answer_timestamp', 0):.1f}s\n"
                qa_text += f"   Confidence: {pair.get('confidence', 0):.2f}\n\n"
            
            self.qa_text.delete(1.0, tk.END)
            self.qa_text.insert(tk.END, qa_text)
        
        # Topics
        if "topics" in annotations:
            # Clear existing items
            for item in self.topics_tree.get_children():
                self.topics_tree.delete(item)
            
            for topic in annotations["topics"]:
                self.topics_tree.insert('', tk.END,
                                      text=f"{topic.get('topic', 'N/A')} ({', '.join(topic.get('keywords', []))})",
                                      values=(
                                          topic.get('mentions', 0),
                                          f"{topic.get('first_mentioned', 0):.1f}s"
                                      ))
        
        # Decisions
        if "decisions" in annotations:
            decisions_text = "🎯 DECISIONS & CONCLUSIONS\n" + "-"*15 + "\n"
            for i, decision in enumerate(annotations["decisions"], 1):
                decisions_text += f"{i}. {decision.get('decision', 'N/A')}\n"
                decisions_text += f"   Speaker: {decision.get('speaker', 'N/A')} at {decision.get('timestamp', 0):.1f}s\n"
                decisions_text += f"   Sentiment: {decision.get('sentiment', 'N/A')}\n"
                decisions_text += f"   Context: {decision.get('context', 'N/A')[:100]}...\n\n"
            
            self.decisions_text.delete(1.0, tk.END)
            self.decisions_text.insert(tk.END, decisions_text)
    
    def export_summary(self):
        """Export summary results"""
        if not hasattr(self, 'current_analysis'):
            messagebox.showwarning("Warning", "No analysis results to export!")
            return
        
        self._export_data(self.current_analysis.get("summaries", {}), "summary")
    
    def export_annotations(self):
        """Export annotation results"""
        if not hasattr(self, 'current_analysis'):
            messagebox.showwarning("Warning", "No analysis results to export!")
            return
        
        self._export_data(self.current_analysis.get("annotations", {}), "annotations")
    
    def export_all(self):
        """Export all analysis results"""
        if not hasattr(self, 'current_analysis'):
            messagebox.showwarning("Warning", "No analysis results to export!")
            return
        
        self._export_data(self.current_analysis, "complete_analysis")
    
    def _export_data(self, data, data_type):
        """Export data in the selected format"""
        if not data:
            messagebox.showwarning("Warning", "No data to export!")
            return
        
        # Check if privacy mode is active
        privacy_mode = hasattr(self, 'current_analysis') and self.current_analysis.get('privacy_mode', False)
        
        # Get export format
        export_format = self.export_format.get()
        
        # Generate filename with privacy indicator
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        privacy_suffix = "_privacy" if privacy_mode else ""
        filename_base = f"oreja_{data_type}{privacy_suffix}_{timestamp}"
        
        if export_format == "json":
            filename = filename_base + ".json"
            default_ext = ".json"
            filetypes = [("JSON Files", "*.json"), ("All Files", "*.*")]
        else:
            filename = filename_base + ".txt"
            default_ext = ".txt"
            filetypes = [("Text Files", "*.txt"), ("All Files", "*.*")]
        
        # Show save dialog
        file_path = filedialog.asksaveasfilename(
            title=f"Export {data_type.replace('_', ' ').title()}" + (" (Privacy Mode)" if privacy_mode else ""),
            initialvalue=filename,
            defaultextension=default_ext,
            filetypes=filetypes
        )
        
        if not file_path:
            return
        
        try:
            if export_format == "json":
                # For JSON export, always include privacy mode indicator
                export_data = data.copy() if isinstance(data, dict) else data
                if privacy_mode:
                    if isinstance(export_data, dict):
                        export_data["__privacy_mode"] = True
                        export_data["__export_notice"] = "This file was exported in Privacy Mode - transcription text has been redacted"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                export_text = self._format_export_text(data, export_format)
                
                # Add privacy mode header for text exports
                if privacy_mode:
                    privacy_header = f"""🔒 PRIVACY MODE EXPORT
{'='*50}
This export was generated in Privacy Mode.
Original transcription text has been redacted for privacy protection.
Only summaries and analyses are included.
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*50}

"""
                    export_text = privacy_header + export_text
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(export_text)
            
            success_msg = f"Data exported to {file_path}"
            if privacy_mode:
                success_msg += "\n\n🔒 Privacy Mode: Transcription text was not included"
            
            messagebox.showinfo("Export Complete", success_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def _format_export_text(self, data, format_type):
        """Format data as text for export"""
        if format_type == "text":
            return self._format_as_report(data)
        elif format_type == "minutes":
            return self._format_as_minutes(data)
        elif format_type == "actions":
            return self._format_actions_only(data)
        else:
            return str(data)
    
    def _format_as_report(self, data):
        """Format as comprehensive report"""
        privacy_mode = hasattr(self, 'current_analysis') and self.current_analysis.get('privacy_mode', False)
        
        report = f"""OREJA CONVERSATION ANALYSIS REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Privacy Mode: {'ACTIVE (transcription text redacted)' if privacy_mode else 'Standard (full transcription available)'}

"""
        
        # Add summaries
        if "summaries" in data:
            summaries = data["summaries"]
            
            if "overall" in summaries:
                overall = summaries["overall"]
                report += f"""OVERALL SUMMARY
{'-'*20}
{overall.get('summary', 'N/A')}

Key Metrics:
• Duration: {overall.get('key_metrics', {}).get('duration_minutes', 0)} minutes
• Speakers: {overall.get('key_metrics', {}).get('speaker_count', 0)}
• Segments: {overall.get('key_metrics', {}).get('segment_count', 0)}
• Dominant Sentiment: {overall.get('key_metrics', {}).get('dominant_sentiment', 'N/A')}

Main Topics: {', '.join(overall.get('main_topics', []))}

"""
        
        # Add annotations
        if "annotations" in data:
            annotations = data["annotations"]
            
            if "action_items" in annotations:
                report += "ACTION ITEMS\n" + "-"*20 + "\n"
                for i, action in enumerate(annotations["action_items"], 1):
                    report += f"{i}. {action.get('action', 'N/A')} ({action.get('speaker', 'N/A')})\n"
                report += "\n"
            
            if "decisions" in annotations:
                report += "DECISIONS MADE\n" + "-"*20 + "\n"
                for i, decision in enumerate(annotations["decisions"], 1):
                    report += f"{i}. {decision.get('decision', 'N/A')} ({decision.get('speaker', 'N/A')})\n"
                report += "\n"
        
        # Add privacy notice if applicable
        if privacy_mode:
            report += f"""
PRIVACY NOTICE
{'-'*20}
This report was generated in Privacy Mode. Original transcription
text has been redacted and is not included in this export.
Only summaries and analytical insights are provided.
"""
        
        return report
    
    def _format_as_minutes(self, data):
        """Format as meeting minutes"""
        privacy_mode = hasattr(self, 'current_analysis') and self.current_analysis.get('privacy_mode', False)
        
        minutes = f"""MEETING MINUTES
{'='*30}
Date: {datetime.now().strftime('%Y-%m-%d')}
Time: {datetime.now().strftime('%H:%M:%S')}
Privacy Mode: {'ACTIVE' if privacy_mode else 'Standard'}

"""
        
        # Add attendees from summaries
        if "summaries" in data and "by_speaker" in data["summaries"]:
            minutes += "ATTENDEES\n" + "-"*10 + "\n"
            for speaker in data["summaries"]["by_speaker"].keys():
                minutes += f"• {speaker}\n"
            minutes += "\n"
        
        # Add key points
        if "summaries" in data and "key_points" in data["summaries"]:
            minutes += "KEY DISCUSSION POINTS\n" + "-"*25 + "\n"
            for point in data["summaries"]["key_points"]:
                minutes += f"• {point.get('text', 'N/A')}\n"
            minutes += "\n"
        
        # Add action items
        if "annotations" in data and "action_items" in data["annotations"]:
            minutes += "ACTION ITEMS\n" + "-"*15 + "\n"
            for action in data["annotations"]["action_items"]:
                minutes += f"• {action.get('action', 'N/A')} - {action.get('speaker', 'N/A')}\n"
            minutes += "\n"
        
        # Add decisions
        if "annotations" in data and "decisions" in data["annotations"]:
            minutes += "DECISIONS MADE\n" + "-"*15 + "\n"
            for decision in data["annotations"]["decisions"]:
                minutes += f"• {decision.get('decision', 'N/A')}\n"
            minutes += "\n"
        
        # Add privacy notice if applicable
        if privacy_mode:
            minutes += f"""PRIVACY NOTICE
{'-'*15}
These minutes were generated in Privacy Mode.
Original conversation text has been redacted.
Only summaries and key points are included.
"""
        
        return minutes
    
    def _format_actions_only(self, data):
        """Format action items only"""
        actions_text = f"""ACTION ITEMS LIST
{'='*20}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if "annotations" in data and "action_items" in data["annotations"]:
            for i, action in enumerate(data["annotations"]["action_items"], 1):
                actions_text += f"{i}. {action.get('action', 'N/A')}\n"
                actions_text += f"   Assigned to: {action.get('speaker', 'N/A')}\n"
                actions_text += f"   Mentioned at: {action.get('timestamp', 0):.1f}s\n\n"
        else:
            actions_text += "No action items found.\n"
        
        return actions_text

    def setup_batch_tab(self):
        """Setup the batch processing tab for recorded calls"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="🎬 Batch Processing")
        
        # Title
        title_frame = ttk.Frame(batch_frame)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(title_frame, text="🎬 Batch Transcription of Recorded Calls", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT)
        
        # Help button
        ttk.Button(title_frame, text="❓ Help", 
                  command=self.show_batch_help).pack(side=tk.RIGHT, padx=5)
        
        # Main content with paned window
        main_paned = ttk.PanedWindow(batch_frame, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top section: File selection and settings
        top_frame = ttk.LabelFrame(main_paned, text="File Selection & Settings", padding=10)
        main_paned.add(top_frame, weight=1)
        
        # File selection section
        file_section = ttk.LabelFrame(top_frame, text="Audio Files", padding=10)
        file_section.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # File selection controls
        file_controls = ttk.Frame(file_section)
        file_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_controls, text="📁 Add Files", 
                  command=self.add_batch_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_controls, text="📂 Add Folder", 
                  command=self.add_batch_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_controls, text="🗑️ Clear All", 
                  command=self.clear_batch_files).pack(side=tk.LEFT, padx=5)
        
        # File list
        file_list_frame = ttk.Frame(file_section)
        file_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for file list
        columns = ('size', 'duration', 'status')
        self.batch_file_tree = ttk.Treeview(file_list_frame, columns=columns, 
                                           show='tree headings', height=8)
        
        self.batch_file_tree.heading('#0', text='Audio File')
        self.batch_file_tree.heading('size', text='Size')
        self.batch_file_tree.heading('duration', text='Duration')
        self.batch_file_tree.heading('status', text='Status')
        
        self.batch_file_tree.column('#0', width=300)
        self.batch_file_tree.column('size', width=80)
        self.batch_file_tree.column('duration', width=80)
        self.batch_file_tree.column('status', width=100)
        
        # File list scrollbar
        file_scroll = ttk.Scrollbar(file_list_frame, orient=tk.VERTICAL, 
                                   command=self.batch_file_tree.yview)
        self.batch_file_tree.configure(yscrollcommand=file_scroll.set)
        
        self.batch_file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Settings section
        settings_section = ttk.LabelFrame(top_frame, text="Processing Settings", padding=15)
        settings_section.pack(fill=tk.X, pady=(10, 10))
        
        settings_grid = ttk.Frame(settings_section)
        settings_grid.pack(fill=tk.X, pady=(0, 10))
        
        # Output directory
        ttk.Label(settings_grid, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_output_var = tk.StringVar(value="transcription_results")
        output_entry = ttk.Entry(settings_grid, textvariable=self.batch_output_var, width=40)
        output_entry.grid(row=0, column=1, padx=5, sticky=tk.W, pady=5)
        ttk.Button(settings_grid, text="Browse", 
                  command=self.select_output_directory).grid(row=0, column=2, padx=5, pady=5)
        
        # Improve speakers checkbox
        self.improve_speakers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_grid, text="Use recordings to improve speaker models", 
                       variable=self.improve_speakers_var).grid(row=1, column=0, columnspan=2, 
                                                               sticky=tk.W, padx=5, pady=8)
        
        # Speaker name mapping
        ttk.Label(settings_grid, text="Speaker Mapping:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.speaker_mapping_var = tk.StringVar()
        mapping_entry = ttk.Entry(settings_grid, textvariable=self.speaker_mapping_var, width=40)
        mapping_entry.grid(row=2, column=1, padx=5, sticky=tk.W, pady=5)
        ttk.Button(settings_grid, text="Browse", 
                  command=self.select_speaker_mapping).grid(row=2, column=2, padx=5, pady=5)
        ttk.Button(settings_grid, text="Create", 
                  command=self.create_speaker_mapping).grid(row=2, column=3, padx=5, pady=5)
        
        # Privacy mode toggle
        self.batch_privacy_mode_var = tk.BooleanVar()
        privacy_checkbox = ttk.Checkbutton(settings_grid, text="🔒 Legal-Safe Mode (no verbatim text saved)",
                                           variable=self.batch_privacy_mode_var,
                                           command=self.on_batch_privacy_mode_changed)
        privacy_checkbox.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=8)
        
        # Privacy mode info label
        self.batch_privacy_info = ttk.Label(settings_grid, text="Legal-Safe Mode disabled: Full transcription will be saved",
                                            font=("Arial", 8), foreground="gray")
        self.batch_privacy_info.grid(row=4, column=0, columnspan=2, sticky="w", padx=15, pady=(0, 10))
        
        # Processing controls section with better spacing
        controls_section = ttk.LabelFrame(settings_section, text="Processing Controls", padding=15)
        controls_section.pack(fill=tk.X, pady=15)
        
        # Button row
        button_frame = ttk.Frame(controls_section)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.batch_process_btn = ttk.Button(button_frame, text="🚀 Start Processing", 
                                           command=self.start_batch_processing)
        self.batch_process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.batch_stop_btn = ttk.Button(button_frame, text="⏹️ Stop", 
                                        command=self.stop_batch_processing, state=tk.DISABLED)
        self.batch_stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress row
        progress_frame = ttk.Frame(controls_section)
        progress_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar
        self.batch_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.batch_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Progress label
        self.batch_progress_label = ttk.Label(progress_frame, text="Ready")
        self.batch_progress_label.pack(side=tk.RIGHT)
        
        # Bottom section: Results and log
        bottom_frame = ttk.LabelFrame(main_paned, text="Processing Results", padding=10)
        main_paned.add(bottom_frame, weight=2)
        
        # Results notebook
        results_notebook = ttk.Notebook(bottom_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        results_tab = ttk.Frame(results_notebook)
        results_notebook.add(results_tab, text="📊 Results")
        
        # Results tree
        results_columns = ('file', 'status', 'speakers', 'segments', 'confidence', 'processing_time')
        self.batch_results_tree = ttk.Treeview(results_tab, columns=results_columns, 
                                              show='headings', height=10)
        
        for col in results_columns:
            self.batch_results_tree.heading(col, text=col.replace('_', ' ').title())
            self.batch_results_tree.column(col, width=120)
        
        # Results scrollbar
        results_scroll = ttk.Scrollbar(results_tab, orient=tk.VERTICAL, 
                                      command=self.batch_results_tree.yview)
        self.batch_results_tree.configure(yscrollcommand=results_scroll.set)
        
        self.batch_results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Log tab
        log_tab = ttk.Frame(results_notebook)
        results_notebook.add(log_tab, text="📝 Processing Log")
        
        self.batch_log = tk.Text(log_tab, height=10, font=('Consolas', 9))
        log_scroll_v = ttk.Scrollbar(log_tab, orient=tk.VERTICAL, command=self.batch_log.yview)
        self.batch_log.configure(yscrollcommand=log_scroll_v.set)
        
        self.batch_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll_v.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize batch processing variables
        self.batch_files = []
        self.batch_processor = None
        self.processing_thread = None
        
    def setup_management_tab(self):
        """Setup the enhanced database management tab"""
        mgmt_frame = ttk.Frame(self.notebook)
        self.notebook.add(mgmt_frame, text="⚙️ Management")
        
        # Create paned window for better layout
        mgmt_paned = ttk.PanedWindow(mgmt_frame, orient=tk.HORIZONTAL)
        mgmt_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Speaker management
        left_mgmt_frame = ttk.LabelFrame(mgmt_paned, text="Speaker Profile Management", padding=10)
        mgmt_paned.add(left_mgmt_frame, weight=2)
        
        # Search and filter controls
        search_frame = ttk.Frame(left_mgmt_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="🔍 Search/Filter:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(0, 5))
        search_entry.bind('<KeyRelease>', self.filter_speakers)
        
        ttk.Button(search_frame, text="Clear", 
                  command=lambda: (self.search_var.set(""), self.filter_speakers())).pack(side=tk.LEFT, padx=5)
        
        # Speaker management controls
        mgmt_controls = ttk.Frame(left_mgmt_frame)
        mgmt_controls.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1: Individual operations
        row1 = ttk.Frame(mgmt_controls)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(row1, text="✏️ Edit Selected", 
                  command=self.edit_selected_speaker).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row1, text="🔗 Merge Selected", 
                  command=self.merge_selected_speakers).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="🗑️ Delete Selected", 
                  command=self.delete_selected_speakers).pack(side=tk.LEFT, padx=5)
        
        # Row 2: Bulk operations
        row2 = ttk.Frame(mgmt_controls)
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(row2, text="🔍 Find Similar Names", 
                  command=self.find_similar_speakers).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2, text="🧹 Bulk Cleanup", 
                  command=self.show_bulk_cleanup_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="📋 Bulk Rename", 
                  command=self.show_bulk_rename_dialog).pack(side=tk.LEFT, padx=5)
        
        # Enhanced speaker list with checkboxes for multi-select
        list_frame = ttk.Frame(left_mgmt_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create enhanced treeview with checkboxes
        self.mgmt_speaker_tree = ttk.Treeview(list_frame, 
                                             columns=('select', 'type', 'samples', 'confidence', 'last_seen', 'id'), 
                                             show='tree headings', height=20)
        
        # Configure columns
        self.mgmt_speaker_tree.heading('#0', text='Speaker Name')
        self.mgmt_speaker_tree.heading('select', text='☐')
        self.mgmt_speaker_tree.heading('type', text='Type')
        self.mgmt_speaker_tree.heading('samples', text='Samples')
        self.mgmt_speaker_tree.heading('confidence', text='Confidence')
        self.mgmt_speaker_tree.heading('last_seen', text='Last Seen')
        self.mgmt_speaker_tree.heading('id', text='Speaker ID')
        
        self.mgmt_speaker_tree.column('#0', width=200)
        self.mgmt_speaker_tree.column('select', width=30)
        self.mgmt_speaker_tree.column('type', width=80)
        self.mgmt_speaker_tree.column('samples', width=70)
        self.mgmt_speaker_tree.column('confidence', width=80)
        self.mgmt_speaker_tree.column('last_seen', width=100)
        self.mgmt_speaker_tree.column('id', width=150)
        
        # Scrollbars
        tree_scroll_y = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.mgmt_speaker_tree.yview)
        tree_scroll_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.mgmt_speaker_tree.xview)
        self.mgmt_speaker_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        
        self.mgmt_speaker_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind events for multi-select
        self.mgmt_speaker_tree.bind('<Button-1>', self.on_tree_click)
        self.mgmt_speaker_tree.bind('<Double-Button-1>', self.on_tree_double_click)
        
        # Track selected speakers
        self.selected_speakers = set()
        
        # Right panel: Operations and logs
        right_mgmt_frame = ttk.Frame(mgmt_paned)
        mgmt_paned.add(right_mgmt_frame, weight=1)
        
        # Database operations
        db_ops_frame = ttk.LabelFrame(right_mgmt_frame, text="Database Operations", padding=10)
        db_ops_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1: Data operations
        db_row1 = ttk.Frame(db_ops_frame)
        db_row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(db_row1, text="🔄 Refresh", 
                  command=self.refresh_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(db_row1, text="📊 Report", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=5)
        
        # Row 2: Export/Import
        db_row2 = ttk.Frame(db_ops_frame)
        db_row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(db_row2, text="💾 Export", 
                  command=self.export_database).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(db_row2, text="⚠️ Reset", 
                  command=self.reset_database).pack(side=tk.LEFT, padx=5)
        
        # Selection info
        info_frame = ttk.LabelFrame(right_mgmt_frame, text="Selection Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.selection_info = tk.Text(info_frame, height=8, font=('Consolas', 9))
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.selection_info.yview)
        self.selection_info.configure(yscrollcommand=info_scroll.set)
        
        self.selection_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Management log
        log_frame = ttk.LabelFrame(right_mgmt_frame, text="Management Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.mgmt_log = tk.Text(log_frame, height=15, font=('Consolas', 9))
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.mgmt_log.yview)
        self.mgmt_log.configure(yscrollcommand=log_scroll.set)
        
        self.mgmt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_user_training_tab(self):
        """Setup the user training tab"""
        try:
            # Import the user embedding trainer
            from user_embedding_trainer import UserEmbeddingTrainer
            
            # Create the trainer module as a tab, sharing our database instance
            self.user_trainer = UserEmbeddingTrainer(self.notebook, shared_db=self.speaker_db)
            
        except ImportError as e:
            # Fallback if dependencies are missing
            training_frame = ttk.Frame(self.notebook)
            self.notebook.add(training_frame, text="🎯 User Training")
            
            error_label = ttk.Label(training_frame, 
                                  text=f"User Training module not available.\n\nMissing dependencies: {e}\n\nPlease install: pip install sounddevice soundfile librosa",
                                  font=('Arial', 12),
                                  justify=tk.CENTER)
            error_label.pack(expand=True)
        except Exception as e:
            # Other errors
            training_frame = ttk.Frame(self.notebook)
            self.notebook.add(training_frame, text="🎯 User Training")
            
            error_label = ttk.Label(training_frame, 
                                  text=f"Error loading User Training module:\n\n{e}",
                                  font=('Arial', 12),
                                  justify=tk.CENTER)
            error_label.pack(expand=True)
        
    def setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.last_update_label = ttk.Label(self.status_bar, text="")
        self.last_update_label.pack(side=tk.RIGHT, padx=5, pady=2)
        
    def setup_auto_refresh(self):
        """Setup automatic refresh thread"""
        def refresh_worker():
            while True:
                if self.auto_refresh.get():
                    try:
                        self.root.after(0, self.refresh_data)
                    except:
                        break
                time.sleep(self.refresh_interval)
        
        refresh_thread = threading.Thread(target=refresh_worker, daemon=True)
        refresh_thread.start()
        
    def refresh_data(self):
        """Refresh all data from the database and backend"""
        try:
            self.update_status("Refreshing data...")
            
            # Load speaker data from database
            self.load_speaker_data()
            
            # Update UI components
            self.update_speaker_list()
            self.update_statistics()
            self.update_analytics()
            self.update_management_speaker_list()  # Update management tab
            
            # Update status
            self.update_status("Data refreshed successfully")
            self.last_update_label.config(text=f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.update_status(f"Error refreshing data: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh data: {str(e)}")
    
    def load_speaker_data(self):
        """Load speaker data from the enhanced database"""
        try:
            # Get all speakers from the enhanced database
            speakers = self.speaker_db.get_all_speakers()
            
            # Convert to the format expected by the dashboard
            self.speaker_data = {}
            for speaker in speakers:
                speaker_id = speaker['speaker_id']
                self.speaker_data[speaker_id] = {
                    'speaker_id': speaker_id,
                    'display_name': speaker['display_name'],
                    'embedding_count': speaker['embedding_count'],
                    'average_confidence': speaker['average_confidence'],
                    'last_seen': speaker['last_seen'],
                    'created_date': speaker['created_date'],
                    'is_enrolled': speaker['is_enrolled'],
                    'is_verified': speaker['is_verified'],
                    'source_type': speaker['source_type'],
                    'type': self.get_speaker_type(speaker_id),
                    'total_audio_seconds': 0.0,  # Add missing field with default value
                    'session_count': 1  # Add missing field with default value
                }
            
            # Load embeddings for selected speaker if any
            if self.selected_speaker and self.selected_speaker in self.speaker_data:
                self.load_speaker_embeddings(self.selected_speaker)
            
        except Exception as e:
            print(f"Error loading speaker data: {e}")
            self.speaker_data = {}
    
    def load_speaker_embeddings(self, speaker_id):
        """Load embedding history for a specific speaker"""
        try:
            # For now, just set empty embeddings since we don't track embedding history in the enhanced database
            # The enhanced database focuses on aggregated stats rather than individual embedding history
            self.speaker_data[speaker_id]['embeddings'] = []
            
        except Exception as e:
            print(f"Error loading embeddings for {speaker_id}: {e}")
            self.speaker_data[speaker_id]['embeddings'] = []
    
    def get_speaker_type(self, speaker_id):
        """Determine speaker type based on ID"""
        # Enhanced database speaker IDs start with 'spk_' (UUID-based)
        if speaker_id.startswith('spk_'):
            # For enhanced database, check the source_type field
            return '✅ Enhanced'
        elif speaker_id.startswith('AUTO_SPEAKER'):
            return '🤖 Auto'
        elif speaker_id.startswith('CORRECTED_SPEAKER'):
            return '✅ Corrected'
        elif speaker_id.startswith('ENROLLED_SPEAKER'):
            return '👤 Enrolled'
        else:
            return '❓ Unknown'
    
    def update_speaker_list(self):
        """Update the speaker list in the overview tab"""
        # Clear existing items
        for item in self.speaker_tree.get_children():
            self.speaker_tree.delete(item)
        
        # Sort speakers by embedding count (most trained first)
        sorted_speakers = sorted(self.speaker_data.items(), 
                               key=lambda x: x[1]['embedding_count'], 
                               reverse=True)
        
        for speaker_id, data in sorted_speakers:
            # Format confidence
            confidence = f"{data['average_confidence']:.3f}" if data['average_confidence'] else "0.000"
            
            # Format last seen
            try:
                last_seen = datetime.fromisoformat(data['last_seen']).strftime('%m/%d %H:%M')
            except:
                last_seen = "Never"
            
            # Get display name (enhanced database uses 'display_name', legacy uses 'name')
            display_name = data.get('display_name', data.get('name', 'Unknown'))
            
            # Insert into tree
            self.speaker_tree.insert('', tk.END, 
                                   values=(data['type'], 
                                          data['embedding_count'],
                                          confidence,
                                          last_seen),
                                   text=display_name,
                                   tags=(data['type'],))
        
        # Configure tag colors
        self.speaker_tree.tag_configure('🤖 Auto', background='#ffe6e6')
        self.speaker_tree.tag_configure('✅ Corrected', background='#e6ffe6')
        self.speaker_tree.tag_configure('✅ Enhanced', background='#e6f7ff')
        self.speaker_tree.tag_configure('👤 Enrolled', background='#e6f3ff')
    
    def update_statistics(self):
        """Update the statistics display"""
        if not self.speaker_data:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "No speaker data available")
            return
        
        # Calculate statistics
        total_speakers = len(self.speaker_data)
        auto_speakers = sum(1 for s in self.speaker_data.values() if s['type'] == '🤖 Auto')
        corrected_speakers = sum(1 for s in self.speaker_data.values() if s['type'] == '✅ Corrected')
        enhanced_speakers = sum(1 for s in self.speaker_data.values() if s['type'] == '✅ Enhanced')
        enrolled_speakers = sum(1 for s in self.speaker_data.values() if s['type'] == '👤 Enrolled')
        
        total_embeddings = sum(s['embedding_count'] for s in self.speaker_data.values())
        avg_confidence = np.mean([s['average_confidence'] for s in self.speaker_data.values() 
                                if s['average_confidence'] is not None])
        
        # Find most active speaker
        most_active = max(self.speaker_data.values(), key=lambda x: x['embedding_count'])
        most_active_name = most_active.get('display_name', most_active.get('name', 'Unknown'))
        
        # Generate statistics text
        stats = f"""📊 SPEAKER DATABASE STATISTICS
{'='*40}

Total Speakers: {total_speakers}
├─ 🤖 Auto-generated: {auto_speakers}
├─ ✅ User-corrected: {corrected_speakers}
├─ ✅ Enhanced: {enhanced_speakers}
└─ 👤 Enrolled: {enrolled_speakers}

Training Data:
├─ Total audio samples: {total_embeddings:,}
├─ Average confidence: {avg_confidence:.3f}
└─ Most active speaker: {most_active_name}
   ({most_active['embedding_count']} samples)

Recent Activity:
├─ Active speakers (24h): {self.count_recent_speakers(24)}
├─ Active speakers (7d): {self.count_recent_speakers(168)}
└─ Last database update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Quality Metrics:
├─ High confidence (>0.8): {self.count_high_confidence()}
├─ Medium confidence (0.5-0.8): {self.count_medium_confidence()}
└─ Low confidence (<0.5): {self.count_low_confidence()}

Storage:
├─ Database size: {self.get_database_size()}
└─ Average samples per speaker: {total_embeddings/total_speakers:.1f}
"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats)
    
    def count_recent_speakers(self, hours):
        """Count speakers active in the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        count = 0
        for speaker in self.speaker_data.values():
            try:
                last_seen = datetime.fromisoformat(speaker['last_seen'])
                if last_seen > cutoff:
                    count += 1
            except:
                pass
        return count
    
    def count_high_confidence(self):
        """Count speakers with high confidence"""
        return sum(1 for s in self.speaker_data.values() 
                  if s['average_confidence'] and s['average_confidence'] > 0.8)
    
    def count_medium_confidence(self):
        """Count speakers with medium confidence"""
        return sum(1 for s in self.speaker_data.values() 
                  if s['average_confidence'] and 0.5 <= s['average_confidence'] <= 0.8)
    
    def count_low_confidence(self):
        """Count speakers with low confidence"""
        return sum(1 for s in self.speaker_data.values() 
                  if s['average_confidence'] and s['average_confidence'] < 0.5)
    
    def get_database_size(self):
        """Get database file size"""
        try:
            size_bytes = self.db_path.stat().st_size
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.1f} KB"
            else:
                return f"{size_bytes/1024**2:.1f} MB"
        except:
            return "Unknown"
    
    def on_speaker_select(self, event):
        """Handle speaker selection in the list"""
        selection = self.speaker_tree.selection()
        if not selection:
            return
        
        item = self.speaker_tree.item(selection[0])
        speaker_name = item['text']
        
        # Find speaker by name (check both display_name and name for compatibility)
        for speaker_id, data in self.speaker_data.items():
            display_name = data.get('display_name', data.get('name', 'Unknown'))
            if display_name == speaker_name:
                self.selected_speaker = speaker_id
                self.load_speaker_embeddings(speaker_id)
                self.update_speaker_details()
                break
    
    def update_speaker_details(self):
        """Update the speaker details tab"""
        if not self.selected_speaker or self.selected_speaker not in self.speaker_data:
            return
        
        data = self.speaker_data[self.selected_speaker]
        
        # Update details text
        display_name = data.get('display_name', data.get('name', 'Unknown'))
        details = f"""🎤 SPEAKER PROFILE: {display_name}
{'='*60}

Basic Information:
├─ Speaker ID: {data['speaker_id']}
├─ Type: {data['type']}
├─ Created: {data['created_date']}
└─ Last Seen: {data['last_seen']}

Training Statistics:
├─ Audio samples: {data['embedding_count']}
├─ Total audio time: {data['total_audio_seconds']:.1f} seconds
├─ Session count: {data['session_count']}
└─ Average confidence: {data['average_confidence']:.3f}

Learning Progress:
├─ First sample: {self.get_first_sample_date(data)}
├─ Latest sample: {self.get_latest_sample_date(data)}
├─ Learning span: {self.get_learning_span(data)}
└─ Samples per day: {self.get_samples_per_day(data)}

Quality Assessment:
├─ Recognition quality: {self.assess_recognition_quality(data)}
├─ Confidence trend: {self.get_confidence_trend(data)}
└─ Training completeness: {self.assess_training_completeness(data)}
"""
        
        self.details_info.delete(1.0, tk.END)
        self.details_info.insert(tk.END, details)
        
        # Update charts
        self.update_speaker_charts()
    
    def update_speaker_charts(self):
        """Update charts for the selected speaker"""
        if not self.selected_speaker or 'embeddings' not in self.speaker_data[self.selected_speaker]:
            return
        
        embeddings = self.speaker_data[self.selected_speaker]['embeddings']
        if not embeddings:
            return
        
        # Clear previous plots
        self.fig.clear()
        
        # Create subplots
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax3 = self.fig.add_subplot(2, 2, 3)
        ax4 = self.fig.add_subplot(2, 2, 4)
        
        # Extract data
        timestamps = [e['timestamp'] for e in embeddings]
        confidences = [e['confidence'] for e in embeddings]
        audio_lengths = [e['audio_length'] for e in embeddings]
        
        # Plot 1: Confidence over time
        ax1.plot(timestamps, confidences, 'b-o', markersize=3)
        ax1.set_title('Confidence Over Time')
        ax1.set_ylabel('Confidence')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Confidence distribution
        ax2.hist(confidences, bins=20, alpha=0.7, color='green')
        ax2.set_title('Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Audio length distribution
        ax3.hist(audio_lengths, bins=15, alpha=0.7, color='orange')
        ax3.set_title('Audio Length Distribution')
        ax3.set_xlabel('Audio Length (seconds)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning progress (cumulative samples)
        cumulative_samples = list(range(1, len(embeddings) + 1))
        ax4.plot(timestamps, cumulative_samples, 'r-', linewidth=2)
        ax4.set_title('Learning Progress (Cumulative Samples)')
        ax4.set_ylabel('Total Samples')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # Adjust layout and refresh
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_analytics(self):
        """Update the learning analytics tab"""
        # Clear previous plots
        self.analytics_fig.clear()
        
        # Create subplots for system-wide analytics
        ax1 = self.analytics_fig.add_subplot(2, 3, 1)
        ax2 = self.analytics_fig.add_subplot(2, 3, 2)
        ax3 = self.analytics_fig.add_subplot(2, 3, 3)
        ax4 = self.analytics_fig.add_subplot(2, 3, 4)
        ax5 = self.analytics_fig.add_subplot(2, 3, 5)
        ax6 = self.analytics_fig.add_subplot(2, 3, 6)
        
        # Analytics data preparation
        if not self.speaker_data:
            return
        
        # 1. Speaker type distribution
        type_counts = {}
        for speaker in self.speaker_data.values():
            speaker_type = speaker['type'].split()[1] if ' ' in speaker['type'] else speaker['type']
            type_counts[speaker_type] = type_counts.get(speaker_type, 0) + 1
        
        ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        ax1.set_title('Speaker Type Distribution')
        
        # 2. Confidence distribution
        confidences = [s['average_confidence'] for s in self.speaker_data.values() 
                      if s['average_confidence'] is not None]
        if confidences:
            ax2.hist(confidences, bins=15, alpha=0.7, color='blue')
            ax2.set_title('System Confidence Distribution')
            ax2.set_xlabel('Average Confidence')
            ax2.set_ylabel('Number of Speakers')
            ax2.grid(True, alpha=0.3)
        
        # 3. Embedding count distribution
        embedding_counts = [s['embedding_count'] for s in self.speaker_data.values()]
        if embedding_counts:
            ax3.hist(embedding_counts, bins=20, alpha=0.7, color='green')
            ax3.set_title('Training Data Distribution')
            ax3.set_xlabel('Number of Samples')
            ax3.set_ylabel('Number of Speakers')
            ax3.grid(True, alpha=0.3)
        
        # 4. Top speakers by sample count
        top_speakers = sorted(self.speaker_data.items(), 
                            key=lambda x: x[1]['embedding_count'], 
                            reverse=True)[:10]
        
        names = [s[1].get('display_name', s[1].get('name', 'Unknown'))[:15] + '...' 
                 if len(s[1].get('display_name', s[1].get('name', 'Unknown'))) > 15 
                 else s[1].get('display_name', s[1].get('name', 'Unknown')) 
                 for s in top_speakers]
        counts = [s[1]['embedding_count'] for s in top_speakers]
        
        if names and counts:
            ax4.barh(names, counts, color='purple', alpha=0.7)
            ax4.set_title('Top 10 Most Trained Speakers')
            ax4.set_xlabel('Number of Samples')
        
        # 5. Confidence vs Sample count scatter
        x_data = [s['embedding_count'] for s in self.speaker_data.values()]
        y_data = [s['average_confidence'] for s in self.speaker_data.values() 
                 if s['average_confidence'] is not None]
        
        if len(x_data) == len(y_data) and x_data:
            ax5.scatter(x_data, y_data, alpha=0.6, color='red')
            ax5.set_title('Confidence vs Training Data')
            ax5.set_xlabel('Number of Samples')
            ax5.set_ylabel('Average Confidence')
            ax5.grid(True, alpha=0.3)
        
        # 6. Recent activity timeline
        self.plot_recent_activity(ax6)
        
        # Adjust layout and refresh
        self.analytics_fig.tight_layout()
        self.analytics_canvas.draw()
    
    def plot_recent_activity(self, ax):
        """Plot recent speaker activity timeline"""
        # Get speakers with last_seen data
        recent_speakers = []
        for speaker in self.speaker_data.values():
            try:
                last_seen = datetime.fromisoformat(speaker['last_seen'])
                display_name = speaker.get('display_name', speaker.get('name', 'Unknown'))
                recent_speakers.append((last_seen, display_name))
            except:
                pass
        
        if not recent_speakers:
            ax.text(0.5, 0.5, 'No recent activity data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Recent Activity Timeline')
            return
        
        # Sort by timestamp
        recent_speakers.sort()
        
        # Take last 20 speakers
        recent_speakers = recent_speakers[-20:]
        
        timestamps = [s[0] for s in recent_speakers]
        names = [s[1][:10] + '...' if len(s[1]) > 10 else s[1] for s in recent_speakers]
        
        y_pos = range(len(names))
        
        ax.barh(y_pos, [1] * len(names), color='lightblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_title('Recent Speaker Activity')
        ax.set_xlabel('Timeline')
    
    # Helper methods for speaker details
    def get_first_sample_date(self, data):
        if 'embeddings' in data and data['embeddings']:
            return data['embeddings'][0]['timestamp'].strftime('%Y-%m-%d %H:%M')
        return "No samples"
    
    def get_latest_sample_date(self, data):
        if 'embeddings' in data and data['embeddings']:
            return data['embeddings'][-1]['timestamp'].strftime('%Y-%m-%d %H:%M')
        return "No samples"
    
    def get_learning_span(self, data):
        if 'embeddings' in data and len(data['embeddings']) > 1:
            first = data['embeddings'][0]['timestamp']
            last = data['embeddings'][-1]['timestamp']
            span = last - first
            return f"{span.days} days, {span.seconds//3600} hours"
        return "Single session"
    
    def get_samples_per_day(self, data):
        if 'embeddings' in data and len(data['embeddings']) > 1:
            first = data['embeddings'][0]['timestamp']
            last = data['embeddings'][-1]['timestamp']
            span = (last - first).total_seconds() / 86400  # days
            if span > 0:
                return f"{len(data['embeddings'])/span:.1f}"
        return "N/A"
    
    def assess_recognition_quality(self, data):
        conf = data['average_confidence']
        if conf is None:
            return "Unknown"
        elif conf > 0.8:
            return "Excellent ⭐⭐⭐"
        elif conf > 0.6:
            return "Good ⭐⭐"
        elif conf > 0.4:
            return "Fair ⭐"
        else:
            return "Poor ❌"
    
    def get_confidence_trend(self, data):
        if 'embeddings' not in data or len(data['embeddings']) < 3:
            return "Insufficient data"
        
        confidences = [e['confidence'] for e in data['embeddings']]
        recent_avg = np.mean(confidences[-5:])  # Last 5 samples
        early_avg = np.mean(confidences[:5])    # First 5 samples
        
        if recent_avg > early_avg + 0.1:
            return "Improving ↗️"
        elif recent_avg < early_avg - 0.1:
            return "Declining ↘️"
        else:
            return "Stable ➡️"
    
    def assess_training_completeness(self, data):
        count = data['embedding_count']
        if count > 50:
            return "Well-trained ✅"
        elif count > 20:
            return "Moderately trained ⚠️"
        elif count > 5:
            return "Lightly trained ⚠️"
        else:
            return "Insufficient training ❌"
    
    # Management functions
    def cleanup_speakers(self):
        """Clean up empty or low-quality speakers"""
        if messagebox.askyesno("Confirm Cleanup", 
                              "This will remove speakers with fewer than 3 samples. Continue?"):
            try:
                response = requests.post(f"{self.backend_url}/speakers/cleanup")
                if response.status_code == 200:
                    result = response.json()
                    self.log_management(f"Cleanup completed: {result}")
                    self.refresh_data()
                else:
                    self.log_management(f"Cleanup failed: {response.status_code}")
            except Exception as e:
                self.log_management(f"Cleanup error: {str(e)}")
    
    def generate_report(self):
        """Generate a comprehensive report"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Speaker Analytics Report"
        )
        
        if filename:
            try:
                report = self.create_comprehensive_report()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.log_management(f"Report saved to: {filename}")
                messagebox.showinfo("Success", f"Report saved to {filename}")
            except Exception as e:
                self.log_management(f"Report generation failed: {str(e)}")
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")
    
    def export_database(self):
        """Export database to JSON"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Speaker Database"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.speaker_data, f, indent=2, default=str)
                self.log_management(f"Database exported to: {filename}")
                messagebox.showinfo("Success", f"Database exported to {filename}")
            except Exception as e:
                self.log_management(f"Export failed: {str(e)}")
                messagebox.showerror("Error", f"Failed to export database: {str(e)}")
    
    def export_analytics(self):
        """Export analytics data to CSV"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Analytics Data"
        )
        
        if filename:
            try:
                # Create DataFrame from speaker data
                df_data = []
                for speaker_id, data in self.speaker_data.items():
                    df_data.append({
                        'speaker_id': speaker_id,
                        'name': data.get('display_name', data.get('name', 'Unknown')),
                        'type': data['type'],
                        'embedding_count': data['embedding_count'],
                        'average_confidence': data['average_confidence'],
                        'total_audio_seconds': data.get('total_audio_seconds', 0.0),
                        'created_date': data['created_date'],
                        'last_seen': data['last_seen']
                    })
                
                df = pd.DataFrame(df_data)
                df.to_csv(filename, index=False)
                
                self.log_management(f"Analytics exported to: {filename}")
                messagebox.showinfo("Success", f"Analytics data exported to {filename}")
            except Exception as e:
                self.log_management(f"Analytics export failed: {str(e)}")
                messagebox.showerror("Error", f"Failed to export analytics: {str(e)}")
    
    def reset_database(self):
        """Reset the speaker database"""
        if messagebox.askyesno("Confirm Reset", 
                              "⚠️ This will permanently delete ALL speaker data. Are you sure?"):
            if messagebox.askyesno("Final Confirmation", 
                                  "This action CANNOT be undone. Proceed with database reset?"):
                try:
                    response = requests.post(f"{self.backend_url}/speakers/reset")
                    if response.status_code == 200:
                        self.log_management("Database reset completed")
                        self.refresh_data()
                        messagebox.showinfo("Success", "Database has been reset")
                    else:
                        self.log_management(f"Reset failed: {response.status_code}")
                except Exception as e:
                    self.log_management(f"Reset error: {str(e)}")
    
    def create_comprehensive_report(self):
        """Create a comprehensive analytics report"""
        report = f"""OREJA SPEAKER ANALYTICS REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
{'='*60}
Total Speakers: {len(self.speaker_data)}
Total Audio Samples: {sum(s['embedding_count'] for s in self.speaker_data.values())}
Average Confidence: {np.mean([s['average_confidence'] for s in self.speaker_data.values() if s['average_confidence'] is not None]):.3f}

SPEAKER BREAKDOWN
{'='*60}
"""
        
        # Add detailed speaker information
        for speaker_id, data in sorted(self.speaker_data.items(), 
                                     key=lambda x: x[1]['embedding_count'], 
                                     reverse=True):
            report += f"""
Speaker: {data.get('display_name', data.get('name', 'Unknown'))}
├─ ID: {data['speaker_id']}
├─ Type: {data['type']}
├─ Samples: {data['embedding_count']}
├─ Confidence: {data['average_confidence']:.3f}
├─ Audio Time: {data.get('total_audio_seconds', 0.0):.1f}s
└─ Last Seen: {data['last_seen']}
"""
        
        return report
    
    def log_management(self, message):
        """Log a management action"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        self.mgmt_log.insert(tk.END, log_entry)
        self.mgmt_log.see(tk.END)
    
    def update_status(self, message):
        """Update the status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nApplication closed by user")

    def show_batch_help(self):
        """Show help for batch processing"""
        help_text = """
🎬 Batch Processing Help

This feature allows you to transcribe recorded calls using your existing speaker embeddings:

📁 Adding Files:
• Use "Add Files" to select individual audio files
• Use "Add Folder" to add all audio files from a directory  
• Supported formats: WAV, MP3, FLAC, M4A, OGG

⚙️ Settings:
• Output Directory: Where results will be saved
• Improve Speaker Models: Use recordings to enhance recognition
• Speaker Mapping: JSON file to map auto-detected speakers to known names
• 🔒 Privacy Mode: Anonymize speaker IDs for privacy protection

🔒 Privacy Mode:
• When enabled, speaker IDs are replaced with anonymous labels (Speaker_A, Speaker_B, etc.)
• Speaker model improvements are disabled to protect user data
• Confidence scores and detailed metadata are hidden in output files
• Use this for sensitive recordings where speaker identity must be protected

🚀 Processing:
• The system will:
  1. Transcribe each recording using Whisper
  2. Identify speakers using your existing embeddings (unless in privacy mode)
  3. Enhance speaker identification accuracy (unless in privacy mode)
  4. Optionally improve speaker models with high-confidence segments (disabled in privacy mode)

📊 Results:
• View processing results in the Results tab
• Check detailed logs in the Processing Log tab
• Find output files in your specified directory
• Privacy mode status is clearly indicated in all output files

💡 Tips:
• For best results, ensure your speaker embeddings are well-trained
• Use speaker mapping to assign meaningful names to speakers
• Enable "Improve Speaker Models" to enhance future recognition
• Use Privacy Mode for sensitive recordings where anonymity is required
• Privacy mode and speaker mapping are mutually exclusive features
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Batch Processing Help")
        help_window.geometry("700x600")
        
        help_text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        help_text_widget.pack(fill=tk.BOTH, expand=True)
        help_text_widget.insert(tk.END, help_text)
        help_text_widget.config(state=tk.DISABLED)
        
        ttk.Button(help_window, text="Close", 
                  command=help_window.destroy).pack(pady=10)
    
    def add_batch_files(self):
        """Add individual audio files to batch"""
        filetypes = [
            ("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("FLAC Files", "*.flac"),
            ("M4A Files", "*.m4a"),
            ("OGG Files", "*.ogg"),
            ("All Files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=filetypes
        )
        
        for file_path in files:
            self.add_file_to_batch(Path(file_path))
    
    def add_batch_folder(self):
        """Add all audio files from a folder"""
        folder = filedialog.askdirectory(title="Select Folder with Audio Files")
        if folder:
            folder_path = Path(folder)
            extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
            
            audio_files = []
            for ext in extensions:
                audio_files.extend(folder_path.glob(f"*{ext}"))
                audio_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
            for file_path in audio_files:
                self.add_file_to_batch(file_path)
            
            self.log_batch(f"Added {len(audio_files)} files from {folder}")
    
    def add_file_to_batch(self, file_path: Path):
        """Add a single file to the batch processing list"""
        if file_path in self.batch_files:
            return  # Already added
        
        try:
            # Get file info
            file_size = file_path.stat().st_size
            size_str = self.format_file_size(file_size)
            
            # Try to get duration (this is basic, could be enhanced)
            duration_str = "Unknown"
            try:
                import torchaudio
                info = torchaudio.info(str(file_path))
                duration = info.num_frames / info.sample_rate
                duration_str = f"{duration:.1f}s"
            except:
                pass
            
            # Add to list and tree
            self.batch_files.append(file_path)
            self.batch_file_tree.insert('', tk.END, 
                                       text=file_path.name,
                                       values=(size_str, duration_str, "Pending"))
            
            self.log_batch(f"Added file: {file_path.name}")
            
        except Exception as e:
            self.log_batch(f"Error adding {file_path.name}: {e}")
    
    def clear_batch_files(self):
        """Clear all files from batch processing list"""
        self.batch_files.clear()
        for item in self.batch_file_tree.get_children():
            self.batch_file_tree.delete(item)
        self.log_batch("Cleared all files from batch")
    
    def select_output_directory(self):
        """Select output directory for batch results"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.batch_output_var.set(directory)
    
    def select_speaker_mapping(self):
        """Select speaker mapping JSON file"""
        file_path = filedialog.askopenfilename(
            title="Select Speaker Mapping File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.speaker_mapping_var.set(file_path)
    
    def create_speaker_mapping(self):
        """Create a new speaker mapping file"""
        mapping_window = tk.Toplevel(self.root)
        mapping_window.title("Create Speaker Mapping")
        mapping_window.geometry("500x400")
        
        ttk.Label(mapping_window, text="Speaker Name Mapping", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        ttk.Label(mapping_window, text="Map auto-detected speaker IDs to meaningful names:").pack(pady=5)
        
        # Mapping entries frame
        mapping_frame = ttk.Frame(mapping_window)
        mapping_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Headers
        ttk.Label(mapping_frame, text="Auto-detected ID", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(mapping_frame, text="Friendly Name", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=5)
        
        # Add some example rows
        self.mapping_entries = []
        for i in range(10):
            id_entry = ttk.Entry(mapping_frame, width=25)
            id_entry.grid(row=i+1, column=0, padx=5, pady=2)
            
            name_entry = ttk.Entry(mapping_frame, width=25)
            name_entry.grid(row=i+1, column=1, padx=5, pady=2)
            
            self.mapping_entries.append((id_entry, name_entry))
        
        # Buttons
        button_frame = ttk.Frame(mapping_window)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Save Mapping", 
                  command=lambda: self.save_speaker_mapping(mapping_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=mapping_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def save_speaker_mapping(self, window):
        """Save the speaker mapping to a JSON file"""
        mapping = {}
        for id_entry, name_entry in self.mapping_entries:
            speaker_id = id_entry.get().strip()
            speaker_name = name_entry.get().strip()
            if speaker_id and speaker_name:
                mapping[speaker_id] = speaker_name
        
        if not mapping:
            messagebox.showwarning("Warning", "No mappings entered!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Speaker Mapping",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, indent=2, ensure_ascii=False)
                
                self.speaker_mapping_var.set(file_path)
                self.log_batch(f"Saved speaker mapping to {file_path}")
                messagebox.showinfo("Success", f"Speaker mapping saved to {file_path}")
                window.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mapping: {e}")
    
    def start_batch_processing(self):
        """Start batch processing of audio files"""
        if not self.batch_files:
            messagebox.showwarning("Warning", "No audio files selected!")
            return
        
        # Prepare processing parameters
        output_dir = Path(self.batch_output_var.get())
        improve_speakers = self.improve_speakers_var.get()
        privacy_mode = self.batch_privacy_mode_var.get()  # Get privacy mode setting
        
        # Load speaker mapping if provided
        speaker_mapping = None
        mapping_file = self.speaker_mapping_var.get()
        if mapping_file and Path(mapping_file).exists():
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    speaker_mapping = json.load(f)
                self.log_batch(f"Loaded speaker mapping with {len(speaker_mapping)} entries")
            except Exception as e:
                self.log_batch(f"Error loading speaker mapping: {e}")
        
        # Log privacy mode status
        if privacy_mode:
            self.log_batch("Privacy mode ENABLED: Speaker IDs will be anonymized")
        else:
            self.log_batch("Privacy mode disabled: Original speaker IDs will be preserved")
        
        # Initialize processor
        try:
            from batch_transcription import BatchTranscriptionProcessor
            self.batch_processor = BatchTranscriptionProcessor(self.backend_url)
            
            # Update UI
            self.batch_process_btn.config(state=tk.DISABLED)
            self.batch_stop_btn.config(state=tk.NORMAL)
            self.batch_progress['value'] = 0
            self.batch_progress['maximum'] = 100
            self.batch_progress_label.config(text="Starting...")
            
            # Clear previous results
            for item in self.batch_results_tree.get_children():
                self.batch_results_tree.delete(item)
            
            # Start processing in separate thread
            self.processing_thread = threading.Thread(
                target=self.run_batch_processing,
                args=(output_dir, improve_speakers, speaker_mapping, privacy_mode),  # Pass privacy mode
                daemon=True
            )
            self.processing_thread.start()
            
            self.log_batch(f"Started batch processing of {len(self.batch_files)} files")
            
        except Exception as e:
            self.log_batch(f"Error starting batch processing: {e}")
            messagebox.showerror("Error", f"Failed to start processing: {e}")
            self.reset_batch_ui()
    
    def run_batch_processing(self, output_dir: Path, improve_speakers: bool, speaker_mapping: dict, privacy_mode: bool):
        """Run batch processing in background thread"""
        try:
            total_files = len(self.batch_files)
            
            for i, audio_file in enumerate(self.batch_files):
                if not hasattr(self, 'batch_processor') or self.batch_processor is None:
                    break  # Processing was stopped
                
                # Update overall file progress
                file_progress = (i / total_files) * 100
                self.root.after(0, lambda prog=file_progress, f=audio_file, idx=i: 
                    self.update_batch_progress(prog, f, idx, total_files))
                
                # Process file with detailed progress tracking
                try:
                    # Create progress callback for this file
                    def progress_callback(stage: str, progress: float):
                        # Calculate overall progress: file progress + current file internal progress
                        overall_progress = (i / total_files) * 100 + (progress / total_files)
                        self.root.after(0, lambda: self.update_transcription_progress(
                            overall_progress, audio_file, stage, i, total_files))
                    
                    # Try the new progress method first, fallback to old method
                    if hasattr(self.batch_processor, 'process_recording_with_progress'):
                        result = self.batch_processor.process_recording_with_progress(
                            audio_file, output_dir, improve_speakers, speaker_mapping, 
                            privacy_mode, progress_callback
                        )
                    else:
                        # Fallback to old method without progress
                        result = self.batch_processor.process_recording(
                            audio_file, output_dir, improve_speakers, speaker_mapping, privacy_mode
                        )
                    
                    # Update results in UI thread
                    self.root.after(0, lambda r=result, f=audio_file: self.add_batch_result(r, f))
                    
                except Exception as e:
                    error_msg = f"Error processing {audio_file.name}: {e}"
                    self.root.after(0, lambda msg=error_msg: self.log_batch(msg))
            
            # Processing complete
            self.root.after(0, self.finish_batch_processing)
            
        except Exception as e:
            error_msg = f"Batch processing failed: {e}"
            self.root.after(0, lambda: self.log_batch(error_msg))
            self.root.after(0, self.reset_batch_ui)
    
    def update_batch_progress(self, progress: float, current_file: Path, file_index: int, total_files: int):
        """Update progress bar and label for file-level progress"""
        self.batch_progress['value'] = progress
        self.batch_progress_label.config(text=f"Processing {current_file.name} ({file_index + 1}/{total_files})")
        
        # Update file status in tree
        for item in self.batch_file_tree.get_children():
            if self.batch_file_tree.item(item)['text'] == current_file.name:
                values = list(self.batch_file_tree.item(item)['values'])
                values[2] = "Processing..."  # Status column
                self.batch_file_tree.item(item, values=values)
                break
    
    def update_transcription_progress(self, overall_progress: float, current_file: Path, stage: str, file_index: int, total_files: int):
        """Update progress bar with detailed transcription progress"""
        self.batch_progress['value'] = overall_progress
        stage_text = {
            'loading': 'Loading audio',
            'transcribing': 'Transcribing audio',
            'diarizing': 'Identifying speakers',
            'enhancing': 'Enhancing speakers',
            'improving': 'Improving models',
            'saving': 'Saving results'
        }.get(stage, stage)
        
        self.batch_progress_label.config(
            text=f"{stage_text}: {current_file.name} ({file_index + 1}/{total_files}) - {overall_progress:.1f}%"
        )
    
    def add_batch_result(self, result: dict, audio_file: Path):
        """Add processing result to results tree"""
        try:
            if 'error' in result:
                status = "Error"
                speakers = "-"
                segments = "-"
                confidence = "-"
                processing_time = "-"
                self.log_batch(f"Error in {audio_file.name}: {result['error']}")
            else:
                status = "Success"
                enhancement_info = result.get('enhancement_info', {})
                segments_info = result.get('segments', [])
                
                # Get unique speakers
                unique_speakers = set()
                total_confidence = 0
                confident_segments = 0
                
                for segment in segments_info:
                    speaker = segment.get('speaker', 'Unknown')
                    if speaker != 'Unknown':
                        unique_speakers.add(speaker)
                    
                    conf = segment.get('speaker_confidence', 0)
                    if conf > 0:
                        total_confidence += conf
                        confident_segments += 1
                
                speakers = str(len(unique_speakers))
                segments = str(len(segments_info))
                
                if confident_segments > 0:
                    avg_confidence = total_confidence / confident_segments
                    confidence = f"{avg_confidence:.2f}"
                else:
                    confidence = "N/A"
                
                processing_time = result.get('processing_time', 'N/A')
                
                self.log_batch(f"Successfully processed {audio_file.name}: {speakers} speakers, {segments} segments")
            
            # Add to results tree
            self.batch_results_tree.insert('', tk.END, values=(
                audio_file.name, status, speakers, segments, confidence, processing_time
            ))
            
            # Update file status in files tree
            for item in self.batch_file_tree.get_children():
                if self.batch_file_tree.item(item)['text'] == audio_file.name:
                    values = list(self.batch_file_tree.item(item)['values'])
                    values[2] = status
                    self.batch_file_tree.item(item, values=values)
                    break
            
        except Exception as e:
            self.log_batch(f"Error updating results for {audio_file.name}: {e}")
    
    def finish_batch_processing(self):
        """Finish batch processing and update UI"""
        self.batch_progress_label.config(text="Complete!")
        self.log_batch("Batch processing completed")
        
        # Show completion message
        successful = len([item for item in self.batch_results_tree.get_children() 
                         if self.batch_results_tree.item(item)['values'][1] == "Success"])
        total = len(self.batch_files)
        
        completion_msg = f"Batch processing complete!\nProcessed {successful}/{total} files successfully."
        if successful < total:
            completion_msg += f"\n{total - successful} files failed to process."
        
        messagebox.showinfo("Batch Processing Complete", completion_msg)
        
        self.reset_batch_ui()
    
    def stop_batch_processing(self):
        """Stop batch processing"""
        if hasattr(self, 'batch_processor'):
            self.batch_processor = None
        
        self.log_batch("Batch processing stopped by user")
        self.reset_batch_ui()
    
    def reset_batch_ui(self):
        """Reset batch processing UI to initial state"""
        self.batch_process_btn.config(state=tk.NORMAL)
        self.batch_stop_btn.config(state=tk.DISABLED)
        self.batch_progress['value'] = 0
        self.batch_progress_label.config(text="Ready")
    
    def log_batch(self, message: str):
        """Add message to batch processing log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        self.batch_log.insert(tk.END, log_entry)
        self.batch_log.see(tk.END)
        
        # Also log to console
        print(f"BATCH: {message}")
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/1024**2:.1f} MB"
        else:
            return f"{size_bytes/1024**3:.1f} GB"

    def setup_summary_results_tab(self, parent_notebook):
        """Setup the summary results display tab"""
        summary_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(summary_frame, text="📋 Summaries")
        
        # Create paned window for different summary types
        summary_paned = ttk.PanedWindow(summary_frame, orient=tk.VERTICAL)
        summary_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Overall Summary
        overall_frame = ttk.LabelFrame(summary_paned, text="Overall Summary", padding=5)
        summary_paned.add(overall_frame, weight=1)
        
        self.overall_summary_text = tk.Text(overall_frame, height=8, font=('Consolas', 10))
        overall_scroll = ttk.Scrollbar(overall_frame, orient=tk.VERTICAL, command=self.overall_summary_text.yview)
        self.overall_summary_text.configure(yscrollcommand=overall_scroll.set)
        self.overall_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        overall_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Speaker Summaries
        speaker_frame = ttk.LabelFrame(summary_paned, text="Speaker Summaries", padding=5)
        summary_paned.add(speaker_frame, weight=1)
        
        self.speaker_summary_text = tk.Text(speaker_frame, height=8, font=('Consolas', 10))
        speaker_scroll = ttk.Scrollbar(speaker_frame, orient=tk.VERTICAL, command=self.speaker_summary_text.yview)
        self.speaker_summary_text.configure(yscrollcommand=speaker_scroll.set)
        self.speaker_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        speaker_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Key Points
        keypoints_frame = ttk.LabelFrame(summary_paned, text="Key Points", padding=5)
        summary_paned.add(keypoints_frame, weight=1)
        
        self.keypoints_text = tk.Text(keypoints_frame, height=8, font=('Consolas', 10))
        keypoints_scroll = ttk.Scrollbar(keypoints_frame, orient=tk.VERTICAL, command=self.keypoints_text.yview)
        self.keypoints_text.configure(yscrollcommand=keypoints_scroll.set)
        self.keypoints_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        keypoints_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_annotation_results_tab(self, parent_notebook):
        """Setup the annotation results display tab"""
        annotation_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(annotation_frame, text="🏷️ Annotations")
        
        # Create notebook for different annotation types
        ann_notebook = ttk.Notebook(annotation_frame)
        ann_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Action Items Tab
        actions_frame = ttk.Frame(ann_notebook)
        ann_notebook.add(actions_frame, text="✅ Action Items")
        
        # Create treeview for action items
        self.actions_tree = ttk.Treeview(actions_frame, columns=('speaker', 'timestamp', 'confidence'), 
                                        show='tree headings', height=12)
        self.actions_tree.heading('#0', text='Action Item')
        self.actions_tree.heading('speaker', text='Speaker')
        self.actions_tree.heading('timestamp', text='Time')
        self.actions_tree.heading('confidence', text='Confidence')
        
        self.actions_tree.column('#0', width=400)
        self.actions_tree.column('speaker', width=100)
        self.actions_tree.column('timestamp', width=80)
        self.actions_tree.column('confidence', width=80)
        
        actions_scroll = ttk.Scrollbar(actions_frame, orient=tk.VERTICAL, command=self.actions_tree.yview)
        self.actions_tree.configure(yscrollcommand=actions_scroll.set)
        self.actions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        actions_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Q&A Pairs Tab
        qa_frame = ttk.Frame(ann_notebook)
        ann_notebook.add(qa_frame, text="❓ Q&A Pairs")
        
        self.qa_text = tk.Text(qa_frame, font=('Consolas', 10))
        qa_scroll = ttk.Scrollbar(qa_frame, orient=tk.VERTICAL, command=self.qa_text.yview)
        self.qa_text.configure(yscrollcommand=qa_scroll.set)
        self.qa_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        qa_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Topics Tab
        topics_frame = ttk.Frame(ann_notebook)
        ann_notebook.add(topics_frame, text="🏷️ Topics")
        
        self.topics_tree = ttk.Treeview(topics_frame, columns=('mentions', 'first_mentioned'), 
                                       show='tree headings', height=12)
        self.topics_tree.heading('#0', text='Topic')
        self.topics_tree.heading('mentions', text='Mentions')
        self.topics_tree.heading('first_mentioned', text='First Mentioned')
        
        self.topics_tree.column('#0', width=300)
        self.topics_tree.column('mentions', width=100)
        self.topics_tree.column('first_mentioned', width=120)
        
        topics_scroll = ttk.Scrollbar(topics_frame, orient=tk.VERTICAL, command=self.topics_tree.yview)
        self.topics_tree.configure(yscrollcommand=topics_scroll.set)
        self.topics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        topics_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Decisions Tab
        decisions_frame = ttk.Frame(ann_notebook)
        ann_notebook.add(decisions_frame, text="🎯 Decisions")
        
        self.decisions_text = tk.Text(decisions_frame, font=('Consolas', 10))
        decisions_scroll = ttk.Scrollbar(decisions_frame, orient=tk.VERTICAL, command=self.decisions_text.yview)
        self.decisions_text.configure(yscrollcommand=decisions_scroll.set)
        self.decisions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        decisions_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_export_tab(self, parent_notebook):
        """Setup the export options tab"""
        export_frame = ttk.Frame(parent_notebook)
        parent_notebook.add(export_frame, text="💾 Export")
        
        # Privacy mode indicator
        privacy_indicator_frame = ttk.Frame(export_frame)
        privacy_indicator_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.export_privacy_label = ttk.Label(privacy_indicator_frame, text="", font=('Arial', 10, 'bold'))
        self.export_privacy_label.pack()
        
        # Export options
        export_options_frame = ttk.LabelFrame(export_frame, text="Export Options", padding=10)
        export_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Export format selection
        format_frame = ttk.Frame(export_options_frame)
        format_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(format_frame, text="Export Format:").pack(side=tk.LEFT, padx=(0, 10))
        self.export_format = tk.StringVar(value="json")
        
        # Update radio buttons based on privacy mode
        self.json_radio = ttk.Radiobutton(format_frame, text="JSON (Analytics Only)", variable=self.export_format, value="json")
        self.json_radio.pack(side=tk.LEFT, padx=5)
        
        self.text_radio = ttk.Radiobutton(format_frame, text="Text Report", variable=self.export_format, value="text")
        self.text_radio.pack(side=tk.LEFT, padx=5)
        
        self.minutes_radio = ttk.Radiobutton(format_frame, text="Meeting Minutes", variable=self.export_format, value="minutes")
        self.minutes_radio.pack(side=tk.LEFT, padx=5)
        
        self.actions_radio = ttk.Radiobutton(format_frame, text="Action Items Only", variable=self.export_format, value="actions")
        self.actions_radio.pack(side=tk.LEFT, padx=5)
        
        # Export buttons
        export_buttons_frame = ttk.Frame(export_options_frame)
        export_buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(export_buttons_frame, text="📋 Export Summary", 
                  command=self.export_summary).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_buttons_frame, text="🏷️ Export Annotations", 
                  command=self.export_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_buttons_frame, text="📦 Export All", 
                  command=self.export_all).pack(side=tk.LEFT, padx=5)
        
        # Export preview
        preview_frame = ttk.LabelFrame(export_frame, text="Export Preview", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.export_preview_text = tk.Text(preview_frame, font=('Consolas', 10))
        preview_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.export_preview_text.yview)
        self.export_preview_text.configure(yscrollcommand=preview_scroll.set)
        self.export_preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def select_transcription_file(self):
        """Select transcription file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Transcription File",
            filetypes=[
                ("JSON Files", "*.json"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.conv_file_var.set(file_path)
    
    # Enhanced speaker management methods
    
    def filter_speakers(self, event=None):
        """Filter speakers based on search term"""
        search_term = self.search_var.get().lower()
        self.update_management_speaker_list(filter_term=search_term)
    
    def update_management_speaker_list(self, filter_term=""):
        """Update the management speaker list with optional filtering"""
        if not hasattr(self, 'mgmt_speaker_tree'):
            return
            
        # Clear existing items
        for item in self.mgmt_speaker_tree.get_children():
            self.mgmt_speaker_tree.delete(item)
        
        if not hasattr(self, 'speaker_data'):
            return
        
        # Sort speakers by name for easier management
        sorted_speakers = sorted(self.speaker_data.items(), 
                               key=lambda x: x[1].get('display_name', x[1].get('name', 'Unknown')).lower())
        
        for speaker_id, data in sorted_speakers:
            # Apply filter if provided
            display_name = data.get('display_name', data.get('name', 'Unknown'))
            if filter_term and filter_term not in display_name.lower() and filter_term not in speaker_id.lower():
                continue
            
            # Format data
            confidence = f"{data['average_confidence']:.3f}" if data['average_confidence'] else "0.000"
            
            try:
                last_seen = datetime.fromisoformat(data['last_seen']).strftime('%m/%d %H:%M')
            except:
                last_seen = "Never"
            
            # Check if selected
            checkbox = "☑" if speaker_id in self.selected_speakers else "☐"
            
            # Insert into tree
            item_id = self.mgmt_speaker_tree.insert('', tk.END, 
                                   values=(checkbox,
                                          data['type'], 
                                          data['embedding_count'],
                                          confidence,
                                          last_seen,
                                          speaker_id),
                                   text=display_name,
                                   tags=(data['type'], speaker_id))
        
        # Configure tag colors and update selection info
        self.mgmt_speaker_tree.tag_configure('🤖 Auto', background='#ffe6e6')
        self.mgmt_speaker_tree.tag_configure('✅ Corrected', background='#e6ffe6')
        self.mgmt_speaker_tree.tag_configure('👤 Enrolled', background='#e6f3ff')
        
        self.update_selection_info()
    
    def on_tree_click(self, event):
        """Handle tree item click for selection"""
        item = self.mgmt_speaker_tree.identify('item', event.x, event.y)
        if item:
            # Get speaker ID from values
            values = self.mgmt_speaker_tree.item(item, 'values')
            if len(values) >= 6:
                speaker_id = values[5]  # Speaker ID is in the 6th column
                
                # Toggle selection
                if speaker_id in self.selected_speakers:
                    self.selected_speakers.remove(speaker_id)
                else:
                    self.selected_speakers.add(speaker_id)
                
                # Update display
                self.update_management_speaker_list(self.search_var.get().lower())
    
    def on_tree_double_click(self, event):
        """Handle double-click to edit speaker"""
        item = self.mgmt_speaker_tree.identify('item', event.x, event.y)
        if item:
            values = self.mgmt_speaker_tree.item(item, 'values')
            if len(values) >= 6:
                speaker_id = values[5]
                self.selected_speakers = {speaker_id}
                self.edit_selected_speaker()
    
    def update_selection_info(self):
        """Update the selection info panel"""
        if not hasattr(self, 'selection_info'):
            return
            
        self.selection_info.delete(1.0, tk.END)
        
        if not self.selected_speakers:
            self.selection_info.insert(tk.END, "No speakers selected\n\nClick speakers to select them for operations.")
            return
        
        info = f"📋 Selected: {len(self.selected_speakers)} speakers\n\n"
        
        total_samples = 0
        speaker_types = {}
        
        for speaker_id in self.selected_speakers:
            if speaker_id in self.speaker_data:
                data = self.speaker_data[speaker_id]
                display_name = data.get('display_name', data.get('name', 'Unknown'))
                info += f"• {display_name} ({speaker_id[:12]}...)\n"
                info += f"  Type: {data['type']} | Samples: {data['embedding_count']}\n\n"
                
                total_samples += data['embedding_count']
                speaker_type = data['type']
                speaker_types[speaker_type] = speaker_types.get(speaker_type, 0) + 1
        
        info += f"📊 Total samples: {total_samples}\n"
        for stype, count in speaker_types.items():
            info += f"   {stype}: {count}\n"
        
        self.selection_info.insert(tk.END, info)
    
    def edit_selected_speaker(self):
        """Edit the selected speaker (single selection only)"""
        if len(self.selected_speakers) != 1:
            messagebox.showerror("Error", "Please select exactly one speaker to edit.")
            return
        
        speaker_id = list(self.selected_speakers)[0]
        if speaker_id not in self.speaker_data:
            messagebox.showerror("Error", "Speaker not found in database.")
            return
        
        self.show_edit_speaker_dialog(speaker_id)
    
    def show_edit_speaker_dialog(self, speaker_id):
        """Show dialog to edit speaker details"""
        speaker_data = self.speaker_data[speaker_id]
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Speaker Profile")
        dialog.geometry("500x400")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 100))
        
        # Header
        tk.Label(dialog, text="✏️ Edit Speaker Profile", 
                font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=(20, 15))
        
        # Speaker info frame
        info_frame = ttk.LabelFrame(dialog, text="Speaker Information", padding=15)
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Speaker name
        tk.Label(info_frame, text="Speaker Name:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        name_var = tk.StringVar(value=speaker_data.get('display_name', speaker_data.get('name', 'Unknown')))
        name_entry = tk.Entry(info_frame, textvariable=name_var, width=40, font=("Arial", 11))
        name_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Speaker ID (read-only)
        tk.Label(info_frame, text="Speaker ID:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=5)
        tk.Label(info_frame, text=speaker_id, font=("Arial", 10)).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Statistics (read-only)
        tk.Label(info_frame, text="Type:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=5)
        tk.Label(info_frame, text=speaker_data['type'], font=("Arial", 10)).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        tk.Label(info_frame, text="Samples:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky=tk.W, pady=5)
        tk.Label(info_frame, text=str(speaker_data['embedding_count']), font=("Arial", 10)).grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        tk.Label(info_frame, text="Confidence:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, pady=5)
        confidence = f"{speaker_data['average_confidence']:.3f}" if speaker_data['average_confidence'] else "0.000"
        tk.Label(info_frame, text=confidence, font=("Arial", 10)).grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        def save_changes():
            new_name = name_var.get().strip()
            if not new_name:
                messagebox.showerror("Error", "Speaker name cannot be empty.")
                return
            
            current_name = speaker_data.get('display_name', speaker_data.get('name', 'Unknown'))
            if new_name != current_name:
                self.update_speaker_name_in_backend(speaker_id, new_name, dialog)
            else:
                dialog.destroy()
        
        def delete_speaker():
            display_name = speaker_data.get('display_name', speaker_data.get('name', 'Unknown'))
            if messagebox.askyesno("Confirm Delete", 
                                 f"Are you sure you want to delete speaker '{display_name}'?\n\n"
                                 f"This will permanently remove:\n"
                                 f"• {speaker_data['embedding_count']} voice samples\n"
                                 f"• All learning data\n"
                                 f"• Speaker profile\n\n"
                                 f"This action cannot be undone!"):
                self.delete_speaker_from_backend(speaker_id, dialog)
        
        ttk.Button(button_frame, text="💾 Save Changes", command=save_changes).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="🗑️ Delete Speaker", command=delete_speaker).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
        
        name_entry.focus()
        name_entry.bind('<Return>', lambda e: save_changes())
    
    def update_speaker_name_in_backend(self, speaker_id, new_name, dialog):
        """Update speaker name via backend API"""
        try:
            response = requests.put(
                f"{self.backend_url}/speakers/{speaker_id}/name",
                params={"new_name": new_name},
                timeout=10
            )
            
            if response.status_code == 200:
                self.log_management(f"✅ Updated speaker name: {speaker_id} -> {new_name}")
                messagebox.showinfo("Success", f"Speaker name updated to: {new_name}")
                dialog.destroy()
                self.refresh_data()
            else:
                error_msg = f"Failed to update speaker name: HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg += f"\n{error_detail}"
                except:
                    pass
                messagebox.showerror("Update Failed", error_msg)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update speaker name:\n{str(e)}")
    
    def delete_speaker_from_backend(self, speaker_id, dialog):
        """Delete speaker via backend API"""
        try:
            response = requests.delete(f"{self.backend_url}/speakers/{speaker_id}", timeout=10)
            
            if response.status_code == 200:
                self.log_management(f"🗑️ Deleted speaker: {speaker_id}")
                messagebox.showinfo("Success", "Speaker deleted successfully.")
                dialog.destroy()
                self.selected_speakers.discard(speaker_id)
                self.refresh_data()
            else:
                error_msg = f"Failed to delete speaker: HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg += f"\n{error_detail}"
                except:
                    pass
                messagebox.showerror("Delete Failed", error_msg)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete speaker:\n{str(e)}")
    
    def merge_selected_speakers(self):
        """Merge selected speakers"""
        if len(self.selected_speakers) < 2:
            messagebox.showerror("Error", "Please select at least 2 speakers to merge.")
            return
        
        self.show_merge_speakers_dialog()
    
    def show_merge_speakers_dialog(self):
        """Show dialog to merge selected speakers"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Merge Speaker Profiles")
        dialog.geometry("700x600")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Header
        tk.Label(dialog, text="🔗 Merge Speaker Profiles", 
                font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=(20, 10))
        
        tk.Label(dialog, text=f"Selected {len(self.selected_speakers)} speakers to merge:", 
                font=("Arial", 11), bg='#f0f0f0').pack(pady=(0, 15))
        
        # Show selected speakers
        speakers_frame = ttk.LabelFrame(dialog, text="Speakers to Merge", padding=10)
        speakers_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create listbox for speakers
        speakers_listbox = tk.Listbox(speakers_frame, font=("Arial", 10), height=8)
        speakers_scroll = ttk.Scrollbar(speakers_frame, orient="vertical", command=speakers_listbox.yview)
        speakers_listbox.configure(yscrollcommand=speakers_scroll.set)
        
        speakers_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        speakers_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with selected speakers
        speaker_list = []
        total_samples = 0
        for speaker_id in self.selected_speakers:
            if speaker_id in self.speaker_data:
                data = self.speaker_data[speaker_id]
                display_name = data.get('display_name', data.get('name', 'Unknown'))
                display_text = f"{display_name} ({speaker_id[:12]}...) - {data['embedding_count']} samples"
                speakers_listbox.insert(tk.END, display_text)
                speaker_list.append((speaker_id, data))
                total_samples += data['embedding_count']
        
        # Target selection
        target_frame = ttk.LabelFrame(dialog, text="Merge Target", padding=10)
        target_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(target_frame, text="Select target speaker (will be kept):").pack(anchor=tk.W, pady=(0, 5))
        
        target_var = tk.StringVar()
        target_dropdown = ttk.Combobox(target_frame, textvariable=target_var, width=50)
        target_values = [f"{data.get('display_name', data.get('name', 'Unknown'))} ({speaker_id[:12]}...)" for speaker_id, data in speaker_list]
        target_dropdown['values'] = target_values
        target_dropdown.pack(fill=tk.X, pady=5)
        
        tk.Label(target_frame, text="Final speaker name:").pack(anchor=tk.W, pady=(10, 5))
        final_name_var = tk.StringVar()
        final_name_entry = tk.Entry(target_frame, textvariable=final_name_var, font=("Arial", 11))
        final_name_entry.pack(fill=tk.X, pady=5)
        
        # Update final name when target changes
        def on_target_change(event):
            selection = target_dropdown.current()
            if selection >= 0:
                _, data = speaker_list[selection]
                display_name = data.get('display_name', data.get('name', 'Unknown'))
                final_name_var.set(display_name)
        
        target_dropdown.bind('<<ComboboxSelected>>', on_target_change)
        
        # Info
        tk.Label(dialog, text=f"Total samples to merge: {total_samples}", 
                font=("Arial", 10, "bold"), bg='#f0f0f0').pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=15)
        
        def perform_merge():
            if not target_var.get():
                messagebox.showerror("Error", "Please select a target speaker.")
                return
            
            target_index = target_dropdown.current()
            if target_index < 0:
                messagebox.showerror("Error", "Invalid target selection.")
                return
            
            target_id, _ = speaker_list[target_index]
            final_name = final_name_var.get().strip()
            
            if not final_name:
                messagebox.showerror("Error", "Please enter a final speaker name.")
                return
            
            # Confirm merge
            source_speakers = [sid for sid, _ in speaker_list if sid != target_id]
            confirm_msg = (
                f"Merge {len(source_speakers)} speakers into '{final_name}'?\n\n"
                f"Target: {target_id[:12]}...\n"
                f"Sources to be deleted: {len(source_speakers)}\n"
                f"Total samples: {total_samples}\n\n"
                f"This action cannot be undone!"
            )
            
            if messagebox.askyesno("Confirm Merge", confirm_msg):
                self.perform_speaker_merge(source_speakers, target_id, final_name, dialog)
        
        ttk.Button(button_frame, text="🔗 Merge Speakers", command=perform_merge).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def perform_speaker_merge(self, source_speakers, target_id, final_name, dialog):
        """Perform the actual speaker merge"""
        try:
            success_count = 0
            
            for source_id in source_speakers:
                response = requests.post(
                    f"{self.backend_url}/speakers/merge",
                    params={
                        "source_speaker_id": source_id,
                        "target_speaker_id": target_id,
                        "target_name": final_name
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    success_count += 1
                    self.log_management(f"✅ Merged {source_id[:12]}... into {target_id[:12]}...")
                else:
                    self.log_management(f"❌ Failed to merge {source_id[:12]}...")
            
            if success_count > 0:
                messagebox.showinfo("Merge Complete", 
                    f"✅ Successfully merged {success_count} speaker profiles!\n\n"
                    f"Final speaker: {final_name}\n"
                    f"Merged profiles have been combined.")
                dialog.destroy()
                self.selected_speakers.clear()
                self.refresh_data()
            else:
                messagebox.showerror("Merge Failed", "No speakers were successfully merged.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to merge speakers:\n{str(e)}")
    
    def delete_selected_speakers(self):
        """Delete selected speakers"""
        if not self.selected_speakers:
            messagebox.showerror("Error", "Please select speakers to delete.")
            return
        
        # Confirm deletion
        speaker_names = [self.speaker_data[sid].get('display_name', self.speaker_data[sid].get('name', 'Unknown')) for sid in self.selected_speakers if sid in self.speaker_data]
        total_samples = sum(self.speaker_data[sid]['embedding_count'] for sid in self.selected_speakers if sid in self.speaker_data)
        
        confirm_msg = (
            f"Delete {len(self.selected_speakers)} speakers?\n\n"
            f"Speakers: {', '.join(speaker_names[:5])}"
            f"{'...' if len(speaker_names) > 5 else ''}\n\n"
            f"Total samples to be deleted: {total_samples}\n\n"
            f"This action cannot be undone!"
        )
        
        if messagebox.askyesno("Confirm Delete", confirm_msg):
            self.perform_bulk_delete()
    
    def perform_bulk_delete(self):
        """Perform bulk deletion of selected speakers"""
        try:
            success_count = 0
            
            for speaker_id in list(self.selected_speakers):
                response = requests.delete(f"{self.backend_url}/speakers/{speaker_id}", timeout=10)
                
                if response.status_code == 200:
                    success_count += 1
                    self.log_management(f"🗑️ Deleted speaker: {speaker_id[:12]}...")
                    self.selected_speakers.discard(speaker_id)
                else:
                    self.log_management(f"❌ Failed to delete: {speaker_id[:12]}...")
            
            if success_count > 0:
                messagebox.showinfo("Delete Complete", 
                    f"✅ Successfully deleted {success_count} speaker profiles.")
                self.refresh_data()
            else:
                messagebox.showerror("Delete Failed", "No speakers were successfully deleted.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete speakers:\n{str(e)}")
    
    def find_similar_speakers(self):
        """Find speakers with similar names"""
        similar_groups = {}
        
        # Group speakers by similarity
        for speaker_id, data in self.speaker_data.items():
            display_name = data.get('display_name', data.get('name', 'Unknown'))
            name = display_name.lower().strip()
            
            # Simple similarity - same first 3 characters or contains same words
            found_group = False
            for group_key in similar_groups:
                if (name[:3] == group_key[:3] and len(name) > 3) or \
                   any(word in name.split() for word in group_key.split() if len(word) > 2):
                    similar_groups[group_key].append((speaker_id, data))
                    found_group = True
                    break
            
            if not found_group:
                similar_groups[name] = [(speaker_id, data)]
        
        # Filter groups with multiple speakers
        similar_groups = {k: v for k, v in similar_groups.items() if len(v) > 1}
        
        if not similar_groups:
            messagebox.showinfo("No Similar Names", "No speakers with similar names found.")
            return
        
        self.show_similar_speakers_dialog(similar_groups)
    
    def show_similar_speakers_dialog(self, similar_groups):
        """Show dialog with similar speaker groups"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Similar Speaker Names")
        dialog.geometry("800x600")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        tk.Label(dialog, text="🔍 Similar Speaker Names Found", 
                font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=(20, 10))
        
        tk.Label(dialog, text=f"Found {len(similar_groups)} groups of similar speakers:", 
                font=("Arial", 11), bg='#f0f0f0').pack(pady=(0, 15))
        
        # Similar groups list
        groups_frame = ttk.LabelFrame(dialog, text="Similar Groups", padding=10)
        groups_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create treeview for groups
        groups_tree = ttk.Treeview(groups_frame, columns=('count', 'samples'), show='tree headings', height=15)
        groups_tree.heading('#0', text='Group / Speaker Names')
        groups_tree.heading('count', text='Count')
        groups_tree.heading('samples', text='Total Samples')
        
        groups_tree.column('#0', width=400)
        groups_tree.column('count', width=100)
        groups_tree.column('samples', width=100)
        
        groups_scroll = ttk.Scrollbar(groups_frame, orient="vertical", command=groups_tree.yview)
        groups_tree.configure(yscrollcommand=groups_scroll.set)
        
        groups_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        groups_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate groups
        for group_name, speakers in similar_groups.items():
            total_samples = sum(data['embedding_count'] for _, data in speakers)
            
            # Add group header
            group_item = groups_tree.insert('', tk.END, 
                                          text=f"📁 {group_name.title()}",
                                          values=(len(speakers), total_samples),
                                          tags=('group',))
            
            # Add speakers in group
            for speaker_id, data in speakers:
                display_name = data.get('display_name', data.get('name', 'Unknown'))
                groups_tree.insert(group_item, tk.END,
                                 text=f"  • {display_name} ({speaker_id[:12]}...)",
                                 values=('', data['embedding_count']),
                                 tags=('speaker', speaker_id))
        
        groups_tree.tag_configure('group', background='#e6f3ff')
        
        # Expand all groups
        for item in groups_tree.get_children():
            groups_tree.item(item, open=True)
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=15)
        
        def select_group_for_merge():
            selection = groups_tree.selection()
            if not selection:
                messagebox.showerror("Error", "Please select speakers to merge.")
                return
            
            # Get all speaker IDs from selection
            speaker_ids = set()
            for item in selection:
                tags = groups_tree.item(item, 'tags')
                if 'speaker' in tags:
                    speaker_id = tags[1] if len(tags) > 1 else None
                    if speaker_id:
                        speaker_ids.add(speaker_id)
            
            if len(speaker_ids) < 2:
                messagebox.showerror("Error", "Please select at least 2 speakers to merge.")
                return
            
            # Set selection and show merge dialog
            self.selected_speakers = speaker_ids
            dialog.destroy()
            self.show_merge_speakers_dialog()
        
        ttk.Button(button_frame, text="🔗 Merge Selected", command=select_group_for_merge).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="❌ Close", command=dialog.destroy).pack(side=tk.LEFT)
    
    def show_bulk_cleanup_dialog(self):
        """Show dialog for bulk cleanup operations"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Bulk Cleanup")
        dialog.geometry("600x500")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        tk.Label(dialog, text="🧹 Bulk Cleanup Operations", 
                font=("Arial", 14, "bold"), bg='#f0f0f0').pack(pady=20)
        
        # Cleanup options frame
        options_frame = ttk.LabelFrame(dialog, text="Cleanup Options", padding=15)
        options_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Remove empty speakers
        ttk.Button(options_frame, text="🗑️ Remove Empty Speakers", 
                  command=lambda: (self.cleanup_speakers(), dialog.destroy())).pack(fill=tk.X, pady=5)
        
        # Find and merge duplicates
        ttk.Button(options_frame, text="🔍 Find Similar Names", 
                  command=lambda: (dialog.destroy(), self.find_similar_speakers())).pack(fill=tk.X, pady=5)
        
        ttk.Button(options_frame, text="❌ Close", command=dialog.destroy).pack(pady=20)
    
    def show_bulk_rename_dialog(self):
        """Show dialog for bulk rename operations"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Bulk Rename")
        dialog.geometry("600x400")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        tk.Label(dialog, text="📋 Bulk Rename Operations", 
                font=("Arial", 14, "bold"), bg='#f0f0f0').pack(pady=20)
        
        # Info text
        info_text = """Select speakers from the main list, then use the
'Edit Selected' or 'Merge Selected' buttons
for individual or bulk operations."""
        
        tk.Label(dialog, text=info_text, bg='#f0f0f0', justify=tk.CENTER).pack(pady=20)
        
        ttk.Button(dialog, text="❌ Close", command=dialog.destroy).pack(pady=20)

def main():
    """Main entry point"""
    print("🎤 Starting Oreja Speaker Analytics Dashboard...")
    app = SpeakerAnalyticsDashboard()
    app.run()

if __name__ == "__main__":
    main() 