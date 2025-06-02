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

class SpeakerAnalyticsDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Oreja Speaker Analytics Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Database path
        self.db_path = Path("speaker_data/speaker_database.db")
        
        # Backend URL
        self.backend_url = "http://127.0.0.1:8000"
        
        # Auto-refresh settings
        self.auto_refresh = tk.BooleanVar(value=True)
        self.refresh_interval = 5  # seconds
        
        # Data storage
        self.speaker_data = {}
        self.selected_speaker = None
        
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
        
        # Tab 4: Database Management
        self.setup_management_tab()
        
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
        
    def setup_management_tab(self):
        """Setup the database management tab"""
        mgmt_frame = ttk.Frame(self.notebook)
        self.notebook.add(mgmt_frame, text="⚙️ Management")
        
        # Management controls
        controls_frame = ttk.LabelFrame(mgmt_frame, text="Database Management", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1: Speaker operations
        row1 = ttk.Frame(controls_frame)
        row1.pack(fill=tk.X, pady=5)
        
        ttk.Button(row1, text="🔄 Refresh All Data", 
                  command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="🧹 Cleanup Empty Speakers", 
                  command=self.cleanup_speakers).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="📊 Generate Report", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=5)
        
        # Row 2: Data operations
        row2 = ttk.Frame(controls_frame)
        row2.pack(fill=tk.X, pady=5)
        
        ttk.Button(row2, text="💾 Export Database", 
                  command=self.export_database).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="📈 Export Analytics", 
                  command=self.export_analytics).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="⚠️ Reset Database", 
                  command=self.reset_database).pack(side=tk.LEFT, padx=5)
        
        # Management log
        log_frame = ttk.LabelFrame(mgmt_frame, text="Management Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.mgmt_log = tk.Text(log_frame, height=15, font=('Consolas', 9))
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.mgmt_log.yview)
        self.mgmt_log.configure(yscrollcommand=log_scroll.set)
        
        self.mgmt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
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
            
            # Update status
            self.update_status("Data refreshed successfully")
            self.last_update_label.config(text=f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.update_status(f"Error refreshing data: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh data: {str(e)}")
    
    def load_speaker_data(self):
        """Load speaker data from the database"""
        if not self.db_path.exists():
            self.speaker_data = {}
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            
            # Load speaker profiles
            cursor = conn.execute("""
                SELECT speaker_id, name, created_date, last_seen, 
                       session_count, total_audio_seconds, 
                       embedding_count, average_confidence
                FROM speaker_profiles
            """)
            
            self.speaker_data = {}
            for row in cursor.fetchall():
                speaker_id, name, created_date, last_seen, session_count, \
                total_audio_seconds, embedding_count, avg_confidence = row
                
                self.speaker_data[speaker_id] = {
                    'id': speaker_id,
                    'name': name,
                    'created_date': created_date,
                    'last_seen': last_seen,
                    'session_count': session_count,
                    'total_audio_seconds': total_audio_seconds,
                    'embedding_count': embedding_count,
                    'average_confidence': avg_confidence,
                    'type': self.get_speaker_type(speaker_id)
                }
            
            # Load embeddings for selected speaker if any
            if self.selected_speaker and self.selected_speaker in self.speaker_data:
                self.load_speaker_embeddings(self.selected_speaker)
            
            conn.close()
            
        except Exception as e:
            print(f"Error loading speaker data: {e}")
            self.speaker_data = {}
    
    def load_speaker_embeddings(self, speaker_id):
        """Load embedding history for a specific speaker"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            
            cursor = conn.execute("""
                SELECT timestamp, confidence, audio_length
                FROM speaker_embeddings
                WHERE speaker_id = ?
                ORDER BY timestamp
            """, (speaker_id,))
            
            embeddings = []
            for row in cursor.fetchall():
                timestamp, confidence, audio_length = row
                embeddings.append({
                    'timestamp': datetime.fromisoformat(timestamp),
                    'confidence': confidence,
                    'audio_length': audio_length
                })
            
            self.speaker_data[speaker_id]['embeddings'] = embeddings
            conn.close()
            
        except Exception as e:
            print(f"Error loading embeddings for {speaker_id}: {e}")
            self.speaker_data[speaker_id]['embeddings'] = []
    
    def get_speaker_type(self, speaker_id):
        """Determine speaker type based on ID"""
        if speaker_id.startswith('AUTO_SPEAKER'):
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
            
            # Insert into tree
            self.speaker_tree.insert('', tk.END, 
                                   values=(data['type'], 
                                          data['embedding_count'],
                                          confidence,
                                          last_seen),
                                   text=data['name'],
                                   tags=(data['type'],))
        
        # Configure tag colors
        self.speaker_tree.tag_configure('🤖 Auto', background='#ffe6e6')
        self.speaker_tree.tag_configure('✅ Corrected', background='#e6ffe6')
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
        enrolled_speakers = sum(1 for s in self.speaker_data.values() if s['type'] == '👤 Enrolled')
        
        total_embeddings = sum(s['embedding_count'] for s in self.speaker_data.values())
        avg_confidence = np.mean([s['average_confidence'] for s in self.speaker_data.values() 
                                if s['average_confidence'] is not None])
        
        # Find most active speaker
        most_active = max(self.speaker_data.values(), key=lambda x: x['embedding_count'])
        
        # Generate statistics text
        stats = f"""📊 SPEAKER DATABASE STATISTICS
{'='*40}

Total Speakers: {total_speakers}
├─ 🤖 Auto-generated: {auto_speakers}
├─ ✅ User-corrected: {corrected_speakers}
└─ 👤 Enrolled: {enrolled_speakers}

Training Data:
├─ Total audio samples: {total_embeddings:,}
├─ Average confidence: {avg_confidence:.3f}
└─ Most active speaker: {most_active['name']}
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
        
        # Find speaker by name
        for speaker_id, data in self.speaker_data.items():
            if data['name'] == speaker_name:
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
        details = f"""🎤 SPEAKER PROFILE: {data['name']}
{'='*60}

Basic Information:
├─ Speaker ID: {data['id']}
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
        
        names = [s[1]['name'][:15] + '...' if len(s[1]['name']) > 15 else s[1]['name'] 
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
                recent_speakers.append((last_seen, speaker['name']))
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
                        'name': data['name'],
                        'type': data['type'],
                        'embedding_count': data['embedding_count'],
                        'average_confidence': data['average_confidence'],
                        'total_audio_seconds': data['total_audio_seconds'],
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
        report = f"""
OREJA SPEAKER ANALYTICS REPORT
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
Speaker: {data['name']}
├─ ID: {speaker_id}
├─ Type: {data['type']}
├─ Samples: {data['embedding_count']}
├─ Confidence: {data['average_confidence']:.3f}
├─ Audio Time: {data['total_audio_seconds']:.1f}s
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

def main():
    """Main entry point"""
    print("🎤 Starting Oreja Speaker Analytics Dashboard...")
    app = SpeakerAnalyticsDashboard()
    app.run()

if __name__ == "__main__":
    main() 