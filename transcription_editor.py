#!/usr/bin/env python3
"""
Enhanced Post-Transcription Editor GUI
Opens automatically after transcription to edit speaker names with advanced features
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
import json
import os
from pathlib import Path
import requests
from datetime import datetime
import re
import colorsys


class TranscriptionEditor:
    def __init__(self, transcription_file=None):
        """Initialize the transcription editor"""
        self.root = tk.Tk()
        self.root.title("🎨 Enhanced Transcription Editor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.transcription_data = None
        self.transcription_file = transcription_file
        self.segments = []
        self.speaker_names = set()
        self.speaker_entries = {}
        self.segment_checkboxes = {}
        self.selected_segments = set()
        
        # Track changes for better save detection
        self.original_segments = []  # Store original state
        self.has_unsaved_changes = False
        
        # Split mode state
        self.split_mode_active = False
        
        # Speaker colors and emotional analysis
        self.speaker_colors = {}  # Color mapping for speakers
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#F4A261', '#2A9D8F', '#E76F51', '#F72585',
            '#7209B7', '#560BAD', '#480CA8', '#3A0CA3', '#3F37C9'
        ]
        self.color_index = 0
        
        # Emotional tone indicators
        self.emotion_icons = {
            'positive': '😊', 'negative': '😞', 'neutral': '😐',
            'happy': '😄', 'sad': '😢', 'angry': '😠', 'excited': '🤩',
            'confused': '😕', 'surprised': '😲', 'question': '❓'
        }
        
        # UI components
        self.file_label = None
        self.speaker_canvas = None
        self.speaker_scrollbar = None
        self.speaker_frame_inner = None
        self.text_area = None
        self.bulk_rename_btn = None
        self.split_text_btn = None
        
        self.setup_ui()
        
        if transcription_file:
            self.load_transcription(transcription_file)
    
    def get_speaker_color(self, speaker):
        """Get or assign a color to a speaker"""
        if speaker not in self.speaker_colors:
            self.speaker_colors[speaker] = self.color_palette[self.color_index % len(self.color_palette)]
            self.color_index += 1
        return self.speaker_colors[speaker]
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'yes', 'sure', 'absolutely']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'no', 'never', 'wrong', 'error', 'problem', 'issue', 'fail']
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which', '?']
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in question_words):
            return 'question'
        elif any(word in text_lower for word in positive_words):
            return 'positive'
        elif any(word in text_lower for word in negative_words):
            return 'negative'
        else:
            return 'neutral'
    
    def setup_ui(self):
        """Setup the enhanced user interface"""
        # Title and file info
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🎨 Enhanced Transcription Editor", 
                             font=("Arial", 18, "bold"), fg='white', bg='#2c3e50')
        title_label.pack(pady=15)
        
        # File info frame
        info_frame = tk.Frame(self.root, bg='#ecf0f1', height=40)
        info_frame.pack(fill=tk.X)
        info_frame.pack_propagate(False)
        
        self.file_label = tk.Label(info_frame, text="No file loaded", font=("Arial", 11), 
                                  bg='#ecf0f1', fg='#2c3e50')
        self.file_label.pack(pady=10)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Enhanced multi-select toolbar
        toolbar_frame = tk.Frame(main_container, bg='#34495e', relief=tk.RAISED, bd=2)
        toolbar_frame.pack(fill=tk.X, pady=(0, 10))
        
        toolbar_inner = tk.Frame(toolbar_frame, bg='#34495e')
        toolbar_inner.pack(pady=8, padx=15)
        
        tk.Label(toolbar_inner, text="📋 Multi-Select Operations:", font=("Arial", 10, "bold"), 
                fg='white', bg='#34495e').pack(side=tk.LEFT, padx=(0, 15))
        
        self.select_all_btn = tk.Button(toolbar_inner, text="☑ Select All", command=self.select_all_segments,
                                       bg='#2ecc71', fg='white', font=("Arial", 9, "bold"), 
                                       relief=tk.FLAT, padx=15, pady=5)
        self.select_all_btn.pack(side=tk.LEFT, padx=(10, 5))
        
        self.clear_selection_btn = tk.Button(toolbar_inner, text="☐ Clear Selection", command=self.clear_selection,
                                           bg='#e74c3c', fg='white', font=("Arial", 9, "bold"),
                                           relief=tk.FLAT, padx=15, pady=5)
        self.clear_selection_btn.pack(side=tk.LEFT, padx=5)
        
        self.bulk_rename_btn = tk.Button(toolbar_inner, text="🏷 Bulk Rename (0)", command=self.bulk_rename_speakers,
                                        bg='#f39c12', fg='white', font=("Arial", 9, "bold"),
                                        relief=tk.FLAT, padx=15, pady=5, state=tk.DISABLED)
        self.bulk_rename_btn.pack(side=tk.LEFT, padx=5)
        
        # Split mode toggle button
        self.split_text_btn = tk.Button(toolbar_inner, text="✂️ Split Mode: OFF", command=self.toggle_split_mode,
                                       bg='#9b59b6', fg='white', font=("Arial", 9, "bold"),
                                       relief=tk.FLAT, padx=15, pady=5)
        self.split_text_btn.pack(side=tk.LEFT, padx=5)
        
        # MAIN LAYOUT: Use PanedWindow for resizable panels
        main_paned = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, 
                                   sashwidth=6, bg='#bdc3c7')
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # LEFT PANEL CONTAINER (will contain stats and speaker mapping)
        left_container = tk.Frame(main_paned, bg='#f0f0f0')
        main_paned.add(left_container, minsize=350, width=400)  # Resizable with minimum width
        
        # STATS PANEL (separate from speaker mapping)
        stats_panel = tk.LabelFrame(left_container, text="📊 Transcription Statistics", 
                                   font=("Arial", 11, "bold"), bg='#ecf0f1', fg='#2c3e50',
                                   relief=tk.GROOVE, bd=2, padx=8, pady=8)
        stats_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Stats content frame
        stats_content = tk.Frame(stats_panel, bg='#ecf0f1')
        stats_content.pack(fill=tk.X)
        
        self.stats_label = tk.Label(stats_content, text="No transcription loaded", 
                                   font=("Arial", 10), bg='#ecf0f1', fg='#34495e',
                                   justify=tk.LEFT, anchor=tk.W)
        self.stats_label.pack(fill=tk.X, pady=5)
        
        # SPEAKER MAPPING PANEL (separate and expandable)
        speaker_panel = tk.LabelFrame(left_container, text="🎨 Speaker Name Mapping", 
                                    font=("Arial", 11, "bold"), bg='#ecf0f1', fg='#2c3e50',
                                    relief=tk.GROOVE, bd=2, padx=8, pady=8)
        speaker_panel.pack(fill=tk.BOTH, expand=True)
        
        # Speaker mapping headers
        header_frame = tk.Frame(speaker_panel, bg='#ecf0f1')
        header_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(header_frame, text="Original Speaker", font=("Arial", 10, "bold"), 
                bg='#ecf0f1', fg='#34495e').pack(side=tk.LEFT)
        tk.Label(header_frame, text="→ Corrected Name", font=("Arial", 10, "bold"), 
                bg='#ecf0f1', fg='#34495e').pack(side=tk.RIGHT)
        
        # Add New Speaker button
        add_speaker_frame = tk.Frame(speaker_panel, bg='#ecf0f1')
        add_speaker_frame.pack(fill=tk.X, pady=(0, 8))
        
        add_speaker_btn = tk.Button(add_speaker_frame, text="✨ Add New Speaker", 
                                   command=self.add_new_speaker_globally,
                                   bg='#3498db', fg='white', font=("Arial", 10, "bold"),
                                   relief=tk.FLAT, padx=15, pady=5)
        add_speaker_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        merge_speaker_btn = tk.Button(add_speaker_frame, text="🔗 Merge Speakers", 
                                     command=self.show_merge_speakers_dialog,
                                     bg='#9b59b6', fg='white', font=("Arial", 10, "bold"),
                                     relief=tk.FLAT, padx=15, pady=5)
        merge_speaker_btn.pack(side=tk.LEFT, padx=5)
        
        # Scrollable speaker mapping area
        mapping_frame = tk.Frame(speaker_panel, bg='#ecf0f1')
        mapping_frame.pack(fill=tk.BOTH, expand=True)
        
        self.speaker_canvas = tk.Canvas(mapping_frame, bg='white', highlightthickness=0)
        self.speaker_scrollbar = ttk.Scrollbar(mapping_frame, orient="vertical", command=self.speaker_canvas.yview)
        self.speaker_frame_inner = tk.Frame(self.speaker_canvas, bg='white')
        
        self.speaker_canvas.configure(yscrollcommand=self.speaker_scrollbar.set)
        self.speaker_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.speaker_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.speaker_canvas.create_window((0, 0), window=self.speaker_frame_inner, anchor="nw")
        
        # RIGHT PANEL: Enhanced transcription preview
        preview_panel = tk.LabelFrame(main_paned, text="📝 Enhanced Transcription Preview", 
                                    font=("Arial", 11, "bold"), bg='#ecf0f1', fg='#2c3e50',
                                    relief=tk.GROOVE, bd=2, padx=8, pady=8)
        main_paned.add(preview_panel, minsize=500)  # Resizable with minimum width
        
        # Preview controls
        preview_controls = tk.Frame(preview_panel, bg='#ecf0f1')
        preview_controls.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(preview_controls, text="💡 Features: ✓ Color-coded speakers ✓ Emotional indicators ✓ Multi-select ✓ Text splitting", 
                font=("Arial", 9), bg='#ecf0f1', fg='#7f8c8d', wraplength=500).pack(anchor=tk.W)
        
        # Scrollable transcription area
        text_frame = tk.Frame(preview_panel, bg='#ecf0f1')
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_area = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10), 
                               bg='white', fg='#2c3e50', selectbackground='#3498db',
                               relief=tk.SUNKEN, bd=1)
        text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=text_scrollbar.set)
        
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enhanced buttons panel
        button_panel = tk.Frame(main_container, bg='#f0f0f0')
        button_panel.pack(fill=tk.X, pady=(15, 0))
        
        # Create styled buttons
        self.create_styled_button(button_panel, "📁 Open File", self.open_file, '#3498db')
        self.create_styled_button(button_panel, "👁 Preview Changes", self.preview_changes, '#2ecc71')
        self.create_styled_button(button_panel, "💾 Save Corrected", self.save_corrected, '#e67e22')
        self.create_styled_button(button_panel, "📤 Send Feedback", self.send_feedback, '#9b59b6')
        
        self.speaker_entries = {}
        self.segment_checkboxes = {}
    
    def create_styled_button(self, parent, text, command, color):
        """Create a styled button"""
        btn = tk.Button(parent, text=text, command=command, 
                       bg=color, fg='white', font=("Arial", 10, "bold"),
                       relief=tk.FLAT, padx=20, pady=8, cursor='hand2')
        btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add hover effects
        def on_enter(e):
            btn.configure(bg=self.darken_color(color))
        def on_leave(e):
            btn.configure(bg=color)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def darken_color(self, color):
        """Darken a hex color"""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, int(c * 0.8)) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
    
    def load_transcription(self, file_path):
        """Load transcription from JSON or TXT file"""
        try:
            self.transcription_file = file_path
            self.file_label.config(text=f"File: {Path(file_path).name}")
            
            if file_path.endswith('.json'):
                # Load JSON format (structured data)
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.transcription_data = json.load(f)
                
                # Extract segments and speaker names from JSON
                self.segments = self.transcription_data.get('segments', [])
                self.speaker_names = set()
                
                for segment in self.segments:
                    speaker = segment.get('speaker', 'Unknown')
                    self.speaker_names.add(speaker)
                    
            else:
                # Load TXT format (parse structured text)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.segments, self.speaker_names = self.parse_txt_transcription(content)
                
                # Create basic transcription data structure
                self.transcription_data = {
                    'metadata': {
                        'source': 'Oreja TXT Import',
                        'version': '1.0'
                    },
                    'segments': self.segments,
                    'full_text': content
                }
            
            # Store original state for change tracking
            import copy
            self.original_segments = copy.deepcopy(self.segments)
            self.has_unsaved_changes = False
            
            self.update_stats()
            self.setup_speaker_mapping()
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load transcription: {e}")
    
    def parse_txt_transcription(self, content):
        """Parse TXT transcription content into segments"""
        segments = []
        speaker_names = set()
        segment_id = 0
        
        lines = content.split('\n')
        current_speaker = None
        current_text = ""
        current_time = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for timestamp and speaker patterns: [MM:SS] Speaker: Text
            import re
            
            # Pattern for: [MM:SS] [Source] Speaker: Text
            full_pattern = r'\[(\d{2}:\d{2})\]\s*\[([^\]]+)\]\s*([^:]+):\s*(.+)'
            match = re.match(full_pattern, line)
            
            if match:
                time_str, source, speaker, text = match.groups()
                minutes, seconds = map(int, time_str.split(':'))
                start_time = minutes * 60 + seconds
                
                segments.append({
                    'id': segment_id,
                    'speaker': speaker.strip(),
                    'text': text.strip(),
                    'start': start_time,
                    'end': start_time + 5,  # Estimate 5 second duration
                    'source': source.strip()
                })
                speaker_names.add(speaker.strip())
                segment_id += 1
                continue
            
            # Pattern for: [MM:SS] Speaker: Text  
            simple_pattern = r'\[(\d{2}:\d{2})\]\s*([^:]+):\s*(.+)'
            match = re.match(simple_pattern, line)
            
            if match:
                time_str, speaker, text = match.groups()
                minutes, seconds = map(int, time_str.split(':'))
                start_time = minutes * 60 + seconds
                
                segments.append({
                    'id': segment_id,
                    'speaker': speaker.strip(),
                    'text': text.strip(),
                    'start': start_time,
                    'end': start_time + 5,  # Estimate 5 second duration
                    'source': 'Unknown'
                })
                speaker_names.add(speaker.strip())
                segment_id += 1
                continue
            
            # Pattern for: Speaker: Text (no timestamp)
            speaker_pattern = r'^([^:]+):\s*(.+)'
            match = re.match(speaker_pattern, line)
            
            if match:
                speaker, text = match.groups()
                segments.append({
                    'id': segment_id,
                    'speaker': speaker.strip(),
                    'text': text.strip(),
                    'start': current_time,
                    'end': current_time + 5,
                    'source': 'Unknown'
                })
                speaker_names.add(speaker.strip())
                segment_id += 1
                current_time += 5  # Increment time for next segment
                continue
            
            # If no pattern matches, treat as continuation of previous speaker
            if line and segments:
                segments[-1]['text'] += ' ' + line
        
        # If no structured data found, create a single segment
        if not segments:
            segments.append({
                'id': 0,
                'speaker': 'Unknown',
                'text': content,
                'start': 0,
                'end': len(content.split()) * 0.5,  # Rough estimate
                'source': 'Unknown'
            })
            speaker_names.add('Unknown')
        
        return segments, speaker_names
    
    def setup_speaker_mapping(self):
        """Setup enhanced speaker name mapping interface with colors"""
        # Clear existing entries
        for widget in self.speaker_frame_inner.winfo_children():
            widget.destroy()
        
        self.speaker_entries = {}
        
        row = 0
        for speaker_id in sorted(self.speaker_names):
            speaker_color = self.get_speaker_color(speaker_id)
            
            # Container frame for each speaker row
            speaker_row = tk.Frame(self.speaker_frame_inner, bg='white', relief=tk.RIDGE, bd=1)
            speaker_row.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2, padx=2)
            speaker_row.columnconfigure(1, weight=1)
            
            # Color indicator
            color_label = tk.Label(speaker_row, text="●", font=("Arial", 20), 
                                 fg=speaker_color, bg='white')
            color_label.grid(row=0, column=0, padx=(5, 10), pady=5)
            
            # Original speaker ID label with colored background
            speaker_label = tk.Label(speaker_row, text=speaker_id, font=("Arial", 10, "bold"),
                                   bg=speaker_color, fg='white', padx=8, pady=4, relief=tk.RAISED)
            speaker_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 10), pady=5)
            
            # Entry for corrected name
            entry = tk.Entry(speaker_row, width=25, font=("Arial", 10), relief=tk.GROOVE, bd=2)
            entry.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(0, 10), pady=5)
            entry.insert(0, speaker_id)  # Default to original name
            # Only update when done editing (not on every keystroke to avoid lag)
            entry.bind('<FocusOut>', self.on_name_change)  # When user clicks away
            entry.bind('<Return>', self.on_name_change)    # When user presses Enter
            
            # Segment count indicator
            segment_count = len([s for s in self.segments if s.get('speaker') == speaker_id])
            count_label = tk.Label(speaker_row, text=f"({segment_count} segments)", 
                                 font=("Arial", 9), fg='#7f8c8d', bg='white')
            count_label.grid(row=0, column=3, padx=(0, 5), pady=5)
            
            self.speaker_entries[speaker_id] = entry
            row += 1
        
        # Update scroll region
        self.speaker_frame_inner.update_idletasks()
        self.speaker_canvas.configure(scrollregion=self.speaker_canvas.bbox("all"))
    
    def on_name_change(self, event=None):
        """Handle speaker name changes and sync with backend"""
        self.has_unsaved_changes = True
        self.send_speaker_changes_to_backend()
        self.update_preview()
    
    def get_speaker_mapping(self):
        """Get current speaker name mapping from UI entries"""
        mapping = {}
        for original_id, entry in self.speaker_entries.items():
            new_name = entry.get().strip()
            if new_name and new_name != original_id:
                mapping[original_id] = new_name
        return mapping
    
    def get_all_changes(self):
        """Get all changes including bulk renames by comparing current segments to original"""
        if not self.original_segments:
            return {}
        
        changes = {}
        
        # First, get UI mapping changes
        ui_mapping = self.get_speaker_mapping()
        changes.update(ui_mapping)
        
        # Then, detect direct segment changes (from bulk rename)
        original_speaker_map = {}
        current_speaker_map = {}
        
        for i, (orig_seg, curr_seg) in enumerate(zip(self.original_segments, self.segments)):
            orig_speaker = orig_seg.get('speaker', 'Unknown')
            curr_speaker = curr_seg.get('speaker', 'Unknown')
            
            if orig_speaker != curr_speaker:
                # This segment was changed
                if orig_speaker not in changes:  # Don't override UI mapping
                    changes[orig_speaker] = curr_speaker
        
        return changes
    
    def show_new_speaker_dialog(self):
        """Show dialog to select existing speaker or create a new one"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Speaker")
        dialog.geometry("500x400")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 100))
        
        result = tk.StringVar()
        
        # Header
        tk.Label(dialog, text="✨ Add Speaker", 
                font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=(20, 10))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tab 1: Select from existing enrolled speakers
        existing_frame = tk.Frame(notebook, bg='#f0f0f0')
        notebook.add(existing_frame, text="📋 Select Existing")
        
        tk.Label(existing_frame, text="Select from enrolled speakers in database:", 
                font=("Arial", 11), bg='#f0f0f0').pack(pady=(15, 10))
        
        # Listbox for existing speakers
        listbox_frame = tk.Frame(existing_frame, bg='#f0f0f0')
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        speaker_listbox = tk.Listbox(listbox_frame, font=("Arial", 10), height=8)
        scrollbar_existing = ttk.Scrollbar(listbox_frame, orient="vertical", command=speaker_listbox.yview)
        speaker_listbox.configure(yscrollcommand=scrollbar_existing.set)
        
        speaker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_existing.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load existing speakers from backend
        existing_speakers = self.get_enrolled_speakers()
        if existing_speakers:
            for speaker in existing_speakers:
                name = speaker.get('name', 'Unknown')
                speaker_id = speaker.get('id', '')
                display_text = f"{name} ({speaker_id[:8]}...)"
                speaker_listbox.insert(tk.END, display_text)
        else:
            speaker_listbox.insert(tk.END, "No enrolled speakers found")
            speaker_listbox.config(state=tk.DISABLED)
        
        # Tab 2: Create new speaker
        new_frame = tk.Frame(notebook, bg='#f0f0f0')
        notebook.add(new_frame, text="✨ Create New")
        
        tk.Label(new_frame, text="Enter name for completely new speaker:", 
                font=("Arial", 11), bg='#f0f0f0').pack(pady=(30, 10))
        
        name_entry = tk.Entry(new_frame, width=30, font=("Arial", 11))
        name_entry.pack(pady=10)
        
        tk.Label(new_frame, text="(This creates a new speaker not in the database)", 
                font=("Arial", 9), bg='#f0f0f0', fg='#7f8c8d').pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=15)
        
        def select_existing():
            selection = speaker_listbox.curselection()
            if selection and existing_speakers:
                speaker = existing_speakers[selection[0]]
                result.set(speaker.get('name', 'Unknown'))
                dialog.destroy()
            else:
                messagebox.showerror("Error", "Please select a speaker from the list.")
        
        def create_new():
            name = name_entry.get().strip()
            if name:
                result.set(name)
                dialog.destroy()
            else:
                messagebox.showerror("Error", "Please enter a speaker name.")
        
        def cancel():
            result.set("")
            dialog.destroy()
        
        # Determine which button to show based on current tab
        def on_tab_change(event):
            current_tab = notebook.index(notebook.select())
            for widget in button_frame.winfo_children():
                widget.destroy()
            
            if current_tab == 0:  # Existing speakers tab
                select_btn = tk.Button(button_frame, text="✅ Select Speaker", command=select_existing,
                                      bg='#3498db', fg='white', font=("Arial", 10, "bold"),
                                      padx=20, pady=5)
                select_btn.pack(side=tk.LEFT, padx=(0, 10))
            else:  # Create new tab
                create_btn = tk.Button(button_frame, text="✅ Create", command=create_new,
                                      bg='#2ecc71', fg='white', font=("Arial", 10, "bold"),
                                      padx=20, pady=5)
                create_btn.pack(side=tk.LEFT, padx=(0, 10))
                name_entry.focus()
            
            cancel_btn = tk.Button(button_frame, text="❌ Cancel", command=cancel,
                                  bg='#e74c3c', fg='white', font=("Arial", 10, "bold"),
                                  padx=20, pady=5)
            cancel_btn.pack(side=tk.LEFT)
        
        notebook.bind("<<NotebookTabChanged>>", on_tab_change)
        
        # Initialize with first tab
        on_tab_change(None)
        
        # Allow Enter key in new speaker entry
        name_entry.bind('<Return>', lambda e: create_new())
        
        # Allow double-click on listbox
        speaker_listbox.bind('<Double-Button-1>', lambda e: select_existing())
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result.get() if result.get() else None
    
    def get_enrolled_speakers(self):
        """Get list of enrolled speakers from the backend"""
        try:
            response = requests.get("http://127.0.0.1:8000/speakers", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('speakers', [])
        except Exception as e:
            print(f"Could not fetch enrolled speakers: {e}")
        return []
    
    def send_speaker_changes_to_backend(self):
        """Send speaker name changes to backend for permanent storage"""
        try:
            mapping = self.get_speaker_mapping()
            for original_id, new_name in mapping.items():
                if original_id != new_name:  # Only send changes
                    response = requests.post(
                        "http://127.0.0.1:8000/speakers/name_mapping",
                        params={
                            "old_speaker_id": original_id,
                            "new_speaker_name": new_name
                        },
                        timeout=5
                    )
                    if response.status_code == 200:
                        result = response.json()
                        status = result.get("status", "unknown")
                        if status == "speakers_merged":
                            print(f"✅ Merged speaker {original_id} into existing {new_name}")
                        elif status == "name_updated":
                            print(f"✅ Updated speaker {original_id} name to {new_name}")
                    else:
                        print(f"⚠️ Failed to update speaker {original_id}: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not sync speaker changes to backend: {e}")
    
    def add_new_speaker_globally(self):
        """Add a new speaker to the system globally"""
        new_speaker = self.show_new_speaker_dialog()
        if new_speaker:
            # Add to speaker names
            self.speaker_names.add(new_speaker)
            
            # Mark as changed and refresh
            self.has_unsaved_changes = True
            self.update_stats()
            self.setup_speaker_mapping()
            self.update_preview()
            
            messagebox.showinfo("Success", f"Added new speaker: {new_speaker}\n\nYou can now assign segments to this speaker using the dropdown menus.")
    
    def show_merge_speakers_dialog(self):
        """Show dialog to merge duplicate speaker profiles"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Merge Speaker Profiles")
        dialog.geometry("600x500")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Header
        tk.Label(dialog, text="🔗 Merge Speaker Profiles", 
                font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=(20, 10))
        
        tk.Label(dialog, text="Select speakers to merge (combines voice profiles & transcription history):", 
                font=("Arial", 11), bg='#f0f0f0').pack(pady=(0, 15))
        
        # Get enrolled speakers with more details
        enrolled_speakers = self.get_enrolled_speakers()
        if not enrolled_speakers:
            tk.Label(dialog, text="No speakers found in database to merge.", 
                    font=("Arial", 11), bg='#f0f0f0', fg='#e74c3c').pack(pady=20)
            
            tk.Button(dialog, text="Close", command=dialog.destroy,
                     bg='#95a5a6', fg='white', font=("Arial", 10, "bold"),
                     padx=20, pady=5).pack(pady=10)
            return
        
        # Two-column layout for source and target selection
        selection_frame = tk.Frame(dialog, bg='#f0f0f0')
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Source speaker (to be merged and deleted)
        source_frame = tk.Frame(selection_frame, bg='#f0f0f0')
        source_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(source_frame, text="Source Speaker (will be deleted):", 
                font=("Arial", 10, "bold"), bg='#f0f0f0', fg='#e74c3c').pack(pady=(0, 5))
        
        source_listbox = tk.Listbox(source_frame, font=("Arial", 9), height=12)
        source_scrollbar = ttk.Scrollbar(source_frame, orient="vertical", command=source_listbox.yview)
        source_listbox.configure(yscrollcommand=source_scrollbar.set)
        source_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        source_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Target speaker (will be kept)
        target_frame = tk.Frame(selection_frame, bg='#f0f0f0')
        target_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        tk.Label(target_frame, text="Target Speaker (will be kept):", 
                font=("Arial", 10, "bold"), bg='#f0f0f0', fg='#2ecc71').pack(pady=(0, 5))
        
        target_listbox = tk.Listbox(target_frame, font=("Arial", 9), height=12)
        target_scrollbar = ttk.Scrollbar(target_frame, orient="vertical", command=target_listbox.yview)
        target_listbox.configure(yscrollcommand=target_scrollbar.set)
        target_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        target_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate both lists with speaker information
        for speaker in enrolled_speakers:
            name = speaker.get('name', 'Unknown')
            speaker_id = speaker.get('id', '')
            embedding_count = speaker.get('embedding_count', 0)
            display_text = f"{name}\n  ID: {speaker_id[:12]}...\n  Samples: {embedding_count}\n"
            
            source_listbox.insert(tk.END, display_text)
            target_listbox.insert(tk.END, display_text)
        
        # Name entry for merged speaker
        name_frame = tk.Frame(dialog, bg='#f0f0f0')
        name_frame.pack(pady=10)
        
        tk.Label(name_frame, text="Final speaker name:", 
                font=("Arial", 10, "bold"), bg='#f0f0f0').pack(side=tk.LEFT, padx=(0, 10))
        
        final_name_entry = tk.Entry(name_frame, width=25, font=("Arial", 10))
        final_name_entry.pack(side=tk.LEFT)
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=15)
        
        def perform_merge():
            source_selection = source_listbox.curselection()
            target_selection = target_listbox.curselection()
            
            if not source_selection or not target_selection:
                messagebox.showerror("Error", "Please select both source and target speakers.")
                return
            
            if source_selection[0] == target_selection[0]:
                messagebox.showerror("Error", "Cannot merge a speaker with itself. Please select different speakers.")
                return
            
            source_speaker = enrolled_speakers[source_selection[0]]
            target_speaker = enrolled_speakers[target_selection[0]]
            final_name = final_name_entry.get().strip()
            
            if not final_name:
                final_name = target_speaker.get('name', 'Merged Speaker')
            
            # Confirm the merge
            confirm_msg = (
                f"Merge '{source_speaker.get('name')}' into '{target_speaker.get('name')}'?\n\n"
                f"• Source speaker ({source_speaker.get('id', '')[:12]}...) will be DELETED\n"
                f"• All voice data will be moved to target speaker\n"
                f"• Final name will be: {final_name}\n\n"
                f"This action cannot be undone!"
            )
            
            if messagebox.askyesno("Confirm Merge", confirm_msg):
                self.merge_speakers_in_backend(
                    source_speaker.get('id'),
                    target_speaker.get('id'),
                    final_name,
                    dialog
                )
        
        merge_btn = tk.Button(button_frame, text="🔗 Merge Speakers", command=perform_merge,
                             bg='#e74c3c', fg='white', font=("Arial", 10, "bold"),
                             padx=20, pady=5)
        merge_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_btn = tk.Button(button_frame, text="❌ Cancel", command=dialog.destroy,
                              bg='#95a5a6', fg='white', font=("Arial", 10, "bold"),
                              padx=20, pady=5)
        cancel_btn.pack(side=tk.LEFT)
        
        # Set initial target name when target is selected
        def on_target_select(event):
            selection = target_listbox.curselection()
            if selection:
                target_speaker = enrolled_speakers[selection[0]]
                final_name_entry.delete(0, tk.END)
                final_name_entry.insert(0, target_speaker.get('name', ''))
        
        target_listbox.bind('<<ListboxSelect>>', on_target_select)
    
    def merge_speakers_in_backend(self, source_id, target_id, final_name, dialog):
        """Perform the actual speaker merge via backend API"""
        try:
            response = requests.post(
                "http://127.0.0.1:8000/speakers/merge",
                params={
                    "source_speaker_id": source_id,
                    "target_speaker_id": target_id,
                    "target_name": final_name
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                messagebox.showinfo("Success", 
                    f"✅ Successfully merged speakers!\n\n"
                    f"• {source_id[:12]}... has been deleted\n"
                    f"• All data merged into: {final_name}\n"
                    f"• Voice profiles combined\n\n"
                    f"The speaker database has been updated.")
                dialog.destroy()
                
                # Refresh the speaker mapping to reflect changes
                self.setup_speaker_mapping()
                self.update_preview()
                
            else:
                error_msg = f"Failed to merge speakers: HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg += f"\n{error_detail}"
                except:
                    pass
                messagebox.showerror("Merge Failed", error_msg)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to merge speakers:\n{str(e)}")
    
    def update_preview(self):
        """Update the enhanced transcription preview with colors and emotions"""
        mapping = self.get_speaker_mapping()
        
        self.text_area.delete('1.0', tk.END)
        self.text_area.configure(state=tk.NORMAL)
        
        # Clear existing checkboxes
        self.segment_checkboxes.clear()
        
        for i, segment in enumerate(self.segments):
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            segment_id = segment.get('id', i)
            
            # Apply speaker mapping
            display_speaker = mapping.get(speaker, speaker)
            speaker_color = self.get_speaker_color(speaker)
            
            # Analyze emotional tone
            emotion = self.analyze_sentiment(text)
            emotion_icon = self.emotion_icons.get(emotion, '😐')
            
            # Format timestamp
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            
            # Create checkbox for multi-select (disabled in split mode)
            checkbox_var = tk.BooleanVar()
            checkbox = tk.Checkbutton(self.text_area, variable=checkbox_var, 
                                    command=lambda sid=segment_id: self.toggle_segment_selection(sid),
                                    bg='white', activebackground='#3498db')
            
            # In split mode, hide checkboxes to prevent click conflicts
            if self.split_mode_active:
                checkbox.config(state=tk.DISABLED)
                # Add a split icon instead
                self.text_area.insert(tk.END, "✂️ ")
            else:
                # Insert checkbox into text area
                self.text_area.window_create(tk.END, window=checkbox)
            
            self.segment_checkboxes[segment_id] = (checkbox, checkbox_var)
            
            # Add speaker reassignment dropdown for each segment (not in split mode)
            if not self.split_mode_active:
                # Get current speaker mapping to show proper names
                mapping = self.get_speaker_mapping()
                
                # Create list of available speakers (both mapped names and original IDs)
                available_speakers = []
                for orig_id in self.speaker_names:
                    mapped_name = mapping.get(orig_id, orig_id)
                    if mapped_name != orig_id:
                        available_speakers.append(mapped_name)  # Show mapped name
                    else:
                        available_speakers.append(orig_id)  # Show original ID
                
                # Add option to create new speaker
                available_speakers.append("+ Add New Speaker...")
                
                # Set current value to mapped name if available
                current_display_name = mapping.get(speaker, speaker)
                
                # Create speaker dropdown for individual segment reassignment
                speaker_var = tk.StringVar(value=current_display_name)
                speaker_dropdown = ttk.Combobox(self.text_area, textvariable=speaker_var,
                                              values=available_speakers, width=12, height=8)
                
                # Bind change event to update this specific segment
                def make_speaker_change_handler(seg_id, var, current_mapping):
                    def on_speaker_change(event=None):
                        selected_name = var.get()
                        
                        # Check if user selected "Add New Speaker"
                        if selected_name == "+ Add New Speaker...":
                            new_speaker = self.show_new_speaker_dialog()
                            if new_speaker:
                                var.set(new_speaker)
                                selected_name = new_speaker
                            else:
                                # User cancelled, restore original value
                                var.set(current_display_name)
                                return
                        
                        # Convert from display name back to original speaker ID if needed
                        target_speaker = selected_name
                        
                        # Check if this is a mapped name - if so, find the original ID
                        reverse_mapping = {v: k for k, v in current_mapping.items()}
                        if selected_name in reverse_mapping:
                            target_speaker = reverse_mapping[selected_name]
                        
                        # Update the specific segment
                        for seg in self.segments:
                            if seg.get('id', 0) == seg_id:
                                seg['speaker'] = target_speaker
                                break
                        
                        # Add to speaker names if it's a new speaker
                        self.speaker_names.add(target_speaker)
                        
                        # Mark as changed and refresh
                        self.has_unsaved_changes = True
                        self.update_stats()
                        self.setup_speaker_mapping()
                        self.update_preview()
                    return on_speaker_change
                
                speaker_dropdown.bind('<<ComboboxSelected>>', make_speaker_change_handler(segment_id, speaker_var, mapping))
                self.text_area.insert(tk.END, " → ")
                self.text_area.window_create(tk.END, window=speaker_dropdown)
            
            # Add formatted text with colors
            self.text_area.insert(tk.END, f" {timestamp} ")
            
            # Insert speaker name with color
            speaker_start = self.text_area.index(tk.END)
            self.text_area.insert(tk.END, f"{display_speaker}")
            speaker_end = self.text_area.index(tk.END)
            
            # Apply color tag to speaker name
            tag_name = f"speaker_{speaker.replace(' ', '_')}"
            self.text_area.tag_add(tag_name, speaker_start, speaker_end)
            self.text_area.tag_configure(tag_name, foreground=speaker_color, font=("Arial", 11, "bold"))
            
            # Add emotional indicator and text
            self.text_area.insert(tk.END, f" {emotion_icon}: {text}\n\n")
            
            # Store segment boundaries for click detection in split mode
            if self.split_mode_active:
                # Find where this segment's text actually starts and ends
                current_pos = self.text_area.index(tk.END)
                # Go back to find the text part (after timestamp and speaker name)
                segment_start_line = int(current_pos.split('.')[0]) - 2  # Account for \n\n
                
                # Create a mapping of text positions to segment info
                if not hasattr(self, 'segment_positions'):
                    self.segment_positions = {}
                
                self.segment_positions[segment_start_line] = {
                    'id': segment_id,
                    'text': text,
                    'speaker': speaker
                }
            

        
        # Configure split mode click behavior
        if self.split_mode_active:
            # In split mode: yellow background and crosshair cursor
            self.text_area.configure(bg='#fffbf0', cursor='crosshair')
            self.text_area.bind("<Button-1>", self.handle_split_click)
        else:
            # Normal mode: white background and normal cursor
            self.text_area.configure(bg='white', cursor='')
            self.text_area.unbind("<Button-1>")
        
        # Keep text area enabled for click events, but prevent typing
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.bind("<Key>", lambda e: "break")  # Prevent typing but allow clicks
    
    def toggle_segment_selection(self, segment_id):
        """Toggle selection of a segment for multi-select operations"""
        if segment_id in self.selected_segments:
            self.selected_segments.remove(segment_id)
        else:
            self.selected_segments.add(segment_id)
        
        self.update_multi_select_ui()
    
    def select_all_segments(self):
        """Select all segments"""
        self.selected_segments = set(range(len(self.segments)))
        
        # Update all checkboxes
        for segment_id, (checkbox, var) in self.segment_checkboxes.items():
            var.set(True)
        
        self.update_multi_select_ui()
    
    def clear_selection(self):
        """Clear all selections"""
        self.selected_segments.clear()
        
        # Update all checkboxes
        for segment_id, (checkbox, var) in self.segment_checkboxes.items():
            var.set(False)
        
        self.update_multi_select_ui()
    
    def update_multi_select_ui(self):
        """Update the multi-select UI elements"""
        count = len(self.selected_segments)
        self.bulk_rename_btn.config(text=f"👥 Reassign Speaker ({count})", 
                                   state=tk.NORMAL if count > 0 else tk.DISABLED)
        # Split mode button is always available, no longer depends on selection
    
    def bulk_rename_speakers(self):
        """Bulk reassign selected segments to a different speaker"""
        if not self.selected_segments:
            messagebox.showwarning("No Selection", "Please select segments to reassign.")
            return
        
        # Create dialog for bulk reassignment
        dialog = tk.Toplevel(self.root)
        dialog.title("Reassign Speaker")
        dialog.geometry("450x250")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Show what's currently selected
        current_speakers = set()
        for segment_id in self.selected_segments:
            if segment_id < len(self.segments):
                current_speakers.add(self.segments[segment_id].get('speaker', 'Unknown'))
        
        tk.Label(dialog, text=f"👥 Reassign {len(self.selected_segments)} segments", 
                font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=(15, 5))
        
        tk.Label(dialog, text=f"Currently: {', '.join(current_speakers)}", 
                font=("Arial", 10), bg='#f0f0f0', fg='#7f8c8d').pack(pady=(0, 15))
        
        tk.Label(dialog, text="Move to speaker:", 
                font=("Arial", 11, "bold"), bg='#f0f0f0').pack(pady=(0, 5))
        
        # Get current mapping to show proper names in dropdown
        mapping = self.get_speaker_mapping()
        available_speakers = []
        for orig_id in self.speaker_names:
            mapped_name = mapping.get(orig_id, orig_id)
            if mapped_name != orig_id:
                available_speakers.append(mapped_name)
            else:
                available_speakers.append(orig_id)
        
        # Add option to create new speaker
        available_speakers.append("+ Add New Speaker...")
        
        new_speaker_var = tk.StringVar()
        speaker_combo = ttk.Combobox(dialog, textvariable=new_speaker_var, 
                                   values=available_speakers, width=30)
        speaker_combo.pack(pady=10)
        speaker_combo.focus()
        
        tk.Label(dialog, text="(or type new speaker name)", 
                font=("Arial", 9), bg='#f0f0f0', fg='#95a5a6').pack()
        
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        def apply_rename():
            selected_name = new_speaker_var.get().strip()
            if not selected_name:
                messagebox.showerror("Error", "Please select or enter a speaker name.")
                return
            
            # Check if user selected "Add New Speaker"
            if selected_name == "+ Add New Speaker...":
                new_speaker = self.show_new_speaker_dialog()
                if new_speaker:
                    selected_name = new_speaker
                else:
                    return  # User cancelled
            
            # Convert from display name back to original speaker ID if needed
            target_speaker = selected_name
            reverse_mapping = {v: k for k, v in mapping.items()}
            if selected_name in reverse_mapping:
                target_speaker = reverse_mapping[selected_name]
            
            # Apply reassignment to selected segments
            changed_count = 0
            for i, segment in enumerate(self.segments):
                if i in self.selected_segments or segment.get('id', i) in self.selected_segments:
                    segment['speaker'] = target_speaker
                    changed_count += 1
            
            # Update speaker names set
            self.speaker_names.add(target_speaker)
            
            # Mark as having changes
            self.has_unsaved_changes = True
            
            # Refresh UI
            self.update_stats()
            self.setup_speaker_mapping()
            self.update_preview()
            self.clear_selection()
            
            dialog.destroy()
            messagebox.showinfo("✅ Reassignment Complete", 
                              f"Moved {changed_count} segments to '{selected_name}'!")
        
        tk.Button(button_frame, text="👥 Reassign", command=apply_rename,
                 bg='#3498db', fg='white', font=("Arial", 10, "bold"), 
                 relief=tk.FLAT, padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                 bg='#95a5a6', fg='white', font=("Arial", 10, "bold"), 
                 relief=tk.FLAT, padx=20, pady=5).pack(side=tk.LEFT, padx=5)
    
    def toggle_split_mode(self):
        """Toggle split mode on/off"""
        self.split_mode_active = not self.split_mode_active
        
        if self.split_mode_active:
            # Entering split mode
            self.split_text_btn.config(text="✂️ Split Mode: ON", bg='#e74c3c')
            messagebox.showinfo("Split Mode Activated", 
                              "Split Mode is now ON!\n\n" +
                              "📝 Instructions:\n" +
                              "1. Click on any text segment to split it\n" +
                              "2. Choose split point and assign speakers\n" +
                              "3. Click 'Split Mode: ON' again to exit\n\n" +
                              "💡 Tip: Text segments will highlight when you hover over them")
        else:
            # Exiting split mode
            self.split_text_btn.config(text="✂️ Split Mode: OFF", bg='#9b59b6')
        
        # Update the preview to show/hide split mode indicators
        self.update_preview()
    
    def handle_split_click(self, event):
        """Handle clicks in split mode - find segment and split immediately"""
        if not self.split_mode_active:
            return
        
        try:
            # Get click position
            click_index = self.text_area.index(f"@{event.x},{event.y}")
            
            # Get the clicked line content
            line_num = int(click_index.split('.')[0])
            click_char = int(click_index.split('.')[1])
            
            line_start = f"{line_num}.0"
            line_end = f"{line_num}.end"
            line_content = self.text_area.get(line_start, line_end)
            
            print(f"DEBUG: Clicked line {line_num}, char {click_char}")
            print(f"DEBUG: Line content: '{line_content}'")
            
            # Find which segment this line belongs to by checking line content
            target_segment = None
            segment_index = None
            
            for i, segment in enumerate(self.segments):
                segment_text = segment.get('text', '').strip()
                if segment_text and segment_text in line_content:
                    target_segment = segment
                    segment_index = i
                    print(f"DEBUG: Found segment {i}: '{segment_text[:50]}...'")
                    break
            
            if not target_segment:
                print("DEBUG: No segment found - line might be empty or timestamp only")
                return
            
            # Find where the text starts in the line (after ": ")
            text_start_marker = ": "
            text_start_pos = line_content.find(text_start_marker)
            if text_start_pos == -1:
                print("DEBUG: Could not find text start marker")
                return
            
            text_start_char = text_start_pos + len(text_start_marker)
            
            # Check if click was in the text portion
            if click_char < text_start_char:
                print("DEBUG: Click was before text content")
                return
            
            # Calculate position within the actual text
            char_pos_in_text = click_char - text_start_char
            segment_text = target_segment['text']
            char_pos_in_text = max(0, min(char_pos_in_text, len(segment_text)))
            
            print(f"DEBUG: Split position: {char_pos_in_text} in text '{segment_text}'")
            
            # Split the text immediately
            first_part = segment_text[:char_pos_in_text].strip()
            second_part = segment_text[char_pos_in_text:].strip()
            
            if not first_part or not second_part:
                messagebox.showwarning("Invalid Split", "Cannot split here - need text on both sides.")
                return
            
            # Perform the split immediately
            original_speaker = target_segment.get('speaker', 'Unknown')
            
            # Update original segment with first part
            target_segment['text'] = first_part
            
            # Create new segment for second part
            new_segment = {
                'id': len(self.segments),
                'speaker': original_speaker,  # Same speaker initially
                'text': second_part,
                'start': target_segment.get('start', 0) + 1,
                'end': target_segment.get('end', 5),
                'source': target_segment.get('source', 'Unknown')
            }
            
            # Insert new segment after original
            self.segments.insert(segment_index + 1, new_segment)
            
            # Mark as having changes
            self.has_unsaved_changes = True
            
            # Refresh UI
            self.update_stats()
            self.setup_speaker_mapping()
            self.update_preview()
            
            # Show success message
            messagebox.showinfo("✂️ Split Complete!", 
                              f"Text split into 2 parts:\n\n" +
                              f"Part 1: '{first_part[:50]}...'\n" +
                              f"Part 2: '{second_part[:50]}...'\n\n" +
                              f"Both assigned to: {original_speaker}")
            
        except Exception as e:
            print(f"DEBUG: Error in handle_split_click: {e}")
            import traceback
            traceback.print_exc()
    
    def show_split_at_position_dialog(self, segment_id, text, split_position):
        """Show split dialog with pre-determined split position"""
        # Find the segment
        segment = None
        for s in self.segments:
            if s.get('id', 0) == segment_id:
                segment = s
                break
        
        if not segment:
            messagebox.showerror("Error", "Segment not found for splitting.")
            return
        
        # Split the text at the specified position
        first_part = text[:split_position].strip()
        second_part = text[split_position:].strip()
        
        if not first_part or not second_part:
            messagebox.showwarning("Invalid Split", 
                                 "Cannot split here - both parts must contain text.")
            return
        
        # Create split dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("✂️ Confirm Split")
        dialog.geometry("600x400")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Title
        tk.Label(dialog, text="✂️ Confirm Text Split", 
                font=("Arial", 16, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=15)
        
        # Preview frame
        preview_frame = tk.LabelFrame(dialog, text="📋 Split Preview", 
                                    bg='#f0f0f0', font=("Arial", 11, "bold"))
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # First part
        tk.Label(preview_frame, text="Part 1:", font=("Arial", 10, "bold"), 
                bg='#f0f0f0', fg='#27ae60').pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        first_text = tk.Text(preview_frame, height=3, wrap=tk.WORD, font=("Arial", 10),
                           relief=tk.GROOVE, bd=1, bg='#e8f5e8')
        first_text.pack(fill=tk.X, padx=10, pady=(5, 10))
        first_text.insert('1.0', first_part)
        first_text.config(state=tk.DISABLED)
        
        # Second part
        tk.Label(preview_frame, text="Part 2:", font=("Arial", 10, "bold"), 
                bg='#f0f0f0', fg='#3498db').pack(anchor=tk.W, padx=10)
        
        second_text = tk.Text(preview_frame, height=3, wrap=tk.WORD, font=("Arial", 10),
                            relief=tk.GROOVE, bd=1, bg='#e8f4fd')
        second_text.pack(fill=tk.X, padx=10, pady=(5, 10))
        second_text.insert('1.0', second_part)
        second_text.config(state=tk.DISABLED)
        
        # Speaker assignment
        speaker_frame = tk.LabelFrame(dialog, text="👥 Assign Speakers", 
                                    bg='#f0f0f0', font=("Arial", 11, "bold"))
        speaker_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # First speaker
        first_row = tk.Frame(speaker_frame, bg='#f0f0f0')
        first_row.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(first_row, text="Part 1 Speaker:", font=("Arial", 10), 
                bg='#f0f0f0', width=15, anchor=tk.W).pack(side=tk.LEFT)
        
        first_speaker_var = tk.StringVar(value=segment.get('speaker', 'Unknown'))
        first_combo = ttk.Combobox(first_row, textvariable=first_speaker_var, 
                                 values=list(self.speaker_names), width=25)
        first_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Second speaker
        second_row = tk.Frame(speaker_frame, bg='#f0f0f0')
        second_row.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(second_row, text="Part 2 Speaker:", font=("Arial", 10), 
                bg='#f0f0f0', width=15, anchor=tk.W).pack(side=tk.LEFT)
        
        second_speaker_var = tk.StringVar(value=segment.get('speaker', 'Unknown'))
        second_combo = ttk.Combobox(second_row, textvariable=second_speaker_var, 
                                  values=list(self.speaker_names), width=25)
        second_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        def apply_split():
            try:
                # Get speakers
                first_speaker = first_speaker_var.get().strip()
                second_speaker = second_speaker_var.get().strip()
                
                if not first_speaker or not second_speaker:
                    messagebox.showerror("Missing Speakers", 
                                       "Please assign speakers to both parts.")
                    return
                
                # Perform the split
                original_index = self.segments.index(segment)
                
                # Update original segment with first part
                segment['text'] = first_part
                segment['speaker'] = first_speaker
                
                # Create new segment for second part
                new_segment = {
                    'id': len(self.segments),
                    'speaker': second_speaker,
                    'text': second_part,
                    'start': segment.get('start', 0) + 2,  # Offset by 2 seconds
                    'end': segment.get('end', 5),
                    'source': segment.get('source', 'Unknown')
                }
                
                # Insert new segment after original
                self.segments.insert(original_index + 1, new_segment)
                
                # Update speaker names
                self.speaker_names.add(first_speaker)
                self.speaker_names.add(second_speaker)
                
                # Mark as having changes
                self.has_unsaved_changes = True
                
                # Refresh UI
                self.update_stats()
                self.setup_speaker_mapping()
                self.update_preview()
                
                dialog.destroy()
                messagebox.showinfo("✅ Split Complete", 
                                  f"Text segment successfully split!\n\n" +
                                  f"Part 1: {first_speaker}\n" +
                                  f"Part 2: {second_speaker}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error splitting text: {e}")
        
        # Buttons
        split_button = tk.Button(button_frame, text="✂️ SPLIT", command=apply_split,
                               bg='#e74c3c', fg='white', font=("Arial", 12, "bold"), 
                               relief=tk.RAISED, bd=3, padx=30, pady=8, cursor='hand2')
        split_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                                bg='#95a5a6', fg='white', font=("Arial", 11, "bold"), 
                                relief=tk.FLAT, padx=20, pady=8)
        cancel_button.pack(side=tk.LEFT, padx=10)
    
    def show_simple_split_dialog(self, segment_id, text):
        """Show simplified split dialog - all in one step"""
        # Find the segment
        segment = None
        for s in self.segments:
            if s.get('id', 0) == segment_id:
                segment = s
                break
        
        if not segment:
            messagebox.showerror("Error", "Segment not found for splitting.")
            return
        
        # Create simple split dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Split Text Segment")
        dialog.geometry("700x500")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Title
        tk.Label(dialog, text="✂️ Split Text Segment", 
                font=("Arial", 16, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=10)
        
        # Instructions
        instruction_frame = tk.Frame(dialog, bg='#ecf0f1', relief=tk.GROOVE, bd=2)
        instruction_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        tk.Label(instruction_frame, text="📝 Instructions: Position cursor where you want to split, then click SPLIT", 
                font=("Arial", 11, "bold"), bg='#ecf0f1', fg='#2c3e50').pack(pady=8)
        
        # Text editing area
        text_frame = tk.Frame(dialog, bg='#f0f0f0')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(text_frame, text="Edit text and position cursor where you want to split:", 
                font=("Arial", 10, "bold"), bg='#f0f0f0').pack(anchor=tk.W, pady=(0, 5))
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, height=6, font=("Arial", 11), 
                             relief=tk.GROOVE, bd=2)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert('1.0', text)
        text_widget.focus()
        
        # Speaker assignment frame
        speaker_frame = tk.LabelFrame(dialog, text="👥 Assign Speakers", 
                                    bg='#f0f0f0', font=("Arial", 11, "bold"))
        speaker_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # First speaker
        first_row = tk.Frame(speaker_frame, bg='#f0f0f0')
        first_row.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(first_row, text="First part speaker:", font=("Arial", 10), 
                bg='#f0f0f0', width=15, anchor=tk.W).pack(side=tk.LEFT)
        
        first_speaker_var = tk.StringVar(value=segment.get('speaker', 'Unknown'))
        first_combo = ttk.Combobox(first_row, textvariable=first_speaker_var, 
                                 values=list(self.speaker_names), width=25)
        first_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Second speaker
        second_row = tk.Frame(speaker_frame, bg='#f0f0f0')
        second_row.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(second_row, text="Second part speaker:", font=("Arial", 10), 
                bg='#f0f0f0', width=15, anchor=tk.W).pack(side=tk.LEFT)
        
        second_speaker_var = tk.StringVar(value=segment.get('speaker', 'Unknown'))
        second_combo = ttk.Combobox(second_row, textvariable=second_speaker_var, 
                                  values=list(self.speaker_names), width=25)
        second_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        def perform_split():
            try:
                # Get cursor position
                cursor_pos = text_widget.index(tk.INSERT)
                char_pos = int(cursor_pos.split('.')[1])
                
                # Get the current text (in case user edited it)
                current_text = text_widget.get('1.0', tk.END).strip()
                
                if char_pos == 0 or char_pos >= len(current_text):
                    messagebox.showwarning("Invalid Split Position", 
                                         "Please position the cursor in the middle of the text.")
                    return
                
                # Split the text
                first_part = current_text[:char_pos].strip()
                second_part = current_text[char_pos:].strip()
                
                if not first_part or not second_part:
                    messagebox.showwarning("Invalid Split", 
                                         "Both parts must contain text after splitting.")
                    return
                
                # Get speakers
                first_speaker = first_speaker_var.get().strip()
                second_speaker = second_speaker_var.get().strip()
                
                if not first_speaker or not second_speaker:
                    messagebox.showerror("Missing Speakers", 
                                       "Please assign speakers to both parts.")
                    return
                
                # Perform the split
                original_index = self.segments.index(segment)
                
                # Update original segment with first part
                segment['text'] = first_part
                segment['speaker'] = first_speaker
                
                # Create new segment for second part
                new_segment = {
                    'id': len(self.segments),
                    'speaker': second_speaker,
                    'text': second_part,
                    'start': segment.get('start', 0) + 2,  # Offset by 2 seconds
                    'end': segment.get('end', 5),
                    'source': segment.get('source', 'Unknown')
                }
                
                # Insert new segment after original
                self.segments.insert(original_index + 1, new_segment)
                
                # Update speaker names
                self.speaker_names.add(first_speaker)
                self.speaker_names.add(second_speaker)
                
                # Mark as having changes
                self.has_unsaved_changes = True
                
                # Refresh UI
                self.update_stats()
                self.setup_speaker_mapping()
                self.update_preview()
                
                dialog.destroy()
                messagebox.showinfo("✅ Split Successful", 
                                  f"Text segment split into 2 parts!\n\n" +
                                  f"Part 1: {first_speaker}\n" +
                                  f"Part 2: {second_speaker}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error splitting text: {e}")
        
        # Big, obvious SPLIT button
        split_button = tk.Button(button_frame, text="✂️ SPLIT", command=perform_split,
                               bg='#e74c3c', fg='white', font=("Arial", 14, "bold"), 
                               relief=tk.RAISED, bd=3, padx=30, pady=10, cursor='hand2')
        split_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                                bg='#95a5a6', fg='white', font=("Arial", 11, "bold"), 
                                relief=tk.FLAT, padx=20, pady=8)
        cancel_button.pack(side=tk.LEFT, padx=10)
    
    def show_split_dialog(self, segment_id, text):
        """Show dialog for splitting text"""
        # Find the segment
        segment = None
        for s in self.segments:
            if s.get('id', 0) == segment_id:
                segment = s
                break
        
        if not segment:
            messagebox.showerror("Error", "Segment not found for splitting.")
            return
        
        # Create split dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Split Text Segment")
        dialog.geometry("600x400")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Split this text into multiple segments:", 
                font=("Arial", 12, "bold"), bg='#f0f0f0').pack(pady=(10, 5))
        
        # Text display
        text_frame = tk.Frame(dialog, bg='#f0f0f0')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, height=8, font=("Arial", 11))
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert('1.0', text)
        
        tk.Label(dialog, text="Instructions: Select text and click 'Split Here' to create a split point", 
                font=("Arial", 10), bg='#f0f0f0', fg='#7f8c8d').pack(pady=5)
        
        # Split controls
        split_frame = tk.Frame(dialog, bg='#f0f0f0')
        split_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def split_at_cursor():
            try:
                cursor_pos = text_widget.index(tk.INSERT)
                char_pos = int(cursor_pos.split('.')[1])
                
                if char_pos == 0 or char_pos >= len(text):
                    messagebox.showwarning("Invalid Split", "Please place cursor in the middle of the text.")
                    return
                
                # Split the text
                first_part = text[:char_pos].strip()
                second_part = text[char_pos:].strip()
                
                if not first_part or not second_part:
                    messagebox.showwarning("Invalid Split", "Both parts must contain text.")
                    return
                
                # Show speaker assignment dialog
                self.show_speaker_assignment_dialog(segment, first_part, second_part, dialog)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error splitting text: {e}")
        
        tk.Button(split_frame, text="Split Here", command=split_at_cursor,
                 bg='#9b59b6', fg='white', font=("Arial", 10, "bold"), 
                 relief=tk.FLAT, padx=20, pady=5).pack(side=tk.LEFT)
        
        tk.Button(split_frame, text="Cancel", command=dialog.destroy,
                 bg='#95a5a6', fg='white', font=("Arial", 10, "bold"), 
                 relief=tk.FLAT, padx=20, pady=5).pack(side=tk.RIGHT)
    
    def show_speaker_assignment_dialog(self, original_segment, first_part, second_part, parent_dialog):
        """Show dialog for assigning speakers to split segments"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Assign Speakers to Split Segments")
        dialog.geometry("500x300")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Assign speakers to each part:", 
                font=("Arial", 12, "bold"), bg='#f0f0f0').pack(pady=10)
        
        # First segment
        first_frame = tk.LabelFrame(dialog, text="First Segment", bg='#f0f0f0', font=("Arial", 10, "bold"))
        first_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(first_frame, text=f'"{first_part}"', bg='#f0f0f0', wraplength=400, 
                font=("Arial", 10), justify=tk.LEFT).pack(pady=5)
        
        first_speaker_var = tk.StringVar(value=original_segment.get('speaker', 'Unknown'))
        first_combo = ttk.Combobox(first_frame, textvariable=first_speaker_var, 
                                 values=list(self.speaker_names), width=30)
        first_combo.pack(pady=5)
        
        # Second segment  
        second_frame = tk.LabelFrame(dialog, text="Second Segment", bg='#f0f0f0', font=("Arial", 10, "bold"))
        second_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(second_frame, text=f'"{second_part}"', bg='#f0f0f0', wraplength=400,
                font=("Arial", 10), justify=tk.LEFT).pack(pady=5)
        
        second_speaker_var = tk.StringVar(value=original_segment.get('speaker', 'Unknown'))
        second_combo = ttk.Combobox(second_frame, textvariable=second_speaker_var, 
                                  values=list(self.speaker_names), width=30)
        second_combo.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        def apply_split():
            first_speaker = first_speaker_var.get().strip()
            second_speaker = second_speaker_var.get().strip()
            
            if not first_speaker or not second_speaker:
                messagebox.showerror("Error", "Please assign speakers to both segments.")
                return
            
            # Create new segments
            original_index = self.segments.index(original_segment)
            
            # Update original segment with first part
            original_segment['text'] = first_part
            original_segment['speaker'] = first_speaker
            
            # Create new segment for second part
            new_segment = {
                'id': len(self.segments),
                'speaker': second_speaker,
                'text': second_part,
                'start': original_segment.get('start', 0) + 2,  # Offset by 2 seconds
                'end': original_segment.get('end', 5),
                'source': original_segment.get('source', 'Unknown')
            }
            
            # Insert new segment after original
            self.segments.insert(original_index + 1, new_segment)
            
            # Update speaker names
            self.speaker_names.add(first_speaker)
            self.speaker_names.add(second_speaker)
            
            # Refresh UI
            self.setup_speaker_mapping()
            self.update_preview()
            
            dialog.destroy()
            parent_dialog.destroy()
            messagebox.showinfo("Success", "Text segment split successfully!")
        
        tk.Button(button_frame, text="Apply Split", command=apply_split,
                 bg='#2ecc71', fg='white', font=("Arial", 10, "bold"), 
                 relief=tk.FLAT, padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                 bg='#e74c3c', fg='white', font=("Arial", 10, "bold"), 
                 relief=tk.FLAT, padx=20, pady=5).pack(side=tk.LEFT, padx=5)

    def preview_changes(self):
        """Preview the changes"""
        self.update_preview()
        messagebox.showinfo("Preview Updated", "✅ Enhanced transcription preview updated!\n\n" +
                          "🎨 Features active:\n" +
                          "• Color-coded speaker names\n" +
                          "• Emotional tone indicators\n" +
                          "• Multi-select checkboxes\n" +
                          "• Text splitting (Ctrl+Click)\n" +
                          "• Bulk operations toolbar")
    
    def save_corrected(self):
        """Save the corrected transcription"""
        if not self.transcription_data:
            messagebox.showerror("Error", "No transcription data to save")
            return
        
        # Get ALL changes (UI mapping + bulk renames)
        all_changes = self.get_all_changes()
        
        if not all_changes and not self.has_unsaved_changes:
            messagebox.showinfo("No Changes", "No speaker name changes detected.")
            return
        
        # Create corrected data using current segments (which include bulk changes)
        corrected_data = self.transcription_data.copy()
        corrected_data['segments'] = self.segments.copy()  # Use current segments
        
        # Also apply any UI mapping changes
        corrected_count = 0
        ui_mapping = self.get_speaker_mapping()
        
        for segment in corrected_data.get('segments', []):
            original_speaker = segment.get('speaker')
            if original_speaker in ui_mapping:
                segment['speaker'] = ui_mapping[original_speaker]
                corrected_count += 1
        
        # Count total changes (bulk + UI)
        total_changes = len(all_changes)
        
        # Save to new file
        if self.transcription_file:
            input_path = Path(self.transcription_file)
            output_path = input_path.parent / f"{input_path.stem}_corrected{input_path.suffix}"
        else:
            output_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not output_path:
                return
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(corrected_data, f, indent=2, ensure_ascii=False)
            
            # Reset change tracking
            import copy
            self.original_segments = copy.deepcopy(self.segments)
            self.has_unsaved_changes = False
            self.update_stats()
            
            messagebox.showinfo("Success", f"Corrected transcription saved!\n\n"
                               f"File: {output_path}\n"
                               f"Total corrections: {total_changes}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save corrected transcription: {e}")
    
    def send_feedback(self):
        """Send speaker corrections back to Oreja for learning"""
        mapping = self.get_speaker_mapping()
        
        if not mapping:
            messagebox.showinfo("No Changes", "No speaker corrections to send.")
            return
        
        try:
            success_count = 0
            error_count = 0
            
            # Send each mapping individually using the name_mapping endpoint
            for old_speaker_id, new_speaker_name in mapping.items():
                try:
                    response = requests.post(
                        f"http://127.0.0.1:8000/speakers/name_mapping",
                        params={
                            "old_speaker_id": old_speaker_id,
                            "new_speaker_name": new_speaker_name
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        error_count += 1
                        print(f"Failed to map {old_speaker_id} -> {new_speaker_name}: {response.text}")
                        
                except Exception as e:
                    error_count += 1
                    print(f"Error mapping {old_speaker_id} -> {new_speaker_name}: {e}")
            
            # Show results
            if success_count > 0 and error_count == 0:
                messagebox.showinfo("Success", f"Successfully sent {success_count} speaker corrections to Oreja!\n\n"
                                   "The system will learn from your corrections to improve "
                                   "future speaker recognition.")
            elif success_count > 0:
                messagebox.showwarning("Partial Success", f"Sent {success_count} corrections successfully.\n"
                                      f"{error_count} corrections failed.\n\n"
                                      "Check console for details.")
            else:
                messagebox.showerror("Error", f"Failed to send speaker corrections.\n"
                                    f"All {error_count} corrections failed.\n\n"
                                    "Make sure the Oreja backend is running.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send feedback to Oreja: {e}")
    
    def open_file(self):
        """Open a transcription file"""
        file_path = filedialog.askopenfilename(
            title="Select Transcription File",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_transcription(file_path)
    
    def update_stats(self):
        """Update the statistics panel"""
        if not self.segments:
            self.stats_label.config(text="No transcription loaded")
            return
        
        total_segments = len(self.segments)
        unique_speakers = len(self.speaker_names)
        
        # Calculate total duration if available
        total_duration = 0
        if self.segments:
            for segment in self.segments:
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                total_duration += (end - start)
        
        duration_text = f"{int(total_duration // 60)}:{int(total_duration % 60):02d}" if total_duration > 0 else "Unknown"
        
        stats_text = f"""📊 Total segments: {total_segments}
👥 Unique speakers: {unique_speakers}  
⏱️ Duration: {duration_text}
📝 Changes: {'Yes' if self.has_unsaved_changes else 'None'}"""
        
        self.stats_label.config(text=stats_text)
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def open_transcription_editor(transcription_file=None):
    """Open the transcription editor GUI"""
    editor = TranscriptionEditor(transcription_file)
    editor.run()


if __name__ == "__main__":
    import sys
    
    transcription_file = sys.argv[1] if len(sys.argv) > 1 else None
    open_transcription_editor(transcription_file) 