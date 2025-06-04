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
        self.root = tk.Tk()
        self.root.title("🎙️ Oreja - Enhanced Transcription Editor")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        self.transcription_file = transcription_file
        self.transcription_data = {}
        self.segments = []
        self.speaker_names = set()
        self.selected_segments = set()  # For multi-select
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
        """Setup the enhanced GUI interface"""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg='#f0f0f0', padx=15, pady=15)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title with enhanced styling
        title_frame = tk.Frame(main_container, bg='#f0f0f0')
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_font = font.Font(family="Arial", size=18, weight="bold")
        title_label = tk.Label(title_frame, text="🎙️ Enhanced Transcription Editor", 
                              font=title_font, bg='#f0f0f0', fg='#2c3e50')
        title_label.pack()
        
        # File info with better styling
        self.file_label = tk.Label(title_frame, text="No file loaded", 
                                  font=("Arial", 11), bg='#f0f0f0', fg='#7f8c8d')
        self.file_label.pack(pady=(5, 0))
        
        # Enhanced multi-select toolbar
        toolbar_frame = tk.Frame(main_container, bg='#3498db', relief=tk.RAISED, bd=1)
        toolbar_frame.pack(fill=tk.X, pady=(0, 10))
        
        toolbar_inner = tk.Frame(toolbar_frame, bg='#3498db', padx=10, pady=8)
        toolbar_inner.pack(fill=tk.X)
        
        # Multi-select controls
        tk.Label(toolbar_inner, text="🔧 Multi-Select Tools:", font=("Arial", 10, "bold"), 
                bg='#3498db', fg='white').pack(side=tk.LEFT)
        
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
        
        # Split text button
        self.split_text_btn = tk.Button(toolbar_inner, text="✂️ Split Selected", command=self.split_selected_text,
                                       bg='#9b59b6', fg='white', font=("Arial", 9, "bold"),
                                       relief=tk.FLAT, padx=15, pady=5, state=tk.DISABLED)
        self.split_text_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content area with enhanced layout
        content_frame = tk.Frame(main_container, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Enhanced speaker mapping panel
        speaker_panel = tk.LabelFrame(content_frame, text="🎨 Color-Coded Speaker Mapping", 
                                    font=("Arial", 12, "bold"), bg='#ecf0f1', fg='#2c3e50',
                                    relief=tk.GROOVE, bd=2, padx=10, pady=10)
        speaker_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        speaker_panel.columnconfigure(1, weight=1)
        
        # Speaker mapping headers
        header_frame = tk.Frame(speaker_panel, bg='#ecf0f1')
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        tk.Label(header_frame, text="Original Speaker", font=("Arial", 10, "bold"), 
                bg='#ecf0f1', fg='#34495e').grid(row=0, column=0, sticky=tk.W)
        tk.Label(header_frame, text="Corrected Name", font=("Arial", 10, "bold"), 
                bg='#ecf0f1', fg='#34495e').grid(row=0, column=1, sticky=tk.W, padx=(120, 0))
        
        # Scrollable speaker mapping area
        self.speaker_canvas = tk.Canvas(speaker_panel, height=300, bg='white', highlightthickness=0)
        self.speaker_scrollbar = ttk.Scrollbar(speaker_panel, orient="vertical", command=self.speaker_canvas.yview)
        self.speaker_frame_inner = tk.Frame(self.speaker_canvas, bg='white')
        
        self.speaker_canvas.configure(yscrollcommand=self.speaker_scrollbar.set)
        self.speaker_canvas.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        self.speaker_scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S), pady=(5, 0))
        
        self.speaker_canvas.create_window((0, 0), window=self.speaker_frame_inner, anchor="nw")
        
        # Enhanced transcription preview with emotional indicators
        preview_panel = tk.LabelFrame(content_frame, text="📝 Enhanced Transcription Preview", 
                                    font=("Arial", 12, "bold"), bg='#ecf0f1', fg='#2c3e50',
                                    relief=tk.GROOVE, bd=2, padx=10, pady=10)
        preview_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(1, weight=1)
        
        # Preview controls
        preview_controls = tk.Frame(preview_panel, bg='#ecf0f1')
        preview_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        tk.Label(preview_controls, text="💡 Features: ✓ Color-coded speakers ✓ Emotional indicators ✓ Multi-select ✓ Text splitting", 
                font=("Arial", 9), bg='#ecf0f1', fg='#7f8c8d', wraplength=400).pack(anchor=tk.W)
        
        # Scrollable transcription area
        text_frame = tk.Frame(preview_panel, bg='#ecf0f1')
        text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.text_area = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 11), 
                               bg='white', fg='#2c3e50', selectbackground='#3498db',
                               relief=tk.SUNKEN, bd=1)
        text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=text_scrollbar.set)
        
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
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
            entry.bind('<KeyRelease>', self.on_name_change)
            
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
        """Handle speaker name changes"""
        self.update_preview()
    
    def get_speaker_mapping(self):
        """Get current speaker name mapping"""
        mapping = {}
        for original_id, entry in self.speaker_entries.items():
            new_name = entry.get().strip()
            if new_name and new_name != original_id:
                mapping[original_id] = new_name
        return mapping
    
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
            
            # Create checkbox for multi-select
            checkbox_var = tk.BooleanVar()
            checkbox = tk.Checkbutton(self.text_area, variable=checkbox_var, 
                                    command=lambda sid=segment_id: self.toggle_segment_selection(sid),
                                    bg='white', activebackground='#3498db')
            
            # Insert checkbox into text area
            self.text_area.window_create(tk.END, window=checkbox)
            self.segment_checkboxes[segment_id] = (checkbox, checkbox_var)
            
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
            
            # Make the line selectable for text splitting
            line_start = f"{self.text_area.index(tk.END).split('.')[0]}.0"
            line_end = f"{int(self.text_area.index(tk.END).split('.')[0])-1}.end"
            
            # Add click binding for text splitting
            def make_split_handler(seg_id, line_text):
                def split_handler(event):
                    if event.state & 0x4:  # Ctrl+Click
                        self.show_split_dialog(seg_id, line_text)
                return split_handler
            
            self.text_area.tag_add(f"segment_{segment_id}", line_start, line_end)
            self.text_area.tag_bind(f"segment_{segment_id}", "<Control-Button-1>", 
                                  make_split_handler(segment_id, text))
            self.text_area.tag_configure(f"segment_{segment_id}", background='#f8f9fa')
        
        self.text_area.configure(state=tk.DISABLED)
    
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
        self.bulk_rename_btn.config(text=f"🏷 Bulk Rename ({count})", 
                                   state=tk.NORMAL if count > 0 else tk.DISABLED)
        self.split_text_btn.config(state=tk.NORMAL if count > 0 else tk.DISABLED)
    
    def bulk_rename_speakers(self):
        """Bulk rename selected speakers"""
        if not self.selected_segments:
            messagebox.showwarning("No Selection", "Please select segments to rename.")
            return
        
        # Create dialog for bulk rename
        dialog = tk.Toplevel(self.root)
        dialog.title("Bulk Rename Speakers")
        dialog.geometry("400x200")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        tk.Label(dialog, text=f"Rename {len(self.selected_segments)} selected segments to:", 
                font=("Arial", 12, "bold"), bg='#f0f0f0').pack(pady=(20, 10))
        
        new_speaker_var = tk.StringVar()
        speaker_combo = ttk.Combobox(dialog, textvariable=new_speaker_var, 
                                   values=list(self.speaker_names), width=30)
        speaker_combo.pack(pady=10)
        speaker_combo.focus()
        
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        def apply_rename():
            new_speaker = new_speaker_var.get().strip()
            if not new_speaker:
                messagebox.showerror("Error", "Please enter a speaker name.")
                return
            
            # Apply rename to selected segments
            for i, segment in enumerate(self.segments):
                if i in self.selected_segments or segment.get('id', i) in self.selected_segments:
                    segment['speaker'] = new_speaker
            
            # Update speaker names set
            self.speaker_names.add(new_speaker)
            
            # Refresh UI
            self.setup_speaker_mapping()
            self.update_preview()
            self.clear_selection()
            
            dialog.destroy()
            messagebox.showinfo("Success", f"Renamed {len(self.selected_segments)} segments to '{new_speaker}'!")
        
        tk.Button(button_frame, text="Apply Rename", command=apply_rename,
                 bg='#2ecc71', fg='white', font=("Arial", 10, "bold"), 
                 relief=tk.FLAT, padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                 bg='#e74c3c', fg='white', font=("Arial", 10, "bold"), 
                 relief=tk.FLAT, padx=20, pady=5).pack(side=tk.LEFT, padx=5)
    
    def split_selected_text(self):
        """Split selected text segments"""
        if not self.selected_segments:
            messagebox.showwarning("No Selection", "Please select segments to split.")
            return
        
        messagebox.showinfo("Text Splitting", 
                          "To split text:\n\n" +
                          "1. Select the segment(s) you want to split\n" +
                          "2. Hold Ctrl and click where you want to split the text\n" +
                          "3. Choose new speakers for each part\n\n" +
                          "Selected segments are ready for splitting!")
    
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
        
        mapping = self.get_speaker_mapping()
        
        if not mapping:
            messagebox.showinfo("No Changes", "No speaker name changes detected.")
            return
        
        # Apply corrections to transcription data
        corrected_data = self.transcription_data.copy()
        corrected_count = 0
        
        for segment in corrected_data.get('segments', []):
            original_speaker = segment.get('speaker')
            if original_speaker in mapping:
                segment['speaker'] = mapping[original_speaker]
                corrected_count += 1
        
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
            
            messagebox.showinfo("Success", f"Corrected transcription saved!\n\n"
                               f"File: {output_path}\n"
                               f"Corrections applied: {corrected_count}")
            
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