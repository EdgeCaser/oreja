#!/usr/bin/env python3
"""
Post-Transcription Editor GUI
Opens automatically after transcription to edit speaker names
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from pathlib import Path
import requests
from datetime import datetime


class TranscriptionEditor:
    def __init__(self, transcription_file=None):
        self.root = tk.Tk()
        self.root.title("Oreja - Transcription Editor")
        self.root.geometry("1000x700")
        
        self.transcription_file = transcription_file
        self.transcription_data = {}
        self.segments = []
        self.speaker_names = set()
        
        self.setup_ui()
        
        if transcription_file:
            self.load_transcription(transcription_file)
    
    def setup_ui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎙️ Transcription Editor", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # File info
        self.file_label = ttk.Label(main_frame, text="No file loaded", 
                                   font=("Arial", 10))
        self.file_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Speaker mapping frame
        speaker_frame = ttk.LabelFrame(main_frame, text="Speaker Name Mapping", padding="5")
        speaker_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        speaker_frame.columnconfigure(1, weight=1)
        
        ttk.Label(speaker_frame, text="Original ID").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(speaker_frame, text="Corrected Name").grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Scrollable frame for speaker mappings
        self.speaker_canvas = tk.Canvas(speaker_frame, height=200)
        self.speaker_scrollbar = ttk.Scrollbar(speaker_frame, orient="vertical", command=self.speaker_canvas.yview)
        self.speaker_frame_inner = ttk.Frame(self.speaker_canvas)
        
        self.speaker_canvas.configure(yscrollcommand=self.speaker_scrollbar.set)
        self.speaker_canvas.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        self.speaker_scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S), pady=(5, 0))
        
        self.speaker_canvas.create_window((0, 0), window=self.speaker_frame_inner, anchor="nw")
        
        # Transcription text area
        text_frame = ttk.LabelFrame(main_frame, text="Transcription Preview", padding="5")
        text_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.text_area = tk.Text(text_frame, wrap=tk.WORD, width=60, height=20)
        text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=text_scrollbar.set)
        
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Open File", command=self.open_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Preview Changes", command=self.preview_changes).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Save Corrected", command=self.save_corrected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Send Feedback to Oreja", command=self.send_feedback).pack(side=tk.LEFT, padx=(0, 5))
        
        self.speaker_entries = {}
    
    def load_transcription(self, file_path):
        """Load transcription from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.transcription_data = json.load(f)
            
            self.transcription_file = file_path
            self.file_label.config(text=f"File: {Path(file_path).name}")
            
            # Extract segments and speaker names
            self.segments = self.transcription_data.get('segments', [])
            self.speaker_names = set()
            
            for segment in self.segments:
                speaker = segment.get('speaker', 'Unknown')
                self.speaker_names.add(speaker)
            
            self.setup_speaker_mapping()
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load transcription: {e}")
    
    def setup_speaker_mapping(self):
        """Setup speaker name mapping interface"""
        # Clear existing entries
        for widget in self.speaker_frame_inner.winfo_children():
            widget.destroy()
        
        self.speaker_entries = {}
        
        row = 0
        for speaker_id in sorted(self.speaker_names):
            # Original speaker ID label
            ttk.Label(self.speaker_frame_inner, text=speaker_id).grid(
                row=row, column=0, sticky=tk.W, pady=2)
            
            # Entry for corrected name
            entry = ttk.Entry(self.speaker_frame_inner, width=30)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
            entry.insert(0, speaker_id)  # Default to original name
            entry.bind('<KeyRelease>', self.on_name_change)
            
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
        """Update the transcription preview"""
        mapping = self.get_speaker_mapping()
        
        self.text_area.delete('1.0', tk.END)
        
        for segment in self.segments:
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            
            # Apply speaker mapping
            display_speaker = mapping.get(speaker, speaker)
            
            # Format timestamp
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            
            # Add to preview
            self.text_area.insert(tk.END, f"{timestamp} {display_speaker}: {text}\n\n")
    
    def preview_changes(self):
        """Preview the changes"""
        self.update_preview()
        messagebox.showinfo("Preview Updated", "Transcription preview has been updated with your changes.")
    
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
            title="Select Transcription JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
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