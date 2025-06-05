#!/usr/bin/env python3
"""
Oreja User Embedding Trainer
A module for creating and updating speaker embeddings for specific users
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging

# Audio recording dependencies
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("Audio recording dependencies not available. Install sounddevice and soundfile.")

# Import existing database and embedding modules
from speaker_database_v2 import EnhancedSpeakerDatabase
from speaker_embeddings import OfflineSpeakerEmbeddingManager

logger = logging.getLogger(__name__)

class UserEmbeddingTrainer:
    """GUI module for training and updating user embeddings"""
    
    # Standard text for consistent recording across users
    # "The Story of the Hare Who Lost His Spectacles" from Jethro Tull's "A Passion Play"
    STANDARD_TEXT = """
    This is the story of the hare who lost his spectacles.
    Owl loved to rest quietly whilst no one was watching.
    Sitting on a fence one day,
    He was surprised when suddenly a kangaroo ran close by.
    Now this may not seem strange, but when Owl overheard Kangaroo whisper to no one in
    Particular,
    "The hare has lost his spectacles, " well, he began to wonder.
    Presently, the moon appeared from behind a cloud and there, lying on the grass was hare.
    In the stream that flowed by the grass a newt.
    And sitting astride a twig of a bush a bee. Ostensibly motionless, the hare was trembling with
    Excitement, for without his spectacles he was completely helpless.
    Where were his spectacles?
    Could someone have stolen them?
    Had he mislaid them?
    What was he to do?
    Bee wanted to help, and thinking he had the answer began:
    "You probably ate them thinking they were a carrot."
    "No!" interrupted Owl, who was wise.
    "I have good eye-sight, insight, and foresight. How could an intelligent hare make such a silly mistake?"
    But all this time, Owl had been sitting on the fence, scowling!
    Kangaroo were hopping mad at this sort of talk.
    She thought herself far superior in intelligence to the others.
    She was their leader, their guru.
    She had the answer: "Hare, you must go in search of the optician."
    But then she realized that Hare was completely helpless without his spectacles.
    And so, Kangaroo loudly proclaimed, "I can't send Hare in search of anything!"
    "You can guru, you can!" shouted Newt.
    "You can send him with Owl."
    But Owl had gone to sleep.
    Newt knew too much to be stopped by so small a problem
    "You can take him in your pouch."
    But alas, Hare was much too big to fit into
    Kangaroo's pouch.
    All this time, it had been quite plain to hare that the others knew nothing about spectacles.
    As for all their tempting ideas, well Hare didn't care.
    The lost spectacles were his own affair.
    And after all, Hare did have a spare a-pair. A-pair.
    """
    
    def __init__(self, parent_notebook, backend_url="http://127.0.0.1:8000", shared_db=None):
        self.parent_notebook = parent_notebook
        self.backend_url = backend_url
        
        # Initialize database connections
        if shared_db is not None:
            # Use the shared database instance from the main dashboard
            self.speaker_db = shared_db
        else:
            # Create a new instance if no shared database provided
            self.speaker_db = EnhancedSpeakerDatabase()
        self.embedding_manager = OfflineSpeakerEmbeddingManager()
        
        # Recording state
        self.is_recording = False
        self.current_recording = None
        self.temp_audio_file = None
        self.selected_user_id = None
        self.confidence_before = 0.0
        self.confidence_after = 0.0
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        
        # Setup the GUI tab
        self.setup_ui()
        self.refresh_user_list()
        
    def setup_ui(self):
        """Setup the user embedding trainer interface"""
        # Create main frame
        self.main_frame = ttk.Frame(self.parent_notebook)
        self.parent_notebook.add(self.main_frame, text="🎯 User Training")
        
        # Title
        title_frame = ttk.Frame(self.main_frame)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(title_frame, text="🎯 User Embedding Trainer", 
                 font=('Arial', 16, 'bold')).pack(side=tk.LEFT)
        
        ttk.Button(title_frame, text="🔄 Refresh", 
                  command=self.refresh_user_list).pack(side=tk.RIGHT)
        
        # Create main paned window
        paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel: User selection
        self.setup_user_selection_panel(paned)
        
        # Right panel: Training interface
        self.setup_training_panel(paned)
        
    def setup_user_selection_panel(self, parent):
        """Setup the user selection panel"""
        user_frame = ttk.LabelFrame(parent, text="1. Select User", padding=10)
        parent.add(user_frame, weight=1)
        
        # User list
        list_frame = ttk.Frame(user_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for users
        self.user_tree = ttk.Treeview(list_frame, 
                                     columns=('samples', 'confidence', 'last_seen'), 
                                     show='tree headings', height=15)
        
        # Configure columns
        self.user_tree.heading('#0', text='User Name')
        self.user_tree.heading('samples', text='Samples')
        self.user_tree.heading('confidence', text='Avg Confidence')
        self.user_tree.heading('last_seen', text='Last Seen')
        
        self.user_tree.column('#0', width=200)
        self.user_tree.column('samples', width=80)
        self.user_tree.column('confidence', width=100)
        self.user_tree.column('last_seen', width=120)
        
        # Scrollbar
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.user_tree.yview)
        self.user_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.user_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection
        self.user_tree.bind('<<TreeviewSelect>>', self.on_user_select)
        
        # Create new user section
        new_user_frame = ttk.LabelFrame(user_frame, text="Create New User", padding=5)
        new_user_frame.pack(fill=tk.X, pady=(10, 0))
        
        create_frame = ttk.Frame(new_user_frame)
        create_frame.pack(fill=tk.X)
        
        ttk.Label(create_frame, text="Name:").pack(side=tk.LEFT)
        self.new_user_name = tk.StringVar()
        self.new_user_entry = ttk.Entry(create_frame, textvariable=self.new_user_name, width=20)
        self.new_user_entry.pack(side=tk.LEFT, padx=(5, 10), fill=tk.X, expand=True)
        
        ttk.Button(create_frame, text="➕ Create User", 
                  command=self.create_new_user).pack(side=tk.LEFT)
        
        # Bind Enter key to create user
        self.new_user_entry.bind('<Return>', lambda e: self.create_new_user())
        
        # User info display
        info_frame = ttk.LabelFrame(user_frame, text="User Information", padding=5)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.user_info = tk.Text(info_frame, height=8, font=('Consolas', 9))
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.user_info.yview)
        self.user_info.configure(yscrollcommand=info_scroll.set)
        
        self.user_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_training_panel(self, parent):
        """Setup the training interface panel"""
        training_frame = ttk.LabelFrame(parent, text="2. Record & Train", padding=10)
        parent.add(training_frame, weight=2)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(training_frame, text="Instructions", padding=10)
        instructions_frame.pack(fill=tk.X, pady=(0, 10))
        
        instructions_text = """Please read the following text clearly and naturally:
        
{}
        
Tips:
• Speak at a normal conversational pace
• Ensure you're in a quiet environment  
• Speak directly towards your microphone
• Try to match your natural speaking style

Getting Started:
• Select an existing user OR create a new one
• New users will need their first voice sample to establish their profile
• Existing users can improve their recognition accuracy""".format(self.STANDARD_TEXT.strip())
        
        instructions_label = tk.Text(instructions_frame, height=12, wrap=tk.WORD, 
                                   font=('Arial', 10))
        instructions_label.insert(tk.END, instructions_text)
        instructions_label.config(state=tk.DISABLED)  # Disable after inserting text
        instructions_label.pack(fill=tk.BOTH, expand=True)
        
        # Recording controls
        controls_frame = ttk.LabelFrame(training_frame, text="Recording Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Audio input selection
        device_frame = ttk.Frame(controls_frame)
        device_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(device_frame, text="Audio Input:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, state="readonly")
        self.device_combo.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        # Populate audio devices
        self.populate_audio_devices()
        
        # Recording buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill=tk.X)
        
        self.record_button = ttk.Button(button_frame, text="🎤 Start Recording", 
                                      command=self.toggle_recording, state=tk.DISABLED)
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.upload_button = ttk.Button(button_frame, text="📁 Upload Audio File", 
                                      command=self.upload_audio_file, state=tk.DISABLED)
        self.upload_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Recording status
        self.status_var = tk.StringVar(value="Select a user to begin")
        self.status_label = ttk.Label(button_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Results display
        results_frame = ttk.LabelFrame(training_frame, text="Training Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Transcription display
        transcription_frame = ttk.LabelFrame(results_frame, text="What You Said (Transcription)", padding=5)
        transcription_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.transcription_text = tk.Text(transcription_frame, height=4, wrap=tk.WORD, 
                                        font=('Arial', 10), state=tk.DISABLED)
        transcription_scroll = ttk.Scrollbar(transcription_frame, orient=tk.VERTICAL, command=self.transcription_text.yview)
        self.transcription_text.configure(yscrollcommand=transcription_scroll.set)
        
        self.transcription_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        transcription_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Before/After confidence display
        confidence_frame = ttk.Frame(results_frame)
        confidence_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(confidence_frame, text="Confidence Before:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.confidence_before_var = tk.StringVar(value="N/A")
        ttk.Label(confidence_frame, textvariable=self.confidence_before_var, 
                 font=('Arial', 10, 'bold')).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(confidence_frame, text="Confidence After:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.confidence_after_var = tk.StringVar(value="N/A")
        ttk.Label(confidence_frame, textvariable=self.confidence_after_var, 
                 font=('Arial', 10, 'bold')).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(confidence_frame, text="Improvement:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.improvement_var = tk.StringVar(value="N/A")
        self.improvement_label = ttk.Label(confidence_frame, textvariable=self.improvement_var, 
                                         font=('Arial', 10, 'bold'))
        self.improvement_label.grid(row=2, column=1, sticky=tk.W)
        
        confidence_frame.columnconfigure(1, weight=1)
        
        # Action buttons
        action_frame = ttk.Frame(results_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.train_again_button = ttk.Button(action_frame, text="🔄 Train Again", 
                                           command=self.prepare_for_next_training, state=tk.DISABLED)
        self.train_again_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="📊 View User Analytics", 
                  command=self.view_user_analytics).pack(side=tk.LEFT)
        
    def populate_audio_devices(self):
        """Populate the audio device selection dropdown"""
        if not AUDIO_AVAILABLE:
            self.device_combo['values'] = ["Audio recording not available"]
            self.device_combo.set("Audio recording not available")
            return
            
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = f"{device['name']} (ID: {i})"
                    input_devices.append((device_name, i))
            
            if input_devices:
                device_names = [name for name, _ in input_devices]
                self.device_combo['values'] = device_names
                self.device_combo.set(device_names[0])  # Select first device by default
                self.device_ids = {name: device_id for name, device_id in input_devices}
            else:
                self.device_combo['values'] = ["No input devices found"]
                self.device_combo.set("No input devices found")
                
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
            self.device_combo['values'] = ["Error detecting devices"]
            self.device_combo.set("Error detecting devices")
    
    def refresh_user_list(self):
        """Refresh the list of users from the database"""
        try:
            # Clear existing items
            for item in self.user_tree.get_children():
                self.user_tree.delete(item)
            
            # Get all speakers from database using the same method as the main dashboard
            speakers = self.speaker_db.get_all_speakers()
            logger.info(f"Found {len(speakers)} speakers in database")
            
            # Process each speaker with robust error handling
            for speaker in speakers:
                try:
                    # Use .get() with fallbacks for safety
                    speaker_id = speaker.get('speaker_id', 'unknown')
                    display_name = speaker.get('display_name', 'Unknown User')
                    embedding_count = speaker.get('embedding_count', 0)
                    avg_confidence = speaker.get('average_confidence', 0.0)
                    last_seen = speaker.get('last_seen', '')
                    
                    # Format data
                    confidence_str = f"{avg_confidence:.2f}" if avg_confidence else "N/A"
                    
                    # Format last seen date
                    formatted_last_seen = last_seen
                    if last_seen:
                        try:
                            dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                            formatted_last_seen = dt.strftime("%Y-%m-%d %H:%M")
                        except:
                            pass  # Use original string if parsing fails
                    
                    # Insert into tree
                    self.user_tree.insert('', tk.END, 
                                        iid=speaker_id,
                                        text=display_name,
                                        values=(embedding_count, confidence_str, formatted_last_seen))
                                        
                except Exception as speaker_error:
                    logger.error(f"Error processing individual speaker: {speaker_error}")
                    # Skip this speaker and continue with others
                    continue
                    
        except Exception as e:
            logger.error(f"Error refreshing user list: {e}")
            # Show a simple error without the problematic string interpolation
            messagebox.showerror("Error", "Failed to refresh user list. Check console for details.")
    
    def on_user_select(self, event):
        """Handle user selection"""
        selected_items = self.user_tree.selection()
        if not selected_items:
            self.selected_user_id = None
            self.record_button.config(state=tk.DISABLED)
            self.upload_button.config(state=tk.DISABLED)
            self.status_var.set("Select a user to begin")
            return
        
        self.selected_user_id = selected_items[0]
        
        # Enable controls
        if AUDIO_AVAILABLE and "not available" not in self.device_var.get().lower():
            self.record_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)
        
        # Update user info display
        self.update_user_info()
        
        # Store current confidence as "before" value
        speakers = self.speaker_db.get_all_speakers()
        selected_speaker = next((s for s in speakers if s['speaker_id'] == self.selected_user_id), None)
        if selected_speaker:
            self.confidence_before = selected_speaker['average_confidence'] or 0.0
            if selected_speaker['embedding_count'] == 0:
                self.confidence_before_var.set("0.000 (new user)")
            else:
                self.confidence_before_var.set(f"{self.confidence_before:.3f}")
        
        self.status_var.set("Ready to record or upload audio")
    
    def create_new_user(self):
        """Create a new user and add them to the database"""
        user_name = self.new_user_name.get().strip()
        
        if not user_name:
            messagebox.showwarning("Warning", "Please enter a name for the new user.")
            self.new_user_entry.focus()
            return
        
        # Check if user already exists
        speakers = self.speaker_db.get_all_speakers()
        existing_names = [s['display_name'].lower() for s in speakers]
        
        if user_name.lower() in existing_names:
            messagebox.showwarning("Warning", f"A user named '{user_name}' already exists. Please choose a different name.")
            self.new_user_entry.focus()
            self.new_user_entry.select_range(0, tk.END)
            return
        
        try:
            # Create new speaker in database
            speaker_id = self.speaker_db.create_speaker(
                display_name=user_name,
                source_type="enrolled",
                is_enrolled=True
            )
            
            # Force save the database to ensure the new user is persisted
            self.speaker_db._save_database()
            
            # Note: We don't create embeddings here - they'll be generated when the user first records audio
            
            # Refresh the user list
            self.refresh_user_list()
            
            # Select the newly created user
            for item in self.user_tree.get_children():
                if self.user_tree.item(item)['text'] == user_name:
                    self.user_tree.selection_set(item)
                    self.user_tree.focus(item)
                    # Trigger the selection event
                    self.selected_user_id = item
                    self.on_user_select(None)
                    break
            
            # Clear the entry field
            self.new_user_name.set("")
            
            # Show success message
            messagebox.showinfo("Success", f"New user '{user_name}' created successfully!\n\nYou can now record their voice to create their speaker profile.")
            
            # Update status
            self.status_var.set(f"New user '{user_name}' created - ready to record first sample")
            
            logger.info(f"New user '{user_name}' successfully created and saved to database")
            
        except Exception as e:
            logger.error(f"Error creating new user: {e}")
            messagebox.showerror("Error", f"Failed to create new user: {e}")
    
    def update_user_info(self):
        """Update the user information display"""
        if not self.selected_user_id:
            return
        
        self.user_info.config(state=tk.NORMAL)
        self.user_info.delete(1.0, tk.END)
        
        # Get speaker details
        speakers = self.speaker_db.get_all_speakers()
        selected_speaker = next((s for s in speakers if s['speaker_id'] == self.selected_user_id), None)
        
        if selected_speaker:
            # Handle new users with no embeddings
            avg_confidence = selected_speaker['average_confidence'] or 0.0
            confidence_display = f"{avg_confidence:.3f}" if selected_speaker['embedding_count'] > 0 else "0.000 (new user)"
            
            info_text = f"""User ID: {selected_speaker['speaker_id']}
Display Name: {selected_speaker['display_name']}
Created: {selected_speaker['created_date'][:19] if selected_speaker['created_date'] else 'N/A'}
Last Seen: {selected_speaker['last_seen'][:19] if selected_speaker['last_seen'] else 'N/A'}
Total Samples: {selected_speaker['embedding_count']}
Average Confidence: {confidence_display}
Source: {selected_speaker['source_type']}
Enrolled: {'Yes' if selected_speaker['is_enrolled'] else 'No'}
Verified: {'Yes' if selected_speaker['is_verified'] else 'No'}

Status: {'New user - needs first voice sample' if selected_speaker['embedding_count'] == 0 else 'Ready for additional training'}"""
            
            self.user_info.insert(tk.END, info_text)
        
        self.user_info.config(state=tk.DISABLED)
    
    def toggle_recording(self):
        """Start or stop audio recording"""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio recording not available. Please install sounddevice and soundfile.")
            return
        
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording"""
        if not self.selected_user_id:
            messagebox.showwarning("Warning", "Please select a user first.")
            return
        
        try:
            # Get selected device
            device_name = self.device_var.get()
            if device_name in getattr(self, 'device_ids', {}):
                device_id = self.device_ids[device_name]
            else:
                device_id = None
            
            # Create temporary file
            self.temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self.temp_audio_file.close()
            
            # Start recording in a separate thread
            self.is_recording = True
            self.current_recording = []
            
            self.record_button.config(text="⏹️ Stop Recording", style="Accent.TButton")
            self.status_var.set("🔴 Recording... Click 'Stop Recording' when finished")
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._record_audio, args=(device_id,))
            self.recording_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            self.is_recording = False
    
    def _record_audio(self, device_id):
        """Record audio in a separate thread"""
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio recording status: {status}")
                if self.is_recording:
                    self.current_recording.extend(indata.copy())
            
            with sd.InputStream(callback=audio_callback, 
                              device=device_id,
                              channels=self.channels, 
                              samplerate=self.sample_rate,
                              dtype=np.float32):
                while self.is_recording:
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop audio recording and process"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.record_button.config(text="🎤 Start Recording")
        self.status_var.set("Processing recording...")
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        
        try:
            # Save recorded audio
            if self.current_recording:
                audio_data = np.array(self.current_recording, dtype=np.float32)
                sf.write(self.temp_audio_file.name, audio_data, self.sample_rate)
                
                # Process the recording
                self.process_audio_file(self.temp_audio_file.name)
            else:
                messagebox.showwarning("Warning", "No audio was recorded.")
                self.status_var.set("Ready to record or upload audio")
                
        except Exception as e:
            logger.error(f"Error processing recording: {e}")
            messagebox.showerror("Error", f"Failed to process recording: {e}")
            self.status_var.set("Error processing recording")
    
    def upload_audio_file(self):
        """Upload an audio file for processing"""
        if not self.selected_user_id:
            messagebox.showwarning("Warning", "Please select a user first.")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.process_audio_file(file_path)
    
    def process_audio_file(self, audio_file_path):
        """Process the audio file and update embeddings"""
        try:
            self.status_var.set("Processing audio...")
            
            # Read audio file
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_file_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if necessary
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Transcribe the audio to show what was said
            self.status_var.set("Transcribing audio...")
            transcription = self.transcribe_audio(audio_file_path)
            
            # Display transcription
            self.transcription_text.config(state=tk.NORMAL)
            self.transcription_text.delete(1.0, tk.END)
            if transcription:
                self.transcription_text.insert(tk.END, transcription)
            else:
                self.transcription_text.insert(tk.END, "No transcription available")
            self.transcription_text.config(state=tk.DISABLED)
            
            # Generate embedding
            self.status_var.set("Generating voice embedding...")
            embedding = self.embedding_manager.extract_embedding(audio_data)
            
            if embedding is None:
                messagebox.showerror("Error", "Failed to extract embedding from audio.")
                self.status_var.set("Error processing audio")
                return
            
            # Add embedding to user
            confidence = 0.9  # High confidence for manual training
            self.speaker_db.add_embedding(self.selected_user_id, embedding, confidence)
            
            # Force save the database to ensure changes are persisted
            self.speaker_db._save_database()
            
            # Calculate new average confidence
            speakers = self.speaker_db.get_all_speakers()
            selected_speaker = next((s for s in speakers if s['speaker_id'] == self.selected_user_id), None)
            
            if selected_speaker:
                self.confidence_after = selected_speaker['average_confidence']
                self.confidence_after_var.set(f"{self.confidence_after:.3f}")
                
                # Calculate improvement
                improvement = self.confidence_after - self.confidence_before
                self.improvement_var.set(f"{improvement:+.3f}")
                
                # Color code the improvement
                if improvement > 0:
                    self.improvement_label.config(foreground='green')
                elif improvement < 0:
                    self.improvement_label.config(foreground='red')
                else:
                    self.improvement_label.config(foreground='black')
            
            # Enable train again button
            self.train_again_button.config(state=tk.NORMAL)
            
            # Update displays
            self.refresh_user_list()
            self.update_user_info()
            
            self.status_var.set("✅ Training completed successfully!")
            
            # Show success message
            messagebox.showinfo("Success", 
                              f"Embedding updated successfully!\n\n"
                              f"Confidence improved by: {improvement:+.3f}\n"
                              f"New average confidence: {self.confidence_after:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            messagebox.showerror("Error", f"Failed to process audio: {e}")
            self.status_var.set("Error processing audio")
        
        finally:
            # Clean up temporary file
            self.cleanup_temp_files()
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file using the Oreja backend"""
        try:
            import requests
            
            # Send audio to local Oreja backend for transcription
            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                response = requests.post(f"{self.backend_url}/transcribe", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract just the text from the transcription result
                    if 'segments' in result:
                        transcription = ' '.join([seg.get('text', '') for seg in result['segments']])
                        return transcription.strip()
                    elif 'text' in result:
                        return result['text'].strip()
                    else:
                        return "Transcription completed but no text found"
                else:
                    logger.warning(f"Transcription failed with status {response.status_code}")
                    return "Transcription service unavailable"
                    
        except requests.exceptions.ConnectionError:
            logger.warning("Could not connect to Oreja backend for transcription")
            return "Transcription service not available (backend not running)"
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return f"Transcription error: {str(e)}"
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files"""
        if self.temp_audio_file and os.path.exists(self.temp_audio_file.name):
            try:
                os.unlink(self.temp_audio_file.name)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")
    
    def prepare_for_next_training(self):
        """Prepare for another training session"""
        # Update the "before" confidence to current level
        self.confidence_before = self.confidence_after
        self.confidence_before_var.set(f"{self.confidence_before:.3f}")
        
        # Reset after values
        self.confidence_after_var.set("N/A")
        self.improvement_var.set("N/A")
        self.improvement_label.config(foreground='black')
        
        # Reset status
        self.status_var.set("Ready for next training session")
        self.train_again_button.config(state=tk.DISABLED)
    
    def view_user_analytics(self):
        """Switch to analytics view for the selected user"""
        if not self.selected_user_id:
            messagebox.showwarning("Warning", "Please select a user first.")
            return
        
        # Switch to details tab (assuming it's tab index 1)
        try:
            self.parent_notebook.select(1)  # Details tab
            messagebox.showinfo("Info", f"Switched to details view for user: {self.selected_user_id}")
        except:
            messagebox.showinfo("Info", "Please manually switch to the 'Speaker Details' tab to view analytics.")


def integrate_with_existing_gui(analytics_dashboard):
    """
    Integration function to add the User Embedding Trainer to an existing 
    SpeakerAnalyticsDashboard instance
    """
    try:
        trainer = UserEmbeddingTrainer(analytics_dashboard.notebook)
        return trainer
    except Exception as e:
        logger.error(f"Failed to integrate User Embedding Trainer: {e}")
        return None


if __name__ == "__main__":
    # Standalone testing
    root = tk.Tk()
    root.title("User Embedding Trainer - Test")
    root.geometry("1200x800")
    
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    trainer = UserEmbeddingTrainer(notebook)
    
    root.mainloop()