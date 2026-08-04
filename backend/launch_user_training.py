#!/usr/bin/env python3
"""
Oreja User Training Module Launcher
Simple launcher to test the user training functionality
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []
    
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")
    
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    
    try:
        from speaker_database_v2 import EnhancedSpeakerDatabase
    except ImportError:
        missing.append("speaker_database_v2 (Oreja core)")
    
    # NOTE: speaker_embeddings (OfflineSpeakerEmbeddingManager) was removed.
    # Voice embeddings now come from server.extract_embedding_from_audio, which
    # is loaded lazily at training time - it is not a launch prerequisite, so
    # it is deliberately not checked here.

    return missing

def main():
    """Main launcher function"""
    print("🎯 Oreja User Training Module Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nTo install audio dependencies, run:")
        print("  pip install -r requirements_user_training.txt")
        
        # Show GUI error
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        message = f"Missing dependencies:\n\n" + "\n".join(f"• {dep}" for dep in missing_deps)
        message += f"\n\nTo install, run:\npip install -r requirements_user_training.txt"
        
        messagebox.showerror("Missing Dependencies", message)
        return 1
    
    print("✅ All dependencies found")
    
    # Check if database exists
    db_path = Path("speaker_data_v2/speaker_records.json")
    if not db_path.exists():
        print("⚠️  No speaker database found. Creating empty database...")
        db_path.parent.mkdir(exist_ok=True)
        # The database will be created automatically when first accessed
    
    print("🚀 Launching User Training Module...")
    
    try:
        # Create main window
        root = tk.Tk()
        root.title("Oreja User Training Module")
        root.geometry("1200x800")
        root.configure(bg='#f0f0f0')
        
        # Create notebook for tabs
        from tkinter import ttk
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Import and create the user training module
        from user_embedding_trainer import UserEmbeddingTrainer
        trainer = UserEmbeddingTrainer(notebook)
        
        # Add some basic instructions
        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text="ℹ️ Instructions")
        
        instructions = """
        🎯 User Embedding Trainer Instructions
        
                 1. Select or Create a User:
            • Choose an existing speaker from the database
            • Or create a brand new user for first-time enrollment
            • View their current confidence and sample count
        
        2. Record or Upload Audio:
           • Record yourself reading the standard text
           • Or upload an existing audio file
           • Ensure good audio quality for best results
        
        3. View Results:
           • See confidence improvement after training
           • Train multiple times to further improve recognition
        
                 4. Standard Text:
            Please read "The Story of the Hare Who Lost His Spectacles"
            clearly and naturally at your normal speaking pace.
        
        Audio Requirements:
        • Quiet environment (minimal background noise)
        • Clear speech directed toward microphone
        • At least 10-15 seconds of audio
        • WAV, MP3, M4A, FLAC, or OGG format
        
        Privacy Note:
        • All processing is done locally on your machine
        • Audio files are automatically deleted after processing
        • No data is sent to external servers
        """
        
        instructions_text = tk.Text(info_frame, wrap=tk.WORD, font=('Arial', 11), 
                                  padx=20, pady=20)
        instructions_text.insert(tk.END, instructions)
        instructions_text.config(state=tk.DISABLED)
        instructions_text.pack(fill=tk.BOTH, expand=True)
        
        # Select the training tab by default
        notebook.select(0)
        
        print("✅ User Training Module launched successfully!")
        print("\nGUI Instructions:")
        print("1. Select a user from the left panel")
        print("2. Record or upload audio")
        print("3. View confidence improvements")
        print("4. Train again to further improve recognition")
        
        # Start the GUI
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error launching User Training Module: {e}")
        
        # Show GUI error
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Launch Error", f"Failed to launch User Training Module:\n\n{e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 