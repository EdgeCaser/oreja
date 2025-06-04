#!/usr/bin/env python3
"""
Oreja Complete Launcher
Single-click launcher for backend + frontend components
"""

import sys
import os
import subprocess
import time
import threading
import requests
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import webbrowser
import shutil

class OrejaLauncher:
    def __init__(self):
        self.backend_process = None
        self.live_transcription_process = None  # Track live transcription process
        self.backend_ready = False
        
        # Setup main window
        self.root = tk.Tk()
        self.root.title("🎙️ Oreja Launcher")
        self.root.geometry("600x800")  # Keep original width (600), but taller (800)
        self.root.resizable(True, True)
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the launcher interface"""
        # Create main canvas with scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎙️ Oreja Transcription System", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.backend_status = ttk.Label(status_frame, text="❌ Backend: Not Running", 
                                       font=("Arial", 10))
        self.backend_status.pack(anchor=tk.W)
        
        # Backend controls
        backend_frame = ttk.LabelFrame(main_frame, text="Backend Server", padding="10")
        backend_frame.pack(fill=tk.X, pady=(0, 20))
        
        backend_buttons = ttk.Frame(backend_frame)
        backend_buttons.pack(fill=tk.X)
        
        self.start_backend_btn = ttk.Button(backend_buttons, text="🚀 Start Backend", 
                                          command=self.start_backend)
        self.start_backend_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_backend_btn = ttk.Button(backend_buttons, text="⏹️ Stop Backend", 
                                         command=self.stop_backend, state="disabled")
        self.stop_backend_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(backend_buttons, text="🌐 Open API Docs", 
                  command=self.open_api_docs).pack(side=tk.LEFT)
        
        # Frontend options
        frontend_frame = ttk.LabelFrame(main_frame, text="Frontend Applications", padding="10")
        frontend_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Live Transcription (Primary Tool)
        live_frame = ttk.Frame(frontend_frame)
        live_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(live_frame, text="🎙️ Live Transcription", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(live_frame, text="Full WPF app with independent mic/system audio, volume meters, live transcription", 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        
        # Live transcription status and buttons
        live_controls = ttk.Frame(live_frame)
        live_controls.pack(fill=tk.X, pady=(5, 0))
        
        self.live_btn = ttk.Button(live_controls, text="Start Live Transcription", 
                                  command=self.launch_live_transcription, state="disabled")
        self.live_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_live_btn = ttk.Button(live_controls, text="Stop Live Transcription", 
                                       command=self.stop_live_transcription, state="disabled")
        self.stop_live_btn.pack(side=tk.LEFT)
        
        self.live_status = ttk.Label(live_controls, text="⏸️ Not Running", 
                                    font=("Arial", 8), foreground="gray")
        self.live_status.pack(side=tk.RIGHT)
        
        # Speaker Analytics
        analytics_frame = ttk.Frame(frontend_frame)
        analytics_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(analytics_frame, text="📊 Speaker Analytics Dashboard", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(analytics_frame, text="View and manage speaker profiles, analyze conversations", 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        
        self.analytics_btn = ttk.Button(analytics_frame, text="Launch Analytics GUI", 
                                       command=self.launch_analytics, state="disabled")
        self.analytics_btn.pack(anchor=tk.W, pady=(5, 0))
        
        # Transcription Editor
        editor_frame = ttk.Frame(frontend_frame)
        editor_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(editor_frame, text="✏️ Transcription Editor", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(editor_frame, text="Edit and correct transcription results", 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        
        self.editor_btn = ttk.Button(editor_frame, text="Launch Editor", 
                                    command=self.launch_editor, state="disabled")
        self.editor_btn.pack(anchor=tk.W, pady=(5, 0))
        
        # File Transcription Tool
        file_frame = ttk.Frame(frontend_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="📁 File Transcription", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(file_frame, text="Transcribe individual audio files", 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        
        self.file_btn = ttk.Button(file_frame, text="Launch File Transcriber", 
                                  command=self.launch_file_transcriber, state="disabled")
        self.file_btn.pack(anchor=tk.W, pady=(5, 0))
        
        # C# Frontend (if available)
        csharp_frame = ttk.Frame(frontend_frame)
        csharp_frame.pack(fill=tk.X)
        
        ttk.Label(csharp_frame, text="🖥️ Desktop Application", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(csharp_frame, text="Alternative desktop interface", 
                 font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        
        self.csharp_btn = ttk.Button(csharp_frame, text="Launch Desktop App", 
                                    command=self.launch_csharp, state="disabled")
        self.csharp_btn.pack(anchor=tk.W, pady=(5, 0))
        
        # Quick start button
        quick_frame = ttk.Frame(main_frame)
        quick_frame.pack(fill=tk.X, pady=(10, 0))
        
        quick_buttons = ttk.Frame(quick_frame)
        quick_buttons.pack(fill=tk.X)
        
        ttk.Button(quick_buttons, text="⚡ Quick Start: Live Transcription", 
                  command=self.quick_start_live, 
                  style="Accent.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(quick_buttons, text="📊 Quick Start: Analytics", 
                  command=self.quick_start_analytics).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Check initial state
        self.check_backend_status()
        self.check_frontend_availability()
        
    def check_backend_status(self):
        """Check if backend is already running"""
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if response.status_code == 200:
                self.backend_ready = True
                self.backend_status.config(text="✅ Backend: Running")
                self.start_backend_btn.config(state="disabled")
                self.stop_backend_btn.config(state="normal")
                self.enable_frontend_buttons()
            else:
                self.backend_ready = False
        except:
            self.backend_ready = False
    
    def check_frontend_availability(self):
        """Check which frontend components are available"""
        # Check C# frontend
        csharp_exe = Path("publish/Oreja.exe")
        if csharp_exe.exists():
            self.csharp_btn.config(state="normal" if self.backend_ready else "disabled")
        
        # Python GUIs should always be available if backend works
        if self.backend_ready:
            self.enable_frontend_buttons()
    
    def enable_frontend_buttons(self):
        """Enable frontend buttons when backend is ready"""
        # Only enable start button if live transcription is not running
        if not (self.live_transcription_process and self.live_transcription_process.poll() is None):
            self.live_btn.config(state="normal")
            
        self.analytics_btn.config(state="normal")
        self.editor_btn.config(state="normal")
        self.file_btn.config(state="normal")
        if Path("publish/Oreja.exe").exists():
            self.csharp_btn.config(state="normal")
    
    def disable_frontend_buttons(self):
        """Disable frontend buttons when backend is not ready"""
        self.live_btn.config(state="disabled")
        self.stop_live_btn.config(state="disabled")
        self.live_status.config(text="⏸️ Backend Required", foreground="gray")
        self.analytics_btn.config(state="disabled")
        self.editor_btn.config(state="disabled")
        self.file_btn.config(state="disabled")
        self.csharp_btn.config(state="disabled")
    
    def start_backend(self):
        """Start the backend server"""
        backend_dir = Path("backend")
        if not backend_dir.exists():
            messagebox.showerror("Error", "Backend directory not found!")
            return
        
        server_file = backend_dir / "server.py"
        if not server_file.exists():
            messagebox.showerror("Error", "Backend server.py not found!")
            return
        
        try:
            # Start backend process
            if sys.platform == "win32":
                # Use the virtual environment python if available
                venv_python = Path("venv/Scripts/python.exe")
                if venv_python.exists():
                    python_cmd = str(venv_python)
                else:
                    python_cmd = sys.executable
                
                self.backend_process = subprocess.Popen([
                    python_cmd, "server.py"
                ], cwd=backend_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                self.backend_process = subprocess.Popen([
                    sys.executable, "server.py"
                ], cwd=backend_dir)
            
            self.backend_status.config(text="⏳ Backend: Starting...")
            self.start_backend_btn.config(state="disabled")
            
            # Check if backend started successfully
            threading.Thread(target=self.wait_for_backend, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start backend: {e}")
    
    def wait_for_backend(self):
        """Wait for backend to be ready"""
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://127.0.0.1:8000/health", timeout=2)
                if response.status_code == 200:
                    self.backend_ready = True
                    self.root.after(0, self._backend_ready_callback)
                    return
            except:
                pass
            time.sleep(1)
        
        # Backend failed to start
        self.root.after(0, self._backend_failed_callback)
    
    def _backend_ready_callback(self):
        """Called when backend is ready"""
        self.backend_status.config(text="✅ Backend: Running")
        self.stop_backend_btn.config(state="normal")
        self.enable_frontend_buttons()
    
    def _backend_failed_callback(self):
        """Called when backend failed to start"""
        self.backend_status.config(text="❌ Backend: Failed to Start")
        self.start_backend_btn.config(state="normal")
        messagebox.showerror("Error", "Backend failed to start. Check console for errors.")
    
    def stop_backend(self):
        """Stop the backend server"""
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process = None
        
        self.backend_ready = False
        self.backend_status.config(text="❌ Backend: Not Running")
        self.start_backend_btn.config(state="normal")
        self.stop_backend_btn.config(state="disabled")
        self.disable_frontend_buttons()
        
        # Also stop live transcription if running since it needs the backend
        if self.live_transcription_process and self.live_transcription_process.poll() is None:
            try:
                self.live_transcription_process.terminate()
                self.live_transcription_process = None
                self.live_status.config(text="⏸️ Stopped (Backend Down)", foreground="gray")
                self.stop_live_btn.config(state="disabled")
            except:
                pass  # Process might already be dead
    
    def launch_analytics(self):
        """Launch speaker analytics GUI"""
        if not self.backend_ready:
            messagebox.showerror("Error", "Backend must be running first!")
            return
        
        try:
            backend_dir = Path("backend")
            subprocess.Popen([sys.executable, "start_analytics.py"], cwd=backend_dir)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch analytics: {e}")
    
    def launch_live_transcription(self):
        """Launch live transcription (C# WPF App via dotnet run)"""
        if not self.backend_ready:
            messagebox.showerror("Error", "Backend must be running first!")
            return
        
        # Check if dotnet is available
        if not shutil.which("dotnet"):
            messagebox.showerror("Error", ".NET SDK not found! Please install .NET 8.0 SDK first.")
            return
        
        # Check if project file exists
        project_file = Path("oreja.csproj")
        if not project_file.exists():
            messagebox.showerror("Error", "Project file 'oreja.csproj' not found! Make sure you're in the correct directory.")
            return
        
        # Check if there's already a live transcription process running
        if self.live_transcription_process and self.live_transcription_process.poll() is None:
            response = messagebox.askyesno("Live Transcription Running", 
                                         "Live transcription is already running. Do you want to start another instance?")
            if not response:
                return
        
        try:
            # Update UI to show starting state
            self.live_status.config(text="⏳ Starting...", foreground="orange")
            self.live_btn.config(state="disabled")
            
            # Launch using dotnet run (the working method)
            self.live_transcription_process = subprocess.Popen([
                "dotnet", "run", "--project", "oreja.csproj"
            ], cwd=Path.cwd())
            
            messagebox.showinfo("Live Transcription", 
                               "✅ Working live transcription GUI launched!\n\n" +
                               "🎙️ Features:\n" +
                               "• ✅ Independent microphone & system audio selection\n" +
                               "• ✅ Real-time volume meters for both sources\n" +
                               "• ✅ Live transcription with speaker recognition\n" +
                               "• ✅ Speaker renaming and assignment\n" +
                               "• ✅ Audio monitoring controls\n" +
                               "• ✅ Save transcription functionality\n" +
                               "• ✅ Professional WPF interface\n\n" +
                               "The window should appear shortly...")
            
            # Update UI to show running state
            self.live_status.config(text="🎙️ Running", foreground="green")
            self.stop_live_btn.config(state="normal")
            
            # Monitor the process in a separate thread
            threading.Thread(target=self._monitor_live_transcription, daemon=True).start()
            
        except Exception as e:
            self.live_status.config(text="❌ Failed", foreground="red")
            self.live_btn.config(state="normal")
            messagebox.showerror("Error", f"Failed to launch live transcription: {e}\n\n" +
                               "Make sure .NET 8.0 SDK is installed and the project builds successfully.")

    def stop_live_transcription(self):
        """Stop the live transcription process"""
        if self.live_transcription_process and self.live_transcription_process.poll() is None:
            try:
                self.live_transcription_process.terminate()
                self.live_transcription_process = None
                
                # Update UI
                self.live_status.config(text="⏸️ Stopped", foreground="gray")
                self.live_btn.config(state="normal")
                self.stop_live_btn.config(state="disabled")
                
                messagebox.showinfo("Live Transcription", "Live transcription has been stopped.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to stop live transcription: {e}")
        else:
            messagebox.showwarning("Warning", "No live transcription process is currently running.")

    def _monitor_live_transcription(self):
        """Monitor the live transcription process"""
        if self.live_transcription_process:
            # Wait a moment for the process to start
            time.sleep(2)
            
            # Check if process is still running
            if self.live_transcription_process.poll() is not None:
                # Process has exited, update UI on main thread
                self.root.after(0, self._live_transcription_stopped_callback)
            else:
                # Process is running, continue monitoring
                self.root.after(0, lambda: self.live_status.config(text="🎙️ Running", foreground="green"))

    def _live_transcription_stopped_callback(self):
        """Called when live transcription process stops unexpectedly"""
        self.live_status.config(text="❌ Stopped Unexpectedly", foreground="red")
        self.live_btn.config(state="normal")
        self.stop_live_btn.config(state="disabled")
        
        messagebox.showerror(
            "Live Transcription Error", 
            "Live transcription process exited unexpectedly.\n\n" +
            "This might be due to:\n" +
            "• Missing dependencies\n" +
            "• Build errors in the project\n" +
            "• Audio device access issues\n\n" +
            "Try building the project manually first:\n" +
            "dotnet build oreja.csproj"
        )
    
    def launch_file_transcriber(self):
        """Launch file transcription tool"""
        if not self.backend_ready:
            messagebox.showerror("Error", "Backend must be running first!")
            return
        
        # Create a simple file transcription dialog
        file_path = tk.filedialog.askopenfilename(
            title="Select Audio File to Transcribe",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Launch simple_batch.py with the selected file
                subprocess.Popen([sys.executable, "simple_batch.py", file_path])
                messagebox.showinfo("File Transcription", 
                                   f"Transcribing: {Path(file_path).name}\n\n" +
                                   "Check the console window for progress.\n" +
                                   "Results will be saved to 'transcription_results' folder.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start file transcription: {e}")
    
    def launch_editor(self):
        """Launch transcription editor"""
        if not self.backend_ready:
            messagebox.showerror("Error", "Backend must be running first!")
            return
        
        try:
            subprocess.Popen([sys.executable, "transcription_editor.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch editor: {e}")
    
    def launch_csharp(self):
        """Launch desktop application (C# Live Transcription)"""
        if not self.backend_ready:
            messagebox.showerror("Error", "Backend must be running first!")
            return
        
        csharp_exe = Path("publish/Oreja.exe")
        if not csharp_exe.exists():
            messagebox.showerror("Error", "C# application not found! Build it first.")
            return
        
        try:
            subprocess.Popen([str(csharp_exe)])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch desktop app: {e}")
    
    def open_api_docs(self):
        """Open API documentation in browser"""
        if self.backend_ready:
            webbrowser.open("http://127.0.0.1:8000/docs")
        else:
            messagebox.showwarning("Warning", "Backend must be running to view API docs!")
    
    def quick_start_live(self):
        """Quick start: Launch live transcription"""
        if not self.backend_ready:
            self.start_backend()
            # Wait a moment then launch live transcription
            self.root.after(3000, lambda: self.launch_live_transcription() if self.backend_ready else None)
        else:
            self.launch_live_transcription()
    
    def quick_start_analytics(self):
        """Quick start: Launch analytics"""
        if not self.backend_ready:
            self.start_backend()
            # Wait a moment then launch analytics
            self.root.after(3000, lambda: self.launch_analytics() if self.backend_ready else None)
        else:
            self.launch_analytics()
    
    def on_closing(self):
        """Handle application closing"""
        try:
            if self.live_transcription_process and self.live_transcription_process.poll() is None:
                self.live_transcription_process.terminate()
        except:
            pass
            
        try:
            if self.backend_process:
                self.backend_process.terminate()
        except:
            pass
            
        self.root.destroy()
    
    def run(self):
        """Run the launcher"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """Main entry point"""
    print("🎙️ Starting Oreja Launcher...")
    
    # Check if we're in the right directory
    if not Path("backend").exists():
        print("❌ Backend directory not found!")
        print("Please run this script from the Oreja root directory.")
        sys.exit(1)
    
    launcher = OrejaLauncher()
    launcher.run()

if __name__ == "__main__":
    main() 