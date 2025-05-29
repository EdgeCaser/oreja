#!/usr/bin/env python3
"""
Install script for CUDA-enabled PyTorch to enable GPU acceleration for Oreja.
This will dramatically speed up transcription from 15-27 seconds to 2-3 seconds per chunk.
"""

import subprocess
import sys
import platform

def run_command(command):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ Success: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e.stderr}")
        return False

def main():
    print("🚀 Installing CUDA-enabled PyTorch for GPU acceleration...")
    print("This will enable your RTX 3080 for extremely fast transcription!")
    
    # Uninstall CPU-only PyTorch first
    print("\n📦 Uninstalling CPU-only PyTorch...")
    uninstall_commands = [
        "pip uninstall torch torchvision torchaudio -y",
    ]
    
    for cmd in uninstall_commands:
        run_command(cmd)
    
    # Install CUDA-enabled PyTorch (CUDA 12.1 is compatible with CUDA 12.9)
    print("\n🔥 Installing CUDA-enabled PyTorch...")
    install_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    
    if run_command(install_command):
        print("\n✅ GPU PyTorch installation completed!")
        
        # Test CUDA availability
        print("\n🧪 Testing CUDA availability...")
        test_code = """
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available - please check installation')
"""
        
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True)
        print(result.stdout)
        
        if "CUDA available: True" in result.stdout:
            print("\n🎉 SUCCESS! Your RTX 3080 is ready for GPU acceleration!")
            print("Restart the Oreja backend server to see dramatic speed improvements.")
            print("Expected transcription time: 2-3 seconds instead of 15-27 seconds!")
        else:
            print("\n⚠️  CUDA installation may have failed. Please restart your terminal and try again.")
    
    else:
        print("\n❌ Installation failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main() 