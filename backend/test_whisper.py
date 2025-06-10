#!/usr/bin/env python3
"""
Test Whisper model loading to debug the issue
"""
import sys
import traceback
from transformers import pipeline
import torch

def test_whisper_loading():
    """Test Whisper model loading step by step"""
    print("🧪 Testing Whisper model loading...")
    
    try:
        # Step 1: Test device setup
        print("Step 1: Setting up device...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("✓ Using CPU")
        
        # Step 2: Test basic pipeline import
        print("Step 2: Testing transformers pipeline...")
        from transformers import pipeline
        print("✓ Pipeline import successful")
        
        # Step 3: Test Whisper large-v3-turbo
        print("Step 3: Loading Whisper large-v3-turbo...")
        try:
            whisper_model = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                device=device,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                return_timestamps=True,
                chunk_length_s=30,
                stride_length_s=5,
            )
            print("✅ Whisper large-v3-turbo loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load whisper-large-v3-turbo: {e}")
            traceback.print_exc()
            
            # Step 4: Try fallback model
            print("Step 4: Trying fallback model (whisper-base)...")
            try:
                whisper_model = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-base",
                    device=device,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    return_timestamps=True,
                    chunk_length_s=30,
                    stride_length_s=5,
                )
                print("✅ Whisper base model loaded successfully (fallback)!")
                return True
            except Exception as e2:
                print(f"❌ Failed to load whisper-base: {e2}")
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"❌ Critical error during setup: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_whisper_loading()
    if success:
        print("\n🎉 Whisper model loading test passed!")
        sys.exit(0)
    else:
        print("\n💥 Whisper model loading test failed!")
        sys.exit(1) 