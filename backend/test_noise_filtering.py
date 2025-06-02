#!/usr/bin/env python3
"""
Test script to verify enhanced noise filtering in Oreja backend.
This script tests various audio scenarios to ensure keyboard clicks and other noises are properly filtered.
"""

import requests
import numpy as np
import io
import wave
import time

BACKEND_URL = "http://127.0.0.1:8000"

def create_test_audio(duration_seconds, sample_rate=16000, audio_type="silence"):
    """Create test audio data for different scenarios."""
    samples = int(duration_seconds * sample_rate)
    
    if audio_type == "silence":
        # Pure silence
        audio = np.zeros(samples, dtype=np.float32)
    elif audio_type == "keyboard_click":
        # Simulate keyboard click - short burst of high-frequency noise
        audio = np.zeros(samples, dtype=np.float32)
        click_samples = int(0.01 * sample_rate)  # 10ms click
        click_start = samples // 2
        # High frequency noise burst
        t = np.linspace(0, 0.01, click_samples)
        click = 0.3 * np.sin(2 * np.pi * 8000 * t) * np.exp(-t * 100)  # 8kHz decaying sine
        audio[click_start:click_start + click_samples] = click
    elif audio_type == "constant_tone":
        # Constant 1kHz tone (not speech-like)
        t = np.linspace(0, duration_seconds, samples)
        audio = 0.1 * np.sin(2 * np.pi * 1000 * t)
    elif audio_type == "speech_like":
        # Simulate speech-like audio with varying frequencies and amplitude
        t = np.linspace(0, duration_seconds, samples)
        # Mix of frequencies typical in speech (200-3000 Hz)
        audio = (0.1 * np.sin(2 * np.pi * 200 * t) + 
                0.08 * np.sin(2 * np.pi * 500 * t) + 
                0.06 * np.sin(2 * np.pi * 1000 * t) + 
                0.04 * np.sin(2 * np.pi * 2000 * t))
        # Add amplitude modulation to simulate speech patterns
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3Hz modulation
        audio = audio * envelope
        # Add some noise
        audio += 0.01 * np.random.normal(0, 1, samples)
    elif audio_type == "very_short":
        # Very short audio burst (should be filtered)
        audio = np.zeros(samples, dtype=np.float32)
        burst_samples = int(0.05 * sample_rate)  # 50ms burst
        burst_start = samples // 2
        audio[burst_start:burst_start + burst_samples] = 0.2
    else:
        audio = np.zeros(samples, dtype=np.float32)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16

def create_wav_bytes(audio_data, sample_rate=16000):
    """Convert audio data to WAV format bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    return buffer.getvalue()

def test_audio_scenario(audio_type, duration=1.0, description=""):
    """Test a specific audio scenario."""
    print(f"\n--- Testing: {audio_type} ({description}) ---")
    
    # Create test audio
    audio_data = create_test_audio(duration, audio_type=audio_type)
    wav_bytes = create_wav_bytes(audio_data)
    
    # Send to backend
    try:
        files = {'audio': ('test.wav', wav_bytes, 'audio/wav')}
        response = requests.post(f"{BACKEND_URL}/transcribe", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if transcription was skipped
            if 'skipped_reason' in result:
                print(f"✅ CORRECTLY FILTERED: {result['skipped_reason']}")
                print(f"   Segments: {len(result.get('segments', []))}")
            else:
                segments = result.get('segments', [])
                full_text = result.get('full_text', '')
                print(f"⚠️  TRANSCRIBED: {len(segments)} segments")
                if full_text:
                    print(f"   Text: '{full_text[:100]}{'...' if len(full_text) > 100 else ''}'")
                else:
                    print("   No text transcribed")
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST ERROR: {e}")

def main():
    """Run all noise filtering tests."""
    print("🔧 Testing Enhanced Noise Filtering in Oreja Backend")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
        else:
            print("❌ Backend health check failed")
            return
    except requests.exceptions.RequestException:
        print("❌ Backend is not accessible. Please start the backend server first.")
        return
    
    # Test scenarios that should be filtered out
    test_audio_scenario("silence", 2.0, "Pure silence - should be filtered")
    test_audio_scenario("keyboard_click", 1.0, "Keyboard click simulation - should be filtered")
    test_audio_scenario("constant_tone", 2.0, "Constant 1kHz tone - should be filtered")
    test_audio_scenario("very_short", 0.5, "Very short audio burst - should be filtered")
    
    # Test scenarios that should pass through
    test_audio_scenario("speech_like", 2.0, "Speech-like audio - should be transcribed")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    print("\nExpected results:")
    print("- Silence, keyboard clicks, constant tones, and very short audio should be FILTERED")
    print("- Speech-like audio should be TRANSCRIBED (though may produce gibberish)")
    print("\nIf keyboard clicks are still being transcribed as Japanese, the filtering is working!")

if __name__ == "__main__":
    main() 