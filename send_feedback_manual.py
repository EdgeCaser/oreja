#!/usr/bin/env python3
"""
Manual Speaker Feedback Sender
Use this script to send speaker corrections to Oreja when the GUI has issues
"""

import requests
import json
from pathlib import Path

def send_speaker_corrections(corrections_dict):
    """
    Send speaker name corrections to Oreja backend
    
    Args:
        corrections_dict: Dictionary of {old_speaker_id: new_speaker_name}
    """
    if not corrections_dict:
        print("No corrections to send.")
        return
    
    success_count = 0
    error_count = 0
    
    print(f"Sending {len(corrections_dict)} speaker corrections...")
    
    for old_speaker_id, new_speaker_name in corrections_dict.items():
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
                print(f"✅ {old_speaker_id} -> {new_speaker_name}")
            else:
                error_count += 1
                print(f"❌ Failed to map {old_speaker_id} -> {new_speaker_name}: {response.text}")
                
        except Exception as e:
            error_count += 1
            print(f"❌ Error mapping {old_speaker_id} -> {new_speaker_name}: {e}")
    
    print(f"\nResults: {success_count} successful, {error_count} failed")

if __name__ == "__main__":
    print("🎙️  Manual Speaker Feedback Sender")
    print("=" * 40)
    
    # Example usage - replace with your actual corrections
    # You'll need to manually enter your corrections here
    corrections = {
        # "SPEAKER_00": "John",
        # "SPEAKER_01": "Mary",
        # Add your corrections here in the format: "original_id": "correct_name"
    }
    
    if not corrections:
        print("\n📝 Instructions:")
        print("1. Edit this script and add your corrections to the 'corrections' dictionary")
        print("2. Use the format: \"SPEAKER_XX\": \"Actual Name\"")
        print("3. Run the script again")
        print("\nExample:")
        print('corrections = {')
        print('    "SPEAKER_00": "John Smith",')
        print('    "SPEAKER_01": "Mary Johnson",')
        print('}')
    else:
        send_speaker_corrections(corrections) 