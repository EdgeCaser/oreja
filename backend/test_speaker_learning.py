#!/usr/bin/env python3
"""
Test script to monitor speaker learning in Oreja.
This helps verify that speaker corrections are being properly learned by the backend.
"""

import requests
import time
import json
from datetime import datetime

BACKEND_URL = "http://127.0.0.1:8000"

def get_speaker_stats():
    """Get current speaker statistics from the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/speakers")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting speaker stats: HTTP {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return None

def display_speakers(stats):
    """Display speaker information in a readable format."""
    if not stats:
        print("No speaker data available")
        return
    
    total_speakers = stats.get('total_speakers', 0)
    speakers = stats.get('speakers', [])
    
    print(f"\n{'='*60}")
    print(f"🎤 SPEAKER DATABASE STATUS - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Total Speakers: {total_speakers}")
    print()
    
    if speakers:
        # Sort speakers by embedding count (most trained first)
        speakers_sorted = sorted(speakers, key=lambda x: x.get('embedding_count', 0), reverse=True)
        
        for speaker in speakers_sorted:
            speaker_id = speaker.get('id', 'Unknown')
            name = speaker.get('name', 'Unnamed')
            embedding_count = speaker.get('embedding_count', 0)
            avg_confidence = speaker.get('avg_confidence', 0.0)
            last_seen = speaker.get('last_seen', 'Never')
            
            # Color coding for different speaker types
            if speaker_id.startswith('AUTO_SPEAKER'):
                status = "🤖 Auto-Generated"
            elif speaker_id.startswith('CORRECTED_SPEAKER'):
                status = "✅ User-Corrected"
            elif speaker_id.startswith('ENROLLED_SPEAKER'):
                status = "👤 Enrolled"
            else:
                status = "❓ Unknown Type"
            
            print(f"📍 {name}")
            print(f"   ID: {speaker_id}")
            print(f"   Status: {status}")
            print(f"   Training Data: {embedding_count} audio samples")
            print(f"   Avg Confidence: {avg_confidence:.3f}")
            print(f"   Last Seen: {last_seen[:19] if len(last_seen) > 19 else last_seen}")
            print()
    else:
        print("No speakers found in database")

def test_name_mapping(old_speaker_id, new_name):
    """Test the speaker name mapping functionality."""
    try:
        print(f"\n🔄 Testing name mapping: '{old_speaker_id}' → '{new_name}'")
        
        response = requests.post(
            f"{BACKEND_URL}/speakers/name_mapping",
            params={
                "old_speaker_id": old_speaker_id,
                "new_speaker_name": new_name
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "unknown")
            
            if status == "speakers_merged":
                print(f"✅ SUCCESS: Speakers merged!")
                print(f"   {old_speaker_id} merged into existing speaker")
                print(f"   Target: {result.get('target_speaker_id')}")
            elif status == "name_updated":
                print(f"✅ SUCCESS: Speaker renamed!")
                print(f"   {old_speaker_id} → {new_name}")
            else:
                print(f"ℹ️  Status: {status}")
                print(f"   Result: {result}")
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")

def monitor_speaker_changes():
    """Monitor speaker database for changes."""
    print("🔍 Starting speaker learning monitor...")
    print("Press Ctrl+C to stop monitoring")
    
    previous_stats = None
    
    try:
        while True:
            current_stats = get_speaker_stats()
            
            if current_stats:
                # Check for changes
                if previous_stats is None:
                    print("📊 Initial speaker database state:")
                    display_speakers(current_stats)
                else:
                    current_total = current_stats.get('total_speakers', 0)
                    previous_total = previous_stats.get('total_speakers', 0)
                    
                    if current_total != previous_total:
                        print(f"\n🔔 CHANGE DETECTED: Speaker count changed from {previous_total} to {current_total}")
                        display_speakers(current_stats)
                    else:
                        # Check for changes in individual speakers
                        current_speakers = {s['id']: s for s in current_stats.get('speakers', [])}
                        previous_speakers = {s['id']: s for s in previous_stats.get('speakers', [])}
                        
                        changes_detected = False
                        for speaker_id, current_speaker in current_speakers.items():
                            if speaker_id in previous_speakers:
                                prev_embeddings = previous_speakers[speaker_id].get('embedding_count', 0)
                                curr_embeddings = current_speaker.get('embedding_count', 0)
                                
                                if curr_embeddings != prev_embeddings:
                                    print(f"\n🔔 LEARNING DETECTED: {current_speaker.get('name')} training data updated")
                                    print(f"   Embeddings: {prev_embeddings} → {curr_embeddings}")
                                    changes_detected = True
                        
                        if changes_detected:
                            display_speakers(current_stats)
                
                previous_stats = current_stats
            
            time.sleep(3)  # Check every 3 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def main():
    """Main function with options for testing."""
    print("🎤 Oreja Speaker Learning Test Tool")
    print("=" * 40)
    
    # Check backend connection
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code != 200:
            print("❌ Backend not accessible. Please start the backend server first.")
            return
    except requests.exceptions.RequestException:
        print("❌ Backend not accessible. Please start the backend server first.")
        return
    
    print("✅ Backend connection successful")
    
    while True:
        print("\nOptions:")
        print("1. Show current speaker database")
        print("2. Test speaker name mapping")
        print("3. Monitor speaker changes (real-time)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            stats = get_speaker_stats()
            display_speakers(stats)
            
        elif choice == "2":
            old_id = input("Enter speaker ID to rename (e.g., AUTO_SPEAKER_001): ").strip()
            new_name = input("Enter new speaker name (e.g., John): ").strip()
            
            if old_id and new_name:
                test_name_mapping(old_id, new_name)
            else:
                print("❌ Both speaker ID and new name are required")
                
        elif choice == "3":
            monitor_speaker_changes()
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 