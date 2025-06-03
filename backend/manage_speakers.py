#!/usr/bin/env python3
"""
Oreja Speaker Management Tool
Delete, rename, and manage speakers in the database
"""

import requests
import json
from datetime import datetime
import time

BACKEND_URL = "http://127.0.0.1:8000"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ Backend not reachable: {e}")
        return None

def get_speaker_stats():
    """Get all speaker statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/speakers", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get speaker stats: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting speaker stats: {e}")
        return None

def delete_speaker(speaker_id):
    """Delete a specific speaker"""
    try:
        response = requests.delete(f"{BACKEND_URL}/speakers/{speaker_id}", timeout=10)
        if response.status_code == 200:
            result = response.json()
            return True, result.get('status', 'deleted')
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Request failed: {e}"

def rename_speaker(speaker_id, new_name):
    """Rename a speaker"""
    try:
        response = requests.put(
            f"{BACKEND_URL}/speakers/{speaker_id}/name", 
            params={"new_name": new_name},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            return True, result.get('status', 'updated')
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Request failed: {e}"

def format_date(date_str):
    """Format ISO date string for display"""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return date_str

def interactive_speaker_cleanup():
    """Interactive speaker cleanup for Ian"""
    print("=" * 80)
    print("🧹 OREJA SPEAKER CLEANUP TOOL")
    print("=" * 80)
    
    # Check backend
    print("📡 Checking backend connection...")
    health = check_backend_health()
    if not health:
        print("❌ Backend not available. Make sure the server is running:")
        print("   cd backend && python server.py")
        return
    
    print(f"✅ Backend connected")
    print()
    
    # Get speakers
    print("🔍 Fetching current speakers...")
    stats = get_speaker_stats()
    if not stats:
        print("❌ Could not retrieve speaker statistics")
        return
    
    speakers = stats.get('speakers', [])
    total_speakers = len(speakers)
    
    if total_speakers == 0:
        print("🤷 No speakers found in database.")
        return
    
    print(f"📊 Found {total_speakers} speakers in database")
    print()
    
    # Display speakers with numbers
    print("📋 CURRENT SPEAKERS:")
    print("-" * 80)
    print(f"{'#':<3} {'Name':<25} {'ID':<18} {'Embeddings':<10} {'Last Seen':<16}")
    print("-" * 80)
    
    for i, speaker in enumerate(speakers, 1):
        name = speaker.get('name', 'Unknown')[:24]
        speaker_id = speaker.get('id', 'N/A')[:17]
        embedding_count = speaker.get('embedding_count', 0)
        last_seen = format_date(speaker.get('last_seen', ''))[:15]
        
        print(f"{i:<3} {name:<25} {speaker_id:<18} {embedding_count:<10} {last_seen:<16}")
    
    print("-" * 80)
    print()
    
    # Help Ian identify which speakers to keep/delete
    print("🤔 SPEAKER IDENTIFICATION:")
    print("Let's identify which speakers are Ian and which should be deleted.")
    print()
    
    ian_speakers = []
    delete_speakers = []
    
    for i, speaker in enumerate(speakers, 1):
        name = speaker.get('name', 'Unknown')
        speaker_id = speaker.get('id', 'N/A')
        
        print(f"Speaker #{i}: {name} (ID: {speaker_id})")
        while True:
            choice = input("Is this Ian? (y/n/skip): ").lower().strip()
            if choice in ['y', 'yes']:
                ian_speakers.append((i, speaker))
                print(f"   ✅ Marked as Ian")
                break
            elif choice in ['n', 'no']:
                delete_speakers.append((i, speaker))
                print(f"   ❌ Marked for deletion")
                break
            elif choice in ['s', 'skip']:
                print(f"   ⏭️ Skipped")
                break
            else:
                print("   Please enter 'y' for yes, 'n' for no, or 'skip'")
        print()
    
    # Summary
    print("📝 SUMMARY:")
    print(f"   🟢 Ian speakers: {len(ian_speakers)}")
    print(f"   🔴 Speakers to delete: {len(delete_speakers)}")
    print(f"   ⚪ Skipped: {total_speakers - len(ian_speakers) - len(delete_speakers)}")
    print()
    
    if len(delete_speakers) == 0:
        print("✅ No speakers marked for deletion. Nothing to do!")
        return
    
    # Confirm deletion
    print("🗑️ SPEAKERS TO DELETE:")
    for i, speaker in delete_speakers:
        name = speaker.get('name', 'Unknown')
        speaker_id = speaker.get('id', 'N/A')
        print(f"   #{i}: {name} (ID: {speaker_id})")
    print()
    
    confirm = input(f"⚠️ Delete {len(delete_speakers)} speakers? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        print("❌ Deletion cancelled.")
        return
    
    # Delete speakers
    print()
    print("🗑️ Deleting speakers...")
    deleted_count = 0
    
    for i, speaker in delete_speakers:
        speaker_id = speaker.get('id')
        name = speaker.get('name', 'Unknown')
        
        print(f"   Deleting #{i}: {name}...", end=" ")
        success, message = delete_speaker(speaker_id)
        
        if success:
            print("✅ Deleted")
            deleted_count += 1
        else:
            print(f"❌ Failed: {message}")
    
    print()
    print(f"🎉 Successfully deleted {deleted_count} out of {len(delete_speakers)} speakers!")
    
    # Option to rename Ian speakers
    if len(ian_speakers) > 0:
        print()
        rename_choice = input("🏷️ Would you like to rename Ian's speakers to 'Ian'? (y/n): ").lower().strip()
        if rename_choice in ['y', 'yes']:
            print("🏷️ Renaming Ian's speakers...")
            for i, speaker in ian_speakers:
                speaker_id = speaker.get('id')
                old_name = speaker.get('name', 'Unknown')
                
                print(f"   Renaming #{i}: {old_name} → Ian...", end=" ")
                success, message = rename_speaker(speaker_id, "Ian")
                
                if success:
                    print("✅ Renamed")
                else:
                    print(f"❌ Failed: {message}")
    
    print()
    print("🔄 Run 'python view_speakers.py' to see the updated database!")
    print("=" * 80)

if __name__ == "__main__":
    interactive_speaker_cleanup() 