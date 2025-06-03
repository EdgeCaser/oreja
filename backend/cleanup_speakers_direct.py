#!/usr/bin/env python3
"""
Oreja Direct Speaker Cleanup Tool
Clean up speaker database by directly modifying files (no API needed)
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime

def backup_speaker_data():
    """Create a backup of speaker data before making changes"""
    speaker_data_dir = Path("speaker_data")
    if not speaker_data_dir.exists():
        print("❌ Speaker data directory not found")
        return False
    
    backup_dir = Path(f"speaker_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    try:
        shutil.copytree(speaker_data_dir, backup_dir)
        print(f"✅ Backup created: {backup_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return False

def load_speaker_profiles():
    """Load speaker profiles from JSON file"""
    profiles_file = Path("speaker_data/speaker_profiles.json")
    if not profiles_file.exists():
        print("❌ Speaker profiles file not found")
        return None
    
    try:
        with open(profiles_file, 'r') as f:
            profiles = json.load(f)
        return profiles
    except Exception as e:
        print(f"❌ Failed to load speaker profiles: {e}")
        return None

def save_speaker_profiles(profiles):
    """Save speaker profiles to JSON file"""
    profiles_file = Path("speaker_data/speaker_profiles.json")
    
    try:
        with open(profiles_file, 'w') as f:
            json.dump(profiles, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Failed to save speaker profiles: {e}")
        return False

def delete_speaker_embeddings(speaker_id):
    """Delete embedding files for a speaker"""
    embeddings_dir = Path("speaker_data/embeddings")
    embedding_file = embeddings_dir / f"{speaker_id}.npy"
    
    if embedding_file.exists():
        try:
            embedding_file.unlink()
            return True
        except Exception as e:
            print(f"❌ Failed to delete embedding file for {speaker_id}: {e}")
            return False
    return True

def format_date(date_str):
    """Format ISO date string for display"""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return date_str

def interactive_cleanup():
    """Interactive speaker cleanup"""
    print("=" * 80)
    print("🧹 OREJA DIRECT SPEAKER CLEANUP TOOL")
    print("=" * 80)
    print("This tool directly modifies speaker database files (no backend needed)")
    print()
    
    # Check if we're in the right directory
    if not Path("speaker_data").exists():
        print("❌ Speaker data directory not found!")
        print("Make sure you're running this from the backend directory:")
        print("   cd backend && python cleanup_speakers_direct.py")
        return
    
    # Create backup
    print("💾 Creating backup of speaker data...")
    if not backup_speaker_data():
        return
    print()
    
    # Load profiles
    print("📂 Loading speaker profiles...")
    profiles = load_speaker_profiles()
    if profiles is None:
        return
    
    if not profiles:
        print("🤷 No speakers found in database.")
        return
    
    print(f"📊 Found {len(profiles)} speakers in database")
    print()
    
    # Display speakers
    print("📋 CURRENT SPEAKERS:")
    print("-" * 80)
    print(f"{'#':<3} {'Name':<25} {'ID':<20} {'Created':<16} {'Last Seen':<16}")
    print("-" * 80)
    
    speaker_list = list(profiles.items())
    for i, (speaker_id, data) in enumerate(speaker_list, 1):
        name = data.get('name', 'Unknown')[:24]
        created = format_date(data.get('created_date', ''))[:15]
        last_seen = format_date(data.get('last_seen', ''))[:15]
        
        print(f"{i:<3} {name:<25} {speaker_id:<20} {created:<16} {last_seen:<16}")
    
    print("-" * 80)
    print()
    
    # Interactive selection
    print("🤔 SPEAKER IDENTIFICATION:")
    print("For each speaker, tell me if this is Ian or should be deleted.")
    print()
    
    ian_speakers = []
    delete_speakers = []
    
    for i, (speaker_id, data) in enumerate(speaker_list, 1):
        name = data.get('name', 'Unknown')
        
        print(f"Speaker #{i}: {name} (ID: {speaker_id})")
        while True:
            choice = input("Is this Ian? (y/n/skip): ").lower().strip()
            if choice in ['y', 'yes']:
                ian_speakers.append((speaker_id, data))
                print(f"   ✅ Marked as Ian")
                break
            elif choice in ['n', 'no']:
                delete_speakers.append((speaker_id, data))
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
    print(f"   ⚪ Skipped: {len(profiles) - len(ian_speakers) - len(delete_speakers)}")
    print()
    
    if len(delete_speakers) == 0:
        print("✅ No speakers marked for deletion. Nothing to do!")
        return
    
    # Confirm deletion
    print("🗑️ SPEAKERS TO DELETE:")
    for speaker_id, data in delete_speakers:
        name = data.get('name', 'Unknown')
        print(f"   {name} (ID: {speaker_id})")
    print()
    
    confirm = input(f"⚠️ Delete {len(delete_speakers)} speakers? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        print("❌ Deletion cancelled.")
        return
    
    # Perform deletion
    print()
    print("🗑️ Deleting speakers...")
    deleted_count = 0
    
    for speaker_id, data in delete_speakers:
        name = data.get('name', 'Unknown')
        
        print(f"   Deleting {name}...", end=" ")
        
        # Remove from profiles
        if speaker_id in profiles:
            del profiles[speaker_id]
        
        # Delete embedding files
        delete_speaker_embeddings(speaker_id)
        
        print("✅ Deleted")
        deleted_count += 1
    
    # Save updated profiles
    print()
    print("💾 Saving updated speaker database...")
    if save_speaker_profiles(profiles):
        print("✅ Database updated successfully!")
    else:
        print("❌ Failed to save database")
        return
    
    print()
    print(f"🎉 Successfully deleted {deleted_count} speakers!")
    
    # Option to rename Ian speakers
    if len(ian_speakers) > 0:
        print()
        rename_choice = input("🏷️ Would you like to rename Ian's speakers to 'Ian'? (y/n): ").lower().strip()
        if rename_choice in ['y', 'yes']:
            print("🏷️ Renaming Ian's speakers...")
            
            for speaker_id, data in ian_speakers:
                old_name = data.get('name', 'Unknown')
                profiles[speaker_id]['name'] = 'Ian'
                print(f"   Renamed: {old_name} → Ian")
            
            print("💾 Saving updated names...")
            if save_speaker_profiles(profiles):
                print("✅ Names updated successfully!")
            else:
                print("❌ Failed to save name updates")
    
    print()
    print("📈 FINAL RESULTS:")
    remaining_speakers = len(profiles)
    print(f"   Total speakers remaining: {remaining_speakers}")
    
    if remaining_speakers > 0:
        print("   Remaining speakers:")
        for speaker_id, data in profiles.items():
            name = data.get('name', 'Unknown')
            print(f"     • {name} (ID: {speaker_id})")
    
    print()
    print("🔄 Start the backend and run 'python view_speakers.py' to verify changes!")
    print("=" * 80)

if __name__ == "__main__":
    # Change to backend directory if not already there
    if Path("backend").exists() and not Path("speaker_data").exists():
        os.chdir("backend")
        print("📁 Changed to backend directory")
    
    interactive_cleanup() 