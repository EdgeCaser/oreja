#!/usr/bin/env python3
"""
Sync Speaker Database for Batch Transcriber
Copies the current speaker database to the legacy format expected by the batch transcriber
"""

import json
import shutil
import numpy as np
from pathlib import Path
from speaker_database_v2 import EnhancedSpeakerDatabase

def sync_batch_transcriber_database():
    """Sync the enhanced database to the legacy format for batch transcriber"""
    print("=" * 60)
    print("SYNCING SPEAKER DATABASE FOR BATCH TRANSCRIBER")
    print("=" * 60)
    
    # Source: Enhanced database
    source_dir = Path("speaker_data_v2")
    source_records = source_dir / "speaker_records.json"
    source_embeddings = source_dir / "embeddings"
    
    # Target: Legacy compatibility directory
    target_dir = Path("speaker_data_v2_legacy_compatibility")
    target_profiles = target_dir / "speaker_profiles.json"
    target_embeddings = target_dir / "embeddings"
    
    # Ensure target directories exist
    target_dir.mkdir(exist_ok=True)
    target_embeddings.mkdir(exist_ok=True)
    
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    # Load enhanced database
    try:
        db = EnhancedSpeakerDatabase()
        speakers = db.get_all_speakers()
        print(f"✅ Loaded {len(speakers)} speakers from enhanced database")
    except Exception as e:
        print(f"❌ Failed to load enhanced database: {e}")
        return False
    
    # Convert to legacy format
    legacy_profiles = {}
    synced_speakers = 0
    synced_embeddings = 0
    
    for speaker in speakers:
        speaker_id = speaker['speaker_id']
        
        # Convert to legacy profile format
        legacy_profile = {
            'speaker_id': speaker_id,
            'name': speaker['display_name'],
            'created_date': speaker['created_date'],
            'last_seen': speaker['last_seen'],
            'session_count': speaker.get('session_count', 0),
            'total_audio_seconds': speaker.get('total_audio_seconds', 0.0),
            'embedding_count': speaker['embedding_count'],
            'average_confidence': speaker['average_confidence']
        }
        
        legacy_profiles[speaker_id] = legacy_profile
        synced_speakers += 1
        
        # Copy embedding file if it exists and has embeddings
        if speaker['embedding_count'] > 0:
            source_embedding_file = source_embeddings / f"{speaker_id}.npy"
            target_embedding_file = target_embeddings / f"{speaker_id}.npy"
            
            if source_embedding_file.exists():
                try:
                    # Copy the embedding file
                    shutil.copy2(source_embedding_file, target_embedding_file)
                    
                    # Verify the copy
                    data = np.load(target_embedding_file, allow_pickle=True).item()
                    embeddings = data.get('embeddings', [])
                    
                    print(f"   ✅ {speaker['display_name']}: {len(embeddings)} embeddings synced")
                    synced_embeddings += len(embeddings)
                    
                except Exception as e:
                    print(f"   ❌ Failed to sync embeddings for {speaker['display_name']}: {e}")
            else:
                print(f"   ⚠️  {speaker['display_name']}: Missing embedding file")
        else:
            print(f"   ⚪ {speaker['display_name']}: No embeddings to sync")
    
    # Save legacy profiles file
    try:
        with open(target_profiles, 'w') as f:
            json.dump(legacy_profiles, f, indent=2)
        print(f"✅ Saved legacy profiles to {target_profiles}")
    except Exception as e:
        print(f"❌ Failed to save legacy profiles: {e}")
        return False
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SYNC SUMMARY")
    print("=" * 60)
    print(f"✅ Synced {synced_speakers} speakers")
    print(f"✅ Synced {synced_embeddings} total embeddings")
    print(f"✅ Legacy database ready at: {target_dir}")
    
    print(f"\n📂 BATCH TRANSCRIBER MASTER FILE:")
    print(f"   {target_profiles}")
    print(f"\n📁 BATCH TRANSCRIBER EMBEDDINGS:")
    print(f"   {target_embeddings}/")
    
    print(f"\n🎯 The Batch Transcriber will now work correctly!")
    print(f"   It will automatically load all speakers from the legacy directory.")
    
    return True

def verify_batch_transcriber_setup():
    """Verify that the batch transcriber can load the synced database"""
    print(f"\n" + "=" * 60)
    print("VERIFYING BATCH TRANSCRIBER SETUP")
    print("=" * 60)
    
    try:
        from speaker_embeddings import OfflineSpeakerEmbeddingManager
        
        # This should load from speaker_data_v2_legacy_compatibility
        speaker_manager = OfflineSpeakerEmbeddingManager()
        
        print(f"✅ Batch transcriber loaded {len(speaker_manager.speaker_profiles)} speakers")
        
        for speaker_id, profile in speaker_manager.speaker_profiles.items():
            print(f"   - {profile.name}: {len(profile.embeddings)} embeddings")
        
        print(f"✅ Batch Transcriber database verification successful!")
        return True
        
    except Exception as e:
        print(f"❌ Batch transcriber verification failed: {e}")
        return False

if __name__ == "__main__":
    success = sync_batch_transcriber_database()
    if success:
        verify_batch_transcriber_setup()
    else:
        print("❌ Sync failed - Batch Transcriber may not work correctly") 