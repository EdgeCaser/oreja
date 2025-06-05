#!/usr/bin/env python3
"""
Fix Orphaned Speaker Embeddings
Either import orphaned speakers into the database or clean them up
"""

import json
import numpy as np
from pathlib import Path
from speaker_database_v2 import EnhancedSpeakerDatabase
from datetime import datetime

def fix_orphaned_speakers():
    """Fix orphaned speaker embeddings"""
    print("=" * 60)
    print("FIXING ORPHANED SPEAKER EMBEDDINGS")
    print("=" * 60)
    
    # Load current database
    db = EnhancedSpeakerDatabase()
    speakers = db.get_all_speakers()
    known_speaker_ids = {s['speaker_id'] for s in speakers}
    
    # Find orphaned files
    embeddings_dir = Path("speaker_data_v2/embeddings")
    all_npy_files = list(embeddings_dir.glob("*.npy"))
    
    orphaned_files = []
    for npy_file in all_npy_files:
        speaker_id = npy_file.stem
        if speaker_id not in known_speaker_ids:
            orphaned_files.append((speaker_id, npy_file))
    
    if not orphaned_files:
        print("✅ No orphaned files found!")
        return True
    
    print(f"Found {len(orphaned_files)} orphaned embedding files:")
    
    # Analyze each orphaned file
    for speaker_id, npy_file in orphaned_files:
        print(f"\n--- {speaker_id} ---")
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            embeddings = data.get('embeddings', [])
            confidences = data.get('confidence_scores', [])
            
            print(f"File: {npy_file.name}")
            print(f"Embeddings: {len(embeddings)}")
            print(f"Average confidence: {np.mean(confidences):.3f}")
            
            if embeddings:
                embedding_shape = np.array(embeddings[0]).shape
                print(f"Embedding shape: {embedding_shape}")
            
        except Exception as e:
            print(f"❌ Error reading {npy_file.name}: {e}")
            continue
    
    # Ask user what to do
    print(f"\n" + "=" * 60)
    print("OPTIONS")
    print("=" * 60)
    print("1. Import orphaned speakers into database (recommended)")
    print("2. Delete orphaned embedding files")
    print("3. Keep orphaned files (no action)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        return import_orphaned_speakers(orphaned_files, db)
    elif choice == "2":
        return delete_orphaned_files(orphaned_files)
    else:
        print("No action taken.")
        return True

def import_orphaned_speakers(orphaned_files, db):
    """Import orphaned speakers into the database"""
    print(f"\n🔄 Importing {len(orphaned_files)} orphaned speakers...")
    
    imported = 0
    failed = 0
    
    for speaker_id, npy_file in orphaned_files:
        try:
            # Load embedding data
            data = np.load(npy_file, allow_pickle=True).item()
            embeddings = data.get('embeddings', [])
            confidences = data.get('confidence_scores', [])
            
            if not embeddings:
                print(f"⚠️  Skipping {speaker_id}: No embeddings")
                continue
            
            # Create a new speaker record with the original speaker_id
            # Generate a reasonable display name
            display_name = f"Imported Speaker {speaker_id[-8:]}"
            
            # Check if this speaker_id already exists
            if speaker_id in {s['speaker_id'] for s in db.get_all_speakers()}:
                print(f"⚠️  Skipping {speaker_id}: Already exists in database")
                continue
            
            # Manually create the speaker record
            from speaker_database_v2 import SpeakerRecord
            from dataclasses import asdict
            
            new_record = SpeakerRecord(
                speaker_id=speaker_id,
                display_name=display_name,
                created_date=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                session_count=0,
                total_audio_seconds=0.0,
                embedding_count=len(embeddings),
                average_confidence=np.mean(confidences) if confidences else 0.0,
                is_enrolled=False,
                is_verified=False,
                source_type="imported"
            )
            
            # Add to database
            db.speaker_records[speaker_id] = new_record
            db.speaker_embeddings[speaker_id] = embeddings
            db.confidence_scores[speaker_id] = confidences
            
            print(f"✅ Imported {display_name} ({speaker_id[:12]}...): {len(embeddings)} embeddings")
            imported += 1
            
        except Exception as e:
            print(f"❌ Failed to import {speaker_id}: {e}")
            failed += 1
    
    # Save the updated database
    try:
        db._save_database()
        db._rebuild_indexes()
        print(f"\n✅ Successfully imported {imported} speakers!")
        if failed > 0:
            print(f"⚠️  Failed to import {failed} speakers")
        return True
    except Exception as e:
        print(f"❌ Failed to save database: {e}")
        return False

def delete_orphaned_files(orphaned_files):
    """Delete orphaned embedding files"""
    print(f"\n🗑️  Deleting {len(orphaned_files)} orphaned files...")
    
    deleted = 0
    failed = 0
    
    for speaker_id, npy_file in orphaned_files:
        try:
            npy_file.unlink()
            print(f"✅ Deleted {npy_file.name}")
            deleted += 1
        except Exception as e:
            print(f"❌ Failed to delete {npy_file.name}: {e}")
            failed += 1
    
    print(f"\n✅ Deleted {deleted} files!")
    if failed > 0:
        print(f"⚠️  Failed to delete {failed} files")
    
    return failed == 0

if __name__ == "__main__":
    fix_orphaned_speakers() 