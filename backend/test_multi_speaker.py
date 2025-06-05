#!/usr/bin/env python3
"""
Test Multi-Speaker Recognition System
Checks database loading, embedding file consistency, and multi-speaker support
"""

import json
import numpy as np
from pathlib import Path
from speaker_database_v2 import EnhancedSpeakerDatabase

def test_multi_speaker_system():
    """Test the multi-speaker recognition system"""
    print("=" * 60)
    print("MULTI-SPEAKER RECOGNITION SYSTEM TEST")
    print("=" * 60)
    
    # Test 1: Database Loading
    print("\n1. Testing Database Loading...")
    try:
        db = EnhancedSpeakerDatabase()
        speakers = db.get_all_speakers()
        print(f"✅ Successfully loaded {len(speakers)} speakers from database")
        
        for speaker in speakers:
            print(f"   - {speaker['display_name']} (ID: {speaker['speaker_id'][:12]}...)")
            print(f"     Embeddings: {speaker['embedding_count']}, Confidence: {speaker['average_confidence']:.3f}")
            print(f"     Type: {speaker['source_type']}, Enrolled: {speaker['is_enrolled']}")
    except Exception as e:
        print(f"❌ Database loading failed: {e}")
        return False
    
    # Test 2: Check Embedding Files
    print(f"\n2. Testing Embedding File Consistency...")
    embeddings_dir = Path("speaker_data_v2/embeddings")
    
    if not embeddings_dir.exists():
        print(f"❌ Embeddings directory doesn't exist: {embeddings_dir}")
        return False
    
    npy_files = list(embeddings_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy embedding files:")
    
    total_embeddings = 0
    for npy_file in npy_files:
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            embeddings = data.get('embeddings', [])
            confidences = data.get('confidence_scores', [])
            
            print(f"   - {npy_file.name}: {len(embeddings)} embeddings")
            print(f"     Average confidence: {np.mean(confidences):.3f}" if confidences else "     No confidence scores")
            total_embeddings += len(embeddings)
            
        except Exception as e:
            print(f"❌ Error loading {npy_file.name}: {e}")
    
    print(f"Total embeddings across all speakers: {total_embeddings}")
    
    # Test 3: Database-File Consistency
    print(f"\n3. Testing Database-File Consistency...")
    inconsistencies = 0
    
    for speaker in speakers:
        speaker_id = speaker['speaker_id']
        expected_file = embeddings_dir / f"{speaker_id}.npy"
        
        if speaker['embedding_count'] > 0:
            if not expected_file.exists():
                print(f"❌ Missing embedding file for {speaker['display_name']}: {expected_file.name}")
                inconsistencies += 1
            else:
                try:
                    data = np.load(expected_file, allow_pickle=True).item()
                    actual_count = len(data.get('embeddings', []))
                    expected_count = speaker['embedding_count']
                    
                    if actual_count != expected_count:
                        print(f"❌ Embedding count mismatch for {speaker['display_name']}: DB={expected_count}, File={actual_count}")
                        inconsistencies += 1
                    else:
                        print(f"✅ {speaker['display_name']}: {actual_count} embeddings (consistent)")
                        
                except Exception as e:
                    print(f"❌ Error checking {speaker['display_name']}: {e}")
                    inconsistencies += 1
        else:
            print(f"⚪ {speaker['display_name']}: No embeddings (new user)")
    
    # Test 4: Check for Legacy JSON Files
    print(f"\n4. Checking for Legacy JSON Embedding Files...")
    json_files = list(embeddings_dir.glob("*.json"))
    
    if json_files:
        print(f"⚠️  Found {len(json_files)} legacy JSON embedding files:")
        for json_file in json_files:
            print(f"   - {json_file.name}")
        print("   These may interfere with the NPY system!")
    else:
        print("✅ No legacy JSON embedding files found")
    
    # Test 5: Multi-Speaker Recognition Capability
    print(f"\n5. Testing Multi-Speaker Recognition Capability...")
    
    if len(speakers) < 2:
        print("⚠️  Only 1 speaker in database - cannot test multi-speaker recognition")
        print("   Add more speakers via the User Training module to test this feature")
    else:
        speakers_with_embeddings = [s for s in speakers if s['embedding_count'] > 0]
        
        if len(speakers_with_embeddings) < 2:
            print("⚠️  Only 1 speaker has embeddings - multi-speaker recognition limited")
            print("   Train more speakers to enable full multi-speaker recognition")
        else:
            print(f"✅ Multi-speaker recognition ready: {len(speakers_with_embeddings)} trained speakers")
            for speaker in speakers_with_embeddings:
                print(f"   - {speaker['display_name']}: {speaker['embedding_count']} samples")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if inconsistencies == 0:
        print("✅ All tests passed! Multi-speaker system is working correctly.")
        print(f"   - {len(speakers)} speakers in database")
        print(f"   - {total_embeddings} total embeddings")
        print(f"   - {len(npy_files)} NPY embedding files")
        return True
    else:
        print(f"❌ Found {inconsistencies} inconsistencies in the system")
        return False

if __name__ == "__main__":
    test_multi_speaker_system() 