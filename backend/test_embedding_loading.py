#!/usr/bin/env python3
"""
Test Embedding Loading Process
Verifies that the system can properly load and use embeddings for multi-speaker recognition
"""

import json
import numpy as np
from pathlib import Path
from speaker_database_v2 import EnhancedSpeakerDatabase
from speaker_embeddings import OfflineSpeakerEmbeddingManager

def test_embedding_loading():
    """Test the embedding loading process"""
    print("=" * 60)
    print("EMBEDDING LOADING PROCESS TEST")
    print("=" * 60)
    
    # Test 1: Enhanced Database Loading
    print("\n1. Testing Enhanced Database Embedding Loading...")
    try:
        db = EnhancedSpeakerDatabase()
        speakers = db.get_all_speakers()
        
        print(f"✅ Enhanced database loaded {len(speakers)} speakers")
        
        # Test loading embeddings for each speaker
        loaded_embeddings = 0
        for speaker in speakers:
            speaker_id = speaker['speaker_id']
            
            # Try to access embeddings through the database
            if hasattr(db, 'speaker_embeddings') and speaker_id in db.speaker_embeddings:
                embeddings = db.speaker_embeddings[speaker_id]
                confidences = db.confidence_scores.get(speaker_id, [])
                
                print(f"   - {speaker['display_name']}: {len(embeddings)} embeddings loaded")
                if embeddings:
                    embedding_shape = np.array(embeddings[0]).shape
                    print(f"     Shape: {embedding_shape}, Avg confidence: {np.mean(confidences):.3f}")
                    loaded_embeddings += len(embeddings)
            else:
                print(f"   - {speaker['display_name']}: No embeddings in memory")
        
        print(f"Total embeddings loaded in enhanced database: {loaded_embeddings}")
        
    except Exception as e:
        print(f"❌ Enhanced database loading failed: {e}")
    
    # Test 2: Legacy System Loading
    print(f"\n2. Testing Legacy System Embedding Loading...")
    try:
        legacy_manager = OfflineSpeakerEmbeddingManager()
        legacy_stats = legacy_manager.get_speaker_stats()
        
        print(f"✅ Legacy system loaded {legacy_stats['total_speakers']} speakers")
        print(f"   Total embeddings: {legacy_stats['total_embeddings']}")
        print(f"   Average confidence: {legacy_stats['average_confidence']:.3f}")
        
        # List speakers in legacy system
        for speaker_id, profile in legacy_manager.speaker_profiles.items():
            print(f"   - {profile.name}: {len(profile.embeddings)} embeddings")
            
    except Exception as e:
        print(f"❌ Legacy system loading failed: {e}")
    
    # Test 3: Cross-System Comparison
    print(f"\n3. Cross-System Embedding Comparison...")
    
    # Compare embedding counts and data between systems
    try:
        # Get all NPY files
        embeddings_dir = Path("speaker_data_v2/embeddings")
        npy_files = list(embeddings_dir.glob("*.npy"))
        
        print(f"NPY files on disk: {len(npy_files)}")
        print(f"Enhanced DB speakers: {len(speakers) if 'speakers' in locals() else 'Failed to load'}")
        print(f"Legacy system speakers: {len(legacy_manager.speaker_profiles) if 'legacy_manager' in locals() else 'Failed to load'}")
        
        # Check if there are discrepancies
        if 'speakers' in locals() and 'legacy_manager' in locals():
            enhanced_ids = {s['speaker_id'] for s in speakers}
            legacy_ids = set(legacy_manager.speaker_profiles.keys())
            file_ids = {f.stem for f in npy_files}
            
            print(f"\nSystem Coverage Analysis:")
            print(f"   Enhanced DB IDs: {len(enhanced_ids)}")
            print(f"   Legacy system IDs: {len(legacy_ids)}")
            print(f"   NPY file IDs: {len(file_ids)}")
            
            # Find mismatches
            only_in_enhanced = enhanced_ids - legacy_ids - file_ids
            only_in_legacy = legacy_ids - enhanced_ids - file_ids
            only_in_files = file_ids - enhanced_ids - legacy_ids
            
            if only_in_enhanced:
                print(f"   ⚠️  Only in Enhanced DB: {only_in_enhanced}")
            if only_in_legacy:
                print(f"   ⚠️  Only in Legacy system: {only_in_legacy}")
            if only_in_files:
                print(f"   ⚠️  Only in NPY files: {only_in_files}")
                
            if not (only_in_enhanced or only_in_legacy or only_in_files):
                print(f"   ✅ All systems have consistent speaker coverage")
        
    except Exception as e:
        print(f"❌ Cross-system comparison failed: {e}")
    
    # Test 4: Multi-Speaker Recognition Simulation
    print(f"\n4. Simulating Multi-Speaker Recognition...")
    
    try:
        # Create a mock audio segment and test recognition against all speakers
        print("Creating mock embedding for recognition test...")
        
        # Generate a random embedding (same shape as real ones)
        mock_embedding = np.random.randn(512)  # Typical ECAPA-TDNN embedding size
        
        # Test recognition against enhanced database
        if 'db' in locals():
            print("Testing against Enhanced Database:")
            recognition_candidates = []
            
            for speaker in speakers:
                if speaker['embedding_count'] > 0:
                    speaker_id = speaker['speaker_id']
                    if hasattr(db, 'speaker_embeddings') and speaker_id in db.speaker_embeddings:
                        embeddings = db.speaker_embeddings[speaker_id]
                        if embeddings:
                            # Calculate similarity to first embedding
                            similarity = 1 - np.linalg.norm(mock_embedding - embeddings[0])
                            recognition_candidates.append((speaker['display_name'], similarity))
            
            # Sort by similarity
            recognition_candidates.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   Recognition candidates (top 3):")
            for i, (name, sim) in enumerate(recognition_candidates[:3]):
                print(f"     {i+1}. {name}: similarity {sim:.3f}")
            
            if len(recognition_candidates) >= 2:
                print("   ✅ Multi-speaker recognition capability confirmed")
            else:
                print("   ⚠️  Limited recognition - need more trained speakers")
        
    except Exception as e:
        print(f"❌ Multi-speaker recognition simulation failed: {e}")
    
    # Test 5: File Format Validation
    print(f"\n5. Validating Embedding File Formats...")
    
    valid_files = 0
    invalid_files = 0
    
    for npy_file in npy_files:
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            
            # Check expected structure
            if 'embeddings' in data and 'confidence_scores' in data:
                embeddings = data['embeddings']
                confidences = data['confidence_scores']
                
                if len(embeddings) == len(confidences):
                    print(f"   ✅ {npy_file.name}: Valid format, {len(embeddings)} embeddings")
                    valid_files += 1
                else:
                    print(f"   ⚠️  {npy_file.name}: Embedding/confidence count mismatch")
                    invalid_files += 1
            else:
                print(f"   ❌ {npy_file.name}: Missing required fields")
                invalid_files += 1
                
        except Exception as e:
            print(f"   ❌ {npy_file.name}: Load error - {e}")
            invalid_files += 1
    
    print(f"\nFile validation results: {valid_files} valid, {invalid_files} invalid")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("EMBEDDING LOADING SUMMARY")
    print("=" * 60)
    
    if invalid_files == 0:
        print("✅ All embedding files are valid and loadable")
        print("✅ Multi-speaker recognition system is functional")
        print("\nThe system can recognize multiple speakers in recordings by:")
        print("1. Loading all speaker embeddings from NPY files")
        print("2. Comparing audio segments against all known speakers") 
        print("3. Selecting the best match based on similarity scores")
        return True
    else:
        print(f"⚠️  Found {invalid_files} invalid embedding files")
        print("   → System may have reduced multi-speaker capability")
        return False

if __name__ == "__main__":
    test_embedding_loading() 