#!/usr/bin/env python3
"""
Check for Legacy JSON Files and System Inconsistencies
"""

import json
import numpy as np
from pathlib import Path
from speaker_database_v2 import EnhancedSpeakerDatabase

def check_legacy_and_inconsistencies():
    """Check for legacy files and system inconsistencies"""
    print("=" * 60)
    print("LEGACY FILES AND INCONSISTENCY CHECK")
    print("=" * 60)
    
    # Check 1: Look for old speaker data directories
    print("\n1. Checking for Legacy Directories...")
    legacy_dirs = [
        "speaker_data",
        "speaker_data_backup_20250602_125402",
        "speaker_data_backup_20250602_125328", 
        "speaker_data_backup_20250602_125132",
        "speaker_data_backup_pre_enhanced_20250605_103940",
        "speaker_data_v2_legacy_compatibility"
    ]
    
    found_legacy = False
    for legacy_dir in legacy_dirs:
        legacy_path = Path(legacy_dir)
        if legacy_path.exists():
            print(f"✅ Found legacy directory: {legacy_dir}")
            found_legacy = True
            
            # Check for legacy embedding files
            if (legacy_path / "embeddings").exists():
                legacy_embeddings = list((legacy_path / "embeddings").glob("*"))
                print(f"   Contains {len(legacy_embeddings)} embedding files:")
                for file in legacy_embeddings[:5]:  # Show first 5
                    print(f"     - {file.name}")
                if len(legacy_embeddings) > 5:
                    print(f"     ... and {len(legacy_embeddings) - 5} more")
        else:
            print(f"⚪ No legacy directory: {legacy_dir}")
    
    if not found_legacy:
        print("✅ No legacy directories found")
    
    # Check 2: Orphaned embedding files
    print(f"\n2. Checking for Orphaned Embedding Files...")
    
    # Load current database
    db = EnhancedSpeakerDatabase()
    speakers = db.get_all_speakers()
    known_speaker_ids = {s['speaker_id'] for s in speakers}
    
    embeddings_dir = Path("speaker_data_v2/embeddings")
    all_npy_files = list(embeddings_dir.glob("*.npy"))
    
    orphaned_files = []
    for npy_file in all_npy_files:
        # Extract speaker ID from filename (remove .npy extension)
        speaker_id = npy_file.stem
        if speaker_id not in known_speaker_ids:
            orphaned_files.append(npy_file)
            
    if orphaned_files:
        print(f"⚠️  Found {len(orphaned_files)} orphaned embedding files:")
        for orphaned in orphaned_files:
            print(f"   - {orphaned.name} (speaker not in database)")
            
            # Try to load and show info about orphaned file
            try:
                data = np.load(orphaned, allow_pickle=True).item()
                embeddings = data.get('embeddings', [])
                confidences = data.get('confidence_scores', [])
                print(f"     Contains: {len(embeddings)} embeddings, avg confidence: {np.mean(confidences):.3f}")
            except Exception as e:
                print(f"     Error reading file: {e}")
    else:
        print("✅ No orphaned embedding files found")
    
    # Check 3: Missing embedding files for speakers
    print(f"\n3. Checking for Missing Embedding Files...")
    
    missing_files = []
    for speaker in speakers:
        if speaker['embedding_count'] > 0:
            expected_file = embeddings_dir / f"{speaker['speaker_id']}.npy"
            if not expected_file.exists():
                missing_files.append((speaker, expected_file))
    
    if missing_files:
        print(f"❌ Found {len(missing_files)} speakers with missing embedding files:")
        for speaker, missing_file in missing_files:
            print(f"   - {speaker['display_name']} ({speaker['speaker_id']})")
            print(f"     Expected: {missing_file.name}")
            print(f"     Database claims: {speaker['embedding_count']} embeddings")
    else:
        print("✅ All speakers with embeddings have corresponding files")
    
    # Check 4: JSON embedding files (legacy format)
    print(f"\n4. Checking for Legacy JSON Embedding Files...")
    
    json_files = list(embeddings_dir.glob("*.json"))
    if json_files:
        print(f"⚠️  Found {len(json_files)} JSON embedding files (legacy format):")
        for json_file in json_files:
            print(f"   - {json_file.name}")
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                print(f"     JSON structure: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
            except Exception as e:
                print(f"     Error reading JSON: {e}")
    else:
        print("✅ No legacy JSON embedding files found")
    
    # Check 5: File formats and sizes
    print(f"\n5. Analyzing Embedding File Formats...")
    
    for npy_file in all_npy_files:
        try:
            file_size = npy_file.stat().st_size
            data = np.load(npy_file, allow_pickle=True).item()
            
            embeddings = data.get('embeddings', [])
            confidences = data.get('confidence_scores', [])
            
            print(f"   - {npy_file.name}: {file_size} bytes")
            print(f"     Structure: {len(embeddings)} embeddings, {len(confidences)} confidence scores")
            
            if embeddings:
                embedding_shape = np.array(embeddings[0]).shape if embeddings else "N/A"
                print(f"     Embedding shape: {embedding_shape}")
                
        except Exception as e:
            print(f"   - {npy_file.name}: ❌ Error reading file: {e}")
    
    # Summary and recommendations
    print(f"\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    issues_found = len(orphaned_files) + len(missing_files) + len(json_files)
    
    if issues_found == 0:
        print("✅ No issues found! System is clean.")
    else:
        print(f"⚠️  Found {issues_found} potential issues:")
        
        if orphaned_files:
            print(f"   • {len(orphaned_files)} orphaned embedding files")
            print("     → Consider cleaning up or re-importing these speakers")
            
        if missing_files:
            print(f"   • {len(missing_files)} speakers missing embedding files")
            print("     → Database may need correction or speakers need re-training")
            
        if json_files:
            print(f"   • {len(json_files)} legacy JSON embedding files")
            print("     → Consider migrating to NPY format or removing if no longer needed")
    
    return issues_found == 0

if __name__ == "__main__":
    check_legacy_and_inconsistencies() 