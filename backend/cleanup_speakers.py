#!/usr/bin/env python3
"""
Clean up speaker database for fresh testing.
This removes all speaker profiles and embeddings while preserving pretrained models.
"""

import os
import shutil
from pathlib import Path

def cleanup_speaker_database():
    """Remove all speaker data while preserving pretrained models."""
    speaker_data_dir = Path("speaker_data")
    
    if not speaker_data_dir.exists():
        print("✅ No speaker data directory found - already clean!")
        return
    
    print("🧹 Cleaning up speaker database...")
    
    # Remove speaker profiles JSON file
    profiles_file = speaker_data_dir / "speaker_profiles.json"
    if profiles_file.exists():
        profiles_file.unlink()
        print("✅ Removed speaker profiles")
    
    # Remove all embedding files but keep the directory
    embeddings_dir = speaker_data_dir / "embeddings"
    if embeddings_dir.exists():
        embedding_files = list(embeddings_dir.glob("*.npy"))
        if embedding_files:
            for file in embedding_files:
                file.unlink()
            print(f"✅ Removed {len(embedding_files)} embedding files")
        else:
            print("✅ No embedding files found")
    
    # Keep pretrained_models directory intact
    pretrained_dir = speaker_data_dir / "pretrained_models"
    if pretrained_dir.exists():
        print("✅ Preserved pretrained models (no re-download needed)")
    
    print("\n🎉 Speaker database cleaned successfully!")
    print("Ready for fresh testing of the speaker learning system.")

if __name__ == "__main__":
    cleanup_speaker_database() 