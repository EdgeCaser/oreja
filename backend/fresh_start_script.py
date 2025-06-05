#!/usr/bin/env python3
"""
Fresh Start Script for Enhanced Speaker Database
Safely backs up current database and initializes enhanced architecture
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

def backup_current_database():
    """Create a backup of the current speaker database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"speaker_data_backup_pre_enhanced_{timestamp}")
    
    print(f"Creating backup at: {backup_dir}")
    
    # Create backup directory
    backup_dir.mkdir(exist_ok=True)
    
    # Backup current speaker_data directory
    if Path("speaker_data").exists():
        shutil.copytree("speaker_data", backup_dir / "speaker_data")
        print(f"✓ Backed up speaker_data/ to {backup_dir}/speaker_data/")
    
    # Create backup summary
    summary = {
        "backup_date": datetime.now().isoformat(),
        "reason": "Pre-enhanced architecture fresh start",
        "original_location": "speaker_data/",
        "backup_location": str(backup_dir),
        "notes": "Database contained ~461 auto-generated speakers with minimal training data"
    }
    
    with open(backup_dir / "backup_info.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Created backup summary at {backup_dir}/backup_info.json")
    return backup_dir

def analyze_current_database():
    """Analyze what's being backed up"""
    speaker_file = Path("speaker_data/speaker_profiles.json")
    
    if not speaker_file.exists():
        print("No existing speaker database found.")
        return {"total_speakers": 0}
    
    with open(speaker_file) as f:
        data = json.load(f)
    
    analysis = {
        "total_speakers": len(data),
        "auto_speakers": len([k for k in data.keys() if k.startswith("AUTO_SPEAKER_")]),
        "enrolled_speakers": len([k for k in data.keys() if k.startswith("ENROLLED_")]),
        "corrected_speakers": len([k for k in data.keys() if k.startswith("CORRECTED_")]),
        "total_embeddings": sum(profile.get("embedding_count", 0) for profile in data.values()),
        "speakers_with_multiple_embeddings": len([p for p in data.values() if p.get("embedding_count", 0) > 1])
    }
    
    print("\n" + "="*50)
    print("CURRENT DATABASE ANALYSIS")
    print("="*50)
    print(f"Total speakers: {analysis['total_speakers']}")
    print(f"Auto-generated speakers: {analysis['auto_speakers']}")
    print(f"Enrolled speakers: {analysis['enrolled_speakers']}")
    print(f"Corrected speakers: {analysis['corrected_speakers']}")
    print(f"Total embeddings: {analysis['total_embeddings']}")
    print(f"Speakers with >1 embedding: {analysis['speakers_with_multiple_embeddings']}")
    print("="*50)
    
    return analysis

def clear_current_database():
    """Safely remove current database after backup"""
    if Path("speaker_data").exists():
        shutil.rmtree("speaker_data")
        print("✓ Removed current speaker_data directory")
    else:
        print("No speaker_data directory to remove")

def initialize_enhanced_database():
    """Initialize the enhanced speaker database v2"""
    try:
        from speaker_database_v2 import EnhancedSpeakerDatabase
        
        # Initialize enhanced database
        enhanced_db = EnhancedSpeakerDatabase("speaker_data_v2")
        
        print("✓ Initialized enhanced speaker database v2")
        print(f"✓ Database location: speaker_data_v2/")
        
        # Verify initialization
        stats = enhanced_db.get_all_speakers()
        print(f"✓ Enhanced database initialized with {len(stats)} speakers")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize enhanced database: {e}")
        return False

def main():
    """Main fresh start process"""
    print("="*60)
    print("OREJA SPEAKER DATABASE FRESH START")
    print("="*60)
    print("This script will:")
    print("1. Analyze your current database")
    print("2. Create a backup of existing data")
    print("3. Clear the current database")
    print("4. Initialize the enhanced speaker database v2")
    print()
    
    # Step 1: Analyze current database
    print("Step 1: Analyzing current database...")
    analysis = analyze_current_database()
    
    # Check if there's anything valuable to lose
    valuable_data = (
        analysis.get("enrolled_speakers", 0) > 0 or 
        analysis.get("corrected_speakers", 0) > 0 or
        analysis.get("speakers_with_multiple_embeddings", 0) > 10
    )
    
    if valuable_data:
        print("\n⚠️  WARNING: Your database contains potentially valuable data!")
        print("Consider migration instead of fresh start.")
        response = input("Continue with fresh start anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted. Consider using the migration endpoints instead.")
            return
    else:
        print("\n✓ Database contains mostly auto-generated data - safe for fresh start")
    
    # Step 2: Create backup
    print("\nStep 2: Creating backup...")
    backup_dir = backup_current_database()
    
    # Step 3: Clear current database
    print("\nStep 3: Clearing current database...")
    clear_current_database()
    
    # Step 4: Initialize enhanced database
    print("\nStep 4: Initializing enhanced database...")
    success = initialize_enhanced_database()
    
    if success:
        print("\n" + "="*60)
        print("✅ FRESH START COMPLETE!")
        print("="*60)
        print(f"• Backup created at: {backup_dir}")
        print("• Enhanced speaker database v2 initialized")
        print("• All new speakers will use the enhanced architecture")
        print("• Speaker names will no longer reset to SPEAKER_NNNN")
        print("• Merging will be intelligent (based on sample count)")
        print("• Save and feedback operations are now separate")
        print()
        print("Next steps:")
        print("1. Start the server: python server.py")
        print("2. Check system status: GET /speakers/system_status")
        print("3. Begin using the enhanced speaker management!")
    else:
        print("\n❌ Fresh start incomplete - enhanced database failed to initialize")
        print("Your backup is safe. Check the error above and try again.")

if __name__ == "__main__":
    main() 