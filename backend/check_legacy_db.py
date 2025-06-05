#!/usr/bin/env python3
import json
from pathlib import Path

try:
    legacy_db = Path("speaker_data/speaker_profiles.json")
    enhanced_db = Path("speaker_data_v2/speaker_records.json")

    print("DATABASE STATUS CHECK")
    print("=" * 40)

    if legacy_db.exists():
        try:
            with open(legacy_db, 'r') as f:
                legacy_data = json.load(f)
            print(f"❌ Legacy DB (speaker_data): {len(legacy_data)} speakers")
            first_few = list(legacy_data.items())[:3]
            for speaker_id, data in first_few:
                name = data.get('name', 'no name') if isinstance(data, dict) else str(data)
                print(f"   {speaker_id}: {name}")
        except Exception as e:
            print(f"❌ Legacy DB: Error reading - {e}")
    else:
        print("✅ Legacy DB: Not found (good!)")

    if enhanced_db.exists():
        try:
            with open(enhanced_db, 'r') as f:
                enhanced_data = json.load(f)
            print(f"✅ Enhanced DB (speaker_data_v2): {len(enhanced_data)} speakers")
            if enhanced_data:
                for speaker_id, data in enhanced_data.items():
                    name = data.get('display_name', 'no name') if isinstance(data, dict) else str(data)
                    print(f"   {speaker_id}: {name}")
            else:
                print("   (Empty - ready for new speakers)")
        except Exception as e:
            print(f"❌ Enhanced DB: Error reading - {e}")
    else:
        print("❌ Enhanced DB: Not found!")

    print("\n" + "=" * 40)
    print("DIAGNOSIS:")
    if legacy_db.exists() and enhanced_db.exists():
        print("❌ PROBLEM: Both databases exist!")
        print("   The backend is recreating the legacy database.")
        print("   Need to update backend to use enhanced DB only.")
    elif legacy_db.exists():
        print("❌ PROBLEM: Only legacy database exists!")
        print("   Enhanced database not created properly.")
    elif enhanced_db.exists():
        print("✅ SUCCESS: Only enhanced database exists!")
        print("   System properly migrated.")
    else:
        print("❌ PROBLEM: No database found!")

except Exception as e:
    print(f"Script error: {e}")
    import traceback
    traceback.print_exc() 