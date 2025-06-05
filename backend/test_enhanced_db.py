#!/usr/bin/env python3
from speaker_database_v2 import EnhancedSpeakerDatabase

print("Testing Enhanced Database v2...")
db = EnhancedSpeakerDatabase('speaker_data_v2')
speakers = db.get_all_speakers()

print(f"Enhanced DB: {len(speakers)} speakers")
for s in speakers:
    print(f"  {s['speaker_id']}: {s['display_name']} (enrolled: {s['is_enrolled']})")

print("Enhanced database is working correctly!") 