#!/usr/bin/env python3
"""
Re-transcription Helper
Analyzes existing transcriptions and helps decide which ones benefit most from re-running
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

def analyze_transcription_quality(transcription_file: Path) -> Dict:
    """Analyze the quality of an existing transcription"""
    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = data.get('segments', [])
        if not segments:
            return {'error': 'No segments found'}
        
        analysis = {
            'file': transcription_file.name,
            'total_segments': len(segments),
            'duration_minutes': 0,
            'unique_speakers': set(),
            'speaker_quality': {
                'generic_speakers': 0,  # SPEAKER_XX names
                'unknown_speakers': 0,  # UNKNOWN_SPEAKER_XX
                'zero_confidence': 0,   # confidence = 0.0
                'fallback_method': 0    # using fallback diarization
            },
            'priority_score': 0  # Higher = more important to re-transcribe
        }
        
        total_duration = 0
        
        for segment in segments:
            # Calculate duration
            start = segment.get('start', 0)
            end = segment.get('end', start)
            total_duration = max(total_duration, end)
            
            # Analyze speaker quality
            speaker = segment.get('speaker', '')
            embedding_speaker = segment.get('embedding_speaker', '')
            confidence = segment.get('speaker_confidence', 0.0)
            identification_method = segment.get('identification_method', '')
            
            # Track unique speakers
            analysis['unique_speakers'].add(speaker)
            
            # Count quality issues
            if 'SPEAKER_' in speaker and speaker.startswith('Speaker SPEAKER_'):
                analysis['speaker_quality']['generic_speakers'] += 1
            
            if 'UNKNOWN_SPEAKER_' in embedding_speaker:
                analysis['speaker_quality']['unknown_speakers'] += 1
            
            if confidence == 0.0:
                analysis['speaker_quality']['zero_confidence'] += 1
            
            if identification_method == 'fallback_diarization':
                analysis['speaker_quality']['fallback_method'] += 1
        
        analysis['duration_minutes'] = total_duration / 60
        analysis['unique_speakers'] = len(analysis['unique_speakers'])
        
        # Calculate priority score (0-100)
        total_segments = analysis['total_segments']
        if total_segments > 0:
            generic_ratio = analysis['speaker_quality']['generic_speakers'] / total_segments
            unknown_ratio = analysis['speaker_quality']['unknown_speakers'] / total_segments  
            zero_conf_ratio = analysis['speaker_quality']['zero_confidence'] / total_segments
            fallback_ratio = analysis['speaker_quality']['fallback_method'] / total_segments
            
            # Higher score = more benefit from re-transcription
            priority_score = (
                generic_ratio * 30 +      # Generic speakers
                unknown_ratio * 25 +      # Unknown speakers  
                zero_conf_ratio * 25 +    # Zero confidence
                fallback_ratio * 20       # Fallback method
            )
            
            # Bonus for longer recordings (more valuable to fix)
            if analysis['duration_minutes'] > 30:
                priority_score += 10
            elif analysis['duration_minutes'] > 10:
                priority_score += 5
            
            # Bonus for multiple speakers (more complex to fix manually)
            if analysis['unique_speakers'] > 2:
                priority_score += 10
            
            analysis['priority_score'] = min(100, priority_score)
        
        return analysis
        
    except Exception as e:
        return {'error': f'Failed to analyze {transcription_file}: {e}'}

def find_source_audio_files() -> List[Path]:
    """Find available source audio files for re-transcription"""
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    audio_files = []
    
    # Check common audio directories
    search_paths = [
        Path('.'),
        Path('..'),
        Path('../audio'),
        Path('./audio'),
        Path('./recordings')
    ]
    
    for search_path in search_paths:
        if search_path.exists():
            for file_path in search_path.rglob('*'):
                if file_path.suffix.lower() in audio_extensions:
                    audio_files.append(file_path)
    
    return audio_files

def match_transcriptions_to_audio(transcriptions: List[Dict], audio_files: List[Path]) -> List[Dict]:
    """Try to match transcription files to their source audio files"""
    matched = []
    
    for transcription in transcriptions:
        transcription_name = transcription['file']
        best_match = None
        best_score = 0
        
        # Try to find matching audio file
        for audio_file in audio_files:
            audio_name = audio_file.stem
            
            # Simple name matching - could be improved
            if audio_name in transcription_name or transcription_name.startswith(audio_name):
                score = len(audio_name)  # Longer matches are better
                if score > best_score:
                    best_match = audio_file
                    best_score = score
        
        transcription_copy = transcription.copy()
        transcription_copy['source_audio'] = str(best_match) if best_match else 'NOT_FOUND'
        transcription_copy['can_retranscribe'] = best_match is not None
        matched.append(transcription_copy)
    
    return matched

def generate_retranscription_plan(transcriptions: List[Dict]) -> Dict:
    """Generate a strategic re-transcription plan"""
    
    # Sort by priority score (highest first)
    transcriptions.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
    
    plan = {
        'high_priority': [],      # Score > 70
        'medium_priority': [],    # Score 40-70  
        'low_priority': [],       # Score < 40
        'cannot_retranscribe': [] # No source audio found
    }
    
    for transcription in transcriptions:
        score = transcription.get('priority_score', 0)
        
        if not transcription.get('can_retranscribe', False):
            plan['cannot_retranscribe'].append(transcription)
        elif score > 70:
            plan['high_priority'].append(transcription)
        elif score > 40:
            plan['medium_priority'].append(transcription)
        else:
            plan['low_priority'].append(transcription)
    
    return plan

def main():
    """Main analysis and planning function"""
    print("="*60)
    print("OREJA RE-TRANSCRIPTION ANALYSIS")
    print("="*60)
    
    # Find all transcription files
    transcription_dir = Path("transcription_results")
    
    if not transcription_dir.exists():
        print("❌ No transcription_results directory found")
        return
    
    transcription_files = list(transcription_dir.glob("*.json"))
    
    if not transcription_files:
        print("❌ No transcription JSON files found")
        return
    
    print(f"Found {len(transcription_files)} transcription files")
    
    # Analyze each transcription
    print("\n🔍 Analyzing transcription quality...")
    transcription_analyses = []
    
    for transcription_file in transcription_files:
        analysis = analyze_transcription_quality(transcription_file)
        if 'error' not in analysis:
            transcription_analyses.append(analysis)
        else:
            print(f"⚠️  {analysis['error']}")
    
    if not transcription_analyses:
        print("❌ No valid transcriptions to analyze")
        return
    
    # Find source audio files
    print("\n🎵 Searching for source audio files...")
    audio_files = find_source_audio_files()
    print(f"Found {len(audio_files)} audio files")
    
    # Match transcriptions to audio
    matched_transcriptions = match_transcriptions_to_audio(transcription_analyses, audio_files)
    
    # Generate re-transcription plan
    plan = generate_retranscription_plan(matched_transcriptions)
    
    # Display results
    print("\n" + "="*60)
    print("RE-TRANSCRIPTION RECOMMENDATIONS")
    print("="*60)
    
    print(f"\n🔥 HIGH PRIORITY (Re-transcribe ASAP):")
    for item in plan['high_priority']:
        print(f"  📁 {item['file']}")
        print(f"     Priority Score: {item['priority_score']:.1f}/100")
        print(f"     Duration: {item['duration_minutes']:.1f} min")
        print(f"     Speakers: {item['unique_speakers']}")
        print(f"     Audio: {item['source_audio']}")
        print(f"     Issues: {item['speaker_quality']['zero_confidence']}/{item['total_segments']} zero confidence")
        print()
    
    print(f"\n⚡ MEDIUM PRIORITY:")
    for item in plan['medium_priority'][:3]:  # Show top 3
        print(f"  📁 {item['file']} (Score: {item['priority_score']:.1f}, {item['duration_minutes']:.1f} min)")
    
    if len(plan['medium_priority']) > 3:
        print(f"  ... and {len(plan['medium_priority']) - 3} more")
    
    print(f"\n📊 SUMMARY:")
    print(f"  High Priority: {len(plan['high_priority'])} files")
    print(f"  Medium Priority: {len(plan['medium_priority'])} files") 
    print(f"  Low Priority: {len(plan['low_priority'])} files")
    print(f"  Cannot Re-transcribe: {len(plan['cannot_retranscribe'])} files (no source audio)")
    
    total_can_retranscribe = len(plan['high_priority']) + len(plan['medium_priority']) + len(plan['low_priority'])
    
    print(f"\n💡 RECOMMENDATIONS:")
    if len(plan['high_priority']) > 0:
        print(f"  1. Start with {len(plan['high_priority'])} high-priority files")
        print(f"  2. These have the most speaker identification issues")
        print(f"  3. Enhanced database will provide immutable speaker IDs")
        print(f"  4. Audio-based splitting will improve accuracy")
    
    if total_can_retranscribe == 0:
        print("  ❌ No source audio files found - cannot re-transcribe")
    else:
        print(f"  ✅ {total_can_retranscribe} files can be re-transcribed with enhanced system")
    
    # Save detailed plan
    plan_file = f"retranscription_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(plan_file, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        serializable_plan = {}
        for key, items in plan.items():
            serializable_plan[key] = []
            for item in items:
                item_copy = item.copy()
                if 'source_audio' in item_copy and isinstance(item_copy['source_audio'], Path):
                    item_copy['source_audio'] = str(item_copy['source_audio'])
                serializable_plan[key].append(item_copy)
        
        json.dump(serializable_plan, f, indent=2)
    
    print(f"\n📄 Detailed plan saved to: {plan_file}")

if __name__ == "__main__":
    main() 