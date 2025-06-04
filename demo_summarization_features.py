#!/usr/bin/env python3
"""
Demo script to showcase the new conversation summarization and annotation features
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_transcription_processor import EnhancedTranscriptionProcessor
except ImportError:
    print("Enhanced transcription processor not available")
    print("Make sure you have installed the enhanced requirements:")
    print("pip install -r requirements_enhanced.txt")
    sys.exit(1)


def create_sample_transcription() -> Dict[str, Any]:
    """Create a sample transcription for demonstration"""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 3.5,
                "text": "Good morning everyone, thanks for joining today's product planning meeting.",
                "speaker": "Alice"
            },
            {
                "start": 4.0,
                "end": 8.2,
                "text": "What are our main priorities for the Q4 roadmap?",
                "speaker": "Alice"
            },
            {
                "start": 9.1,
                "end": 15.7,
                "text": "I think we need to focus on the mobile app improvements and the new API integration.",
                "speaker": "Bob"
            },
            {
                "start": 16.2,
                "end": 22.4,
                "text": "Agreed. The customer feedback has been clear about mobile performance issues.",
                "speaker": "Charlie"
            },
            {
                "start": 23.1,
                "end": 28.9,
                "text": "Should we prioritize the performance fixes or the new features first?",
                "speaker": "Alice"
            },
            {
                "start": 29.5,
                "end": 36.8,
                "text": "I believe we must fix the performance issues first. New features won't matter if the app is slow.",
                "speaker": "Bob"
            },
            {
                "start": 37.3,
                "end": 42.1,
                "text": "That's a good point. Let's make performance our top priority.",
                "speaker": "Charlie"
            },
            {
                "start": 43.0,
                "end": 48.5,
                "text": "Excellent. So our action items are: Bob will lead the performance optimization team.",
                "speaker": "Alice"
            },
            {
                "start": 49.2,
                "end": 55.3,
                "text": "And Charlie will coordinate with the design team for the mobile improvements.",
                "speaker": "Alice"
            },
            {
                "start": 56.0,
                "end": 61.7,
                "text": "Perfect. When do we need to have this completed?",
                "speaker": "Charlie"
            },
            {
                "start": 62.4,
                "end": 67.8,
                "text": "Let's target end of November for the performance fixes.",
                "speaker": "Alice"
            },
            {
                "start": 68.5,
                "end": 73.2,
                "text": "That sounds reasonable. I'll start working on this immediately.",
                "speaker": "Bob"
            }
        ],
        "full_text": "Good morning everyone, thanks for joining today's product planning meeting. What are our main priorities for the Q4 roadmap? I think we need to focus on the mobile app improvements and the new API integration. Agreed. The customer feedback has been clear about mobile performance issues. Should we prioritize the performance fixes or the new features first? I believe we must fix the performance issues first. New features won't matter if the app is slow. That's a good point. Let's make performance our top priority. Excellent. So our action items are: Bob will lead the performance optimization team. And Charlie will coordinate with the design team for the mobile improvements. Perfect. When do we need to have this completed? Let's target end of November for the performance fixes. That sounds reasonable. I'll start working on this immediately.",
        "processing_time": 2.3,
        "timestamp": 1698765432.0,
        "audio_duration": 73.2,
        "sample_rate": 16000
    }


def add_mock_sentiment_analysis(transcription: Dict[str, Any]) -> Dict[str, Any]:
    """Add mock sentiment analysis to demonstrate the features"""
    # Mock sentiment analysis for each segment
    sentiment_map = {
        0: {"sentiment": "positive", "confidence": 0.8},
        1: {"sentiment": "neutral", "confidence": 0.6},
        2: {"sentiment": "positive", "confidence": 0.7},
        3: {"sentiment": "positive", "confidence": 0.9},
        4: {"sentiment": "neutral", "confidence": 0.5},
        5: {"sentiment": "positive", "confidence": 0.8},
        6: {"sentiment": "positive", "confidence": 0.9},
        7: {"sentiment": "positive", "confidence": 0.8},
        8: {"sentiment": "positive", "confidence": 0.7},
        9: {"sentiment": "neutral", "confidence": 0.6},
        10: {"sentiment": "neutral", "confidence": 0.5},
        11: {"sentiment": "positive", "confidence": 0.8}
    }
    
    for i, segment in enumerate(transcription["segments"]):
        if i in sentiment_map:
            segment["sentiment_analysis"] = sentiment_map[i]
    
    return transcription


def demonstrate_summarization_features():
    """Demonstrate the conversation summarization and annotation features"""
    
    print("🎙️  OREJA CONVERSATION SUMMARIZATION & ANNOTATION DEMO")
    print("=" * 60)
    
    # Create enhanced processor
    print("\n1. Initializing Enhanced Transcription Processor...")
    processor = EnhancedTranscriptionProcessor(sentiment_model="vader")
    
    # Create sample transcription
    print("\n2. Creating sample meeting transcription...")
    transcription = create_sample_transcription()
    transcription = add_mock_sentiment_analysis(transcription)
    
    print(f"   📊 Sample conversation: {len(transcription['segments'])} segments, {transcription['audio_duration']} seconds")
    print(f"   👥 Speakers: {len(set(seg['speaker'] for seg in transcription['segments']))}")
    
    # Generate summaries
    print("\n3. Generating Conversation Summaries...")
    print("   🔄 Processing...")
    
    summary_types = ["overall", "by_speaker", "by_time", "key_points"]
    summaries = processor.generate_conversation_summary(transcription, summary_types)
    
    # Display Overall Summary
    print("\n   📋 OVERALL SUMMARY:")
    overall = summaries.get("overall", {})
    print(f"   Summary: {overall.get('summary', 'N/A')}")
    print(f"   Duration: {overall.get('key_metrics', {}).get('duration_minutes', 0)} minutes")
    print(f"   Speakers: {overall.get('key_metrics', {}).get('speaker_count', 0)}")
    print(f"   Dominant Sentiment: {overall.get('key_metrics', {}).get('dominant_sentiment', 'N/A')}")
    print(f"   Main Topics: {', '.join(overall.get('main_topics', []))}")
    
    # Display Speaker Summaries
    print("\n   👥 SPEAKER SUMMARIES:")
    by_speaker = summaries.get("by_speaker", {})
    for speaker, data in by_speaker.items():
        stats = data.get("participation_stats", {})
        print(f"   {speaker}:")
        print(f"      Summary: {data.get('summary', 'N/A')}")
        print(f"      Participation: {stats.get('participation_percentage', 0)}% ({stats.get('total_time_seconds', 0)}s)")
        print(f"      Sentiment: {stats.get('dominant_sentiment', 'N/A')}")
    
    # Display Time-based Summaries
    print("\n   ⏰ TIME-BASED SUMMARIES:")
    by_time = summaries.get("by_time", [])
    for interval in by_time[:3]:  # Show first 3 intervals
        time_range = interval.get("time_range", {})
        print(f"   {time_range.get('start_minutes', 0):.1f}-{time_range.get('end_minutes', 0):.1f} min:")
        print(f"      {interval.get('summary', 'N/A')}")
        print(f"      Speakers: {', '.join(interval.get('active_speakers', []))}")
    
    # Display Key Points
    print("\n   🎯 KEY POINTS:")
    key_points = summaries.get("key_points", [])
    for i, point in enumerate(key_points[:5], 1):  # Show top 5
        print(f"   {i}. [{point.get('type', 'general')}] {point.get('text', 'N/A')}")
        print(f"      Speaker: {point.get('speaker', 'N/A')} at {point.get('timestamp', 0):.1f}s")
    
    # Generate annotations
    print("\n4. Generating Conversation Annotations...")
    print("   🔄 Processing...")
    
    annotation_types = ["topics", "action_items", "questions_answers", "decisions", "emotional_moments"]
    annotations = processor.generate_conversation_annotations(transcription, annotation_types)
    
    # Display Topics
    print("\n   🏷️  IDENTIFIED TOPICS:")
    topics = annotations.get("topics", [])
    for topic in topics[:5]:  # Show top 5 topics
        print(f"   Topic: {topic.get('topic', 'N/A')} ({topic.get('mentions', 0)} mentions)")
        print(f"      Keywords: {', '.join(topic.get('keywords', []))}")
        print(f"      First mentioned: {topic.get('first_mentioned', 0):.1f}s")
    
    # Display Action Items
    print("\n   ✅ ACTION ITEMS:")
    action_items = annotations.get("action_items", [])
    for i, item in enumerate(action_items, 1):
        print(f"   {i}. {item.get('action', 'N/A')}")
        print(f"      Assigned to: {item.get('speaker', 'N/A')} at {item.get('timestamp', 0):.1f}s")
    
    # Display Q&A Pairs
    print("\n   ❓ QUESTION & ANSWER PAIRS:")
    qa_pairs = annotations.get("questions_answers", [])
    for i, pair in enumerate(qa_pairs[:3], 1):  # Show top 3
        print(f"   {i}. Q: {pair.get('question', 'N/A')} ({pair.get('question_speaker', 'N/A')})")
        print(f"      A: {pair.get('answer', 'N/A')} ({pair.get('answer_speaker', 'N/A')})")
        print(f"      Confidence: {pair.get('confidence', 0):.2f}")
    
    # Display Decisions
    print("\n   🎯 DECISIONS & CONCLUSIONS:")
    decisions = annotations.get("decisions", [])
    for i, decision in enumerate(decisions, 1):
        print(f"   {i}. {decision.get('decision', 'N/A')}")
        print(f"      By: {decision.get('speaker', 'N/A')} at {decision.get('timestamp', 0):.1f}s")
        print(f"      Sentiment: {decision.get('sentiment', 'N/A')}")
    
    # Display Emotional Moments
    print("\n   😊 EMOTIONAL MOMENTS:")
    emotions = annotations.get("emotional_moments", [])
    for i, moment in enumerate(emotions[:3], 1):  # Show top 3
        print(f"   {i}. {moment.get('emotional_indicator', 'N/A')}")
        print(f"      Text: {moment.get('text', 'N/A')[:80]}...")
        print(f"      Speaker: {moment.get('speaker', 'N/A')} at {moment.get('timestamp', 0):.1f}s")
    
    # Summary statistics
    print("\n5. Feature Summary:")
    print(f"   📊 Generated {len(summary_types)} summary types")
    print(f"   🏷️  Identified {len(topics)} topics")
    print(f"   ✅ Found {len(action_items)} action items")
    print(f"   ❓ Detected {len(qa_pairs)} Q&A pairs")
    print(f"   🎯 Extracted {len(decisions)} decisions")
    print(f"   😊 Found {len(emotions)} emotional moments")
    
    # Export example
    print("\n6. Export Options:")
    print("   💾 These features can be exported as:")
    print("      - Enhanced JSON with summaries and annotations")
    print("      - Executive summary report")
    print("      - Action items list")
    print("      - Meeting minutes format")
    
    print("\n" + "=" * 60)
    print("✨ Demo completed! These features turn raw transcription into actionable insights.")
    print("🎯 Benefits:")
    print("   • Quick understanding of long conversations")
    print("   • Automatic action item extraction")
    print("   • Speaker participation analysis") 
    print("   • Topic and decision tracking")
    print("   • Emotional tone monitoring")


def save_demo_results():
    """Save demonstration results to files"""
    print("\nSaving demo results...")
    
    processor = EnhancedTranscriptionProcessor(sentiment_model="vader")
    transcription = create_sample_transcription()
    transcription = add_mock_sentiment_analysis(transcription)
    
    # Generate full enhanced result
    enhanced_result = processor.process_with_summary_and_annotations(
        transcription, 
        None,  # No audio waveform for demo
        16000,
        include_summary=True,
        include_annotations=True
    )
    
    # Save to file
    output_file = "demo_enhanced_transcription_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_result, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 Enhanced result saved to: {output_file}")
    print(f"   📂 File size: {os.path.getsize(output_file)} bytes")


if __name__ == "__main__":
    try:
        demonstrate_summarization_features()
        
        # Ask user if they want to save results
        save_results = input("\nWould you like to save the demo results to a file? (y/n): ").lower().strip()
        if save_results in ['y', 'yes']:
            save_demo_results()
        
        print("\n🎉 Thank you for trying the Oreja enhanced features!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have installed the enhanced requirements:")
        print("pip install -r requirements_enhanced.txt")
        sys.exit(1) 