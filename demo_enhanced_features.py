#!/usr/bin/env python3
"""
Demo: Enhanced Transcription Features
Demonstrates sentiment analysis and audio features capabilities
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

def create_demo_data():
    """Create realistic demo transcription data"""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 4.2,
                "text": "Hi everyone! I'm really excited to discuss our quarterly results today.",
                "speaker": "CEO"
            },
            {
                "start": 4.5,
                "end": 8.1,
                "text": "Thank you Sarah. I have to say, I'm quite worried about our customer retention numbers.",
                "speaker": "Marketing_Director"
            },
            {
                "start": 8.5,
                "end": 12.0,
                "text": "That's a valid concern, but I think we're overreacting. The data shows improvement.",
                "speaker": "Data_Analyst"
            },
            {
                "start": 12.2,
                "end": 15.8,
                "text": "I disagree completely! We need immediate action or we'll lose more customers.",
                "speaker": "Sales_Manager"
            },
            {
                "start": 16.0,
                "end": 19.5,
                "text": "Let's stay calm and look at this objectively. What do the numbers actually tell us?",
                "speaker": "CEO"
            },
            {
                "start": 20.0,
                "end": 24.2,
                "text": "You're absolutely right. I apologize for getting heated. Let me show you the analysis.",
                "speaker": "Sales_Manager"
            },
            {
                "start": 24.5,
                "end": 28.0,
                "text": "Great! I love seeing the team work together like this. Very encouraging!",
                "speaker": "Marketing_Director"
            }
        ],
        "full_text": "Complete meeting transcript...",
        "processing_time": 45.2,
        "timestamp": datetime.now().timestamp()
    }

def create_demo_audio(duration_seconds=30, sample_rate=16000):
    """Create synthetic audio with varying characteristics"""
    t = np.linspace(0, duration_seconds, duration_seconds * sample_rate)
    
    # Create segments with different characteristics
    audio = np.zeros_like(t)
    
    # Segment 1: Excited speech (higher frequency, more energy)
    mask1 = (t >= 0) & (t < 4.2)
    audio[mask1] = 0.3 * np.sin(2 * np.pi * 200 * t[mask1]) + 0.1 * np.random.normal(0, 0.1, mask1.sum())
    
    # Segment 2: Worried speech (lower frequency, moderate energy)
    mask2 = (t >= 4.5) & (t < 8.1)
    audio[mask2] = 0.2 * np.sin(2 * np.pi * 150 * t[mask2]) + 0.05 * np.random.normal(0, 0.1, mask2.sum())
    
    # Segment 3: Confident speech (steady frequency, good energy)
    mask3 = (t >= 8.5) & (t < 12.0)
    audio[mask3] = 0.25 * np.sin(2 * np.pi * 180 * t[mask3]) + 0.02 * np.random.normal(0, 0.1, mask3.sum())
    
    # Segment 4: Angry speech (higher frequency, high energy, irregular)
    mask4 = (t >= 12.2) & (t < 15.8)
    angry_freq = 220 + 30 * np.sin(10 * t[mask4])  # Varying frequency
    audio[mask4] = 0.4 * np.sin(2 * np.pi * angry_freq * t[mask4]) + 0.15 * np.random.normal(0, 0.2, mask4.sum())
    
    # Segment 5: Calm speech (steady, moderate)
    mask5 = (t >= 16.0) & (t < 19.5)
    audio[mask5] = 0.22 * np.sin(2 * np.pi * 170 * t[mask5]) + 0.03 * np.random.normal(0, 0.1, mask5.sum())
    
    # Segment 6: Apologetic speech (softer, lower energy)
    mask6 = (t >= 20.0) & (t < 24.2)
    audio[mask6] = 0.15 * np.sin(2 * np.pi * 140 * t[mask6]) + 0.02 * np.random.normal(0, 0.05, mask6.sum())
    
    # Segment 7: Happy speech (bright, energetic)
    mask7 = (t >= 24.5) & (t < 28.0)
    audio[mask7] = 0.3 * np.sin(2 * np.pi * 210 * t[mask7]) + 0.08 * np.random.normal(0, 0.1, mask7.sum())
    
    return torch.tensor(audio).unsqueeze(0), sample_rate

def run_sentiment_analysis_demo():
    """Demonstrate different sentiment analysis models"""
    print("🎭 SENTIMENT ANALYSIS DEMO")
    print("=" * 50)
    
    try:
        from enhanced_transcription_processor import EnhancedTranscriptionProcessor
        
        # Test different models
        models = ["vader", "textblob", "transformer"]
        test_texts = [
            "I'm absolutely thrilled about this opportunity!",
            "This is completely unacceptable and frustrating.",
            "The weather is okay, I guess.",
            "I love this product! It's amazing and works perfectly!",
            "I hate waiting in long lines. It's so annoying.",
            "The presentation was informative and well-structured."
        ]
        
        for model in models:
            print(f"\n📊 Testing {model.upper()} Model:")
            print("-" * 30)
            
            try:
                processor = EnhancedTranscriptionProcessor(sentiment_model=model)
                
                for text in test_texts:
                    result = processor.analyze_sentiment(text)
                    sentiment = result["sentiment"]
                    confidence = result["confidence"]
                    
                    # Emoji for sentiment
                    emoji = "😊" if sentiment == "positive" else "😢" if sentiment == "negative" else "😐"
                    
                    print(f"{emoji} {sentiment.title():8} ({confidence:.2f}) | {text}")
                    
            except Exception as e:
                print(f"❌ {model} model failed: {e}")
        
    except ImportError:
        print("❌ Enhanced features not available. Run: pip install -r requirements_enhanced.txt")

def run_audio_features_demo():
    """Demonstrate audio feature extraction"""
    print("\n\n🎵 AUDIO FEATURES DEMO")
    print("=" * 50)
    
    try:
        from enhanced_transcription_processor import EnhancedTranscriptionProcessor
        
        processor = EnhancedTranscriptionProcessor()
        
        # Create demo audio with different characteristics
        print("\n🔊 Analyzing Audio Characteristics...")
        
        # Different audio scenarios
        scenarios = [
            ("Excited Speaker", 0.4, 220, "high energy, fast speech"),
            ("Calm Speaker", 0.15, 160, "low energy, steady speech"),
            ("Stressed Speaker", 0.35, 200, "irregular energy, tense speech"),
            ("Quiet Speaker", 0.08, 140, "very low energy, slow speech")
        ]
        
        for name, amplitude, frequency, description in scenarios:
            print(f"\n🎤 {name} ({description}):")
            print("-" * 40)
            
            # Generate test audio
            duration = 3.0
            sample_rate = 16000
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Add some variation for realism
            freq_variation = frequency + 20 * np.sin(5 * t)
            audio = amplitude * np.sin(2 * np.pi * freq_variation * t)
            
            # Add some noise and energy variation
            noise = 0.02 * np.random.normal(0, 1, len(audio))
            energy_envelope = 1 + 0.3 * np.sin(2 * t)  # Energy variation
            audio = (audio + noise) * energy_envelope
            
            waveform = torch.tensor(audio).unsqueeze(0)
            
            # Extract features
            features = processor.extract_audio_features(waveform, sample_rate, 0.0, duration)
            
            print(f"   Volume Level: {features['energy']['volume_level']}")
            print(f"   Speaking Rate: {features['speaking_rate']['rate_category']}")
            print(f"   Voice Characteristics: {features['spectral']['voice_characteristics']}")
            print(f"   Stress Level: {features['stress']['stress_level']}")
            print(f"   Speaking Style: {features['pauses']['speaking_style']}")
            
    except ImportError:
        print("❌ Enhanced features not available. Run: pip install -r requirements_enhanced.txt")

def run_full_enhanced_demo():
    """Demonstrate complete enhanced transcription"""
    print("\n\n🚀 FULL ENHANCED TRANSCRIPTION DEMO")
    print("=" * 50)
    
    try:
        from enhanced_transcription_processor import EnhancedTranscriptionProcessor
        
        processor = EnhancedTranscriptionProcessor(sentiment_model="vader")
        
        # Create demo data
        transcription_data = create_demo_data()
        demo_audio, sample_rate = create_demo_audio()
        
        print("\n📝 Original Transcription:")
        print("-" * 30)
        for i, segment in enumerate(transcription_data["segments"], 1):
            print(f"{i}. [{segment['start']:.1f}s] {segment['speaker']}: {segment['text']}")
        
        # Process with enhancements
        enhanced_result = processor.process_enhanced_transcription(
            transcription_data, demo_audio, sample_rate
        )
        
        print("\n✨ Enhanced Analysis:")
        print("-" * 30)
        
        for i, segment in enumerate(enhanced_result["segments"], 1):
            speaker = segment["speaker"]
            text = segment["text"]
            
            # Sentiment analysis
            if "sentiment_analysis" in segment:
                sentiment_info = segment["sentiment_analysis"]
                sentiment = sentiment_info["sentiment"]
                confidence = sentiment_info["confidence"]
                emoji = "😊" if sentiment == "positive" else "😢" if sentiment == "negative" else "😐"
                
                print(f"\n{i}. 👤 {speaker}")
                print(f"   💬 \"{text}\"")
                print(f"   {emoji} Sentiment: {sentiment.title()} (confidence: {confidence:.2f})")
                
                # Audio features (if available)
                if "audio_features" in segment:
                    features = segment["audio_features"]
                    print(f"   🔊 Volume: {features['energy']['volume_level']}")
                    print(f"   ⚡ Energy: {features['speaking_rate']['rate_category']}")
                    print(f"   🎵 Voice: {features['spectral']['voice_characteristics']}")
                    print(f"   😰 Stress: {features['stress']['stress_level']}")
        
        # Conversation analysis
        if "conversation_analysis" in enhanced_result:
            conv_analysis = enhanced_result["conversation_analysis"]
            
            print(f"\n🗣️ CONVERSATION ANALYSIS:")
            print("-" * 30)
            
            metrics = conv_analysis["conversation_metrics"]
            print(f"Overall Sentiment: {metrics['overall_sentiment_label'].title()}")
            print(f"Duration: {metrics['total_duration']:.1f} seconds")
            print(f"Speakers: {metrics['total_speakers']}")
            print(f"Segments: {metrics['total_segments']}")
            
            print(f"\n👥 SPEAKER STATISTICS:")
            print("-" * 30)
            
            for speaker, stats in conv_analysis["speaker_statistics"].items():
                print(f"{speaker}:")
                print(f"  Speaking time: {stats['total_time']:.1f}s")
                print(f"  Segments: {stats['segment_count']}")
                print(f"  Avg sentiment: {stats['avg_sentiment']:.2f}")
                if stats['interruption_count'] > 0:
                    print(f"  Interruptions: {stats['interruption_count']}")
        
        # Save enhanced results
        output_file = "demo_enhanced_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Enhanced results saved to: {output_file}")
        
    except ImportError:
        print("❌ Enhanced features not available. Run: pip install -r requirements_enhanced.txt")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def show_integration_example():
    """Show how to integrate with existing code"""
    print("\n\n🔧 INTEGRATION EXAMPLE")
    print("=" * 50)
    
    code_example = '''
# Example: Adding enhanced features to your existing transcription

from enhanced_server_integration import EnhancedTranscriptionService

# Initialize enhanced service
enhanced_service = EnhancedTranscriptionService(
    sentiment_model="vader",  # or "textblob", "transformer"
    enable_audio_features=True
)

# After getting your normal transcription result:
def process_audio_file(audio_file_path):
    # Your existing transcription code
    transcription_result = your_existing_transcription_function(audio_file_path)
    waveform, sample_rate = load_audio(audio_file_path)
    
    # Add enhanced features
    enhanced_result = enhanced_service.enhance_transcription_result(
        transcription_result, waveform, sample_rate
    )
    
    # Now you have:
    # - Original transcription + speaker identification
    # - Sentiment analysis for each segment
    # - Audio characteristics (volume, stress, speaking rate)
    # - Conversation dynamics (participation, interruptions)
    
    return enhanced_result

# Use in your FastAPI server:
from enhanced_server_integration import add_enhanced_endpoints

app = FastAPI()
add_enhanced_endpoints(app)  # Adds /transcribe_enhanced endpoint
'''
    
    print(code_example)

def main():
    """Run all demos"""
    print("🎙️ OREJA ENHANCED TRANSCRIPTION FEATURES DEMO")
    print("=" * 60)
    print("This demo showcases advanced sentiment analysis and audio features")
    print("that can be added to your transcription pipeline.")
    print()
    
    # Check if enhanced features are available
    try:
        from enhanced_transcription_processor import EnhancedTranscriptionProcessor
        print("✅ Enhanced features are available!")
    except ImportError:
        print("❌ Enhanced features not available.")
        print("📦 Install with: pip install -r requirements_enhanced.txt")
        print("\nRequired packages:")
        print("  - vaderSentiment (sentiment analysis)")
        print("  - textblob (alternative sentiment)")
        print("  - transformers (advanced sentiment)")
        print("  - librosa (audio features)")
        return
    
    # Run demos
    run_sentiment_analysis_demo()
    run_audio_features_demo() 
    run_full_enhanced_demo()
    show_integration_example()
    
    print("\n\n🎯 NEXT STEPS")
    print("=" * 50)
    print("1. Install enhanced requirements: pip install -r requirements_enhanced.txt")
    print("2. Try the demo: python demo_enhanced_features.py")
    print("3. Integrate with your server using enhanced_server_integration.py")
    print("4. Access new endpoints: /transcribe_enhanced, /analyze_sentiment")
    print("5. Enjoy sentiment analysis and audio insights! 🚀")

if __name__ == "__main__":
    main() 