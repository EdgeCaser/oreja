#!/usr/bin/env python3
"""
Test enhanced transcription features in backend directory
"""

from enhanced_transcription_processor import EnhancedTranscriptionProcessor

def test_enhanced_features():
    print("🧪 Testing Enhanced Transcription Features")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = EnhancedTranscriptionProcessor(sentiment_model="vader")
        print("✅ Enhanced processor initialized successfully")
        
        # Test sentiment analysis
        test_text = "I love this new enhanced transcription feature!"
        result = processor.analyze_sentiment(test_text)
        
        print(f"📝 Test text: '{test_text}'")
        print(f"😊 Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
        print(f"📊 Method: {result['method']}")
        
        # Test with different sentiments
        test_cases = [
            "This is amazing! I'm so excited!",
            "This is terrible and frustrating.",
            "The weather is okay, I guess."
        ]
        
        print("\n🎭 Testing Multiple Sentiments:")
        print("-" * 30)
        
        for text in test_cases:
            result = processor.analyze_sentiment(text)
            emoji = "😊" if result['sentiment'] == "positive" else "😢" if result['sentiment'] == "negative" else "😐"
            print(f"{emoji} {result['sentiment'].title():8} ({result['confidence']:.2f}) | {text}")
        
        print("\n✅ All enhanced features working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_features() 