#!/usr/bin/env python3
"""
Oreja Speaker Database Viewer
View all speakers and their recognition quality metrics
"""

import requests
import json
from datetime import datetime
import time

BACKEND_URL = "http://127.0.0.1:8000"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return health_data
        else:
            return None
    except Exception as e:
        print(f"❌ Backend not reachable: {e}")
        return None

def get_speaker_stats():
    """Get all speaker statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/speakers", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get speaker stats: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting speaker stats: {e}")
        return None

def calculate_recognition_quality(speaker):
    """Calculate recognition quality score based on available data"""
    embedding_count = speaker.get('embedding_count', 0)
    avg_confidence = speaker.get('avg_confidence', 0.0)
    
    # Quality factors:
    # 1. Number of embeddings (more = better recognition)
    # 2. Average confidence (higher = better quality recordings)
    
    if embedding_count == 0:
        return 0.0, "No data"
    
    # Normalize embedding count (1-10 embeddings = 0.1-1.0 score)
    embedding_score = min(embedding_count / 10.0, 1.0)
    
    # Confidence score (0.0-1.0)
    confidence_score = avg_confidence
    
    # Combined score (weighted average)
    quality_score = (embedding_score * 0.4) + (confidence_score * 0.6)
    
    # Quality ratings
    if quality_score >= 0.8:
        rating = "Excellent"
    elif quality_score >= 0.6:
        rating = "Good"
    elif quality_score >= 0.4:
        rating = "Fair"
    elif quality_score >= 0.2:
        rating = "Poor"
    else:
        rating = "Very Poor"
    
    return quality_score, rating

def format_date(date_str):
    """Format ISO date string for display"""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return date_str

def print_speaker_database():
    """Print formatted speaker database"""
    print("=" * 80)
    print("🎙️  OREJA SPEAKER DATABASE")
    print("=" * 80)
    
    # Check backend health
    print("📡 Checking backend connection...")
    health = check_backend_health()
    
    if not health:
        print("❌ Backend not available. Make sure the server is running:")
        print("   cd backend && python server.py")
        return
    
    print(f"✅ Backend connected - Device: {health.get('device', 'unknown')}")
    models = health.get('models', {})
    print(f"   Models: Whisper={models.get('whisper', False)}, "
          f"Embeddings={models.get('speaker_embeddings', False)}")
    print()
    
    # Get speaker stats
    print("🔍 Fetching speaker statistics...")
    stats = get_speaker_stats()
    
    if not stats:
        print("❌ Could not retrieve speaker statistics")
        return
    
    total_speakers = stats.get('total_speakers', 0)
    speakers = stats.get('speakers', [])
    
    print(f"📊 Total speakers in database: {total_speakers}")
    print()
    
    if total_speakers == 0:
        print("🤷 No speakers found in database.")
        print("💡 Start recording to automatically detect and learn speakers!")
        return
    
    # Table header
    print("📋 SPEAKER RECOGNITION QUALITY REPORT")
    print("-" * 80)
    print(f"{'Name':<20} {'ID':<15} {'Quality':<12} {'Embeddings':<10} {'Confidence':<10} {'Last Seen':<16}")
    print("-" * 80)
    
    # Sort speakers by quality score (best first)
    speakers_with_quality = []
    for speaker in speakers:
        quality_score, quality_rating = calculate_recognition_quality(speaker)
        speakers_with_quality.append((speaker, quality_score, quality_rating))
    
    speakers_with_quality.sort(key=lambda x: x[1], reverse=True)
    
    # Print each speaker
    for speaker, quality_score, quality_rating in speakers_with_quality:
        name = speaker.get('name', 'Unknown')[:19]
        speaker_id = speaker.get('id', 'N/A')[:14]
        embedding_count = speaker.get('embedding_count', 0)
        avg_confidence = speaker.get('avg_confidence', 0.0)
        last_seen = format_date(speaker.get('last_seen', ''))[:15]
        
        # Color coding for quality
        if quality_score >= 0.8:
            quality_display = f"🟢 {quality_rating}"
        elif quality_score >= 0.6:
            quality_display = f"🟡 {quality_rating}"
        elif quality_score >= 0.4:
            quality_display = f"🟠 {quality_rating}"
        else:
            quality_display = f"🔴 {quality_rating}"
        
        print(f"{name:<20} {speaker_id:<15} {quality_display:<12} "
              f"{embedding_count:<10} {avg_confidence:<10.3f} {last_seen:<16}")
    
    print("-" * 80)
    print()
    
    # Quality summary
    excellent = sum(1 for _, score, _ in speakers_with_quality if score >= 0.8)
    good = sum(1 for _, score, _ in speakers_with_quality if 0.6 <= score < 0.8)
    fair = sum(1 for _, score, _ in speakers_with_quality if 0.4 <= score < 0.6)
    poor = sum(1 for _, score, _ in speakers_with_quality if score < 0.4)
    
    print("📈 RECOGNITION QUALITY SUMMARY:")
    print(f"   🟢 Excellent: {excellent} speakers")
    print(f"   🟡 Good:      {good} speakers") 
    print(f"   🟠 Fair:      {fair} speakers")
    print(f"   🔴 Poor:      {poor} speakers")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if poor > 0:
        print(f"   • {poor} speakers need more training data")
        print("   • Record longer sessions with these speakers")
        print("   • Ensure clear audio quality")
    
    if excellent + good >= total_speakers * 0.8:
        print("   • 🎉 Great speaker recognition quality!")
        print("   • System should reliably identify most speakers")
    
    print()
    print("🔄 To refresh this view, run: python view_speakers.py")
    print("=" * 80)

if __name__ == "__main__":
    print_speaker_database() 