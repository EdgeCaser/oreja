using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Oreja.Models;

/// <summary>
/// Represents the result of audio transcription and diarization.
/// </summary>
public class TranscriptionResult
{
    [JsonPropertyName("segments")]
    public List<TranscriptionSegment> Segments { get; set; } = new();
    
    [JsonPropertyName("full_text")]
    public string FullText { get; set; } = string.Empty;
    
    [JsonPropertyName("processing_time")]
    public double ProcessingTime { get; set; }
    
    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Represents a single segment of transcribed audio with speaker information.
/// </summary>
public class TranscriptionSegment
{
    [JsonPropertyName("start")]
    public double Start { get; set; }
    
    [JsonPropertyName("end")]
    public double End { get; set; }
    
    [JsonPropertyName("text")]
    public string Text { get; set; } = string.Empty;
    
    [JsonPropertyName("speaker")]
    public string Speaker { get; set; } = string.Empty;
    
    [JsonPropertyName("confidence")]
    public float Confidence { get; set; }
    
    /// <summary>
    /// Duration of the segment in seconds.
    /// </summary>
    public double Duration => End - Start;
}

/// <summary>
/// Represents speaker information stored in the local database.
/// </summary>
public class Speaker
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string EmbeddingId { get; set; } = string.Empty;
    public byte[] Embedding { get; set; } = Array.Empty<byte>();
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime LastSeen { get; set; } = DateTime.UtcNow;
    public int OccurrenceCount { get; set; } = 1;
}

/// <summary>
/// Request model for renaming a speaker.
/// </summary>
public class RenameSpeakerRequest
{
    public string CurrentName { get; set; } = string.Empty;
    public string NewName { get; set; } = string.Empty;
}

/// <summary>
/// Response model for speaker operations.
/// </summary>
public class SpeakerResponse
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public Speaker? Speaker { get; set; }
}

/// <summary>
/// Model for audio device information.
/// </summary>
public class AudioDevice
{
    public int Index { get; set; }
    public string Name { get; set; } = string.Empty;
    public string DeviceId { get; set; } = string.Empty;
    public bool IsDefault { get; set; }
    public bool IsEnabled { get; set; } = true;
}

/// <summary>
/// Configuration model for transcription settings.
/// </summary>
public class TranscriptionSettings
{
    public bool EnableMicrophone { get; set; } = true;
    public bool EnableSystemAudio { get; set; } = true;
    public int SelectedMicrophoneIndex { get; set; } = 0;
    public float MicrophoneGain { get; set; } = 1.0f;
    public float SystemGain { get; set; } = 1.0f;
    public bool AutoSave { get; set; } = false;
    public string SaveDirectory { get; set; } = string.Empty;
    public int BufferSizeMs { get; set; } = 1000;
    public float ConfidenceThreshold { get; set; } = 0.5f;
} 