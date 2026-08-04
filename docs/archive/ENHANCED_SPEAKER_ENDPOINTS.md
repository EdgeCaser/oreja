# Enhanced Speaker Management API Endpoints

## Overview

The enhanced speaker management system provides a comprehensive solution to the architectural issues identified in the original speaker database. Here are all the new endpoints and their purposes:

## New API Endpoints

### 1. **Migration & Setup**

#### `POST /speakers/migrate_to_enhanced`
- **Purpose**: Migrate from legacy speaker database to enhanced v2 architecture
- **Parameters**: 
  - `dry_run` (bool): If true, analyze what would be migrated without making changes
- **Features**:
  - Detects and merges duplicate speakers with same names
  - Preserves all embeddings and metadata
  - Creates backup of legacy database
  - Provides detailed migration statistics

**Example Usage:**
```python
# Analyze migration first
response = requests.post("/speakers/migrate_to_enhanced?dry_run=true")

# Then perform actual migration
response = requests.post("/speakers/migrate_to_enhanced?dry_run=false")
```

#### `GET /speakers/system_status`
- **Purpose**: Get comprehensive status of both legacy and enhanced systems
- **Returns**: Speaker counts, migration status, recommendations
- **Use Case**: Check if migration is needed, system health monitoring

### 2. **Enhanced Statistics**

#### `GET /speakers/enhanced_stats`
- **Purpose**: Get detailed statistics from enhanced speaker database
- **Features**:
  - Enrollment status breakdown
  - Confidence distribution
  - Source type analysis (auto, enrolled, corrected)
  - Migration completion status

### 3. **Separated Save vs Feedback Operations**

#### `POST /speakers/save_transcription_enhanced`
- **Purpose**: **ONLY** save transcription file with corrections applied
- **Parameters**:
  - `transcription_data`: The transcription to save
  - `speaker_corrections`: Name corrections to apply
  - `output_file`: Optional output path
- **Key Feature**: **Does NOT send learning feedback** - addresses save/feedback confusion

#### `POST /speakers/enhanced_feedback`
- **Purpose**: **ONLY** send learning feedback to improve speaker recognition
- **Parameters**:
  - `corrections`: Speaker name corrections for learning
- **Features**:
  - Intelligent merging based on sample count
  - Immutable speaker IDs
  - Proper confidence recalculation
  - Fallback to legacy system if needed

### 4. **Enhanced Segment Splitting**

#### `POST /segments/split_with_audio_analysis`
- **Purpose**: Split segments with proper audio re-analysis and embedding extraction
- **Parameters**:
  - `audio_file`: Path to audio file
  - `original_segment`: Segment to split
  - `split_text_position`: Position in text (0.0-1.0) 
  - `first_speaker`: Speaker for first part
  - `second_speaker`: Speaker for second part
- **Features**:
  - **Actual audio loading and analysis**
  - **Optimal split point detection** (silence/energy-based)
  - **Separate embedding extraction** for each part
  - Confidence scoring for split quality
  - Validation and suggestions

#### `POST /segments/reprocess_embeddings`
- **Purpose**: Reprocess embeddings after speaker name corrections
- **Parameters**:
  - `audio_file`: Audio file path
  - `segments`: Segments to reprocess
  - `force_update`: Whether to update even if embeddings exist
- **Features**:
  - Re-extracts embeddings from actual audio
  - Improves speaker recognition based on corrections
  - Provides detailed processing results

## Key Architectural Improvements

### ✅ **Fixed Segment Splitting**
**Before**: Text-based timing estimation, wrong embeddings
```csharp
// OLD - BROKEN
var firstPartDuration = totalDuration * (firstPartLength / (double)totalLength);
```

**After**: Audio-based analysis with proper embedding extraction
```python
# NEW - FIXED
optimal_split_time = self._find_optimal_split_point(segment_waveform, sr, ...)
first_embedding = self._extract_embedding_from_waveform(first_audio, sr)
second_embedding = self._extract_embedding_from_waveform(second_audio, sr)
```

### ✅ **Immutable Speaker IDs**
**Before**: Speaker ID served as both ID and display name
**After**: UUID-based immutable IDs with mutable display names
```python
@dataclass
class SpeakerRecord:
    speaker_id: str  # UUID-based, NEVER changes  
    display_name: str  # User-friendly, mutable
```

### ✅ **Intelligent Merging**
**Before**: Could merge higher-confidence speaker into lower-confidence one
**After**: Speaker with more samples always becomes the target
```python
# Rule: Speaker with MORE samples becomes the target
if len(source_embeddings) > len(target_embeddings):
    source_speaker_id, target_speaker_id = target_speaker_id, source_speaker_id
```

### ✅ **Clear Operation Separation**
**Before**: Save and feedback happened simultaneously
**After**: Completely separate operations
- `/speakers/save_transcription_enhanced` - ONLY saves file
- `/speakers/enhanced_feedback` - ONLY sends learning feedback

## Migration Workflow

1. **Check Status**:
   ```bash
   GET /speakers/system_status
   ```

2. **Analyze Migration** (recommended):
   ```bash
   POST /speakers/migrate_to_enhanced?dry_run=true
   ```

3. **Perform Migration**:
   ```bash
   POST /speakers/migrate_to_enhanced?dry_run=false
   ```

4. **Verify Results**:
   ```bash
   GET /speakers/enhanced_stats
   ```

## Usage Examples

### Enhanced Segment Splitting
```python
# Split with proper audio analysis
response = requests.post("/segments/split_with_audio_analysis", json={
    "audio_file": "recording.wav",
    "original_segment": {"start": 10.0, "end": 15.0, "text": "Hello there how are you"},
    "split_text_position": 0.4,  # 40% through text
    "first_speaker": "Alice", 
    "second_speaker": "Bob"
})

# Response includes confidence and validation
{
    "status": "split_successful",
    "validation": {"confidence": 0.85},
    "embeddings_extracted": {"first_segment": true, "second_segment": true},
    "suggestions": ["High confidence split - good speaker separation detected"]
}
```

### Separate Save and Feedback
```python
# 1. Save transcription (no learning)
response = requests.post("/speakers/save_transcription_enhanced", json={
    "transcription_data": {...},
    "speaker_corrections": {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
})
# Returns: {"learning_feedback_sent": false}

# 2. Send feedback separately (learning only)  
response = requests.post("/speakers/enhanced_feedback", json={
    "corrections": {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
})
# Returns: enhanced learning results
```

## Benefits Summary

- ✅ **Accurate embeddings** from actual audio segments
- ✅ **Immutable speaker identity** prevents name resets  
- ✅ **Intelligent merging** keeps best training data
- ✅ **Clear operations** - user controls when learning happens
- ✅ **Automatic improvement** - system gets smarter with corrections
- ✅ **Fallback compatibility** - works alongside legacy system
- ✅ **Migration support** - smooth transition from legacy database 