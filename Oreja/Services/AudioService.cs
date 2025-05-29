using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Oreja.Models;

namespace Oreja.Services;

/// <summary>
/// Service for capturing microphone and system audio using NAudio.
/// Processes audio in memory and sends to Python backend for transcription.
/// </summary>
public class AudioService : IAudioService, IDisposable
{
    private readonly ILogger<AudioService> _logger;
    private readonly HttpClient _httpClient;
    private readonly object _lockObject = new();
    private DateTime _lastTranscriptionRequest = DateTime.MinValue;
    private const int TRANSCRIPTION_COOLDOWN_MS = 5000; // Wait 5 seconds between transcription requests
    
    private WasapiCapture? _microphoneCapture;
    private WasapiLoopbackCapture? _systemCapture;
    private MixingSampleProvider? _mixer;
    private BufferedWaveProvider? _microphoneBuffer;
    private BufferedWaveProvider? _systemBuffer;
    
    private bool _isCapturing;
    private bool _isMonitoring;
    private const int SAMPLE_RATE = 16000;
    private const int BUFFER_DURATION_MS = 1000;
    private const string BACKEND_URL = "http://127.0.0.1:8000";

    public event EventHandler<AudioLevelEventArgs>? MicrophoneLevelChanged;
    public event EventHandler<AudioLevelEventArgs>? SystemLevelChanged;
    public event EventHandler<TranscriptionEventArgs>? TranscriptionReceived;

    public AudioService(ILogger<AudioService> logger, HttpClient httpClient)
    {
        _logger = logger;
        _httpClient = httpClient;
        _httpClient.BaseAddress = new Uri(BACKEND_URL);
    }

    /// <summary>
    /// Starts audio capture from specified microphone and system audio devices.
    /// </summary>
    /// <param name="microphoneDeviceIndex">Index of microphone device</param>
    /// <param name="enableSystemCapture">Whether to capture system audio</param>
    /// <param name="systemAudioDeviceIndex">Index of system audio device (optional)</param>
    /// <param name="microphoneSensitivity">Microphone sensitivity multiplier (0.1 to 2.0)</param>
    public async Task StartCaptureAsync(int microphoneDeviceIndex, bool enableSystemCapture = true, int systemAudioDeviceIndex = -1, float microphoneSensitivity = 1.0f)
    {
        try
        {
            lock (_lockObject)
            {
                if (_isCapturing)
                    return;

                InitializeMicrophoneCapture(microphoneDeviceIndex, microphoneSensitivity);
                
                if (enableSystemCapture)
                    InitializeSystemCapture(systemAudioDeviceIndex);

                InitializeMixer();
                
                _microphoneCapture?.StartRecording();
                _systemCapture?.StartRecording();
                
                _isCapturing = true;
            }

            _logger.LogInformation("Audio capture started successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start audio capture");
            throw;
        }
    }

    /// <summary>
    /// Stops audio capture and cleans up resources.
    /// </summary>
    public async Task StopCaptureAsync()
    {
        try
        {
            lock (_lockObject)
            {
                if (!_isCapturing)
                    return;

                _microphoneCapture?.StopRecording();
                _systemCapture?.StopRecording();
                
                _isCapturing = false;
            }

            _logger.LogInformation("Audio capture stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping audio capture");
            throw;
        }
    }

    /// <summary>
    /// Starts audio monitoring for volume levels only (no transcription).
    /// </summary>
    /// <param name="microphoneDeviceIndex">Index of microphone device</param>
    /// <param name="enableSystemCapture">Whether to monitor system audio</param>
    /// <param name="systemAudioDeviceIndex">Index of system audio device (optional)</param>
    /// <param name="microphoneSensitivity">Microphone sensitivity multiplier (0.1 to 2.0)</param>
    public async Task StartMonitoringAsync(int microphoneDeviceIndex, bool enableSystemCapture = true, int systemAudioDeviceIndex = -1, float microphoneSensitivity = 1.0f)
    {
        try
        {
            _logger.LogInformation("StartMonitoringAsync called with micIndex={MicIndex}, enableSystem={EnableSystem}, systemIndex={SystemIndex}, sensitivity={Sensitivity}", 
                microphoneDeviceIndex, enableSystemCapture, systemAudioDeviceIndex, microphoneSensitivity);

            lock (_lockObject)
            {
                if (_isMonitoring || _isCapturing)
                {
                    _logger.LogWarning("Audio monitoring already active - isMonitoring={IsMonitoring}, isCapturing={IsCapturing}", _isMonitoring, _isCapturing);
                    return;
                }

                try
                {
                    _logger.LogInformation("Initializing microphone monitoring...");
                    InitializeMicrophoneMonitoring(microphoneDeviceIndex, microphoneSensitivity);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to initialize microphone monitoring");
                    throw new InvalidOperationException($"Failed to initialize microphone (device {microphoneDeviceIndex}): {ex.Message}", ex);
                }
                
                if (enableSystemCapture)
                {
                    try
                    {
                        _logger.LogInformation("Initializing system audio monitoring...");
                        InitializeSystemMonitoring(systemAudioDeviceIndex);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Failed to initialize system audio monitoring");
                        // Don't throw here - continue with just microphone monitoring
                        _logger.LogWarning("Continuing with microphone monitoring only");
                    }
                }
                
                try
                {
                    _logger.LogInformation("Starting audio capture devices...");
                    _microphoneCapture?.StartRecording();
                    _logger.LogInformation("Microphone capture started");
                    
                    if (enableSystemCapture && _systemCapture != null)
                    {
                        _systemCapture?.StartRecording();
                        _logger.LogInformation("System audio capture started");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to start audio recording");
                    throw new InvalidOperationException($"Failed to start audio recording: {ex.Message}", ex);
                }
                
                _isMonitoring = true;
                _logger.LogInformation("Audio monitoring started successfully");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start audio monitoring");
            // Clean up any partially initialized resources
            try
            {
                _microphoneCapture?.StopRecording();
                _systemCapture?.StopRecording();
                _microphoneCapture?.Dispose();
                _systemCapture?.Dispose();
                _microphoneCapture = null;
                _systemCapture = null;
            }
            catch (Exception cleanupEx)
            {
                _logger.LogError(cleanupEx, "Error during cleanup after monitoring failure");
            }
            throw;
        }
    }

    /// <summary>
    /// Stops audio monitoring.
    /// </summary>
    public async Task StopMonitoringAsync()
    {
        try
        {
            lock (_lockObject)
            {
                if (!_isMonitoring)
                    return;

                _microphoneCapture?.StopRecording();
                _systemCapture?.StopRecording();
                
                _isMonitoring = false;
            }

            _logger.LogInformation("Audio monitoring stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping audio monitoring");
            throw;
        }
    }

    private void InitializeMicrophoneCapture(int deviceIndex, float sensitivity)
    {
        var devices = new MMDeviceEnumerator().EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active);
        if (deviceIndex >= devices.Count)
            throw new ArgumentException("Invalid microphone device index");

        var device = devices[deviceIndex];
        _microphoneCapture = new WasapiCapture(device);
        _microphoneCapture.WaveFormat = new WaveFormat(SAMPLE_RATE, 1);
        
        _microphoneBuffer = new BufferedWaveProvider(_microphoneCapture.WaveFormat)
        {
            BufferDuration = TimeSpan.FromMilliseconds(BUFFER_DURATION_MS * 10),
            DiscardOnBufferOverflow = true
        };

        _microphoneCapture.DataAvailable += (s, e) =>
        {
            _microphoneBuffer.AddSamples(e.Buffer, 0, e.BytesRecorded);
            
            // Calculate and report audio level
            var level = CalculateAudioLevel(e.Buffer, e.BytesRecorded) * sensitivity;
            MicrophoneLevelChanged?.Invoke(this, new AudioLevelEventArgs(level));
            
            // Process audio buffer for transcription asynchronously
            _ = Task.Run(async () =>
            {
                try
                {
                    await ProcessAudioBufferAsync(e.Buffer, e.BytesRecorded, "microphone");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing microphone audio buffer");
                }
            });
        };
    }

    private void InitializeSystemCapture(int deviceIndex)
    {
        if (deviceIndex >= 0)
        {
            // Use specific system audio device
            var devices = new MMDeviceEnumerator().EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active);
            if (deviceIndex < devices.Count)
            {
                var device = devices[deviceIndex];
                _systemCapture = new WasapiLoopbackCapture(device);
            }
            else
            {
                _systemCapture = new WasapiLoopbackCapture(); // Fallback to default
            }
        }
        else
        {
            _systemCapture = new WasapiLoopbackCapture(); // Use default system audio
        }
        
        _systemBuffer = new BufferedWaveProvider(_systemCapture.WaveFormat)
        {
            BufferDuration = TimeSpan.FromMilliseconds(BUFFER_DURATION_MS * 10),
            DiscardOnBufferOverflow = true
        };

        _systemCapture.DataAvailable += (s, e) =>
        {
            _systemBuffer.AddSamples(e.Buffer, 0, e.BytesRecorded);
            
            // Calculate and report audio level
            var level = CalculateAudioLevel(e.Buffer, e.BytesRecorded);
            SystemLevelChanged?.Invoke(this, new AudioLevelEventArgs(level));
            
            // Process audio buffer for transcription asynchronously
            _ = Task.Run(async () =>
            {
                try
                {
                    await ProcessAudioBufferAsync(e.Buffer, e.BytesRecorded, "system");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing system audio buffer");
                }
            });
        };
    }

    private void InitializeMixer()
    {
        if (_microphoneBuffer == null) return;

        var micSample = _microphoneBuffer.ToSampleProvider();
        _mixer = new MixingSampleProvider(micSample.WaveFormat);
        _mixer.AddMixerInput(micSample);
        
        if (_systemBuffer != null)
        {
            var systemSample = _systemBuffer.ToSampleProvider();
            if (systemSample.WaveFormat.SampleRate == micSample.WaveFormat.SampleRate)
            {
                _mixer.AddMixerInput(systemSample);
            }
        }
    }

    private void InitializeMicrophoneMonitoring(int deviceIndex, float sensitivity)
    {
        _logger.LogInformation("InitializeMicrophoneMonitoring - deviceIndex={DeviceIndex}, sensitivity={Sensitivity}", deviceIndex, sensitivity);
        
        var devices = new MMDeviceEnumerator().EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active);
        _logger.LogInformation("Found {DeviceCount} capture devices", devices.Count);
        
        if (deviceIndex >= devices.Count)
        {
            _logger.LogError("Invalid microphone device index {DeviceIndex} - only {DeviceCount} devices available", deviceIndex, devices.Count);
            throw new ArgumentException("Invalid microphone device index");
        }

        var device = devices[deviceIndex];
        _logger.LogInformation("Using microphone device: {DeviceName} (ID: {DeviceId})", device.FriendlyName, device.ID);
        
        // Test device accessibility
        try
        {
            _logger.LogInformation("Testing device accessibility...");
            var testFormat = device.AudioClient.MixFormat;
            _logger.LogInformation("Device supports format: {SampleRate}Hz, {Channels} channels, {BitsPerSample} bits", 
                testFormat.SampleRate, testFormat.Channels, testFormat.BitsPerSample);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Device accessibility test failed");
            throw new InvalidOperationException($"Cannot access microphone device '{device.FriendlyName}': {ex.Message}", ex);
        }
        
        try
        {
            _microphoneCapture = new WasapiCapture(device);
            _microphoneCapture.WaveFormat = new WaveFormat(SAMPLE_RATE, 1);
            _logger.LogInformation("Microphone capture initialized with format: {SampleRate}Hz, {Channels} channel(s)", SAMPLE_RATE, 1);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create WasapiCapture");
            throw new InvalidOperationException($"Failed to create audio capture for device '{device.FriendlyName}': {ex.Message}", ex);
        }
        
        // For monitoring, we only need level calculation, no buffering for transcription
        _microphoneCapture.DataAvailable += (s, e) =>
        {
            try
            {
                // Calculate and report audio level only
                var level = CalculateAudioLevel(e.Buffer, e.BytesRecorded) * sensitivity;
                _logger.LogDebug("Microphone level: {Level:F3} (raw bytes: {Bytes})", level, e.BytesRecorded);
                MicrophoneLevelChanged?.Invoke(this, new AudioLevelEventArgs(level));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in microphone DataAvailable event");
            }
        };
        
        _logger.LogInformation("Microphone monitoring event handler attached");
    }

    private void InitializeSystemMonitoring(int deviceIndex)
    {
        _logger.LogInformation("InitializeSystemMonitoring - deviceIndex={DeviceIndex}", deviceIndex);
        
        if (deviceIndex >= 0)
        {
            // Use specific system audio device
            var devices = new MMDeviceEnumerator().EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active);
            _logger.LogInformation("Found {DeviceCount} render devices", devices.Count);
            
            if (deviceIndex < devices.Count)
            {
                var device = devices[deviceIndex];
                _logger.LogInformation("Using specific system audio device: {DeviceName} (ID: {DeviceId})", device.FriendlyName, device.ID);
                _systemCapture = new WasapiLoopbackCapture(device);
            }
            else
            {
                _logger.LogWarning("Invalid system audio device index {DeviceIndex} - falling back to default", deviceIndex);
                _systemCapture = new WasapiLoopbackCapture(); // Fallback to default
            }
        }
        else
        {
            _logger.LogInformation("Using default system audio device");
            _systemCapture = new WasapiLoopbackCapture(); // Use default system audio
        }
        
        _logger.LogInformation("System capture initialized");
        
        // For monitoring, we only need level calculation, no buffering for transcription
        _systemCapture.DataAvailable += (s, e) =>
        {
            try
            {
                // Calculate and report audio level only
                var level = CalculateAudioLevel(e.Buffer, e.BytesRecorded);
                _logger.LogDebug("System level: {Level:F3} (raw bytes: {Bytes})", level, e.BytesRecorded);
                SystemLevelChanged?.Invoke(this, new AudioLevelEventArgs(level));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in system audio DataAvailable event");
            }
        };
        
        _logger.LogInformation("System audio monitoring event handler attached");
    }

    private async Task ProcessAudioBufferAsync(byte[] buffer, int bytesRecorded, string source)
    {
        try
        {
            // Only process if we have sufficient audio data (at least 3 seconds)
            if (bytesRecorded < SAMPLE_RATE * 6) // Less than 3 seconds of audio at 16-bit (2 bytes per sample)
                return;

            // Create a copy of the buffer to avoid memory issues
            var audioData = new byte[bytesRecorded];
            Array.Copy(buffer, audioData, bytesRecorded);

            // Send to Python backend for transcription
            await SendAudioForTranscriptionAsync(audioData);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing audio buffer from {Source}", source);
        }
    }

    private async Task SendAudioForTranscriptionAsync(byte[] audioData)
    {
        try
        {
            // Rate limiting - don't send requests too frequently
            var timeSinceLastRequest = DateTime.Now - _lastTranscriptionRequest;
            if (timeSinceLastRequest.TotalMilliseconds < TRANSCRIPTION_COOLDOWN_MS)
            {
                _logger.LogDebug("Skipping transcription request - rate limited (last request {Ms}ms ago)", 
                    timeSinceLastRequest.TotalMilliseconds);
                return;
            }

            _lastTranscriptionRequest = DateTime.Now;

            // Create a proper WAV file from the raw audio data
            using var memoryStream = new MemoryStream();
            using var waveFileWriter = new WaveFileWriter(memoryStream, new WaveFormat(SAMPLE_RATE, 16, 1));
            
            // Write the audio data to create a proper WAV file
            waveFileWriter.Write(audioData, 0, audioData.Length);
            waveFileWriter.Flush();
            
            // Get the WAV file bytes
            var wavFileBytes = memoryStream.ToArray();

            using var content = new MultipartFormDataContent();
            using var audioContent = new ByteArrayContent(wavFileBytes);
            audioContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("audio/wav");
            content.Add(audioContent, "audio", "audio.wav");

            _logger.LogInformation("Sending {Bytes} bytes of audio for transcription", wavFileBytes.Length);
            var response = await _httpClient.PostAsync("/transcribe", content);
            if (response.IsSuccessStatusCode)
            {
                var jsonResponse = await response.Content.ReadAsStringAsync();
                var transcriptionResult = JsonSerializer.Deserialize<TranscriptionResult>(jsonResponse);
                
                if (transcriptionResult != null)
                {
                    TranscriptionReceived?.Invoke(this, new TranscriptionEventArgs(transcriptionResult));
                }
            }
            else
            {
                _logger.LogWarning("Transcription request failed with status {StatusCode}: {Reason}", 
                    response.StatusCode, response.ReasonPhrase);
            }
        }
        catch (HttpRequestException ex)
        {
            _logger.LogWarning(ex, "Failed to send audio to backend - service may be unavailable");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending audio for transcription");
        }
    }

    private static float CalculateAudioLevel(byte[] buffer, int bytesRecorded)
    {
        if (bytesRecorded == 0) return 0f;

        long sum = 0;
        for (int i = 0; i < bytesRecorded; i += 2)
        {
            if (i + 1 < bytesRecorded)
            {
                short sample = BitConverter.ToInt16(buffer, i);
                sum += Math.Abs(sample);
            }
        }

        var average = sum / (bytesRecorded / 2.0);
        return (float)(average / short.MaxValue);
    }

    public void Dispose()
    {
        StopCaptureAsync().GetAwaiter().GetResult();
        
        _microphoneCapture?.Dispose();
        _systemCapture?.Dispose();
        _microphoneBuffer?.ClearBuffer();
        _systemBuffer?.ClearBuffer();
        
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Interface for audio capture service.
/// </summary>
public interface IAudioService : IDisposable
{
    event EventHandler<AudioLevelEventArgs>? MicrophoneLevelChanged;
    event EventHandler<AudioLevelEventArgs>? SystemLevelChanged;
    event EventHandler<TranscriptionEventArgs>? TranscriptionReceived;
    
    Task StartCaptureAsync(int microphoneDeviceIndex, bool enableSystemCapture = true, int systemAudioDeviceIndex = -1, float microphoneSensitivity = 1.0f);
    Task StopCaptureAsync();
    Task StartMonitoringAsync(int microphoneDeviceIndex, bool enableSystemCapture = true, int systemAudioDeviceIndex = -1, float microphoneSensitivity = 1.0f);
    Task StopMonitoringAsync();
}

/// <summary>
/// Event arguments for audio level changes.
/// </summary>
public class AudioLevelEventArgs : EventArgs
{
    public float Level { get; }
    
    public AudioLevelEventArgs(float level)
    {
        Level = Math.Max(0f, Math.Min(1f, level));
    }
}

/// <summary>
/// Event arguments for transcription results.
/// </summary>
public class TranscriptionEventArgs : EventArgs
{
    public TranscriptionResult Result { get; }
    
    public TranscriptionEventArgs(TranscriptionResult result)
    {
        Result = result;
    }
} 