using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using Microsoft.Extensions.Logging;
using NAudio.CoreAudioApi;
using Oreja.Models;
using Oreja.Services;

namespace Oreja.ViewModels;

/// <summary>
/// Main view model for the Oreja application implementing MVVM pattern.
/// </summary>
public class MainViewModel : INotifyPropertyChanged, IDisposable
{
    private readonly IAudioService _audioService;
    private readonly ISpeakerService _speakerService;
    private readonly ILogger<MainViewModel> _logger;

    private bool _isRecording;
    private float _microphoneLevel;
    private float _systemLevel;
    private string _transcriptionText = string.Empty;
    private string _statusMessage = "Ready";
    private AudioDevice? _selectedMicrophone;
    private bool _enableSystemCapture = true;
    private string _currentSpeakerName = string.Empty;
    private string _newSpeakerName = string.Empty;

    public MainViewModel(IAudioService audioService, ISpeakerService speakerService, ILogger<MainViewModel> logger)
    {
        _audioService = audioService;
        _speakerService = speakerService;
        _logger = logger;

        // Initialize commands
        StartRecordingCommand = new RelayCommand(async () => await StartRecordingAsync(), () => !IsRecording);
        StopRecordingCommand = new RelayCommand(async () => await StopRecordingAsync(), () => IsRecording);
        SaveTranscriptCommand = new RelayCommand(async () => await SaveTranscriptAsync(), () => !string.IsNullOrEmpty(TranscriptionText));
        ClearTranscriptCommand = new RelayCommand(ClearTranscript, () => !string.IsNullOrEmpty(TranscriptionText));
        RenameSpeakerCommand = new RelayCommand(async () => await RenameSpeakerAsync(), () => !string.IsNullOrEmpty(CurrentSpeakerName) && !string.IsNullOrEmpty(NewSpeakerName));
        RefreshDevicesCommand = new RelayCommand(RefreshAudioDevices);

        // Initialize collections
        AvailableMicrophones = new ObservableCollection<AudioDevice>();
        TranscriptionSegments = new ObservableCollection<TranscriptionSegment>();

        // Subscribe to audio service events
        _audioService.MicrophoneLevelChanged += OnMicrophoneLevelChanged;
        _audioService.SystemLevelChanged += OnSystemLevelChanged;
        _audioService.TranscriptionReceived += OnTranscriptionReceived;

        // Initialize
        RefreshAudioDevices();
    }

    #region Properties

    /// <summary>
    /// Indicates whether audio recording is currently active.
    /// </summary>
    public bool IsRecording
    {
        get => _isRecording;
        set
        {
            if (SetProperty(ref _isRecording, value))
            {
                OnPropertyChanged(nameof(IsNotRecording));
                CommandManager.InvalidateRequerySuggested();
            }
        }
    }

    /// <summary>
    /// Inverse of IsRecording for UI binding convenience.
    /// </summary>
    public bool IsNotRecording => !IsRecording;

    /// <summary>
    /// Current microphone audio level (0.0 to 1.0).
    /// </summary>
    public float MicrophoneLevel
    {
        get => _microphoneLevel;
        set => SetProperty(ref _microphoneLevel, Math.Max(0f, Math.Min(1f, value)));
    }

    /// <summary>
    /// Current system audio level (0.0 to 1.0).
    /// </summary>
    public float SystemLevel
    {
        get => _systemLevel;
        set => SetProperty(ref _systemLevel, Math.Max(0f, Math.Min(1f, value)));
    }

    /// <summary>
    /// Complete transcription text for display.
    /// </summary>
    public string TranscriptionText
    {
        get => _transcriptionText;
        set
        {
            if (SetProperty(ref _transcriptionText, value))
            {
                CommandManager.InvalidateRequerySuggested();
            }
        }
    }

    /// <summary>
    /// Current application status message.
    /// </summary>
    public string StatusMessage
    {
        get => _statusMessage;
        set => SetProperty(ref _statusMessage, value);
    }

    /// <summary>
    /// Currently selected microphone device.
    /// </summary>
    public AudioDevice? SelectedMicrophone
    {
        get => _selectedMicrophone;
        set => SetProperty(ref _selectedMicrophone, value);
    }

    /// <summary>
    /// Whether to capture system audio along with microphone.
    /// </summary>
    public bool EnableSystemCapture
    {
        get => _enableSystemCapture;
        set => SetProperty(ref _enableSystemCapture, value);
    }

    /// <summary>
    /// Name of speaker selected for renaming.
    /// </summary>
    public string CurrentSpeakerName
    {
        get => _currentSpeakerName;
        set
        {
            if (SetProperty(ref _currentSpeakerName, value))
            {
                CommandManager.InvalidateRequerySuggested();
            }
        }
    }

    /// <summary>
    /// New name for speaker renaming operation.
    /// </summary>
    public string NewSpeakerName
    {
        get => _newSpeakerName;
        set
        {
            if (SetProperty(ref _newSpeakerName, value))
            {
                CommandManager.InvalidateRequerySuggested();
            }
        }
    }

    /// <summary>
    /// Collection of available microphone devices.
    /// </summary>
    public ObservableCollection<AudioDevice> AvailableMicrophones { get; }

    /// <summary>
    /// Collection of transcription segments for detailed view.
    /// </summary>
    public ObservableCollection<TranscriptionSegment> TranscriptionSegments { get; }

    #endregion

    #region Commands

    public ICommand StartRecordingCommand { get; }
    public ICommand StopRecordingCommand { get; }
    public ICommand SaveTranscriptCommand { get; }
    public ICommand ClearTranscriptCommand { get; }
    public ICommand RenameSpeakerCommand { get; }
    public ICommand RefreshDevicesCommand { get; }

    #endregion

    #region Methods

    private async Task StartRecordingAsync()
    {
        try
        {
            if (SelectedMicrophone == null)
            {
                StatusMessage = "Please select a microphone device";
                return;
            }

            StatusMessage = "Starting audio capture...";
            await _audioService.StartCaptureAsync(SelectedMicrophone.Index, EnableSystemCapture);
            
            IsRecording = true;
            StatusMessage = "Recording and transcribing...";
            
            _logger.LogInformation("Recording started successfully");
        }
        catch (Exception ex)
        {
            StatusMessage = $"Failed to start recording: {ex.Message}";
            _logger.LogError(ex, "Error starting recording");
        }
    }

    private async Task StopRecordingAsync()
    {
        try
        {
            StatusMessage = "Stopping audio capture...";
            await _audioService.StopCaptureAsync();
            
            IsRecording = false;
            MicrophoneLevel = 0f;
            SystemLevel = 0f;
            StatusMessage = "Recording stopped";
            
            _logger.LogInformation("Recording stopped successfully");
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error stopping recording: {ex.Message}";
            _logger.LogError(ex, "Error stopping recording");
        }
    }

    private async Task SaveTranscriptAsync()
    {
        try
        {
            if (string.IsNullOrEmpty(TranscriptionText))
            {
                StatusMessage = "No transcription to save";
                return;
            }

            var fileName = $"transcript_{DateTime.Now:yyyyMMdd_HHmmss}.txt";
            var filePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), fileName);
            
            await File.WriteAllTextAsync(filePath, TranscriptionText, Encoding.UTF8);
            
            StatusMessage = $"Transcript saved to {fileName}";
            _logger.LogInformation("Transcript saved to {FilePath}", filePath);
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error saving transcript: {ex.Message}";
            _logger.LogError(ex, "Error saving transcript");
        }
    }

    private void ClearTranscript()
    {
        TranscriptionText = string.Empty;
        TranscriptionSegments.Clear();
        StatusMessage = "Transcript cleared";
    }

    private async Task RenameSpeakerAsync()
    {
        try
        {
            if (string.IsNullOrEmpty(CurrentSpeakerName) || string.IsNullOrEmpty(NewSpeakerName))
                return;

            var success = await _speakerService.RenameSpeakerAsync(CurrentSpeakerName, NewSpeakerName);
            if (success)
            {
                // Update existing segments with new speaker name
                foreach (var segment in TranscriptionSegments.Where(s => s.Speaker == CurrentSpeakerName))
                {
                    segment.Speaker = NewSpeakerName;
                }

                // Rebuild transcription text
                RebuildTranscriptionText();
                
                StatusMessage = $"Speaker renamed from '{CurrentSpeakerName}' to '{NewSpeakerName}'";
                CurrentSpeakerName = string.Empty;
                NewSpeakerName = string.Empty;
            }
            else
            {
                StatusMessage = "Failed to rename speaker";
            }
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error renaming speaker: {ex.Message}";
            _logger.LogError(ex, "Error renaming speaker");
        }
    }

    private void RefreshAudioDevices()
    {
        try
        {
            AvailableMicrophones.Clear();
            
            using var enumerator = new MMDeviceEnumerator();
            var devices = enumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active);
            
            for (int i = 0; i < devices.Count; i++)
            {
                var device = devices[i];
                AvailableMicrophones.Add(new AudioDevice
                {
                    Index = i,
                    Name = device.FriendlyName,
                    DeviceId = device.ID,
                    IsDefault = device.DataFlow == DataFlow.Capture && device.State == DeviceState.Active
                });
            }

            // Select default microphone if available
            SelectedMicrophone = AvailableMicrophones.FirstOrDefault(d => d.IsDefault) ?? AvailableMicrophones.FirstOrDefault();
            StatusMessage = $"Found {AvailableMicrophones.Count} microphone(s)";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error refreshing devices: {ex.Message}";
            _logger.LogError(ex, "Error refreshing audio devices");
        }
    }

    private void OnMicrophoneLevelChanged(object? sender, AudioLevelEventArgs e)
    {
        // Update on UI thread
        App.Current.Dispatcher.BeginInvoke(() => MicrophoneLevel = e.Level);
    }

    private void OnSystemLevelChanged(object? sender, AudioLevelEventArgs e)
    {
        // Update on UI thread
        App.Current.Dispatcher.BeginInvoke(() => SystemLevel = e.Level);
    }

    private void OnTranscriptionReceived(object? sender, TranscriptionEventArgs e)
    {
        // Update on UI thread
        App.Current.Dispatcher.BeginInvoke(() =>
        {
            foreach (var segment in e.Result.Segments)
            {
                TranscriptionSegments.Add(segment);
            }
            
            RebuildTranscriptionText();
            StatusMessage = $"Transcribed {e.Result.Segments.Count} segment(s) in {e.Result.ProcessingTime:F2}s";
        });
    }

    private void RebuildTranscriptionText()
    {
        var sb = new StringBuilder();
        string? currentSpeaker = null;
        
        foreach (var segment in TranscriptionSegments.OrderBy(s => s.Start))
        {
            if (segment.Speaker != currentSpeaker)
            {
                if (sb.Length > 0)
                    sb.AppendLine();
                
                sb.Append($"{segment.Speaker}: ");
                currentSpeaker = segment.Speaker;
            }
            
            sb.Append(segment.Text);
            if (!segment.Text.EndsWith(" "))
                sb.Append(" ");
        }
        
        TranscriptionText = sb.ToString();
    }

    #endregion

    #region INotifyPropertyChanged

    public event PropertyChangedEventHandler? PropertyChanged;

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    protected bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (Equals(field, value)) return false;
        field = value;
        OnPropertyChanged(propertyName);
        return true;
    }

    #endregion

    #region IDisposable

    public void Dispose()
    {
        _audioService.MicrophoneLevelChanged -= OnMicrophoneLevelChanged;
        _audioService.SystemLevelChanged -= OnSystemLevelChanged;
        _audioService.TranscriptionReceived -= OnTranscriptionReceived;
        
        _audioService?.Dispose();
        GC.SuppressFinalize(this);
    }

    #endregion
}

/// <summary>
/// Simple implementation of ICommand for view model commands.
/// </summary>
public class RelayCommand : ICommand
{
    private readonly Action _execute;
    private readonly Func<bool>? _canExecute;

    public RelayCommand(Action execute, Func<bool>? canExecute = null)
    {
        _execute = execute ?? throw new ArgumentNullException(nameof(execute));
        _canExecute = canExecute;
    }

    public event EventHandler? CanExecuteChanged
    {
        add { CommandManager.RequerySuggested += value; }
        remove { CommandManager.RequerySuggested -= value; }
    }

    public bool CanExecute(object? parameter) => _canExecute?.Invoke() ?? true;

    public void Execute(object? parameter) => _execute();
} 