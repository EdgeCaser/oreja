using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Collections.ObjectModel;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wasapi;
using System;
using System.Threading.Tasks;
using System.Windows.Threading;
using System.Net.Http;
using System.IO;
using System.Text.Json;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Oreja;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool AllocConsole();
    
    private ComboBox? _microphoneComboBox;
    private ComboBox? _systemAudioComboBox;
    private Button? _startRecordingButton;
    private Button? _stopRecordingButton;
    private Rectangle? _microphoneVolumeBar;
    private Rectangle? _systemAudioVolumeBar;
    private TextBlock? _statusText;
    private ScrollViewer? _transcriptionScrollViewer;
    private StackPanel? _transcriptionPanel;
    private ObservableCollection<MMDevice>? _availableMicrophones;
    private ObservableCollection<MMDevice>? _availableSystemAudioDevices;
    private DispatcherTimer? _volumeTimer;
    private int _selectedMicrophoneIndex = 0;
    private int _selectedSystemAudioIndex = 0;
    private bool _isRecording = false;
    
    // Audio capture components
    private WaveInEvent? _waveIn;
    private WasapiLoopbackCapture? _systemAudioCapture;
    private MMDeviceEnumerator? _deviceEnumerator;
    private MMDevice? _selectedMicrophone;
    private MMDevice? _defaultSystemAudio;
    
    // Volume level tracking
    private float _microphoneLevel = 0f;
    private float _systemAudioLevel = 0f;
    
    // Transcription components
    private HttpClient? _httpClient;
    private List<byte> _audioBuffer = new List<byte>();
    private List<byte> _systemAudioBuffer = new List<byte>(); // New: System audio buffer
    private DispatcherTimer? _transcriptionTimer;
    private const string BACKEND_URL = "http://127.0.0.1:8000";
    private const int AUDIO_CHUNK_DURATION_MS = 5000; // Send audio every 5 seconds (increased from 3)
    private bool _isProcessingTranscription = false; // Prevent overlapping requests
    
    // Speaker renaming and save functionality
    private Dictionary<string, string> _speakerNames = new Dictionary<string, string>();
    private Button? _saveTranscriptionButton;
    private List<TranscriptionSegment> _transcriptionHistory = new List<TranscriptionSegment>();

    // Helper class for transcription segments
    public class TranscriptionSegment
    {
        public string? Speaker { get; set; }
        public string? Text { get; set; }
        public double StartTime { get; set; }
        public string Source { get; set; } = "";
        public DateTime Timestamp { get; set; } = DateTime.Now;
    }

    protected override void OnStartup(StartupEventArgs e)
    {
        // Allocate console for debugging
        AllocConsole();
        Console.WriteLine("=== Oreja Application Starting ===");
        
        base.OnStartup(e);

        // Add global exception handler
        this.DispatcherUnhandledException += App_DispatcherUnhandledException;

        try
        {
            Console.WriteLine("Initializing components...");
            _availableMicrophones = new ObservableCollection<MMDevice>();
            _availableSystemAudioDevices = new ObservableCollection<MMDevice>();
            _deviceEnumerator = new MMDeviceEnumerator();
            _httpClient = new HttpClient();
            _httpClient.Timeout = TimeSpan.FromSeconds(60);
            
            Console.WriteLine("Creating window...");
            // Create window entirely in code to bypass XAML issues
            var window = new Window
            {
                Title = "Oreja - Real-time Conference Transcription",
                Width = 800,
                Height = 700,
                WindowStartupLocation = WindowStartupLocation.CenterScreen,
                WindowState = WindowState.Normal,
                Topmost = false
            };
            
            var mainPanel = new StackPanel { Margin = new Thickness(20) };
            
            // Title
            var titleText = new TextBlock 
            { 
                Text = "Oreja - Audio Capture & Transcription",
                FontSize = 24,
                FontWeight = FontWeights.Bold,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 0, 0, 20)
            };
            
            // Microphone selection
            var microphoneLabel = new TextBlock 
            { 
                Text = "Select Microphone:",
                FontSize = 14,
                Margin = new Thickness(0, 0, 0, 5)
            };
            
            _microphoneComboBox = new ComboBox
            {
                Width = 500,
                HorizontalAlignment = HorizontalAlignment.Left,
                Margin = new Thickness(0, 0, 0, 15)
            };
            _microphoneComboBox.SelectionChanged += MicrophoneComboBox_SelectionChanged;
            
            // System audio selection
            var systemAudioLabel = new TextBlock 
            { 
                Text = "Select System Audio:",
                FontSize = 14,
                Margin = new Thickness(0, 0, 0, 5)
            };
            
            _systemAudioComboBox = new ComboBox
            {
                Width = 500,
                HorizontalAlignment = HorizontalAlignment.Left,
                Margin = new Thickness(0, 0, 0, 15)
            };
            _systemAudioComboBox.SelectionChanged += SystemAudioComboBox_SelectionChanged;
            
            // Volume meters section
            var volumeLabel = new TextBlock 
            { 
                Text = "Audio Levels:",
                FontSize = 14,
                FontWeight = FontWeights.Bold,
                Margin = new Thickness(0, 0, 0, 10)
            };
            
            // Microphone volume meter
            var micVolumePanel = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 0, 10) };
            var micVolumeLabel = new TextBlock { Text = "Microphone: ", Width = 120, VerticalAlignment = VerticalAlignment.Center };
            var micVolumeBorder = new Border 
            { 
                Width = 300, 
                Height = 20, 
                BorderBrush = Brushes.Gray, 
                BorderThickness = new Thickness(1),
                Background = Brushes.LightGray
            };
            _microphoneVolumeBar = new Rectangle 
            { 
                Fill = Brushes.LimeGreen, 
                HorizontalAlignment = HorizontalAlignment.Left,
                VerticalAlignment = VerticalAlignment.Stretch,
                Width = 0
            };
            micVolumeBorder.Child = _microphoneVolumeBar;
            micVolumePanel.Children.Add(micVolumeLabel);
            micVolumePanel.Children.Add(micVolumeBorder);
            
            // System audio volume meter
            var sysVolumePanel = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 0, 20) };
            var sysVolumeLabel = new TextBlock { Text = "System Audio: ", Width = 120, VerticalAlignment = VerticalAlignment.Center };
            var sysVolumeBorder = new Border 
            { 
                Width = 300, 
                Height = 20, 
                BorderBrush = Brushes.Gray, 
                BorderThickness = new Thickness(1),
                Background = Brushes.LightGray
            };
            _systemAudioVolumeBar = new Rectangle 
            { 
                Fill = Brushes.DodgerBlue, 
                HorizontalAlignment = HorizontalAlignment.Left,
                VerticalAlignment = VerticalAlignment.Stretch,
                Width = 0
            };
            sysVolumeBorder.Child = _systemAudioVolumeBar;
            sysVolumePanel.Children.Add(sysVolumeLabel);
            sysVolumePanel.Children.Add(sysVolumeBorder);
            
            // Recording controls
            var buttonPanel = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Center, Margin = new Thickness(0, 0, 0, 20) };
            
            _startRecordingButton = new Button 
            { 
                Content = "▶ Start Recording",
                Width = 150,
                Height = 40,
                Margin = new Thickness(0, 0, 10, 0),
                FontSize = 14,
                Background = Brushes.LightGreen
            };
            _startRecordingButton.Click += StartRecordingButton_Click;
            
            _stopRecordingButton = new Button 
            { 
                Content = "⏹ Stop Recording",
                Width = 150,
                Height = 40,
                IsEnabled = false,
                FontSize = 14,
                Background = Brushes.LightCoral
            };
            _stopRecordingButton.Click += StopRecordingButton_Click;
            
            _saveTranscriptionButton = new Button 
            { 
                Content = "💾 Save Transcription",
                Width = 150,
                Height = 40,
                Margin = new Thickness(10, 0, 0, 0),
                FontSize = 14,
                Background = Brushes.LightBlue,
                IsEnabled = false // Initially disabled until we have transcriptions
            };
            _saveTranscriptionButton.Click += SaveTranscriptionButton_Click;
            
            buttonPanel.Children.Add(_startRecordingButton);
            buttonPanel.Children.Add(_stopRecordingButton);
            buttonPanel.Children.Add(_saveTranscriptionButton);
            
            // Status text
            _statusText = new TextBlock 
            { 
                Text = "Initializing audio devices...",
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 0, 0, 15),
                TextWrapping = TextWrapping.Wrap,
                FontSize = 12
            };
            
            // Transcription section
            var transcriptionLabel = new TextBlock 
            { 
                Text = "Live Transcription:",
                FontSize = 16,
                FontWeight = FontWeights.Bold,
                Margin = new Thickness(0, 0, 0, 10)
            };
            
            _transcriptionPanel = new StackPanel 
            { 
                Margin = new Thickness(10),
                Background = Brushes.White
            };
            
            _transcriptionScrollViewer = new ScrollViewer 
            { 
                Height = 250,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
                BorderBrush = Brushes.Gray,
                BorderThickness = new Thickness(1),
                Background = Brushes.WhiteSmoke,
                Content = _transcriptionPanel
            };
            
            // Add all controls to main panel
            mainPanel.Children.Add(titleText);
            mainPanel.Children.Add(microphoneLabel);
            mainPanel.Children.Add(_microphoneComboBox);
            mainPanel.Children.Add(systemAudioLabel);
            mainPanel.Children.Add(_systemAudioComboBox);
            mainPanel.Children.Add(volumeLabel);
            mainPanel.Children.Add(micVolumePanel);
            mainPanel.Children.Add(sysVolumePanel);
            mainPanel.Children.Add(buttonPanel);
            mainPanel.Children.Add(_statusText);
            mainPanel.Children.Add(transcriptionLabel);
            mainPanel.Children.Add(_transcriptionScrollViewer);
            
            window.Content = mainPanel;
            window.Closing += Window_Closing;
            
            Console.WriteLine("Setting as main window...");
            // Set as main window and show
            this.MainWindow = window;
            
            Console.WriteLine("Showing window...");
            // Show and activate the window FIRST
            window.Show();
            window.Activate();
            window.Focus();
            
            Console.WriteLine("Window shown, starting async initialization...");
            // Initialize everything else asynchronously to prevent blocking
            this.Dispatcher.BeginInvoke(new Action(() =>
            {
                try
                {
                    Console.WriteLine("Loading audio devices...");
                    // Initialize audio devices
                    LoadAudioDevices();
                    
                    Console.WriteLine("Checking backend connection...");
                    // Check backend connectivity
                    CheckBackendConnection();
                    
                    Console.WriteLine("Setting up timers...");
                    // Setup volume monitoring timer - less frequent to improve UI responsiveness
                    _volumeTimer = new DispatcherTimer();
                    _volumeTimer.Interval = TimeSpan.FromMilliseconds(200); // Reduced frequency
                    _volumeTimer.Tick += VolumeTimer_Tick;
                    _volumeTimer.Start();
                    
                    // Setup transcription timer
                    _transcriptionTimer = new DispatcherTimer();
                    _transcriptionTimer.Interval = TimeSpan.FromMilliseconds(AUDIO_CHUNK_DURATION_MS);
                    _transcriptionTimer.Tick += TranscriptionTimer_Tick;
                    
                    Console.WriteLine("Initialization complete!");
                    if (_statusText != null)
                    {
                        _statusText.Text = "Ready! Application initialized successfully.";
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Async initialization error: {ex.Message}");
                    if (_statusText != null)
                    {
                        _statusText.Text = $"Initialization error: {ex.Message}";
                    }
                }
            }), System.Windows.Threading.DispatcherPriority.Background);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Main startup error: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            MessageBox.Show($"Error starting application: {ex.Message}\n\nStack trace: {ex.StackTrace}", "Oreja Error", MessageBoxButton.OK, MessageBoxImage.Error);
            this.Shutdown();
        }
    }
    
    private async void CheckBackendConnection()
    {
        try
        {
            var response = await _httpClient!.GetAsync($"{BACKEND_URL}/health");
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var healthData = JsonSerializer.Deserialize<JsonElement>(content);
                
                if (_statusText != null)
                {
                    _statusText.Text += " Backend connected ✓";
                }
            }
        }
        catch (Exception ex)
        {
            if (_statusText != null)
            {
                _statusText.Text += $" Backend connection failed: {ex.Message}";
            }
        }
    }
    
    private void LoadAudioDevices()
    {
        try
        {
            _availableMicrophones?.Clear();
            _availableSystemAudioDevices?.Clear();
            
            if (_deviceEnumerator != null)
            {
                // Load microphone input devices
                var inputDevices = _deviceEnumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active);
                foreach (var device in inputDevices)
                {
                    _availableMicrophones?.Add(device);
                }
                
                // Load system audio output devices
                var outputDevices = _deviceEnumerator.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active);
                foreach (var device in outputDevices)
                {
                    _availableSystemAudioDevices?.Add(device);
                }
                
                // Set default system audio device
                _defaultSystemAudio = _deviceEnumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
            }
            
            if (_microphoneComboBox != null && _availableMicrophones != null)
            {
                _microphoneComboBox.ItemsSource = _availableMicrophones;
                _microphoneComboBox.DisplayMemberPath = "FriendlyName";
                
                if (_availableMicrophones.Count > 0)
                {
                    _microphoneComboBox.SelectedIndex = 0;
                    _selectedMicrophoneIndex = 0;
                    _selectedMicrophone = _availableMicrophones[0];
                }
            }
            
            if (_systemAudioComboBox != null && _availableSystemAudioDevices != null)
            {
                _systemAudioComboBox.ItemsSource = _availableSystemAudioDevices;
                _systemAudioComboBox.DisplayMemberPath = "FriendlyName";
                
                if (_availableSystemAudioDevices.Count > 0)
                {
                    _systemAudioComboBox.SelectedIndex = 0;
                    _selectedSystemAudioIndex = 0;
                }
            }
            
            if (_statusText != null)
            {
                _statusText.Text = $"Found {_availableMicrophones?.Count ?? 0} microphone(s) and {_availableSystemAudioDevices?.Count ?? 0} system audio device(s).";
            }
        }
        catch (Exception ex)
        {
            if (_statusText != null)
            {
                _statusText.Text = $"Error loading audio devices: {ex.Message}";
            }
        }
    }
    
    private void MicrophoneComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_microphoneComboBox != null && _availableMicrophones != null)
        {
            _selectedMicrophoneIndex = _microphoneComboBox.SelectedIndex;
            if (_selectedMicrophoneIndex >= 0 && _selectedMicrophoneIndex < _availableMicrophones.Count)
            {
                _selectedMicrophone = _availableMicrophones[_selectedMicrophoneIndex];
            }
        }
    }
    
    private void SystemAudioComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_systemAudioComboBox != null && _availableSystemAudioDevices != null)
        {
            _selectedSystemAudioIndex = _systemAudioComboBox.SelectedIndex;
            if (_selectedSystemAudioIndex >= 0 && _selectedSystemAudioIndex < _availableSystemAudioDevices.Count)
            {
                _defaultSystemAudio = _availableSystemAudioDevices[_selectedSystemAudioIndex];
            }
        }
    }
    
    private void StartRecordingButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            if (_selectedMicrophone != null)
            {
                // Clear previous transcriptions
                _transcriptionPanel?.Children.Clear();
                _audioBuffer.Clear();
                _systemAudioBuffer.Clear();
                
                // Initialize WaveIn for actual audio capture
                _waveIn = new WaveInEvent
                {
                    DeviceNumber = _selectedMicrophoneIndex,
                    WaveFormat = new WaveFormat(16000, 1), // 16kHz, mono - matching backend requirements
                    BufferMilliseconds = 100, // Increased buffer for better capture
                    NumberOfBuffers = 3 // More buffers for smoother capture
                };
                
                _waveIn.DataAvailable += WaveIn_DataAvailable;
                _waveIn.RecordingStopped += WaveIn_RecordingStopped;
                
                _waveIn.StartRecording();
                
                // Initialize system audio capture (loopback)
                if (_defaultSystemAudio != null)
                {
                    _systemAudioCapture = new WasapiLoopbackCapture(_defaultSystemAudio);
                    _systemAudioCapture.DataAvailable += SystemAudio_DataAvailable;
                    _systemAudioCapture.RecordingStopped += SystemAudio_RecordingStopped;
                    _systemAudioCapture.StartRecording();
                }
                
                _transcriptionTimer?.Start();
                _isRecording = true;
                
                if (_statusText != null) _statusText.Text = $"Recording from: {_selectedMicrophone.FriendlyName} - Transcribing in real-time...";
                if (_startRecordingButton != null) _startRecordingButton.IsEnabled = false;
                if (_stopRecordingButton != null) _stopRecordingButton.IsEnabled = true;
            }
            else
            {
                if (_statusText != null) _statusText.Text = "Please select a microphone first.";
            }
        }
        catch (Exception ex)
        {
            if (_statusText != null) _statusText.Text = $"Error starting recording: {ex.Message}";
            if (_startRecordingButton != null) _startRecordingButton.IsEnabled = true;
        }
    }
    
    private void StopRecordingButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            _waveIn?.StopRecording();
            _systemAudioCapture?.StopRecording();
            _transcriptionTimer?.Stop();
            _isRecording = false;
            
            // Process any remaining audio
            if (_audioBuffer.Count > 0)
            {
                _ = ProcessAudioChunkAsync(_audioBuffer.ToArray(), "Microphone");
                _audioBuffer.Clear();
            }
            
            if (_systemAudioBuffer.Count > 0)
            {
                _ = ProcessAudioChunkAsync(_systemAudioBuffer.ToArray(), "System Audio");
                _systemAudioBuffer.Clear();
            }
            
            if (_statusText != null) _statusText.Text = "Recording stopped. Transcription complete.";
            if (_stopRecordingButton != null) _stopRecordingButton.IsEnabled = false;
            if (_startRecordingButton != null) _startRecordingButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            if (_statusText != null) _statusText.Text = $"Error stopping recording: {ex.Message}";
        }
    }
    
    private void WaveIn_DataAvailable(object? sender, WaveInEventArgs e)
    {
        // Calculate audio level from the captured data
        if (e.Buffer.Length > 0)
        {
            float max = 0;
            
            // Apply gain/amplification for better sensitivity (amplify by 2x)
            for (int index = 0; index < e.BytesRecorded; index += 2)
            {
                short sample = (short)((e.Buffer[index + 1] << 8) | e.Buffer[index]);
                
                // Apply 2x gain for better microphone sensitivity
                sample = (short)Math.Max(Math.Min(sample * 2, short.MaxValue), short.MinValue);
                
                // Update the buffer with amplified audio
                e.Buffer[index] = (byte)(sample & 0xFF);
                e.Buffer[index + 1] = (byte)(sample >> 8);
                
                var sample32 = sample / 32768f;
                if (sample32 < 0) sample32 = -sample32;
                if (sample32 > max) max = sample32;
            }
            _microphoneLevel = max;
            
            // Add audio data to buffer for transcription
            _audioBuffer.AddRange(e.Buffer.Take(e.BytesRecorded));
        }
    }
    
    private void WaveIn_RecordingStopped(object? sender, StoppedEventArgs e)
    {
        _waveIn?.Dispose();
        _waveIn = null;
        _microphoneLevel = 0f;
    }
    
    private void SystemAudio_DataAvailable(object? sender, WaveInEventArgs e)
    {
        // Add system audio data to buffer for transcription
        if (e.Buffer.Length > 0)
        {
            // Convert system audio to the same format as microphone (16kHz mono)
            // Note: System audio is typically 44.1kHz stereo, so we need to resample
            _systemAudioBuffer.AddRange(e.Buffer.Take(e.BytesRecorded));
        }
    }
    
    private void SystemAudio_RecordingStopped(object? sender, StoppedEventArgs e)
    {
        _systemAudioCapture?.Dispose();
        _systemAudioCapture = null;
        _systemAudioLevel = 0f;
    }
    
    private async void TranscriptionTimer_Tick(object? sender, EventArgs e)
    {
        if (_isRecording && !_isProcessingTranscription)
        {
            // Process microphone audio if available
            if (_audioBuffer.Count > 0)
            {
                var micAudioData = _audioBuffer.ToArray();
                _audioBuffer.Clear();
                _ = ProcessAudioChunkAsync(micAudioData, "Microphone");
            }
            
            // Process system audio if available
            if (_systemAudioBuffer.Count > 0)
            {
                var sysAudioData = _systemAudioBuffer.ToArray();
                _systemAudioBuffer.Clear();
                _ = ProcessAudioChunkAsync(sysAudioData, "System Audio");
            }
        }
    }
    
    private async Task ProcessAudioChunkAsync(byte[] audioData, string source)
    {
        if (_isProcessingTranscription) return; // Skip if already processing
        
        _isProcessingTranscription = true;
        
        try
        {
            Console.WriteLine($"Processing audio chunk of {audioData.Length} bytes from {source}...");
            
            // Convert raw audio data to WAV format
            var wavData = CreateWavFile(audioData, 16000, 1);
            Console.WriteLine($"Created WAV file of {wavData.Length} bytes");
            
            // Send to backend for transcription
            using var content = new MultipartFormDataContent();
            using var audioContent = new ByteArrayContent(wavData);
            audioContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("audio/wav");
            content.Add(audioContent, "audio", $"{source}.wav");
            
            Console.WriteLine("Sending request to backend...");
            var response = await _httpClient!.PostAsync($"{BACKEND_URL}/transcribe", content);
            
            Console.WriteLine($"Backend response status: {response.StatusCode}");
            
            if (response.IsSuccessStatusCode)
            {
                var jsonResponse = await response.Content.ReadAsStringAsync();
                Console.WriteLine($"Backend response: {jsonResponse}");
                
                var transcriptionResult = JsonSerializer.Deserialize<JsonElement>(jsonResponse);
                
                // Extract transcription segments
                if (transcriptionResult.TryGetProperty("segments", out var segments))
                {
                    Console.WriteLine($"Found {segments.GetArrayLength()} segments");
                    
                    foreach (var segment in segments.EnumerateArray())
                    {
                        var text = segment.GetProperty("text").GetString();
                        var speaker = segment.GetProperty("speaker").GetString();
                        var startTime = segment.GetProperty("start").GetDouble();
                        
                        Console.WriteLine($"Segment: Speaker='{speaker}', Text='{text}', StartTime={startTime}");
                        
                        if (!string.IsNullOrWhiteSpace(text))
                        {
                            // Add transcription to UI (run on UI thread)
                            Dispatcher.Invoke(() => AddTranscriptionSegment(speaker, text, startTime, source));
                        }
                    }
                }
                else
                {
                    Console.WriteLine("No 'segments' property found in response");
                }
            }
            else
            {
                Console.WriteLine($"Backend request failed with status: {response.StatusCode}");
                var errorContent = await response.Content.ReadAsStringAsync();
                Console.WriteLine($"Error content: {errorContent}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ProcessAudioChunkAsync error: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            
            Dispatcher.Invoke(() =>
            {
                if (_statusText != null)
                {
                    _statusText.Text = $"Transcription error: {ex.Message}";
                }
            });
        }
        finally
        {
            _isProcessingTranscription = false;
        }
    }
    
    private void AddTranscriptionSegment(string? speaker, string? text, double startTime, string source)
    {
        Console.WriteLine($"AddTranscriptionSegment called: Speaker='{speaker}', Text='{text}', StartTime={startTime}, Source={source}");
        
        if (_transcriptionPanel == null || string.IsNullOrWhiteSpace(text)) 
        {
            Console.WriteLine("Skipping - transcription panel is null or text is empty");
            return;
        }
        
        // Store in transcription history
        var segment = new TranscriptionSegment
        {
            Speaker = speaker,
            Text = text,
            StartTime = startTime,
            Source = source,
            Timestamp = DateTime.Now
        };
        _transcriptionHistory.Add(segment);
        
        // Enable save button once we have transcriptions
        if (_saveTranscriptionButton != null)
        {
            _saveTranscriptionButton.IsEnabled = true;
        }
        
        var segmentPanel = new StackPanel 
        { 
            Orientation = Orientation.Horizontal, 
            Margin = new Thickness(0, 5, 0, 5) 
        };
        
        // Timestamp
        var timestampText = new TextBlock 
        { 
            Text = $"[{TimeSpan.FromSeconds(startTime):mm\\:ss}]",
            FontWeight = FontWeights.Bold,
            Foreground = Brushes.Gray,
            Width = 60,
            Margin = new Thickness(0, 0, 10, 0)
        };
        
        // Speaker (now clickable for renaming)
        var displaySpeaker = GetDisplaySpeakerName(speaker);
        var speakerButton = new Button 
        { 
            Content = $"{displaySpeaker}:",
            FontWeight = FontWeights.Bold,
            Foreground = GetSpeakerColor(speaker),
            Background = Brushes.Transparent,
            BorderThickness = new Thickness(0),
            Width = 100,
            Margin = new Thickness(0, 0, 10, 0),
            Cursor = System.Windows.Input.Cursors.Hand,
            ToolTip = "Click to rename speaker"
        };
        speakerButton.Click += (s, e) => ShowSpeakerRenameDialog(speaker);
        
        // Source indicator
        var sourceText = new TextBlock 
        { 
            Text = $"[{source}]",
            FontWeight = FontWeights.Normal,
            Foreground = source == "Microphone" ? Brushes.Blue : Brushes.Green,
            Width = 80,
            FontSize = 10,
            Margin = new Thickness(0, 0, 10, 0)
        };
        
        // Transcribed text
        var textBlock = new TextBlock 
        { 
            Text = text,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 0, 10, 0)
        };
        
        segmentPanel.Children.Add(timestampText);
        segmentPanel.Children.Add(speakerButton);
        segmentPanel.Children.Add(sourceText);
        segmentPanel.Children.Add(textBlock);
        
        _transcriptionPanel.Children.Add(segmentPanel);
        
        // Auto-scroll to bottom
        _transcriptionScrollViewer?.ScrollToEnd();
        
        Console.WriteLine("Transcription segment added to UI successfully");
    }
    
    private string GetDisplaySpeakerName(string? originalSpeaker)
    {
        if (string.IsNullOrEmpty(originalSpeaker))
            return "Unknown";
            
        // Check if we have a custom name for this speaker
        if (_speakerNames.ContainsKey(originalSpeaker))
            return _speakerNames[originalSpeaker];
            
        return originalSpeaker;
    }
    
    private void ShowSpeakerRenameDialog(string? originalSpeaker)
    {
        if (string.IsNullOrEmpty(originalSpeaker))
            return;
            
        var currentName = GetDisplaySpeakerName(originalSpeaker);
        
        // Create simple input dialog
        var inputWindow = new Window
        {
            Title = "Rename Speaker",
            Width = 350,
            Height = 200,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = MainWindow,
            ResizeMode = ResizeMode.NoResize
        };
        
        var panel = new StackPanel { Margin = new Thickness(20) };
        
        var label = new TextBlock 
        { 
            Text = $"Enter new name for '{currentName}':",
            Margin = new Thickness(0, 0, 0, 10)
        };
        
        var textBox = new TextBox 
        { 
            Text = currentName,
            Margin = new Thickness(0, 0, 0, 20),
            Padding = new Thickness(5)
        };
        textBox.SelectAll();
        textBox.Focus();
        
        var buttonPanel = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        
        var okButton = new Button 
        { 
            Content = "OK",
            Width = 70,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            IsDefault = true
        };
        
        var cancelButton = new Button 
        { 
            Content = "Cancel",
            Width = 70,
            Height = 30,
            IsCancel = true
        };
        
        okButton.Click += (s, e) =>
        {
            var newName = textBox.Text.Trim();
            if (!string.IsNullOrEmpty(newName))
            {
                _speakerNames[originalSpeaker] = newName;
                RefreshTranscriptionDisplay();
                inputWindow.DialogResult = true;
            }
        };
        
        cancelButton.Click += (s, e) => inputWindow.DialogResult = false;
        
        // Handle Enter key
        textBox.KeyDown += (s, e) =>
        {
            if (e.Key == System.Windows.Input.Key.Enter)
            {
                okButton.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
            }
        };
        
        buttonPanel.Children.Add(okButton);
        buttonPanel.Children.Add(cancelButton);
        
        panel.Children.Add(label);
        panel.Children.Add(textBox);
        panel.Children.Add(buttonPanel);
        
        inputWindow.Content = panel;
        inputWindow.ShowDialog();
    }
    
    private void RefreshTranscriptionDisplay()
    {
        if (_transcriptionPanel == null)
            return;
            
        // Clear current display
        _transcriptionPanel.Children.Clear();
        
        // Redraw all segments with updated speaker names
        foreach (var segment in _transcriptionHistory)
        {
            var segmentPanel = new StackPanel 
            { 
                Orientation = Orientation.Horizontal, 
                Margin = new Thickness(0, 5, 0, 5) 
            };
            
            // Timestamp
            var timestampText = new TextBlock 
            { 
                Text = $"[{TimeSpan.FromSeconds(segment.StartTime):mm\\:ss}]",
                FontWeight = FontWeights.Bold,
                Foreground = Brushes.Gray,
                Width = 60,
                Margin = new Thickness(0, 0, 10, 0)
            };
            
            // Speaker (clickable)
            var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);
            var speakerButton = new Button 
            { 
                Content = $"{displaySpeaker}:",
                FontWeight = FontWeights.Bold,
                Foreground = GetSpeakerColor(segment.Speaker),
                Background = Brushes.Transparent,
                BorderThickness = new Thickness(0),
                Width = 100,
                Margin = new Thickness(0, 0, 10, 0),
                Cursor = System.Windows.Input.Cursors.Hand,
                ToolTip = "Click to rename speaker"
            };
            speakerButton.Click += (s, e) => ShowSpeakerRenameDialog(segment.Speaker);
            
            // Source indicator
            var sourceText = new TextBlock 
            { 
                Text = $"[{segment.Source}]",
                FontWeight = FontWeights.Normal,
                Foreground = segment.Source == "Microphone" ? Brushes.Blue : Brushes.Green,
                Width = 80,
                FontSize = 10,
                Margin = new Thickness(0, 0, 10, 0)
            };
            
            // Transcribed text
            var textBlock = new TextBlock 
            { 
                Text = segment.Text,
                TextWrapping = TextWrapping.Wrap,
                Margin = new Thickness(0, 0, 10, 0)
            };
            
            segmentPanel.Children.Add(timestampText);
            segmentPanel.Children.Add(speakerButton);
            segmentPanel.Children.Add(sourceText);
            segmentPanel.Children.Add(textBlock);
            
            _transcriptionPanel.Children.Add(segmentPanel);
        }
        
        // Auto-scroll to bottom
        _transcriptionScrollViewer?.ScrollToEnd();
    }
    
    private void SaveTranscriptionButton_Click(object sender, RoutedEventArgs e)
    {
        if (_transcriptionHistory.Count == 0)
        {
            MessageBox.Show("No transcription data to save.", "Oreja", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        
        var saveFileDialog = new Microsoft.Win32.SaveFileDialog
        {
            Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*",
            DefaultExt = "txt",
            FileName = $"Oreja_Transcription_{DateTime.Now:yyyy-MM-dd_HH-mm-ss}.txt"
        };
        
        if (saveFileDialog.ShowDialog() == true)
        {
            try
            {
                var content = GenerateTranscriptionReport();
                File.WriteAllText(saveFileDialog.FileName, content);
                
                MessageBox.Show($"Transcription saved successfully to:\n{saveFileDialog.FileName}", 
                    "Oreja", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error saving transcription:\n{ex.Message}", 
                    "Oreja Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }
    
    private string GenerateTranscriptionReport()
    {
        var report = new System.Text.StringBuilder();
        
        report.AppendLine("=== OREJA TRANSCRIPTION REPORT ===");
        report.AppendLine($"Generated on: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        report.AppendLine($"Total segments: {_transcriptionHistory.Count}");
        report.AppendLine();
        
        // Group by source
        var microphoneSegments = _transcriptionHistory.Where(s => s.Source == "Microphone").ToList();
        var systemAudioSegments = _transcriptionHistory.Where(s => s.Source == "System Audio").ToList();
        
        if (microphoneSegments.Any())
        {
            report.AppendLine("=== MICROPHONE AUDIO ===");
            foreach (var segment in microphoneSegments)
            {
                var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);
                report.AppendLine($"[{TimeSpan.FromSeconds(segment.StartTime):mm\\:ss}] {displaySpeaker}: {segment.Text}");
            }
            report.AppendLine();
        }
        
        if (systemAudioSegments.Any())
        {
            report.AppendLine("=== SYSTEM AUDIO ===");
            foreach (var segment in systemAudioSegments)
            {
                var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);
                report.AppendLine($"[{TimeSpan.FromSeconds(segment.StartTime):mm\\:ss}] {displaySpeaker}: {segment.Text}");
            }
            report.AppendLine();
        }
        
        // Combined chronological view
        report.AppendLine("=== CHRONOLOGICAL TRANSCRIPT ===");
        var sortedSegments = _transcriptionHistory.OrderBy(s => s.Timestamp).ToList();
        foreach (var segment in sortedSegments)
        {
            var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);
            report.AppendLine($"[{TimeSpan.FromSeconds(segment.StartTime):mm\\:ss}] [{segment.Source}] {displaySpeaker}: {segment.Text}");
        }
        
        return report.ToString();
    }

    private Brush GetSpeakerColor(string? speaker)
    {
        // Assign colors to speakers for better visualization
        return speaker switch
        {
            "Speaker 1" or "Speaker SPEAKER_00" => Brushes.Blue,
            "Speaker 2" or "Speaker SPEAKER_01" => Brushes.Green,
            "Speaker 3" or "Speaker SPEAKER_02" => Brushes.Purple,
            "Speaker 4" or "Speaker SPEAKER_03" => Brushes.Orange,
            _ => Brushes.Black
        };
    }
    
    private byte[] CreateWavFile(byte[] audioData, int sampleRate, int channels)
    {
        using var memoryStream = new MemoryStream();
        using var writer = new BinaryWriter(memoryStream);
        
        // WAV header
        writer.Write("RIFF".ToCharArray());
        writer.Write(36 + audioData.Length);
        writer.Write("WAVE".ToCharArray());
        writer.Write("fmt ".ToCharArray());
        writer.Write(16); // PCM
        writer.Write((short)1); // Format
        writer.Write((short)channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * 2); // Byte rate
        writer.Write((short)(channels * 2)); // Block align
        writer.Write((short)16); // Bits per sample
        writer.Write("data".ToCharArray());
        writer.Write(audioData.Length);
        writer.Write(audioData);
        
        return memoryStream.ToArray();
    }
    
    private void VolumeTimer_Tick(object? sender, EventArgs e)
    {
        try
        {
            // Update microphone volume meter (use cached level from recording)
            if (_microphoneVolumeBar != null)
            {
                var level = _isRecording ? _microphoneLevel : GetMicrophoneLevel();
                _microphoneVolumeBar.Width = level * 298; // 298 = 300 - 2 for border
                
                // Change color based on level
                if (level > 0.8)
                    _microphoneVolumeBar.Fill = Brushes.Red;
                else if (level > 0.5)
                    _microphoneVolumeBar.Fill = Brushes.Orange;
                else
                    _microphoneVolumeBar.Fill = Brushes.LimeGreen;
            }
            
            // Update system audio volume meter (cached device)
            if (_systemAudioVolumeBar != null && _defaultSystemAudio?.AudioMeterInformation != null)
            {
                var sysLevel = _defaultSystemAudio.AudioMeterInformation.MasterPeakValue;
                _systemAudioVolumeBar.Width = sysLevel * 298;
                
                // Change color based on level
                if (sysLevel > 0.8)
                    _systemAudioVolumeBar.Fill = Brushes.Red;
                else if (sysLevel > 0.5)
                    _systemAudioVolumeBar.Fill = Brushes.Orange;
                else
                    _systemAudioVolumeBar.Fill = Brushes.DodgerBlue;
            }
        }
        catch
        {
            // Ignore errors in volume monitoring
        }
    }
    
    private float GetMicrophoneLevel()
    {
        try
        {
            return _selectedMicrophone?.AudioMeterInformation?.MasterPeakValue ?? 0f;
        }
        catch
        {
            return 0f;
        }
    }
    
    private void Window_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        // Clean up resources
        _volumeTimer?.Stop();
        _transcriptionTimer?.Stop();
        _waveIn?.StopRecording();
        _waveIn?.Dispose();
        _systemAudioCapture?.StopRecording();
        _systemAudioCapture?.Dispose();
        _httpClient?.Dispose();
        _deviceEnumerator?.Dispose();
    }

    private void App_DispatcherUnhandledException(object sender, System.Windows.Threading.DispatcherUnhandledExceptionEventArgs e)
    {
        MessageBox.Show($"An unhandled exception occurred: {e.Exception.Message}", "Oreja Error", MessageBoxButton.OK, MessageBoxImage.Error);
        e.Handled = true;
    }
} 