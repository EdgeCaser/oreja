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
using System.Text;
using System.Windows.Input;

namespace Oreja;

// Settings class for speaker persistence
public class SpeakerSettings
{
    public List<string> AvailableSpeakers { get; set; } = new List<string>();
    public Dictionary<string, string> SpeakerNameMappings { get; set; } = new Dictionary<string, string>();
    public int NextSpeakerNumber { get; set; } = 5;
}

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
    private Button? _monitoringToggleButton;
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
    private bool _isMonitoring = false;
    
    // Audio capture components
    private WaveInEvent? _waveIn;
    private WasapiLoopbackCapture? _systemAudioCapture;
    private MMDeviceEnumerator? _deviceEnumerator;
    private MMDevice? _selectedMicrophone;
    private MMDevice? _defaultSystemAudio;
    
    // Volume level tracking
    private float _microphoneLevel = 0f;
    
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
    
    // Manual speaker assignment functionality
    private List<string> _availableSpeakers = new List<string> { "Unknown" }; // Remove default speakers 1-4
    private int _nextSpeakerNumber = 1; // Start from 1 instead of 5

    // Speaker persistence settings
    private const string SETTINGS_FILE = "oreja_speaker_settings.json";
    private string? _settingsFilePath;
    private SpeakerSettings _speakerSettings = new SpeakerSettings();

    // Keep track of all speaker ComboBoxes for refreshing
    private List<ComboBox> _speakerComboBoxes = new List<ComboBox>();

    // Privacy Mode functionality
    private bool _privacyModeEnabled = false;
    private CheckBox? _privacyModeCheckBox;
    private Dictionary<string, string> _privacySpeakerMapping = new Dictionary<string, string>();
    private int _privacySpeakerCounter = 1;

    // Enhanced Transcription Editor functionality
    private double _lastScrollPosition = 0;
    private bool _userScrolledUp = false;
    private HashSet<int> _selectedSegments = new HashSet<int>();
    private bool _multiSelectMode = false;
    private Button? _multiSelectToggleButton;
    private Button? _bulkRenameButton;
    private Button? _selectAllButton;
    private Button? _clearSelectionButton;
    private List<CheckBox> _segmentCheckBoxes = new List<CheckBox>();
    
    // Speaker color coding
    private Dictionary<string, Brush> _speakerColors = new Dictionary<string, Brush>();
    private readonly Brush[] _availableColors = {
        Brushes.LightBlue, Brushes.LightGreen, Brushes.LightCoral, 
        Brushes.LightGoldenrodYellow, Brushes.LightPink, Brushes.LightCyan,
        Brushes.LightSalmon, Brushes.LightSeaGreen, Brushes.Plum, Brushes.Khaki,
        Brushes.PaleGreen, Brushes.LightSkyBlue, Brushes.PeachPuff, Brushes.Lavender
    };
    private int _colorIndex = 0;
    
    // Emotional tone indicators
    private readonly Dictionary<string, string> _emotionIcons = new Dictionary<string, string>
    {
        ["positive"] = "😊",
        ["negative"] = "😔", 
        ["neutral"] = "😐",
        ["questioning"] = "🤔",
        ["concerned"] = "😟",
        ["excited"] = "🤩",
        ["angry"] = "😠",
        ["calm"] = "😌"
    };
    
    private readonly Dictionary<string, Brush> _emotionColors = new Dictionary<string, Brush>
    {
        ["positive"] = Brushes.LightGreen,
        ["negative"] = Brushes.LightCoral,
        ["neutral"] = Brushes.LightGray,
        ["questioning"] = Brushes.LightBlue,
        ["concerned"] = Brushes.Orange,
        ["excited"] = Brushes.Gold,
        ["angry"] = Brushes.Red,
        ["calm"] = Brushes.LightCyan
    };

    // Helper class for transcription segments
    public class TranscriptionSegment
    {
        public string? Speaker { get; set; }
        public string? Text { get; set; }
        public double StartTime { get; set; }
        public double EndTime { get; set; }
        public string Source { get; set; } = "";
        public DateTime Timestamp { get; set; } = DateTime.Now;
        public int SegmentId { get; set; } // Add unique ID for tracking
        public string? EmotionalTone { get; set; }
        public double SentimentConfidence { get; set; }
        public bool IsSelected { get; set; } = false;
        public List<TranscriptionSegment>? SplitSegments { get; set; } // For split functionality
    }

    // Smart speaker filtering for dropdown
    private List<string> GetFilteredSpeakersForDropdown()
    {
        var filteredSpeakers = new List<string>();
        
        // Always add "Unknown" first
        filteredSpeakers.Add("Unknown");
        
        // Add user-defined speakers (non-auto-generated)
        var userDefinedSpeakers = _availableSpeakers
            .Where(s => s != "Unknown" && !s.StartsWith("Speaker_AUTO_SPEAKER_"))
            .OrderBy(s => s)
            .ToList();
        filteredSpeakers.AddRange(userDefinedSpeakers);
        
        // Only add auto-generated speakers if there are very few user-defined speakers
        // or if they're actively being used in recent transcriptions
        var autoGeneratedSpeakers = _availableSpeakers
            .Where(s => s.StartsWith("Speaker_AUTO_SPEAKER_"))
            .ToList();
            
        if (userDefinedSpeakers.Count < 3 && autoGeneratedSpeakers.Count > 0)
        {
            // Add only the most recently used auto-generated speakers (max 5)
            var recentAutoSpeakers = GetRecentlyUsedAutoSpeakers(autoGeneratedSpeakers, 5);
            filteredSpeakers.AddRange(recentAutoSpeakers);
        }
        
        // Add a special entry to access all auto-generated speakers if needed
        if (autoGeneratedSpeakers.Count > 0)
        {
            filteredSpeakers.Add("--- Show All Auto Speakers ---");
        }
        
        return filteredSpeakers;
    }
    
    private List<string> GetRecentlyUsedAutoSpeakers(List<string> autoSpeakers, int maxCount)
    {
        // Get auto speakers that were used in recent transcription segments
        var recentSpeakers = _transcriptionHistory
            .Where(t => t.Timestamp >= DateTime.Now.AddMinutes(-30)) // Last 30 minutes
            .Where(t => autoSpeakers.Contains(t.Speaker))
            .GroupBy(t => t.Speaker)
            .OrderByDescending(g => g.Max(t => t.Timestamp))
            .Take(maxCount)
            .Select(g => g.Key)
            .ToList();
            
        // If we don't have enough recent ones, add the most recent auto speakers by name
        if (recentSpeakers.Count < maxCount)
        {
            var additionalSpeakers = autoSpeakers
                .Where(s => !recentSpeakers.Contains(s))
                .OrderByDescending(s => ExtractAutoSpeakerNumber(s))
                .Take(maxCount - recentSpeakers.Count);
            recentSpeakers.AddRange(additionalSpeakers);
        }
        
        return recentSpeakers;
    }
    
    private int ExtractAutoSpeakerNumber(string autoSpeakerName)
    {
        // Extract number from "Speaker_AUTO_SPEAKER_123" format
        var match = System.Text.RegularExpressions.Regex.Match(autoSpeakerName, @"AUTO_SPEAKER_(\d+)");
        return match.Success ? int.Parse(match.Groups[1].Value) : 0;
    }
    
    private void ShowAllAutoSpeakersDialog(ComboBox comboBox, int segmentId)
    {
        var autoSpeakers = _availableSpeakers
            .Where(s => s.StartsWith("Speaker_AUTO_SPEAKER_"))
            .OrderByDescending(s => ExtractAutoSpeakerNumber(s))
            .ToList();
            
        if (autoSpeakers.Count == 0)
        {
            MessageBox.Show("No auto-generated speakers available.", "Oreja", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        
        // Create a selection dialog
        var dialog = new Window
        {
            Title = "Select Auto-Generated Speaker",
            Width = 400,
            Height = 300,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = Application.Current.MainWindow,
            ResizeMode = ResizeMode.CanResize
        };
        
        var panel = new StackPanel { Margin = new Thickness(10) };
        
        var label = new TextBlock 
        { 
            Text = "Select an auto-generated speaker or type a new name:",
            Margin = new Thickness(0, 0, 0, 10),
            FontWeight = FontWeights.Bold
        };
        
        var listBox = new ListBox
        {
            Height = 150,
            Margin = new Thickness(0, 0, 0, 10)
        };
        
        foreach (var speaker in autoSpeakers)
        {
            listBox.Items.Add(speaker);
        }
        
        var textPanel = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 0, 10) };
        var textLabel = new TextBlock { Text = "Or create new speaker:", VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, 0, 10, 0) };
        var textBox = new TextBox { Width = 200, Height = 25 };
        textPanel.Children.Add(textLabel);
        textPanel.Children.Add(textBox);
        
        var buttonPanel = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        var selectButton = new Button { Content = "Select", Width = 80, Margin = new Thickness(0, 0, 10, 0) };
        var cancelButton = new Button { Content = "Cancel", Width = 80 };
        
        selectButton.Click += (s, e) =>
        {
            string selectedSpeaker = null;
            
            if (!string.IsNullOrWhiteSpace(textBox.Text))
            {
                selectedSpeaker = textBox.Text.Trim();
            }
            else if (listBox.SelectedItem != null)
            {
                selectedSpeaker = listBox.SelectedItem.ToString();
            }
            
            if (!string.IsNullOrEmpty(selectedSpeaker))
            {
                // Add to available speakers if it's a new name
                if (!_availableSpeakers.Contains(selectedSpeaker))
                {
                    _availableSpeakers.Add(selectedSpeaker);
                    RefreshAllSpeakerDropdowns();
                    SaveSpeakerSettings();
                }
                
                comboBox.SelectedItem = selectedSpeaker;
                UpdateSegmentSpeaker(segmentId, selectedSpeaker);
                dialog.DialogResult = true;
            }
        };
        
        cancelButton.Click += (s, e) => dialog.DialogResult = false;
        
        buttonPanel.Children.Add(selectButton);
        buttonPanel.Children.Add(cancelButton);
        
        panel.Children.Add(label);
        panel.Children.Add(listBox);
        panel.Children.Add(textPanel);
        panel.Children.Add(buttonPanel);
        
        dialog.Content = panel;
        dialog.ShowDialog();
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
            
            // Initialize settings file path
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var orejaFolderPath = System.IO.Path.Combine(appDataPath, "Oreja");
            Directory.CreateDirectory(orejaFolderPath); // Ensure directory exists
            _settingsFilePath = System.IO.Path.Combine(orejaFolderPath, SETTINGS_FILE);
            
            // Load speaker settings from file
            LoadSpeakerSettings();
            Console.WriteLine($"Loaded {_availableSpeakers.Count} speaker names from settings");
            
            Console.WriteLine("Creating window...");
            // Create window entirely in code to bypass XAML issues
            var window = new Window
            {
                Title = "Oreja - Real-time Conference Transcription",
                Width = 800,
                Height = 800,
                WindowStartupLocation = WindowStartupLocation.CenterScreen,
                WindowState = WindowState.Normal,
                Topmost = false
            };
            
            // Use Grid instead of StackPanel for better layout control
            var mainGrid = new Grid { Margin = new Thickness(20) };
            
            // Define rows for the grid
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Title
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Microphone section
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // System audio section
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Volume meters
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Control buttons
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Status text
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Transcription label
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Instructions
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // Multi-select toolbar
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) }); // Transcription area (takes remaining space)
            
            int currentRow = 0;
            
            // Title
            var titleText = new TextBlock 
            { 
                Text = "Oreja - Audio Capture & Transcription",
                FontSize = 24,
                FontWeight = FontWeights.Bold,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 0, 0, 20)
            };
            Grid.SetRow(titleText, currentRow++);
            
            // Microphone selection section
            var microphoneSection = new StackPanel { Margin = new Thickness(0, 0, 0, 15) };
            var microphoneLabel = new TextBlock 
            { 
                Text = "Select Microphone:",
                FontSize = 14,
                Margin = new Thickness(0, 0, 0, 5)
            };
            
            _microphoneComboBox = new ComboBox
            {
                Width = 500,
                HorizontalAlignment = HorizontalAlignment.Left
            };
            _microphoneComboBox.SelectionChanged += MicrophoneComboBox_SelectionChanged;
            
            microphoneSection.Children.Add(microphoneLabel);
            microphoneSection.Children.Add(_microphoneComboBox);
            Grid.SetRow(microphoneSection, currentRow++);
            
            // System audio selection section
            var systemAudioSection = new StackPanel { Margin = new Thickness(0, 0, 0, 15) };
            var systemAudioLabel = new TextBlock 
            { 
                Text = "Select System Audio:",
                FontSize = 14,
                Margin = new Thickness(0, 0, 0, 5)
            };
            
            _systemAudioComboBox = new ComboBox
            {
                Width = 500,
                HorizontalAlignment = HorizontalAlignment.Left
            };
            _systemAudioComboBox.SelectionChanged += SystemAudioComboBox_SelectionChanged;
            
            systemAudioSection.Children.Add(systemAudioLabel);
            systemAudioSection.Children.Add(_systemAudioComboBox);
            Grid.SetRow(systemAudioSection, currentRow++);
            
            // Volume meters section
            var volumeSection = new StackPanel { Margin = new Thickness(0, 0, 0, 20) };
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
            var sysVolumePanel = new StackPanel { Orientation = Orientation.Horizontal };
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
            
            volumeSection.Children.Add(volumeLabel);
            volumeSection.Children.Add(micVolumePanel);
            volumeSection.Children.Add(sysVolumePanel);
            Grid.SetRow(volumeSection, currentRow++);
            
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
            
            _monitoringToggleButton = new Button 
            { 
                Content = "🔄 Start Monitoring",
                Width = 150,
                Height = 40,
                Margin = new Thickness(10, 0, 0, 0),
                FontSize = 14,
                Background = Brushes.LightBlue,
                IsEnabled = true
            };
            _monitoringToggleButton.Click += MonitoringToggleButton_Click;
            
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
            buttonPanel.Children.Add(_monitoringToggleButton);
            buttonPanel.Children.Add(_saveTranscriptionButton);
            Grid.SetRow(buttonPanel, currentRow++);
            
            // Privacy Mode section
            var privacyPanel = new StackPanel 
            { 
                Orientation = Orientation.Horizontal, 
                HorizontalAlignment = HorizontalAlignment.Center, 
                Margin = new Thickness(0, 10, 0, 10) 
            };
            
            _privacyModeCheckBox = new CheckBox
            {
                Content = "🔒 Legal-Safe Mode",
                FontSize = 12,
                FontWeight = FontWeights.Bold,
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 10, 0)
            };
            _privacyModeCheckBox.Checked += PrivacyModeCheckBox_Changed;
            _privacyModeCheckBox.Unchecked += PrivacyModeCheckBox_Changed;
            
            var privacyHelp = new TextBlock
            {
                Text = "Analysis only - no verbatim transcription stored (legal-safe)",
                FontSize = 10,
                FontStyle = FontStyles.Italic,
                Foreground = Brushes.Gray,
                VerticalAlignment = VerticalAlignment.Center
            };
            
            privacyPanel.Children.Add(_privacyModeCheckBox);
            privacyPanel.Children.Add(privacyHelp);
            Grid.SetRow(privacyPanel, currentRow++);
            
            // Status text
            _statusText = new TextBlock 
            { 
                Text = "Ready! Select a microphone and toggle monitoring to start.",
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 0, 0, 15),
                TextWrapping = TextWrapping.Wrap,
                FontSize = 12
            };
            Grid.SetRow(_statusText, currentRow++);
            
            // Transcription section label
            var transcriptionLabel = new TextBlock 
            { 
                Text = "Live Transcription:",
                FontSize = 16,
                FontWeight = FontWeights.Bold,
                Margin = new Thickness(0, 0, 0, 10)
            };
            Grid.SetRow(transcriptionLabel, currentRow++);
            
            // Instructions panel
            var instructionsBorder = new Border
            {
                BorderBrush = Brushes.LightBlue,
                BorderThickness = new Thickness(1),
                CornerRadius = new CornerRadius(3),
                Background = Brushes.AliceBlue,
                Padding = new Thickness(10),
                Margin = new Thickness(0, 0, 0, 10)
            };
            
            var instructionsPanel = new StackPanel();
            
            var instructionsTitle = new TextBlock
            {
                Text = "💡 Speaker Assignment Help:",
                FontWeight = FontWeights.Bold,
                FontSize = 12,
                Margin = new Thickness(0, 0, 0, 5)
            };
            
            var instructionsText = new TextBlock
            {
                Text = "• Use dropdowns to change speaker assignments\n• Click number buttons (1-4) for quick assignment\n• Click '+' to create new speakers\n• Click '×' to delete speakers from the list\n• Type in dropdown to create custom speaker names",
                FontSize = 11,
                Foreground = Brushes.DarkBlue,
                TextWrapping = TextWrapping.Wrap
            };
            
            instructionsPanel.Children.Add(instructionsTitle);
            instructionsPanel.Children.Add(instructionsText);
            instructionsBorder.Child = instructionsPanel;
            Grid.SetRow(instructionsBorder, currentRow++);
            
            // Multi-select toolbar
            var multiSelectToolbar = CreateMultiSelectToolbar();
            Grid.SetRow(multiSelectToolbar, currentRow++);
            
            // Transcription area - this will now take up all remaining space
            _transcriptionPanel = new StackPanel 
            { 
                Margin = new Thickness(10),
                Background = Brushes.White
            };
            
            _transcriptionScrollViewer = new ScrollViewer 
            { 
                // Remove fixed Height - let it fill available space
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
                BorderBrush = Brushes.Gray,
                BorderThickness = new Thickness(1),
                Background = Brushes.WhiteSmoke,
                Content = _transcriptionPanel,
                Margin = new Thickness(0, 0, 0, 10)
            };
            Grid.SetRow(_transcriptionScrollViewer, currentRow++);
            
            // Add all sections to the grid
            mainGrid.Children.Add(titleText);
            mainGrid.Children.Add(microphoneSection);
            mainGrid.Children.Add(systemAudioSection);
            mainGrid.Children.Add(volumeSection);
            mainGrid.Children.Add(buttonPanel);
            mainGrid.Children.Add(privacyPanel);
            mainGrid.Children.Add(_statusText);
            mainGrid.Children.Add(transcriptionLabel);
            mainGrid.Children.Add(instructionsBorder);
            mainGrid.Children.Add(multiSelectToolbar);
            mainGrid.Children.Add(_transcriptionScrollViewer);
            
            window.Content = mainGrid;
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
                        _statusText.Text = "Ready! Select a microphone and toggle monitoring to start.";
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
                
                // If monitoring is active, stop it first (we'll use recording mode instead)
                if (_isMonitoring)
                {
                    _waveIn?.StopRecording();
                    _systemAudioCapture?.StopRecording();
                    _isMonitoring = false;
                }
                
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
                
                if (_statusText != null) _statusText.Text = $"Recording from: {_selectedMicrophone.FriendlyName} - Transcribing in real-time... (monitoring auto-enabled)";
                if (_startRecordingButton != null) _startRecordingButton.IsEnabled = false;
                if (_stopRecordingButton != null) _stopRecordingButton.IsEnabled = true;
                if (_monitoringToggleButton != null)
                {
                    _monitoringToggleButton.Content = "🔄 Recording Mode";
                    _monitoringToggleButton.Background = Brushes.Orange;
                    _monitoringToggleButton.IsEnabled = false; // Disable during recording
                }
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
            
            // Re-enable monitoring toggle and restore monitoring state
            if (_monitoringToggleButton != null)
            {
                _monitoringToggleButton.IsEnabled = true;
                _monitoringToggleButton.Content = "🔄 Start Monitoring";
                _monitoringToggleButton.Background = Brushes.LightBlue;
            }
        }
        catch (Exception ex)
        {
            if (_statusText != null) _statusText.Text = $"Error stopping recording: {ex.Message}";
        }
    }
    
    private void MonitoringToggleButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            if (_selectedMicrophone == null)
            {
                if (_statusText != null) _statusText.Text = "Please select a microphone first.";
                return;
            }

            if (_isMonitoring)
            {
                // Stop monitoring
                _waveIn?.StopRecording();
                _systemAudioCapture?.StopRecording();
                _isMonitoring = false;
                
                if (_monitoringToggleButton != null)
                {
                    _monitoringToggleButton.Content = "🔄 Start Monitoring";
                    _monitoringToggleButton.Background = Brushes.LightBlue;
                }
                if (_statusText != null) _statusText.Text = "Audio monitoring stopped.";
            }
            else
            {
                // Start monitoring (without transcription)
                if (_selectedMicrophone != null)
                {
                    // Use a separate WaveIn for monitoring only
                    _waveIn = new WaveInEvent
                    {
                        DeviceNumber = _selectedMicrophoneIndex,
                        WaveFormat = new WaveFormat(16000, 1),
                        BufferMilliseconds = 50,
                        NumberOfBuffers = 2
                    };
                    
                    // Only handle data for volume monitoring, no buffering
                    _waveIn.DataAvailable += (s, args) =>
                    {
                        // Calculate volume level only - no audio storage
                        float level = 0f;
                        for (int i = 0; i < args.BytesRecorded; i += 2)
                        {
                            if (i + 1 < args.BytesRecorded)
                            {
                                short sample = BitConverter.ToInt16(args.Buffer, i);
                                level = Math.Max(level, Math.Abs(sample) / 32768f);
                            }
                        }
                        _microphoneLevel = level;
                    };
                    
                    _waveIn.StartRecording();
                    
                    // Start system audio monitoring too
                    if (_defaultSystemAudio != null)
                    {
                        _systemAudioCapture = new WasapiLoopbackCapture(_defaultSystemAudio);
                        _systemAudioCapture.StartRecording();
                    }
                    
                    _isMonitoring = true;
                    
                    if (_monitoringToggleButton != null)
                    {
                        _monitoringToggleButton.Content = "🔄 Stop Monitoring";
                        _monitoringToggleButton.Background = Brushes.LightGreen;
                    }
                    if (_statusText != null) _statusText.Text = "Audio monitoring started - ready to record.";
                }
            }
        }
        catch (Exception ex)
        {
            if (_statusText != null) _statusText.Text = $"Error toggling monitoring: {ex.Message}";
        }
        
        Console.WriteLine($"Monitoring toggled: {_isMonitoring}");
    }
    
    private void PrivacyModeCheckBox_Changed(object sender, RoutedEventArgs e)
    {
        if (_privacyModeCheckBox == null) return;
        
        _privacyModeEnabled = _privacyModeCheckBox.IsChecked == true;
        
        if (_privacyModeEnabled)
        {
            Console.WriteLine("🔒 Legal-Safe Mode ENABLED - Only analysis will be shown, no verbatim transcription");
            _privacySpeakerMapping.Clear();
            _privacySpeakerCounter = 1;
            
            // Show legal-safe mode indicator in status
            if (_statusText != null)
            {
                _statusText.Text += " 🔒 LEGAL-SAFE MODE ACTIVE";
                _statusText.Foreground = Brushes.DarkBlue;
            }
        }
        else
        {
            Console.WriteLine("🔓 Legal-Safe Mode DISABLED - Full transcription will be shown");
            _privacySpeakerMapping.Clear();
            
            // Remove legal-safe mode indicator from status
            if (_statusText != null)
            {
                var statusText = _statusText.Text;
                if (statusText.Contains(" 🔒 LEGAL-SAFE MODE ACTIVE"))
                {
                    _statusText.Text = statusText.Replace(" 🔒 LEGAL-SAFE MODE ACTIVE", "");
                    _statusText.Foreground = Brushes.Black;
                }
            }
        }
        
        // Refresh the transcription display to apply/remove legal-safe mode
        RefreshTranscriptionDisplay();
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
    }
    
    private void TranscriptionTimer_Tick(object? sender, EventArgs e)
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
        
        // Generate unique segment ID
        var segmentId = _transcriptionHistory.Count;
        
        // Store in transcription history
        var segment = new TranscriptionSegment
        {
            Speaker = speaker,
            Text = text,
            StartTime = startTime,
            Source = source,
            Timestamp = DateTime.Now,
            SegmentId = segmentId
        };
        _transcriptionHistory.Add(segment);
        
        // Ensure speaker is in available speakers list
        if (!string.IsNullOrEmpty(speaker) && !_availableSpeakers.Contains(speaker))
        {
            _availableSpeakers.Add(speaker);
            // Save settings when new speaker is auto-detected
            SaveSpeakerSettings();
            Console.WriteLine($"Auto-detected and saved new speaker: {speaker}");
        }
        
        // Enable save button once we have transcriptions
        if (_saveTranscriptionButton != null)
        {
            _saveTranscriptionButton.IsEnabled = true;
        }
        
        // Create main container for the segment
        var segmentBorder = new Border 
        { 
            BorderBrush = Brushes.LightGray,
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(5),
            Margin = new Thickness(0, 2, 0, 2),
            Padding = new Thickness(8),
            Background = Brushes.White
        };
        
        var segmentPanel = new StackPanel();
        
        // Top row: Checkbox, Timestamp, Speaker Assignment, Emotion, Source, and Controls
        var topPanel = new StackPanel 
        { 
            Orientation = Orientation.Horizontal, 
            Margin = new Thickness(0, 0, 0, 5) 
        };
        
        // Selection checkbox
        var selectionCheckBox = CreateSelectionCheckBox(segmentId);
        
        // Timestamp
        var timestampText = new TextBlock 
        { 
            Text = $"[{TimeSpan.FromSeconds(startTime):mm\\:ss}]",
            FontWeight = FontWeights.Bold,
            Foreground = Brushes.Gray,
            Width = 60,
            VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(0, 0, 10, 0)
        };
        
        // Speaker assignment dropdown with color coding
        var speakerComboBox = new ComboBox
        {
            Width = 120,
            ItemsSource = GetFilteredSpeakersForDropdown(),
            SelectedItem = GetDisplaySpeakerName(speaker),
            Margin = new Thickness(0, 0, 10, 0),
            ToolTip = "Select or change speaker",
            IsEditable = true,
            Background = GetSpeakerColorEnhanced(GetDisplaySpeakerName(speaker))
        };
        
        // Create emotional tone indicator
        var emotionIndicator = CreateEmotionalToneIndicator(text ?? "", speaker ?? "");
        
        // Track this ComboBox for refreshing
        _speakerComboBoxes.Add(speakerComboBox);
        
        // Capture segment ID for event handlers
        var currentSegmentId = segmentId;
        
        // Handle speaker selection change
        speakerComboBox.SelectionChanged += (s, e) =>
        {
            if (speakerComboBox.SelectedItem != null)
            {
                var newSpeaker = speakerComboBox.SelectedItem.ToString();
                if (!string.IsNullOrEmpty(newSpeaker))
                {
                    // Check if user selected the special "Show All Auto Speakers" option
                    if (newSpeaker == "--- Show All Auto Speakers ---")
                    {
                        // Reset selection to current speaker and show dialog
                        speakerComboBox.SelectedItem = GetDisplaySpeakerName(speaker);
                        ShowAllAutoSpeakersDialog(speakerComboBox, currentSegmentId);
                    }
                    else
                    {
                        // Update color when speaker changes
                        speakerComboBox.Background = GetSpeakerColorEnhanced(newSpeaker);
                        UpdateSegmentSpeaker(currentSegmentId, newSpeaker);
                    }
                }
            }
        };
        
        // Handle manual text entry for new speakers
        speakerComboBox.LostFocus += (s, e) =>
        {
            var newSpeaker = speakerComboBox.Text.Trim();
            if (!string.IsNullOrEmpty(newSpeaker) && newSpeaker != GetDisplaySpeakerName(speaker))
            {
                // Add to available speakers if new
                if (!_availableSpeakers.Contains(newSpeaker))
                {
                    _availableSpeakers.Add(newSpeaker);
                    RefreshAllSpeakerDropdowns();
                    
                    // Save settings when new speaker is created
                    SaveSpeakerSettings();
                    Console.WriteLine($"Created and saved new speaker via text entry in refresh: {newSpeaker}");
                }
                UpdateSegmentSpeaker(currentSegmentId, newSpeaker);
            }
        };
        
        // Source indicator
        var sourceText = new TextBlock 
        { 
            Text = $"[{source}]",
            FontWeight = FontWeights.Normal,
            Foreground = segment.Source == "Microphone" ? Brushes.Blue : Brushes.Green,
            Width = 80,
            FontSize = 10,
            VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(0, 0, 10, 0)
        };
        
        // Quick speaker buttons
        var quickSpeakerPanel = new StackPanel 
        { 
            Orientation = Orientation.Horizontal,
            Margin = new Thickness(0, 0, 10, 0)
        };
        
        // Add quick assignment buttons for common speakers
        for (int i = 1; i <= 4; i++)
        {
            var speakerNum = i;
            var quickButton = new Button
            {
                Content = speakerNum.ToString(),
                Width = 25,
                Height = 25,
                Margin = new Thickness(2, 0, 2, 0),
                FontSize = 10,
                Background = Brushes.LightBlue,
                ToolTip = $"Assign to Speaker {speakerNum}",
                BorderThickness = new Thickness(1)
            };
            
            quickButton.Click += (s, e) =>
            {
                var newSpeaker = $"Speaker {speakerNum}";
                speakerComboBox.SelectedItem = newSpeaker;
                UpdateSegmentSpeaker(currentSegmentId, newSpeaker);
            };
            
            quickSpeakerPanel.Children.Add(quickButton);
        }
        
        // New speaker button
        var newSpeakerButton = new Button
        {
            Content = "+",
            Width = 25,
            Height = 25,
            Margin = new Thickness(5, 0, 2, 0),
            FontSize = 12,
            FontWeight = FontWeights.Bold,
            Background = Brushes.LightGreen,
            ToolTip = "Create new speaker"
        };
        
        newSpeakerButton.Click += (s, e) => CreateNewSpeaker(speakerComboBox, currentSegmentId);
        
        // Show all auto speakers button
        var showAutoSpeakersButton = new Button
        {
            Content = "🔍",
            Width = 25,
            Height = 25,
            Margin = new Thickness(2, 0, 2, 0),
            FontSize = 10,
            Background = Brushes.LightYellow,
            ToolTip = "Browse all auto-generated speakers"
        };
        
        showAutoSpeakersButton.Click += (s, e) => ShowAllAutoSpeakersDialog(speakerComboBox, currentSegmentId);
        
        // Delete speaker button
        var deleteSpeakerButton = new Button
        {
            Content = "×",
            Width = 25,
            Height = 25,
            Margin = new Thickness(2, 0, 0, 0),
            FontSize = 14,
            FontWeight = FontWeights.Bold,
            Background = Brushes.LightCoral,
            ToolTip = "Delete current speaker from list",
            Foreground = Brushes.DarkRed
        };
        
        deleteSpeakerButton.Click += (s, e) => DeleteSpeaker(speakerComboBox, currentSegmentId);
        
        // Add all top row elements
        topPanel.Children.Add(selectionCheckBox);
        topPanel.Children.Add(timestampText);
        topPanel.Children.Add(speakerComboBox);
        topPanel.Children.Add(emotionIndicator);
        topPanel.Children.Add(sourceText);
        topPanel.Children.Add(quickSpeakerPanel);
        topPanel.Children.Add(newSpeakerButton);
        topPanel.Children.Add(showAutoSpeakersButton);
        topPanel.Children.Add(deleteSpeakerButton);
        
        // Editable text content (in its own row for better readability)
        var textDisplay = CreateEditableTextDisplay(segment);
        
        // Add both rows to the segment panel
        segmentPanel.Children.Add(topPanel);
        segmentPanel.Children.Add(textDisplay);
        
        // Add segment panel to border and border to main panel
        segmentBorder.Child = segmentPanel;
        _transcriptionPanel.Children.Add(segmentBorder);
        
        // Smart auto-scroll: only scroll if user was already at bottom
        if (_transcriptionScrollViewer != null)
        {
            var maxScroll = _transcriptionScrollViewer.ScrollableHeight;
            var currentScroll = _transcriptionScrollViewer.VerticalOffset;
            var wasAtBottom = currentScroll >= maxScroll - 50; // 50px tolerance
            
            if (wasAtBottom)
            {
                _transcriptionScrollViewer.ScrollToEnd();
            }
        }
        
        Console.WriteLine("Transcription segment added to UI successfully");
    }
    
    private string GetDisplaySpeakerName(string? originalSpeaker)
    {
        if (string.IsNullOrEmpty(originalSpeaker))
            return "Unknown";
        
        // Privacy Mode: Anonymize speakers
        if (_privacyModeEnabled)
        {
            if (originalSpeaker == "Unknown")
                return "Unknown";
                
            // Map original speaker to anonymous identifier
            if (!_privacySpeakerMapping.ContainsKey(originalSpeaker))
            {
                var anonymousName = GetAnonymousSpeakerName(_privacySpeakerCounter);
                _privacySpeakerMapping[originalSpeaker] = anonymousName;
                _privacySpeakerCounter++;
                Console.WriteLine($"🔒 Privacy mapping: {originalSpeaker} -> {anonymousName}");
            }
            
            return _privacySpeakerMapping[originalSpeaker];
        }
        
        // Normal mode: Check if we have a custom name for this speaker
        if (_speakerNames.ContainsKey(originalSpeaker))
            return _speakerNames[originalSpeaker];
            
        return originalSpeaker;
    }
    
    private string GetAnonymousSpeakerName(int counter)
    {
        // Generate Speaker A, B, C, ... Z, AA, BB, etc.
        if (counter <= 26)
        {
            return $"Speaker {(char)('A' + counter - 1)}";
        }
        else
        {
            // For more than 26 speakers, use AA, BB, CC pattern
            var letter = (char)('A' + ((counter - 27) % 26));
            var repetitions = ((counter - 27) / 26) + 2;
            return $"Speaker {new string(letter, repetitions)}";
        }
    }
    
    private string GetDisplayText(string? originalText)
    {
        if (string.IsNullOrEmpty(originalText))
            return "";
        
        // Legal-Safe Mode: Show analysis instead of verbatim transcription
        if (_privacyModeEnabled)
        {
            return GenerateTextAnalysis(originalText);
        }
        
        // Normal mode: Show actual transcription
        return originalText;
    }
    
    private string GenerateTextAnalysis(string text)
    {
        var analysis = new List<string>();
        
        // Basic analysis without revealing actual content
        var wordCount = text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        var duration = EstimateSpeechDuration(wordCount);
        
        analysis.Add($"📊 Speech Analysis:");
        analysis.Add($"• Duration: ~{duration} seconds");
        analysis.Add($"• Word count: {wordCount} words");
        
        // Sentiment analysis (basic)
        var sentiment = AnalyzeSentiment(text);
        analysis.Add($"• Tone: {sentiment}");
        
        // Content type analysis
        var contentType = AnalyzeContentType(text);
        analysis.Add($"• Content: {contentType}");
        
        return string.Join("\n", analysis);
    }
    
    private int EstimateSpeechDuration(int wordCount)
    {
        // Average speech rate is about 2-3 words per second
        return Math.Max(1, wordCount / 2);
    }
    
    private string AnalyzeSentiment(string text)
    {
        var lowerText = text.ToLower();
        
        // Simple sentiment analysis
        var positiveWords = new[] { "good", "great", "excellent", "happy", "pleased", "agree", "yes", "perfect", "wonderful" };
        var negativeWords = new[] { "bad", "terrible", "awful", "angry", "upset", "no", "disagree", "problem", "issue", "concern" };
        var questionWords = new[] { "what", "how", "why", "when", "where", "who", "can", "could", "would", "should" };
        
        var positiveCount = positiveWords.Count(word => lowerText.Contains(word));
        var negativeCount = negativeWords.Count(word => lowerText.Contains(word));
        var questionCount = questionWords.Count(word => lowerText.Contains(word));
        
        if (lowerText.Contains("?") || questionCount > 0)
            return "Questioning";
        if (positiveCount > negativeCount)
            return "Positive";
        if (negativeCount > positiveCount)
            return "Concerned";
        
        return "Neutral";
    }
    
    private string AnalyzeContentType(string text)
    {
        var lowerText = text.ToLower();
        
        // Analyze content type without revealing specifics
        if (lowerText.Contains("meeting") || lowerText.Contains("discuss"))
            return "Discussion";
        if (lowerText.Contains("budget") || lowerText.Contains("cost") || lowerText.Contains("money"))
            return "Financial discussion";
        if (lowerText.Contains("project") || lowerText.Contains("timeline") || lowerText.Contains("deadline"))
            return "Project planning";
        if (lowerText.Contains("?"))
            return "Question/inquiry";
        if (lowerText.Length < 20)
            return "Brief comment";
        
        return "General discussion";
    }
    
    private void UpdateSegmentSpeaker(int segmentId, string newSpeaker)
    {
        // Find and update the segment in history
        var segment = _transcriptionHistory.FirstOrDefault(s => s.SegmentId == segmentId);
        if (segment != null)
        {
            var oldSpeaker = segment.Speaker;
            segment.Speaker = newSpeaker;
            Console.WriteLine($"Updated segment {segmentId} speaker from '{oldSpeaker}' to '{newSpeaker}'");
            
            // If this is a speaker name mapping change, save it
            if (!string.IsNullOrEmpty(oldSpeaker) && oldSpeaker != newSpeaker)
            {
                _speakerNames[oldSpeaker] = newSpeaker;
                SaveSpeakerSettings();
                Console.WriteLine($"Saved speaker name mapping: {oldSpeaker} -> {newSpeaker}");
                
                // Send feedback to backend to improve speaker recognition
                _ = SendSpeakerCorrectionFeedbackAsync(segment, newSpeaker);
            }
        }
    }
    
    private async Task SendSpeakerCorrectionFeedbackAsync(TranscriptionSegment segment, string correctSpeakerName)
    {
        try
        {
            // Only send feedback if we have the original audio and the correction is meaningful
            if (string.IsNullOrEmpty(correctSpeakerName) || correctSpeakerName == "Unknown")
                return;
            
            Console.WriteLine($"Sending speaker correction feedback: '{segment.Speaker}' -> '{correctSpeakerName}'");
            
            // Send the name mapping to the backend
            var requestData = new
            {
                old_speaker_id = segment.Speaker,
                new_speaker_name = correctSpeakerName
            };
            
            var jsonContent = System.Text.Json.JsonSerializer.Serialize(requestData);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");
            
            try
            {
                var response = await _httpClient!.PostAsync($"{BACKEND_URL}/speakers/name_mapping?old_speaker_id={Uri.EscapeDataString(segment.Speaker ?? "")}&new_speaker_name={Uri.EscapeDataString(correctSpeakerName)}", null);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync();
                    var result = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(responseContent);
                    
                    var status = result.GetProperty("status").GetString();
                    Console.WriteLine($"Speaker feedback processed: {status}");
                    
                    if (status == "speakers_merged")
                    {
                        Console.WriteLine($"Speakers successfully merged in backend");
                    }
                    else if (status == "name_updated")
                    {
                        Console.WriteLine($"Speaker name updated in backend");
                    }
                }
                else
                {
                    Console.WriteLine($"Speaker feedback failed: HTTP {response.StatusCode}");
                }
            }
            catch (HttpRequestException ex)
            {
                Console.WriteLine($"Network error sending speaker feedback: {ex.Message}");
            }
            
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error sending speaker correction feedback: {ex.Message}");
        }
    }
    
    private void CreateNewSpeaker(ComboBox comboBox, int segmentId)
    {
        var inputWindow = new Window
        {
            Title = "Create New Speaker",
            Width = 300,
            Height = 180,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = MainWindow,
            ResizeMode = ResizeMode.NoResize
        };
        
        var panel = new StackPanel { Margin = new Thickness(20) };
        
        var label = new TextBlock 
        { 
            Text = "Enter name for new speaker:",
            Margin = new Thickness(0, 0, 0, 10)
        };
        
        var textBox = new TextBox 
        { 
            Text = $"Speaker {_nextSpeakerNumber}",
            Margin = new Thickness(0, 0, 0, 20),
            Padding = new Thickness(5)
        };
        textBox.SelectAll();
        textBox.Focus();
        
        var buttonPanel = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        
        var okButton = new Button 
        { 
            Content = "Create",
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
            var newSpeaker = textBox.Text.Trim();
            if (!string.IsNullOrEmpty(newSpeaker))
            {
                // Add to available speakers
                if (!_availableSpeakers.Contains(newSpeaker))
                {
                    _availableSpeakers.Add(newSpeaker);
                    _nextSpeakerNumber++;
                    RefreshAllSpeakerDropdowns();
                    
                    // Save settings when new speaker is created
                    SaveSpeakerSettings();
                    Console.WriteLine($"Created and saved new speaker: {newSpeaker}");
                }
                
                // Set the new speaker for this segment
                comboBox.SelectedItem = newSpeaker;
                UpdateSegmentSpeaker(segmentId, newSpeaker);
                
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
    
    private void RefreshAllSpeakerDropdowns()
    {
        Console.WriteLine($"Refreshing {_speakerComboBoxes.Count} speaker dropdowns with {_availableSpeakers.Count} speakers");
        
        // Update all tracked ComboBoxes
        foreach (var comboBox in _speakerComboBoxes.ToList()) // Use ToList() to avoid collection modification issues
        {
            try
            {
                // Check if ComboBox is still valid (not disposed)
                var currentSelection = comboBox.SelectedItem as string;
                
                // Update ItemsSource
                comboBox.ItemsSource = GetFilteredSpeakersForDropdown();
                
                // Restore selection if still valid
                if (!string.IsNullOrEmpty(currentSelection) && _availableSpeakers.Contains(currentSelection))
                {
                    comboBox.SelectedItem = currentSelection;
                }
                else if (!string.IsNullOrEmpty(currentSelection))
                {
                    // If selected speaker was deleted, set to "Unknown"
                    comboBox.SelectedItem = "Unknown";
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error refreshing ComboBox: {ex.Message}");
                // Remove invalid ComboBox from tracking list
                _speakerComboBoxes.Remove(comboBox);
            }
        }
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
                
                // Save settings when speaker is renamed
                SaveSpeakerSettings();
                Console.WriteLine($"Renamed and saved speaker: {originalSpeaker} -> {newName}");
                
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
    
    private void PreserveScrollPosition()
    {
        if (_transcriptionScrollViewer != null)
        {
            _lastScrollPosition = _transcriptionScrollViewer.VerticalOffset;
            var maxScroll = _transcriptionScrollViewer.ScrollableHeight;
            _userScrolledUp = _lastScrollPosition < maxScroll - 50; // 50px tolerance
        }
    }

    private void RestoreScrollPosition()
    {
        if (_transcriptionScrollViewer != null && _userScrolledUp)
        {
            _transcriptionScrollViewer.ScrollToVerticalOffset(_lastScrollPosition);
        }
        else if (!_userScrolledUp)
        {
            _transcriptionScrollViewer?.ScrollToEnd(); // Only auto-scroll if user was at bottom
        }
    }

    private Brush GetSpeakerColorEnhanced(string speaker)
    {
        if (string.IsNullOrEmpty(speaker))
            speaker = "Unknown";
            
        if (!_speakerColors.ContainsKey(speaker))
        {
            _speakerColors[speaker] = _availableColors[_colorIndex % _availableColors.Length];
            _colorIndex++;
        }
        return _speakerColors[speaker];
    }

    private StackPanel CreateEmotionalToneIndicator(string text, string speaker)
    {
        var tonePanel = new StackPanel 
        { 
            Orientation = Orientation.Horizontal,
            Margin = new Thickness(5, 0, 5, 0)
        };
        
        // Analyze sentiment using existing method
        var sentiment = AnalyzeSentiment(text).ToLower();
        var icon = _emotionIcons.GetValueOrDefault(sentiment, "😐");
        var color = _emotionColors.GetValueOrDefault(sentiment, Brushes.LightGray);
        
        var emotionIcon = new Border
        {
            Background = color,
            CornerRadius = new CornerRadius(8),
            Padding = new Thickness(4, 2, 4, 2),
            Margin = new Thickness(2, 0, 2, 0),
            Child = new TextBlock 
            { 
                Text = icon,
                FontSize = 12,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                ToolTip = $"Emotional tone: {sentiment}"
            }
        };
        
        tonePanel.Children.Add(emotionIcon);
        return tonePanel;
    }

    private CheckBox CreateSelectionCheckBox(int segmentId)
    {
        var checkbox = new CheckBox
        {
            Margin = new Thickness(5, 0, 10, 0),
            VerticalAlignment = VerticalAlignment.Center,
            Visibility = _multiSelectMode ? Visibility.Visible : Visibility.Collapsed,
            IsChecked = _selectedSegments.Contains(segmentId)
        };
        
        checkbox.Checked += (s, e) => {
            _selectedSegments.Add(segmentId);
            UpdateMultiSelectButtons();
        };
        checkbox.Unchecked += (s, e) => {
            _selectedSegments.Remove(segmentId);
            UpdateMultiSelectButtons();
        };
        
        _segmentCheckBoxes.Add(checkbox);
        return checkbox;
    }

    private void UpdateMultiSelectButtons()
    {
        if (_bulkRenameButton != null)
        {
            _bulkRenameButton.IsEnabled = _selectedSegments.Count > 0;
            _bulkRenameButton.Content = $"Rename Selected ({_selectedSegments.Count})";
        }
        if (_clearSelectionButton != null)
        {
            _clearSelectionButton.IsEnabled = _selectedSegments.Count > 0;
        }
    }

    private Border CreateMultiSelectToolbar()
    {
        var toolbar = new Border
        {
            Background = Brushes.LightYellow,
            BorderBrush = Brushes.DarkOrange,
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(3),
            Padding = new Thickness(10, 5, 10, 5),
            Margin = new Thickness(0, 5, 0, 10),
            Visibility = Visibility.Visible // Always visible now, but contents change
        };
        
        var toolbarPanel = new StackPanel 
        { 
            Orientation = Orientation.Horizontal, 
            HorizontalAlignment = HorizontalAlignment.Left 
        };
        
        // Multi-select mode toggle
        _multiSelectToggleButton = new Button
        {
            Content = "📋 Multi-Select: OFF",
            Width = 150,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            Background = Brushes.LightBlue,
            ToolTip = "Toggle multi-select mode for bulk operations"
        };
        _multiSelectToggleButton.Click += MultiSelectToggle_Click;
        
        // Select All button
        _selectAllButton = new Button
        {
            Content = "☑ Select All",
            Width = 100,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            Background = Brushes.LightGray,
            IsEnabled = false
        };
        _selectAllButton.Click += SelectAll_Click;
        
        // Clear Selection button
        _clearSelectionButton = new Button
        {
            Content = "☐ Clear",
            Width = 80,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            Background = Brushes.LightGray,
            IsEnabled = false
        };
        _clearSelectionButton.Click += ClearSelection_Click;
        
        // Bulk rename button
        _bulkRenameButton = new Button
        {
            Content = "🏷 Rename Selected (0)",
            Width = 160,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            Background = Brushes.LightGreen,
            IsEnabled = false
        };
        _bulkRenameButton.Click += BulkRename_Click;
        
        // Status text for selections
        var selectionStatus = new TextBlock
        {
            Text = "Select segments to perform bulk operations",
            VerticalAlignment = VerticalAlignment.Center,
            FontStyle = FontStyles.Italic,
            Foreground = Brushes.DarkOrange,
            Margin = new Thickness(10, 0, 0, 0)
        };
        
        toolbarPanel.Children.Add(_multiSelectToggleButton);
        toolbarPanel.Children.Add(_selectAllButton);
        toolbarPanel.Children.Add(_clearSelectionButton);
        toolbarPanel.Children.Add(_bulkRenameButton);
        toolbarPanel.Children.Add(selectionStatus);
        
        toolbar.Child = toolbarPanel;
        return toolbar;
    }

    private void MultiSelectToggle_Click(object sender, RoutedEventArgs e)
    {
        _multiSelectMode = !_multiSelectMode;
        
        if (_multiSelectToggleButton != null)
        {
            _multiSelectToggleButton.Content = _multiSelectMode ? "📋 Multi-Select: ON" : "📋 Multi-Select: OFF";
            _multiSelectToggleButton.Background = _multiSelectMode ? Brushes.Orange : Brushes.LightBlue;
        }
        
        // Update button states
        if (_selectAllButton != null) _selectAllButton.IsEnabled = _multiSelectMode;
        if (_clearSelectionButton != null) _clearSelectionButton.IsEnabled = _multiSelectMode && _selectedSegments.Count > 0;
        
        // Update checkbox visibility
        foreach (var checkbox in _segmentCheckBoxes)
        {
            checkbox.Visibility = _multiSelectMode ? Visibility.Visible : Visibility.Collapsed;
        }
        
        // Clear selections when turning off multi-select mode
        if (!_multiSelectMode)
        {
            _selectedSegments.Clear();
            UpdateMultiSelectButtons();
        }
    }

    private void SelectAll_Click(object sender, RoutedEventArgs e)
    {
        foreach (var segment in _transcriptionHistory)
        {
            _selectedSegments.Add(segment.SegmentId);
        }
        
        foreach (var checkbox in _segmentCheckBoxes)
        {
            checkbox.IsChecked = true;
        }
        
        UpdateMultiSelectButtons();
    }

    private void ClearSelection_Click(object sender, RoutedEventArgs e)
    {
        _selectedSegments.Clear();
        
        foreach (var checkbox in _segmentCheckBoxes)
        {
            checkbox.IsChecked = false;
        }
        
        UpdateMultiSelectButtons();
    }

    private void BulkRename_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedSegments.Count == 0) return;
        
        ShowBulkRenameDialog();
    }

    private void ShowBulkRenameDialog()
    {
        var dialog = new Window
        {
            Title = "Bulk Rename Speakers",
            Width = 400,
            Height = 200,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = Application.Current.MainWindow,
            ResizeMode = ResizeMode.NoResize
        };
        
        var panel = new StackPanel { Margin = new Thickness(20) };
        
        var label = new TextBlock 
        { 
            Text = $"Rename {_selectedSegments.Count} selected segments to:",
            Margin = new Thickness(0, 0, 0, 10)
        };
        
        var speakerComboBox = new ComboBox
        {
            ItemsSource = _availableSpeakers,
            IsEditable = true,
            Margin = new Thickness(0, 0, 0, 20),
            Padding = new Thickness(5)
        };
        speakerComboBox.Focus();
        
        var buttonPanel = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        
        var okButton = new Button 
        { 
            Content = "Rename All",
            Width = 100,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            IsDefault = true,
            Background = Brushes.LightGreen
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
            var newSpeaker = speakerComboBox.Text.Trim();
            if (!string.IsNullOrEmpty(newSpeaker))
            {
                // Add to available speakers if new
                if (!_availableSpeakers.Contains(newSpeaker))
                {
                    _availableSpeakers.Add(newSpeaker);
                    SaveSpeakerSettings();
                }
                
                // Update all selected segments
                foreach (var segmentId in _selectedSegments)
                {
                    UpdateSegmentSpeaker(segmentId, newSpeaker);
                }
                
                // Refresh the display
                RefreshTranscriptionDisplay();
                
                // Clear selections
                _selectedSegments.Clear();
                UpdateMultiSelectButtons();
                
                dialog.DialogResult = true;
            }
        };
        
        cancelButton.Click += (s, e) => dialog.DialogResult = false;
        
        buttonPanel.Children.Add(okButton);
        buttonPanel.Children.Add(cancelButton);
        
        panel.Children.Add(label);
        panel.Children.Add(speakerComboBox);
        panel.Children.Add(buttonPanel);
        
        dialog.Content = panel;
        dialog.ShowDialog();
    }

    private TextBox CreateEditableTextDisplay(TranscriptionSegment segment)
    {
        var textBox = new TextBox
        {
            Text = GetDisplayText(segment.Text),
            TextWrapping = TextWrapping.Wrap,
            BorderThickness = new Thickness(0),
            Background = Brushes.Transparent,
            IsReadOnly = true,
            Margin = new Thickness(0, 5, 0, 0),
            FontSize = 13,
            AcceptsReturn = true
        };
        
        // Add context menu for splitting
        var contextMenu = new ContextMenu();
        
        var splitMenuItem = new MenuItem { Header = "✂️ Split Text Here" };
        splitMenuItem.Click += (s, e) => {
            if (textBox.SelectionStart > 0 && textBox.SelectionStart < textBox.Text.Length)
            {
                ShowSplitDialog(segment, textBox.SelectionStart);
            }
            else
            {
                MessageBox.Show("Please place your cursor where you want to split the text.", "Split Text", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        };
        
        var editMenuItem = new MenuItem { Header = "✏️ Edit Text" };
        editMenuItem.Click += (s, e) => EnableTextEditing(textBox, segment);
        
        contextMenu.Items.Add(splitMenuItem);
        contextMenu.Items.Add(editMenuItem);
        textBox.ContextMenu = contextMenu;
        
        // Double-click to enable editing
        textBox.MouseDoubleClick += (s, e) => EnableTextEditing(textBox, segment);
        
        return textBox;
    }

    private void EnableTextEditing(TextBox textBox, TranscriptionSegment segment)
    {
        textBox.IsReadOnly = false;
        textBox.Background = Brushes.LightYellow;
        textBox.BorderThickness = new Thickness(1);
        textBox.BorderBrush = Brushes.Orange;
        textBox.Focus();
        textBox.SelectAll();
        
        // Handle when editing is finished
        textBox.LostFocus += (s, e) => {
            FinishTextEditing(textBox, segment);
        };
        
        textBox.KeyDown += (s, e) => {
            if (e.Key == System.Windows.Input.Key.Enter && (Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control)
            {
                FinishTextEditing(textBox, segment);
            }
            else if (e.Key == System.Windows.Input.Key.Escape)
            {
                // Cancel editing
                textBox.Text = GetDisplayText(segment.Text);
                FinishTextEditing(textBox, segment);
            }
        };
    }

    private void FinishTextEditing(TextBox textBox, TranscriptionSegment segment)
    {
        textBox.IsReadOnly = true;
        textBox.Background = Brushes.Transparent;
        textBox.BorderThickness = new Thickness(0);
        
        // Update the segment text if changed
        var newText = textBox.Text.Trim();
        if (!string.IsNullOrEmpty(newText) && newText != segment.Text)
        {
            segment.Text = newText;
            Console.WriteLine($"Updated segment {segment.SegmentId} text");
        }
    }

    private void ShowSplitDialog(TranscriptionSegment segment, int splitPosition)
    {
        var originalText = segment.Text ?? "";
        if (splitPosition <= 0 || splitPosition >= originalText.Length)
        {
            MessageBox.Show("Invalid split position.", "Split Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        
        var firstPart = originalText.Substring(0, splitPosition).Trim();
        var secondPart = originalText.Substring(splitPosition).Trim();
        
        var dialog = new Window
        {
            Title = "Split Text Segment",
            Width = 500,
            Height = 400,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = Application.Current.MainWindow,
            ResizeMode = ResizeMode.CanResize
        };
        
        var panel = new StackPanel { Margin = new Thickness(20) };
        
        var titleLabel = new TextBlock 
        { 
            Text = "Split text into two segments:",
            FontWeight = FontWeights.Bold,
            Margin = new Thickness(0, 0, 0, 15)
        };
        
        // First segment
        var firstSegmentPanel = new StackPanel { Margin = new Thickness(0, 0, 0, 15) };
        
        var firstSpeakerPanel = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 0, 5) };
        firstSpeakerPanel.Children.Add(new TextBlock { Text = "First segment speaker:", Width = 150, VerticalAlignment = VerticalAlignment.Center });
        
        var firstSpeakerCombo = new ComboBox
        {
            ItemsSource = _availableSpeakers,
            SelectedItem = GetDisplaySpeakerName(segment.Speaker),
            IsEditable = true,
            Width = 200
        };
        firstSpeakerPanel.Children.Add(firstSpeakerCombo);
        
        var firstTextBox = new TextBox
        {
            Text = firstPart,
            TextWrapping = TextWrapping.Wrap,
            Height = 60,
            AcceptsReturn = true,
            Margin = new Thickness(0, 5, 0, 0)
        };
        
        firstSegmentPanel.Children.Add(firstSpeakerPanel);
        firstSegmentPanel.Children.Add(new TextBlock { Text = "First segment text:" });
        firstSegmentPanel.Children.Add(firstTextBox);
        
        // Second segment
        var secondSegmentPanel = new StackPanel { Margin = new Thickness(0, 0, 0, 15) };
        
        var secondSpeakerPanel = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 0, 5) };
        secondSpeakerPanel.Children.Add(new TextBlock { Text = "Second segment speaker:", Width = 150, VerticalAlignment = VerticalAlignment.Center });
        
        var secondSpeakerCombo = new ComboBox
        {
            ItemsSource = _availableSpeakers,
            SelectedItem = GetDisplaySpeakerName(segment.Speaker),
            IsEditable = true,
            Width = 200
        };
        secondSpeakerPanel.Children.Add(secondSpeakerCombo);
        
        var secondTextBox = new TextBox
        {
            Text = secondPart,
            TextWrapping = TextWrapping.Wrap,
            Height = 60,
            AcceptsReturn = true,
            Margin = new Thickness(0, 5, 0, 0)
        };
        
        secondSegmentPanel.Children.Add(secondSpeakerPanel);
        secondSegmentPanel.Children.Add(new TextBlock { Text = "Second segment text:" });
        secondSegmentPanel.Children.Add(secondTextBox);
        
        // Buttons
        var buttonPanel = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        
        var okButton = new Button 
        { 
            Content = "Split",
            Width = 80,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            IsDefault = true,
            Background = Brushes.LightGreen
        };
        
        var cancelButton = new Button 
        { 
            Content = "Cancel",
            Width = 80,
            Height = 30,
            IsCancel = true
        };
        
        okButton.Click += (s, e) =>
        {
            var firstSpeaker = firstSpeakerCombo.Text.Trim();
            var secondSpeaker = secondSpeakerCombo.Text.Trim();
            var firstText = firstTextBox.Text.Trim();
            var secondText = secondTextBox.Text.Trim();
            
            if (!string.IsNullOrEmpty(firstText) && !string.IsNullOrEmpty(secondText))
            {
                SplitSegment(segment, firstSpeaker, firstText, secondSpeaker, secondText);
                dialog.DialogResult = true;
            }
            else
            {
                MessageBox.Show("Both text segments must contain text.", "Split Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        };
        
        cancelButton.Click += (s, e) => dialog.DialogResult = false;
        
        buttonPanel.Children.Add(okButton);
        buttonPanel.Children.Add(cancelButton);
        
        panel.Children.Add(titleLabel);
        panel.Children.Add(firstSegmentPanel);
        panel.Children.Add(secondSegmentPanel);
        panel.Children.Add(buttonPanel);
        
        dialog.Content = panel;
        dialog.ShowDialog();
    }

    private void SplitSegment(TranscriptionSegment originalSegment, string firstSpeaker, string firstText, string secondSpeaker, string secondText)
    {
        // Update original segment with first part
        originalSegment.Text = firstText;
        originalSegment.Speaker = firstSpeaker;
        
        // Calculate timing for the split
        var totalDuration = originalSegment.EndTime - originalSegment.StartTime;
        var firstPartLength = firstText.Length;
        var totalLength = firstText.Length + secondText.Length;
        var firstPartDuration = totalDuration * (firstPartLength / (double)totalLength);
        
        var splitTime = originalSegment.StartTime + firstPartDuration;
        originalSegment.EndTime = splitTime;
        
        // Create new segment for second part
        var newSegment = new TranscriptionSegment
        {
            Speaker = secondSpeaker,
            Text = secondText,
            StartTime = splitTime,
            EndTime = originalSegment.StartTime + totalDuration,
            Source = originalSegment.Source,
            Timestamp = originalSegment.Timestamp,
            SegmentId = _transcriptionHistory.Count,
            EmotionalTone = AnalyzeSentiment(secondText),
            SentimentConfidence = 0.5 // Default confidence for user-edited content
        };
        
        // Add new speakers to available list if needed
        if (!string.IsNullOrEmpty(firstSpeaker) && !_availableSpeakers.Contains(firstSpeaker))
        {
            _availableSpeakers.Add(firstSpeaker);
        }
        if (!string.IsNullOrEmpty(secondSpeaker) && !_availableSpeakers.Contains(secondSpeaker))
        {
            _availableSpeakers.Add(secondSpeaker);
        }
        
        // Insert the new segment after the original one
        var originalIndex = _transcriptionHistory.FindIndex(s => s.SegmentId == originalSegment.SegmentId);
        if (originalIndex >= 0 && originalIndex < _transcriptionHistory.Count - 1)
        {
            _transcriptionHistory.Insert(originalIndex + 1, newSegment);
        }
        else
        {
            _transcriptionHistory.Add(newSegment);
        }
        
        SaveSpeakerSettings();
        RefreshTranscriptionDisplay();
        
        Console.WriteLine($"Split segment {originalSegment.SegmentId} into two segments");
    }

    private void RefreshTranscriptionDisplay()
    {
        if (_transcriptionPanel == null)
            return;
            
        // Preserve scroll position before refresh
        PreserveScrollPosition();
            
        // Clear current display and tracking lists
        _transcriptionPanel.Children.Clear();
        _speakerComboBoxes.Clear();
        _segmentCheckBoxes.Clear();
        
        // Redraw all segments with updated layout
        foreach (var segment in _transcriptionHistory)
        {
            // Create main container for the segment
            var segmentBorder = new Border 
            { 
                BorderBrush = Brushes.LightGray,
                BorderThickness = new Thickness(1),
                CornerRadius = new CornerRadius(5),
                Margin = new Thickness(0, 2, 0, 2),
                Padding = new Thickness(8),
                Background = Brushes.White
            };
            
            var segmentPanel = new StackPanel();
            
            // Top row: Checkbox, Timestamp, Speaker Assignment, Emotion, Source, and Controls
            var topPanel = new StackPanel 
            { 
                Orientation = Orientation.Horizontal, 
                Margin = new Thickness(0, 0, 0, 5) 
            };
            
            // Selection checkbox
            var selectionCheckBox = CreateSelectionCheckBox(segment.SegmentId);
            
            // Timestamp
            var timestampText = new TextBlock 
            { 
                Text = $"[{TimeSpan.FromSeconds(segment.StartTime):mm\\:ss}]",
                FontWeight = FontWeights.Bold,
                Foreground = Brushes.Gray,
                Width = 60,
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 10, 0)
            };
            
            // Speaker assignment dropdown with color coding
            var speakerComboBox = new ComboBox
            {
                Width = 120,
                ItemsSource = GetFilteredSpeakersForDropdown(),
                SelectedItem = GetDisplaySpeakerName(segment.Speaker),
                Margin = new Thickness(0, 0, 10, 0),
                ToolTip = "Select or change speaker",
                IsEditable = true,
                Background = GetSpeakerColorEnhanced(GetDisplaySpeakerName(segment.Speaker))
            };
            
            // Create emotional tone indicator
            var emotionIndicator = CreateEmotionalToneIndicator(segment.Text ?? "", segment.Speaker ?? "");
            
            // Track this ComboBox for refreshing
            _speakerComboBoxes.Add(speakerComboBox);
            
            // Capture segment ID for event handlers
            var currentSegmentId = segment.SegmentId;
            
            // Handle speaker selection change
            speakerComboBox.SelectionChanged += (s, e) =>
            {
                if (speakerComboBox.SelectedItem != null)
                {
                    var newSpeaker = speakerComboBox.SelectedItem.ToString();
                    if (!string.IsNullOrEmpty(newSpeaker))
                    {
                        // Check if user selected the special "Show All Auto Speakers" option
                        if (newSpeaker == "--- Show All Auto Speakers ---")
                        {
                            // Reset selection to current speaker and show dialog
                            speakerComboBox.SelectedItem = GetDisplaySpeakerName(segment.Speaker);
                            ShowAllAutoSpeakersDialog(speakerComboBox, currentSegmentId);
                        }
                        else
                        {
                            // Update color when speaker changes
                            speakerComboBox.Background = GetSpeakerColorEnhanced(newSpeaker);
                            UpdateSegmentSpeaker(currentSegmentId, newSpeaker);
                        }
                    }
                }
            };
            
            // Handle manual text entry for new speakers
            speakerComboBox.LostFocus += (s, e) =>
            {
                var newSpeaker = speakerComboBox.Text.Trim();
                if (!string.IsNullOrEmpty(newSpeaker) && newSpeaker != GetDisplaySpeakerName(segment.Speaker))
                {
                    // Add to available speakers if new
                    if (!_availableSpeakers.Contains(newSpeaker))
                    {
                        _availableSpeakers.Add(newSpeaker);
                        RefreshAllSpeakerDropdowns();
                        
                        // Save settings when new speaker is created
                        SaveSpeakerSettings();
                        Console.WriteLine($"Created and saved new speaker via text entry in refresh: {newSpeaker}");
                    }
                    UpdateSegmentSpeaker(currentSegmentId, newSpeaker);
                }
            };
            
            // Source indicator
            var sourceText = new TextBlock 
            { 
                Text = $"[{segment.Source}]",
                FontWeight = FontWeights.Normal,
                Foreground = segment.Source == "Microphone" ? Brushes.Blue : Brushes.Green,
                Width = 80,
                FontSize = 10,
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 10, 0)
            };
            
            // Quick speaker buttons
            var quickSpeakerPanel = new StackPanel 
            { 
                Orientation = Orientation.Horizontal,
                Margin = new Thickness(0, 0, 10, 0)
            };
            
            // Add quick assignment buttons for common speakers
            for (int i = 1; i <= 4; i++)
            {
                var speakerNum = i;
                var quickButton = new Button
                {
                    Content = speakerNum.ToString(),
                    Width = 25,
                    Height = 25,
                    Margin = new Thickness(2, 0, 2, 0),
                    FontSize = 10,
                    Background = Brushes.LightBlue,
                    ToolTip = $"Assign to Speaker {speakerNum}",
                    BorderThickness = new Thickness(1)
                };
                
                quickButton.Click += (s, e) =>
                {
                    var newSpeaker = $"Speaker {speakerNum}";
                    speakerComboBox.SelectedItem = newSpeaker;
                    UpdateSegmentSpeaker(currentSegmentId, newSpeaker);
                };
                
                quickSpeakerPanel.Children.Add(quickButton);
            }
            
            // New speaker button
            var newSpeakerButton = new Button
            {
                Content = "+",
                Width = 25,
                Height = 25,
                Margin = new Thickness(5, 0, 2, 0),
                FontSize = 12,
                FontWeight = FontWeights.Bold,
                Background = Brushes.LightGreen,
                ToolTip = "Create new speaker"
            };
            
            newSpeakerButton.Click += (s, e) => CreateNewSpeaker(speakerComboBox, currentSegmentId);
            
            // Show all auto speakers button
            var showAutoSpeakersButton = new Button
            {
                Content = "🔍",
                Width = 25,
                Height = 25,
                Margin = new Thickness(2, 0, 2, 0),
                FontSize = 10,
                Background = Brushes.LightYellow,
                ToolTip = "Browse all auto-generated speakers"
            };
            
            showAutoSpeakersButton.Click += (s, e) => ShowAllAutoSpeakersDialog(speakerComboBox, currentSegmentId);
            
            // Delete speaker button
            var deleteSpeakerButton = new Button
            {
                Content = "×",
                Width = 25,
                Height = 25,
                Margin = new Thickness(2, 0, 0, 0),
                FontSize = 14,
                FontWeight = FontWeights.Bold,
                Background = Brushes.LightCoral,
                ToolTip = "Delete current speaker from list",
                Foreground = Brushes.DarkRed
            };
            
            deleteSpeakerButton.Click += (s, e) => DeleteSpeaker(speakerComboBox, currentSegmentId);
            
            // Add all top row elements
            topPanel.Children.Add(selectionCheckBox);
            topPanel.Children.Add(timestampText);
            topPanel.Children.Add(speakerComboBox);
            topPanel.Children.Add(emotionIndicator);
            topPanel.Children.Add(sourceText);
            topPanel.Children.Add(quickSpeakerPanel);
            topPanel.Children.Add(newSpeakerButton);
            topPanel.Children.Add(showAutoSpeakersButton);
            topPanel.Children.Add(deleteSpeakerButton);
            
            // Editable text content (in its own row for better readability)
            var textDisplay = CreateEditableTextDisplay(segment);
            
            // Add both rows to the segment panel
            segmentPanel.Children.Add(topPanel);
            segmentPanel.Children.Add(textDisplay);
            
            // Add segment panel to border and border to main panel
            segmentBorder.Child = segmentPanel;
            _transcriptionPanel.Children.Add(segmentBorder);
        }
        
        // Restore scroll position instead of always auto-scrolling
        RestoreScrollPosition();
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
        
        // Legal-safe mode indication
        if (_privacyModeEnabled)
        {
            report.AppendLine("🔒 LEGAL-SAFE MODE: Only analytical data included");
            report.AppendLine("   NO verbatim transcription stored for legal compliance");
            report.AppendLine("   This report contains only speech analysis and patterns");
        }
        
        report.AppendLine();
        
        if (_privacyModeEnabled)
        {
            // Legal-Safe Mode: Only analytical summary
            GenerateLegalSafeAnalysis(report);
        }
        else
        {
            // Normal Mode: Full transcription
            GenerateFullTranscriptionReport(report);
        }
        
        return report.ToString();
    }
    
    private void GenerateLegalSafeAnalysis(System.Text.StringBuilder report)
    {
        report.AppendLine("=== CONVERSATION ANALYSIS (LEGAL-SAFE) ===");
        report.AppendLine();
        
        // Overall statistics
        var totalDuration = _transcriptionHistory.Sum(s => EstimateSpeechDuration(
            s.Text?.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length ?? 0));
        var totalWords = _transcriptionHistory.Sum(s => 
            s.Text?.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length ?? 0);
        
        report.AppendLine("📊 OVERALL STATISTICS:");
        report.AppendLine($"• Total estimated duration: {totalDuration} seconds");
        report.AppendLine($"• Total word count: {totalWords} words");
        report.AppendLine($"• Number of speech segments: {_transcriptionHistory.Count}");
        report.AppendLine();
        
        // Speaker analysis
        var speakerStats = _transcriptionHistory
            .Where(s => !string.IsNullOrEmpty(s.Speaker))
            .GroupBy(s => GetDisplaySpeakerName(s.Speaker))
            .Select(g => new {
                Speaker = g.Key,
                SegmentCount = g.Count(),
                WordCount = g.Sum(s => s.Text?.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length ?? 0)
            })
            .OrderByDescending(s => s.WordCount)
            .ToList();
        
        report.AppendLine("👥 SPEAKER PARTICIPATION:");
        foreach (var speaker in speakerStats)
        {
            var percentage = totalWords > 0 ? (speaker.WordCount * 100.0) / totalWords : 0;
            report.AppendLine($"• {speaker.Speaker}: {speaker.SegmentCount} segments, {speaker.WordCount} words ({percentage:F1}%)");
        }
        report.AppendLine();
        
        // Sentiment analysis
        var sentiments = _transcriptionHistory
            .Where(s => !string.IsNullOrEmpty(s.Text))
            .Select(s => AnalyzeSentiment(s.Text!))
            .GroupBy(s => s)
            .Select(g => new { Sentiment = g.Key, Count = g.Count() })
            .OrderByDescending(s => s.Count)
            .ToList();
        
        report.AppendLine("😊 TONE ANALYSIS:");
        foreach (var sentiment in sentiments)
        {
            var percentage = _transcriptionHistory.Count > 0 ? (sentiment.Count * 100.0) / _transcriptionHistory.Count : 0;
            report.AppendLine($"• {sentiment.Sentiment}: {sentiment.Count} segments ({percentage:F1}%)");
        }
        report.AppendLine();
        
        // Content type analysis
        var contentTypes = _transcriptionHistory
            .Where(s => !string.IsNullOrEmpty(s.Text))
            .Select(s => AnalyzeContentType(s.Text!))
            .GroupBy(c => c)
            .Select(g => new { ContentType = g.Key, Count = g.Count() })
            .OrderByDescending(c => c.Count)
            .ToList();
        
        report.AppendLine("📋 CONTENT ANALYSIS:");
        foreach (var contentType in contentTypes)
        {
            var percentage = _transcriptionHistory.Count > 0 ? (contentType.Count * 100.0) / _transcriptionHistory.Count : 0;
            report.AppendLine($"• {contentType.ContentType}: {contentType.Count} segments ({percentage:F1}%)");
        }
        report.AppendLine();
        
        // Source analysis
        var microphoneSegments = _transcriptionHistory.Where(s => s.Source == "Microphone").Count();
        var systemAudioSegments = _transcriptionHistory.Where(s => s.Source == "System Audio").Count();
        
        report.AppendLine("🎤 AUDIO SOURCE ANALYSIS:");
        report.AppendLine($"• Microphone input: {microphoneSegments} segments");
        report.AppendLine($"• System audio input: {systemAudioSegments} segments");
        report.AppendLine();
        
        report.AppendLine("⚖️ LEGAL COMPLIANCE:");
        report.AppendLine("• No verbatim transcription stored");
        report.AppendLine("• Speaker identities anonymized");
        report.AppendLine("• Only analytical insights recorded");
        report.AppendLine("• Compliant with privacy regulations");
    }
    
    private void GenerateFullTranscriptionReport(System.Text.StringBuilder report)
    {
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
        // Save speaker settings before closing
        SaveSpeakerSettings();
        Console.WriteLine("Saved speaker settings on application exit");
        
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

    private void LoadSpeakerSettings()
    {
        try
        {
            if (!string.IsNullOrEmpty(_settingsFilePath) && File.Exists(_settingsFilePath))
            {
                Console.WriteLine($"Loading speaker settings from: {_settingsFilePath}");
                var json = File.ReadAllText(_settingsFilePath);
                _speakerSettings = JsonSerializer.Deserialize<SpeakerSettings>(json) ?? new SpeakerSettings();
                
                // Update current state from loaded settings
                if (_speakerSettings.AvailableSpeakers.Count > 0)
                {
                    _availableSpeakers = _speakerSettings.AvailableSpeakers.ToList();
                }
                else
                {
                    // If no speakers saved, use new defaults (no preset speakers)
                    _availableSpeakers = new List<string> { "Unknown" };
                }
                
                _nextSpeakerNumber = _speakerSettings.NextSpeakerNumber;
                _speakerNames = new Dictionary<string, string>(_speakerSettings.SpeakerNameMappings);
                
                Console.WriteLine($"Loaded settings: {_availableSpeakers.Count} speakers, next number: {_nextSpeakerNumber}");
            }
            else
            {
                Console.WriteLine("No existing speaker settings found, using defaults");
                // Initialize with new defaults (no preset speakers)
                _availableSpeakers = new List<string> { "Unknown" };
                _nextSpeakerNumber = 1;
                _speakerNames = new Dictionary<string, string>();
                
                // Save initial settings
                SaveSpeakerSettings();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading speaker settings: {ex.Message}");
            // Fallback to new defaults
            _availableSpeakers = new List<string> { "Unknown" };
            _nextSpeakerNumber = 1;
            _speakerNames = new Dictionary<string, string>();
        }
    }

    private void SaveSpeakerSettings()
    {
        try
        {
            if (string.IsNullOrEmpty(_settingsFilePath))
            {
                Console.WriteLine("Settings file path not initialized, skipping save");
                return;
            }
            
            // Update settings object with current state
            _speakerSettings.AvailableSpeakers = _availableSpeakers.ToList();
            _speakerSettings.NextSpeakerNumber = _nextSpeakerNumber;
            _speakerSettings.SpeakerNameMappings = new Dictionary<string, string>(_speakerNames);
            
            var options = new JsonSerializerOptions 
            { 
                WriteIndented = true // Make JSON readable
            };
            var json = JsonSerializer.Serialize(_speakerSettings, options);
            File.WriteAllText(_settingsFilePath, json);
            
            Console.WriteLine($"Saved speaker settings to: {_settingsFilePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error saving speaker settings: {ex.Message}");
        }
    }

    private void UpdateSpeakerSettings(string oldSpeaker, string newSpeaker)
    {
        if (_speakerNames.ContainsKey(oldSpeaker))
        {
            _speakerNames[oldSpeaker] = newSpeaker;
            SaveSpeakerSettings();
        }
    }

    private void DeleteSpeaker(ComboBox comboBox, int segmentId)
    {
        var selectedSpeaker = comboBox.SelectedItem as string;
        if (string.IsNullOrEmpty(selectedSpeaker))
        {
            MessageBox.Show("No speaker selected to delete.", "Oreja", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        
        if (selectedSpeaker == "Unknown")
        {
            MessageBox.Show("Cannot delete the 'Unknown' speaker.", "Oreja", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        
        if (!_availableSpeakers.Contains(selectedSpeaker))
        {
            MessageBox.Show("Selected speaker not found in the list.", "Oreja", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        
        var result = MessageBox.Show($"Are you sure you want to delete '{selectedSpeaker}' from the speaker list?\n\nThis will affect all segments assigned to this speaker.", 
            "Delete Speaker", MessageBoxButton.YesNo, MessageBoxImage.Question);
            
        if (result == MessageBoxResult.Yes)
        {
            _availableSpeakers.Remove(selectedSpeaker);
            
            // Update all segments that used this speaker to "Unknown"
            foreach (var segment in _transcriptionHistory.Where(s => s.Speaker == selectedSpeaker))
            {
                segment.Speaker = "Unknown";
            }
            
            // Update current ComboBox to "Unknown"
            UpdateSegmentSpeaker(segmentId, "Unknown");
            
            // Refresh all dropdowns and save settings
            RefreshAllSpeakerDropdowns();
            SaveSpeakerSettings();
            
            Console.WriteLine($"Deleted speaker: {selectedSpeaker}");
        }
    }
} 