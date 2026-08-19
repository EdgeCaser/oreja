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
using System.Diagnostics;

namespace Oreja;

// Persisted application settings. Originally only speaker data was saved here (and only on
// clean shutdown); this now covers everything the app should remember between launches. The
// file name and the original three properties are kept as-is so an existing settings file on
// disk keeps loading (missing new properties just take their default below - no migration step
// needed).
public class AppSettings
{
    // --- Speaker data (existing) ---
    public List<string> AvailableSpeakers { get; set; } = new List<string>();
    public Dictionary<string, string> SpeakerNameMappings { get; set; } = new Dictionary<string, string>();
    public int NextSpeakerNumber { get; set; } = 5;

    // --- Device selection (NAudio MMDevice.ID strings; null/absent = "use the first device") ---
    public string? MicDeviceId { get; set; }
    public string? SystemDeviceId { get; set; }

    // --- Privacy ---
    public bool PrivacyMode { get; set; } = false;

    // --- Window geometry. Left/Top are NaN until the user has moved the window at least once,
    // which we treat as "let WPF center the window" instead of a saved position. ---
    public double WindowWidth { get; set; } = 800;
    public double WindowHeight { get; set; } = 800;
    public double WindowLeft { get; set; } = double.NaN;
    public double WindowTop { get; set; } = double.NaN;

    // --- Backend connection ---
    public string BackendUrl { get; set; } = "http://127.0.0.1:8000";
    public bool AutoStartBackend { get; set; } = true;

    // --- Audio ---
    public float MicGain { get; set; } = 1.0f;

    // --- Transcription language: a Whisper language code ("en", "es", ...) pins the decode
    // to that language for every chunk; "auto" lets the backend detect per chunk, which is
    // the better mode for sessions that mix languages. Defaults to English - per-chunk
    // auto-detection occasionally misfires on short/noisy chunks and produces garbage in
    // the wrong language, so pinning is the accuracy-preserving default. ---
    public string Language { get; set; } = "en";

    // --- Session audio: tee each live recording to per-source 16 kHz mono WAVs
    // (~115 MB/hour) in Documents\Oreja Recordings, so any transcript segment can be
    // replayed via its ▶ button. Ignored while Legal-Safe Mode is on. ---
    public bool SaveSessionAudio { get; set; } = true;

    // --- Keyword alerts: any transcript segment whose speaker or text contains one of these
    // (case-insensitive) is highlighted and flashes the status text. Edited via the inline
    // "🔔 Keyword Alerts" expander above the transcript. ---
    public List<string> KeywordAlerts { get; set; } = new List<string>();
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
    private ComboBox? _languageComboBox;

    // Current transcription language selection, mirrored from the combo box so the
    // capture threads never have to touch UI elements. "auto" = per-chunk detection.
    private volatile string _selectedLanguageCode = "en";

    // (Display, Whisper language code) pairs offered in the language combo. "auto" maps to
    // per-chunk detection on the backend; every other code must be one faster-whisper accepts.
    private static readonly (string Display, string Code)[] LANGUAGE_OPTIONS = new[]
    {
        ("English", "en"),
        ("Auto-detect (multilingual)", "auto"),
        ("Spanish", "es"),
        ("French", "fr"),
        ("German", "de"),
        ("Italian", "it"),
        ("Portuguese", "pt"),
        ("Dutch", "nl"),
        ("Russian", "ru"),
        ("Japanese", "ja"),
        ("Korean", "ko"),
        ("Chinese", "zh"),
        ("Hindi", "hi"),
        ("Arabic", "ar"),
    };
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

    // Microphone input gain.
    // 1.0 = pass captured samples straight through (no destructive amplification).
    // Values > 1.0 amplify quiet microphones; the result is clamped to the Int16 range.
    // Persisted as AppSettings.MicGain (loaded/saved by LoadAppSettings/SaveAppSettings).
    // TODO: no slider/control adjusts this yet - a later change can wire one up to the setting.
    private float _microphoneGain = 1.0f;

    // Transcription components
    private HttpClient? _httpClient;
    private DispatcherTimer? _transcriptionTimer;
    private const string DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";
    // Backing field for the configurable backend base URL (settings.BackendUrl). Kept mutable -
    // unlike the old const - so a saved setting actually takes effect.
    private string _backendUrl = DEFAULT_BACKEND_URL;
    // Dispatch is pause-aware rather than fixed-interval: the timer below only POLLS at this
    // rate; a chunk is actually sent when the speaker pauses (tail of the buffer goes silent)
    // or the buffer hits MAX_CHUNK_SECONDS. Cutting on pauses instead of a hard 5-second timer
    // stops sentences from being split mid-word and gives Whisper full utterances of context.
    private const int DISPATCH_POLL_INTERVAL_MS = 1000;
    private const double MIN_CHUNK_SECONDS = 3.0;   // never send fragments shorter than this
    private const double MAX_CHUNK_SECONDS = 15.0;  // hard cap: flush mid-speech at this size
    private const int SILENCE_WINDOW_MS = 400;      // trailing window inspected for a pause
    private const double SILENCE_RMS_THRESHOLD = 300.0; // int16 RMS ≈ -40 dBFS

    // Display merging: a newly arrived fragment is appended to the previous card when it is
    // the same speaker and source and starts within this gap of the card's end.
    private const double SEGMENT_MERGE_MAX_GAP_SECONDS = 2.0;
    private const int SEGMENT_MERGE_MAX_CHARS = 500; // start a fresh card beyond this length

    // --- Backend status indicator -------------------------------------------------------
    private enum BackendStatus { Connecting, Connected, Offline }
    private Ellipse? _backendStatusDot;
    private TextBlock? _backendStatusText;
    private DispatcherTimer? _backendHealthTimer;
    private bool _backendHealthCheckInFlight = false;
    private string? _lastBackendError;

    // Whether the last contact with the backend (health check or transcribe request) succeeded.
    // DispatchPendingAudio consults this BEFORE cutting a chunk: while false, buffered audio is
    // held (and keeps accumulating, up to MAX_BUFFERED_AUDIO_BYTES) instead of being fired at a
    // port nothing is listening on and lost. The main case is the auto-started backend's model
    // load: uvicorn does not bind the port until whisper/pyannote finish loading (~20-60s), and
    // before this flag every chunk sent in that window died with "connection refused". Volatile:
    // written from health-check paths, read from the dispatch timer.
    private volatile bool _backendReachable = false;
    // True once any health check has succeeded this session. Distinguishes "auto-started backend
    // is still loading models" (status: Connecting) from "backend went away" (status: Offline).
    private volatile bool _backendEverConnected = false;

    // Best-effort auto-started backend process. Only killed on exit if we are the ones who
    // started it - an already-running backend the user launched themselves is left alone.
    private Process? _autoStartedBackendProcess;
    private bool _weStartedBackend = false;

    // The auto-started backend's stdout/stderr are captured to a per-session log file (next to
    // the settings file) so a backend that dies at startup leaves a diagnosable trail - without
    // this its output went nowhere and a crash was indistinguishable from slow model loading.
    private StreamWriter? _backendLogWriter;
    private readonly object _backendLogSync = new object();
    private string? _backendLogPath;
    private int _backendLogStreamsEnded = 0; // stdout + stderr EOFs seen (2 = close the file)

    // Debounces settings writes so dragging/resizing the window doesn't hammer the disk.
    private DispatcherTimer? _settingsSaveTimer;
    private bool _settingsLoaded = false; // Guard against saving while still applying loaded settings.

    // Canonical wire format for everything we send to the backend: 16 kHz mono 16-bit LE PCM.
    private const int TRANSCRIPTION_SAMPLE_RATE = 16000;
    private const int TRANSCRIPTION_CHANNELS = 1;
    private const int TRANSCRIPTION_BYTES_PER_SAMPLE = 2;

    // 32000 bytes = exactly one second of the wire format above. Used to turn a source's
    // cumulative "bytes consumed so far" counter into a recording-relative offset in seconds
    // (see AudioSourceState.ConsumedBytes and ProcessAudioChunkAsync).
    private const int TRANSCRIPTION_BYTES_PER_SECOND =
        TRANSCRIPTION_SAMPLE_RATE * TRANSCRIPTION_CHANNELS * TRANSCRIPTION_BYTES_PER_SAMPLE;

    // Never keep more than this much un-sent audio per source. If the backend is down or
    // slow the oldest audio is dropped instead of growing the buffer without bound. Sized to
    // ride out the auto-started backend's model-load window (~20-60s) with room to spare, since
    // dispatch now holds chunks while the backend is unreachable; 120s is still only ~3.8 MB
    // per source at the 32 KB/s wire format.
    private const int MAX_BUFFERED_AUDIO_SECONDS = 120;
    private const int MAX_BUFFERED_AUDIO_BYTES =
        TRANSCRIPTION_BYTES_PER_SECOND * MAX_BUFFERED_AUDIO_SECONDS;

    // Independent per-source capture/dispatch state. Previously a single shared
    // "_isProcessingTranscription" flag was used, which meant the microphone always won the
    // race and the system-audio chunk was cleared and then dropped at the guard - the far end
    // of the call was never transcribed. Each source now has its own buffer, lock and in-flight
    // flag so the two never interfere.
    private readonly AudioSourceState _micSource = new AudioSourceState("Microphone", "mic");
    private readonly AudioSourceState _systemSource = new AudioSourceState("System Audio", "system");

    // Converts WASAPI loopback audio (whatever format the endpoint reports) to the wire format.
    private readonly SystemAudioFormatConverter _systemAudioConverter = new SystemAudioFormatConverter();
    private bool _systemAudioFormatUnsupported = false; // Set once if the format cannot be converted

    // Speaker renaming and save functionality
    private Dictionary<string, string> _speakerNames = new Dictionary<string, string>();
    private Button? _saveTranscriptionButton;
    private List<TranscriptionSegment> _transcriptionHistory = new List<TranscriptionSegment>();

    // File transcription: one at a time, mutually exclusive with live recording so the
    // transcript timeline never interleaves file-relative and recording-relative times.
    private Button? _transcribeFileButton;
    private bool _isFileTranscriptionRunning = false;
    
    // Manual speaker assignment functionality
    private List<string> _availableSpeakers = new List<string> { "Unknown" }; // Remove default speakers 1-4
    private int _nextSpeakerNumber = 1; // Start from 1 instead of 5

    // Speaker persistence settings
    private const string SETTINGS_FILE = "oreja_speaker_settings.json";
    private string? _settingsFilePath;
    private AppSettings _appSettings = new AppSettings();

    // Keep track of all speaker ComboBoxes for refreshing
    private List<ComboBox> _speakerComboBoxes = new List<ComboBox>();

    // Set while a speaker ComboBox is being updated programmatically (see
    // UpdateSegmentCardInPlace). WPF raises SelectionChanged for a programmatic assignment
    // exactly as it does for a user pick, so without this guard a pure *display* refresh is
    // indistinguishable from the user choosing a new speaker: the handler would rewrite
    // segment.Speaker to the display name, persist a bogus entry in _speakerNames, and POST a
    // phantom correction to /speakers/name_mapping. All of this runs on the UI thread, so a
    // plain bool is sufficient.
    private bool _suppressSpeakerSelectionChanged = false;

    // Privacy Mode functionality
    private bool _privacyModeEnabled = false;
    private CheckBox? _privacyModeCheckBox;
    private Dictionary<string, string> _privacySpeakerMapping = new Dictionary<string, string>();
    private int _privacySpeakerCounter = 1;

    // Enhanced Transcription Editor functionality
    private double _lastScrollPosition = 0;
    private bool _userScrolledUp = false;
    private HashSet<int> _selectedSegments = new HashSet<int>();
    // Selection is always live: click a card to select it, Ctrl+click to toggle,
    // Shift+click to extend from the anchor, or use the checkboxes directly.
    // _selectionAnchorId is the SegmentId the next Shift+click ranges from.
    private int? _selectionAnchorId;
    private static readonly Brush _selectionCardBackground =
        new SolidColorBrush(Color.FromRgb(219, 234, 254)); // light blue, distinct from the alert yellow

    // Monotonic segment id source. SegmentIds used to be _transcriptionHistory.Count,
    // which collides with surviving ids as soon as deletion exists (delete one of three
    // segments and the next arrival would reuse id 2). Never reset - stale gaps are fine.
    private int _nextSegmentId = 0;

    // --- Session speaker roster -------------------------------------------------------
    // Who the user said is in this session (👥 Session Speakers dialog). Roster names
    // lead the per-segment dropdowns, map onto the 1-4 quick-assign buttons in roster
    // order, and the roster size is sent as the diarization max_speakers ceiling.
    // Guests are session-only identities: they never join _availableSpeakers (so they
    // are not persisted to settings) and corrections to them never send enrollment
    // feedback to the backend, so no voiceprint is learned for them.
    private List<string> _sessionRoster = new List<string>();
    private HashSet<string> _sessionGuestNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    private Button? _bulkRenameButton;
    private Button? _selectAllButton;
    private Button? _clearSelectionButton;
    private Button? _deleteSelectedButton;
    private List<CheckBox> _segmentCheckBoxes = new List<CheckBox>();

    // Transcript search: a TextBox above the transcript filters segment cards by text/speaker.
    // Filtering itself is applied via ApplySearchFilter(), debounced ~300ms from keystrokes by
    // _searchDebounceTimer so a fast typist doesn't trigger a filter pass on every character.
    private TextBox? _searchTextBox;
    private TextBlock? _searchMatchCountText;
    private DispatcherTimer? _searchDebounceTimer;
    private string _currentSearchQuery = "";

    // Keyword alerts: comma-separated list edited via _keywordAlertsTextBox (inline expander),
    // persisted as AppSettings.KeywordAlerts. Matching segments get a highlighted card
    // (ApplyKeywordAlertHighlight) and a brief status-text flash when they arrive live.
    private TextBox? _keywordAlertsTextBox;
    private static readonly Brush _keywordAlertCardBackground = new SolidColorBrush(Color.FromRgb(0xFF, 0xF3, 0xCD));
    private static readonly Brush _keywordAlertCardBorder = Brushes.Goldenrod;

    // Single reusable status-text flash timer (see FlashStatusTextForKeywordAlert). One shared
    // timer that is restarted per alert, rather than one timer per alert, so overlapping alerts
    // extend the flash instead of the first tick cancelling all of them - and so a burst of
    // matches doesn't leak a DispatcherTimer per match. _statusTextBackgroundBeforeFlash holds
    // the brush to put back (null is a normal value: TextBlock.Background defaults to null).
    private DispatcherTimer? _keywordFlashTimer;
    private Brush? _statusTextBackgroundBeforeFlash;

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

        // --- Live UI element cache (not persisted - these are WPF elements, not settings data)---
        // Populated when this segment's card is built (AddTranscriptionSegment for a newly
        // arrived segment, RefreshTranscriptionDisplay for a full rebuild) so a speaker rename,
        // privacy-mode toggle, or keyword-alert re-scan can patch the existing card in place
        // instead of tearing down and rebuilding every segment card in the transcript.
        public Border? CardElement { get; set; }
        public ComboBox? SpeakerComboBoxElement { get; set; }
        public TextBox? TextDisplayElement { get; set; }
        public TextBlock? AlertBellElement { get; set; }
        public CheckBox? SelectionCheckBoxElement { get; set; }
        public Rectangle? SpeakerStripeElement { get; set; }

        // Where this segment's audio can be replayed from: the original media file for
        // file transcriptions, or null for live segments (those resolve through the
        // per-source session recording, see ResolveSegmentAudioPath).
        public string? AudioFilePath { get; set; }
    }

    // Smart speaker filtering for dropdown
    private List<string> GetFilteredSpeakersForDropdown()
    {
        var filteredSpeakers = new List<string>();

        // Always add "Unknown" first
        filteredSpeakers.Add("Unknown");

        // Session roster leads, in roster order: these are the names the user said are
        // actually in the room, so they are the overwhelmingly likely corrections. This
        // is also how guests become assignable - they are never in _availableSpeakers.
        filteredSpeakers.AddRange(_sessionRoster.Where(n => n != "Unknown"));

        // Add user-defined speakers (non-auto-generated)
        var userDefinedSpeakers = _availableSpeakers
            .Where(s => s != "Unknown" && !s.StartsWith("Speaker_AUTO_SPEAKER_")
                && !filteredSpeakers.Contains(s))
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
            .Where(t => t.Speaker != null && autoSpeakers.Contains(t.Speaker))
            .GroupBy(t => t.Speaker)
            .OrderByDescending(g => g.Max(t => t.Timestamp))
            .Take(maxCount)
            .Select(g => g.Key!)
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
            string? selectedSpeaker = null;
            
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
                    SaveAppSettings();
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
#if DEBUG
        // Allocate a console window for debug output only - Release builds run silently
        // with no visible console, which is what an installed app should do.
        AllocConsole();
#endif
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

            // Load persisted settings synchronously (speaker data, backend URL, privacy mode,
            // mic gain, window geometry, device IDs) before the window is built so geometry and
            // the backend URL are available immediately. Device selection is restored later,
            // once LoadAudioDevices() has actually enumerated the devices to match against.
            LoadAppSettings();

            // Load speakers from enhanced backend (with fallback to local settings loaded above)
            Console.WriteLine("🔄 Initializing enhanced speaker system...");
            _ = LoadSpeakersFromBackend();

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
            ApplyWindowGeometry(window, _appSettings);
            
            // Use Grid instead of StackPanel for better layout control
            var mainGrid = new Grid { Margin = new Thickness(20) };
            
            // Define rows for the grid
            // NOTE: this list must have exactly one entry per Grid.SetRow(..., currentRow++) call
            // below, in the same order - a previous mismatch here (the "Privacy row" had no
            // dedicated entry, so every row from Privacy onward silently used the next row's
            // definition) left the transcription area's Star sizing attached to the multi-select
            // toolbar instead of the scroll viewer, so the transcript no longer filled the window.
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 0: Title
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 1: Microphone section
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 2: System audio section
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 3: Language section
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 4: Volume meters
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 5: Control buttons
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 6: Backend status indicator
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 7: Privacy mode row
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 8: Status text
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 9: Transcription label
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 10: Instructions
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 11: Keyword alerts expander
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 12: Transcript search row
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto }); // 13: Multi-select toolbar
            mainGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) }); // 14: Transcription area (takes remaining space)
            
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

            // Transcription language section
            var languageSection = new StackPanel { Margin = new Thickness(0, 0, 0, 15) };
            var languageLabel = new TextBlock
            {
                Text = "Transcription Language:",
                FontSize = 14,
                Margin = new Thickness(0, 0, 0, 5)
            };

            _languageComboBox = new ComboBox
            {
                Width = 250,
                HorizontalAlignment = HorizontalAlignment.Left
            };
            foreach (var (display, code) in LANGUAGE_OPTIONS)
            {
                _languageComboBox.Items.Add(new ComboBoxItem { Content = display, Tag = code });
            }
            // Restore the saved selection (LoadAppSettings has already run). An unknown saved
            // code (hand-edited file, future option removed) falls back to English at index 0.
            var savedLanguage = _appSettings.Language;
            var savedIndex = Array.FindIndex(LANGUAGE_OPTIONS, o => o.Code == savedLanguage);
            _languageComboBox.SelectedIndex = savedIndex >= 0 ? savedIndex : 0;
            _selectedLanguageCode = LANGUAGE_OPTIONS[_languageComboBox.SelectedIndex].Code;
            _languageComboBox.SelectionChanged += LanguageComboBox_SelectionChanged;

            languageSection.Children.Add(languageLabel);
            languageSection.Children.Add(_languageComboBox);
            Grid.SetRow(languageSection, currentRow++);
            
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

            _transcribeFileButton = new Button
            {
                Content = "📂 Transcribe File",
                Width = 150,
                Height = 40,
                Margin = new Thickness(10, 0, 0, 0),
                FontSize = 14,
                Background = Brushes.LightBlue,
                ToolTip = "Transcribe an audio file (WAV/FLAC/OGG/MP3) through the same pipeline as live audio"
            };
            _transcribeFileButton.Click += TranscribeFileButton_Click;

            var sessionSpeakersButton = new Button
            {
                Content = "👥 Session Speakers",
                Width = 160,
                Height = 40,
                Margin = new Thickness(10, 0, 0, 0),
                FontSize = 14,
                Background = Brushes.Lavender,
                ToolTip = "Declare who is in this session so the tool doesn't have to guess:\n" +
                          "known speakers keep learning voiceprints, guests stay session-only."
            };
            sessionSpeakersButton.Click += (s, e) => ShowSessionSpeakersDialog();

            buttonPanel.Children.Add(_startRecordingButton);
            buttonPanel.Children.Add(_stopRecordingButton);
            buttonPanel.Children.Add(_monitoringToggleButton);
            buttonPanel.Children.Add(_saveTranscriptionButton);
            buttonPanel.Children.Add(_transcribeFileButton);
            buttonPanel.Children.Add(sessionSpeakersButton);
            Grid.SetRow(buttonPanel, currentRow++);

            // Backend status indicator: a small colored dot + short text, updated by the
            // periodic health poll (see PollBackendHealthAsync / UpdateBackendStatusUI).
            var backendStatusPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 0, 0, 10)
            };

            _backendStatusDot = new Ellipse
            {
                Width = 10,
                Height = 10,
                Fill = Brushes.Gray,
                Margin = new Thickness(0, 0, 6, 0),
                VerticalAlignment = VerticalAlignment.Center
            };

            _backendStatusText = new TextBlock
            {
                Text = "Backend: checking...",
                FontSize = 11,
                VerticalAlignment = VerticalAlignment.Center,
                Foreground = Brushes.Gray
            };

            backendStatusPanel.Children.Add(_backendStatusDot);
            backendStatusPanel.Children.Add(_backendStatusText);
            Grid.SetRow(backendStatusPanel, currentRow++);

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
            // Restore the saved privacy mode. Both _statusText and _transcriptionPanel are still
            // null at this point in window construction; the handler null-checks both, so this
            // is safe and simply primes _privacyModeEnabled before anything is shown.
            _privacyModeCheckBox.IsChecked = _appSettings.PrivacyMode;

            var privacyHelp = new TextBlock
            {
                Text = "Analysis only - no verbatim transcription stored (legal-safe)",
                FontSize = 10,
                FontStyle = FontStyles.Italic,
                Foreground = Brushes.Gray,
                VerticalAlignment = VerticalAlignment.Center
            };
            
            var saveAudioCheckBox = new CheckBox
            {
                Content = "💾 Save session audio",
                FontSize = 12,
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(20, 0, 10, 0),
                IsChecked = _appSettings.SaveSessionAudio,
                ToolTip = "Keep a compact 16 kHz recording of each session (Documents\\Oreja Recordings, ~115 MB/hour)\n" +
                          "so the ▶ button on a segment can replay its audio. Ignored in Legal-Safe Mode.\n" +
                          "Takes effect at the next recording start."
            };
            saveAudioCheckBox.Checked += (s, e) => { _appSettings.SaveSessionAudio = true; SaveAppSettings(); };
            saveAudioCheckBox.Unchecked += (s, e) => { _appSettings.SaveSessionAudio = false; SaveAppSettings(); };

            privacyPanel.Children.Add(_privacyModeCheckBox);
            privacyPanel.Children.Add(privacyHelp);
            privacyPanel.Children.Add(saveAudioCheckBox);
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
                Text = "💡 Transcript Help:",
                FontWeight = FontWeights.Bold,
                FontSize = 12,
                Margin = new Thickness(0, 0, 0, 5)
            };

            var instructionsText = new TextBlock
            {
                Text = "• Use the speaker dropdown on a segment to reassign just that segment (type a new name to create one)\n• Click '+' to create a new speaker, '🔍' to browse auto-detected speakers, '✏' to rename a speaker everywhere, '×' to delete one\n• Click a card to select it - Shift+click selects a range, Ctrl+click toggles, right-click selects by speaker or deletes\n• ▶ replays a segment's audio (kept in Documents\\Oreja Recordings while 💾 Save session audio is on)\n• Use Search above the transcript to filter segments by text or speaker (Esc clears); Select All picks every match\n• Expand 🔔 Keyword Alerts to highlight segments containing chosen words as they arrive\n• Right-click a segment's text to split it, or double-click to edit it directly\n• Save Transcription exports to JSON, TXT, SRT, WebVTT, or Markdown",
                FontSize = 11,
                Foreground = Brushes.DarkBlue,
                TextWrapping = TextWrapping.Wrap
            };

            instructionsPanel.Children.Add(instructionsTitle);
            instructionsPanel.Children.Add(instructionsText);
            instructionsBorder.Child = instructionsPanel;
            Grid.SetRow(instructionsBorder, currentRow++);

            // Keyword alerts: collapsed-by-default expander holding a comma-separated keyword
            // list (persisted as AppSettings.KeywordAlerts). Any segment whose speaker or text
            // contains one of these (case-insensitive, whole-word-ish) gets a highlighted card
            // and a 🔔 prefix; a live match also flashes the status text briefly.
            var keywordAlertsExpander = new Expander
            {
                Header = "🔔 Keyword Alerts",
                FontSize = 12,
                Margin = new Thickness(0, 0, 0, 8),
                IsExpanded = false
            };

            var keywordAlertsPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Margin = new Thickness(4, 6, 4, 4)
            };

            _keywordAlertsTextBox = new TextBox
            {
                Width = 320,
                Padding = new Thickness(4, 2, 4, 2),
                FontSize = 12,
                Text = string.Join(", ", _appSettings.KeywordAlerts),
                ToolTip = "Comma-separated keywords. Matching segments are highlighted."
            };

            var applyKeywordAlertsButton = new Button
            {
                Content = "Apply",
                Width = 70,
                Height = 24,
                Margin = new Thickness(6, 0, 0, 0),
                FontSize = 11
            };
            applyKeywordAlertsButton.Click += ApplyKeywordAlertsButton_Click;

            keywordAlertsPanel.Children.Add(_keywordAlertsTextBox);
            keywordAlertsPanel.Children.Add(applyKeywordAlertsButton);
            keywordAlertsExpander.Content = keywordAlertsPanel;
            Grid.SetRow(keywordAlertsExpander, currentRow++);

            // Transcript search: filters segment cards by text or speaker name as the user types.
            // Filtering is debounced ~300ms (SearchTextBox_TextChanged / _searchDebounceTimer) so
            // fast typing doesn't re-scan the transcript on every keystroke; Esc clears it.
            var searchPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Margin = new Thickness(0, 0, 0, 8)
            };

            var searchLabel = new TextBlock
            {
                Text = "🔎 Search:",
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 6, 0),
                FontSize = 12
            };

            _searchTextBox = new TextBox
            {
                Width = 220,
                Padding = new Thickness(4, 2, 4, 2),
                FontSize = 12,
                ToolTip = "Filter the transcript by text or speaker name (Esc to clear)"
            };
            _searchTextBox.TextChanged += SearchTextBox_TextChanged;
            _searchTextBox.KeyDown += SearchTextBox_KeyDown;

            _searchMatchCountText = new TextBlock
            {
                Text = "",
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(8, 0, 0, 0),
                FontSize = 11,
                FontStyle = FontStyles.Italic,
                Foreground = Brushes.Gray
            };

            searchPanel.Children.Add(searchLabel);
            searchPanel.Children.Add(_searchTextBox);
            searchPanel.Children.Add(_searchMatchCountText);
            Grid.SetRow(searchPanel, currentRow++);

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
            mainGrid.Children.Add(languageSection);
            mainGrid.Children.Add(volumeSection);
            mainGrid.Children.Add(buttonPanel);
            mainGrid.Children.Add(backendStatusPanel);
            mainGrid.Children.Add(privacyPanel);
            mainGrid.Children.Add(_statusText);
            mainGrid.Children.Add(transcriptionLabel);
            mainGrid.Children.Add(instructionsBorder);
            mainGrid.Children.Add(keywordAlertsExpander);
            mainGrid.Children.Add(searchPanel);
            mainGrid.Children.Add(multiSelectToolbar);
            mainGrid.Children.Add(_transcriptionScrollViewer);
            
            window.Content = mainGrid;
            window.Closing += Window_Closing;
            // Debounced (~1s) geometry save - dragging/resizing fires these continuously, and
            // RequestSettingsSave collapses a burst of them into a single write.
            window.SizeChanged += (s, args) => RequestSettingsSave();
            window.LocationChanged += (s, args) => RequestSettingsSave();
            
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
                    // Re-select whatever devices were saved from a previous run, if still present.
                    RestoreDeviceSelection();

                    Console.WriteLine("Checking backend connection...");
                    // Check backend connectivity - and, on the very first check, best-effort
                    // auto-start the backend if it isn't reachable yet (InitializeBackendConnectivityAsync
                    // also starts the recurring 10s health poll used for the rest of the session).
                    _ = InitializeBackendConnectivityAsync();

                    Console.WriteLine("Setting up timers...");
                    // Setup volume monitoring timer - less frequent to improve UI responsiveness
                    _volumeTimer = new DispatcherTimer();
                    _volumeTimer.Interval = TimeSpan.FromMilliseconds(200); // Reduced frequency
                    _volumeTimer.Tick += VolumeTimer_Tick;
                    _volumeTimer.Start();
                    
                    // Setup transcription timer
                    _transcriptionTimer = new DispatcherTimer();
                    _transcriptionTimer.Interval = TimeSpan.FromMilliseconds(DISPATCH_POLL_INTERVAL_MS);
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
    
    // ---------------------------------------------------------------------------------
    // Backend connectivity: status indicator, periodic health poll, and best-effort auto-start.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Runs the first health check, best-effort auto-starts the backend if that check fails,
    /// then starts the recurring 10s poll that keeps the status indicator (and _lastBackendError)
    /// current for the rest of the session.
    /// </summary>
    private async Task InitializeBackendConnectivityAsync()
    {
        UpdateBackendStatusUI(BackendStatus.Connecting);

        bool healthy = await CheckBackendHealthAsync().ConfigureAwait(true);
        if (!healthy)
        {
            TryAutoStartBackend();

            if (_appSettings.AutoStartBackend && _weStartedBackend)
            {
                // Give a freshly-spawned uvicorn a little time to bind the port before the next
                // check, so the indicator doesn't flash "offline" for a process we just started.
                await Task.Delay(3000).ConfigureAwait(true);
                healthy = await CheckBackendHealthAsync().ConfigureAwait(true);
            }
        }

        UpdateBackendStatusUI(
            healthy ? BackendStatus.Connected : BackendStartupInProgress() ? BackendStatus.Connecting : BackendStatus.Offline,
            healthy ? null : _lastBackendError);

        _backendHealthTimer = new DispatcherTimer();
        _backendHealthTimer.Interval = TimeSpan.FromSeconds(10);
        _backendHealthTimer.Tick += async (s, e) => await PollBackendHealthAsync().ConfigureAwait(true);
        _backendHealthTimer.Start();
    }

    /// <summary>
    /// One health-check tick. Re-entrancy-guarded so an overlapping call (from the timer firing
    /// again, or from ReportBackendFailure below) is a no-op instead of stacking requests.
    /// </summary>
    private async Task PollBackendHealthAsync()
    {
        if (_backendHealthCheckInFlight)
        {
            return;
        }
        _backendHealthCheckInFlight = true;
        try
        {
            bool healthy = await CheckBackendHealthAsync().ConfigureAwait(true);
            UpdateBackendStatusUI(
                healthy ? BackendStatus.Connected : BackendStartupInProgress() ? BackendStatus.Connecting : BackendStatus.Offline,
                healthy ? null : _lastBackendError);
        }
        finally
        {
            _backendHealthCheckInFlight = false;
        }
    }

    private async Task<bool> CheckBackendHealthAsync()
    {
        try
        {
            var response = await _httpClient!.GetAsync($"{_backendUrl}/health").ConfigureAwait(true);
            if (response.IsSuccessStatusCode)
            {
                _lastBackendError = null;
                _backendReachable = true;
                _backendEverConnected = true;
                return true;
            }

            _lastBackendError = $"HTTP {(int)response.StatusCode}";
            _backendReachable = false;
            return false;
        }
        catch (Exception ex)
        {
            _lastBackendError = ex.Message;
            _backendReachable = false;
            return false;
        }
    }

    /// <summary>
    /// True while the backend we auto-started is alive but has not yet answered a health check -
    /// i.e. uvicorn is still loading models and hasn't bound the port. Used to show "Connecting"
    /// instead of "Offline" during that window: the old label read as a failure when the honest
    /// state was "starting". Once the process dies (or the backend has connected at least once),
    /// this returns false and a failed check means genuinely Offline again.
    /// </summary>
    private bool BackendStartupInProgress()
    {
        if (_backendEverConnected || !_weStartedBackend)
        {
            return false;
        }

        var process = _autoStartedBackendProcess;
        if (process == null)
        {
            return false;
        }

        try
        {
            return !process.HasExited;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Called from the transcription/feedback request paths right after a failure. Sets the
    /// indicator to Offline immediately (instead of only console logging) and kicks off an
    /// out-of-band health check so the indicator clears itself automatically the moment the
    /// backend is reachable again, without waiting for the next 10s tick. Safe to call from any
    /// thread; UI updates are marshalled to the dispatcher.
    /// </summary>
    private void ReportBackendFailure(string errorMessage)
    {
        // Close the dispatch gate immediately (not from the dispatcher callback below): the next
        // timer tick must already see the backend as unreachable so only ONE chunk is ever lost
        // to a dead backend - everything after it is held in the source buffers instead.
        _backendReachable = false;

        // The poll is started INSIDE the dispatcher callback, not alongside it. PollBackendHealthAsync
        // awaits with ConfigureAwait(true) and then writes to _backendStatusDot/_backendStatusText;
        // started from a thread-pool thread (which FlushSourceOnStopAsync's retry path does reach -
        // its Task.Delay uses ConfigureAwait(false)) there is no SynchronizationContext to capture,
        // so the continuation would resume off the UI thread and throw "the calling thread cannot
        // access this object", inside a discarded Task where nobody ever observes it. Starting it
        // on the dispatcher gives it the UI SynchronizationContext to come back to.
        Dispatcher.BeginInvoke(new Action(() =>
        {
            _lastBackendError = errorMessage;
            UpdateBackendStatusUI(BackendStatus.Offline, errorMessage);
            _ = PollBackendHealthAsync();
        }));
    }

    /// <summary>Called right after a transcription/feedback request succeeds, so the indicator
    /// flips back to Connected immediately rather than waiting for the next poll.</summary>
    private void ReportBackendSuccess()
    {
        _backendReachable = true;
        _backendEverConnected = true;
        Dispatcher.BeginInvoke(new Action(() => UpdateBackendStatusUI(BackendStatus.Connected)));
    }

    /// <summary>
    /// Writes the status dot/label. Self-marshalling: every caller is somewhere downstream of an
    /// await, so rather than auditing each path for thread affinity this hops to the dispatcher
    /// itself when called off the UI thread.
    /// </summary>
    private void UpdateBackendStatusUI(BackendStatus status, string? detail = null)
    {
        if (!Dispatcher.CheckAccess())
        {
            Dispatcher.BeginInvoke(new Action(() => UpdateBackendStatusUI(status, detail)));
            return;
        }

        if (_backendStatusDot == null || _backendStatusText == null)
        {
            return;
        }

        switch (status)
        {
            case BackendStatus.Connected:
                _backendStatusDot.Fill = Brushes.LimeGreen;
                _backendStatusText.Text = "Backend: Connected";
                _backendStatusText.Foreground = Brushes.DarkGreen;
                break;
            case BackendStatus.Connecting:
                _backendStatusDot.Fill = Brushes.Orange;
                _backendStatusText.Text = "Backend: Connecting...";
                _backendStatusText.Foreground = Brushes.DarkOrange;
                break;
            case BackendStatus.Offline:
            default:
                _backendStatusDot.Fill = Brushes.Red;
                var shortDetail = string.IsNullOrWhiteSpace(detail) ? null : detail.Length > 80 ? detail.Substring(0, 80) + "..." : detail;
                _backendStatusText.Text = shortDetail == null ? "Backend offline" : $"Backend offline: {shortDetail}";
                _backendStatusText.Foreground = Brushes.DarkRed;
                break;
        }
    }

    /// <summary>
    /// Best-effort: locate a python interpreter and the backend folder, then spawn
    /// "python -m uvicorn server:app --host &lt;host&gt; --port &lt;port&gt;" hidden, with the backend
    /// folder as its working directory. Host and port come from the configured backend URL, so a
    /// user who moved the backend off :8000 doesn't get a uvicorn bound to a port nothing polls.
    /// A non-loopback URL means the backend lives on another machine and is not ours to start.
    /// Any failure just leaves the offline indicator showing - this is a convenience, not a
    /// requirement for the app to run against an already-running backend.
    /// </summary>
    private void TryAutoStartBackend()
    {
        try
        {
            if (!_appSettings.AutoStartBackend)
            {
                Console.WriteLine("Auto-start backend disabled by settings.");
                return;
            }

            if (!Uri.TryCreate(_backendUrl, UriKind.Absolute, out var backendUri))
            {
                Console.WriteLine($"Auto-start backend: '{_backendUrl}' is not a valid absolute URL; not starting anything.");
                return;
            }

            if (!backendUri.IsLoopback)
            {
                Console.WriteLine($"Auto-start backend: {backendUri.Host} is not a loopback address - the backend is remote, leaving it alone.");
                return;
            }

            string? backendDir = FindBackendDirectory();
            if (backendDir == null)
            {
                Console.WriteLine("Auto-start backend: could not find a 'backend' folder containing server.py near the executable.");
                return;
            }

            string pythonExe = FindPythonExecutable();

            var startInfo = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"-m uvicorn server:app --host {backendUri.Host} --port {backendUri.Port}",
                WorkingDirectory = backendDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
                // Capture the backend's output to a log file (see OpenBackendLog). Without this
                // its stdout/stderr went nowhere and a crash at startup was indistinguishable
                // from slow model loading. Redirection makes the child block once the pipe
                // buffer fills, so the Begin*ReadLine calls below are mandatory, not optional.
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = new UTF8Encoding(false),
                StandardErrorEncoding = new UTF8Encoding(false),
            };
            // Make python actually emit UTF-8 on redirected pipes; its default on Windows is the
            // ANSI codepage, which mangles the backend's emoji-laden log lines.
            startInfo.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

            OpenBackendLog(pythonExe, startInfo.Arguments, backendDir);

            Console.WriteLine($"Auto-starting backend: \"{pythonExe}\" {startInfo.Arguments} (cwd: {backendDir})");
            _autoStartedBackendProcess = Process.Start(startInfo);
            _weStartedBackend = _autoStartedBackendProcess != null;

            if (_weStartedBackend)
            {
                var process = _autoStartedBackendProcess!;
                process.EnableRaisingEvents = true;
                process.Exited += OnAutoStartedBackendExited;
                process.OutputDataReceived += (s, e) => WriteBackendLogLine(e.Data);
                process.ErrorDataReceived += (s, e) => WriteBackendLogLine(e.Data);
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                Console.WriteLine($"Auto-started backend process id {process.Id}. Output -> {_backendLogPath ?? "(log file unavailable)"}");
            }
            else
            {
                CloseBackendLog();
            }
        }
        catch (Exception ex)
        {
            // Best-effort only: no python/uvicorn available, permissions issue, etc. The offline
            // indicator (already showing) communicates this to the user; nothing else to do.
            Console.WriteLine($"Auto-start backend failed: {ex.Message}");
            _weStartedBackend = false;
            CloseBackendLog();
        }
    }

    /// <summary>
    /// Opens (truncating) the auto-started backend's output log, next to the settings file:
    /// %APPDATA%\Oreja\backend.log. One file per app session - "what did the backend say last
    /// time it ran" is the question this answers; unbounded append is not worth it. Failure to
    /// open just means backend output is discarded (WriteBackendLogLine tolerates a null
    /// writer); the backend itself still runs.
    /// </summary>
    private void OpenBackendLog(string pythonExe, string arguments, string backendDir)
    {
        lock (_backendLogSync)
        {
            try
            {
                var orejaFolder = System.IO.Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Oreja");
                Directory.CreateDirectory(orejaFolder);
                _backendLogPath = System.IO.Path.Combine(orejaFolder, "backend.log");

                _backendLogStreamsEnded = 0;
                _backendLogWriter = new StreamWriter(_backendLogPath, append: false, new UTF8Encoding(false))
                {
                    // Line-level flushing so a crash (theirs or ours) never loses the tail of
                    // the log - which is exactly the part a startup failure is diagnosed from.
                    AutoFlush = true,
                };
                _backendLogWriter.WriteLine($"=== Oreja backend started {DateTime.Now:yyyy-MM-dd HH:mm:ss} ===");
                _backendLogWriter.WriteLine($"=== \"{pythonExe}\" {arguments} (cwd: {backendDir}) ===");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Could not open backend log file: {ex.Message}");
                _backendLogWriter = null;
            }
        }
    }

    /// <summary>
    /// Sink for the backend's OutputDataReceived/ErrorDataReceived events (thread-pool threads).
    /// A null line is that stream's EOF; after both streams end the file is closed. Writes after
    /// close are silently dropped - EOF/exit ordering isn't guaranteed and a lost trailing line
    /// beats a crash in an event handler.
    /// </summary>
    private void WriteBackendLogLine(string? line)
    {
        lock (_backendLogSync)
        {
            if (line == null)
            {
                if (++_backendLogStreamsEnded == 2)
                {
                    CloseBackendLogLocked();
                }
                return;
            }

            try
            {
                _backendLogWriter?.WriteLine(line);
            }
            catch
            {
                // Disposed underneath us or disk trouble; the log is best-effort.
            }
        }
    }

    private void CloseBackendLog()
    {
        lock (_backendLogSync)
        {
            CloseBackendLogLocked();
        }
    }

    private void CloseBackendLogLocked()
    {
        try
        {
            _backendLogWriter?.Dispose();
        }
        catch
        {
        }
        _backendLogWriter = null;
    }

    /// <summary>
    /// The auto-started backend exited on its own (a crash, or clean shutdown we didn't ask
    /// for). Record the exit code in both the console and the log file, and point at the log -
    /// this line is what turns "connection refused forever" into a diagnosable failure.
    /// </summary>
    private void OnAutoStartedBackendExited(object? sender, EventArgs e)
    {
        int? exitCode = null;
        try
        {
            exitCode = (sender as Process)?.ExitCode;
        }
        catch
        {
            // Process already disposed (app shutdown path); the console line below still lands.
        }

        var codeText = exitCode?.ToString() ?? "unknown";
        WriteBackendLogLine($"=== backend process exited (code {codeText}) at {DateTime.Now:yyyy-MM-dd HH:mm:ss} ===");
        Console.WriteLine($"Auto-started backend exited (code {codeText}). Its output is in: {_backendLogPath ?? "(log file unavailable)"}");
    }

    /// <summary>
    /// Searches "&lt;repo&gt;/venv/Scripts/python.exe" relative to the executable's directory and
    /// each of its parents; falls back to "python" (resolved via PATH by Process.Start) if none
    /// of them exist.
    /// </summary>
    private static string FindPythonExecutable()
    {
        try
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null)
            {
                // Fully qualified: System.Windows.Shapes.Path is also in scope (via
                // "using System.Windows.Shapes;"), and a bare "Path" is ambiguous between the two.
                var candidate = System.IO.Path.Combine(dir.FullName, "venv", "Scripts", "python.exe");
                if (File.Exists(candidate))
                {
                    return candidate;
                }
                dir = dir.Parent;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error searching for a venv python executable: {ex.Message}");
        }

        return "python";
    }

    /// <summary>
    /// Searches the executable's directory and each of its parents for a folder named "backend"
    /// that contains server.py.
    /// </summary>
    private static string? FindBackendDirectory()
    {
        try
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null)
            {
                var candidate = System.IO.Path.Combine(dir.FullName, "backend");
                if (Directory.Exists(candidate) && File.Exists(System.IO.Path.Combine(candidate, "server.py")))
                {
                    return candidate;
                }
                dir = dir.Parent;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error searching for the backend folder: {ex.Message}");
        }

        return null;
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
                    _selectedMicrophone = _availableMicrophones[0];
                    // Translated, not assumed to be 0 - see ResolveWaveInDeviceNumber. (Assigning
                    // SelectedIndex above already ran the handler, which does the same thing; this
                    // just keeps the two paths from disagreeing if that ever stops firing.)
                    _selectedMicrophoneIndex = ResolveWaveInDeviceNumber(_selectedMicrophone, 0);
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

    /// <summary>
    /// Re-selects the microphone/system-audio devices saved from a previous run, matched by
    /// NAudio's stable MMDevice.ID. Must run after LoadAudioDevices has populated the combo
    /// boxes. A device that is no longer present (unplugged, renamed) is silently left at
    /// LoadAudioDevices' default (index 0) instead of failing.
    ///
    /// Setting SelectedIndex here runs MicrophoneComboBox_SelectionChanged, which is where the
    /// dropdown position gets translated into a waveIn device number - so restoring a saved
    /// selection opens the device the dropdown names, not whatever sits at the same position in
    /// the other enumeration.
    /// </summary>
    private void RestoreDeviceSelection()
    {
        try
        {
            if (_microphoneComboBox != null && _availableMicrophones != null && !string.IsNullOrEmpty(_appSettings.MicDeviceId))
            {
                for (int i = 0; i < _availableMicrophones.Count; i++)
                {
                    if (_availableMicrophones[i].ID == _appSettings.MicDeviceId)
                    {
                        _microphoneComboBox.SelectedIndex = i; // Fires MicrophoneComboBox_SelectionChanged
                        break;
                    }
                }
            }

            if (_systemAudioComboBox != null && _availableSystemAudioDevices != null && !string.IsNullOrEmpty(_appSettings.SystemDeviceId))
            {
                for (int i = 0; i < _availableSystemAudioDevices.Count; i++)
                {
                    if (_availableSystemAudioDevices[i].ID == _appSettings.SystemDeviceId)
                    {
                        _systemAudioComboBox.SelectedIndex = i; // Fires SystemAudioComboBox_SelectionChanged
                        break;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error restoring saved device selection: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Maps an MMDevice (WASAPI enumeration - what the dropdown is built from) to the legacy
    /// waveIn device number that WaveInEvent.DeviceNumber expects.
    ///
    /// These are two different enumerations with two different orderings, and the app was feeding
    /// a position in the first straight into the second: picking the third microphone in the
    /// dropdown could open a different physical microphone than the one named next to it, with no
    /// error anywhere. Matching on the device name restores the invariant that
    /// _selectedMicrophoneIndex is valid for how the capture object actually opens the device.
    ///
    /// waveIn product names are truncated to 31 characters by MMSYSTEM
    /// (WaveInCapabilities.MaxProductNameLength), so a long WASAPI FriendlyName such as
    /// "Microphone Array (Realtek(R) Audio)" is matched by prefix rather than equality.
    /// </summary>
    private static int ResolveWaveInDeviceNumber(MMDevice? device, int enumerationIndex)
    {
        int waveInCount;
        try
        {
            waveInCount = WaveInEvent.DeviceCount;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Could not enumerate waveIn devices ({ex.Message}); using the dropdown index as-is.");
            return enumerationIndex;
        }

        if (waveInCount <= 0)
        {
            return enumerationIndex; // Nothing to match against - leave the caller's value alone.
        }

        var friendlyName = device?.FriendlyName;
        if (!string.IsNullOrEmpty(friendlyName))
        {
            for (int i = 0; i < waveInCount; i++)
            {
                string productName;
                try
                {
                    productName = WaveInEvent.GetCapabilities(i).ProductName ?? "";
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Could not read waveIn device {i} capabilities: {ex.Message}");
                    continue;
                }

                if (productName.Length == 0)
                {
                    continue;
                }

                if (string.Equals(productName, friendlyName, StringComparison.OrdinalIgnoreCase) ||
                    friendlyName.StartsWith(productName, StringComparison.OrdinalIgnoreCase))
                {
                    return i;
                }
            }

            Console.WriteLine($"No waveIn device name matched '{friendlyName}'; falling back to a positional guess.");
        }

        // No name match: keep the dropdown index if it is at least a valid waveIn device number,
        // otherwise fall back to the first waveIn device rather than opening nothing.
        return (enumerationIndex >= 0 && enumerationIndex < waveInCount) ? enumerationIndex : 0;
    }

    private void MicrophoneComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_microphoneComboBox != null && _availableMicrophones != null)
        {
            var enumerationIndex = _microphoneComboBox.SelectedIndex;
            if (enumerationIndex >= 0 && enumerationIndex < _availableMicrophones.Count)
            {
                _selectedMicrophone = _availableMicrophones[enumerationIndex];
                // NOT the dropdown index: _selectedMicrophoneIndex is used verbatim as
                // WaveInEvent.DeviceNumber, which indexes a different enumeration entirely.
                _selectedMicrophoneIndex = ResolveWaveInDeviceNumber(_selectedMicrophone, enumerationIndex);
                RequestSettingsSave();
            }
        }
    }

    private void LanguageComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_languageComboBox?.SelectedItem is ComboBoxItem item && item.Tag is string code)
        {
            _selectedLanguageCode = code;
            _appSettings.Language = code;
            // Language changes are rare and cheap to persist; save immediately so a crash
            // before clean shutdown does not lose the choice.
            SaveAppSettings();
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
                RequestSettingsSave();
            }
        }
    }

    private void StartRecordingButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            if (_selectedMicrophone != null)
            {
                // Starting a new recording used to silently wipe the existing transcript. Ask
                // first if there is anything to lose.
                if (_transcriptionHistory.Count > 0)
                {
                    var confirmClear = MessageBox.Show(
                        "Clear current transcript and start new recording?",
                        "Start New Recording",
                        MessageBoxButton.YesNo,
                        MessageBoxImage.Question);

                    if (confirmClear != MessageBoxResult.Yes)
                    {
                        return;
                    }
                }

                // Clear previous transcriptions
                _transcriptionPanel?.Children.Clear();
                _transcriptionHistory.Clear();
                if (_saveTranscriptionButton != null)
                {
                    _saveTranscriptionButton.IsEnabled = false;
                }
                // A fresh recording has nothing to filter yet - clear any leftover search query
                // so segments from the new recording don't start out hidden by an old filter.
                _searchDebounceTimer?.Stop();
                if (_searchTextBox != null) _searchTextBox.Text = "";
                _currentSearchQuery = "";
                if (_searchMatchCountText != null) _searchMatchCountText.Text = "";
                ResetAudioSource(_micSource);
                ResetAudioSource(_systemSource);
                _systemAudioConverter.Reset();
                _systemAudioFormatUnsupported = false;

                // Fresh per-recording session audio (for segment playback). Never under
                // Legal-Safe Mode - keeping raw audio would contradict "no verbatim
                // transcription stored" - and off when the user disabled it.
                StopSegmentPlayback();
                _sessionAudio?.Dispose();
                _sessionAudio = (_appSettings.SaveSessionAudio && !_privacyModeEnabled)
                    ? new SessionAudioWriter()
                    : null;

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
                if (_transcribeFileButton != null) _transcribeFileButton.IsEnabled = false;
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
            
            // Flush whatever is still buffered for each source. Each flush waits (off the UI
            // thread) for that source's in-flight request to finish rather than dropping the
            // tail of the recording, and the two sources are flushed independently.
            FlushSourceOnStop(_micSource);
            FlushSourceOnStop(_systemSource);

            if (_statusText != null) _statusText.Text = "Recording stopped. Transcription complete.";
            if (_stopRecordingButton != null) _stopRecordingButton.IsEnabled = false;
            if (_startRecordingButton != null) _startRecordingButton.IsEnabled = true;
            if (_transcribeFileButton != null) _transcribeFileButton.IsEnabled = !_isFileTranscriptionRunning;
            
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
                                // Note: negate in float space - Math.Abs(short.MinValue) throws.
                                float magnitude = sample / 32768f;
                                if (magnitude < 0f) magnitude = -magnitude;
                                if (magnitude > level) level = magnitude;
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
        
        // Apply/remove legal-safe mode to every already-displayed card in place. Privacy mode
        // changes what every segment SHOWS (speaker name + text) but doesn't add, remove, or
        // reorder any segment, so patching each card is enough - no full rebuild needed.
        foreach (var segment in _transcriptionHistory)
        {
            UpdateSegmentCardInPlace(segment);
        }
        RequestSettingsSave();
    }

    private void WaveIn_DataAvailable(object? sender, WaveInEventArgs e)
    {
        if (e.BytesRecorded <= 0)
        {
            return;
        }

        // The microphone is opened as 16 kHz mono 16-bit PCM, so every two bytes is one sample
        // and the buffer can go straight to the backend once the (optional) gain is applied.
        float max = 0f;
        bool applyGain = Math.Abs(_microphoneGain - 1.0f) > 0.0001f;

        for (int index = 0; index + 1 < e.BytesRecorded; index += 2)
        {
            short sample = (short)((e.Buffer[index + 1] << 8) | e.Buffer[index]);

            if (applyGain)
            {
                // Clamped so loud input saturates instead of wrapping around to the
                // opposite sign (which is what an unchecked cast would do).
                sample = ClampToInt16(sample * _microphoneGain);
                e.Buffer[index] = (byte)(sample & 0xFF);
                e.Buffer[index + 1] = (byte)((sample >> 8) & 0xFF);
            }

            var sample32 = sample / 32768f;
            if (sample32 < 0) sample32 = -sample32;
            if (sample32 > max) max = sample32;
        }

        _microphoneLevel = max;

        // Add audio data to the microphone buffer for transcription
        AppendAudioToSource(_micSource, e.Buffer, e.BytesRecorded);
    }

    private void WaveIn_RecordingStopped(object? sender, StoppedEventArgs e)
    {
        _waveIn?.Dispose();
        _waveIn = null;
        _microphoneLevel = 0f;
    }
    
    private void SystemAudio_DataAvailable(object? sender, WaveInEventArgs e)
    {
        if (e.BytesRecorded <= 0 || _systemAudioFormatUnsupported)
        {
            return;
        }

        // WASAPI loopback hands us whatever the endpoint's mix format is - commonly 32-bit
        // IEEE float, stereo, at 44.1 kHz or 48 kHz. Convert it to 16 kHz mono 16-bit PCM
        // here, BEFORE buffering, so the WAV header written later actually describes the
        // bytes it wraps. Nothing is hardcoded: the real format is read from the capture.
        WaveFormat? sourceFormat = (sender as IWaveIn)?.WaveFormat ?? _systemAudioCapture?.WaveFormat;
        if (sourceFormat == null)
        {
            return;
        }

        byte[] converted;
        try
        {
            converted = _systemAudioConverter.ConvertToPcm16Mono16k(sourceFormat, e.Buffer, e.BytesRecorded);
        }
        catch (NotSupportedException ex)
        {
            // Log once and stop capturing system audio rather than shipping garbage bytes
            // to the backend and polluting the transcript with hallucinated text.
            _systemAudioFormatUnsupported = true;
            Console.WriteLine($"System audio format not supported ({ex.Message}); system audio will be skipped for this session.");
            Dispatcher.BeginInvoke(new Action(() =>
            {
                if (_statusText != null)
                {
                    _statusText.Text = "System audio format unsupported - transcribing microphone only.";
                }
            }));
            return;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"System audio conversion error: {ex.Message}");
            return;
        }

        if (converted.Length > 0)
        {
            AppendAudioToSource(_systemSource, converted, converted.Length);
        }
    }

    private void SystemAudio_RecordingStopped(object? sender, StoppedEventArgs e)
    {
        _systemAudioCapture?.Dispose();
        _systemAudioCapture = null;
    }
    
    private void TranscriptionTimer_Tick(object? sender, EventArgs e)
    {
        if (!_isRecording)
        {
            return;
        }

        // Each source is dispatched independently. A source that is still waiting on the
        // backend keeps its buffered audio (it is NOT cleared) and goes out on a later tick.
        DispatchPendingAudio(_micSource);
        DispatchPendingAudio(_systemSource);
    }

    /// <summary>
    /// Sends the audio buffered for one source, if any, and if that source does not already
    /// have a request in flight. The buffer is drained only at the moment we actually take
    /// ownership of the bytes, so audio can never be discarded by a guard further downstream.
    /// </summary>
    private void DispatchPendingAudio(AudioSourceState state)
    {
        byte[] chunk;
        double chunkStartSeconds;

        lock (state.Sync)
        {
            if (state.IsProcessing || state.Buffer.Count == 0)
            {
                // Still busy (or nothing to send): leave the buffer intact for the next tick.
                return;
            }

            double bufferedSeconds = state.Buffer.Count / (double)TRANSCRIPTION_BYTES_PER_SECOND;

            if (!_backendReachable)
            {
                // Backend not up yet (auto-started uvicorn loading models) or gone away: hold
                // the audio instead of firing it at a closed port and losing it. The buffer
                // keeps accumulating (AppendAudioToSource drops-oldest past
                // MAX_BUFFERED_AUDIO_BYTES, with the timeline kept honest) and the health poll
                // reopens this gate the moment the backend answers, at which point everything
                // held goes out through the normal cut logic below.
                if (!state.HoldLogged)
                {
                    state.HoldLogged = true;
                    Console.WriteLine($"Backend not reachable; holding {bufferedSeconds:F1}s of {state.DisplayName} audio until it is.");
                }
                return;
            }

            if (state.HoldLogged)
            {
                state.HoldLogged = false;
                Console.WriteLine($"Backend reachable again; releasing {bufferedSeconds:F1}s of held {state.DisplayName} audio.");
            }

            if (bufferedSeconds < MAX_CHUNK_SECONDS)
            {
                if (bufferedSeconds < MIN_CHUNK_SECONDS)
                {
                    // Too little audio to be worth a request; keep accumulating.
                    return;
                }

                if (!PcmTailIsSilent(state.Buffer))
                {
                    // Mid-utterance: wait for a natural pause so the cut does not
                    // land mid-word. The MAX_CHUNK_SECONDS cap above bounds how
                    // long an uninterrupted speaker can defer the flush.
                    return;
                }

                if (PcmIsAllSilent(state.Buffer))
                {
                    // Nothing but silence buffered (e.g. an idle microphone).
                    // Drop it instead of posting dead air, but keep the timeline
                    // accounting intact - these bytes were still consumed.
                    state.ConsumedBytes += state.Buffer.Count;
                    state.Buffer.Clear();
                    return;
                }
            }

            // How much of the buffer leaves in this chunk. Normally all of it (the
            // tail-silence gate above means the buffer already ends on a pause), but a
            // cap flush lands here mid-speech: then prefer the most recent pause WITHIN
            // the buffer as the cut point and carry the tail into the next chunk, so the
            // boundary does not split a word. No pause anywhere => send everything,
            // exactly as before.
            int cutBytes = state.Buffer.Count;
            if (bufferedSeconds >= MAX_CHUNK_SECONDS)
            {
                cutBytes = FindPauseCutOffset(state.Buffer);
            }

            if (cutBytes >= state.Buffer.Count)
            {
                chunk = state.Buffer.ToArray();
                state.Buffer.Clear();
            }
            else
            {
                chunk = new byte[cutBytes];
                state.Buffer.CopyTo(0, chunk, 0, cutBytes);
                state.Buffer.RemoveRange(0, cutBytes);
            }

            // Everything consumed before this chunk IS this chunk's start on the recording
            // timeline. Captured under the same lock that drains the buffer so two dispatches
            // can never be handed the same offset. Adding only chunk.Length (not the whole
            // pre-cut buffer) keeps the carried-forward tail's timeline accounting intact.
            chunkStartSeconds = state.ConsumedBytes / (double)TRANSCRIPTION_BYTES_PER_SECOND;
            state.ConsumedBytes += chunk.Length;

            state.IsProcessing = true; // Released in ProcessAudioChunkAsync's finally block
        }

        // Fire and forget - the returned task clears state.IsProcessing when it completes.
        _ = ProcessAudioChunkAsync(chunk, state, chunkStartSeconds);
    }

    /// <summary>
    /// Final flush when recording stops: hand off to an async helper so the UI thread is never
    /// blocked, and so a request already in flight for this source is awaited rather than
    /// causing the tail of the recording to be thrown away.
    /// </summary>
    private void FlushSourceOnStop(AudioSourceState state)
    {
        _ = FlushSourceOnStopAsync(state);
    }

    private async Task FlushSourceOnStopAsync(AudioSourceState state)
    {
        // Bounded retry so a wedged backend cannot leave this task alive forever.
        const int maxAttempts = 60;
        const int retryDelayMs = 500;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            byte[] chunk = Array.Empty<byte>();
            double chunkStartSeconds = 0;

            lock (state.Sync)
            {
                if (state.Buffer.Count == 0)
                {
                    return; // Nothing left to flush
                }

                // Also wait out an unreachable backend (same bounded loop): stopping while the
                // auto-started backend is still loading models should deliver the tail once it
                // comes up, not throw the tail at a closed port.
                if (!state.IsProcessing && _backendReachable)
                {
                    chunk = state.Buffer.ToArray();
                    state.Buffer.Clear();

                    // Same bookkeeping as DispatchPendingAudio: this tail chunk starts where
                    // everything consumed so far ended.
                    chunkStartSeconds = state.ConsumedBytes / (double)TRANSCRIPTION_BYTES_PER_SECOND;
                    state.ConsumedBytes += chunk.Length;

                    state.IsProcessing = true;
                }
            }

            if (chunk.Length > 0)
            {
                await ProcessAudioChunkAsync(chunk, state, chunkStartSeconds);
                return;
            }

            // A request for this source is still in flight; wait for it and retry.
            await Task.Delay(retryDelayMs).ConfigureAwait(false);
        }

        Console.WriteLine($"Final flush for {state.DisplayName} gave up waiting for the backend; buffered audio discarded.");
        lock (state.Sync)
        {
            state.Buffer.Clear();
        }
    }

    /// <summary>
    /// Sends one chunk to /transcribe and turns the response into transcript segments.
    ///
    /// chunkStartSeconds is where this chunk begins on the recording's timeline (see
    /// AudioSourceState.ConsumedBytes). The backend transcribes each uploaded WAV in isolation,
    /// so the start/end it returns are offsets WITHIN the chunk and every chunk restarts near
    /// zero; adding chunkStartSeconds here is what makes TranscriptionSegment.StartTime/EndTime
    /// recording-relative, which the on-screen timestamps and the SRT/VTT/JSON exports all rely
    /// on being true.
    /// </summary>
    private async Task ProcessAudioChunkAsync(byte[] audioData, AudioSourceState state, double chunkStartSeconds)
    {
        string source = state.DisplayName;

        // Tee the outgoing PCM to the session recording so segments can be replayed
        // later - the same bytes the backend hears, at the same timeline offsets.
        // Never under Legal-Safe Mode: "no verbatim transcription stored" has to
        // extend to the raw audio, or the mode is a fig leaf.
        if (!_privacyModeEnabled)
        {
            _sessionAudio?.Write(source, chunkStartSeconds, audioData);
        }

        try
        {
            Console.WriteLine($"Processing audio chunk of {audioData.Length} bytes from {source}...");

            // Convert raw audio data to WAV format
            var wavData = CreateWavFile(audioData, TRANSCRIPTION_SAMPLE_RATE, TRANSCRIPTION_CHANNELS);
            Console.WriteLine($"Created WAV file of {wavData.Length} bytes");

            // Send to backend for transcription
            using var content = new MultipartFormDataContent();
            using var audioContent = new ByteArrayContent(wavData);
            audioContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("audio/wav");
            content.Add(audioContent, "audio", $"{source}.wav");

            // Tag the chunk's origin so the backend can tell the near end from the far end.
            // Unknown query parameters are tolerated by the backend, so this stays compatible.
            // "language" pins the decode language ("auto" = per-chunk detection for mixed-
            // language sessions); read from the volatile mirror, never the ComboBox itself.
            // A session roster caps how many distinct speakers diarization may report.
            var requestUrl = $"{_backendUrl}/transcribe?source={Uri.EscapeDataString(state.SourceTag)}&language={Uri.EscapeDataString(_selectedLanguageCode)}";
            if (_sessionRoster.Count > 0)
            {
                requestUrl += $"&max_speakers={_sessionRoster.Count}";
            }

            Console.WriteLine("Sending request to backend...");
            var response = await _httpClient!.PostAsync(requestUrl, content);

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
                        // "end" is part of the documented /transcribe contract, but fall back to
                        // startTime (zero-duration) rather than throwing if an older/odd backend
                        // response ever omits it - export formats (SRT/VTT) need SOME end value.
                        var endTime = segment.TryGetProperty("end", out var endProp) ? endProp.GetDouble() : startTime;

                        // Chunk-relative -> recording-relative. Without this every chunk's
                        // segments would sit at ~0-5s, making the timestamps meaningless and
                        // every SRT/VTT cue overlap inside the first five seconds.
                        var absoluteStart = chunkStartSeconds + startTime;
                        var absoluteEnd = chunkStartSeconds + endTime;

                        Console.WriteLine($"Segment: Speaker='{speaker}', Text='{text}', StartTime={absoluteStart} (chunk-relative {startTime}), EndTime={absoluteEnd}");

                        if (!string.IsNullOrWhiteSpace(text))
                        {
                            // Add transcription to UI (run on UI thread)
                            Dispatcher.Invoke(() => AddTranscriptionSegment(speaker, text, absoluteStart, absoluteEnd, source));
                        }
                    }
                }
                else
                {
                    Console.WriteLine("No 'segments' property found in response");
                }

                ReportBackendSuccess();
            }
            else
            {
                Console.WriteLine($"Backend request failed with status: {response.StatusCode}");
                var errorContent = await response.Content.ReadAsStringAsync();
                Console.WriteLine($"Error content: {errorContent}");
                ReportBackendFailure($"HTTP {(int)response.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ProcessAudioChunkAsync error: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");

            // Visible error state instead of console-only: flips the status indicator to
            // Offline immediately and triggers an out-of-band health check.
            ReportBackendFailure(ex.Message);

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
            // Release only THIS source. The other source's capture and dispatch are unaffected.
            lock (state.Sync)
            {
                state.IsProcessing = false;
            }
        }
    }

    /// <summary>
    /// Picks an audio file and runs it through the same /transcribe pipeline as live audio.
    /// Segments land in the transcript view with file-relative timestamps, so display
    /// merging, exports, search and keyword alerts all work unchanged. Mutually exclusive
    /// with live recording (see _isFileTranscriptionRunning) - mixing file-relative and
    /// recording-relative timestamps in one transcript would corrupt SRT/VTT exports.
    /// </summary>
    private async void TranscribeFileButton_Click(object sender, RoutedEventArgs e)
    {
        if (_isRecording || _isFileTranscriptionRunning)
        {
            return;
        }

        var dialog = new Microsoft.Win32.OpenFileDialog
        {
            Title = "Choose an audio file to transcribe",
            Filter = "Audio files (*.wav;*.mp3;*.flac;*.ogg;*.m4a)|*.wav;*.mp3;*.flac;*.ogg;*.m4a|All files (*.*)|*.*"
        };
        if (dialog.ShowDialog() != true)
        {
            return;
        }

        var filePath = dialog.FileName;
        var fileName = System.IO.Path.GetFileName(filePath);

        _isFileTranscriptionRunning = true;
        if (_transcribeFileButton != null)
        {
            _transcribeFileButton.IsEnabled = false;
            _transcribeFileButton.Content = "⏳ Transcribing...";
        }
        if (_startRecordingButton != null) _startRecordingButton.IsEnabled = false;
        if (_statusText != null)
        {
            _statusText.Text = $"Transcribing '{fileName}'... long files can take a few minutes.";
        }

        try
        {
            byte[] fileBytes = await System.IO.File.ReadAllBytesAsync(filePath);

            using var content = new MultipartFormDataContent();
            var audioContent = new ByteArrayContent(fileBytes);
            audioContent.Headers.ContentType =
                new System.Net.Http.Headers.MediaTypeHeaderValue("application/octet-stream");
            content.Add(audioContent, "audio", fileName);

            // "file" is normalized to an unscoped probe by the backend: file audio is
            // neither the near end nor the far end, so voices enrolled from either
            // source may match. The language selection applies to files the same as
            // to live chunks. accuracy=true buys a better decode (higher beam size,
            // optionally a stronger model) - offline files have no latency pressure.
            // A session roster caps how many distinct speakers diarization may report.
            var requestUrl = $"{_backendUrl}/transcribe?source=file&language={Uri.EscapeDataString(_selectedLanguageCode)}&accuracy=true";
            if (_sessionRoster.Count > 0)
            {
                requestUrl += $"&max_speakers={_sessionRoster.Count}";
            }

            // The shared client's 60s timeout is tuned for short live chunks; a long
            // file legitimately transcribes for minutes.
            using var fileClient = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
            var response = await fileClient.PostAsync(requestUrl, content);

            if (!response.IsSuccessStatusCode)
            {
                var error = await response.Content.ReadAsStringAsync();
                if (error.Length > 200) error = error.Substring(0, 200);
                if (_statusText != null)
                {
                    _statusText.Text = $"File transcription failed (HTTP {(int)response.StatusCode}): {error}";
                }
                return;
            }

            var jsonResponse = await response.Content.ReadAsStringAsync();
            var transcriptionResult = JsonSerializer.Deserialize<JsonElement>(jsonResponse);

            int added = 0;
            if (transcriptionResult.TryGetProperty("segments", out var segments))
            {
                foreach (var segment in segments.EnumerateArray())
                {
                    var text = segment.GetProperty("text").GetString();
                    var speaker = segment.GetProperty("speaker").GetString();
                    var startTime = segment.GetProperty("start").GetDouble();
                    var endTime = segment.TryGetProperty("end", out var endProp)
                        ? endProp.GetDouble()
                        : startTime;

                    if (string.IsNullOrWhiteSpace(text))
                    {
                        continue;
                    }

                    // File timestamps are already absolute within the file - no chunk
                    // offset to add, unlike the live path.
                    AddTranscriptionSegment(speaker, text, startTime, endTime, $"File: {fileName}", filePath);
                    added++;
                }
            }

            ReportBackendSuccess();
            if (_statusText != null)
            {
                _statusText.Text = added > 0
                    ? $"Transcribed '{fileName}': {added} segments."
                    : $"'{fileName}' contained no recognizable speech.";
            }
        }
        catch (Exception ex)
        {
            ReportBackendFailure(ex.Message);
            if (_statusText != null)
            {
                _statusText.Text = $"File transcription error: {ex.Message}";
            }
        }
        finally
        {
            _isFileTranscriptionRunning = false;
            if (_transcribeFileButton != null)
            {
                _transcribeFileButton.Content = "📂 Transcribe File";
                _transcribeFileButton.IsEnabled = !_isRecording;
            }
            if (_startRecordingButton != null) _startRecordingButton.IsEnabled = !_isRecording;
        }
    }

    private void AddTranscriptionSegment(string? speaker, string? text, double startTime, double endTime, string source,
        string? audioFilePath = null)
    {
        Console.WriteLine($"AddTranscriptionSegment called: Speaker='{speaker}', Text='{text}', StartTime={startTime}, EndTime={endTime}, Source={source}");

        if (_transcriptionPanel == null || string.IsNullOrWhiteSpace(text))
        {
            Console.WriteLine("Skipping - transcription panel is null or text is empty");
            return;
        }

        // Consecutive fragments from the same speaker and source read as one
        // utterance, so absorb this fragment into the previous card when the two
        // are nearly contiguous - instead of stacking one card per audio chunk.
        // Skipped while that card's text is being edited (the in-place update
        // would be suppressed and history would silently diverge from the UI).
        var previous = _transcriptionHistory.Count > 0
            ? _transcriptionHistory[_transcriptionHistory.Count - 1]
            : null;
        if (previous != null
            && previous.Source == source
            && string.Equals(previous.Speaker, speaker, StringComparison.Ordinal)
            && startTime >= previous.StartTime
            && (startTime - previous.EndTime) <= SEGMENT_MERGE_MAX_GAP_SECONDS
            && (previous.Text?.Length ?? 0) + text.Length <= SEGMENT_MERGE_MAX_CHARS
            && previous.SplitSegments == null
            && (previous.TextDisplayElement?.IsReadOnly ?? true))
        {
            previous.Text = string.IsNullOrWhiteSpace(previous.Text)
                ? text
                : $"{previous.Text} {text}";
            previous.EndTime = Math.Max(previous.EndTime, endTime);

            UpdateSegmentCardInPlace(previous);
            if (ApplyKeywordAlertHighlight(previous))
            {
                FlashStatusTextForKeywordAlert();
            }
            return;
        }

        // Generate unique segment ID (monotonic counter - see _nextSegmentId)
        var segmentId = _nextSegmentId++;

        // Store in transcription history
        var segment = new TranscriptionSegment
        {
            Speaker = speaker,
            Text = text,
            StartTime = startTime,
            EndTime = endTime,
            Source = source,
            Timestamp = DateTime.Now,
            SegmentId = segmentId,
            // File transcriptions pass their media file; live segments resolve to the
            // session recording for their source (null when session audio is off).
            AudioFilePath = audioFilePath ?? _sessionAudio?.GetPathIfWritten(source)
        };
        _transcriptionHistory.Add(segment);
        
        // Ensure speaker is in available speakers list
        if (!string.IsNullOrEmpty(speaker) && !_availableSpeakers.Contains(speaker))
        {
            _availableSpeakers.Add(speaker);
            // Save settings when new speaker is auto-detected
            SaveAppSettings();
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
        
        // Keyword-alert bell: hidden unless this segment matches a configured alert keyword
        // (ApplyKeywordAlertHighlight below sets its Visibility - see also UpdateSegmentCardInPlace).
        var alertBellIcon = new TextBlock
        {
            Text = "🔔",
            FontSize = 14,
            Margin = new Thickness(0, 0, 6, 0),
            VerticalAlignment = VerticalAlignment.Center,
            Visibility = Visibility.Collapsed,
            ToolTip = "Matches a keyword alert"
        };

        // Selection checkbox
        var selectionCheckBox = CreateSelectionCheckBox(segmentId);

        // Timestamp (recording-relative - see ProcessAudioChunkAsync's chunkStartSeconds)
        var timestampText = new TextBlock
        {
            Text = $"[{FormatClockTimestamp(startTime)}]",
            FontWeight = FontWeights.Bold,
            Foreground = Brushes.Gray,
            Width = 70,
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
            // Ignore selection changes we caused ourselves while patching the card's display.
            if (_suppressSpeakerSelectionChanged)
                return;

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
                    SaveAppSettings();
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
                // Roster members map onto the quick buttons in roster order; without a
                // roster the legacy "Speaker N" placeholders apply.
                var newSpeaker = speakerNum <= _sessionRoster.Count
                    ? _sessionRoster[speakerNum - 1]
                    : $"Speaker {speakerNum}";
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

        // Rename speaker button. The dropdown next to it reassigns THIS segment to a different
        // speaker; this renames the speaker itself everywhere they appear. segment.Speaker is
        // read at click time (not the captured parameter) so a later reassignment is respected.
        var renameSpeakerButton = new Button
        {
            Content = "✏",
            Width = 25,
            Height = 25,
            Margin = new Thickness(2, 0, 2, 0),
            FontSize = 10,
            Background = Brushes.Lavender,
            ToolTip = "Rename this speaker everywhere they appear"
        };

        renameSpeakerButton.Click += (s, e) => ShowSpeakerRenameDialog(segment.Speaker);

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
        topPanel.Children.Add(alertBellIcon);
        topPanel.Children.Add(selectionCheckBox);
        topPanel.Children.Add(CreatePlaySegmentButton(segment));
        topPanel.Children.Add(timestampText);
        topPanel.Children.Add(speakerComboBox);
        topPanel.Children.Add(emotionIndicator);
        topPanel.Children.Add(sourceText);
        topPanel.Children.Add(quickSpeakerPanel);
        topPanel.Children.Add(newSpeakerButton);
        topPanel.Children.Add(showAutoSpeakersButton);
        topPanel.Children.Add(renameSpeakerButton);
        topPanel.Children.Add(deleteSpeakerButton);

        // Editable text content (in its own row for better readability)
        var textDisplay = CreateEditableTextDisplay(segment);
        
        // Add both rows to the segment panel
        segmentPanel.Children.Add(topPanel);
        segmentPanel.Children.Add(textDisplay);

        // Speaker stripe + click-to-select + context menu, then attach to the panel.
        AttachCardChrome(segment, segmentBorder, segmentPanel);
        _transcriptionPanel.Children.Add(segmentBorder);

        // Cache the card elements on the segment itself so a later rename, privacy toggle, or
        // keyword-alert change can patch this card in place (UpdateSegmentCardInPlace /
        // ApplyKeywordAlertHighlight) instead of requiring a full display rebuild.
        segment.CardElement = segmentBorder;
        segment.SpeakerComboBoxElement = speakerComboBox;
        segment.TextDisplayElement = textDisplay;
        segment.AlertBellElement = alertBellIcon;
        segment.SelectionCheckBoxElement = selectionCheckBox;

        // Highlight + bell prefix if this segment matches a configured alert keyword, and flash
        // the status text so a live match is noticeable even if the transcript isn't in view.
        if (ApplyKeywordAlertHighlight(segment))
        {
            FlashStatusTextForKeywordAlert();
        }

        // If a search filter is active, a newly-arrived segment must be evaluated against it too
        // (this only flips Visibility on existing elements - it doesn't rebuild any cards).
        if (!string.IsNullOrEmpty(_currentSearchQuery))
        {
            ApplySearchFilter(_currentSearchQuery);
        }

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
                SaveAppSettings();
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

            // Guests are session-only by contract: no name mapping, no enrollment, no
            // voiceprint. The label lives purely in this transcript.
            if (_sessionGuestNames.Contains(correctSpeakerName))
            {
                Console.WriteLine($"'{correctSpeakerName}' is a session guest - skipping backend feedback");
                return;
            }
            
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
                var response = await _httpClient!.PostAsync($"{_backendUrl}/speakers/name_mapping?old_speaker_id={Uri.EscapeDataString(segment.Speaker ?? "")}&new_speaker_name={Uri.EscapeDataString(correctSpeakerName)}", null);
                
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
                    SaveAppSettings();
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

        // Everything below is a PROGRAMMATIC repopulate of the dropdowns, exactly like
        // UpdateSegmentCardInPlace: assigning ItemsSource resets SelectedItem and assigning
        // SelectedItem raises SelectionChanged, which the per-segment handler treats as the user
        // correcting that segment's speaker (rewriting segment.Speaker to the *display* name,
        // persisting a bogus _speakerNames entry and POSTing a phantom /speakers/name_mapping).
        // A refresh must not look like N speaker corrections, so suppress the handler throughout.
        // The previous value is saved/restored rather than blindly cleared so a caller that is
        // itself suppressing (e.g. a rename flow) stays suppressed afterwards.
        bool previousSuppress = _suppressSpeakerSelectionChanged;
        _suppressSpeakerSelectionChanged = true;
        try
        {
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
        finally
        {
            _suppressSpeakerSelectionChanged = previousSuppress;
        }
    }
    
    /// <summary>
    /// Renames one underlying speaker everywhere it appears, by adding/updating its entry in
    /// _speakerNames rather than rewriting any segment's raw Speaker id. Reached from the "✏"
    /// button on each segment card; distinct from the card's speaker dropdown, which reassigns
    /// only that one segment.
    /// </summary>
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

                // The new name has to exist in the dropdown list before any card selects it -
                // a ComboBox silently coerces a SelectedItem that isn't in its ItemsSource to
                // null, which would blank out every renamed card's speaker box.
                if (!_availableSpeakers.Contains(newName))
                {
                    _availableSpeakers.Add(newName);
                    RefreshAllSpeakerDropdowns();
                }

                // Only the segments spoken by this raw speaker ID display differently now - patch
                // just their cards in place rather than rebuilding the whole transcript.
                foreach (var affectedSegment in _transcriptionHistory.Where(s => s.Speaker == originalSpeaker))
                {
                    UpdateSegmentCardInPlace(affectedSegment);
                }

                // Save settings when speaker is renamed
                SaveAppSettings();
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
            IsChecked = _selectedSegments.Contains(segmentId),
            ToolTip = "Select this segment (Shift+click a card selects a range, Ctrl+click toggles)"
        };

        // The HashSet add/remove and the visual refresh are both idempotent, so these
        // handlers are safe to fire from programmatic IsChecked changes too - which is
        // exactly how SetSegmentSelected keeps everything in sync through one path.
        checkbox.Checked += (s, e) => {
            _selectedSegments.Add(segmentId);
            _selectionAnchorId = segmentId;
            RefreshCardSelectionVisual(segmentId);
            UpdateMultiSelectButtons();
        };
        checkbox.Unchecked += (s, e) => {
            _selectedSegments.Remove(segmentId);
            RefreshCardSelectionVisual(segmentId);
            UpdateMultiSelectButtons();
        };

        _segmentCheckBoxes.Add(checkbox);
        return checkbox;
    }

    // --- Selection model --------------------------------------------------------------
    // Selection is always active. The checkbox is the explicit control; the card itself
    // is the fast path: plain click selects just that card, Ctrl+click toggles it,
    // Shift+click extends from the anchor. Clicks that land on interactive children
    // (buttons, the speaker dropdown, the text box) never count as selection clicks.

    private void RefreshCardSelectionVisual(int segmentId)
    {
        var segment = _transcriptionHistory.FirstOrDefault(s => s.SegmentId == segmentId);
        if (segment != null)
        {
            // ApplyKeywordAlertHighlight owns the card background and is selection-aware.
            ApplyKeywordAlertHighlight(segment);
        }
    }

    private void SetSegmentSelected(TranscriptionSegment segment, bool selected)
    {
        var checkbox = segment.SelectionCheckBoxElement;
        if (checkbox != null)
        {
            checkbox.IsChecked = selected; // fires the handlers above, which do the rest
        }
        else if (selected)
        {
            _selectedSegments.Add(segment.SegmentId);
        }
        else
        {
            _selectedSegments.Remove(segment.SegmentId);
        }
    }

    /// <summary>True when a click's original source sits inside an interactive control of the card.</summary>
    private static bool ClickLandedOnControl(object originalSource, Border card)
    {
        var node = originalSource as DependencyObject;
        while (node != null && !ReferenceEquals(node, card))
        {
            if (node is Button || node is ComboBox || node is ComboBoxItem
                || node is TextBox || node is CheckBox || node is System.Windows.Controls.Primitives.ScrollBar)
            {
                return true;
            }
            node = node is Visual || node is System.Windows.Media.Media3D.Visual3D
                ? VisualTreeHelper.GetParent(node)
                : LogicalTreeHelper.GetParent(node);
        }
        return false;
    }

    private void HandleCardClick(int segmentId, MouseButtonEventArgs e)
    {
        var segment = _transcriptionHistory.FirstOrDefault(s => s.SegmentId == segmentId);
        if (segment == null) return;

        bool ctrl = Keyboard.Modifiers.HasFlag(ModifierKeys.Control);
        bool shift = Keyboard.Modifiers.HasFlag(ModifierKeys.Shift);

        if (shift && _selectionAnchorId.HasValue)
        {
            int anchorIndex = _transcriptionHistory.FindIndex(s => s.SegmentId == _selectionAnchorId.Value);
            int thisIndex = _transcriptionHistory.FindIndex(s => s.SegmentId == segmentId);
            if (anchorIndex >= 0 && thisIndex >= 0)
            {
                if (!ctrl)
                {
                    ClearAllSelections();
                }
                int from = Math.Min(anchorIndex, thisIndex);
                int to = Math.Max(anchorIndex, thisIndex);
                for (int i = from; i <= to; i++)
                {
                    var inRange = _transcriptionHistory[i];
                    // A search filter hides cards; a range never selects what the user can't see.
                    if (inRange.CardElement?.Visibility == Visibility.Collapsed) continue;
                    SetSegmentSelected(inRange, true);
                }
                // Anchor deliberately stays put so consecutive Shift+clicks re-range from
                // the same origin, matching Explorer/list-view semantics.
                return;
            }
        }

        if (ctrl)
        {
            SetSegmentSelected(segment, !_selectedSegments.Contains(segmentId));
            _selectionAnchorId = segmentId;
            return;
        }

        // Plain click: this card becomes the whole selection - unless it already is,
        // in which case clicking deselects it (an easy way out of a stray selection).
        bool wasOnlySelection = _selectedSegments.Count == 1 && _selectedSegments.Contains(segmentId);
        ClearAllSelections();
        if (!wasOnlySelection)
        {
            SetSegmentSelected(segment, true);
            _selectionAnchorId = segmentId;
        }
    }

    private void ClearAllSelections()
    {
        // Snapshot: SetSegmentSelected mutates _selectedSegments via the checkbox handlers.
        foreach (var id in _selectedSegments.ToList())
        {
            var segment = _transcriptionHistory.FirstOrDefault(s => s.SegmentId == id);
            if (segment != null)
            {
                SetSegmentSelected(segment, false);
            }
            else
            {
                _selectedSegments.Remove(id);
            }
        }
        UpdateMultiSelectButtons();
    }

    private void SelectAllFromSpeaker(string? rawSpeaker, bool additive)
    {
        var displayName = GetDisplaySpeakerName(rawSpeaker);
        if (!additive)
        {
            ClearAllSelections();
        }
        foreach (var segment in _transcriptionHistory)
        {
            if (segment.CardElement?.Visibility == Visibility.Collapsed) continue;
            if (GetDisplaySpeakerName(segment.Speaker) == displayName)
            {
                SetSegmentSelected(segment, true);
            }
        }
        UpdateMultiSelectButtons();
    }

    /// <summary>
    /// Deletes the given segments from the transcript: history, card UI, selection state,
    /// and the stale entries in the flat checkbox/dropdown tracking lists. Purely a
    /// transcript edit - speakers and their voiceprints are untouched.
    /// </summary>
    private void DeleteSegments(List<int> segmentIds)
    {
        foreach (var id in segmentIds)
        {
            var segment = _transcriptionHistory.FirstOrDefault(s => s.SegmentId == id);
            if (segment == null) continue;

            if (segment.CardElement != null)
            {
                _transcriptionPanel?.Children.Remove(segment.CardElement);
            }
            if (segment.SpeakerComboBoxElement != null)
            {
                _speakerComboBoxes.Remove(segment.SpeakerComboBoxElement);
            }
            if (segment.SelectionCheckBoxElement != null)
            {
                _segmentCheckBoxes.Remove(segment.SelectionCheckBoxElement);
            }
            _transcriptionHistory.Remove(segment);
            _selectedSegments.Remove(id);
            if (_selectionAnchorId == id)
            {
                _selectionAnchorId = null;
            }
        }

        if (_transcriptionHistory.Count == 0 && _saveTranscriptionButton != null)
        {
            _saveTranscriptionButton.IsEnabled = false;
        }
        UpdateMultiSelectButtons();
    }

    private void DeleteSelected_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedSegments.Count == 0) return;

        var count = _selectedSegments.Count;
        var confirm = MessageBox.Show(
            $"Delete {count} selected segment{(count == 1 ? "" : "s")} from the transcript?\n\n" +
            "This only removes the text segments - speakers and their voiceprints are not affected.",
            "Delete Segments", MessageBoxButton.YesNo, MessageBoxImage.Warning);
        if (confirm == MessageBoxResult.Yes)
        {
            DeleteSegments(_selectedSegments.ToList());
        }
    }

    private void SelectBySpeaker_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button button) return;

        // Build the menu fresh each time from the speakers actually present, with counts.
        var menu = new ContextMenu();
        var groups = _transcriptionHistory
            .GroupBy(s => GetDisplaySpeakerName(s.Speaker))
            .OrderByDescending(g => g.Count())
            .ToList();

        if (groups.Count == 0)
        {
            menu.Items.Add(new MenuItem { Header = "(no segments yet)", IsEnabled = false });
        }
        foreach (var group in groups)
        {
            var item = new MenuItem { Header = $"{group.Key}  ({group.Count()})" };
            var rawSpeaker = group.First().Speaker;
            item.Click += (_, __) => SelectAllFromSpeaker(rawSpeaker,
                additive: Keyboard.Modifiers.HasFlag(ModifierKeys.Control));
            menu.Items.Add(item);
        }

        menu.PlacementTarget = button;
        menu.Placement = System.Windows.Controls.Primitives.PlacementMode.Bottom;
        menu.IsOpen = true;
    }

    /// <summary>
    /// Card chrome shared by both construction sites (live arrival and full rebuild):
    /// the per-speaker color stripe, click-to-select, and the card context menu.
    /// Sets segmentBorder.Child, so callers must not assign it themselves.
    /// </summary>
    private void AttachCardChrome(TranscriptionSegment segment, Border segmentBorder, StackPanel segmentPanel)
    {
        var stripe = new Rectangle
        {
            Width = 5,
            RadiusX = 2,
            RadiusY = 2,
            Fill = GetSpeakerColorEnhanced(GetDisplaySpeakerName(segment.Speaker)),
            Margin = new Thickness(0, 0, 6, 0)
        };
        segment.SpeakerStripeElement = stripe;

        var layout = new DockPanel();
        DockPanel.SetDock(stripe, Dock.Left);
        layout.Children.Add(stripe);
        layout.Children.Add(segmentPanel);
        segmentBorder.Child = layout;

        segmentBorder.MouseLeftButtonUp += (s, e) =>
        {
            // Clicks that land on the card's own controls (buttons, dropdown, text box,
            // checkbox) belong to those controls, not to selection.
            if (!ClickLandedOnControl(e.OriginalSource, segmentBorder))
            {
                HandleCardClick(segment.SegmentId, e);
            }
        };

        // Card context menu. segment.Speaker is read at click time, so a reassignment
        // between opening the transcript and using the menu is respected. The text box
        // keeps its own Edit/Split menu; this one covers the rest of the card.
        var menu = new ContextMenu();

        var selectSpeakerItem = new MenuItem { Header = "👤 Select all from this speaker" };
        selectSpeakerItem.Click += (_, __) => SelectAllFromSpeaker(segment.Speaker, additive: false);
        menu.Items.Add(selectSpeakerItem);

        var deleteItem = new MenuItem { Header = "🗑 Delete this segment" };
        deleteItem.Click += (_, __) =>
        {
            var confirm = MessageBox.Show(
                "Delete this segment from the transcript?",
                "Delete Segment", MessageBoxButton.YesNo, MessageBoxImage.Warning);
            if (confirm == MessageBoxResult.Yes)
            {
                DeleteSegments(new List<int> { segment.SegmentId });
            }
        };
        menu.Items.Add(deleteItem);

        segmentBorder.ContextMenu = menu;
    }

    // --- Segment audio playback -------------------------------------------------------
    // Live recordings are teed to per-source session WAVs (16 kHz mono 16-bit - the exact
    // stream the ASR consumes, ~115 MB/hour) so any segment can be replayed to verify the
    // transcription. File transcriptions replay straight from the original media file.

    private SessionAudioWriter? _sessionAudio;
    private WaveOutEvent? _segmentPlayer;
    private AudioFileReader? _segmentPlayerReader;
    private Button? _activePlayButton;
    private DispatcherTimer? _segmentPlaybackTimer;
    private double _segmentPlaybackEndSeconds;

    private Button CreatePlaySegmentButton(TranscriptionSegment segment)
    {
        var playButton = new Button
        {
            Content = "▶",
            Width = 25,
            Height = 25,
            Margin = new Thickness(0, 0, 6, 0),
            FontSize = 10,
            Background = Brushes.WhiteSmoke,
            ToolTip = "Play this segment's audio (click again to stop)"
        };
        playButton.Click += (s, e) => ToggleSegmentPlayback(segment, playButton);
        return playButton;
    }

    private void ToggleSegmentPlayback(TranscriptionSegment segment, Button playButton)
    {
        if (ReferenceEquals(_activePlayButton, playButton))
        {
            StopSegmentPlayback();
            return;
        }
        StopSegmentPlayback();

        if (_isRecording)
        {
            // The loopback capture would hear the playback and transcribe it right back
            // into the session. Verification is an after-the-fact activity anyway.
            if (_statusText != null) _statusText.Text = "Stop the recording before playing segments back.";
            return;
        }

        var path = segment.AudioFilePath;
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            if (_statusText != null)
            {
                _statusText.Text = _privacyModeEnabled
                    ? "No audio saved for this segment (Legal-Safe Mode does not keep session audio)."
                    : "No saved audio for this segment - enable 'Save session audio' before recording.";
            }
            return;
        }

        try
        {
            _segmentPlayerReader = new AudioFileReader(path);
            var start = TimeSpan.FromSeconds(Math.Max(0, segment.StartTime));
            if (start < _segmentPlayerReader.TotalTime)
            {
                _segmentPlayerReader.CurrentTime = start;
            }
            // Small pad so a word that runs slightly past the segment boundary isn't clipped.
            _segmentPlaybackEndSeconds = segment.EndTime + 0.25;

            var player = new WaveOutEvent();
            _segmentPlayer = player;
            player.Init(_segmentPlayerReader);
            player.PlaybackStopped += (s, e) => Dispatcher.BeginInvoke(new Action(() =>
            {
                // Only react if this event belongs to the CURRENT player - a stale stop
                // event from a previous playback must not kill a newly started one.
                if (ReferenceEquals(_segmentPlayer, player))
                {
                    StopSegmentPlayback();
                }
            }));
            player.Play();

            _activePlayButton = playButton;
            playButton.Content = "⏹";

            if (_segmentPlaybackTimer == null)
            {
                _segmentPlaybackTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
                _segmentPlaybackTimer.Tick += (s, e) =>
                {
                    if (_segmentPlayerReader != null
                        && _segmentPlayerReader.CurrentTime.TotalSeconds >= _segmentPlaybackEndSeconds)
                    {
                        StopSegmentPlayback();
                    }
                };
            }
            _segmentPlaybackTimer.Start();
        }
        catch (Exception ex)
        {
            StopSegmentPlayback();
            if (_statusText != null) _statusText.Text = $"Playback failed: {ex.Message}";
        }
    }

    private void StopSegmentPlayback()
    {
        _segmentPlaybackTimer?.Stop();
        if (_activePlayButton != null)
        {
            _activePlayButton.Content = "▶";
            _activePlayButton = null;
        }
        var player = _segmentPlayer;
        _segmentPlayer = null; // cleared first so the PlaybackStopped callback sees it stale
        try { player?.Stop(); } catch { /* already stopped */ }
        player?.Dispose();
        _segmentPlayerReader?.Dispose();
        _segmentPlayerReader = null;
    }

    /// <summary>
    /// Per-recording session audio files: one WAV per source, 16 kHz mono 16-bit PCM.
    /// Chunks are written at their timeline byte offsets (offset = seconds * bytes/sec),
    /// so silence the dispatcher dropped stays as literal silence in the file and every
    /// segment's StartTime seeks to exactly the right audio. The header sizes are
    /// re-stamped after every write, so the file is valid WAV at all times - there is no
    /// finalize step to forget, and a crash mid-recording still leaves playable audio.
    /// </summary>
    private sealed class SessionAudioWriter : IDisposable
    {
        private readonly object _sync = new object();
        private readonly Dictionary<string, FileStream> _streams = new Dictionary<string, FileStream>();
        private readonly Dictionary<string, string> _paths = new Dictionary<string, string>();
        private readonly string _directory;
        private readonly string _stamp;
        private bool _disposed;

        public SessionAudioWriter()
        {
            _directory = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "Oreja Recordings");
            _stamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        }

        /// <summary>The session file a source has actually written to, or null if none yet.</summary>
        public string? GetPathIfWritten(string sourceName)
        {
            lock (_sync)
            {
                return _streams.ContainsKey(sourceName) && _paths.TryGetValue(sourceName, out var path)
                    ? path
                    : null;
            }
        }

        public void Write(string sourceName, double startSeconds, byte[] pcm)
        {
            if (pcm.Length == 0) return;
            lock (_sync)
            {
                if (_disposed) return;
                try
                {
                    if (!_streams.TryGetValue(sourceName, out var stream))
                    {
                        System.IO.Directory.CreateDirectory(_directory);
                        var safeName = string.Concat(sourceName.Split(System.IO.Path.GetInvalidFileNameChars()))
                            .Replace(' ', '-');
                        var path = System.IO.Path.Combine(_directory, $"Oreja_{_stamp}_{safeName}.wav");
                        stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.Read);
                        stream.Write(BuildWavHeader(0), 0, 44);
                        _paths[sourceName] = path;
                        _streams[sourceName] = stream;
                    }

                    long offset = 44 + (((long)Math.Round(startSeconds * TRANSCRIPTION_BYTES_PER_SECOND)) & ~1L);
                    stream.Seek(offset, SeekOrigin.Begin);
                    stream.Write(pcm, 0, pcm.Length);

                    long dataLength = Math.Max(0, stream.Length - 44);
                    var header = BuildWavHeader(dataLength);
                    stream.Seek(0, SeekOrigin.Begin);
                    stream.Write(header, 0, 44);
                    stream.Flush();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Session audio write failed ({sourceName}): {ex.Message}");
                }
            }
        }

        private static byte[] BuildWavHeader(long dataLength)
        {
            const int sampleRate = TRANSCRIPTION_SAMPLE_RATE;
            const short channels = 1;
            const short bitsPerSample = 16;
            using var ms = new MemoryStream(44);
            using var w = new BinaryWriter(ms);
            w.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            w.Write((uint)(36 + dataLength));
            w.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));
            w.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            w.Write(16);                                    // fmt chunk size
            w.Write((short)1);                              // PCM
            w.Write(channels);
            w.Write(sampleRate);
            w.Write(sampleRate * channels * (bitsPerSample / 8)); // byte rate
            w.Write((short)(channels * (bitsPerSample / 8)));     // block align
            w.Write(bitsPerSample);
            w.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            w.Write((uint)dataLength);
            return ms.ToArray();
        }

        public void Dispose()
        {
            lock (_sync)
            {
                _disposed = true;
                foreach (var stream in _streams.Values)
                {
                    try { stream.Dispose(); } catch { /* best effort */ }
                }
                _streams.Clear();
            }
        }
    }

    // --- Session speaker roster dialog --------------------------------------------------

    private void ShowSessionSpeakersDialog()
    {
        var dialog = new Window
        {
            Title = "Session Speakers",
            Width = 440,
            Height = 560,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = MainWindow,
            ResizeMode = ResizeMode.NoResize
        };

        var root = new StackPanel { Margin = new Thickness(15) };

        root.Children.Add(new TextBlock
        {
            Text = "Declare who is in this session so speaker identification doesn't have to guess. " +
                   "Known speakers keep learning voiceprints from your corrections; guests are " +
                   "session-only and never saved.",
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 0, 0, 12)
        });

        // Known speakers checklist
        root.Children.Add(new TextBlock
        {
            Text = "Known speakers present:",
            FontWeight = FontWeights.Bold,
            Margin = new Thickness(0, 0, 0, 5)
        });
        var knownPanel = new StackPanel();
        var knownNames = _availableSpeakers
            .Where(s => s != "Unknown"
                && !s.StartsWith("Speaker_AUTO_SPEAKER_")
                && !_sessionGuestNames.Contains(s))
            .Distinct()
            .OrderBy(s => s)
            .ToList();
        var knownChecks = new List<CheckBox>();
        foreach (var name in knownNames)
        {
            var check = new CheckBox
            {
                Content = name,
                Margin = new Thickness(2),
                IsChecked = _sessionRoster.Contains(name)
            };
            knownChecks.Add(check);
            knownPanel.Children.Add(check);
        }
        if (knownNames.Count == 0)
        {
            knownPanel.Children.Add(new TextBlock
            {
                Text = "(no saved speakers yet - name them during a session and they'll show up here)",
                FontStyle = FontStyles.Italic,
                Foreground = Brushes.Gray
            });
        }
        root.Children.Add(new ScrollViewer
        {
            Content = knownPanel,
            MaxHeight = 170,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            Margin = new Thickness(0, 0, 0, 12)
        });

        // Guests
        root.Children.Add(new TextBlock
        {
            Text = "Guest speakers (this session only, no voiceprint saved):",
            FontWeight = FontWeights.Bold,
            Margin = new Thickness(0, 0, 0, 5)
        });
        var guestNames = new List<string>(_sessionGuestNames.Where(g => _sessionRoster.Contains(g)));
        var guestPanel = new StackPanel();

        void RebuildGuestPanel()
        {
            guestPanel.Children.Clear();
            foreach (var guest in guestNames)
            {
                var row = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(2) };
                var removeButton = new Button
                {
                    Content = "×",
                    Width = 20,
                    Height = 20,
                    Margin = new Thickness(0, 0, 6, 0),
                    Background = Brushes.LightCoral
                };
                var captured = guest;
                removeButton.Click += (_, __) => { guestNames.Remove(captured); RebuildGuestPanel(); };
                row.Children.Add(removeButton);
                row.Children.Add(new TextBlock { Text = $"{guest}  (guest)", VerticalAlignment = VerticalAlignment.Center });
                guestPanel.Children.Add(row);
            }
        }
        RebuildGuestPanel();
        root.Children.Add(guestPanel);

        var addGuestRow = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 5, 0, 12) };
        var guestNameBox = new TextBox { Width = 220, Padding = new Thickness(4) };
        var addGuestButton = new Button
        {
            Content = "＋ Add Guest",
            Width = 100,
            Margin = new Thickness(8, 0, 0, 0),
            Background = Brushes.LightYellow
        };
        void AddGuest()
        {
            var name = guestNameBox.Text.Trim();
            if (name.Length == 0) return;
            bool duplicate = guestNames.Contains(name, StringComparer.OrdinalIgnoreCase)
                || knownNames.Contains(name, StringComparer.OrdinalIgnoreCase);
            if (!duplicate)
            {
                guestNames.Add(name);
                RebuildGuestPanel();
            }
            guestNameBox.Text = "";
            guestNameBox.Focus();
        }
        addGuestButton.Click += (_, __) => AddGuest();
        guestNameBox.KeyDown += (_, e) => { if (e.Key == Key.Enter) AddGuest(); };
        addGuestRow.Children.Add(guestNameBox);
        addGuestRow.Children.Add(addGuestButton);
        root.Children.Add(addGuestRow);

        // Action buttons
        var actionRow = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        var okButton = new Button
        {
            Content = "Set Roster",
            Width = 100,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            IsDefault = true,
            Background = Brushes.LightGreen
        };
        var clearButton = new Button
        {
            Content = "No Roster",
            Width = 90,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            ToolTip = "Clear the roster: back to unconstrained identification"
        };
        var cancelButton = new Button { Content = "Cancel", Width = 70, Height = 30, IsCancel = true };

        okButton.Click += (_, __) =>
        {
            var selectedKnown = knownChecks.Where(c => c.IsChecked == true)
                .Select(c => c.Content?.ToString() ?? "")
                .Where(n => n.Length > 0)
                .ToList();
            _sessionRoster = selectedKnown.Concat(guestNames).ToList();
            _sessionGuestNames = new HashSet<string>(guestNames, StringComparer.OrdinalIgnoreCase);
            RefreshAllSpeakerDropdowns();
            if (_statusText != null)
            {
                _statusText.Text = _sessionRoster.Count == 0
                    ? "Session roster cleared."
                    : "Session speakers: " + string.Join(", ",
                        _sessionRoster.Select(n => _sessionGuestNames.Contains(n) ? $"{n} (guest)" : n));
            }
            dialog.DialogResult = true;
        };
        clearButton.Click += (_, __) =>
        {
            _sessionRoster = new List<string>();
            _sessionGuestNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            RefreshAllSpeakerDropdowns();
            if (_statusText != null) _statusText.Text = "Session roster cleared.";
            dialog.DialogResult = true;
        };

        actionRow.Children.Add(okButton);
        actionRow.Children.Add(clearButton);
        actionRow.Children.Add(cancelButton);
        root.Children.Add(actionRow);

        dialog.Content = new ScrollViewer { Content = root, VerticalScrollBarVisibility = ScrollBarVisibility.Auto };
        dialog.ShowDialog();
    }

    private void UpdateMultiSelectButtons()
    {
        var count = _selectedSegments.Count;
        if (_bulkRenameButton != null)
        {
            _bulkRenameButton.IsEnabled = count > 0;
            _bulkRenameButton.Content = $"🏷 Rename Selected ({count})";
        }
        if (_deleteSelectedButton != null)
        {
            _deleteSelectedButton.IsEnabled = count > 0;
            _deleteSelectedButton.Content = $"🗑 Delete Selected ({count})";
        }
        if (_clearSelectionButton != null)
        {
            _clearSelectionButton.IsEnabled = count > 0;
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
        
        // Select All button (respects an active search filter: hidden cards stay unselected,
        // which turns Search + Select All into "select every match")
        _selectAllButton = new Button
        {
            Content = "☑ Select All",
            Width = 100,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            Background = Brushes.LightGray,
            ToolTip = "Select every visible segment (with a search active, that means every match)"
        };
        _selectAllButton.Click += SelectAll_Click;

        // Select-by-speaker menu button
        var selectSpeakerButton = new Button
        {
            Content = "👤 Select Speaker…",
            Width = 140,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            Background = Brushes.LightGray,
            ToolTip = "Select every segment from one speaker (Ctrl+choose adds to the current selection)"
        };
        selectSpeakerButton.Click += SelectBySpeaker_Click;

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

        // Bulk delete button
        _deleteSelectedButton = new Button
        {
            Content = "🗑 Delete Selected (0)",
            Width = 160,
            Height = 30,
            Margin = new Thickness(0, 0, 10, 0),
            Background = Brushes.MistyRose,
            IsEnabled = false,
            ToolTip = "Remove the selected segments from the transcript (speakers are not affected)"
        };
        _deleteSelectedButton.Click += DeleteSelected_Click;

        // Status text for selections
        var selectionStatus = new TextBlock
        {
            Text = "Click a card to select · Shift+click for a range · Ctrl+click to toggle",
            VerticalAlignment = VerticalAlignment.Center,
            FontStyle = FontStyles.Italic,
            Foreground = Brushes.DarkOrange,
            Margin = new Thickness(10, 0, 0, 0)
        };

        toolbarPanel.Children.Add(_selectAllButton);
        toolbarPanel.Children.Add(selectSpeakerButton);
        toolbarPanel.Children.Add(_clearSelectionButton);
        toolbarPanel.Children.Add(_bulkRenameButton);
        toolbarPanel.Children.Add(_deleteSelectedButton);
        toolbarPanel.Children.Add(selectionStatus);

        toolbar.Child = toolbarPanel;
        return toolbar;
    }

    private void SelectAll_Click(object sender, RoutedEventArgs e)
    {
        foreach (var segment in _transcriptionHistory)
        {
            if (segment.CardElement?.Visibility == Visibility.Collapsed) continue;
            SetSegmentSelected(segment, true);
        }
        UpdateMultiSelectButtons();
    }

    private void ClearSelection_Click(object sender, RoutedEventArgs e)
    {
        ClearAllSelections();
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
                // Add to available speakers if new. RefreshAllSpeakerDropdowns() must run so the
                // new name shows up in every OTHER segment's dropdown too, since (unlike a full
                // RefreshTranscriptionDisplay()) the targeted update below only touches the
                // dropdowns of the segments actually being renamed.
                if (!_availableSpeakers.Contains(newSpeaker))
                {
                    _availableSpeakers.Add(newSpeaker);
                    RefreshAllSpeakerDropdowns();
                    SaveAppSettings();
                }

                // Update all selected segments and patch just their cards in place - a bulk
                // rename doesn't add/remove/reorder segments, so it doesn't need a full
                // RefreshTranscriptionDisplay() rebuild.
                foreach (var segmentId in _selectedSegments)
                {
                    UpdateSegmentSpeaker(segmentId, newSpeaker);
                    var renamedSegment = _transcriptionHistory.FirstOrDefault(seg => seg.SegmentId == segmentId);
                    if (renamedSegment != null)
                    {
                        UpdateSegmentCardInPlace(renamedSegment);
                    }
                }

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
            SegmentId = _nextSegmentId++,
            AudioFilePath = originalSegment.AudioFilePath,
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
        
        SaveAppSettings();
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

            // Keyword-alert bell: hidden unless this segment matches a configured alert keyword.
            var alertBellIcon = new TextBlock
            {
                Text = "🔔",
                FontSize = 14,
                Margin = new Thickness(0, 0, 6, 0),
                VerticalAlignment = VerticalAlignment.Center,
                Visibility = Visibility.Collapsed,
                ToolTip = "Matches a keyword alert"
            };

            // Selection checkbox
            var selectionCheckBox = CreateSelectionCheckBox(segment.SegmentId);

            // Timestamp (recording-relative - see ProcessAudioChunkAsync's chunkStartSeconds)
            var timestampText = new TextBlock
            {
                Text = $"[{FormatClockTimestamp(segment.StartTime)}]",
                FontWeight = FontWeights.Bold,
                Foreground = Brushes.Gray,
                Width = 70,
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
                // Ignore selection changes we caused ourselves while patching the card's display.
                if (_suppressSpeakerSelectionChanged)
                    return;

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
                        SaveAppSettings();
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
                    // Mirrors AddTranscriptionSegment: roster order first, then legacy names.
                    var newSpeaker = speakerNum <= _sessionRoster.Count
                        ? _sessionRoster[speakerNum - 1]
                        : $"Speaker {speakerNum}";
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

            // Rename speaker button - mirrors AddTranscriptionSegment's card layout. Renames the
            // speaker everywhere they appear, as opposed to reassigning just this segment.
            var renameSpeakerButton = new Button
            {
                Content = "✏",
                Width = 25,
                Height = 25,
                Margin = new Thickness(2, 0, 2, 0),
                FontSize = 10,
                Background = Brushes.Lavender,
                ToolTip = "Rename this speaker everywhere they appear"
            };

            renameSpeakerButton.Click += (s, e) => ShowSpeakerRenameDialog(segment.Speaker);

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
            topPanel.Children.Add(alertBellIcon);
            topPanel.Children.Add(selectionCheckBox);
            topPanel.Children.Add(CreatePlaySegmentButton(segment));
            topPanel.Children.Add(timestampText);
            topPanel.Children.Add(speakerComboBox);
            topPanel.Children.Add(emotionIndicator);
            topPanel.Children.Add(sourceText);
            topPanel.Children.Add(quickSpeakerPanel);
            topPanel.Children.Add(newSpeakerButton);
            topPanel.Children.Add(showAutoSpeakersButton);
            topPanel.Children.Add(renameSpeakerButton);
            topPanel.Children.Add(deleteSpeakerButton);

            // Editable text content (in its own row for better readability)
            var textDisplay = CreateEditableTextDisplay(segment);

            // Add both rows to the segment panel
            segmentPanel.Children.Add(topPanel);
            segmentPanel.Children.Add(textDisplay);

            // Speaker stripe + click-to-select + context menu, then attach to the panel.
            AttachCardChrome(segment, segmentBorder, segmentPanel);
            _transcriptionPanel.Children.Add(segmentBorder);

            // Cache the card elements and re-apply the keyword-alert highlight, mirroring what
            // AddTranscriptionSegment does for a newly-arrived segment.
            segment.CardElement = segmentBorder;
            segment.SpeakerComboBoxElement = speakerComboBox;
            segment.TextDisplayElement = textDisplay;
            segment.AlertBellElement = alertBellIcon;
            segment.SelectionCheckBoxElement = selectionCheckBox;
            ApplyKeywordAlertHighlight(segment);
        }

        // Re-apply the active search filter (if any) to the freshly-built cards - a rebuild
        // replaces every card's Border instance, so the previous filter pass no longer applies.
        if (!string.IsNullOrEmpty(_currentSearchQuery))
        {
            ApplySearchFilter(_currentSearchQuery);
        }

        // Restore scroll position instead of always auto-scrolling
        RestoreScrollPosition();
    }

    /// <summary>
    /// Patches an already-built segment card to reflect the segment's current speaker/text/
    /// privacy state, without tearing down and rebuilding the card. Used for speaker renames and
    /// the privacy-mode toggle - changes that affect how a segment is *displayed* but don't add,
    /// remove, or reorder any segments (those still go through RefreshTranscriptionDisplay).
    /// </summary>
    private void UpdateSegmentCardInPlace(TranscriptionSegment segment)
    {
        var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);

        // Element references are copied to locals before use (rather than re-reading
        // segment.XxxElement each time) so the null-check below stays valid across the
        // subsequent GetFilteredSpeakersForDropdown()/GetSpeakerColorEnhanced() calls.
        var speakerComboBox = segment.SpeakerComboBoxElement;
        if (speakerComboBox != null)
        {
            // These are PROGRAMMATIC updates. Assigning ItemsSource resets SelectedItem, and
            // assigning SelectedItem raises SelectionChanged - the very handler attached in
            // AddTranscriptionSegment / RefreshTranscriptionDisplay, which treats any change as
            // a user speaker correction. Re-entering it from here rewrote segment.Speaker to the
            // display name, saved a spurious _speakerNames mapping, and fired a phantom
            // /speakers/name_mapping POST on every privacy toggle, rename and keyword-alert
            // refresh. Suppress the handler for the duration (saving/restoring the previous value
            // so a caller that is already suppressing stays suppressed afterwards).
            bool previousSuppress = _suppressSpeakerSelectionChanged;
            _suppressSpeakerSelectionChanged = true;
            try
            {
                speakerComboBox.ItemsSource = GetFilteredSpeakersForDropdown();
                speakerComboBox.SelectedItem = displaySpeaker;
                speakerComboBox.Background = GetSpeakerColorEnhanced(displaySpeaker);
            }
            finally
            {
                _suppressSpeakerSelectionChanged = previousSuppress;
            }
        }

        // The stripe tracks the speaker, so a reassignment recolors the card edge too.
        if (segment.SpeakerStripeElement != null)
        {
            segment.SpeakerStripeElement.Fill = GetSpeakerColorEnhanced(displaySpeaker);
        }

        // Don't clobber text the user is actively editing - EnableTextEditing sets IsReadOnly to
        // false while an edit is in progress, so leave that card's text box alone until it finishes.
        var textDisplay = segment.TextDisplayElement;
        if (textDisplay != null && textDisplay.IsReadOnly)
        {
            textDisplay.Text = GetDisplayText(segment.Text);
        }

        ApplyKeywordAlertHighlight(segment);

        var cardElement = segment.CardElement;
        if (cardElement != null && !string.IsNullOrEmpty(_currentSearchQuery))
        {
            cardElement.Visibility = SegmentMatchesSearch(segment, _currentSearchQuery)
                ? Visibility.Visible
                : Visibility.Collapsed;
        }
    }

    // --- Keyword alerts -------------------------------------------------------------------

    /// <summary>
    /// True if this segment's DISPLAYED speaker or text contains one of AppSettings.KeywordAlerts.
    /// Matching is case-insensitive and "whole-word-ish": a keyword must not be immediately
    /// preceded/followed by another letter or digit, so "cat" doesn't fire on "category" while
    /// still tolerating adjacent punctuation ("cat." or "(cat)" both match).
    ///
    /// The display forms are used (as SegmentMatchesSearch does), never the raw speaker/text.
    /// Matching the raw values would defeat Legal-Safe Mode: the card highlight and the visible
    /// 🔔 tell anyone looking at the screen that the redacted segment contained one specific
    /// configured keyword - and the keyword list itself is right there in the expander. Under
    /// Legal-Safe Mode the haystack is therefore the same anonymised name and derived analysis
    /// text the user can already see, so an alert can never reveal more than the card does.
    /// </summary>
    private bool SegmentMatchesKeywordAlert(TranscriptionSegment segment)
    {
        var keywords = _appSettings.KeywordAlerts;
        if (keywords == null || keywords.Count == 0)
            return false;

        var haystack = $"{GetDisplaySpeakerName(segment.Speaker)} {GetDisplayText(segment.Text)}";
        if (string.IsNullOrWhiteSpace(haystack))
            return false;

        foreach (var keyword in keywords)
        {
            var trimmedKeyword = keyword?.Trim();
            if (string.IsNullOrEmpty(trimmedKeyword))
                continue;

            var pattern = @"(?<![A-Za-z0-9])" + System.Text.RegularExpressions.Regex.Escape(trimmedKeyword) + @"(?![A-Za-z0-9])";
            if (System.Text.RegularExpressions.Regex.IsMatch(haystack, pattern, System.Text.RegularExpressions.RegexOptions.IgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Shows/hides a segment's bell prefix and card highlight based on SegmentMatchesKeywordAlert.
    /// Returns whether it matched, so callers (e.g. AddTranscriptionSegment for a live arrival)
    /// can decide whether to also flash the status text.
    /// </summary>
    private bool ApplyKeywordAlertHighlight(TranscriptionSegment segment)
    {
        bool isMatch = SegmentMatchesKeywordAlert(segment);

        var bellIcon = segment.AlertBellElement;
        if (bellIcon != null)
        {
            bellIcon.Visibility = isMatch ? Visibility.Visible : Visibility.Collapsed;
        }

        var cardElement = segment.CardElement;
        if (cardElement != null)
        {
            // Selection owns the background; the alert keeps its border + bell either way,
            // so a selected alert-matching card still reads as both.
            cardElement.Background = _selectedSegments.Contains(segment.SegmentId)
                ? _selectionCardBackground
                : (isMatch ? _keywordAlertCardBackground : Brushes.White);
            cardElement.BorderBrush = isMatch ? _keywordAlertCardBorder : Brushes.LightGray;
            cardElement.BorderThickness = isMatch ? new Thickness(2) : new Thickness(1);
        }

        return isMatch;
    }

    /// <summary>
    /// Briefly flashes the status text's background so a live keyword-alert match is noticeable
    /// even when the transcript itself isn't in view.
    ///
    /// One reusable timer, restarted on each alert: overlapping alerts extend the flash rather
    /// than the first one's tick cutting the rest short (which is what a fresh timer per call
    /// actually did, despite the comment that used to claim the opposite). The pre-flash brush is
    /// captured and restored instead of assuming Transparent - a TextBlock's default Background
    /// is null, and hardcoding Transparent would also stomp any background set elsewhere.
    /// </summary>
    private void FlashStatusTextForKeywordAlert()
    {
        if (_statusText == null) return;

        if (_keywordFlashTimer == null)
        {
            _keywordFlashTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(800) };
            _keywordFlashTimer.Tick += (s, e) =>
            {
                _keywordFlashTimer!.Stop();
                if (_statusText != null) _statusText.Background = _statusTextBackgroundBeforeFlash;
            };
        }

        // Only capture when no flash is already running - otherwise the second alert would
        // "restore" gold and the highlight would never clear.
        if (!_keywordFlashTimer.IsEnabled)
        {
            _statusTextBackgroundBeforeFlash = _statusText.Background;
        }

        _statusText.Background = Brushes.Gold;
        _keywordFlashTimer.Stop();
        _keywordFlashTimer.Start();
    }

    private void ApplyKeywordAlertsButton_Click(object sender, RoutedEventArgs e)
    {
        if (_keywordAlertsTextBox == null) return;

        var keywords = _keywordAlertsTextBox.Text
            .Split(',')
            .Select(k => k.Trim())
            .Where(k => !string.IsNullOrEmpty(k))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        _appSettings.KeywordAlerts = keywords;
        SaveAppSettings();

        // Re-scan every already-displayed segment so the new keyword list takes effect
        // immediately instead of only affecting segments that arrive after this point.
        foreach (var segment in _transcriptionHistory)
        {
            ApplyKeywordAlertHighlight(segment);
        }

        if (_statusText != null)
        {
            _statusText.Text = keywords.Count > 0
                ? $"Keyword alerts updated ({keywords.Count} keyword{(keywords.Count == 1 ? "" : "s")})."
                : "Keyword alerts cleared.";
        }
    }

    // --- Transcript search -----------------------------------------------------------------

    private bool SegmentMatchesSearch(TranscriptionSegment segment, string query)
    {
        var displayedText = GetDisplayText(segment.Text);
        var displayedSpeaker = GetDisplaySpeakerName(segment.Speaker);

        return displayedText.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0 ||
               displayedSpeaker.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0;
    }

    /// <summary>
    /// Shows/hides every already-built segment card based on whether it matches the query, and
    /// updates the match-count label. Only sets Visibility on existing elements - it never
    /// rebuilds a card - so it's cheap to call on every keystroke (post-debounce), on every new
    /// live segment while a filter is active, and after a full RefreshTranscriptionDisplay().
    /// </summary>
    private void ApplySearchFilter(string query)
    {
        _currentSearchQuery = query?.Trim() ?? "";
        bool hasQuery = !string.IsNullOrEmpty(_currentSearchQuery);
        int matchCount = 0;

        foreach (var segment in _transcriptionHistory)
        {
            bool isMatch = !hasQuery || SegmentMatchesSearch(segment, _currentSearchQuery);
            var cardElement = segment.CardElement;
            if (cardElement != null)
            {
                cardElement.Visibility = isMatch ? Visibility.Visible : Visibility.Collapsed;
            }
            if (isMatch) matchCount++;
        }

        if (_searchMatchCountText != null)
        {
            _searchMatchCountText.Text = hasQuery ? $"{matchCount} match{(matchCount == 1 ? "" : "es")}" : "";
        }
    }

    private void SearchTextBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        // Debounce: restart the timer on every keystroke so a fast typist only triggers one
        // filter pass ~300ms after they stop, not once per character.
        if (_searchDebounceTimer == null)
        {
            _searchDebounceTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(300) };
            _searchDebounceTimer.Tick += (s, args) =>
            {
                _searchDebounceTimer!.Stop();
                ApplySearchFilter(_searchTextBox?.Text ?? "");
            };
        }

        _searchDebounceTimer.Stop();
        _searchDebounceTimer.Start();
    }

    private void SearchTextBox_KeyDown(object sender, System.Windows.Input.KeyEventArgs e)
    {
        if (e.Key == System.Windows.Input.Key.Escape)
        {
            _searchDebounceTimer?.Stop();
            if (_searchTextBox != null) _searchTextBox.Text = "";
            ApplySearchFilter("");
            e.Handled = true;
        }
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
            Filter = "JSON Files (*.json)|*.json|Text Files (*.txt)|*.txt|SubRip Subtitles (*.srt)|*.srt|WebVTT (*.vtt)|*.vtt|Markdown (*.md)|*.md|All Files (*.*)|*.*",
            DefaultExt = "json",
            FileName = $"Oreja_Transcription_{DateTime.Now:yyyy-MM-dd_HH-mm-ss}.json"
        };

        if (saveFileDialog.ShowDialog() == true)
        {
            try
            {
                string content;
                if (saveFileDialog.FileName.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                {
                    content = GenerateJsonTranscription();
                }
                else if (saveFileDialog.FileName.EndsWith(".srt", StringComparison.OrdinalIgnoreCase))
                {
                    content = GenerateSrtTranscription();
                }
                else if (saveFileDialog.FileName.EndsWith(".vtt", StringComparison.OrdinalIgnoreCase))
                {
                    content = GenerateVttTranscription();
                }
                else if (saveFileDialog.FileName.EndsWith(".md", StringComparison.OrdinalIgnoreCase))
                {
                    content = GenerateMarkdownTranscription();
                }
                else
                {
                    content = GenerateTranscriptionReport();
                }

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

    private string GenerateJsonTranscription()
    {
        var transcriptionData = new
        {
            metadata = new
            {
                version = "1.0",
                created_at = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                total_segments = _transcriptionHistory.Count,
                privacy_mode = _privacyModeEnabled,
                source = "Oreja Live Transcription"
            },
            // Emitted in _transcriptionHistory (arrival) order, like the SRT/VTT/Markdown
            // exports. start/end are seconds from the beginning of the recording, not from the
            // beginning of the 5-second chunk they were transcribed in.
            segments = _transcriptionHistory.Select(segment => new
            {
                id = segment.SegmentId,
                speaker = GetDisplaySpeakerName(segment.Speaker),
                text = _privacyModeEnabled ? "[REDACTED]" : segment.Text,
                start = segment.StartTime,
                end = segment.EndTime,
                source = segment.Source,
                timestamp = segment.Timestamp.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                emotional_tone = segment.EmotionalTone,
                sentiment_confidence = segment.SentimentConfidence
            }).ToArray(),
            speakers = _availableSpeakers.Where(s => s != "Unknown").ToArray(),
            full_text = _privacyModeEnabled ? "[REDACTED FOR PRIVACY]" : 
                string.Join(" ", _transcriptionHistory.Select(s => $"{GetDisplaySpeakerName(s.Speaker)}: {s.Text}"))
        };

        return JsonSerializer.Serialize(transcriptionData, new JsonSerializerOptions 
        { 
            WriteIndented = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        });
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
                report.AppendLine($"[{FormatClockTimestamp(segment.StartTime)}] {displaySpeaker}: {segment.Text}");
            }
            report.AppendLine();
        }

        if (systemAudioSegments.Any())
        {
            report.AppendLine("=== SYSTEM AUDIO ===");
            foreach (var segment in systemAudioSegments)
            {
                var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);
                report.AppendLine($"[{FormatClockTimestamp(segment.StartTime)}] {displaySpeaker}: {segment.Text}");
            }
            report.AppendLine();
        }

        // Combined chronological view. Ordered by wall-clock arrival (Timestamp), which matches
        // the arrival order the SRT/VTT/Markdown exports emit and the order shown on screen.
        report.AppendLine("=== CHRONOLOGICAL TRANSCRIPT ===");
        var sortedSegments = _transcriptionHistory.OrderBy(s => s.Timestamp).ToList();
        foreach (var segment in sortedSegments)
        {
            var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);
            report.AppendLine($"[{FormatClockTimestamp(segment.StartTime)}] [{segment.Source}] {displaySpeaker}: {segment.Text}");
        }
    }

    /// <summary>
    /// SRT (SubRip) export: sequential 1-based indices, HH:MM:SS,mmm --> HH:MM:SS,mmm ranges from
    /// each segment's start/end, and a "SpeakerName: text" line - using the same mapped display
    /// name and privacy-mode redaction as the other export formats.
    ///
    /// Segments are emitted in _transcriptionHistory order, which is arrival order and therefore
    /// already chronological (SplitSegment inserts its second half directly after the first).
    /// They are deliberately NOT re-sorted: the two capture sources are transcribed independently
    /// and a sort would interleave them into an order the user never saw on screen. When both
    /// sources are active the interleaving that arrival order produces is the honest one.
    /// </summary>
    private string GenerateSrtTranscription()
    {
        var sb = new System.Text.StringBuilder();
        int index = 1;

        foreach (var segment in _transcriptionHistory)
        {
            var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);
            var text = _privacyModeEnabled ? "[REDACTED]" : (segment.Text ?? "");

            sb.AppendLine(index.ToString());
            sb.AppendLine($"{FormatSrtTimestamp(segment.StartTime)} --> {FormatSrtTimestamp(segment.EndTime)}");
            sb.AppendLine($"{displaySpeaker}: {text}");
            sb.AppendLine();
            index++;
        }

        return sb.ToString();
    }

    /// <summary>
    /// WebVTT export: the required "WEBVTT" header followed by cues using dot-millisecond
    /// timestamps (WebVTT's format, vs. SRT's comma-millisecond). Emitted in arrival order for
    /// the same reason as the SRT export above.
    /// </summary>
    private string GenerateVttTranscription()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("WEBVTT");
        sb.AppendLine();

        foreach (var segment in _transcriptionHistory)
        {
            var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);
            var text = _privacyModeEnabled ? "[REDACTED]" : (segment.Text ?? "");

            sb.AppendLine($"{FormatVttTimestamp(segment.StartTime)} --> {FormatVttTimestamp(segment.EndTime)}");
            sb.AppendLine($"{displaySpeaker}: {text}");
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>
    /// Markdown meeting-notes export: a title with today's date, then one bolded-speaker
    /// paragraph per run of consecutive same-speaker segments (so several short segments from the
    /// same person read as one paragraph instead of one bullet per 5-second chunk). Runs are
    /// detected over arrival order (see the SRT export above) - re-sorting would shuffle segments
    /// between speakers and break the grouping.
    /// </summary>
    private string GenerateMarkdownTranscription()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"# Oreja Meeting Notes — {DateTime.Now:yyyy-MM-dd HH:mm}");
        sb.AppendLine();

        if (_privacyModeEnabled)
        {
            sb.AppendLine("_Legal-Safe Mode: verbatim transcription redacted._");
            sb.AppendLine();
        }

        string? currentParagraphSpeaker = null;
        var paragraphText = new System.Text.StringBuilder();

        void FlushParagraph()
        {
            if (currentParagraphSpeaker != null && paragraphText.Length > 0)
            {
                sb.AppendLine($"**{currentParagraphSpeaker}:** {paragraphText.ToString().Trim()}");
                sb.AppendLine();
            }
            paragraphText.Clear();
        }

        foreach (var segment in _transcriptionHistory)
        {
            var displaySpeaker = GetDisplaySpeakerName(segment.Speaker);
            var text = _privacyModeEnabled ? "[REDACTED]" : (segment.Text ?? "");

            if (displaySpeaker != currentParagraphSpeaker)
            {
                FlushParagraph();
                currentParagraphSpeaker = displaySpeaker;
            }

            if (paragraphText.Length > 0) paragraphText.Append(' ');
            paragraphText.Append(text);
        }
        FlushParagraph();

        return sb.ToString();
    }

    /// <summary>
    /// Short human-readable position on the recording timeline, used for the on-screen segment
    /// timestamps and the TXT report. mm:ss while the recording is under an hour, h:mm:ss beyond
    /// it - a plain "mm\:ss" format string silently drops the hours component (a segment at
    /// 1:00:30 would render as "00:30"), which only became reachable once StartTime stopped being
    /// a per-chunk offset and started being recording-relative.
    /// </summary>
    private static string FormatClockTimestamp(double totalSeconds)
    {
        var clamped = TimeSpan.FromSeconds(Math.Max(0, totalSeconds));
        return clamped.TotalHours >= 1
            ? $"{(int)clamped.TotalHours}:{clamped.Minutes:D2}:{clamped.Seconds:D2}"
            : $"{clamped.Minutes:D2}:{clamped.Seconds:D2}";
    }

    /// <summary>SRT timestamp: HH:MM:SS,mmm (comma before milliseconds).</summary>
    private static string FormatSrtTimestamp(double totalSeconds)
    {
        var clamped = TimeSpan.FromSeconds(Math.Max(0, totalSeconds));
        return $"{(int)clamped.TotalHours:D2}:{clamped.Minutes:D2}:{clamped.Seconds:D2},{clamped.Milliseconds:D3}";
    }

    /// <summary>WebVTT timestamp: HH:MM:SS.mmm (dot before milliseconds).</summary>
    private static string FormatVttTimestamp(double totalSeconds)
    {
        var clamped = TimeSpan.FromSeconds(Math.Max(0, totalSeconds));
        return $"{(int)clamped.TotalHours:D2}:{clamped.Minutes:D2}:{clamped.Seconds:D2}.{clamped.Milliseconds:D3}";
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
        // Cancel any pending debounced save and write the final state synchronously - a crash
        // no longer loses settings, but a clean close shouldn't rely on the 1s debounce either.
        _settingsSaveTimer?.Stop();
        SaveAppSettings();
        Console.WriteLine("Saved settings on application exit");

        // Clean up resources
        _volumeTimer?.Stop();
        _transcriptionTimer?.Stop();
        _backendHealthTimer?.Stop();
        _waveIn?.StopRecording();
        _waveIn?.Dispose();
        _systemAudioCapture?.StopRecording();
        _systemAudioCapture?.Dispose();
        _httpClient?.Dispose();
        _deviceEnumerator?.Dispose();

        // Only stop a backend process WE spawned; one the user already had running is left alone.
        var backendProcess = _autoStartedBackendProcess;
        if (_weStartedBackend && backendProcess != null)
        {
            try
            {
                if (!backendProcess.HasExited)
                {
                    backendProcess.Kill(entireProcessTree: true);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error stopping auto-started backend process: {ex.Message}");
            }
            finally
            {
                backendProcess.Dispose();
            }
        }

        // After the kill: the output streams may or may not have EOF'd by now, so close the
        // backend log deterministically here (idempotent; late DataReceived events are dropped).
        CloseBackendLog();
    }

    private void App_DispatcherUnhandledException(object sender, System.Windows.Threading.DispatcherUnhandledExceptionEventArgs e)
    {
        MessageBox.Show($"An unhandled exception occurred: {e.Exception.Message}", "Oreja Error", MessageBoxButton.OK, MessageBoxImage.Error);

        // Only swallow non-fatal exceptions and keep the app running. OutOfMemoryException means
        // the process is already in a bad spot - let it crash honestly instead of limping on.
        // (StackOverflowException can't reach a handler like this one at all, per the CLR.)
        e.Handled = e.Exception is not OutOfMemoryException;
    }

    // Shared serializer options for settings.json.
    //
    // System.Text.Json's default JsonNumberHandling.Strict THROWS on double.NaN
    // ("ArgumentException: .NET number values such as positive and negative infinity cannot be
    // written as valid JSON"). AppSettings.WindowLeft/WindowTop default to NaN as the "window
    // has never been positioned" sentinel, so every save taken while the geometry was still NaN
    // threw - and SaveAppSettings swallows the exception, so the file was silently never
    // written. That happened on the very first run (LoadAppSettings runs before MainWindow is
    // assigned, so the "persist the defaults for a brand-new install" save saw MainWindow ==
    // null and left the geometry at NaN), and again on every save taken while the window was
    // maximized/minimized. Result: no persistence of speakers, keyword alerts or device
    // selection at all for that session.
    //
    // AllowNamedFloatingPointLiterals writes NaN as the JSON string "NaN"; Deserialize needs
    // the same setting for the sentinel to round-trip on read, so both use this instance.
    private static readonly JsonSerializerOptions SettingsJsonOptions = new JsonSerializerOptions
    {
        WriteIndented = true, // Make JSON readable
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    private void LoadAppSettings()
    {
        try
        {
            if (!string.IsNullOrEmpty(_settingsFilePath) && File.Exists(_settingsFilePath))
            {
                Console.WriteLine($"Loading settings from: {_settingsFilePath}");
                var json = File.ReadAllText(_settingsFilePath);
                // Old settings files only ever contained AvailableSpeakers/SpeakerNameMappings/
                // NextSpeakerNumber; System.Text.Json defaults every property this class has
                // added since, so an old file loads cleanly with no migration step.
                _appSettings = JsonSerializer.Deserialize<AppSettings>(json, SettingsJsonOptions) ?? new AppSettings();

                // Update current state from loaded settings
                if (_appSettings.AvailableSpeakers.Count > 0)
                {
                    _availableSpeakers = _appSettings.AvailableSpeakers.ToList();
                }
                else
                {
                    // If no speakers saved, use new defaults (no preset speakers)
                    _availableSpeakers = new List<string> { "Unknown" };
                }

                _nextSpeakerNumber = _appSettings.NextSpeakerNumber;
                _speakerNames = new Dictionary<string, string>(_appSettings.SpeakerNameMappings);

                Console.WriteLine($"Loaded settings: {_availableSpeakers.Count} speakers, next number: {_nextSpeakerNumber}");
            }
            else
            {
                Console.WriteLine("No existing settings file found, using defaults");
                // Initialize with new defaults (no preset speakers)
                _availableSpeakers = new List<string> { "Unknown" };
                _nextSpeakerNumber = 1;
                _speakerNames = new Dictionary<string, string>();
                _appSettings = new AppSettings();
            }

            // Apply the non-speaker settings that don't depend on UI elements existing yet.
            // (Device IDs are applied later by RestoreDeviceSelection, once the device combo
            // boxes have been populated; privacy mode and window geometry are applied while the
            // window is being built, in OnStartup.)
            _backendUrl = string.IsNullOrWhiteSpace(_appSettings.BackendUrl) ? DEFAULT_BACKEND_URL : _appSettings.BackendUrl;
            _microphoneGain = _appSettings.MicGain > 0f ? _appSettings.MicGain : 1.0f;

            if (string.IsNullOrEmpty(_settingsFilePath) == false && !File.Exists(_settingsFilePath))
            {
                SaveAppSettings(); // Persist the defaults for a brand-new install.
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading settings: {ex.Message}");
            // Fallback to new defaults
            _availableSpeakers = new List<string> { "Unknown" };
            _nextSpeakerNumber = 1;
            _speakerNames = new Dictionary<string, string>();
            _appSettings = new AppSettings();
            _backendUrl = DEFAULT_BACKEND_URL;
        }
        finally
        {
            _settingsLoaded = true;
        }
    }

    /// <summary>
    /// Persists the full settings object: speaker data plus device selection, privacy mode,
    /// window geometry, backend URL/auto-start, and mic gain. Called immediately after
    /// speaker-related changes (existing call sites) and via a ~1s debounce from
    /// RequestSettingsSave for high-frequency changes (window drag/resize), plus once more on
    /// close so nothing from the final second is lost.
    /// </summary>
    private void SaveAppSettings()
    {
        try
        {
            if (string.IsNullOrEmpty(_settingsFilePath))
            {
                Console.WriteLine("Settings file path not initialized, skipping save");
                return;
            }

            // Update settings object with current state
            _appSettings.AvailableSpeakers = _availableSpeakers.ToList();
            _appSettings.NextSpeakerNumber = _nextSpeakerNumber;
            _appSettings.SpeakerNameMappings = new Dictionary<string, string>(_speakerNames);

            _appSettings.MicDeviceId = _selectedMicrophone?.ID;
            _appSettings.SystemDeviceId = _defaultSystemAudio?.ID;
            _appSettings.PrivacyMode = _privacyModeEnabled;
            _appSettings.BackendUrl = _backendUrl;
            _appSettings.MicGain = _microphoneGain;
            // AutoStartBackend is not mutated by any UI in this build, so whatever was loaded (or
            // the type default) is simply written back unchanged. KeywordAlerts IS mutated by the
            // 🔔 Keyword Alerts expander (ApplyKeywordAlertsButton_Click sets it directly on
            // _appSettings), so there's nothing extra to sync here.

            if (MainWindow != null)
            {
                // Only persist a real, on-screen size/position - a minimized window reports a
                // near-zero restore rectangle that would be useless to restore next launch.
                if (MainWindow.WindowState == WindowState.Normal)
                {
                    _appSettings.WindowWidth = MainWindow.Width;
                    _appSettings.WindowHeight = MainWindow.Height;
                    _appSettings.WindowLeft = MainWindow.Left;
                    _appSettings.WindowTop = MainWindow.Top;
                }
            }

            // SettingsJsonOptions (not default options): WindowLeft/WindowTop can legitimately
            // be double.NaN, which the default Strict number handling refuses to write.
            var json = JsonSerializer.Serialize(_appSettings, SettingsJsonOptions);
            File.WriteAllText(_settingsFilePath, json);

            Console.WriteLine($"Saved settings to: {_settingsFilePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error saving settings: {ex.Message}");
        }
    }

    /// <summary>
    /// Schedules a settings save ~1s from now, restarting the delay on every call so a burst of
    /// changes (dragging/resizing the window, toggling a checkbox) results in one write instead
    /// of many. Must be called on the UI thread (DispatcherTimer).
    /// </summary>
    private void RequestSettingsSave()
    {
        if (!_settingsLoaded)
        {
            return; // Don't stomp on-disk settings while the initial load is still in progress.
        }

        if (_settingsSaveTimer == null)
        {
            _settingsSaveTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
            _settingsSaveTimer.Tick += (s, e) =>
            {
                _settingsSaveTimer!.Stop();
                SaveAppSettings();
            };
        }

        _settingsSaveTimer.Stop();
        _settingsSaveTimer.Start();
    }

    /// <summary>
    /// Applies saved width/height/left/top to a not-yet-shown window, clamping the position so a
    /// geometry saved while a second monitor was connected can't leave the window unreachable.
    /// Left/Top are left as NaN (and WindowStartupLocation.CenterScreen, already set by the
    /// caller) when no position has ever been saved.
    /// </summary>
    private static void ApplyWindowGeometry(Window window, AppSettings settings)
    {
        try
        {
            double screenWidth = SystemParameters.VirtualScreenWidth;
            double screenHeight = SystemParameters.VirtualScreenHeight;
            double screenLeft = SystemParameters.VirtualScreenLeft;
            double screenTop = SystemParameters.VirtualScreenTop;

            double width = settings.WindowWidth;
            double height = settings.WindowHeight;
            if (double.IsNaN(width) || width < 300) width = 800;
            if (double.IsNaN(height) || height < 200) height = 800;
            if (screenWidth > 0 && width > screenWidth) width = screenWidth;
            if (screenHeight > 0 && height > screenHeight) height = screenHeight;

            window.Width = width;
            window.Height = height;

            if (!double.IsNaN(settings.WindowLeft) && !double.IsNaN(settings.WindowTop) && screenWidth > 0 && screenHeight > 0)
            {
                double left = settings.WindowLeft;
                double top = settings.WindowTop;
                const double minVisible = 80; // Keep at least this many px reachable on-screen.

                if (left + width < screenLeft + minVisible) left = screenLeft;
                if (left > screenLeft + screenWidth - minVisible) left = screenLeft + screenWidth - minVisible;
                if (top < screenTop) top = screenTop;
                if (top > screenTop + screenHeight - minVisible) top = screenTop + screenHeight - minVisible;

                window.WindowStartupLocation = WindowStartupLocation.Manual;
                window.Left = left;
                window.Top = top;
            }
            // else: leave WindowStartupLocation as CenterScreen (set by the caller) - first run.
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error applying window geometry: {ex.Message}");
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

            // Update all segments that used this speaker to "Unknown" and patch their cards in
            // place (previously these cards kept showing the deleted speaker until some other
            // full-rebuild event happened, since this method never triggered one itself).
            foreach (var segment in _transcriptionHistory.Where(s => s.Speaker == selectedSpeaker))
            {
                segment.Speaker = "Unknown";
                UpdateSegmentCardInPlace(segment);
            }

            // Update current ComboBox to "Unknown"
            UpdateSegmentSpeaker(segmentId, "Unknown");

            // Refresh all dropdowns (the deleted speaker must disappear from every ItemsSource)
            // and save settings.
            RefreshAllSpeakerDropdowns();
            SaveAppSettings();

            Console.WriteLine($"Deleted speaker: {selectedSpeaker}");
        }
    }

    // Add after the existing LoadAppSettings method
    private async Task LoadSpeakersFromBackend()
    {
        try
        {
            Console.WriteLine("🔄 Loading speakers from enhanced backend...");
            
            if (_httpClient == null)
            {
                Console.WriteLine("❌ HTTP client not initialized, falling back to local settings");
                LoadAppSettings();
                return;
            }
            
            var response = await _httpClient.GetAsync($"{_backendUrl}/speakers");
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var speakerData = JsonSerializer.Deserialize<JsonElement>(content);
                
                var speakers = speakerData.GetProperty("speakers").EnumerateArray();
                _availableSpeakers.Clear();
                _availableSpeakers.Add("Unknown"); // Always keep Unknown
                
                Console.WriteLine("✅ Loading speakers from enhanced backend:");
                foreach (var speaker in speakers)
                {
                    var name = speaker.GetProperty("name").GetString();
                    var speakerId = speaker.GetProperty("id").GetString();
                    var embeddingCount = speaker.GetProperty("embedding_count").GetInt32();
                    
                    if (!string.IsNullOrEmpty(name) && !_availableSpeakers.Contains(name))
                    {
                        _availableSpeakers.Add(name);
                        Console.WriteLine($"   📢 {name} ({speakerId}, {embeddingCount} embeddings)");
                    }
                }
                
                Console.WriteLine($"✅ Loaded {_availableSpeakers.Count - 1} speakers from enhanced backend");
                RefreshAllSpeakerDropdowns();
                
                // Save the loaded speakers to local settings as backup
                SaveAppSettings();
            }
            else
            {
                Console.WriteLine($"❌ Backend responded with {response.StatusCode}, falling back to local settings");
                LoadAppSettings();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error loading speakers from backend: {ex.Message}");
            Console.WriteLine("📁 Falling back to local speaker settings");
            LoadAppSettings();
        }
    }

    // ---------------------------------------------------------------------------------
    // Audio capture plumbing
    //
    // Concurrency model: every capture source owns an AudioSourceState. The NAudio capture
    // callback thread appends already-normalised bytes under state.Sync; the UI timer thread
    // drains the buffer and claims the source under the same lock. A source that is mid-flight
    // keeps its audio buffered instead of having it cleared and then dropped, and one slow
    // source can no longer starve the other. No await ever happens while a lock is held.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Capture/dispatch state for a single audio source (microphone or system loopback).
    /// </summary>
    private sealed class AudioSourceState
    {
        public AudioSourceState(string displayName, string sourceTag)
        {
            DisplayName = displayName;
            SourceTag = sourceTag;
        }

        /// <summary>Human readable name used in the UI and in log lines.</summary>
        public string DisplayName { get; }

        /// <summary>Short machine tag sent to the backend as ?source=... ("mic" / "system").</summary>
        public string SourceTag { get; }

        /// <summary>Guards the Buffer and IsProcessing members below, together.</summary>
        public object Sync { get; } = new object();

        /// <summary>Pending 16 kHz mono 16-bit little-endian PCM bytes awaiting transcription.</summary>
        public List<byte> Buffer { get; } = new List<byte>();

        /// <summary>True while a /transcribe request for THIS source is outstanding.</summary>
        public bool IsProcessing { get; set; }

        /// <summary>
        /// True after DispatchPendingAudio has logged that this source's audio is being held
        /// because the backend is unreachable; cleared when dispatch resumes. Purely to keep the
        /// console at one line per hold episode instead of one per timer tick. Guarded by Sync.
        /// </summary>
        public bool HoldLogged { get; set; }

        /// <summary>
        /// Total bytes that have left this source's Buffer since the recording started - either
        /// dispatched to the backend or dropped as overflow. Divided by
        /// TRANSCRIPTION_BYTES_PER_SECOND this is the recording-relative start time of the NEXT
        /// chunk to be dispatched, which is what turns the backend's per-chunk (near-zero)
        /// start/end offsets into real timeline positions. Reset by ResetAudioSource when a new
        /// recording starts. Guarded by Sync like the members above.
        /// </summary>
        public long ConsumedBytes { get; set; }
    }

    /// <summary>
    /// RMS of 16-bit little-endian PCM over [start, end) byte offsets of a buffer.
    /// Callers must hold the owning source's Sync lock.
    /// </summary>
    private static double PcmRms(List<byte> buffer, int start, int end)
    {
        if (start < 0) start = 0;
        if (start % 2 != 0) start++; // keep sample alignment
        if (end > buffer.Count) end = buffer.Count;

        double sumSquares = 0;
        int sampleCount = 0;
        for (int i = start; i + 1 < end; i += 2)
        {
            short sample = (short)(buffer[i] | (buffer[i + 1] << 8));
            sumSquares += (double)sample * sample;
            sampleCount++;
        }

        return sampleCount == 0 ? 0.0 : Math.Sqrt(sumSquares / sampleCount);
    }

    /// <summary>True when the trailing SILENCE_WINDOW_MS of the buffer is below the silence floor.</summary>
    private static bool PcmTailIsSilent(List<byte> buffer)
    {
        int windowBytes = TRANSCRIPTION_BYTES_PER_SECOND * SILENCE_WINDOW_MS / 1000;
        return PcmRms(buffer, buffer.Count - windowBytes, buffer.Count) < SILENCE_RMS_THRESHOLD;
    }

    /// <summary>True when the entire buffer is below the silence floor (idle source).</summary>
    private static bool PcmIsAllSilent(List<byte> buffer)
    {
        return PcmRms(buffer, 0, buffer.Count) < SILENCE_RMS_THRESHOLD;
    }

    /// <summary>
    /// Byte offset to cut a cap-flushed chunk at: the end of the most recent
    /// SILENCE_WINDOW_MS-long quiet stretch, so the cut lands in a pause instead of
    /// mid-word. Scans backwards at 100 ms strides (a real inter-sentence pause is
    /// several hundred ms, so the stride cannot step over one). Never cuts closer to
    /// the buffer start than MIN_CHUNK_SECONDS - the outgoing chunk stays worth a
    /// request - and returns buffer.Count (send everything, the pre-existing
    /// behavior) when the range holds no pause at all.
    /// Callers must hold the owning source's Sync lock.
    /// </summary>
    private static int FindPauseCutOffset(List<byte> buffer)
    {
        int windowBytes = TRANSCRIPTION_BYTES_PER_SECOND * SILENCE_WINDOW_MS / 1000;
        int strideBytes = TRANSCRIPTION_BYTES_PER_SECOND / 10;
        int minChunkBytes = (int)(TRANSCRIPTION_BYTES_PER_SECOND * MIN_CHUNK_SECONDS);

        // buffer.Count and both constants are whole-sample (even) sizes, so every
        // candidate cut below stays sample-aligned without explicit rounding.
        for (int cut = buffer.Count; cut - windowBytes >= minChunkBytes; cut -= strideBytes)
        {
            if (PcmRms(buffer, cut - windowBytes, cut) < SILENCE_RMS_THRESHOLD)
            {
                return cut;
            }
        }
        return buffer.Count;
    }

    /// <summary>
    /// Appends already-converted 16 kHz mono 16-bit PCM to a source's buffer, dropping the
    /// oldest audio once the buffer exceeds MAX_BUFFERED_AUDIO_SECONDS seconds of audio.
    /// </summary>
    private static void AppendAudioToSource(AudioSourceState state, byte[] pcmBytes, int count)
    {
        if (count <= 0 || pcmBytes.Length == 0)
        {
            return;
        }

        if (count > pcmBytes.Length)
        {
            count = pcmBytes.Length;
        }

        lock (state.Sync)
        {
            if (count == pcmBytes.Length)
            {
                state.Buffer.AddRange(pcmBytes);
            }
            else
            {
                state.Buffer.AddRange(pcmBytes.Take(count));
            }

            int overflow = state.Buffer.Count - MAX_BUFFERED_AUDIO_BYTES;
            if (overflow > 0)
            {
                // Keep the trim on a 16-bit sample boundary so we never split a sample.
                if ((overflow & 1) != 0)
                {
                    overflow++;
                }

                if (overflow > state.Buffer.Count)
                {
                    overflow = state.Buffer.Count;
                }

                state.Buffer.RemoveRange(0, overflow);

                // Dropped audio still advanced the recording's timeline, so it counts as
                // consumed - otherwise every segment after an overflow would be timestamped
                // earlier than it actually occurred.
                state.ConsumedBytes += overflow;
            }
        }
    }

    /// <summary>Drops any audio buffered for a source (used when a new recording starts).</summary>
    private static void ResetAudioSource(AudioSourceState state)
    {
        lock (state.Sync)
        {
            state.Buffer.Clear();
            state.ConsumedBytes = 0; // New recording => timeline restarts at 0.
        }
    }

    /// <summary>
    /// Scales a float sample (nominally in [-1, 1], or an already-scaled Int16 value) into the
    /// Int16 range with saturation instead of wraparound.
    /// </summary>
    private static short ClampToInt16(float value)
    {
        if (float.IsNaN(value))
        {
            return 0;
        }

        if (value >= short.MaxValue)
        {
            return short.MaxValue;
        }

        if (value <= short.MinValue)
        {
            return short.MinValue;
        }

        return (short)value;
    }

    /// <summary>
    /// Converts WASAPI loopback audio into the 16 kHz mono 16-bit little-endian PCM the
    /// backend expects.
    ///
    /// Three steps, in order:
    ///   1. decode the interleaved source samples (32-bit IEEE float or 16-bit PCM),
    ///   2. downmix to mono by averaging all channels,
    ///   3. resample to 16 kHz by linear interpolation.
    ///
    /// The resampler keeps the previous callback's final sample and its fractional read
    /// position in fields, so successive capture buffers are joined seamlessly - without the
    /// carry, every callback boundary would introduce a phase jump and an audible click that
    /// degrades ASR accuracy. Instances are used from the single NAudio capture thread.
    /// </summary>
    private sealed class SystemAudioFormatConverter
    {
        // KSDATAFORMAT_SUBTYPE_* GUIDs from ksmedia.h. For WAVE_FORMAT_EXTENSIBLE these - not
        // the encoding tag or the bit depth - are what identify the sample layout.
        private static readonly Guid SubTypePcm = new Guid("00000001-0000-0010-8000-00aa00389b71");
        private static readonly Guid SubTypeIeeeFloat = new Guid("00000003-0000-0010-8000-00aa00389b71");

        private float _previousSample;
        private double _sourcePosition = 1.0;
        private int _configuredSampleRate = -1;
        private int _configuredChannels = -1;
        private float[] _monoScratch = Array.Empty<float>();

        // Anti-aliasing low-pass applied BEFORE the 16 kHz downsample. Linear
        // interpolation alone does not band-limit: at a 44.1/48 kHz source rate,
        // everything above 8 kHz (music, hiss, notification chimes in system audio)
        // folds back into the 0-8 kHz speech band as non-harmonic noise the ASR
        // model then has to fight. A symmetric windowed-sinc FIR with cutoff below
        // the 16 kHz Nyquist removes that energy first. Null when the source rate
        // needs no filtering (already at/below 16 kHz).
        private const int FIR_TAPS = 95;               // odd => symmetric, linear phase
        private const double FIR_CUTOFF_HZ = 7000.0;   // below 8 kHz Nyquist; speech content above this is negligible
        private float[]? _firCoefficients;
        private readonly float[] _firHistory = new float[FIR_TAPS - 1]; // last inputs of the previous callback
        private float[] _firExtScratch = Array.Empty<float>();   // history + current buffer
        private float[] _firOutScratch = Array.Empty<float>();   // filtered output

        /// <summary>Clears the interpolation carry; call when (re)starting a capture.</summary>
        public void Reset()
        {
            _previousSample = 0f;
            _sourcePosition = 1.0;
            _configuredSampleRate = -1;
            _configuredChannels = -1;
            Array.Clear(_firHistory, 0, _firHistory.Length);
        }

        /// <summary>
        /// Windowed-sinc (Hamming) low-pass coefficients for the given source rate,
        /// normalized to unity DC gain. Symmetric, so convolution can read taps in
        /// either direction.
        /// </summary>
        private static float[] BuildLowPassCoefficients(int sampleRate)
        {
            var coefficients = new float[FIR_TAPS];
            int m = FIR_TAPS - 1;
            double normalizedCutoff = FIR_CUTOFF_HZ / sampleRate; // cycles per sample
            double sum = 0.0;
            for (int n = 0; n <= m; n++)
            {
                double x = n - (m / 2.0);
                double sinc = x == 0.0
                    ? 2.0 * Math.PI * normalizedCutoff
                    : Math.Sin(2.0 * Math.PI * normalizedCutoff * x) / x;
                double window = 0.54 - (0.46 * Math.Cos(2.0 * Math.PI * n / m));
                coefficients[n] = (float)(sinc * window);
                sum += coefficients[n];
            }
            for (int n = 0; n <= m; n++)
            {
                coefficients[n] = (float)(coefficients[n] / sum);
            }
            return coefficients;
        }

        /// <summary>
        /// Converts the first bytesRecorded bytes of buffer, laid out according to sourceFormat,
        /// into 16 kHz mono 16-bit little-endian PCM.
        /// Throws NotSupportedException when the source format cannot be decoded.
        /// </summary>
        public byte[] ConvertToPcm16Mono16k(WaveFormat? sourceFormat, byte[] buffer, int bytesRecorded)
        {
            if (sourceFormat == null)
            {
                throw new NotSupportedException("the capture reported no wave format");
            }

            if (bytesRecorded <= 0)
            {
                return Array.Empty<byte>();
            }

            int channels = sourceFormat.Channels;
            int sampleRate = sourceFormat.SampleRate;
            if (channels <= 0 || sampleRate <= 0)
            {
                throw new NotSupportedException($"implausible format: {channels} channel(s) @ {sampleRate} Hz");
            }

            // NAudio reports the endpoint's real mix format. WASAPI shared mode is normally
            // 32-bit IEEE float; some drivers describe the same layout as WAVE_FORMAT_EXTENSIBLE.
            bool isFloat32;
            if (sourceFormat is WaveFormatExtensible extensible)
            {
                // Extensible says nothing about the sample layout - only the SubFormat GUID does.
                // Treating "32-bit extensible" as float unconditionally would decode a 32-bit
                // INTEGER PCM stream (sample values around 1e9) as float, saturating every sample
                // to +/-32767: the backend would receive full-scale noise instead of this method
                // throwing NotSupportedException and system audio being cleanly disabled.
                if (extensible.SubFormat == SubTypeIeeeFloat && sourceFormat.BitsPerSample == 32)
                {
                    isFloat32 = true;
                }
                else if (extensible.SubFormat == SubTypePcm && sourceFormat.BitsPerSample == 16)
                {
                    isFloat32 = false;
                }
                else
                {
                    throw new NotSupportedException(
                        $"extensible subformat {extensible.SubFormat} at {sourceFormat.BitsPerSample} bit(s) per sample");
                }
            }
            else if (sourceFormat.BitsPerSample == 32 && sourceFormat.Encoding == WaveFormatEncoding.IeeeFloat)
            {
                isFloat32 = true;
            }
            else if (sourceFormat.BitsPerSample == 16 && sourceFormat.Encoding == WaveFormatEncoding.Pcm)
            {
                isFloat32 = false;
            }
            else if (sourceFormat.Encoding == WaveFormatEncoding.Extensible &&
                     (sourceFormat.BitsPerSample == 32 || sourceFormat.BitsPerSample == 16))
            {
                // Tagged extensible but not surfaced as a WaveFormatExtensible instance, so the
                // SubFormat GUID isn't reachable. Fall back to the old bit-depth heuristic rather
                // than refusing a format that used to work.
                isFloat32 = sourceFormat.BitsPerSample == 32;
            }
            else
            {
                throw new NotSupportedException(
                    $"encoding {sourceFormat.Encoding} at {sourceFormat.BitsPerSample} bit(s) per sample");
            }

            int bytesPerSample = isFloat32 ? 4 : 2;
            int bytesPerFrame = channels * bytesPerSample;
            int frameCount = bytesRecorded / bytesPerFrame;
            if (frameCount <= 0)
            {
                return Array.Empty<byte>();
            }

            // A mid-stream format change would invalidate the carried interpolation state
            // and the FIR history, and the FIR coefficients are a function of the rate.
            if (sampleRate != _configuredSampleRate || channels != _configuredChannels)
            {
                _configuredSampleRate = sampleRate;
                _configuredChannels = channels;
                _previousSample = 0f;
                _sourcePosition = 1.0;
                _firCoefficients = sampleRate > App.TRANSCRIPTION_SAMPLE_RATE
                    ? BuildLowPassCoefficients(sampleRate)
                    : null; // at/below 16 kHz nothing can alias into the target band
                Array.Clear(_firHistory, 0, _firHistory.Length);
            }

            if (_monoScratch.Length < frameCount)
            {
                _monoScratch = new float[frameCount];
            }

            float[] mono = _monoScratch;

            // Step 1 + 2: decode and downmix to mono by averaging the channels.
            if (isFloat32)
            {
                for (int frame = 0; frame < frameCount; frame++)
                {
                    int offset = frame * bytesPerFrame;
                    float sum = 0f;
                    for (int channel = 0; channel < channels; channel++)
                    {
                        sum += BitConverter.ToSingle(buffer, offset + (channel * 4));
                    }
                    mono[frame] = sum / channels;
                }
            }
            else
            {
                for (int frame = 0; frame < frameCount; frame++)
                {
                    int offset = frame * bytesPerFrame;
                    float sum = 0f;
                    for (int channel = 0; channel < channels; channel++)
                    {
                        sum += BitConverter.ToInt16(buffer, offset + (channel * 2)) / 32768f;
                    }
                    mono[frame] = sum / channels;
                }
            }

            // Step 3: anti-aliasing low-pass (see the FIR field comments). Runs at the
            // source rate, on the mono signal, BEFORE the rate change - filtering after
            // decimation would be too late, the folding has already happened. The history
            // buffer supplies the FIR's look-back across callback boundaries, so the
            // filtered stream is seamless; the interpolation state below then sees one
            // continuous band-limited signal.
            if (_firCoefficients != null)
            {
                float[] coefficients = _firCoefficients;
                int historyLength = FIR_TAPS - 1;
                int extendedLength = historyLength + frameCount;
                if (_firExtScratch.Length < extendedLength)
                {
                    _firExtScratch = new float[extendedLength];
                }
                if (_firOutScratch.Length < frameCount)
                {
                    _firOutScratch = new float[frameCount];
                }

                Array.Copy(_firHistory, 0, _firExtScratch, 0, historyLength);
                Array.Copy(mono, 0, _firExtScratch, historyLength, frameCount);

                for (int i = 0; i < frameCount; i++)
                {
                    float acc = 0f;
                    // Symmetric taps, so no reversal needed: this is y[i] = sum h[k]*x[i-k].
                    for (int k = 0; k < FIR_TAPS; k++)
                    {
                        acc += coefficients[k] * _firExtScratch[i + k];
                    }
                    _firOutScratch[i] = acc;
                }

                // Last (FIR_TAPS - 1) input samples become the next callback's look-back.
                Array.Copy(_firExtScratch, frameCount, _firHistory, 0, historyLength);

                mono = _firOutScratch;
            }

            // Step 4: linear-interpolation resample to 16 kHz.
            //
            // The virtual input stream for this callback is
            //     c[0]    = last mono sample of the PREVIOUS callback (_previousSample)
            //     c[k]    = mono[k - 1]   for k in 1..frameCount
            // and _sourcePosition is a fractional index into c carried across callbacks.
            // After the loop it is rebased by frameCount so it stays valid for the next
            // buffer, whose c[0] is this buffer's last sample.
            double step = (double)sampleRate / App.TRANSCRIPTION_SAMPLE_RATE;
            double position = _sourcePosition;

            int estimatedSamples = (int)((frameCount - position) / step) + 2;
            if (estimatedSamples < 0)
            {
                estimatedSamples = 0;
            }

            var output = new List<byte>(estimatedSamples * 2);

            while (position < frameCount)
            {
                int index = (int)position;              // position >= 0, so this is floor()
                double fraction = position - index;
                float a = index == 0 ? _previousSample : mono[index - 1];
                float b = mono[index];                  // c[index + 1]
                float value = (float)(a + ((b - a) * fraction));

                short pcm = App.ClampToInt16(value * 32767f);
                output.Add((byte)(pcm & 0xFF));
                output.Add((byte)((pcm >> 8) & 0xFF));

                position += step;
            }

            _previousSample = mono[frameCount - 1];
            _sourcePosition = position - frameCount;

            // Defensive only, and unreachable by construction: the loop above either never runs
            // (because position was already >= frameCount) or exits the moment position reaches
            // frameCount, so position >= frameCount here in both cases - including the
            // frameCount < step case, where a single iteration still carries position past
            // frameCount. The carry is therefore never clamped away and no phase discontinuity
            // is introduced. Kept as a guard against a future change to the loop condition.
            if (_sourcePosition < 0)
            {
                _sourcePosition = 0;
            }

            return output.ToArray();
        }
    }
} 