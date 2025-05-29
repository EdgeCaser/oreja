using System;
using System.Windows;
using System.Windows.Media;
using SkiaSharp;
using SkiaSharp.Views.Desktop;
using SkiaSharp.Views.WPF;

namespace Oreja.Controls;

/// <summary>
/// Custom volume meter control using SkiaSharp for smooth audio level visualization.
/// </summary>
public class VolumeMeter : SKElement
{
    /// <summary>
    /// Audio level dependency property (0.0 to 1.0).
    /// </summary>
    public static readonly DependencyProperty LevelProperty =
        DependencyProperty.Register(
            nameof(Level),
            typeof(float),
            typeof(VolumeMeter),
            new PropertyMetadata(0.0f, OnLevelChanged));

    /// <summary>
    /// Color for the active (filled) portion of the meter.
    /// </summary>
    public static readonly DependencyProperty ActiveColorProperty =
        DependencyProperty.Register(
            nameof(ActiveColor),
            typeof(Color),
            typeof(VolumeMeter),
            new PropertyMetadata(Colors.LimeGreen, OnVisualPropertyChanged));

    /// <summary>
    /// Color for the inactive (background) portion of the meter.
    /// </summary>
    public static readonly DependencyProperty InactiveColorProperty =
        DependencyProperty.Register(
            nameof(InactiveColor),
            typeof(Color),
            typeof(VolumeMeter),
            new PropertyMetadata(Colors.LightGray, OnVisualPropertyChanged));

    /// <summary>
    /// Color for the peak indicator.
    /// </summary>
    public static readonly DependencyProperty PeakColorProperty =
        DependencyProperty.Register(
            nameof(PeakColor),
            typeof(Color),
            typeof(VolumeMeter),
            new PropertyMetadata(Colors.Red, OnVisualPropertyChanged));

    /// <summary>
    /// Whether to show segment divisions in the meter.
    /// </summary>
    public static readonly DependencyProperty ShowSegmentsProperty =
        DependencyProperty.Register(
            nameof(ShowSegments),
            typeof(bool),
            typeof(VolumeMeter),
            new PropertyMetadata(true, OnVisualPropertyChanged));

    private float _peakLevel = 0.0f;
    private DateTime _peakTime = DateTime.MinValue;
    private const double PEAK_HOLD_DURATION = 1.5; // seconds
    private const double PEAK_DECAY_RATE = 0.95; // per frame

    /// <summary>
    /// Gets or sets the current audio level (0.0 to 1.0).
    /// </summary>
    public float Level
    {
        get => (float)GetValue(LevelProperty);
        set => SetValue(LevelProperty, Math.Max(0.0f, Math.Min(1.0f, value)));
    }

    /// <summary>
    /// Gets or sets the color for the active portion of the meter.
    /// </summary>
    public Color ActiveColor
    {
        get => (Color)GetValue(ActiveColorProperty);
        set => SetValue(ActiveColorProperty, value);
    }

    /// <summary>
    /// Gets or sets the color for the inactive portion of the meter.
    /// </summary>
    public Color InactiveColor
    {
        get => (Color)GetValue(InactiveColorProperty);
        set => SetValue(InactiveColorProperty, value);
    }

    /// <summary>
    /// Gets or sets the color for the peak indicator.
    /// </summary>
    public Color PeakColor
    {
        get => (Color)GetValue(PeakColorProperty);
        set => SetValue(PeakColorProperty, value);
    }

    /// <summary>
    /// Gets or sets whether to show segment divisions.
    /// </summary>
    public bool ShowSegments
    {
        get => (bool)GetValue(ShowSegmentsProperty);
        set => SetValue(ShowSegmentsProperty, value);
    }

    private static void OnLevelChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is VolumeMeter meter)
        {
            meter.UpdatePeakLevel((float)e.NewValue);
            meter.InvalidateVisual();
        }
    }

    private static void OnVisualPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is VolumeMeter meter)
        {
            meter.InvalidateVisual();
        }
    }

    protected override void OnPaintSurface(SKPaintSurfaceEventArgs e)
    {
        var canvas = e.Surface.Canvas;
        var info = e.Info;

        // Clear the canvas
        canvas.Clear(SKColors.Transparent);

        if (info.Width <= 0 || info.Height <= 0)
            return;

        // Calculate dimensions
        var rect = new SKRect(0, 0, info.Width, info.Height);
        var cornerRadius = Math.Min(info.Height * 0.2f, 4.0f);

        // Draw background
        using var backgroundPaint = new SKPaint
        {
            Color = ToSkiaColor(InactiveColor),
            Style = SKPaintStyle.Fill,
            IsAntialias = true
        };
        canvas.DrawRoundRect(rect, cornerRadius, cornerRadius, backgroundPaint);

        // Calculate active width based on level
        var activeWidth = rect.Width * Level;
        if (activeWidth > 0)
        {
            var activeRect = new SKRect(0, 0, activeWidth, rect.Height);
            
            // Create gradient for active area
            var gradient = CreateLevelGradient(activeRect, Level);
            using var activePaint = new SKPaint
            {
                Shader = gradient,
                Style = SKPaintStyle.Fill,
                IsAntialias = true
            };
            
            canvas.DrawRoundRect(activeRect, cornerRadius, cornerRadius, activePaint);
        }

        // Draw peak indicator
        DrawPeakIndicator(canvas, rect, cornerRadius);

        // Draw segments if enabled
        if (ShowSegments)
        {
            DrawSegments(canvas, rect);
        }

        // Draw border
        using var borderPaint = new SKPaint
        {
            Color = SKColors.Gray,
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 1.0f,
            IsAntialias = true
        };
        canvas.DrawRoundRect(rect, cornerRadius, cornerRadius, borderPaint);
    }

    private void UpdatePeakLevel(float currentLevel)
    {
        var now = DateTime.UtcNow;
        
        if (currentLevel > _peakLevel)
        {
            // New peak detected
            _peakLevel = currentLevel;
            _peakTime = now;
        }
        else
        {
            // Decay peak level over time
            var timeSincePeak = (now - _peakTime).TotalSeconds;
            if (timeSincePeak > PEAK_HOLD_DURATION)
            {
                _peakLevel *= (float)PEAK_DECAY_RATE;
                if (_peakLevel < currentLevel)
                {
                    _peakLevel = currentLevel;
                }
            }
        }
    }

    private SKShader CreateLevelGradient(SKRect rect, float level)
    {
        // Create color gradient based on level (green -> yellow -> red)
        var colors = new SKColor[3];
        var positions = new float[] { 0.0f, 0.7f, 1.0f };

        if (level < 0.7f)
        {
            // Green to yellow
            colors[0] = SKColors.LimeGreen;
            colors[1] = SKColors.Yellow;
            colors[2] = SKColors.Yellow;
        }
        else
        {
            // Yellow to red
            colors[0] = SKColors.LimeGreen;
            colors[1] = SKColors.Yellow;
            colors[2] = SKColors.Red;
        }

        return SKShader.CreateLinearGradient(
            new SKPoint(0, rect.MidY),
            new SKPoint(rect.Width, rect.MidY),
            colors,
            positions,
            SKShaderTileMode.Clamp);
    }

    private void DrawPeakIndicator(SKCanvas canvas, SKRect rect, float cornerRadius)
    {
        if (_peakLevel > Level && _peakLevel > 0.01f)
        {
            var peakX = rect.Width * _peakLevel;
            
            using var peakPaint = new SKPaint
            {
                Color = ToSkiaColor(PeakColor),
                Style = SKPaintStyle.Fill,
                IsAntialias = true
            };

            // Draw peak line
            var peakLine = new SKRect(peakX - 1, 0, peakX + 1, rect.Height);
            canvas.DrawRect(peakLine, peakPaint);
        }
    }

    private void DrawSegments(SKCanvas canvas, SKRect rect)
    {
        const int segmentCount = 10;
        var segmentWidth = rect.Width / segmentCount;

        using var segmentPaint = new SKPaint
        {
            Color = SKColors.Gray.WithAlpha(100),
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 0.5f,
            IsAntialias = true
        };

        for (int i = 1; i < segmentCount; i++)
        {
            var x = i * segmentWidth;
            canvas.DrawLine(x, 0, x, rect.Height, segmentPaint);
        }
    }

    private static SKColor ToSkiaColor(Color wpfColor)
    {
        return new SKColor(wpfColor.R, wpfColor.G, wpfColor.B, wpfColor.A);
    }

    /// <summary>
    /// Sets the level with smooth animation support.
    /// </summary>
    /// <param name="level">Target level (0.0 to 1.0)</param>
    public void SetLevelSmooth(float level)
    {
        // For smooth animation, you could implement easing here
        Level = level;
    }

    /// <summary>
    /// Resets the peak indicator.
    /// </summary>
    public void ResetPeak()
    {
        _peakLevel = 0.0f;
        _peakTime = DateTime.MinValue;
        InvalidateVisual();
    }
} 