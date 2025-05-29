using System.Windows;
using Microsoft.Extensions.Logging;
using Oreja.ViewModels;

namespace Oreja.Views;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    private readonly ILogger<MainWindow> _logger;
    private readonly MainViewModel _viewModel;

    public MainWindow(ILogger<MainWindow> logger, MainViewModel viewModel)
    {
        try
        {
            _logger = logger;
            _viewModel = viewModel;
            
            InitializeComponent();
            DataContext = _viewModel;
            
            _logger.LogInformation("MainWindow initialized successfully");
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Error initializing MainWindow");
            throw;
        }
    }

    private void Window_Loaded(object sender, RoutedEventArgs e)
    {
        try
        {
            _logger.LogInformation("MainWindow loaded");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in Window_Loaded");
        }
    }

    private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        try
        {
            _logger.LogInformation("MainWindow closing");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in Window_Closing");
        }
    }
} 