# Code Style for Oreja

This document outlines coding conventions for Oreja.

## C# Conventions
- **Version**: C# 12 with .NET 8.
- **Naming**:
  - Classes/Methods/Properties: PascalCase (e.g., `AudioService`).
  - Private Fields/Variables: camelCase (e.g., `_audioBuffer`).
  - Interfaces: `I` prefix (e.g., `IAudioService`).
  - Constants: UPPER_CASE (e.g., `MAX_BUFFER_SIZE`).
- **Documentation**: XML comments for public members:
  ```csharp
  /// <summary>
  /// Starts audio capture.
  /// </summary>
  public async Task StartCaptureAsync() { ... }
  ```
- **Async**: Use `async/await`, suffix with `Async`.
- **Error Handling**: Try-catch for NAudio, HTTP, SQLite; log with Serilog.
- **Disposables**: Use `using` for `IDisposable`:
  ```csharp
  using var capture = new WasapiCapture(deviceIndex);
  ```
- **Formatting**: `dotnet format`.

## Python Conventions
- **Version**: Python 3.10, follow PEP 8.
- **Naming**:
  - Functions/Variables: snake_case (e.g., `process_audio`).
  - Classes: CamelCase (e.g., `AudioProcessor`).
  - Constants: UPPER_CASE (e.g., `SAMPLE_RATE`).
- **Documentation**: Docstrings for functions/classes:
  ```python
  def transcribe_audio(audio: bytes) -> list:
      """Transcribes audio using Whisper."""
      ...
  ```
- **Async**: Use `async def` for FastAPI routes.
- **Error Handling**: Try-except for model inference, logging with `logging`.
- **Formatting**: `black` and `isort`.

## XAML Conventions
- **Naming**: Descriptive control names (e.g., `MicComboBox`).
- **Binding**: Explicit `Path`:
  ```xaml
  <TextBox Text="{Binding Path=TranscriptionText, Mode=OneWay}"/>
  ```
- **Layout**: Use `Grid`/`StackPanel`, avoid absolute positioning.
- **MVVM**: Bind to view model properties, minimal code-behind.

## File Organization
- C#: One class per file, named after class (e.g., `AudioService.cs`).
- Python: One module per file (e.g., `server.py`).
- Group in folders: `/Oreja/Models`, `/backend`.

## Example Code
**C#**:
```csharp
namespace Oreja.Services;

public class AudioService : IAudioService
{
    private readonly WasapiCapture _capture;

    public async Task StartCaptureAsync(int deviceIndex)
    {
        try
        {
            _capture = new WasapiCapture(deviceIndex);
            _capture.StartRecording();
        }
        catch (Exception ex)
        {
            // Log with Serilog
            throw;
        }
    }
}
```

**Python**:
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile) -> list:
    """Transcribes and diarizes audio."""
    ...
```

**XAML**:
```xaml
<Window x:Class="Oreja.Views.MainWindow">
    <Grid>
        <TextBox Text="{Binding Path=TranscriptionText, Mode=OneWay}"/>
    </Grid>
</Window>
```

## Tools
- **C#**: `dotnet format`, Roslyn analyzers.
- **Python**: `black`, `isort`, `flake8`.
- **IDE**: Visual Studio/Rider/VS Code with Cursor.

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup.