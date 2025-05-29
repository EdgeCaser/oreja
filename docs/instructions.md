# Oreja Setup and Usage Guide

This document provides detailed instructions for setting up and using the Oreja system audio capture application in Windsurf AI.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Project Setup](#project-setup)
3. [Adding the Hugging Face API Token](#adding-the-hugging-face-api-token)
4. [Running the Application](#running-the-application)
5. [Troubleshooting](#troubleshooting)

## Environment Setup

### Prerequisites
- Windows 10 or 11
- .NET 8.0 SDK installed
- Windsurf AI IDE installed
- A Hugging Face account with an API token (for transcription functionality)

### Installing Windsurf AI
1. Download and install Windsurf AI from the official website
2. Launch Windsurf AI and sign in to your account
3. Install the C# Dev Kit extension:
   - Open the Extensions view (Ctrl+Shift+X)
   - Search for "C# Dev Kit"
   - Click Install

## Project Setup

### Cloning the Repository
```powershell
# Clone the repository
git clone https://github.com/yourusername/oreja.git
cd oreja

# Initialize git if not already initialized
git init
```

### Opening the Project in Windsurf AI
1. In Windsurf AI, select File > Open Folder
2. Navigate to the cloned repository folder and click "Select Folder"

### Installing Dependencies
The project uses the following NuGet packages:
- NAudio (for audio capture)
- Microsoft.Extensions.Configuration.UserSecrets (for secure API key storage)
- System.Net.Http (for API requests)

These dependencies are specified in the `Oreja.csproj` file and will be restored automatically when you build the project.

To manually restore packages:
```powershell
dotnet restore
```

## Adding the Hugging Face API Token

To use the transcription functionality, you need to add your Hugging Face API token to the .NET Secret Manager:

1. Sign up for a Hugging Face account at [huggingface.co](https://huggingface.co) if you don't have one
2. Generate an API token in your Hugging Face account settings
3. Add your token to the .NET Secret Manager:

```powershell
dotnet user-secrets init --project Oreja
dotnet user-secrets set "HuggingFace:ApiKey" "your-api-token-here" --project Oreja
```

## Running the Application

### Building and Running
To build and run the application:

```powershell
dotnet run
```

The application will:
1. Capture system audio for 10 seconds
2. Save the audio as `system_audio_output.wav` in the output directory (typically `Oreja\bin\Debug\net8.0\`)

### Enabling Transcription
To enable transcription of the captured audio:

1. Open `Program.cs`
2. Uncomment the line: `// await TranscribeAudio(OutputFilePath);` in the Main method
3. Run the application again

## Troubleshooting

### No Audio Captured
- Ensure your system is playing audio during the capture period
- Check that you have the correct audio device selected as your default playback device
- Verify that NAudio has permission to access your audio devices

### API Key Issues
- Verify that you've correctly added your Hugging Face API token to the Secret Manager
- Check that the token is valid and has not expired
- Ensure you have sufficient quota/credits on your Hugging Face account

### Build Errors
- Make sure you have .NET 8.0 SDK installed
- Restore NuGet packages: `dotnet restore`
- Clean and rebuild the project: `dotnet clean` followed by `dotnet build`

For file-specific Cascade prompts that can help you generate or refine code in this project, please refer to [cascade_prompts.md](cascade_prompts.md).

## Additional Resources
- [NAudio Documentation](https://github.com/naudio/NAudio)
- [Hugging Face API Documentation](https://huggingface.co/docs/api-inference/index)
- [.NET Secret Manager Documentation](https://learn.microsoft.com/en-us/aspnet/core/security/app-secrets)
