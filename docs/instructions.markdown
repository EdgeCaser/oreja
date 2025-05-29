# Capturing System Audio with C# and NAudio in the Oreja Project Using Windsurf AI IDE

This document provides instructions for building a C# tool to capture system audio (e.g., speaker output from applications like video players or conference calls) on Windows using the NAudio library within the **Windsurf AI IDE** (formerly Codeium). The project is hosted in the GitHub repository `C:\Users\ianfe\OneDrive\Documents\GitHub\oreja`. The tool will save captured audio as a WAV file, suitable for transcription with Hugging Face models (e.g., Whisper). These instructions assume you are working on a Windows system with Windsurf AI IDE, .NET 8, and a Hugging Face token.

## Prerequisites
- **Operating System**: Windows 10 or later (WASAPI loopback is supported from Windows Vista onward).
- **Hardware**: A system with audio output capabilities (e.g., speakers or headphones). Your NVIDIA GeForce RTX 3080 system is sufficient.
- **Windsurf AI IDE**: Download from [windsurf.com](https://windsurf.com) for Windows. Ensure it’s installed and logged in with your Codeium account.
- **.NET SDK**: Version 8 or later, downloadable from [dotnet.microsoft.com](https://dotnet.microsoft.com/). Verify installation:
  ```bash
  dotnet --version
  ```
- **Git**: Installed and configured for GitHub. The repository is at `C:\Users\ianfe\OneDrive\Documents\GitHub\oreja`.
- **Hugging Face Token**: Required for accessing Hugging Face models. Store it securely as described below.
- **Windows Sound Settings**: Ensure your audio output device (e.g., speakers) is active. NAudio’s WASAPI loopback does not require “Stereo Mix,” but verify:
  1. Right-click the sound icon in the system tray and select “Sounds.”
  2. In the “Playback” tab, confirm your default output device is set (e.g., “Speakers”).

## Project Setup
1. **Initialize the GitHub Repository**:
   - Navigate to the repository directory:
     ```bash
     cd C:\Users\ianfe\OneDrive\Documents\GitHub\oreja
     ```
   - Initialize the Git repository if not already done:
     ```bash
     git init
     echo "# Oreja: System Audio Capture with Windsurf AI" > README.md
     git add README.md
     git commit -m "Initial commit"
     git remote add origin <your-repo-url>
     git push -u origin main
     ```

2. **Set Up Windsurf AI IDE**:
   - Open Windsurf AI IDE.
   - Import VS Code settings (optional) for familiarity:
     - Go to Settings (bottom right or `Ctrl+,`) and select “Import VS Code/Cursor Configuration.”
   - Install the **C# Dev Kit** extension for .NET support:
     - Open the Extensions panel (`Ctrl+Shift+X`).
     - Search for “C# Dev Kit” by Microsoft and install it.
   - Open the `oreja` folder:
     - Click “Open Folder” and select `C:\Users\ianfe\OneDrive\Documents\GitHub\oreja`.

3. **Create a C# Project**:
   - In Windsurf AI, open the terminal (`Ctrl+``) and create a new Console App:
     ```bash
     cd C:\Users\ianfe\OneDrive\Documents\GitHub\oreja
     dotnet new console -n Oreja
     ```
   - This generates `Oreja.csproj` and `Program.cs` in `oreja\Oreja`.
   - Open the project in Windsurf AI:
     - Select “Open Folder” and choose `C:\Users\ianfe\OneDrive\Documents\GitHub\oreja\Oreja`.

4. **Add NAudio Dependency**:
   - In Windsurf AI’s terminal, add NAudio via NuGet:
     ```bash
     dotnet add package NAudio
     ```
   - Alternatively, use Windsurf AI’s Cascade feature:
     - Open the Cascade panel (right side of the IDE).
     - Type: “Add NAudio package to my C# project.”
     - Cascade will generate the `dotnet add package NAudio` command; confirm execution.

5. **Store the Hugging Face Token**:
   - **Option 1: .NET Secret Manager** (Recommended):
     - In Windsurf AI’s terminal, initialize the Secret Manager:
       ```bash
       cd C:\Users\ianfe\OneDrive\Documents\GitHub\oreja\Oreja
       dotnet user-secrets init
       ```
     - Add your Hugging Face token:
       ```bash
       dotnet user-secrets set "HuggingFace:ApiKey" "<your-hugging-face-token>"
       ```
     - Access in C# (example in `Program.cs` if needed for transcription).
   - **Option 2: Environment Variable**:
     - Set in Windows:
       1. Open Control Panel > System > Advanced system settings > Environment Variables.
       2. Add under “User variables”:
          - Name: `HUGGINGFACE_API_KEY`
          - Value: `<your-hugging-face-token>`
     - Access in C#: `Environment.GetEnvironmentVariable("HUGGINGFACE_API_KEY")`.
   - **Security Note**: Add secrets to `.gitignore`:
     ```bash
     echo "*.secrets.json" >> .gitignore
     git add .gitignore
     git commit -m "Add .gitignore for secrets"
     ```

## Implementation Steps
1. **Update `Program.cs`**:
   - Replace the default `Program.cs` with the code in the separate artifact (Program.cs, artifact ID: `d472d3a8-8f31-4df6-a714-fa07f4e1a6c3`). It captures system audio for 10 seconds and saves it as `system_audio_output.wav`.
   - Use Windsurf AI’s Cascade to assist:
     - Open `Program.cs`, select the code, and type in Cascade: “Explain this C# NAudio code for audio capture.”
     - Cascade will provide insights or suggest improvements.

2. **Use Cascade Prompts for Additional Files**:
   - Refer to `docs/cascade_prompts.md` (artifact ID: `d8bedfd9-36e6-4e22-856b-e8a13b107520`) for file-specific prompts to generate or refine:
     - `Program.cs`: Main audio capture logic.
     - `Oreja.csproj`: Project dependencies.
     - `README.md`: Project documentation.
     - `.gitignore`: Exclude sensitive files.
     - `instructions.md`: This file (optional, for regeneration).
   - In Windsurf AI, open the Cascade panel and paste the relevant prompt to create or edit these files. For example:
     - For `README.md`: “Generate a README.md file for my GitHub repository at ‘C:\Users\ianfe\OneDrive\Documents\GitHub\oreja’.”
   - Save generated files in the appropriate locations (e.g., `oreja\README.md`, `oreja\Oreja\Oreja.csproj`).

3. **Test the Audio Capture**:
   - In Windsurf AI, run the project:
     - Press `F5` or use the terminal:
       ```bash
       dotnet run
       ```
   - Play audio (e.g., a YouTube video) during the 10-second recording.
   - Check the output file `system_audio_output.wav` in `oreja\Oreja\bin\Debug\net8.0`.
   - If errors occur, use Cascade:
     - Type: “Debug why my NAudio code isn’t capturing audio.”
     - Cascade may suggest checking the playback device or NAudio configuration.

4. **Optional: Integrate Hugging Face Models**:
   - For transcription with Hugging Face’s Whisper API, add code to `Program.cs` using the Cascade prompt from `cascade_prompts.md` (Prompt 2 for `Program.cs`).
   - Example placeholder:
     ```csharp
     using System.Net.Http;
     using System.Net.Http.Headers;
     using Microsoft.Extensions.Configuration;

     var configuration = new ConfigurationBuilder().AddUserSecrets<Program>().Build();
     var apiKey = configuration["HuggingFace:ApiKey"];
     using var client = new HttpClient();
     client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
     // Add API call logic here
     ```
   - Use Cascade: Type “Add C# code to call Hugging Face Whisper API with my token.”

5. **Commit Changes**:
   - Use Windsurf AI’s Source Control panel (`Ctrl+Shift+G`) to commit:
     ```bash
     git add .
     git commit -m "Add C# audio capture with NAudio and Cascade prompts"
     git push
     ```
   - Or use Cascade: Type “Generate a commit message for adding NAudio audio capture.”

## Troubleshooting
- **No Audio Captured**:
  - Ensure audio is playing during recording.
  - Verify the default playback device in Windows Sound settings.
  - Use Cascade: “Why is my NAudio WASAPI loopback capture producing an empty WAV file?”
- **Windsurf AI Issues**:
  - If Cascade suggestions are incorrect, type: “Check your guidelines and revise your suggestion.”
  - For login issues, submit a ticket via Codeium’s support forum.
- **Hugging Face Token Errors**:
  - Verify the token in Secret Manager or environment variable.
  - Test with a curl command: `curl -H "Authorization: Bearer <token>" https://api.huggingface.co/models`.
- **C# Extension Issues**:
  - Ensure the C# Dev Kit extension is installed and .NET 8 is recognized (`dotnet --version`).

## Next Steps
- Test the audio capture and verify the WAV file.
- Use `docs/cascade_prompts.md` to generate or refine project files with Cascade.
- If adding transcription, use the Hugging Face API prompt or request further code.
- Update the GitHub repo’s `README.md` using the Cascade prompt.
- Expand the tool (e.g., real-time audio monitoring) with Cascade prompts.