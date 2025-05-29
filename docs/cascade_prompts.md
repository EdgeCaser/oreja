# Windsurf AI Cascade Prompts for Oreja Project Files

These prompts are designed for use in the **Windsurf AI IDE**’s Cascade feature to generate or refine files for the `oreja` project (`C:\Users\ianfe\OneDrive\Documents\GitHub\oreja`). The project captures system audio on Windows using C# and NAudio, saving it as a WAV file, with potential integration of Hugging Face’s Whisper API for transcription. Each prompt targets a specific file to ensure a cohesive C# project compatible with .NET 8 and Windows 10/11. Use these in Cascade’s panel (right side of the IDE) by typing or pasting the prompt and selecting “Generate File” or “Edit File” as needed [,].

## 1. Program.cs
**Purpose**: Main C# code to capture system audio using NAudio’s WASAPI loopback and optionally call the Hugging Face Whisper API for transcription.

**Prompt**:
```
Generate a C# file named 'Program.cs' for my project at 'C:\Users\ianfe\OneDrive\Documents\GitHub\oreja\Oreja'. The code should use NAudio to capture system audio via WASAPI loopback for 10 seconds, saving it as 'system_audio_output.wav'. Include error handling for device issues and invalid audio streams. Add a placeholder method to call Hugging Face’s Whisper API using a token stored in .NET Secret Manager as 'HuggingFace:ApiKey'. Ensure compatibility with .NET 8 and Windows 10/11. Include comments explaining each section and use a try-catch block for robustness. The namespace should be 'Oreja'.
```

**Notes**:
- Cascade will generate code similar to the previous `Program.cs` artifact but with a placeholder for the Hugging Face API.
- If you want to implement transcription now, modify the prompt to: “Add full Hugging Face Whisper API integration to transcribe the WAV file and save the output as 'transcription.txt’.”

## 2. Oreja.csproj
**Purpose**: C# project file defining dependencies (NAudio, System.Net.Http for API calls) and .NET 8 target framework.

**Prompt**:
```
Generate a C# project file named 'Oreja.csproj' for my project at 'C:\Users\ianfe\OneDrive\Documents\GitHub\oreja\Oreja'. Target .NET 8.0 and include dependencies for NAudio (latest version) and System.Net.Http for Hugging Face API calls. Ensure the project is a console application. Add XML comments for clarity and include Microsoft.Extensions.Configuration.UserSecrets for accessing the Hugging Face token stored in .NET Secret Manager.
```

**Expected Output** (approximate):
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <UserSecretsId>oreja-1234-5678-9012-345678901234</UserSecretsId>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="NAudio" Version="2.2.1" />
    <PackageReference Include="Microsoft.Extensions.Configuration.UserSecrets" Version="8.0.0" />
  </ItemGroup>
</Project>
```

## 3. README.md
**Purpose**: Project overview, setup instructions, and usage guide for the `oreja` repository.

**Prompt**:
```
Generate a 'README.md' file for my GitHub repository at 'C:\Users\ianfe\OneDrive\Documents\GitHub\oreja'. The project, Oreja, captures system audio on Windows using C# and NAudio in the Windsurf AI IDE, saving it as a WAV file, with optional Hugging Face Whisper API integration for transcription. Include sections for: project description, prerequisites (.NET 8, Windsurf AI, Hugging Face token), installation steps (clone repo, install dependencies, set up token), usage (run the program, play audio), and contributing guidelines. Mention the repository path and compatibility with Windows 10/11.
```

**Notes**:
- Cascade’s “Write Mode” can refine the markdown for clarity [].
- If the generated README is too verbose, edit it in Windsurf AI or use Cascade: “Simplify this README.md to focus on setup and usage.”

## 4. .gitignore
**Purpose**: Excludes sensitive files (e.g., secrets, build outputs) from Git commits.

**Prompt**:
```
Generate a '.gitignore' file for my C# project at 'C:\Users\ianfe\OneDrive\Documents\GitHub\oreja'. Include standard C#/.NET ignore patterns for bin/, obj/, and .vs/ directories. Add patterns to exclude .NET Secret Manager files (*.secrets.json) and potential .env files. Ensure it’s compatible with a Windsurf AI IDE project using .NET 8.
```

**Expected Output** (approximate):
```
# .NET
bin/
obj/
*.csproj.user
*.secrets.json
.env

# Visual Studio/Windsurf AI
.vscode/
.vs/
```

## 5. docs/instructions.md
**Purpose**: Detailed setup and development guide, as provided in the previous artifact.

**Prompt**:
```
Generate a markdown file named 'instructions.md' in the 'docs' folder of my project at 'C:\Users\ianfe\OneDrive\Documents\GitHub\oreja'. The file should detail how to set up a C# project in Windsurf AI IDE to capture system audio using NAudio on Windows. Include sections for prerequisites (.NET 8, Windsurf AI, Hugging Face token), project setup (create repo, install C# Dev Kit extension, add NAudio), storing the Hugging Face token in .NET Secret Manager, implementation steps (update Program.cs, test capture), and troubleshooting (e.g., empty WAV, token errors). Ensure compatibility with Windows 10/11 and mention the repository path.
```

**Notes**:
- This prompt regenerates the `instructions.md` from the previous artifact but ensures Cascade optimizes it for Windsurf AI.
- If you already have the `instructions.md` artifact, skip this prompt or use Cascade to refine it: “Update this instructions.md to emphasize Windsurf AI’s Cascade feature.”