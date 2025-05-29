# Contributing to Oreja

Thank you for considering contributing to Oreja! This document outlines the process for contributing.

## Getting Started
1. Fork the repository and clone your fork.
2. Create a branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Install prerequisites (see [DEVELOPMENT.md](DEVELOPMENT.md)).
4. Make changes, following the guidelines below.

## Coding Standards
- **C#**: Use C# 12 with .NET 8, MVVM for WPF, PascalCase for public members, camelCase for private fields.
- **Python**: Use Python 3.10, follow PEP 8, prefer async for FastAPI routes.
- **Privacy**: Ensure no audio is stored or sent to the cloud; process in memory.
- **Documentation**: Use XML comments for C# public members, docstrings for Python functions.
- See [CODE_STYLE.md](CODE_STYLE.md) for details.

## Project Structure
- `/Oreja`: C# frontend (WPF, NAudio, SQLite).
  - `/Models`: Data models and DTOs.
  - `/ViewModels`: MVVM view models.
  - `/Views`: XAML files and code-behind.
  - `/Services`: Audio and database services.
  - `/Utilities`: Helpers and extensions.
- `/backend`: Python backend (FastAPI, Whisper, pyannote.audio).
- `/scripts`: Utility scripts for model downloading.

## Pull Request Process
1. Write unit tests for C# (xUnit) and Python (pytest).
2. Ensure code builds and tests pass:
   ```bash
   dotnet test
   pytest backend/tests
   ```
3. Format code:
   - C#: `dotnet format`
   - Python: `black .`
4. Submit a PR with a clear description and reference related issues.
5. Address feedback during review.

## Reporting Issues
- Use GitHub Issues for bugs or feature requests.
- Include Windows version, Python version, and audio setup for bugs.

## Cursor Integration
- Use `.cursorrules` for context-aware completions.
- See [DEVELOPMENT.md](DEVELOPMENT.md) for Cursor tips.

## Questions?
Contact maintainers via GitHub Issues.