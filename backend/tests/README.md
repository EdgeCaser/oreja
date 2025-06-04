# Oreja Backend Test Suite

This directory contains comprehensive unit tests, integration tests, and QA tools for the Oreja audio transcription backend.

## Overview

The test suite covers:
- **API Endpoints** - All FastAPI routes and error handling
- **Speaker Recognition** - Speaker enrollment, identification, and embedding management  
- **Audio Processing** - Audio loading, validation, and transcription pipeline
- **Batch Processing** - File transcription and result management
- **Utility Functions** - Core helper functions and data processing

## Quick Start

### Install Test Dependencies

```bash
# Install test dependencies
cd backend
python run_tests.py --install-deps

# Or manually:
pip install -r requirements_test.txt
```

### Run Tests

```bash
# Check test environment
python run_tests.py --check

# Run quick test suite (default)
python run_tests.py

# Run all tests with coverage
python run_tests.py --all --verbose

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --api           # API endpoint tests
python run_tests.py --speaker       # Speaker recognition tests
python run_tests.py --integration   # Integration tests

# Generate HTML reports
python run_tests.py --report
```

## Test Structure

### Test Files

| File | Purpose | Coverage |
|------|---------|----------|
| `test_api_endpoints.py` | FastAPI endpoint testing | All `/transcribe/*` endpoints, error handling, file uploads |
| `test_speaker_embeddings.py` | Speaker recognition system | SpeakerProfile class, embedding manager, database operations |
| `test_utils.py` | Utility function testing | Audio processing, transcription merging, validation |
| `test_batch_transcription.py` | Batch processing | File transcription, directory processing, result saving |
| `conftest.py` | Shared test fixtures | Mock data, audio samples, validation utilities |

### Test Categories (Markers)

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests (< 1s each)
- `@pytest.mark.integration` - Integration tests requiring multiple components
- `@pytest.mark.api` - API endpoint tests  
- `@pytest.mark.speaker` - Speaker recognition tests
- `@pytest.mark.slow` - Performance/slow tests (> 5s each)

### Fixtures Available

The `conftest.py` provides these reusable fixtures:

**Audio Data:**
- `sample_audio_data` - Basic audio waveform and sample rate
- `sample_audio_bytes` - WAV audio as bytes for API testing
- `sample_short_audio` - Very short audio (< 0.1s)
- `sample_long_audio` - Long audio (> 30s)

**Mock Objects:**
- `mock_whisper_model` - Mocked Whisper transcription model
- `mock_diarization_pipeline` - Mocked speaker diarization  
- `mock_whisper_response` - Sample transcription result
- `mock_diarization_response` - Sample speaker diarization result

**Speaker Data:**
- `sample_speaker_profile` - Complete speaker profile with embeddings
- `speaker_profiles_list` - Multiple speaker profiles for testing

**API Testing:**
- `test_client` - FastAPI test client
- `temp_upload_file` - Temporary file for upload testing

## Running Specific Tests

### By Test Function

```bash
# Run specific test function
python -m pytest tests/test_api_endpoints.py::TestTranscriptionEndpoints::test_transcribe_file_success -v

# Run all tests in a class
python -m pytest tests/test_speaker_embeddings.py::TestSpeakerProfile -v
```

### By Marker

```bash
# Run only unit tests
python -m pytest -m unit

# Run everything except slow tests  
python -m pytest -m "not slow"

# Run API and speaker tests
python -m pytest -m "api or speaker"
```

### With Coverage

```bash
# Generate coverage report
python -m pytest --cov=. --cov-report=term-missing

# Generate HTML coverage report
python -m pytest --cov=. --cov-report=html
```

## Test Configuration

### pytest.ini Settings

```ini
[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers
testpaths = tests
markers =
    unit: Fast unit tests
    integration: Integration tests  
    api: API endpoint tests
    speaker: Speaker recognition tests
    slow: Slow or performance tests
```

### Coverage Configuration

Coverage is configured to:
- Include all Python files in the backend directory
- Exclude test files from coverage measurement
- Generate terminal and HTML reports
- Fail if coverage drops below 80%

## Writing New Tests

### Test File Structure

```python
"""
Brief description of what this test file covers.
"""

import pytest
from unittest.mock import Mock, patch

# Test class for organizing related tests
class TestMyFeature:
    """Test MyFeature functionality."""
    
    @pytest.mark.unit
    def test_basic_functionality(self, relevant_fixture):
        """Test the basic case."""
        # Arrange
        input_data = "test"
        
        # Act  
        result = my_function(input_data)
        
        # Assert
        assert result == expected_output
    
    @pytest.mark.unit
    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(ValueError, match="Expected error message"):
            my_function(invalid_input)
```

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `TestFeatureName`
- Test methods: `test_what_it_tests`
- Use descriptive names that explain the scenario being tested

### Mocking Guidelines

```python
# Mock external dependencies
@patch('module.external_dependency')
def test_with_mock(mock_dependency):
    mock_dependency.return_value = expected_value
    # Test code here

# Use fixtures for complex mock setups
@pytest.fixture
def mock_complex_object():
    mock_obj = Mock()
    mock_obj.method.return_value = "test"
    return mock_obj
```

### Adding Fixtures

Add new fixtures to `conftest.py`:

```python
@pytest.fixture
def my_test_data():
    """Provide test data for my feature."""
    return {
        "field1": "value1", 
        "field2": "value2"
    }
```

## Continuous Integration

### GitHub Actions (if applicable)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r backend/requirements_test.txt
      - name: Run tests
        run: |
          cd backend
          python run_tests.py --all --no-coverage
```

### Local Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/sh
cd backend
python run_tests.py --quick
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

## Test Data Management

### Audio Test Files

Test audio files are generated programmatically to avoid large binary files in the repository:

```python
# Generate test audio in fixtures
sample_rate = 16000
duration = 2.0  # seconds
t = torch.linspace(0, duration, int(sample_rate * duration))
waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz tone
```

### Mock Data

Speaker embeddings and other complex data structures are mocked:

```python
# Mock speaker embedding
mock_embedding = torch.randn(512)  # 512-dimensional embedding
```

## Performance Testing

### Slow Test Guidelines

Tests marked with `@pytest.mark.slow` should:
- Test realistic data sizes and processing times
- Validate memory usage doesn't grow excessively
- Check performance under load
- Be skipped in quick test runs

### Memory Testing

```python
import psutil
import os

def test_memory_usage():
    """Test that processing doesn't use excessive memory."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operation
    large_operation()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Assert memory increase is reasonable
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

## Debugging Tests

### Running with Debug Output

```bash
# Extra verbose output
python -m pytest -vvv

# Show print statements  
python -m pytest -s

# Drop into debugger on failure
python -m pytest --pdb

# Show test duration
python -m pytest --durations=10
```

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the backend directory
2. **Fixture Not Found**: Check fixture is defined in `conftest.py` or imported
3. **Mock Not Working**: Verify the patch path matches the actual import path
4. **Async Tests Failing**: Use `@pytest.mark.asyncio` for async test functions

## Coverage Goals

- **Overall Coverage**: > 80%
- **Critical Functions**: > 95% (audio processing, API endpoints)
- **Error Handling**: All error paths tested
- **Edge Cases**: Boundary conditions and invalid inputs

## Maintenance

### Regular Tasks

- Update test dependencies when main dependencies change
- Add tests for new features before merging
- Review and update slow/integration tests monthly
- Check coverage reports to identify untested code paths

### Test Review Checklist

- [ ] All new features have unit tests
- [ ] Error cases are tested
- [ ] Integration points are tested
- [ ] Performance impact is considered
- [ ] Tests are documented and named clearly
- [ ] Mock objects are used appropriately
- [ ] Test data is realistic but minimal

---

## Getting Help

- Check the [pytest documentation](https://docs.pytest.org/)
- Review existing test files for patterns and examples
- Ask questions in team discussions about testing approaches
- Consider pair programming when writing complex tests

For questions specific to the Oreja test suite, review the test files and fixtures in this directory. 