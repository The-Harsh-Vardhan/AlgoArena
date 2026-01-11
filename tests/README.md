# 🧪 AlgoArena Test Suite

This directory contains automated tests for the AlgoArena project.

## 📁 Structure

- `test_basic.py` - Basic smoke tests for project structure and imports
- `conftest.py` - Pytest configuration and fixtures
- `__init__.py` - Test package initialization

## 🚀 Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ --cov=streamlit_app --cov=utils --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_basic.py -v
```

## 📊 Current Tests

### Basic Tests (`test_basic.py`)
- ✅ **test_import_utils** - Verifies utility modules can be imported
- ✅ **test_import_streamlit_app** - Checks streamlit app files exist
- ✅ **test_requirements_exist** - Validates requirements files
- ✅ **test_docker_files_exist** - Confirms Docker configuration
- ✅ **test_readme_exists** - Ensures documentation is present
- ✅ **test_python_version** - Checks Python version compatibility

## 🔧 Adding New Tests

To add new tests:

1. Create a new file named `test_*.py`
2. Import pytest: `import pytest`
3. Write test functions starting with `test_`
4. Run tests to verify

Example:
```python
def test_my_feature():
    """Test description"""
    result = my_function()
    assert result == expected_value
```

## 📝 Testing Best Practices

- **Keep tests simple** - One assertion per test when possible
- **Use descriptive names** - Test names should describe what they test
- **Use fixtures** - Share setup code via pytest fixtures
- **Mock external calls** - Don't rely on external services
- **Test edge cases** - Include boundary conditions and error cases

## 🎯 CI/CD Integration

Tests automatically run on GitHub Actions for:
- Pull requests to `main` branch
- Pushes to `main` and `develop` branches
- Python versions: 3.10, 3.11

## 📚 Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [GitHub Actions](https://github.com/The-Harsh-Vardhan/AlgoArena/actions)

---

**Note:** This is a minimal test suite. More comprehensive tests will be added as the project grows.
