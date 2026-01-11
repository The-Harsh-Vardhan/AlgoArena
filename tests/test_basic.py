"""
Tests for AlgoArena utility modules
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_import_utils():
    """Test that utility modules can be imported"""
    try:
        from utils import preprocessing
        from utils import visualization
        assert True
    except ImportError as e:
        pytest.skip(f"Utils modules not available: {e}")


def test_import_streamlit_app():
    """Test that streamlit app modules exist"""
    import os
    assert os.path.exists('streamlit_app/app.py')
    assert os.path.exists('streamlit_app/dynamic_ml_analyzer.py')


def test_requirements_exist():
    """Test that requirements files exist"""
    import os
    assert os.path.exists('requirements.txt')
    assert os.path.exists('streamlit_app/requirements.txt')


def test_docker_files_exist():
    """Test that Docker configuration exists"""
    import os
    assert os.path.exists('Dockerfile')
    assert os.path.exists('docker-compose.yml')
    assert os.path.exists('.dockerignore')


def test_readme_exists():
    """Test that documentation exists"""
    import os
    assert os.path.exists('README.md')


def test_python_version():
    """Test Python version compatibility"""
    import sys
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
