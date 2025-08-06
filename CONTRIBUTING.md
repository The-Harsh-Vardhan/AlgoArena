# Contributing to AlgoArena

Thank you for your interest in contributing to AlgoArena! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs
- Provide detailed information about the problem
- Include steps to reproduce the issue
- Specify your environment (Python version, OS, etc.)

### Suggesting Features

- Check if the feature has already been suggested
- Provide a clear description of the feature
- Explain why it would be beneficial
- Consider implementation complexity

### Code Contributions

#### Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature
4. Make your changes
5. Test your changes thoroughly
6. Submit a pull request

#### Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Keep functions focused and small

#### Adding New Algorithms

When adding new ML algorithms:

1. Include proper documentation
2. Add performance evaluation
3. Update the comparison dashboard
4. Test on the existing datasets
5. Include references to algorithm papers/sources

#### Adding New Data Domains

For new data types (e.g., video, 3D data):

1. Create a new folder with appropriate structure
2. Include a comprehensive README
3. Provide sample datasets or links
4. Implement multiple algorithms
5. Create visualizations for results

### Pull Request Process

1. Ensure your PR has a clear title and description
2. Link to any relevant issues
3. Include tests for new functionality
4. Update documentation as needed
5. Ensure all existing tests still pass

## 📝 Development Setup

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/AlgoArena.git
cd AlgoArena

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_tabular.py

# Run with coverage
python -m pytest --cov=algoarena
```

## 🎯 Priority Areas

We're particularly interested in contributions in these areas:

### High Priority

- Implementation of image classification algorithms
- Text processing and NLP algorithms
- Time series forecasting methods
- Interactive dashboard improvements

### Medium Priority

- Performance optimization
- Additional evaluation metrics
- Better visualization options
- Documentation improvements

### Low Priority

- Code refactoring
- Additional datasets
- UI/UX enhancements

## 📚 Resources

### Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Coursera Course](https://www.coursera.org/learn/machine-learning)
- [Towards Data Science](https://towardsdatascience.com/)

### Technical Resources

- [Python Style Guide](https://pep8.org/)
- [Git Workflow](https://guides.github.com/introduction/flow/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🏆 Recognition

Contributors will be recognized in:

- README.md file
- Release notes
- Special contributor badge (for significant contributions)

## 📞 Getting Help

If you need help with your contribution:

- Open an issue with the "help wanted" label
- Join our discussions in the GitHub Discussions tab
- Reach out via email (if provided)

## 🎉 Types of Contributions We're Looking For

- **Bug fixes** - Help us maintain code quality
- **New algorithms** - Expand our algorithm coverage
- **Performance improvements** - Optimize existing code
- **Documentation** - Improve clarity and completeness
- **Testing** - Add comprehensive test coverage
- **Visualizations** - Create better charts and dashboards
- **Datasets** - Suggest interesting datasets to analyze

Thank you for contributing to AlgoArena! 🚀
