# 🚀 Setup & Installation Guide - AlgoArena

This comprehensive guide will help you set up AlgoArena on your system and get started with machine learning analysis.

> **🌟 Want to try AlgoArena instantly?** Skip the installation and **[try our live demo](https://algo-arena.streamlit.app/)** - no setup required!

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher (Python 3.9-3.11 recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: At least 1GB free space
- **Internet**: Required for initial package installation

### Required Software

1. **Python 3.8+**: [Download from python.org](https://www.python.org/downloads/)
2. **pip**: Usually comes with Python (package manager)
3. **Git**: [Download from git-scm.com](https://git-scm.com/downloads)
4. **Web Browser**: Chrome, Firefox, Safari, or Edge

## 🛠️ Installation Methods

### Method 1: Quick Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/The-Harsh-Vardhan/AlgoArena.git
cd AlgoArena

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Launch the application
streamlit run streamlit_app/app.py
```

**That's it!** Your browser should automatically open to `http://localhost:8501`

### Method 2: Virtual Environment (Best Practice)

```bash
# 1. Clone the repository
git clone https://github.com/The-Harsh-Vardhan/AlgoArena.git
cd AlgoArena

# 2. Create virtual environment
python -m venv algoarena-env

# 3. Activate virtual environment
# Windows:
algoarena-env\Scripts\activate
# macOS/Linux:
source algoarena-env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch application
streamlit run streamlit_app/app.py
```

### Method 3: Conda Environment

```bash
# 1. Clone the repository
git clone https://github.com/The-Harsh-Vardhan/AlgoArena.git
cd AlgoArena

# 2. Create conda environment
conda create -n algoarena python=3.9

# 3. Activate environment
conda activate algoarena

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch application
streamlit run streamlit_app/app.py
```

### Method 4: Docker (Advanced)

```bash
# 1. Clone the repository
git clone https://github.com/The-Harsh-Vardhan/AlgoArena.git
cd AlgoArena

# 2. Build Docker image
docker build -t algoarena .

# 3. Run container
docker run -p 8501:8501 algoarena
```

## 🖥️ Platform-Specific Instructions

### Windows

1. **Install Python**: Download from [python.org](https://www.python.org/downloads/windows/)

   - ✅ Check "Add Python to PATH" during installation
   - ✅ Choose "Install for all users" if you have admin rights

2. **Open Command Prompt or PowerShell**

   - Press `Win + R`, type `cmd`, press Enter
   - Or search for "Command Prompt" in Start Menu

3. **Verify Python installation**:

   ```cmd
   python --version
   pip --version
   ```

4. **Follow Method 1 or 2 above**

5. **Alternative: Use the provided batch file**
   ```cmd
   # Simply double-click
   run_dashboard.bat
   ```

### macOS

1. **Install Python**: We recommend using Homebrew

   ```bash
   # Install Homebrew (if not installed)
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install Python
   brew install python
   ```

2. **Open Terminal**: Press `Cmd + Space`, type "Terminal", press Enter

3. **Verify installation**:

   ```bash
   python3 --version
   pip3 --version
   ```

4. **Follow Method 1 or 2 above** (use `python3` and `pip3` instead of `python` and `pip`)

### Linux (Ubuntu/Debian)

1. **Update package lists**:

   ```bash
   sudo apt update
   ```

2. **Install Python and pip**:

   ```bash
   sudo apt install python3 python3-pip python3-venv git
   ```

3. **Verify installation**:

   ```bash
   python3 --version
   pip3 --version
   ```

4. **Follow Method 1 or 2 above** (use `python3` and `pip3`)

### Linux (CentOS/RHEL/Fedora)

1. **Install Python and pip**:

   ```bash
   # CentOS/RHEL
   sudo yum install python3 python3-pip git

   # Fedora
   sudo dnf install python3 python3-pip git
   ```

2. **Follow Method 1 or 2 above** (use `python3` and `pip3`)

## 🔧 Dependencies Explained

### Core Dependencies

```txt
streamlit>=1.28.0          # Web framework for the dashboard
pandas>=2.0.0              # Data manipulation and analysis
numpy>=1.24.0              # Numerical computing
scikit-learn>=1.3.0        # Machine learning library
matplotlib>=3.7.0          # Basic plotting
seaborn>=0.12.0            # Statistical data visualization
plotly>=5.15.0             # Interactive visualizations
```

### Advanced ML Libraries

```txt
xgboost>=1.7.0             # Extreme Gradient Boosting
lightgbm>=4.0.0            # Light Gradient Boosting Machine
catboost>=1.2.0            # Categorical Boosting
```

### Development Tools

```txt
jupyter>=1.0.0             # Jupyter notebooks
requests>=2.31.0           # HTTP library
tqdm>=4.65.0               # Progress bars
```

## 🚨 Troubleshooting

### Common Issues

#### 1. "Command not found: python"

**Solution**:

- Windows: Reinstall Python with "Add to PATH" option
- macOS/Linux: Use `python3` instead of `python`

#### 2. "Permission denied" errors

**Solution**:

```bash
# Use --user flag
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### 3. "pip not found" or outdated pip

**Solution**:

```bash
# Update pip
python -m pip install --upgrade pip

# If pip is missing
python -m ensurepip --upgrade
```

#### 4. Package installation failures

**Solution**:

```bash
# Update setuptools and wheel
pip install --upgrade setuptools wheel

# Try installing packages individually
pip install streamlit
pip install pandas
pip install scikit-learn
```

#### 5. "Port already in use" error

**Solution**:

```bash
# Use different port
streamlit run streamlit_app/app.py --server.port 8502

# Or kill process using port 8501
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F

# macOS/Linux:
lsof -ti:8501 | xargs kill -9
```

#### 6. Slow loading or memory issues

**Solutions**:

- Close other applications to free up memory
- Use smaller datasets (< 50,000 rows for optimal performance)
- Restart the application if it becomes unresponsive

### Platform-Specific Issues

#### Windows

- **Path issues**: Use forward slashes or raw strings in file paths
- **Antivirus**: Add AlgoArena folder to antivirus exceptions
- **PowerShell execution policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### macOS

- **Xcode tools**: If compilation fails, install: `xcode-select --install`
- **Permission issues**: Use `sudo` with caution, prefer virtual environments

#### Linux

- **Missing system libraries**: Install build essentials:
  ```bash
  sudo apt install build-essential python3-dev
  ```

## 🔍 Verification Steps

After installation, verify everything works:

1. **Check Python version**:

   ```bash
   python --version
   # Should show 3.8 or higher
   ```

2. **Test package imports**:

   ```python
   python -c "import streamlit, pandas, sklearn; print('All packages imported successfully!')"
   ```

3. **Run AlgoArena**:

   ```bash
   streamlit run streamlit_app/app.py
   ```

4. **Expected output**:

   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

5. **Test the application**:
   - Browser should open automatically
   - You should see the AlgoArena homepage
   - Try uploading a CSV file to test functionality

## 🚀 Performance Optimization

### For Better Performance

1. **Use SSD storage** for faster file I/O
2. **Increase RAM** if working with large datasets
3. **Close unnecessary applications** while running AlgoArena
4. **Use virtual environments** to avoid package conflicts

### Resource Usage Guidelines

- **Small datasets** (< 1,000 rows): 2GB RAM sufficient
- **Medium datasets** (1,000-10,000 rows): 4GB RAM recommended
- **Large datasets** (10,000-100,000 rows): 8GB+ RAM recommended

## 🆘 Getting Help

### If You're Still Having Issues:

1. **Check Requirements**:

   - Python 3.8+
   - All dependencies installed
   - No proxy/firewall blocking connections

2. **Common Solutions**:

   - Restart your terminal/command prompt
   - Try virtual environment approach
   - Update all packages: `pip install --upgrade -r requirements.txt`

3. **Get Support**:
   - 📋 Create an issue on [GitHub](https://github.com/The-Harsh-Vardhan/AlgoArena/issues)
   - 📧 Email: support@algoarena.dev
   - 💬 Include error messages and system information

### Useful Commands for Debugging

```bash
# Check Python path
python -c "import sys; print(sys.executable)"

# List installed packages
pip list

# Check Streamlit version
streamlit version

# Run with verbose output
streamlit run streamlit_app/app.py --logger.level debug
```

## 🎉 Success!

Once everything is working:

- 🌟 **Star the repository** on GitHub
- 📝 **Share your feedback**
- 🤝 **Contribute** to the project
- 📊 **Start analyzing** your data!

---

<div align="center">
  <h3>🎯 Ready to start your ML journey with AlgoArena!</h3>
  <p>If you found this guide helpful, please ⭐ star the repository!</p>
</div>
