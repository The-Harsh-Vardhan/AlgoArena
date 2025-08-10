# � AlgoArena

<div align="center">

![AlgoArena Logo](images/logo.png)
_Coming Soon - Logo will be added here_

**The Ultimate Machine Learning Playground for Data Scientists & Enthusiasts**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/The-Harsh-Vardhan/AlgoArena)
[![GitHub Stars](https://img.shields.io/github/stars/The-Harsh-Vardhan/AlgoArena?style=social)](https://github.com/The-Harsh-Vardhan/AlgoArena/stargazers)

[🚀 Quick Start](#-quick-start) • [✨ Features](#-features) • [📊 Demo](#-demo) • [🛠️ Installation](#️-installation) • [🤝 Contributing](#-contributing)

</div>

---

## 🌟 Overview

AlgoArena is a comprehensive machine learning platform that transforms the way you approach data analysis. Whether you're a seasoned data scientist or just starting your ML journey, AlgoArena provides an intuitive, powerful environment to explore, analyze, and understand your datasets.

![AlgoArena Dashboard](images/dashboard_overview.png)
_Main Dashboard Interface - Upload any dataset and get instant ML insights_

## 🚀 Dynamic ML Analyzer

**The game-changing feature**: Upload your own tabular dataset (CSV/Excel/JSON) and get instant ML analysis!

![Dynamic Analyzer Demo](images/dynamic_analyzer_demo.gif)
_Dynamic ML Analyzer in Action - From upload to insights in seconds_

### ✨ Key Features:

- 📁 **Universal File Support**: CSV, Excel, JSON
- 🔍 **Smart Problem Detection**: Automatically detects classification vs regression
- 🤖 **Intelligent Preprocessing**: Handles missing values, encoding, scaling
- 📊 **Multiple Algorithms**: 7+ ML algorithms automatically applied
- 📈 **Interactive Visualizations**: Performance comparisons and insights
- 📥 **Downloadable Results**: Export your analysis results

## 🎯 Platform Overview

AlgoArena now offers two types of tabular data analysis:

### 🤖 Dynamic ML Analyzer

**Upload any tabular dataset and get instant analysis**

- Supports CSV, Excel, JSON files
- Automatic problem type detection
- Intelligent data preprocessing
- 7+ ML algorithms automatically applied
- Interactive performance comparison
- Downloadable results

### 📊 Tabular Data Analysis

**Pre-loaded Adult Income dataset**

- 9 different ML algorithms
- Feature importance analysis
- Cross-validation results
- Performance metrics comparison

---

## � Demo

### Live Application Screenshots

<div align="center">

| Feature                        | Screenshot                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **Main Dashboard**             | ![Main Dashboard](images/main_dashboard.png)<br>_Navigate between Dynamic Analyzer and Pre-loaded Analysis_                    |
| **File Upload Interface**      | ![Upload Interface](images/upload_interface.png)<br>_Drag & drop your CSV, Excel, or JSON files_                               |
| **Algorithm Comparison**       | ![Algorithm Results](images/algorithm_comparison.png)<br>_Side-by-side performance comparison of all algorithms_               |
| **Performance Metrics**        | ![Performance Dashboard](images/performance_metrics.png)<br>_Detailed metrics including accuracy, precision, recall, F1-score_ |
| **Interactive Visualizations** | ![Interactive Charts](images/interactive_charts.png)<br>_Plotly-powered charts for deep insights_                              |
| **Results Export**             | ![Export Options](images/export_results.png)<br>_Download comprehensive analysis results as CSV_                               |

</div>

### 🎬 Demo Video

![AlgoArena Demo Video](images/demo_video.gif)
_Complete workflow: Upload → Analyze → Visualize → Export_

### 📈 Sample Analysis Results

Below are examples of the insights AlgoArena provides:

#### Algorithm Performance Comparison

![Performance Comparison Chart](images/performance_comparison_chart.png)
_Visual comparison showing which algorithms work best for your data_

#### Training Time vs Accuracy Analysis

![Time vs Accuracy](images/time_accuracy_analysis.png)
_Find the optimal balance between speed and performance_

#### Feature Importance Analysis

![Feature Importance](images/feature_importance.png)
_Understand which features drive your model's predictions_

---

## �📁 Project Structure

```
AlgoArena/
│
├── 🤖 streamlit_app/
│   ├── app.py                               # Main Streamlit application
│   ├── dynamic_ml_analyzer.py               # 🆕 Dynamic dataset analyzer
│   ├── 01_Tabular_Data.py                   # Adult Income analysis
│   └── requirements.txt                     # Streamlit dependencies
│
├── 📊 01_Tabular_Data/
│   ├── 01_Tabular_Data_algorithms.ipynb     # Complete tabular analysis
│   ├── Dataset/                             # Adult Income dataset
│   └── README.md                            # Documentation
│
├── 🛠️ utils/
│   ├── preprocessing.py                     # Data preprocessing utilities
│   └── visualization.py                    # Visualization utilities
│
└── 📋 requirements.txt                      # Project dependencies
```

## 🚀 Quick Start

### 🎯 Option 1: Dynamic ML Analyzer (Recommended)

Get instant insights from your own datasets in under 2 minutes!

```bash
# 1️⃣ Clone the repository
git clone https://github.com/The-Harsh-Vardhan/AlgoArena.git
cd AlgoArena

# 2️⃣ Install dependencies
pip install -r streamlit_app/requirements.txt

# 3️⃣ Launch the application
streamlit run streamlit_app/app.py
```

**Then simply:**

1. Select "Dynamic ML Analyzer" from the sidebar
2. Upload your CSV/Excel/JSON dataset
3. Choose your target column
4. Click "Run ML Analysis"
5. Explore results and download your analysis!

![Quick Start Guide](images/quick_start_guide.png)
_Step-by-step visual guide to get started_

### 📊 Option 2: Explore Pre-loaded Analysis

Want to see AlgoArena in action first? Try our pre-loaded Adult Income dataset analysis:

```bash
# Follow steps 1-3 above, then:
# Navigate to "Tabular Data" in the sidebar
# Explore the comprehensive Adult Income dataset analysis
```

## 🛠️ Installation

### Prerequisites

- **Python 3.8+** (Required)
- **pip** package manager
- **Git** (for cloning)

### Method 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/The-Harsh-Vardhan/AlgoArena.git
cd AlgoArena

# Create virtual environment (recommended)
python -m venv algoarena_env

# Activate virtual environment
# On Windows:
algoarena_env\Scripts\activate
# On macOS/Linux:
source algoarena_env/bin/activate

# Install dependencies
pip install -r streamlit_app/requirements.txt

# Launch application
streamlit run streamlit_app/app.py
```

### Method 2: Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/The-Harsh-Vardhan/AlgoArena.git
cd AlgoArena

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt

# Run tests (optional)
pytest

# Launch application
streamlit run streamlit_app/app.py
```

### Method 3: Using Docker (Coming Soon)

```bash
# Pull and run Docker image
docker pull harshvardhan/algoarena:latest
docker run -p 8501:8501 harshvardhan/algoarena:latest
```

### Quick Launcher (Windows Users)

For Windows users, we provide a convenient batch script:

```bash
# Double-click to run
run_dashboard.bat
```

This script automatically:

- Checks Python installation
- Installs missing dependencies
- Launches the Streamlit application
- Opens your browser automatically

## ✨ Features

## ✨ Features

### 🎯 Core Capabilities

| Feature                              | Description                               | Benefit                           |
| ------------------------------------ | ----------------------------------------- | --------------------------------- |
| **🔄 Universal Dataset Support**     | CSV, Excel, JSON formats                  | Work with any tabular data        |
| **🤖 Intelligent Problem Detection** | Auto-detects classification vs regression | No manual configuration needed    |
| **🧹 Smart Preprocessing**           | Handles missing values, encoding, scaling | Clean data automatically          |
| **⚡ Multi-Algorithm Analysis**      | 7+ algorithms tested simultaneously       | Find the best performer instantly |
| **📊 Rich Visualizations**           | Interactive Plotly charts                 | Understand results visually       |
| **💾 Export Everything**             | Download results, charts, and reports     | Share insights easily             |
| **🌐 Web Interface**                 | Browser-based, no installation needed     | Access from anywhere              |
| **🔒 Privacy-First**                 | Local processing, no data uploaded        | Your data stays secure            |

### 🧠 Advanced ML Features

#### Automated Data Preprocessing

- **Missing Value Imputation**: Smart strategies for numerical and categorical data
- **Categorical Encoding**: Automatic one-hot encoding and label encoding
- **Feature Scaling**: StandardScaler and MinMaxScaler applied when needed
- **Data Type Detection**: Automatic identification of numerical and categorical features

#### Intelligent Algorithm Selection

- **Problem Type Detection**: Automatically determines classification vs regression
- **Algorithm Optimization**: Each algorithm uses optimal parameters
- **Cross-Validation**: 5-fold CV for reliable performance estimates
- **Performance Metrics**: Comprehensive metrics for each algorithm type

#### Professional Visualizations

- **Performance Comparison Charts**: Side-by-side algorithm comparison
- **Training Time Analysis**: Efficiency vs accuracy trade-offs
- **Feature Importance**: Understand what drives predictions
- **Confusion Matrices**: Detailed classification analysis
- **Residual Plots**: Regression model diagnostics

## 🤖 Supported Algorithms

### 🎯 Classification Algorithms

| Algorithm                  | Description                 | Best For                          | Speed    |
| -------------------------- | --------------------------- | --------------------------------- | -------- |
| **Random Forest**          | Ensemble of decision trees  | High accuracy, feature importance | ⚡⚡⚡   |
| **Logistic Regression**    | Linear probability model    | Interpretable results, baseline   | ⚡⚡⚡⚡ |
| **Support Vector Machine** | Margin-based classification | High-dimensional data             | ⚡⚡     |
| **K-Nearest Neighbors**    | Instance-based learning     | Non-linear patterns, simple       | ⚡⚡⚡   |
| **Naive Bayes**            | Probabilistic classifier    | Text data, fast training          | ⚡⚡⚡⚡ |
| **Decision Tree**          | Rule-based classification   | Interpretable rules               | ⚡⚡⚡⚡ |
| **Gradient Boosting**      | Sequential ensemble         | High performance                  | ⚡⚡     |

### 📈 Regression Algorithms

| Algorithm                       | Description               | Best For                  | Speed    |
| ------------------------------- | ------------------------- | ------------------------- | -------- |
| **Random Forest Regressor**     | Ensemble regression       | Robust predictions        | ⚡⚡⚡   |
| **Linear Regression**           | Classic linear modeling   | Linear relationships      | ⚡⚡⚡⚡ |
| **Support Vector Regression**   | SVM for continuous values | Non-linear relationships  | ⚡⚡     |
| **KNN Regressor**               | Distance-based prediction | Local patterns            | ⚡⚡⚡   |
| **Decision Tree Regressor**     | Rule-based regression     | Interpretable predictions | ⚡⚡⚡⚡ |
| **Gradient Boosting Regressor** | Sequential improvement    | High accuracy             | ⚡⚡     |

## � Performance Metrics & Analysis

### 🎯 Classification Metrics

| Metric            | Description                               | Purpose                           |
| ----------------- | ----------------------------------------- | --------------------------------- |
| **Accuracy**      | Overall correctness percentage            | General performance measure       |
| **Precision**     | True positives / (True + False positives) | Minimize false alarms             |
| **Recall**        | True positives / (True + False negatives) | Catch all positive cases          |
| **F1-Score**      | Harmonic mean of precision and recall     | Balanced performance              |
| **AUC-ROC**       | Area under ROC curve                      | Binary classification performance |
| **Training Time** | Time to train the model                   | Efficiency consideration          |

![Classification Metrics Dashboard](images/classification_metrics.png)
_Comprehensive classification performance dashboard_

### 📊 Regression Metrics

| Metric            | Description                  | Purpose                  |
| ----------------- | ---------------------------- | ------------------------ |
| **R² Score**      | Coefficient of determination | Explained variance       |
| **RMSE**          | Root Mean Square Error       | Prediction accuracy      |
| **MAE**           | Mean Absolute Error          | Average prediction error |
| **MSE**           | Mean Square Error            | Squared prediction error |
| **Training Time** | Time to train the model      | Efficiency consideration |

![Regression Metrics Dashboard](images/regression_metrics.png)
_Detailed regression analysis with residual plots_

### 📊 Advanced Visualizations

#### Performance Comparison Charts

![Algorithm Performance Comparison](images/algorithm_performance.png)
_Side-by-side comparison of all algorithms with confidence intervals_

#### Training Efficiency Analysis

![Training Time vs Performance](images/efficiency_analysis.png)
_Find the optimal balance between speed and accuracy_

#### Cross-Validation Results

![Cross-Validation Scores](images/cv_results.png)
_Robust performance estimation with statistical significance_

## 🎯 Use Cases & Applications

### 👥 Perfect for:

| User Type                | Use Case                                               | Benefits                         |
| ------------------------ | ------------------------------------------------------ | -------------------------------- |
| **🔬 Data Scientists**   | Quick algorithm comparison and baseline establishment  | Save hours of manual coding      |
| **🎓 Students**          | Learning different ML algorithms and their performance | Visual understanding of concepts |
| **🔍 Researchers**       | Rapid prototyping and algorithm selection              | Fast iteration cycles            |
| **💼 Business Analysts** | Understanding ML approach suitability for tabular data | No-code analysis                 |
| **👨‍💻 Developers**        | Proof-of-concept development                           | Quick feasibility studies        |
| **📊 Data Enthusiasts**  | Exploring personal datasets                            | Instant insights                 |

### 🏢 Industry Applications

#### Business & Finance

- **Customer Churn Prediction**: Identify customers likely to leave
- **Credit Risk Assessment**: Evaluate loan default probability
- **Sales Forecasting**: Predict future revenue trends
- **Fraud Detection**: Identify suspicious transactions

#### Healthcare & Life Sciences

- **Medical Diagnosis**: Predict disease outcomes from symptoms
- **Drug Discovery**: Analyze compound effectiveness
- **Patient Risk Assessment**: Identify high-risk patients
- **Clinical Trial Analysis**: Evaluate treatment effectiveness

#### Technology & Engineering

- **Predictive Maintenance**: Forecast equipment failures
- **Quality Control**: Predict defect rates
- **Performance Optimization**: Improve system efficiency
- **Resource Planning**: Optimize capacity allocation

#### Marketing & E-commerce

- **Customer Segmentation**: Group customers by behavior
- **Price Optimization**: Find optimal pricing strategies
- **Recommendation Systems**: Suggest relevant products
- **Campaign Effectiveness**: Measure marketing ROI

![Use Cases Gallery](images/use_cases_gallery.png)
_Real-world applications across industries_

- **Business Analysts**: Understanding which ML approach works best for their tabular data
- **Anyone**: No-code ML analysis for any tabular dataset

### Example Datasets:

- Customer churn prediction
- Sales forecasting
- Medical diagnosis
- Financial risk assessment
- Marketing analytics
- Any tabular dataset!

## 📈 Performance Insights

The platform automatically provides:

- **Best performing algorithm** for your specific dataset
- **Training time vs accuracy** trade-offs
- **Algorithm suitability** recommendations
- **Performance visualizations** for easy interpretation

## 🔧 Technical Features

### Smart Preprocessing

- Automatic handling of missing values
- Intelligent categorical variable encoding
- Feature scaling and normalization
- Data type detection and conversion

### Robust Algorithm Implementation

- Cross-validation for reliable performance estimates
- Proper train/test splits
- Consistent random seeds for reproducibility
- Error handling for problematic datasets

### User-Friendly Interface

- Drag-and-drop file upload
- Real-time progress indicators
- Interactive visualizations
- Downloadable results in CSV format

## 🌐 Deployment Options

### Streamlit (Recommended for Dynamic Analysis)

```bash
streamlit run streamlit_app/app.py
```

### Hugging Face Spaces (For Sharing)

The app can be easily deployed on Hugging Face Spaces for public access.

### Local Development

Perfect for local analysis and experimentation.

## 📋 Requirements

### Core Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
```

### Advanced ML (Optional)

```
xgboost>=1.7.0
lightgbm>=4.0.0
catboost>=1.2.0
tensorflow>=2.13.0
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution:

- Additional ML algorithms
- New visualization options
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **GitHub Repository**: [AlgoArena](https://github.com/The-Harsh-Vardhan/AlgoArena)
- **Documentation**: [README](https://github.com/The-Harsh-Vardhan/AlgoArena#readme)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/The-Harsh-Vardhan/AlgoArena/issues)

## 📊 Example Usage

### 1. Upload Dataset

- Drag and drop your CSV/Excel/JSON file
- Dataset automatically validated and loaded

### 2. Select Target Column

- Choose the variable you want to predict
- System automatically detects problem type

### 3. Run Analysis

- Click "Run ML Analysis"
- Watch as multiple algorithms are trained and compared

### 4. View Results

- Interactive performance comparison charts
- Detailed metrics for each algorithm
- Best algorithm recommendation

### 5. Download Results

- Export analysis results as CSV
- Save visualizations for reports

---

## 🚀 Getting Started - Step by Step

### 📋 Step-by-Step Tutorial

#### 1️⃣ Upload Your Dataset

![Upload Step](images/step1_upload.png)

- Drag and drop your CSV, Excel, or JSON file
- Dataset is automatically validated and loaded
- Preview shows data structure and statistics

#### 2️⃣ Select Target Column

![Target Selection](images/step2_target.png)

- Choose the variable you want to predict
- System automatically detects problem type (classification/regression)
- Preview shows target variable distribution

#### 3️⃣ Run Analysis

![Analysis Running](images/step3_analysis.png)

- Click "Run ML Analysis" button
- Watch real-time progress as algorithms are trained
- Processing time varies based on dataset size

#### 4️⃣ Explore Results

![Results Dashboard](images/step4_results.png)

- Interactive performance comparison charts
- Detailed metrics for each algorithm
- Best algorithm recommendation with confidence scores

#### 5️⃣ Export & Share

![Export Options](images/step5_export.png)

- Download comprehensive analysis results as CSV
- Save visualizations as high-quality PNG images
- Generate shareable reports for stakeholders

---

## 📊 Example Analysis Results

### Sample Dataset: Customer Churn Prediction

![Sample Analysis Results](images/sample_analysis_results.png)
_Complete analysis results showing algorithm comparison, metrics, and insights_

#### Key Insights from This Analysis:

- **Best Algorithm**: Random Forest (94.2% accuracy)
- **Fastest Algorithm**: Logistic Regression (0.03s training time)
- **Most Important Features**: Monthly charges, Contract type, Tenure
- **Recommendation**: Use Random Forest for production, Logistic Regression for real-time predictions

---

## 🔧 Technical Requirements

### System Requirements

| Component   | Minimum               | Recommended   |
| ----------- | --------------------- | ------------- |
| **Python**  | 3.8+                  | 3.9+          |
| **RAM**     | 4GB                   | 8GB+          |
| **Storage** | 500MB                 | 2GB+          |
| **CPU**     | 2 cores               | 4+ cores      |
| **Browser** | Chrome/Firefox/Safari | Latest Chrome |

### Dependency Overview

```bash
# Core ML Stack
scikit-learn>=1.3.0      # Machine learning algorithms
pandas>=2.0.0            # Data manipulation
numpy>=1.24.0            # Numerical computing

# Visualization
plotly>=5.15.0           # Interactive charts
matplotlib>=3.7.0        # Static plots
seaborn>=0.12.0          # Statistical visualizations

# Web Framework
streamlit>=1.28.0        # Web application framework

# Optional Advanced ML
xgboost>=1.7.0           # Gradient boosting
lightgbm>=4.0.0          # Light gradient boosting
catboost>=1.2.0          # Categorical boosting
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🌟 Ways to Contribute

| Contribution Type         | Examples                        | Difficulty   |
| ------------------------- | ------------------------------- | ------------ |
| **🐛 Bug Reports**        | Found an issue? Report it!      | Beginner     |
| **💡 Feature Requests**   | Suggest new algorithms/features | Beginner     |
| **📝 Documentation**      | Improve docs, add examples      | Beginner     |
| **🔧 Code Contributions** | New algorithms, optimizations   | Intermediate |
| **🧪 Testing**            | Add tests, improve coverage     | Intermediate |
| **🚀 Performance**        | Optimize speed, memory usage    | Advanced     |

### 🛠️ Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/AlgoArena.git
cd AlgoArena

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Start development server
streamlit run streamlit_app/app.py
```

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📈 Project Roadmap

### 🚀 Coming Soon (v2.0)

- **🖼️ Image Classification Module**: Upload and analyze image datasets
- **📝 Text Analytics Module**: NLP and sentiment analysis
- **🧠 Deep Learning Integration**: Neural networks for complex patterns
- **☁️ Cloud Deployment**: One-click cloud hosting
- **🔄 Model Persistence**: Save and load trained models
- **📊 Advanced Visualizations**: 3D plots, interactive dashboards

### 🔮 Future Vision (v3.0+)

- **🤖 AutoML Integration**: Automated hyperparameter tuning
- **📱 Mobile App**: Analyze data on the go
- **🔌 API Access**: Programmatic access to analysis
- **👥 Collaboration Features**: Share projects with teams
- **🏢 Enterprise Features**: Advanced security, scaling

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What this means:

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ No warranty provided
- ❌ No liability assumed

---

## 🔗 Links & Resources

### 📚 Documentation

- **[Setup Guide](SETUP.md)** - Detailed installation instructions
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Technical reference for developers
- **[Project Summary](PROJECT_SUMMARY.md)** - Executive overview
- **[Changelog](CHANGELOG.md)** - Version history and updates

### � Online Presence

- **[GitHub Repository](https://github.com/The-Harsh-Vardhan/AlgoArena)** - Source code and issues
- **[Live Demo](https://algoarena.streamlit.app)** - Try it online (Coming Soon)
- **[Documentation Site](https://the-harsh-vardhan.github.io/AlgoArena)** - Full documentation (Coming Soon)

### 🤝 Community

- **[Discussions](https://github.com/The-Harsh-Vardhan/AlgoArena/discussions)** - Q&A and ideas
- **[Issues](https://github.com/The-Harsh-Vardhan/AlgoArena/issues)** - Bug reports and feature requests
- **[Pull Requests](https://github.com/The-Harsh-Vardhan/AlgoArena/pulls)** - Contribute code

---

## �🎉 Get Started Today!

Transform your data analysis workflow with AlgoArena's Dynamic ML Analyzer. Upload any dataset and get instant, professional ML analysis without writing a single line of code!

### ⚡ Quick Start Commands

```bash
# One-liner installation and launch
git clone https://github.com/The-Harsh-Vardhan/AlgoArena.git && cd AlgoArena && pip install -r streamlit_app/requirements.txt && streamlit run streamlit_app/app.py
```

### 🎯 Ready to Analyze?

1. **Upload** your dataset (CSV/Excel/JSON)
2. **Select** your target variable
3. **Click** "Run ML Analysis"
4. **Explore** comprehensive results
5. **Download** insights and visualizations

![Get Started Today](images/get_started_cta.png)
_Your ML analysis journey starts here!_

---

<div align="center">

**⭐ Star this repository if AlgoArena helped you! ⭐**

_Built with ❤️ by [Harsh Vardhan](https://github.com/The-Harsh-Vardhan)_

**Happy Analyzing!** 🚀📊🤖

</div>
