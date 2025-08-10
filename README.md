# 🏟️ AlgoArena: Machine Learning Algorithm Comparison Platform

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.1+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

Welcome to **AlgoArena** - the ultimate machine learning battlefield! This comprehensive platform compares the performance of various ML algorithms across **tabular data** and **image classification** tasks.

## 🎯 Project Overview

AlgoArena provides comprehensive machine learning analysis with two specialized modules:

1. **📊 Tabular Data Analysis**: Compare algorithms on structured datasets
2. **🖼️ Image Classification**: Test deep learning and traditional ML on image data

The platform focuses on providing detailed performance comparisons, interactive visualizations, and educational insights into how different algorithms perform on real-world datasets.

---

## 📁 Project Structure

```
AlgoArena/
│
├── 📊 01_Tabular_Data/
│   ├── 01_Tabular_Data_algorithms.ipynb     # Complete tabular data analysis
│   ├── Dataset/                             # Adult Income dataset files
│   └── README.md                            # Tabular data documentation
│
├── 🖼️ 02_Image_Data/
│   ├── 02_Image_Data_Complete.ipynb         # Complete Fashion-MNIST analysis
│   ├── image/                               # Generated images and visualizations
│   ├── README.md                            # Image data documentation
│   └── QUICK_START.md                       # Quick start guide
│
├── 🌐 streamlit_app/
│   ├── app.py                               # Main Streamlit application
│   ├── 01_Tabular_Data.py                   # Tabular data dashboard
│   ├── 02_Image_Data.py                     # Image data dashboard
│   └── requirements.txt                     # Streamlit dependencies
│
├── 🛠️ utils/
│   ├── preprocessing.py                     # Data preprocessing utilities
│   └── visualization.py                    # Visualization utilities
│
├── requirements.txt                         # Project dependencies
└── README.md                               # Project documentation
```

## 🎯 Project Goals

📊 **Algorithm Comparison**: Implement and compare 15+ ML algorithms across two data types
📈 **Performance Analysis**: Detailed evaluation using multiple metrics (accuracy, precision, recall, F1-score, ROC-AUC)
🖥️ **Interactive Dashboards**: Streamlit-powered visualizations for easy result interpretation
🧠 **Learning Platform**: Hands-on experience with diverse ML techniques and datasets
🔄 **Reproducible Research**: Well-documented code and standardized evaluation procedures

## ✅ Current Progress

### 🟢 Completed: Tabular Data Analysis

- **Dataset**: UCI Adult Income Dataset
- **Problem Type**: Binary Classification (Income >50K or <=50K)
- **Features**: Demographics, education, occupation, work hours, etc.
- **Algorithms Implemented** (9 total):
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Trees
  - Random Forest
  - Naive Bayes
  - Support Vector Machine (SVM)
  - XGBoost
  - LightGBM
  - CatBoost

### 🟢 Completed: Image Data Analysis

- **Dataset**: Fashion-MNIST
- **Problem Type**: Multi-class Classification (10 fashion categories)
- **Features**: 28x28 grayscale images of clothing items
- **Algorithms Implemented** (5 total):
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Convolutional Neural Network (CNN)

## 🚀 Getting Started

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/The-Harsh-Vardhan/AlgoArena.git
   cd AlgoArena
   ```

2. **Quick launch options:**

   ```bash
   # Launch main dashboard
   run_dashboard.bat

   # Or launch image analysis specifically
   run_image_analysis.bat
   ```

3. **Manual setup:**
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app/app.py
   ```

### Prerequisites

```bash
Python 3.8+
pip package manager
```

### Installation

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Jupyter notebooks:**

   ```bash
   jupyter notebook 01_Tabular_Data/01_Tabular_Data_algorithms.ipynb
   ```

3. **Launch Streamlit dashboard:**
   ```bash
   streamlit run streamlit_app/app.py
   ```

## 📊 Key Features

### Algorithm Coverage

**Tabular Data:**

- **Traditional ML**: Logistic Regression, SVM, Decision Trees, Random Forest
- **Ensemble Methods**: XGBoost, LightGBM, CatBoost
- **Instance-based**: K-Nearest Neighbors
- **Probabilistic**: Naive Bayes

**Image Data:**

- **Deep Learning**: Convolutional Neural Networks (CNN)
- **Traditional ML**: Random Forest, SVM, KNN, Logistic Regression
- **Feature Extraction**: HOG, LBP, statistical features

### Evaluation Metrics

- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ ROC-AUC and PR-AUC curves
- ✅ Confusion Matrix with heatmaps
- ✅ Feature importance analysis
- ✅ Cross-validation scores
- ✅ Training time comparison

### Visualization Dashboard

- Interactive performance comparisons
- Algorithm ranking by different metrics
- Feature importance plots
- ROC and PR curve overlays
- Image classification visualizations

## 🛠️ Technologies Used

### Core ML Stack:

- **Python 3.8+** - Primary programming language
- **Scikit-learn** - Traditional ML algorithms
- **XGBoost, LightGBM, CatBoost** - Advanced gradient boosting
- **TensorFlow/Keras** - Deep learning models
- **Pandas & NumPy** - Data manipulation and analysis

### Visualization & UI:

- **Streamlit** - Interactive web dashboards
- **Matplotlib & Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts

### Development Tools:

- **Jupyter Notebook** - Exploratory data analysis
- **Git/GitHub** - Version control
- **VS Code** - Development environment

## 📈 Results Preview

### Tabular Data Performance (Adult Income Dataset):

| Algorithm           | Accuracy | F1-Score | AUC-ROC | Training Time |
| ------------------- | -------- | -------- | ------- | ------------- |
| XGBoost             | 0.874    | 0.721    | 0.923   | 2.3s          |
| Random Forest       | 0.863    | 0.695    | 0.914   | 1.8s          |
| LightGBM            | 0.871    | 0.718    | 0.920   | 1.2s          |
| CatBoost            | 0.869    | 0.715    | 0.918   | 3.1s          |
| Logistic Regression | 0.848    | 0.663    | 0.895   | 0.5s          |

### Image Data Performance (Fashion-MNIST Dataset):

| Algorithm           | Accuracy | Training Time | Model Type     |
| ------------------- | -------- | ------------- | -------------- |
| CNN                 | ~0.920   | ~180s         | Deep Learning  |
| Random Forest       | ~0.880   | ~15s          | Traditional ML |
| Logistic Regression | ~0.840   | ~8s           | Traditional ML |
| KNN                 | ~0.850   | ~2s           | Traditional ML |
| SVM                 | ~0.870   | ~45s          | Traditional ML |

_Full results and analysis available in the notebooks and dashboard_

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Ideas for Contributions:

- Implement new algorithms
- Add new datasets
- Improve visualizations
- Optimize performance
- Add more evaluation metrics
- Enhance documentation

## 📚 Learning Resources

This project serves as a practical learning platform. Each module includes:

- **Theory explanations** of algorithms
- **Step-by-step implementation** with comments
- **Performance analysis** and interpretation
- **Best practices** for each data type
- **Common pitfalls** and how to avoid them

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** - For providing excellent datasets
- **Fashion-MNIST creators** - For the image classification dataset
- **Scikit-learn community** - For comprehensive ML algorithms
- **Streamlit team** - For the amazing web app framework
- **Open source ML community** - For continuous innovation

## 📞 Contact

- **GitHub**: [@The-Harsh-Vardhan](https://github.com/The-Harsh-Vardhan)
- **Project Link**: [https://github.com/The-Harsh-Vardhan/AlgoArena](https://github.com/The-Harsh-Vardhan/AlgoArena)

---

**⭐ If you found AlgoArena helpful, please consider giving it a star on GitHub!**

_Happy Learning and May the Best Algorithm Win! 🏆_
