# �‍💻 Developer Guide - AlgoArena

**A comprehensive guide for developers who want to understand, extend, or contribute to the AlgoArena codebase.**

## 🏗️ Architecture Overview

AlgoArena follows a modular architecture with clear separation of concerns:

```
streamlit_app/
├── app.py                    # Main application entry point
├── dynamic_ml_analyzer.py    # Core ML analysis engine
├── 01_Tabular_Data.py       # Pre-loaded dataset analysis
└── 02_Image_Data.py         # Image analysis (future)
```

## 🔧 Core Components

### 1. Main Application (`app.py`)

**Purpose**: Entry point and navigation controller for the entire application.

#### Key Functions:

```python
def load_module(module_name: str) -> None
```

- **Description**: Safely loads and executes module files
- **Parameters**: `module_name` - Name of the module to load
- **Returns**: None
- **Raises**: FileNotFoundError, ImportError

```python
def main() -> None
```

- **Description**: Main application controller with sidebar navigation
- **Features**:
  - Home page with feature overview
  - Module selection and routing
  - Project statistics display

### 2. Dynamic ML Analyzer (`dynamic_ml_analyzer.py`)

**Purpose**: Core machine learning analysis engine for user-uploaded datasets.

#### Data Exploration Functions:

```python
def show_data_exploration(df: pd.DataFrame, target_column: str) -> None
```

- **Description**: Comprehensive data exploration and visualization
- **Parameters**:
  - `df`: Input DataFrame
  - `target_column`: Target variable column name
- **Features**:
  - Dataset overview metrics
  - Missing value analysis
  - Target variable distribution
  - Data quality assessment

```python
def show_feature_analysis(df: pd.DataFrame, target_column: str) -> None
```

- **Description**: Advanced feature analysis and correlations
- **Parameters**:
  - `df`: Input DataFrame
  - `target_column`: Target variable column name
- **Features**:
  - Correlation heatmaps
  - Feature distributions
  - Feature importance preview

```python
def show_advanced_visualizations(df: pd.DataFrame, target_column: str) -> None
```

- **Description**: Advanced statistical visualizations
- **Parameters**:
  - `df`: Input DataFrame
  - `target_column`: Target variable column name
- **Features**:
  - Scatter plots
  - Box plots
  - Statistical summaries

#### ML Pipeline Functions:

```python
def detect_problem_type(target_series: pd.Series) -> str
```

- **Description**: Automatically detects problem type
- **Parameters**: `target_series` - Target variable series
- **Returns**: 'classification' or 'regression'
- **Logic**:
  - Classification: ≤20 unique values and object/int/category dtype
  - Regression: >20 unique values or float dtype

```python
def preprocess_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, str, Optional[np.ndarray]]
```

- **Description**: Intelligent data preprocessing pipeline
- **Parameters**:
  - `df`: Input DataFrame
  - `target_column`: Target variable column name
- **Returns**:
  - `X`: Preprocessed features
  - `y`: Preprocessed target
  - `problem_type`: 'classification' or 'regression'
  - `target_classes`: Class labels for classification (None for regression)
- **Features**:
  - Missing value imputation
  - Categorical encoding
  - Target encoding

```python
def get_ml_algorithms(problem_type: str) -> Dict[str, Any]
```

- **Description**: Returns appropriate ML algorithms based on problem type
- **Parameters**: `problem_type` - 'classification' or 'regression'
- **Returns**: Dictionary of algorithm name -> sklearn estimator
- **Algorithms**:
  - **Classification**: Random Forest, Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, Gradient Boosting
  - **Regression**: Random Forest, Linear Regression, SVR, KNN, Decision Tree, Gradient Boosting

```python
def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, problem_type: str) -> Tuple[Dict, Dict, pd.DataFrame, pd.Series, StandardScaler]
```

- **Description**: Trains multiple models and returns comprehensive results
- **Parameters**:
  - `X`: Feature matrix
  - `y`: Target vector
  - `problem_type`: 'classification' or 'regression'
- **Returns**:
  - `results`: Performance metrics dictionary
  - `detailed_results`: Detailed model information
  - `X_test`: Test features
  - `y_test`: Test targets
  - `scaler`: Fitted StandardScaler
- **Features**:
  - Train-test split (80-20)
  - Feature scaling
  - Progress tracking
  - Error handling

#### Visualization Functions:

```python
def show_enhanced_results(results_df: pd.DataFrame, detailed_results: Dict, problem_type: str, feature_names: List[str]) -> None
```

- **Description**: Display comprehensive results with advanced visualizations
- **Features**:
  - Performance overview metrics
  - Styled performance tables
  - Advanced charts and comparisons

```python
def show_performance_charts(results_df: pd.DataFrame, problem_type: str) -> None
```

- **Description**: Create comprehensive performance charts
- **Features**:
  - Accuracy/R² comparison bars
  - Training time analysis
  - Performance vs speed trade-offs

```python
def show_model_comparison_radar(results_df: pd.DataFrame, problem_type: str) -> None
```

- **Description**: Create radar chart for top models comparison
- **Features**:
  - Top 3 models visualization
  - Multi-metric comparison
  - Interactive radar plots

```python
def show_feature_importance_analysis(detailed_results: Dict, feature_names: List[str]) -> None
```

- **Description**: Feature importance analysis for applicable models
- **Features**:
  - Top 15 features visualization
  - Feature importance statistics
  - Model-specific importance

```python
def show_confusion_matrices(detailed_results: Dict) -> None
```

- **Description**: Display confusion matrices for classification models
- **Features**:
  - Heatmap visualizations
  - Classification reports
  - Model selection

### 3. Tabular Data Analysis (`01_Tabular_Data.py`)

**Purpose**: Pre-loaded Adult Income dataset analysis and visualization.

```python
def main() -> None
```

- **Description**: Main function for tabular data analysis
- **Features**:
  - Pre-computed results loading
  - Interactive visualizations
  - Performance comparisons
  - Comprehensive metrics display

## 📊 Data Structures

### Results Dictionary Structure

#### Classification Results:

```python
{
    "Algorithm_Name": {
        "Accuracy": float,      # 0.0 to 1.0
        "Precision": float,     # 0.0 to 1.0
        "Recall": float,        # 0.0 to 1.0
        "F1-Score": float,      # 0.0 to 1.0
        "Training Time (s)": float
    }
}
```

#### Regression Results:

```python
{
    "Algorithm_Name": {
        "R² Score": float,      # Can be negative
        "RMSE": float,          # Root Mean Square Error
        "MAE": float,           # Mean Absolute Error
        "Training Time (s)": float
    }
}
```

#### Detailed Results Structure:

```python
{
    "Algorithm_Name": {
        "model": sklearn.estimator,              # Trained model
        "predictions": np.ndarray,               # Test predictions
        "probabilities": np.ndarray,             # Class probabilities (classification)
        "confusion_matrix": np.ndarray,          # Confusion matrix (classification)
        "classification_report": dict,           # Detailed metrics (classification)
        "feature_importance": np.ndarray,        # Feature importance scores
        "mse": float,                           # Mean Squared Error (regression)
        "rmse": float,                          # Root Mean Square Error (regression)
        "mae": float,                           # Mean Absolute Error (regression)
        "r2": float                             # R² Score (regression)
    }
}
```

## 🔌 Configuration Options

### Streamlit Configuration

```python
st.set_page_config(
    page_title="AlgoArena - Dynamic Tabular ML Platform",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Model Parameters

#### Random Forest:

```python
RandomForestClassifier(n_estimators=100, random_state=42)
RandomForestRegressor(n_estimators=100, random_state=42)
```

#### Logistic Regression:

```python
LogisticRegression(random_state=42, max_iter=1000)
```

#### SVM:

```python
SVC(random_state=42, probability=True)
SVR()
```

#### Gradient Boosting:

```python
GradientBoostingClassifier(random_state=42)
GradientBoostingRegressor(random_state=42)
```

### Data Preprocessing Parameters

#### Train-Test Split:

- **Test Size**: 20% (0.2)
- **Random State**: 42
- **Stratify**: True for classification

#### Feature Scaling:

- **Method**: StandardScaler
- **Applied to**: All features before training

#### Missing Value Handling:

- **Numeric**: Median imputation
- **Categorical**: Mode imputation

#### Categorical Encoding:

- **Binary categories** (≤2 unique): Label Encoding
- **Multi-class** (3-10 unique): One-Hot Encoding
- **High cardinality** (>10 unique): Label Encoding

## 🎯 API Usage Examples

### Basic Usage:

```python
import streamlit as st
import dynamic_ml_analyzer

# Load data
df = pd.read_csv("your_dataset.csv")

# Select target column
target_column = "target_variable"

# Preprocess data
X, y, problem_type, target_classes = dynamic_ml_analyzer.preprocess_data(df, target_column)

# Train models
results, detailed_results, X_test, y_test, scaler = dynamic_ml_analyzer.train_and_evaluate_models(X, y, problem_type)

# Display results
results_df = pd.DataFrame(results).T
dynamic_ml_analyzer.show_enhanced_results(results_df, detailed_results, problem_type, X.columns.tolist())
```

### Custom Model Integration:

```python
# Add custom algorithms
def get_custom_algorithms(problem_type):
    if problem_type == 'classification':
        return {
            'Custom Model': YourCustomClassifier(),
            **get_ml_algorithms(problem_type)
        }
    else:
        return {
            'Custom Model': YourCustomRegressor(),
            **get_ml_algorithms(problem_type)
        }
```

## 🚨 Error Handling

### Common Exceptions:

```python
# File format errors
FileNotFoundError: Dataset file not found
UnicodeDecodeError: File encoding issues
pd.errors.EmptyDataError: Empty dataset

# Data processing errors
ValueError: Invalid target column
KeyError: Missing required columns
TypeError: Incompatible data types

# Model training errors
sklearn.exceptions.NotFittedError: Model not trained
MemoryError: Insufficient memory for large datasets
```

### Error Recovery:

- Automatic fallback to simpler algorithms
- Graceful degradation for memory issues
- User-friendly error messages
- Logging for debugging

## 🔧 Extension Points

### Adding New Algorithms:

1. Modify `get_ml_algorithms()` function
2. Add algorithm-specific parameters
3. Update results processing logic
4. Add visualization support

### Custom Preprocessing:

1. Extend `preprocess_data()` function
2. Add new encoding strategies
3. Implement domain-specific transformations

### Additional Visualizations:

1. Create new plotting functions
2. Add to `show_enhanced_results()`
3. Implement interactive features

## 📈 Performance Considerations

### Optimization Strategies:

- **Lazy Loading**: Load heavy libraries only when needed
- **Caching**: Use `@st.cache_data` for expensive operations
- **Chunking**: Process large datasets in chunks
- **Parallel Training**: Concurrent model training where possible

### Memory Management:

- Clear unused variables
- Use generators for large datasets
- Implement data sampling for very large files

### Scalability Limits:

- **Recommended Maximum**: 100K rows, 1000 features
- **Memory Usage**: ~1GB per 50K rows
- **Training Time**: Linear with dataset size

---

<div align="center">
  <h3>📚 For more detailed information, see the source code documentation!</h3>
</div>
