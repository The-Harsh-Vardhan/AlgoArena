import streamlit as st
import sys
import os
import importlib.util

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Function to safely load and execute module files
def load_module(module_name):
    """Safely load and execute a module file with proper encoding handling."""
    try:
        # Try UTF-8 first
        with open(f"{module_name}.py", "r", encoding="utf-8") as f:
            code = f.read()
    except UnicodeDecodeError:
        try:
            # Fallback to latin-1
            with open(f"{module_name}.py", "r", encoding="latin-1") as f:
                code = f.read()
        except UnicodeDecodeError:
            # Final fallback to cp1252 (Windows default)
            with open(f"{module_name}.py", "r", encoding="cp1252", errors="ignore") as f:
                code = f.read()
    except FileNotFoundError:
        st.error(f"{module_name} module not found. Please ensure {module_name}.py exists in the streamlit_app folder.")
        return
    
    try:
        exec(code)
    except Exception as e:
        st.error(f"Error executing {module_name} module: {str(e)}")

# Set page configuration
st.set_page_config(
    page_title="AlgoArena - ML Algorithm Comparison",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🏟️ AlgoArena: Machine Learning Algorithm Comparison Platform")

# Sidebar for navigation
st.sidebar.title("📊 Choose Data Type")
data_type = st.sidebar.selectbox(
    "Select the type of data analysis:",
    ["🏠 Home", "📊 Tabular Data", "🖼️ Image Data"]
)

# Home page
if data_type == "🏠 Home":
    st.markdown("""
    ## Welcome to AlgoArena! 🎉
    
    The ultimate machine learning battlefield where algorithms compete across different data types.
    
    ### 🎯 What can you do here?
    
    - **📊 Tabular Data Analysis**: Compare ML algorithms on structured datasets
    - **🖼️ Image Classification**: Test image recognition algorithms on Fashion-MNIST
    
    ### 🚀 Quick Start
    
    1. Choose a data type from the sidebar
    2. Explore the pre-loaded datasets and analysis
    3. Compare algorithm performance
    4. Learn from detailed visualizations
    
    ### 📈 Featured Algorithms
    
    **Tabular Data:**
    - Logistic Regression, Random Forest, XGBoost, LightGBM
    - SVM, Decision Trees, Naive Bayes, KNN
    - CatBoost and more
    
    **Image Data:**
    - Convolutional Neural Networks (CNN)
    - Traditional ML: Random Forest, SVM, KNN
    - Logistic Regression with feature extraction
    
    Select a data type from the sidebar to begin! 👈
    """)
    
    # Display project statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Data Types",
            value="2",
            help="Tabular and Image data analysis"
        )
    
    with col2:
        st.metric(
            label="🤖 Algorithms",
            value="15+",
            help="Machine learning algorithms implemented"
        )
    
    with col3:
        st.metric(
            label="📈 Metrics",
            value="20+",
            help="Performance metrics tracked"
        )
    
    with col4:
        st.metric(
            label="🎯 Accuracy",
            value="90%+",
            help="Best achieved accuracy"
        )
    
    # Feature highlights
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Tabular Data Analysis
        - Adult Income dataset analysis
        - 9 ML algorithms comparison
        - Interactive performance metrics
        - Feature importance analysis
        - Cross-validation results
        """)
    
    with col2:
        st.markdown("""
        ### 🖼️ Image Classification
        - Fashion-MNIST dataset
        - CNN and traditional ML
        - Image preprocessing pipeline
        - Visual performance comparison
        - Model accuracy analysis
        """)

# Load the appropriate module based on selection
elif data_type == "📊 Tabular Data":
    load_module("01_Tabular_Data")

elif data_type == "🖼️ Image Data":
    load_module("02_Image_Data")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown("[GitHub Repository](https://github.com/The-Harsh-Vardhan/AlgoArena)")
st.sidebar.markdown("[Documentation](https://github.com/The-Harsh-Vardhan/AlgoArena#readme)")

# Additional info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
**AlgoArena** is a comprehensive machine learning platform for comparing algorithm performance across different data types.

**Focus Areas:**
- 📊 Tabular Data Analysis
- 🖼️ Image Classification

Built with:
- 🐍 Python
- 📊 Streamlit
- 🧠 Scikit-learn
- 📈 Plotly
- 🤖 TensorFlow/Keras
""")

# Version info
st.sidebar.markdown("---")
st.sidebar.markdown("**Version:** 2.0.0")
st.sidebar.markdown("**Last Updated:** August 2025")
