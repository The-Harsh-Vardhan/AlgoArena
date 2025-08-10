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
        # Import modules directly instead of using exec for better scope handling
        if module_name == "dynamic_ml_analyzer":
            import dynamic_ml_analyzer
            dynamic_ml_analyzer.main()
        elif module_name == "01_Tabular_Data":
            # Simple and reliable approach for tabular data module
            try:
                # Read and execute the file directly
                module_path = os.path.join(os.path.dirname(__file__), "01_Tabular_Data.py")
                with open(module_path, "r", encoding="utf-8") as f:
                    code = f.read()
                
                # Create a proper execution environment
                exec_globals = globals().copy()
                exec_globals.update({
                    '__name__': '__main__',
                    '__file__': module_path
                })
                
                exec(code, exec_globals)
                
            except FileNotFoundError:
                st.error(f"01_Tabular_Data module not found. Please ensure 01_Tabular_Data.py exists in the streamlit_app folder.")
                return
            except Exception as e:
                st.error(f"Error executing 01_Tabular_Data module: {str(e)}")
                return
        else:
            # For other modules, use the exec method
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
            
            exec(code)
    except FileNotFoundError:
        st.error(f"{module_name} module not found. Please ensure {module_name}.py exists in the streamlit_app folder.")
        return
    except Exception as e:
        st.error(f"Error executing {module_name} module: {str(e)}")
        return

# Set page configuration
st.set_page_config(
    page_title="AlgoArena - Dynamic Tabular ML Platform",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🏟️ AlgoArena: Dynamic Tabular ML Platform")

# Sidebar for navigation
st.sidebar.title("📊 Choose Analysis Type")
analysis_type = st.sidebar.selectbox(
    "Select the type of analysis:",
    ["🏠 Home", "🤖 Dynamic ML Analyzer", "📊 Tabular Data (Adult Dataset)"]
)

# Home page
if analysis_type == "🏠 Home":
    st.markdown("""
    ## Welcome to AlgoArena! 🎉
    
    The ultimate dynamic machine learning platform for tabular data analysis.
    
    ### 🚀 Dynamic ML Analyzer
    
    **Upload your own tabular dataset and get instant ML analysis!**
    - 📁 Support for CSV, Excel, JSON files
    - 🔍 Automatic problem type detection (classification/regression)
    - 🤖 7+ ML algorithms automatically applied
    - 📊 Interactive visualizations and comparisons
    - 📥 Downloadable results
    
    ### 🎯 Analysis Options
    
    1. **🤖 Dynamic ML Analyzer** - Upload ANY tabular dataset for instant analysis
    2. **📊 Tabular Data** - Pre-loaded Adult Income dataset analysis
    
    ### 📈 Supported Algorithms
    
    **Classification:**
    - Random Forest, Logistic Regression, SVM
    - KNN, Naive Bayes, Decision Tree, Gradient Boosting
    
    **Regression:**
    - Random Forest, Linear Regression, SVR
    - KNN, Decision Tree, Gradient Boosting
    
    ### 🎯 Key Features
    
    ✅ **Intelligent Data Preprocessing**
    ✅ **Automatic Algorithm Selection**
    ✅ **Interactive Visualizations**
    ✅ **Performance Comparisons**
    ✅ **Downloadable Results**
    ✅ **No Coding Required**
    
    **Get started by selecting an analysis type from the sidebar!** 👈
    """)
    
    # Display project statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🤖 Analysis Types",
            value="2",
            help="Dynamic and Tabular analysis"
        )
    
    with col2:
        st.metric(
            label="🧠 Algorithms",
            value="7+",
            help="Machine learning algorithms available"
        )
    
    with col3:
        st.metric(
            label="📁 File Formats",
            value="3",
            help="CSV, Excel, JSON supported"
        )
    
    with col4:
        st.metric(
            label="🎯 Problem Types",
            value="2",
            help="Classification and Regression"
        )
    
    # Feature highlights
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 Dynamic ML Analyzer
        - Upload any tabular dataset (CSV/Excel/JSON)
        - Automatic preprocessing
        - Intelligent algorithm selection
        - Real-time performance comparison
        - Download analysis results
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Tabular Data Analysis
        - Adult Income dataset analysis
        - 9 ML algorithms comparison
        - Interactive performance metrics
        - Feature importance analysis
        - Cross-validation results
        """)

# Load the appropriate module based on selection
elif analysis_type == "🤖 Dynamic ML Analyzer":
    load_module("dynamic_ml_analyzer")

elif analysis_type == "📊 Tabular Data (Adult Dataset)":
    load_module("01_Tabular_Data")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown("[GitHub Repository](https://github.com/The-Harsh-Vardhan/AlgoArena)")
st.sidebar.markdown("[Documentation](https://github.com/The-Harsh-Vardhan/AlgoArena#readme)")

# Additional info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About AlgoArena")
st.sidebar.markdown("""
**AlgoArena** is a dynamic machine learning platform specialized for tabular data analysis!

**Key Features:**
- 🤖 Upload your own tabular datasets
- 📊 Automatic ML analysis
-  Interactive visualizations
- 📥 Downloadable results

Built with:
- 🐍 Python & Streamlit
- 🧠 Scikit-learn
- 📊 Plotly & Pandas
""")

# Version info
st.sidebar.markdown("---")
st.sidebar.markdown("**Version:** 3.0.0 - Tabular Focused Edition")
st.sidebar.markdown("**Last Updated:** August 2025")
