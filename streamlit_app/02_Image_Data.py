import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Note: set_page_config is called in the main app.py file

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1E88E5;
    }
    .algorithm-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Fashion-MNIST data for visualization
@st.cache_data
def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return X_train, y_train, X_test, y_test, class_names

# Load results data
@st.cache_data
def load_results():
    try:
        with open('../02_Image_Data/image/image_results.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # Try alternative path
        try:
            with open('02_Image_Data/image/image_results.json', 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Image Data Analysis - Fashion-MNIST</h1>', unsafe_allow_html=True)
    
    # Load data
    X_train, y_train, X_test, y_test, class_names = load_fashion_mnist()
    results_data = load_results()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Analysis",
        ["📊 Dataset Overview", "🤖 Algorithm Comparison", "📈 Performance Analysis", "🎯 Detailed Results"]
    )
    
    if page == "📊 Dataset Overview":
        show_dataset_overview(X_train, y_train, X_test, y_test, class_names)
    elif page == "🤖 Algorithm Comparison":
        show_algorithm_comparison(results_data)
    elif page == "📈 Performance Analysis":
        show_performance_analysis(results_data)
    elif page == "🎯 Detailed Results":
        show_detailed_results(results_data, class_names)

def show_dataset_overview(X_train, y_train, X_test, y_test, class_names):
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", f"{len(X_train):,}")
    with col2:
        st.metric("Test Samples", f"{len(X_test):,}")
    with col3:
        st.metric("Image Size", "28x28")
    with col4:
        st.metric("Classes", len(class_names))
    
    st.markdown("---")
    
    # Class distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Class Distribution")
        unique, counts = np.unique(y_train, return_counts=True)
        
        fig = px.bar(
            x=[class_names[i] for i in unique], 
            y=counts,
            title="Distribution of Fashion Categories",
            color=counts,
            color_continuous_scale="viridis"
        )
        fig.update_layout(
            xaxis_title="Fashion Categories",
            yaxis_title="Number of Samples",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🖼️ Sample Images")
        
        # Display sample images
        selected_class = st.selectbox("Select a class to view samples:", class_names)
        class_idx = class_names.index(selected_class)
        
        # Find samples of selected class
        class_samples = np.where(y_train == class_idx)[0][:9]
        
        # Create a 3x3 grid of images
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        fig.suptitle(f"Sample {selected_class} Images", fontsize=16)
        
        for i, sample_idx in enumerate(class_samples):
            row, col = i // 3, i % 3
            axes[row, col].imshow(X_train[sample_idx], cmap='gray')
            axes[row, col].axis('off')
        
        st.pyplot(fig)
    
    # Dataset information
    st.markdown("---")
    st.subheader("ℹ️ Dataset Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **Fashion-MNIST Details:**
        - Created by Zalando Research
        - Drop-in replacement for MNIST
        - More challenging than digit recognition
        - Grayscale images of clothing items
        - Each image is 28x28 pixels
        - 10 fashion categories
        """)
    
    with info_col2:
        st.markdown("""
        **Fashion Categories:**
        - T-shirt/top
        - Trouser
        - Pullover
        - Dress
        - Coat
        - Sandal
        - Shirt
        - Sneaker
        - Bag
        - Ankle boot
        """)

def show_algorithm_comparison(results_data):
    st.markdown('<h2 class="sub-header">Algorithm Comparison</h2>', unsafe_allow_html=True)
    
    if results_data is None:
        st.warning("⚠️ No results data found. Please run the Image Data notebook first.")
        st.info("Run the `02_Image_Data_algorithms.ipynb` notebook to generate results!")
        return
    
    # Create DataFrame from results
    summary_df = pd.DataFrame(results_data['summary'])
    
    # Performance Overview
    st.subheader("🏆 Performance Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_accuracy = summary_df['Accuracy'].max()
        best_model = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Algorithm']
        st.metric("Best Accuracy", f"{best_accuracy:.4f}", f"({best_model})")
    
    with col2:
        avg_accuracy = summary_df['Accuracy'].mean()
        st.metric("Average Accuracy", f"{avg_accuracy:.4f}")
    
    with col3:
        fastest_time = summary_df['Training Time (s)'].min()
        fastest_model = summary_df.loc[summary_df['Training Time (s)'].idxmin(), 'Algorithm']
        st.metric("Fastest Training", f"{fastest_time:.2f}s", f"({fastest_model})")
    
    st.markdown("---")
    
    # Algorithm Performance Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Accuracy Comparison")
        fig = px.bar(
            summary_df.sort_values('Accuracy', ascending=True), 
            x='Accuracy', 
            y='Algorithm',
            orientation='h',
            color='Accuracy',
            color_continuous_scale="viridis",
            title="Model Accuracy Comparison"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Training Time Comparison")
        fig = px.bar(
            summary_df.sort_values('Training Time (s)', ascending=True), 
            x='Training Time (s)', 
            y='Algorithm',
            orientation='h',
            color='Training Time (s)',
            color_continuous_scale="plasma",
            title="Training Time Comparison"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Accuracy vs Training Time Scatter Plot
    st.subheader("🎯 Accuracy vs Training Time")
    fig = px.scatter(
        summary_df, 
        x='Training Time (s)', 
        y='Accuracy',
        text='Algorithm',
        title="Accuracy vs Training Time Trade-off",
        size_max=15
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_performance_analysis(results_data):
    st.markdown('<h2 class="sub-header">Performance Analysis</h2>', unsafe_allow_html=True)
    
    if results_data is None:
        st.warning("⚠️ No results data found. Please run the Image Data notebook first.")
        return
    
    # Algorithm selector
    available_algorithms = list(results_data['algorithms'].keys())
    selected_algorithms = st.multiselect(
        "Select algorithms to compare:",
        available_algorithms,
        default=available_algorithms[:5]
    )
    
    if not selected_algorithms:
        st.warning("Please select at least one algorithm.")
        return
    
    # Performance metrics comparison
    st.subheader("📊 Performance Metrics")
    
    metrics_data = []
    for alg in selected_algorithms:
        alg_data = results_data['algorithms'][alg]
        metrics_data.append({
            'Algorithm': alg,
            'Accuracy': alg_data['accuracy'],
            'Training Time': alg_data['training_time']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Metrics")
    st.dataframe(metrics_df, use_container_width=True)

def show_detailed_results(results_data, class_names):
    st.markdown('<h2 class="sub-header">Detailed Results</h2>', unsafe_allow_html=True)
    
    if results_data is None:
        st.warning("⚠️ No results data found. Please run the Image Data notebook first.")
        return
    
    # Algorithm selector
    available_algorithms = list(results_data['algorithms'].keys())
    selected_algorithm = st.selectbox(
        "Select an algorithm for detailed analysis:",
        available_algorithms
    )
    
    if selected_algorithm:
        alg_data = results_data['algorithms'][selected_algorithm]
        
        # Algorithm overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{alg_data['accuracy']:.4f}")
        with col2:
            st.metric("Training Time", f"{alg_data['training_time']:.2f}s")
        with col3:
            if 'cv_mean' in alg_data:
                st.metric("CV Score", f"{alg_data['cv_mean']:.4f} ± {alg_data['cv_std']:.4f}")
            elif 'loss' in alg_data:
                st.metric("Loss", f"{alg_data['loss']:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        
        cm = np.array(alg_data['confusion_matrix'])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - {selected_algorithm}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()