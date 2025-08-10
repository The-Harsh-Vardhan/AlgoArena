import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Note: set_page_config is handled by the main app.py file

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .feature-importance {
        background: linear-gradient(90deg, #ff7675 0%, #fd79a8 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def show_data_exploration(df, target_column):
    """Comprehensive data exploration and visualization"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 **Data Exploration & Analysis**")
    
    # Dataset overview
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("📈 Features", f"{df.shape[1]-1}")
    with col3:
        st.metric("🔢 Numeric", f"{len(df.select_dtypes(include=[np.number]).columns)}")
    with col4:
        st.metric("📝 Categorical", f"{len(df.select_dtypes(include=['object']).columns)}")
    with col5:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("❌ Missing %", f"{missing_pct:.1f}%")
    
    # Data types and missing values analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Data Types & Missing Values")
        data_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(data_info, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Target Variable Analysis")
        target_counts = df[target_column].value_counts()
        
        # Target distribution pie chart
        fig_target = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title=f"Distribution of {target_column}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
        # Target statistics
        st.write("**Target Statistics:**")
        st.write(f"- Total classes: {len(target_counts)}")
        st.write(f"- Most common: {target_counts.index[0]} ({target_counts.iloc[0]:,} samples)")
        if len(target_counts) == 2:
            balance_ratio = target_counts.min() / target_counts.max()
            st.write(f"- Class balance ratio: {balance_ratio:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_feature_analysis(df, target_column):
    """Advanced feature analysis and correlations"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("### 📊 **Feature Analysis & Correlations**")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔗 Correlation Heatmap")
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols + [target_column]].corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title('Feature Correlation Matrix')
            st.pyplot(fig)
        
        with col2:
            st.subheader("📈 Feature Distributions")
            
            # Select feature for distribution analysis
            selected_feature = st.selectbox(
                "Select feature for distribution analysis:",
                numeric_cols,
                key="feature_dist"
            )
            
            if selected_feature:
                # Distribution plot
                fig = px.histogram(
                    df, x=selected_feature, color=target_column,
                    title=f"Distribution of {selected_feature} by {target_column}",
                    marginal="box",
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance preview (using target correlation)
    if len(numeric_cols) > 0:
        st.subheader("🎯 Feature Importance Preview (Correlation-based)")
        
        # Calculate correlation with target
        if df[target_column].dtype == 'object':
            # Encode target for correlation calculation
            le = LabelEncoder()
            target_encoded = le.fit_transform(df[target_column])
            correlations = df[numeric_cols].corrwith(pd.Series(target_encoded, index=df.index))
        else:
            correlations = df[numeric_cols].corrwith(df[target_column])
        
        correlations = correlations.abs().sort_values(ascending=False)
        
        # Plot correlation-based importance
        fig = px.bar(
            x=correlations.values[:10],
            y=correlations.index[:10],
            orientation='h',
            title="Top 10 Features by Correlation with Target",
            labels={'x': 'Absolute Correlation', 'y': 'Features'},
            color=correlations.values[:10],
            color_continuous_scale="viridis"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_advanced_visualizations(df, target_column):
    """Advanced statistical visualizations"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("### 📈 **Advanced Statistical Analysis**")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔵 Scatter Plot Analysis")
            x_axis = st.selectbox("X-axis feature:", numeric_cols, key="scatter_x")
            y_axis = st.selectbox("Y-axis feature:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_y")
            
            if x_axis and y_axis and x_axis != y_axis:
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, color=target_column,
                    title=f"{x_axis} vs {y_axis}",
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Box Plot Analysis")
            selected_feature = st.selectbox(
                "Select feature for box plot:",
                numeric_cols,
                key="box_plot"
            )
            
            if selected_feature:
                fig = px.box(
                    df, x=target_column, y=selected_feature,
                    title=f"{selected_feature} Distribution by {target_column}",
                    color=target_column
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("📋 Statistical Summary")
    if len(numeric_cols) > 0:
        summary_stats = df[numeric_cols].describe()
        st.dataframe(summary_stats.round(3), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def detect_problem_type(target_series):
    """Automatically detect if it's classification or regression"""
    unique_values = target_series.nunique()
    if unique_values <= 20 and target_series.dtype in ['object', 'int64', 'category']:
        return 'classification'
    else:
        return 'regression'

def preprocess_data(df, target_column):
    """Intelligent data preprocessing"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True)).fillna(X.mode().iloc[0])
    y = y.fillna(y.mode().iloc[0] if not y.empty else 0)
    
    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    if len(categorical_columns) > 0:
        # Use LabelEncoder for binary categories, OneHot for others
        for col in categorical_columns:
            if X[col].nunique() <= 2:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            else:
                # One-hot encode with max 10 categories to avoid explosion
                if X[col].nunique() <= 10:
                    dummies = pd.get_dummies(X[col], prefix=col)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                else:
                    # For high cardinality, just label encode
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target for classification
    problem_type = detect_problem_type(y)
    if problem_type == 'classification' and y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        target_classes = le_target.classes_
    else:
        target_classes = None
    
    return X, y, problem_type, target_classes

def get_ml_algorithms(problem_type):
    """Get appropriate algorithms based on problem type"""
    if problem_type == 'classification':
        return {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
    else:  # regression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        
        return {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

def train_and_evaluate_models(X, y, problem_type):
    """Train multiple models and return comprehensive results"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    algorithms = get_ml_algorithms(problem_type)
    results = {}
    detailed_results = {}
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(algorithms.items()):
        status_text.text(f'🔄 Training {name}...')
        start_time = time.time()
        
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            training_time = time.time() - start_time
            
            if problem_type == 'classification':
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Store results
                results[name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Training Time (s)': training_time
                }
                
                # Detailed results for visualizations
                detailed_results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                # Feature importance for supported models
                if hasattr(model, 'feature_importances_'):
                    detailed_results[name]['feature_importance'] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    detailed_results[name]['feature_importance'] = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                
            else:  # regression
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'R² Score': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Training Time (s)': training_time
                }
                
                detailed_results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
            
        except Exception as e:
            st.warning(f"⚠️ Error training {name}: {str(e)}")
            results[name] = {'Error': str(e)}
        
        progress_bar.progress((i + 1) / len(algorithms))
    
    status_text.text('✅ Training completed!')
    progress_bar.empty()
    status_text.empty()
    
    return results, detailed_results, X_test, y_test, scaler


def show_enhanced_results(results_df, detailed_results, problem_type, feature_names):
    """Display comprehensive results with advanced visualizations"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("### 🏆 **Model Performance Analysis**")
    
    # Performance overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if problem_type == 'classification':
        with col1:
            best_acc = results_df['Accuracy'].max()
            best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Algorithm']
            st.metric("🎯 Best Accuracy", f"{best_acc:.3f}", f"({best_model})")
        with col2:
            avg_acc = results_df['Accuracy'].mean()
            st.metric("📊 Average Accuracy", f"{avg_acc:.3f}")
        with col3:
            best_f1 = results_df['F1-Score'].max()
            st.metric("🎪 Best F1-Score", f"{best_f1:.3f}")
        with col4:
            fastest_time = results_df['Training Time (s)'].min()
            fastest_model = results_df.loc[results_df['Training Time (s)'].idxmin(), 'Algorithm']
            st.metric("⚡ Fastest", f"{fastest_time:.2f}s", f"({fastest_model})")
    else:
        with col1:
            best_r2 = results_df['R² Score'].max()
            best_model = results_df.loc[results_df['R² Score'].idxmax(), 'Algorithm']
            st.metric("🎯 Best R²", f"{best_r2:.3f}", f"({best_model})")
        with col2:
            avg_r2 = results_df['R² Score'].mean()
            st.metric("📊 Average R²", f"{avg_r2:.3f}")
        with col3:
            best_rmse = results_df['RMSE'].min()
            st.metric("📉 Best RMSE", f"{best_rmse:.3f}")
        with col4:
            fastest_time = results_df['Training Time (s)'].min()
            fastest_model = results_df.loc[results_df['Training Time (s)'].idxmin(), 'Algorithm']
            st.metric("⚡ Fastest", f"{fastest_time:.2f}s", f"({fastest_model})")
    
    # Enhanced performance table
    st.subheader("📊 Comprehensive Performance Table")
    if problem_type == 'classification':
        styled_df = results_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen')\
                                    .highlight_min(subset=['Training Time (s)'], color='lightblue')\
                                    .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 
                                           'F1-Score': '{:.4f}', 'Training Time (s)': '{:.3f}'})
    else:
        styled_df = results_df.style.highlight_max(subset=['R² Score'], color='lightgreen')\
                                    .highlight_min(subset=['RMSE', 'MAE', 'Training Time (s)'], color='lightblue')\
                                    .format({'R² Score': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}', 
                                           'Training Time (s)': '{:.3f}'})
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Advanced visualizations
    show_performance_charts(results_df, problem_type)
    show_model_comparison_radar(results_df, problem_type)
    
    # Feature importance analysis
    if detailed_results:
        show_feature_importance_analysis(detailed_results, feature_names)
    
    # Confusion matrices for classification
    if problem_type == 'classification':
        show_confusion_matrices(detailed_results)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_performance_charts(results_df, problem_type):
    """Create comprehensive performance charts"""
    st.subheader("📈 Performance Comparison Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Main metric comparison
        if problem_type == 'classification':
            fig = px.bar(
                results_df, x='Algorithm', y='Accuracy',
                title='🎯 Model Accuracy Comparison',
                color='Accuracy',
                color_continuous_scale='viridis',
                text='Accuracy'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        else:
            fig = px.bar(
                results_df, x='Algorithm', y='R² Score',
                title='🎯 Model R² Score Comparison',
                color='R² Score',
                color_continuous_scale='viridis',
                text='R² Score'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Training time comparison
        fig = px.bar(
            results_df, x='Algorithm', y='Training Time (s)',
            title='⏱️ Training Time Comparison',
            color='Training Time (s)',
            color_continuous_scale='plasma',
            text='Training Time (s)'
        )
        fig.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance vs Speed scatter plot
    if problem_type == 'classification':
        main_metric = 'Accuracy'
    else:
        main_metric = 'R² Score'
    
    fig = px.scatter(
        results_df, x='Training Time (s)', y=main_metric,
        size=main_metric, hover_name='Algorithm',
        title=f'🔄 {main_metric} vs Training Time Trade-off',
        color=main_metric,
        color_continuous_scale='viridis'
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

def show_model_comparison_radar(results_df, problem_type):
    """Create radar chart for top models comparison"""
    st.subheader("🕸️ Top Models Radar Comparison")
    
    # Select top 3 models
    if problem_type == 'classification':
        top_models = results_df.nlargest(3, 'Accuracy')
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    else:
        top_models = results_df.nlargest(3, 'R² Score')
        metrics = ['R² Score']
        if 'RMSE' in results_df.columns:
            # Invert RMSE for radar chart (lower is better)
            top_models = top_models.copy()
            top_models['Inverted RMSE'] = 1 - (top_models['RMSE'] / top_models['RMSE'].max())
            metrics.extend(['Inverted RMSE'])
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (idx, model_row) in enumerate(top_models.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=[model_row[metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=model_row['Algorithm'],
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="🏆 Top 3 Models - Comprehensive Comparison",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_feature_importance_analysis(detailed_results, feature_names):
    """Show feature importance analysis for applicable models"""
    st.subheader("🔍 Feature Importance Analysis")
    
    # Collect models with feature importance
    importance_models = {name: results for name, results in detailed_results.items() 
                        if 'feature_importance' in results}
    
    if not importance_models:
        st.info("ℹ️ Feature importance analysis not available for the current models.")
        return
    
    # Create feature importance comparison
    model_names = list(importance_models.keys())
    selected_model = st.selectbox("Select model for feature importance:", model_names)
    
    if selected_model and 'feature_importance' in importance_models[selected_model]:
        importance_scores = importance_models[selected_model]['feature_importance']
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importance_scores)],
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        # Top 15 features
        top_features = importance_df.head(15)
        
        fig = px.bar(
            top_features, x='Importance', y='Feature',
            orientation='h',
            title=f'🎯 Top 15 Feature Importance - {selected_model}',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔝 Most Important", top_features.iloc[0]['Feature'])
        with col2:
            st.metric("💪 Top 5 Contribution", f"{top_features.head(5)['Importance'].sum():.3f}")
        with col3:
            st.metric("📊 Total Features", len(importance_df))

def show_confusion_matrices(detailed_results):
    """Display confusion matrices for classification models"""
    st.subheader("🔍 Confusion Matrix Analysis")
    
    # Select model for confusion matrix
    model_names = [name for name, results in detailed_results.items() 
                  if 'confusion_matrix' in results]
    
    if not model_names:
        st.info("ℹ️ Confusion matrix analysis not available.")
        return
    
    selected_model = st.selectbox("Select model for confusion matrix:", model_names, key="cm_model")
    
    if selected_model:
        cm = detailed_results[selected_model]['confusion_matrix']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {selected_model}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        st.pyplot(fig)
        
        # Classification report
        if 'classification_report' in detailed_results[selected_model]:
            st.subheader("📋 Detailed Classification Report")
            report = detailed_results[selected_model]['classification_report']
            
            # Convert to dataframe for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🤖 Dynamic ML Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("Upload your dataset and automatically analyze it with multiple ML algorithms!")
    
    # Sidebar for controls and navigation
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Analysis sections
    analysis_sections = [
        "📁 Data Upload",
        "🔍 Data Exploration", 
        "📊 Feature Analysis",
        "📈 Advanced Visualizations",
        "🤖 ML Training & Results"
    ]
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📁 Upload your dataset",
        type=['csv', 'xlsx', 'json'],
        help="Supported formats: CSV, Excel, JSON"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            # Load data based on file type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Sidebar navigation
            selected_section = st.sidebar.radio("Navigate to:", analysis_sections, index=1)
            
            # Data overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            
            # Show selected section
            if selected_section == "📁 Data Upload":
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data sample
                st.subheader("🔍 Data Info")
                buffer = io.StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                st.text(info_str)
            
            elif selected_section == "🔍 Data Exploration":
                # Select target column first
                st.subheader("🎯 Select Target Column")
                target_column = st.selectbox(
                    "Choose the column you want to predict:",
                    df.columns.tolist(),
                    help="This is the variable you want to predict (dependent variable)"
                )
                
                if target_column:
                    show_data_exploration(df, target_column)
            
            elif selected_section == "📊 Feature Analysis":
                target_column = st.sidebar.selectbox("Target Column:", df.columns.tolist(), key="target_fa")
                if target_column:
                    show_feature_analysis(df, target_column)
            
            elif selected_section == "📈 Advanced Visualizations":
                target_column = st.sidebar.selectbox("Target Column:", df.columns.tolist(), key="target_av")
                if target_column:
                    show_advanced_visualizations(df, target_column)
            
            elif selected_section == "🤖 ML Training & Results":
                # Select target column
                st.subheader("🎯 Machine Learning Configuration")
                target_column = st.selectbox(
                    "Choose the column you want to predict:",
                    df.columns.tolist(),
                    help="This is the variable you want to predict (dependent variable)",
                    key="target_ml"
                )
                
                if target_column:
                    # Prepare features and target
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                    
                    # Handle categorical features in X
                    categorical_cols = X.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        st.info(f"🔄 Automatically encoding {len(categorical_cols)} categorical columns...")
                        
                        # Simple label encoding for categorical features
                        for col in categorical_cols:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                    
                    # Determine problem type
                    if y.dtype == 'object' or y.nunique() <= 10:
                        problem_type = 'classification'
                        st.info("🎯 **Problem Type**: Classification")
                        
                        # Encode target if needed
                        if y.dtype == 'object':
                            le_target = LabelEncoder()
                            y = le_target.fit_transform(y)
                            
                    else:
                        problem_type = 'regression'
                        st.info("📈 **Problem Type**: Regression")
                    
                    # ML Training section
                    if st.button("🚀 Start ML Analysis", type="primary"):
                        with st.spinner("🔄 Training multiple ML models..."):
                            # Train models
                            results, detailed_results, X_test, y_test, scaler = train_and_evaluate_models(X, y, problem_type)
                            
                            # Convert results to DataFrame
                            results_df = pd.DataFrame(results).T
                            results_df = results_df.reset_index().rename(columns={'index': 'Algorithm'})
                            
                            # Sort by primary metric
                            if problem_type == 'classification':
                                results_df = results_df.sort_values('Accuracy', ascending=False)
                            else:
                                results_df = results_df.sort_values('R² Score', ascending=False)
                            
                            # Show enhanced results with all visualizations
                            show_enhanced_results(results_df, detailed_results, problem_type, X.columns.tolist())
                            
                            # Best model summary
                            best_model = results_df.iloc[0]
                            st.success(f"🏆 **Best Model**: {best_model['Algorithm']}")
                            
                            # Download results
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv_buffer.getvalue(),
                                file_name=f"ml_analysis_results_{problem_type}.csv",
                                mime="text/csv"
                            )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please make sure your file is properly formatted and try again.")
    
    else:
        # Show example datasets and instructions
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("📚 Example Datasets & Instructions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🌸 Iris Dataset**
            - **Type**: Classification
            - **Samples**: 150 rows
            - **Features**: 4 numeric
            - **Target**: 3 flower species
            - **Use case**: Multi-class classification
            """)
        
        with col2:
            st.markdown("""
            **🏠 Boston Housing**
            - **Type**: Regression  
            - **Samples**: 506 rows
            - **Features**: 13 numeric
            - **Target**: House prices
            - **Use case**: Price prediction
            """)
        
        with col3:
            st.markdown("""
            **🍷 Wine Quality**
            - **Type**: Classification
            - **Samples**: 1599 rows
            - **Features**: 11 numeric
            - **Target**: Quality rating
            - **Use case**: Quality assessment
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 🚀 **How to Use This Analyzer**
        
        1. **📁 Upload Data**: Start by uploading your CSV, Excel, or JSON file
        2. **🔍 Explore**: Use the Data Exploration section to understand your dataset
        3. **📊 Analyze Features**: Examine correlations and feature distributions  
        4. **📈 Visualize**: Create advanced statistical visualizations
        5. **🤖 Train Models**: Run multiple ML algorithms and compare performance
        6. **📥 Export**: Download results and analysis reports
        
        ### 📋 **Supported Features**
        - **Automatic preprocessing** for categorical and numeric data
        - **7+ ML algorithms** for both classification and regression
        - **Advanced visualizations** including correlation heatmaps, distribution plots
        - **Feature importance analysis** for interpretable models
        - **Comprehensive performance metrics** and comparison charts
        - **Interactive dashboard** with multiple analysis sections
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
