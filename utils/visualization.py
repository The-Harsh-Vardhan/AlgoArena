"""
Visualization Utilities for AlgoArena

This module contains utility functions for creating interactive and static
visualizations for machine learning analysis and data exploration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Union, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style defaults
plt.style.use('default')
sns.set_palette("husl")


class VisualizationEngine:
    """
    Comprehensive visualization engine for AlgoArena.
    
    Provides both static (matplotlib/seaborn) and interactive (plotly)
    visualizations for data exploration and ML analysis.
    """
    
    def __init__(self, theme: str = 'plotly'):
        """
        Initialize visualization engine.
        
        Args:
            theme: Visualization theme ('plotly', 'seaborn', 'minimal')
        """
        self.theme = theme
        self.color_palette = self._get_color_palette()
    
    def _get_color_palette(self) -> List[str]:
        """Get color palette based on theme."""
        palettes = {
            'plotly': px.colors.qualitative.Set1,
            'seaborn': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'minimal': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
        }
        return palettes.get(self.theme, palettes['plotly'])
    
    def create_data_overview(self, df: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive data overview dashboard.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Plotly figure with data overview
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Data Types Distribution', 'Missing Values', 
                          'Numeric Features Summary', 'Categorical Features'],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Data types distribution
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values,
                   name="Data Types"),
            row=1, col=1
        )
        
        # 2. Missing values
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            fig.add_trace(
                go.Bar(x=missing_data.index, y=missing_data.values,
                       name="Missing Values", marker_color='red'),
                row=1, col=2
            )
        
        # 3. Numeric features summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:5]:  # Top 5 numeric columns
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], mode='lines',
                             name=col, opacity=0.7),
                    row=2, col=1
                )
        
        # 4. Categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_unique = [df[col].nunique() for col in categorical_cols[:10]]
            fig.add_trace(
                go.Bar(x=categorical_cols[:10], y=cat_unique,
                       name="Unique Values", marker_color='green'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Data Overview Dashboard")
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame, method: str = 'pearson') -> go.Figure:
        """
        Create interactive correlation heatmap.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Plotly heatmap figure
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation analysis")
        
        corr_matrix = numeric_df.corr(method=method)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Correlation Matrix ({method.capitalize()})',
            xaxis_title='Features',
            yaxis_title='Features',
            height=600
        )
        return fig
    
    def create_feature_distribution(self, df: pd.DataFrame, feature: str, 
                                  target: Optional[str] = None) -> go.Figure:
        """
        Create feature distribution plot.
        
        Args:
            df: Input DataFrame
            feature: Feature column name
            target: Optional target column for coloring
            
        Returns:
            Plotly distribution figure
        """
        if df[feature].dtype in ['int64', 'float64']:
            # Numeric feature
            if target and target in df.columns:
                fig = px.histogram(df, x=feature, color=target, marginal="box",
                                 title=f'Distribution of {feature} by {target}')
            else:
                fig = px.histogram(df, x=feature, marginal="box",
                                 title=f'Distribution of {feature}')
        else:
            # Categorical feature
            value_counts = df[feature].value_counts().head(20)
            if target and target in df.columns:
                fig = px.histogram(df, x=feature, color=target,
                                 title=f'Distribution of {feature} by {target}')
            else:
                fig = go.Figure(data=[
                    go.Bar(x=value_counts.index, y=value_counts.values)
                ])
                fig.update_layout(title=f'Distribution of {feature}',
                                xaxis_title=feature, yaxis_title='Count')
        
        return fig
    
    def create_scatter_matrix(self, df: pd.DataFrame, features: List[str],
                            target: Optional[str] = None) -> go.Figure:
        """
        Create interactive scatter matrix.
        
        Args:
            df: Input DataFrame
            features: List of feature columns
            target: Optional target column for coloring
            
        Returns:
            Plotly scatter matrix figure
        """
        if target and target in df.columns:
            fig = px.scatter_matrix(df, dimensions=features, color=target,
                                  title="Feature Scatter Matrix")
        else:
            fig = px.scatter_matrix(df, dimensions=features,
                                  title="Feature Scatter Matrix")
        
        fig.update_traces(diagonal_visible=False)
        return fig
    
    def create_target_analysis(self, df: pd.DataFrame, target: str) -> go.Figure:
        """
        Create comprehensive target variable analysis.
        
        Args:
            df: Input DataFrame
            target: Target column name
            
        Returns:
            Plotly figure with target analysis
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Target Distribution', 'Target Statistics'],
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        if df[target].dtype in ['object', 'category']:
            # Categorical target
            value_counts = df[target].value_counts()
            
            # Distribution pie chart
            fig.add_trace(
                go.Pie(labels=value_counts.index, values=value_counts.values,
                       name="Target Distribution"),
                row=1, col=1
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values,
                       name="Count", marker_color=self.color_palette),
                row=1, col=2
            )
        else:
            # Numeric target
            fig.add_trace(
                go.Histogram(x=df[target], name="Target Distribution"),
                row=1, col=1
            )
            
            # Statistics
            stats = df[target].describe()
            fig.add_trace(
                go.Bar(x=stats.index, y=stats.values,
                       name="Statistics"),
                row=1, col=2
            )
        
        fig.update_layout(height=400, title_text=f"Target Analysis: {target}")
        return fig
    
    def create_model_comparison(self, results_df: pd.DataFrame, 
                              metric: str = 'Accuracy') -> go.Figure:
        """
        Create model performance comparison chart.
        
        Args:
            results_df: DataFrame with model results
            metric: Metric to compare
            
        Returns:
            Plotly comparison figure
        """
        fig = go.Figure()
        
        # Bar chart
        fig.add_trace(go.Bar(
            x=results_df.index,
            y=results_df[metric] if metric in results_df.columns else results_df.iloc[:, 0],
            text=np.round(results_df[metric] if metric in results_df.columns else results_df.iloc[:, 0], 3),
            textposition='auto',
            marker_color=self.color_palette[:len(results_df)]
        ))
        
        fig.update_layout(
            title=f'Model Comparison: {metric}',
            xaxis_title='Models',
            yaxis_title=metric,
            xaxis_tickangle=-45
        )
        return fig
    
    def create_performance_radar(self, results_df: pd.DataFrame, 
                               metrics: List[str]) -> go.Figure:
        """
        Create radar chart for model performance comparison.
        
        Args:
            results_df: DataFrame with model results
            metrics: List of metrics to include
            
        Returns:
            Plotly radar chart figure
        """
        fig = go.Figure()
        
        colors = self.color_palette[:len(results_df)]
        
        for i, (model, row) in enumerate(results_df.iterrows()):
            values = [row[metric] if metric in row.index else 0 for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Model Performance Comparison"
        )
        return fig
    
    def create_feature_importance(self, importance_scores: Dict[str, float], 
                                top_n: int = 15) -> go.Figure:
        """
        Create feature importance chart.
        
        Args:
            importance_scores: Dictionary of feature:importance
            top_n: Number of top features to show
            
        Returns:
            Plotly feature importance figure
        """
        # Sort and select top features
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_features)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(scores),
            y=list(features),
            orientation='h',
            marker_color=self.color_palette[0],
            text=np.round(scores, 3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, top_n * 25)
        )
        return fig
    
    def create_confusion_matrix(self, cm: np.ndarray, 
                              class_names: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array
            class_names: Optional class names
            
        Returns:
            Plotly confusion matrix figure
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Normalize for better visualization
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=500
        )
        return fig
    
    def create_learning_curve(self, train_scores: List[float], 
                            val_scores: List[float],
                            train_sizes: List[int]) -> go.Figure:
        """
        Create learning curve visualization.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores  
            train_sizes: Training set sizes
            
        Returns:
            Plotly learning curve figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_sizes, y=train_scores,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=self.color_palette[0])
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes, y=val_scores,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=self.color_palette[1])
        ))
        
        fig.update_layout(
            title='Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='Score',
            height=400
        )
        return fig


def quick_visualize(df: pd.DataFrame, plot_type: str, **kwargs) -> go.Figure:
    """
    Quick visualization function for common plots.
    
    Args:
        df: Input DataFrame
        plot_type: Type of plot ('histogram', 'scatter', 'box', 'correlation')
        **kwargs: Additional arguments for specific plots
        
    Returns:
        Plotly figure
    """
    viz = VisualizationEngine()
    
    if plot_type == 'histogram':
        feature = kwargs.get('feature')
        target = kwargs.get('target')
        return viz.create_feature_distribution(df, feature, target)
    
    elif plot_type == 'correlation':
        method = kwargs.get('method', 'pearson')
        return viz.create_correlation_heatmap(df, method)
    
    elif plot_type == 'scatter':
        x = kwargs.get('x')
        y = kwargs.get('y')
        color = kwargs.get('color')
        fig = px.scatter(df, x=x, y=y, color=color)
        return fig
    
    elif plot_type == 'box':
        x = kwargs.get('x')
        y = kwargs.get('y')
        fig = px.box(df, x=x, y=y)
        return fig
    
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def create_eda_report(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, go.Figure]:
    """
    Create comprehensive EDA report with multiple visualizations.
    
    Args:
        df: Input DataFrame
        target_column: Optional target column name
        
    Returns:
        Dictionary of visualization name -> Plotly figure
    """
    viz = VisualizationEngine()
    report = {}
    
    # Data overview
    report['data_overview'] = viz.create_data_overview(df)
    
    # Correlation heatmap (if enough numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        report['correlation_heatmap'] = viz.create_correlation_heatmap(df)
    
    # Target analysis
    if target_column and target_column in df.columns:
        report['target_analysis'] = viz.create_target_analysis(df, target_column)
    
    # Feature distributions (top 5 features)
    features_to_plot = list(df.columns[:5])
    if target_column in features_to_plot:
        features_to_plot.remove(target_column)
    
    for feature in features_to_plot:
        report[f'distribution_{feature}'] = viz.create_feature_distribution(
            df, feature, target_column
        )
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Visualization Utilities for AlgoArena")
    print("Use VisualizationEngine class for comprehensive visualizations")
    print("Use quick_visualize() for fast plotting")
    print("Use create_eda_report() for full EDA report")
