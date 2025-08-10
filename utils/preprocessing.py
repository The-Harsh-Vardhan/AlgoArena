"""
Data Preprocessing Utilities for AlgoArena

This module contains utility functions for data preprocessing,
including data cleaning, feature engineering, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for AlgoArena.
    
    Handles missing values, categorical encoding, feature scaling,
    and data validation for machine learning pipelines.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        self.preprocessing_steps = []
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning operations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()
        
        # Remove completely empty rows/columns
        df_cleaned = df_cleaned.dropna(how='all')
        df_cleaned = df_cleaned.dropna(axis=1, how='all')
        
        # Remove duplicate rows
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        
        if len(df_cleaned) < initial_rows:
            print(f"Removed {initial_rows - len(df_cleaned)} duplicate rows")
        
        # Strip whitespace from string columns
        for col in df_cleaned.select_dtypes(include=['object']).columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            # Replace common missing value indicators
            df_cleaned[col] = df_cleaned[col].replace(['?', 'null', 'NULL', 'none', 'None', ''], np.nan)
        
        self.preprocessing_steps.append("Data cleaning completed")
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values based on data type and strategy.
        
        Args:
            df: Input DataFrame
            strategy: Dictionary mapping column names to imputation strategies
            
        Returns:
            DataFrame with missing values handled
        """
        df_imputed = df.copy()
        
        if strategy is None:
            strategy = {}
        
        # Numeric columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_imputed[col].isnull().sum() > 0:
                impute_strategy = strategy.get(col, 'median')
                imputer = SimpleImputer(strategy=impute_strategy)
                df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                self.imputers[col] = imputer
        
        # Categorical columns
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_imputed[col].isnull().sum() > 0:
                impute_strategy = strategy.get(col, 'most_frequent')
                imputer = SimpleImputer(strategy=impute_strategy)
                df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                self.imputers[col] = imputer
        
        self.preprocessing_steps.append("Missing values handled")
        return df_imputed
    
    def encode_categorical_features(self, df: pd.DataFrame, encoding_strategy: str = 'auto') -> pd.DataFrame:
        """
        Encode categorical features based on strategy.
        
        Args:
            df: Input DataFrame
            encoding_strategy: 'auto', 'label', 'onehot', or 'target'
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_values = df_encoded[col].nunique()
            
            if encoding_strategy == 'auto':
                # Automatic strategy selection
                if unique_values <= 2:
                    # Binary encoding
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = ('label', le)
                elif unique_values <= 10:
                    # One-hot encoding for low cardinality
                    encoded_cols = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_cols], axis=1)
                    self.encoders[col] = ('onehot', encoded_cols.columns.tolist())
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = ('label', le)
            
            elif encoding_strategy == 'label':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = ('label', le)
            
            elif encoding_strategy == 'onehot':
                encoded_cols = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_cols], axis=1)
                self.encoders[col] = ('onehot', encoded_cols.columns.tolist())
        
        self.preprocessing_steps.append(f"Categorical encoding ({encoding_strategy}) completed")
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            method: 'standard', 'minmax', or 'robust'
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if len(numeric_cols) > 0:
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
            self.scalers['features'] = scaler
        
        self.preprocessing_steps.append(f"Feature scaling ({method}) completed")
        return df_scaled
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers in numerical features.
        
        Args:
            df: Input DataFrame
            method: 'iqr', 'zscore', or 'isolation'
            
        Returns:
            DataFrame with outlier information
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = pd.DataFrame(index=df.index)
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_info[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_info[f'{col}_outlier'] = z_scores > 3
            
            elif method == 'isolation':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_info[f'{col}_outlier'] = iso_forest.fit_predict(df[[col]]) == -1
        
        return outlier_info
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        df_features = df.copy()
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        # Create polynomial features for numeric columns (degree 2)
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    # Interaction features
                    df_features[f'{col1}_x_{col2}'] = df_features[col1] * df_features[col2]
        
        # Create ratio features
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    # Avoid division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = df_features[col1] / df_features[col2]
                        ratio = np.where(np.isfinite(ratio), ratio, 0)
                        df_features[f'{col1}_div_{col2}'] = ratio
        
        self.preprocessing_steps.append("Feature engineering completed")
        return df_features
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Union[bool, str, int]]:
        """
        Validate data quality and return report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validation report dictionary
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'is_valid': True,
            'warnings': []
        }
        
        # Check for issues
        if report['missing_percentage'] > 50:
            report['warnings'].append("High percentage of missing values (>50%)")
        
        if report['duplicate_rows'] > len(df) * 0.1:
            report['warnings'].append("High number of duplicate rows (>10%)")
        
        if len(df) < 100:
            report['warnings'].append("Small dataset size (<100 rows)")
        
        if len(df.columns) > len(df):
            report['warnings'].append("More features than samples (curse of dimensionality)")
            report['is_valid'] = False
        
        return report
    
    def get_preprocessing_summary(self) -> Dict[str, Union[List[str], int]]:
        """
        Get summary of all preprocessing steps performed.
        
        Returns:
            Summary dictionary
        """
        return {
            'steps_performed': self.preprocessing_steps,
            'total_steps': len(self.preprocessing_steps),
            'encoders_used': list(self.encoders.keys()),
            'scalers_used': list(self.scalers.keys()),
            'imputers_used': list(self.imputers.keys())
        }


def quick_preprocess(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Quick preprocessing function for immediate use.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        
    Returns:
        Tuple of (preprocessed_features, target)
    """
    preprocessor = DataPreprocessor()
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Apply preprocessing steps
    X = preprocessor.clean_data(X)
    X = preprocessor.handle_missing_values(X)
    X = preprocessor.encode_categorical_features(X)
    
    # Handle target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y


def analyze_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze data quality and return detailed report.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with quality metrics for each column
    """
    quality_report = []
    
    for col in df.columns:
        col_info = {
            'column': col,
            'dtype': str(df[col].dtype),
            'non_null_count': df[col].count(),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'unique_percentage': (df[col].nunique() / len(df)) * 100
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            })
        else:
            col_info.update({
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            })
        
        quality_report.append(col_info)
    
    return pd.DataFrame(quality_report)


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Utilities for AlgoArena")
    print("Use DataPreprocessor class for comprehensive preprocessing")
    print("Use quick_preprocess() for fast preprocessing")
    print("Use analyze_data_quality() for data quality analysis")
