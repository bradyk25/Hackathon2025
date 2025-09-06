"""
Statistical Profiling Engine
Analyzes datasets to extract comprehensive statistical profiles for each column
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StatisticalProfiler:
    """
    Comprehensive statistical profiling for dataset columns
    """
    
    def __init__(self):
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # M/D/YY or MM/DD/YYYY
        ]
        
        self.currency_patterns = [
            r'^\$[\d,]+\.?\d*$',  # $1,234.56
            r'^\d+\.\d{2}$',      # 123.45
            r'^\$\d+$',           # $123
        ]
    
    def profile_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive profile for a single column
        """
        profile = {
            'column_name': column_name,
            'total_count': len(series),
            'null_count': series.isnull().sum(),
            'null_rate': series.isnull().sum() / len(series) if len(series) > 0 else 0,
            'non_null_count': series.notna().sum(),
            'unique_count': series.nunique(),
            'cardinality_rate': series.nunique() / len(series) if len(series) > 0 else 0,
        }
        
        # Get non-null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            profile.update({
                'data_type_hint': 'empty',
                'sample_values': [],
                'value_frequencies': {},
                'statistics': {}
            })
            return profile
        
        # Basic value analysis
        profile['sample_values'] = non_null_series.head(10).tolist()
        
        # Value frequency analysis (top 10)
        value_counts = non_null_series.value_counts().head(10)
        profile['value_frequencies'] = value_counts.to_dict()
        profile['most_common_value'] = value_counts.index[0] if len(value_counts) > 0 else None
        profile['most_common_count'] = value_counts.iloc[0] if len(value_counts) > 0 else 0
        
        # Data type analysis
        profile['data_type_hint'] = self._infer_data_type(non_null_series)
        
        # Type-specific statistics
        if profile['data_type_hint'] in ['integer', 'float', 'currency']:
            profile['statistics'] = self._numeric_statistics(non_null_series)
        elif profile['data_type_hint'] == 'date':
            profile['statistics'] = self._date_statistics(non_null_series)
        elif profile['data_type_hint'] == 'categorical':
            profile['statistics'] = self._categorical_statistics(non_null_series)
        else:  # text
            profile['statistics'] = self._text_statistics(non_null_series)
        
        # Pattern analysis
        profile['patterns'] = self._analyze_patterns(non_null_series)
        
        # Quality indicators
        profile['quality_score'] = self._calculate_quality_score(profile)
        profile['issues'] = self._identify_issues(series, profile)
        
        return profile
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """
        Infer the most likely data type for a series
        """
        # Convert to string for pattern matching
        str_series = series.astype(str)
        
        # Check for numeric types first
        numeric_count = 0
        integer_count = 0
        float_count = 0
        currency_count = 0
        
        for value in str_series:
            if self._is_numeric(value):
                numeric_count += 1
                if self._is_integer(value):
                    integer_count += 1
                elif self._is_float(value):
                    float_count += 1
            elif self._is_currency(value):
                currency_count += 1
        
        total_count = len(str_series)
        
        # If 80%+ are numeric, classify as numeric
        if numeric_count / total_count >= 0.8:
            if integer_count > float_count:
                return 'integer'
            else:
                return 'float'
        
        # Check for currency
        if currency_count / total_count >= 0.7:
            return 'currency'
        
        # Check for dates
        date_count = sum(1 for value in str_series if self._is_date(value))
        if date_count / total_count >= 0.7:
            return 'date'
        
        # Check for boolean
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        boolean_count = sum(1 for value in str_series.str.lower() 
                          if value in boolean_values)
        if boolean_count / total_count >= 0.8:
            return 'boolean'
        
        # Check if categorical (low cardinality)
        if series.nunique() / len(series) <= 0.1 and series.nunique() <= 50:
            return 'categorical'
        
        # Default to text
        return 'text'
    
    def _is_numeric(self, value: str) -> bool:
        """Check if value can be converted to a number"""
        try:
            float(value.replace(',', ''))
            return True
        except (ValueError, AttributeError):
            return False
    
    def _is_integer(self, value: str) -> bool:
        """Check if value is an integer"""
        try:
            val = float(value.replace(',', ''))
            return val.is_integer()
        except (ValueError, AttributeError):
            return False
    
    def _is_float(self, value: str) -> bool:
        """Check if value is a float"""
        try:
            val = float(value.replace(',', ''))
            return not val.is_integer()
        except (ValueError, AttributeError):
            return False
    
    def _is_currency(self, value: str) -> bool:
        """Check if value matches currency patterns"""
        for pattern in self.currency_patterns:
            if re.match(pattern, str(value)):
                return True
        return False
    
    def _is_date(self, value: str) -> bool:
        """Check if value matches date patterns"""
        for pattern in self.date_patterns:
            if re.match(pattern, str(value)):
                return True
        return False
    
    def _numeric_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for numeric columns"""
        # Convert to numeric, handling currency symbols
        numeric_series = pd.to_numeric(
            series.astype(str).str.replace(r'[$,]', '', regex=True), 
            errors='coerce'
        ).dropna()
        
        if len(numeric_series) == 0:
            return {}
        
        return {
            'min': float(numeric_series.min()),
            'max': float(numeric_series.max()),
            'mean': float(numeric_series.mean()),
            'median': float(numeric_series.median()),
            'std': float(numeric_series.std()),
            'q25': float(numeric_series.quantile(0.25)),
            'q75': float(numeric_series.quantile(0.75)),
            'range': float(numeric_series.max() - numeric_series.min()),
            'zeros_count': int((numeric_series == 0).sum()),
            'negative_count': int((numeric_series < 0).sum()),
            'positive_count': int((numeric_series > 0).sum()),
        }
    
    def _date_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for date columns"""
        # Try to parse dates
        date_series = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        date_series = date_series.dropna()
        
        if len(date_series) == 0:
            return {}
        
        return {
            'min_date': str(date_series.min()),
            'max_date': str(date_series.max()),
            'date_range_days': (date_series.max() - date_series.min()).days,
            'unique_dates': date_series.nunique(),
            'most_common_year': date_series.dt.year.mode().iloc[0] if len(date_series.dt.year.mode()) > 0 else None,
            'most_common_month': date_series.dt.month.mode().iloc[0] if len(date_series.dt.month.mode()) > 0 else None,
        }
    
    def _categorical_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for categorical columns"""
        value_counts = series.value_counts()
        
        return {
            'unique_categories': len(value_counts),
            'most_frequent_category': value_counts.index[0],
            'most_frequent_count': int(value_counts.iloc[0]),
            'least_frequent_category': value_counts.index[-1],
            'least_frequent_count': int(value_counts.iloc[-1]),
            'category_distribution': value_counts.head(10).to_dict(),
            'entropy': float(-sum((p := value_counts / len(series)) * np.log2(p + 1e-10))),
        }
    
    def _text_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for text columns"""
        str_series = series.astype(str)
        lengths = str_series.str.len()
        
        return {
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'avg_length': float(lengths.mean()),
            'median_length': float(lengths.median()),
            'empty_strings': int((str_series == '').sum()),
            'contains_numbers': int(str_series.str.contains(r'\d', na=False).sum()),
            'contains_special_chars': int(str_series.str.contains(r'[^a-zA-Z0-9\s]', na=False).sum()),
            'all_uppercase': int(str_series.str.isupper().sum()),
            'all_lowercase': int(str_series.str.islower().sum()),
        }
    
    def _analyze_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze common patterns in the data"""
        str_series = series.astype(str)
        
        patterns = {
            'email_like': int(str_series.str.contains(r'@.*\.', na=False).sum()),
            'phone_like': int(str_series.str.contains(r'\d{3}[-.]?\d{3}[-.]?\d{4}', na=False).sum()),
            'ssn_like': int(str_series.str.contains(r'\d{3}-?\d{2}-?\d{4}', na=False).sum()),
            'url_like': int(str_series.str.contains(r'https?://', na=False).sum()),
            'zip_code_like': int(str_series.str.contains(r'^\d{5}(-\d{4})?$', na=False).sum()),
            'credit_card_like': int(str_series.str.contains(r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}', na=False).sum()),
        }
        
        return patterns
    
    def _calculate_quality_score(self, profile: Dict[str, Any]) -> float:
        """Calculate a quality score for the column (0-1)"""
        score = 1.0
        
        # Penalize high null rates
        score -= profile['null_rate'] * 0.3
        
        # Penalize very low or very high cardinality
        cardinality = profile['cardinality_rate']
        if cardinality < 0.01 or cardinality > 0.95:
            score -= 0.2
        
        # Bonus for consistent data types
        if profile['data_type_hint'] in ['integer', 'float', 'date', 'boolean']:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _identify_issues(self, series: pd.Series, profile: Dict[str, Any]) -> List[str]:
        """Identify potential data quality issues"""
        issues = []
        
        # High null rate
        if profile['null_rate'] > 0.5:
            issues.append(f"High null rate: {profile['null_rate']:.1%}")
        
        # Very low cardinality
        if profile['cardinality_rate'] < 0.01 and profile['unique_count'] > 1:
            issues.append("Very low cardinality - mostly duplicate values")
        
        # Very high cardinality
        if profile['cardinality_rate'] > 0.95 and len(series) > 100:
            issues.append("Very high cardinality - mostly unique values")
        
        # Mixed data types (if we see patterns that don't match the inferred type)
        if profile['data_type_hint'] == 'text':
            patterns = profile.get('patterns', {})
            pattern_count = sum(patterns.values())
            if pattern_count > 0 and pattern_count < len(series) * 0.8:
                issues.append("Mixed data types detected")
        
        return issues
    
    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive profile for entire dataset
        """
        dataset_profile = {
            'dataset_summary': {
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'total_cells': len(df) * len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            },
            'column_profiles': {},
            'dataset_quality': {},
            'recommendations': []
        }
        
        # Profile each column
        for column in df.columns:
            dataset_profile['column_profiles'][column] = self.profile_column(df[column], column)
        
        # Dataset-level quality metrics
        null_rates = [profile['null_rate'] for profile in dataset_profile['column_profiles'].values()]
        quality_scores = [profile['quality_score'] for profile in dataset_profile['column_profiles'].values()]
        
        dataset_profile['dataset_quality'] = {
            'overall_null_rate': sum(null_rates) / len(null_rates) if null_rates else 0,
            'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'columns_with_issues': sum(1 for profile in dataset_profile['column_profiles'].values() if profile['issues']),
            'high_quality_columns': sum(1 for score in quality_scores if score > 0.8),
        }
        
        # Generate recommendations
        dataset_profile['recommendations'] = self._generate_recommendations(dataset_profile)
        
        return dataset_profile
    
    def _generate_recommendations(self, dataset_profile: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on profiling results"""
        recommendations = []
        
        # Check for columns with high null rates
        high_null_columns = [
            col for col, profile in dataset_profile['column_profiles'].items()
            if profile['null_rate'] > 0.3
        ]
        if high_null_columns:
            recommendations.append(f"Consider handling high null rates in: {', '.join(high_null_columns[:3])}")
        
        # Check for potential ID columns
        id_columns = [
            col for col, profile in dataset_profile['column_profiles'].items()
            if profile['cardinality_rate'] > 0.95 and profile['data_type_hint'] in ['integer', 'text']
        ]
        if id_columns:
            recommendations.append(f"Potential ID columns detected: {', '.join(id_columns[:3])}")
        
        # Check for categorical columns that might need encoding
        categorical_columns = [
            col for col, profile in dataset_profile['column_profiles'].items()
            if profile['data_type_hint'] == 'categorical' and profile['unique_count'] > 10
        ]
        if categorical_columns:
            recommendations.append(f"Consider encoding categorical columns: {', '.join(categorical_columns[:3])}")
        
        return recommendations


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function for dataset profiling
    """
    profiler = StatisticalProfiler()
    return profiler.profile_dataset(df)


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        from .ingestion import ingest_data
        result = ingest_data(sys.argv[1])
        profile = profile_data(result['dataframe'])
        print(f"Dataset Quality Score: {profile['dataset_quality']['avg_quality_score']:.2f}")
        print(f"Columns with Issues: {profile['dataset_quality']['columns_with_issues']}")
