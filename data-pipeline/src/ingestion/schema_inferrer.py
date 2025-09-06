"""
Schema inference for healthcare datasets.
Automatically detects data types, patterns, constraints, and relationships.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from collections import Counter
import asyncio

logger = logging.getLogger(__name__)

class SchemaInferrer:
    """
    Infers schema information from healthcare datasets.
    Detects data types, patterns, constraints, and statistical properties.
    """
    
    def __init__(self):
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        self.id_patterns = [
            r'^[A-Z]{1,3}\d{6,10}$',  # Medical record numbers
            r'^P\d{6,8}$',            # Patient IDs
            r'^[A-Z]\d{2}\.\d{1,2}$', # ICD codes
            r'^\d{10}$',              # 10-digit IDs
        ]
        
        # Healthcare-specific field types
        self.healthcare_field_types = {
            'age': 'integer',
            'weight': 'float',
            'height': 'float',
            'temperature': 'float',
            'blood_pressure': 'string',
            'diagnosis': 'categorical',
            'medication': 'categorical',
            'gender': 'categorical',
            'race': 'categorical',
            'ethnicity': 'categorical'
        }
    
    async def infer_schema(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Infer schema from healthcare data.
        
        Args:
            data: List of data records
            
        Returns:
            Schema information dictionary
        """
        
        if not data:
            return {
                "fields": {},
                "constraints": [],
                "record_count": 0,
                "quality_score": 0.0,
                "inference_timestamp": datetime.utcnow().isoformat()
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)
        
        # Infer field information
        fields = {}
        for column in df.columns:
            field_info = await self._infer_field_schema(df[column], column)
            fields[column] = field_info
        
        # Detect constraints and relationships
        constraints = await self._detect_constraints(df, fields)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(df, fields)
        
        schema = {
            "fields": fields,
            "constraints": constraints,
            "record_count": len(data),
            "quality_score": quality_score,
            "inference_timestamp": datetime.utcnow().isoformat(),
            "column_count": len(df.columns),
            "data_types_summary": self._get_type_summary(fields)
        }
        
        return schema
    
    async def _infer_field_schema(self, series: pd.Series, field_name: str) -> Dict[str, Any]:
        """Infer schema for a single field"""
        
        field_info = {
            "name": field_name,
            "nullable": series.isnull().any(),
            "null_count": series.isnull().sum(),
            "null_percentage": round((series.isnull().sum() / len(series)) * 100, 2),
            "unique_count": series.nunique(),
            "unique_percentage": round((series.nunique() / len(series)) * 100, 2)
        }
        
        # Remove null values for type inference
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            field_info.update({
                "type": "unknown",
                "description": "All values are null"
            })
            return field_info
        
        # Infer data type
        inferred_type = self._infer_data_type(clean_series, field_name)
        field_info["type"] = inferred_type
        
        # Add type-specific information
        if inferred_type == "integer":
            field_info.update(self._analyze_integer_field(clean_series))
        elif inferred_type == "float":
            field_info.update(self._analyze_float_field(clean_series))
        elif inferred_type == "categorical":
            field_info.update(self._analyze_categorical_field(clean_series))
        elif inferred_type == "datetime":
            field_info.update(self._analyze_datetime_field(clean_series))
        elif inferred_type == "string":
            field_info.update(self._analyze_string_field(clean_series, field_name))
        elif inferred_type == "boolean":
            field_info.update(self._analyze_boolean_field(clean_series))
        
        # Add healthcare-specific insights
        field_info.update(self._add_healthcare_context(field_name, clean_series, inferred_type))
        
        return field_info
    
    def _infer_data_type(self, series: pd.Series, field_name: str) -> str:
        """Infer the data type of a series"""
        
        field_lower = field_name.lower()
        
        # Check healthcare-specific field types first
        for keyword, data_type in self.healthcare_field_types.items():
            if keyword in field_lower:
                return data_type
        
        # Try to infer from pandas dtype
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        elif pd.api.types.is_float_dtype(series):
            return "float"
        elif pd.api.types.is_bool_dtype(series):
            return "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        
        # For object types, need more analysis
        if series.dtype == 'object':
            return self._infer_object_type(series, field_name)
        
        return "string"
    
    def _infer_object_type(self, series: pd.Series, field_name: str) -> str:
        """Infer type for object/string columns"""
        
        sample_values = series.astype(str).head(100)
        
        # Check for datetime patterns
        datetime_matches = 0
        for value in sample_values:
            for pattern in self.date_patterns:
                if re.search(pattern, value):
                    datetime_matches += 1
                    break
        
        if datetime_matches / len(sample_values) > 0.5:
            return "datetime"
        
        # Check for numeric values stored as strings
        numeric_count = 0
        for value in sample_values:
            try:
                float(value)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        if numeric_count / len(sample_values) > 0.8:
            # Check if they're integers
            integer_count = 0
            for value in sample_values:
                try:
                    val = float(value)
                    if val.is_integer():
                        integer_count += 1
                except (ValueError, TypeError):
                    pass
            
            if integer_count / numeric_count > 0.9:
                return "integer"
            else:
                return "float"
        
        # Check for boolean values
        boolean_values = {'true', 'false', 'yes', 'no', '1', '0', 'y', 'n'}
        unique_lower = set(str(v).lower().strip() for v in series.unique())
        if unique_lower.issubset(boolean_values) and len(unique_lower) <= 2:
            return "boolean"
        
        # Check if it's categorical (limited unique values)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.1 and series.nunique() < 50:
            return "categorical"
        
        return "string"
    
    def _analyze_integer_field(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze integer field"""
        
        # Convert to numeric if needed
        if series.dtype == 'object':
            series = pd.to_numeric(series, errors='coerce').dropna()
        
        return {
            "min": int(series.min()),
            "max": int(series.max()),
            "mean": round(series.mean(), 2),
            "median": int(series.median()),
            "std": round(series.std(), 2),
            "distribution": self._detect_distribution(series)
        }
    
    def _analyze_float_field(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze float field"""
        
        # Convert to numeric if needed
        if series.dtype == 'object':
            series = pd.to_numeric(series, errors='coerce').dropna()
        
        return {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(series.mean(), 4),
            "median": float(series.median()),
            "std": round(series.std(), 4),
            "distribution": self._detect_distribution(series),
            "decimal_places": self._detect_decimal_places(series)
        }
    
    def _analyze_categorical_field(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical field"""
        
        value_counts = series.value_counts()
        frequencies = (value_counts / len(series)).round(4)
        
        return {
            "categories": value_counts.index.tolist(),
            "frequencies": frequencies.tolist(),
            "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
            "least_common": value_counts.index[-1] if len(value_counts) > 0 else None,
            "category_count": len(value_counts)
        }
    
    def _analyze_datetime_field(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime field"""
        
        try:
            dt_series = pd.to_datetime(series, errors='coerce').dropna()
            
            if len(dt_series) == 0:
                return {"error": "Could not parse datetime values"}
            
            return {
                "min_date": dt_series.min().isoformat(),
                "max_date": dt_series.max().isoformat(),
                "date_range_days": (dt_series.max() - dt_series.min()).days,
                "format_detected": self._detect_date_format(series.iloc[0])
            }
        except Exception as e:
            return {"error": f"Datetime analysis failed: {str(e)}"}
    
    def _analyze_string_field(self, series: pd.Series, field_name: str) -> Dict[str, Any]:
        """Analyze string field"""
        
        str_series = series.astype(str)
        
        info = {
            "min_length": str_series.str.len().min(),
            "max_length": str_series.str.len().max(),
            "avg_length": round(str_series.str.len().mean(), 2),
            "contains_numbers": str_series.str.contains(r'\d').any(),
            "contains_special_chars": str_series.str.contains(r'[^a-zA-Z0-9\s]').any()
        }
        
        # Detect patterns
        pattern = self._detect_string_pattern(str_series)
        if pattern:
            info["pattern"] = pattern
        
        return info
    
    def _analyze_boolean_field(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze boolean field"""
        
        value_counts = series.value_counts()
        
        return {
            "true_count": value_counts.get(True, 0) + value_counts.get('true', 0) + value_counts.get('True', 0),
            "false_count": value_counts.get(False, 0) + value_counts.get('false', 0) + value_counts.get('False', 0),
            "true_percentage": round((value_counts.get(True, 0) / len(series)) * 100, 2)
        }
    
    def _detect_distribution(self, series: pd.Series) -> str:
        """Detect the distribution type of numeric data"""
        
        try:
            from scipy import stats
            
            # Test for normality
            _, p_value = stats.normaltest(series)
            if p_value > 0.05:
                return "normal"
            
            # Check skewness
            skewness = stats.skew(series)
            if abs(skewness) < 0.5:
                return "symmetric"
            elif skewness > 0.5:
                return "right_skewed"
            else:
                return "left_skewed"
        
        except:
            return "unknown"
    
    def _detect_decimal_places(self, series: pd.Series) -> int:
        """Detect number of decimal places in float data"""
        
        try:
            str_series = series.astype(str)
            decimal_places = []
            
            for value in str_series.head(100):
                if '.' in value:
                    decimal_places.append(len(value.split('.')[1]))
                else:
                    decimal_places.append(0)
            
            return max(decimal_places) if decimal_places else 0
        
        except:
            return 2  # Default
    
    def _detect_date_format(self, sample_value: str) -> str:
        """Detect date format from sample value"""
        
        sample_str = str(sample_value)
        
        for pattern in self.date_patterns:
            if re.search(pattern, sample_str):
                if re.search(r'\d{4}-\d{2}-\d{2}', sample_str):
                    return "YYYY-MM-DD"
                elif re.search(r'\d{2}/\d{2}/\d{4}', sample_str):
                    return "MM/DD/YYYY"
                elif re.search(r'\d{2}-\d{2}-\d{4}', sample_str):
                    return "MM-DD-YYYY"
                elif re.search(r'\d{4}/\d{2}/\d{2}', sample_str):
                    return "YYYY/MM/DD"
        
        return "unknown"
    
    def _detect_string_pattern(self, series: pd.Series) -> Optional[str]:
        """Detect common patterns in string data"""
        
        sample_values = series.head(50)
        
        # Check for ID patterns
        for pattern in self.id_patterns:
            matches = sum(1 for value in sample_values if re.match(pattern, str(value)))
            if matches / len(sample_values) > 0.8:
                return pattern
        
        return None
    
    def _add_healthcare_context(self, field_name: str, series: pd.Series, data_type: str) -> Dict[str, Any]:
        """Add healthcare-specific context to field analysis"""
        
        context = {}
        field_lower = field_name.lower()
        
        # Healthcare field categories
        if any(term in field_lower for term in ['age', 'birth']):
            context["healthcare_category"] = "demographics"
        elif any(term in field_lower for term in ['diagnosis', 'condition', 'disease']):
            context["healthcare_category"] = "clinical"
        elif any(term in field_lower for term in ['medication', 'drug', 'prescription']):
            context["healthcare_category"] = "medication"
        elif any(term in field_lower for term in ['provider', 'doctor', 'physician']):
            context["healthcare_category"] = "provider"
        elif any(term in field_lower for term in ['insurance', 'payer', 'coverage']):
            context["healthcare_category"] = "financial"
        
        # Validate healthcare-specific constraints
        if 'age' in field_lower and data_type in ['integer', 'float']:
            context["expected_range"] = {"min": 0, "max": 120}
        elif 'weight' in field_lower and data_type in ['integer', 'float']:
            context["expected_range"] = {"min": 0, "max": 1000}  # kg
        elif 'height' in field_lower and data_type in ['integer', 'float']:
            context["expected_range"] = {"min": 0, "max": 300}  # cm
        
        return context
    
    async def _detect_constraints(self, df: pd.DataFrame, fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect data constraints and relationships"""
        
        constraints = []
        
        # Unique constraints
        for column in df.columns:
            if fields[column].get("unique_percentage", 0) == 100:
                constraints.append({
                    "type": "unique",
                    "field": column,
                    "description": f"{column} has unique values"
                })
        
        # Not null constraints
        for column in df.columns:
            if not fields[column].get("nullable", True):
                constraints.append({
                    "type": "not_null",
                    "field": column,
                    "description": f"{column} cannot be null"
                })
        
        # Range constraints for numeric fields
        for column in df.columns:
            field_info = fields[column]
            if field_info["type"] in ["integer", "float"]:
                constraints.append({
                    "type": "range",
                    "field": column,
                    "min": field_info.get("min"),
                    "max": field_info.get("max"),
                    "description": f"{column} values range from {field_info.get('min')} to {field_info.get('max')}"
                })
        
        return constraints
    
    def _calculate_quality_score(self, df: pd.DataFrame, fields: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        
        scores = []
        
        # Completeness score (1 - average null percentage)
        null_percentages = [field.get("null_percentage", 0) for field in fields.values()]
        completeness = 1 - (sum(null_percentages) / len(null_percentages) / 100)
        scores.append(completeness)
        
        # Consistency score (based on type inference confidence)
        type_consistency = 1.0  # Simplified - in practice, measure type inference confidence
        scores.append(type_consistency)
        
        # Uniqueness score (appropriate level of uniqueness)
        uniqueness_scores = []
        for field in fields.values():
            unique_pct = field.get("unique_percentage", 0)
            # Ideal uniqueness depends on field type
            if field["type"] in ["integer", "float"]:
                # Numeric fields should have reasonable uniqueness
                uniqueness_scores.append(min(unique_pct / 50, 1.0))
            elif field["type"] == "categorical":
                # Categorical should have low uniqueness
                uniqueness_scores.append(max(1 - unique_pct / 100, 0.5))
            else:
                uniqueness_scores.append(0.8)  # Default
        
        uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0.8
        scores.append(uniqueness)
        
        return round(sum(scores) / len(scores), 3)
    
    def _get_type_summary(self, fields: Dict[str, Any]) -> Dict[str, int]:
        """Get summary of data types"""
        
        type_counts = Counter(field["type"] for field in fields.values())
        return dict(type_counts)
