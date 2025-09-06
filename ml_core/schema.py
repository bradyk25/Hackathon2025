"""
Schema Inference Core
Intelligent type inference, constraint discovery, and schema generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import re
from datetime import datetime
import json
from .profiling import StatisticalProfiler


class SchemaInferrer:
    """
    Advanced schema inference with constraint discovery and validation
    """
    
    def __init__(self):
        self.profiler = StatisticalProfiler()
        
        # Medical/Healthcare specific patterns
        self.medical_patterns = {
            'icd_10': r'^[A-Z]\d{2}(\.\d{1,3})?$',  # ICD-10 codes
            'cpt_code': r'^\d{5}$',  # CPT codes
            'npi': r'^\d{10}$',  # National Provider Identifier
            'mrn': r'^[A-Z0-9]{6,12}$',  # Medical Record Number (varies)
        }
        
        # Common constraint patterns
        self.constraint_patterns = {
            'age': {'min': 0, 'max': 150, 'type': 'integer'},
            'percentage': {'min': 0, 'max': 100, 'type': 'float'},
            'year': {'min': 1900, 'max': 2030, 'type': 'integer'},
            'month': {'min': 1, 'max': 12, 'type': 'integer'},
            'day': {'min': 1, 'max': 31, 'type': 'integer'},
        }
    
    def infer_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive schema with types, constraints, and relationships
        """
        # First get statistical profiles
        dataset_profile = self.profiler.profile_dataset(df)
        
        schema = {
            'table_name': 'inferred_table',
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': {},
            'constraints': {},
            'relationships': {},
            'quality_assessment': dataset_profile['dataset_quality'],
            'recommendations': dataset_profile['recommendations'],
            'metadata': {
                'inference_timestamp': datetime.now().isoformat(),
                'inference_version': '1.0'
            }
        }
        
        # Infer schema for each column
        for column_name in df.columns:
            column_profile = dataset_profile['column_profiles'][column_name]
            schema['columns'][column_name] = self._infer_column_schema(
                df[column_name], column_profile, column_name
            )
        
        # Discover cross-column constraints and relationships
        schema['constraints'] = self._discover_constraints(df, schema['columns'])
        schema['relationships'] = self._discover_relationships(df, schema['columns'])
        
        # Add domain-specific insights
        schema['domain_insights'] = self._analyze_domain_patterns(df, schema['columns'])
        
        return schema
    
    def _infer_column_schema(self, series: pd.Series, profile: Dict[str, Any], column_name: str) -> Dict[str, Any]:
        """
        Infer detailed schema for a single column
        """
        column_schema = {
            'name': column_name,
            'inferred_type': self._map_to_sql_type(profile['data_type_hint'], profile),
            'python_type': profile['data_type_hint'],
            'nullable': profile['null_rate'] > 0,
            'null_rate': profile['null_rate'],
            'unique_count': profile['unique_count'],
            'cardinality_rate': profile['cardinality_rate'],
            'quality_score': profile['quality_score'],
            'issues': profile['issues']
        }
        
        # Add type-specific constraints
        if profile['data_type_hint'] in ['integer', 'float', 'currency']:
            column_schema.update(self._infer_numeric_constraints(profile))
        elif profile['data_type_hint'] == 'text':
            column_schema.update(self._infer_text_constraints(series, profile))
        elif profile['data_type_hint'] == 'categorical':
            column_schema.update(self._infer_categorical_constraints(profile))
        elif profile['data_type_hint'] == 'date':
            column_schema.update(self._infer_date_constraints(profile))
        elif profile['data_type_hint'] == 'boolean':
            column_schema.update(self._infer_boolean_constraints(series))
        
        # Check for special patterns
        column_schema['special_patterns'] = self._detect_special_patterns(series, column_name)
        
        # Suggest primary key candidacy
        column_schema['primary_key_candidate'] = self._assess_primary_key_candidacy(profile)
        
        # Suggest foreign key candidacy
        column_schema['foreign_key_candidate'] = self._assess_foreign_key_candidacy(profile, column_name)
        
        return column_schema
    
    def _map_to_sql_type(self, python_type: str, profile: Dict[str, Any]) -> str:
        """
        Map Python types to SQL types with appropriate sizing
        """
        type_mapping = {
            'integer': self._infer_integer_sql_type(profile),
            'float': 'DECIMAL(10,2)',  # Default, can be refined
            'currency': 'DECIMAL(10,2)',
            'text': self._infer_text_sql_type(profile),
            'categorical': self._infer_categorical_sql_type(profile),
            'date': 'DATE',
            'boolean': 'BOOLEAN',
            'empty': 'TEXT'
        }
        
        return type_mapping.get(python_type, 'TEXT')
    
    def _infer_integer_sql_type(self, profile: Dict[str, Any]) -> str:
        """Infer appropriate integer SQL type based on range"""
        stats = profile.get('statistics', {})
        if not stats:
            return 'INTEGER'
        
        min_val = stats.get('min', 0)
        max_val = stats.get('max', 0)
        
        # Choose appropriate integer type based on range
        if min_val >= -128 and max_val <= 127:
            return 'TINYINT'
        elif min_val >= -32768 and max_val <= 32767:
            return 'SMALLINT'
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return 'INTEGER'
        else:
            return 'BIGINT'
    
    def _infer_text_sql_type(self, profile: Dict[str, Any]) -> str:
        """Infer appropriate text SQL type based on length"""
        stats = profile.get('statistics', {})
        if not stats:
            return 'TEXT'
        
        max_length = stats.get('max_length', 0)
        
        if max_length <= 50:
            return f'VARCHAR({max(max_length, 50)})'
        elif max_length <= 255:
            return f'VARCHAR({max_length})'
        elif max_length <= 65535:
            return 'TEXT'
        else:
            return 'LONGTEXT'
    
    def _infer_categorical_sql_type(self, profile: Dict[str, Any]) -> str:
        """Infer SQL type for categorical data"""
        unique_count = profile.get('unique_count', 0)
        
        if unique_count <= 10:
            # Small enum, could use ENUM type
            values = list(profile.get('value_frequencies', {}).keys())
            if all(len(str(v)) <= 50 for v in values):
                return f"ENUM({', '.join(repr(str(v)) for v in values[:10])})"
        
        # Fallback to VARCHAR
        stats = profile.get('statistics', {})
        max_length = stats.get('max_length', 50) if stats else 50
        return f'VARCHAR({max(max_length, 50)})'
    
    def _infer_numeric_constraints(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Infer constraints for numeric columns"""
        stats = profile.get('statistics', {})
        if not stats:
            return {}
        
        constraints = {
            'min_value': stats.get('min'),
            'max_value': stats.get('max'),
            'mean': stats.get('mean'),
            'std_dev': stats.get('std'),
            'has_negatives': stats.get('negative_count', 0) > 0,
            'has_zeros': stats.get('zeros_count', 0) > 0,
        }
        
        # Check for common numeric patterns
        if constraints['min_value'] is not None and constraints['max_value'] is not None:
            min_val, max_val = constraints['min_value'], constraints['max_value']
            
            # Age-like pattern
            if 0 <= min_val <= 150 and 0 <= max_val <= 150:
                constraints['likely_age'] = True
            
            # Percentage-like pattern
            if 0 <= min_val <= 100 and 0 <= max_val <= 100:
                constraints['likely_percentage'] = True
            
            # Year-like pattern
            if 1900 <= min_val <= 2030 and 1900 <= max_val <= 2030:
                constraints['likely_year'] = True
        
        return constraints
    
    def _infer_text_constraints(self, series: pd.Series, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Infer constraints for text columns"""
        stats = profile.get('statistics', {})
        patterns = profile.get('patterns', {})
        
        constraints = {
            'min_length': stats.get('min_length', 0),
            'max_length': stats.get('max_length', 0),
            'avg_length': stats.get('avg_length', 0),
            'has_numbers': stats.get('contains_numbers', 0) > 0,
            'has_special_chars': stats.get('contains_special_chars', 0) > 0,
        }
        
        # Check for format patterns
        if patterns.get('email_like', 0) > len(series) * 0.8:
            constraints['format'] = 'email'
        elif patterns.get('phone_like', 0) > len(series) * 0.8:
            constraints['format'] = 'phone'
        elif patterns.get('ssn_like', 0) > len(series) * 0.8:
            constraints['format'] = 'ssn'
        elif patterns.get('url_like', 0) > len(series) * 0.8:
            constraints['format'] = 'url'
        elif patterns.get('zip_code_like', 0) > len(series) * 0.8:
            constraints['format'] = 'zip_code'
        
        # Check for consistent length (fixed-width fields)
        if constraints['min_length'] == constraints['max_length'] and constraints['min_length'] > 0:
            constraints['fixed_length'] = True
            constraints['length'] = constraints['min_length']
        
        return constraints
    
    def _infer_categorical_constraints(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Infer constraints for categorical columns"""
        return {
            'categories': list(profile.get('value_frequencies', {}).keys()),
            'category_count': profile.get('unique_count', 0),
            'most_frequent': profile.get('most_common_value'),
            'entropy': profile.get('statistics', {}).get('entropy', 0),
            'is_ordinal': self._detect_ordinal_pattern(profile)
        }
    
    def _infer_date_constraints(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Infer constraints for date columns"""
        stats = profile.get('statistics', {})
        if not stats:
            return {}
        
        return {
            'min_date': stats.get('min_date'),
            'max_date': stats.get('max_date'),
            'date_range_days': stats.get('date_range_days', 0),
            'unique_dates': stats.get('unique_dates', 0),
            'most_common_year': stats.get('most_common_year'),
            'most_common_month': stats.get('most_common_month'),
        }
    
    def _infer_boolean_constraints(self, series: pd.Series) -> Dict[str, Any]:
        """Infer constraints for boolean columns"""
        value_counts = series.value_counts()
        
        return {
            'true_count': int(value_counts.get(True, 0) + value_counts.get('true', 0) + 
                            value_counts.get('True', 0) + value_counts.get('1', 0) + 
                            value_counts.get(1, 0)),
            'false_count': int(value_counts.get(False, 0) + value_counts.get('false', 0) + 
                             value_counts.get('False', 0) + value_counts.get('0', 0) + 
                             value_counts.get(0, 0)),
            'representation': self._detect_boolean_representation(series)
        }
    
    def _detect_ordinal_pattern(self, profile: Dict[str, Any]) -> bool:
        """Detect if categorical data has ordinal properties"""
        categories = list(profile.get('value_frequencies', {}).keys())
        
        # Check for common ordinal patterns
        ordinal_patterns = [
            ['low', 'medium', 'high'],
            ['small', 'medium', 'large'],
            ['poor', 'fair', 'good', 'excellent'],
            ['never', 'rarely', 'sometimes', 'often', 'always'],
            ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree']
        ]
        
        categories_lower = [str(c).lower() for c in categories]
        
        for pattern in ordinal_patterns:
            if all(item in categories_lower for item in pattern):
                return True
        
        # Check for numeric-like ordering (1st, 2nd, 3rd, etc.)
        if all(re.match(r'\d+(st|nd|rd|th)', str(c).lower()) for c in categories):
            return True
        
        return False
    
    def _detect_boolean_representation(self, series: pd.Series) -> str:
        """Detect how boolean values are represented"""
        unique_values = set(str(v).lower() for v in series.dropna().unique())
        
        if unique_values.issubset({'true', 'false'}):
            return 'true/false'
        elif unique_values.issubset({'1', '0'}):
            return '1/0'
        elif unique_values.issubset({'yes', 'no'}):
            return 'yes/no'
        elif unique_values.issubset({'y', 'n'}):
            return 'y/n'
        else:
            return 'mixed'
    
    def _detect_special_patterns(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Detect special patterns in the data"""
        patterns = {}
        
        # Check medical patterns
        for pattern_name, regex in self.medical_patterns.items():
            matches = series.astype(str).str.match(regex, na=False).sum()
            if matches > len(series) * 0.7:
                patterns[pattern_name] = True
        
        # Check column name hints
        column_lower = column_name.lower()
        if any(keyword in column_lower for keyword in ['id', 'key', 'identifier']):
            patterns['likely_identifier'] = True
        
        if any(keyword in column_lower for keyword in ['name', 'title', 'description']):
            patterns['likely_name_field'] = True
        
        if any(keyword in column_lower for keyword in ['date', 'time', 'created', 'updated']):
            patterns['likely_temporal'] = True
        
        return patterns
    
    def _assess_primary_key_candidacy(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if column could be a primary key"""
        assessment = {
            'is_candidate': False,
            'confidence': 0.0,
            'reasons': []
        }
        
        # High cardinality (unique or near-unique)
        if profile['cardinality_rate'] >= 0.95:
            assessment['confidence'] += 0.4
            assessment['reasons'].append('High cardinality')
        
        # No nulls
        if profile['null_rate'] == 0:
            assessment['confidence'] += 0.3
            assessment['reasons'].append('No null values')
        
        # Appropriate data type
        if profile['data_type_hint'] in ['integer', 'text']:
            assessment['confidence'] += 0.2
            assessment['reasons'].append('Appropriate data type')
        
        # Good quality
        if profile['quality_score'] > 0.8:
            assessment['confidence'] += 0.1
            assessment['reasons'].append('High quality score')
        
        assessment['is_candidate'] = assessment['confidence'] >= 0.7
        
        return assessment
    
    def _assess_foreign_key_candidacy(self, profile: Dict[str, Any], column_name: str) -> Dict[str, Any]:
        """Assess if column could be a foreign key"""
        assessment = {
            'is_candidate': False,
            'confidence': 0.0,
            'reasons': []
        }
        
        # Column name suggests relationship
        if any(suffix in column_name.lower() for suffix in ['_id', 'id', '_key', 'ref']):
            assessment['confidence'] += 0.3
            assessment['reasons'].append('Column name suggests foreign key')
        
        # Moderate cardinality (not unique, not too few values)
        if 0.1 <= profile['cardinality_rate'] <= 0.8:
            assessment['confidence'] += 0.2
            assessment['reasons'].append('Moderate cardinality')
        
        # Appropriate data type
        if profile['data_type_hint'] in ['integer', 'text']:
            assessment['confidence'] += 0.2
            assessment['reasons'].append('Appropriate data type')
        
        # Low null rate
        if profile['null_rate'] < 0.1:
            assessment['confidence'] += 0.2
            assessment['reasons'].append('Low null rate')
        
        # Good quality
        if profile['quality_score'] > 0.7:
            assessment['confidence'] += 0.1
            assessment['reasons'].append('Good quality score')
        
        assessment['is_candidate'] = assessment['confidence'] >= 0.6
        
        return assessment
    
    def _discover_constraints(self, df: pd.DataFrame, column_schemas: Dict[str, Any]) -> Dict[str, Any]:
        """Discover cross-column constraints"""
        constraints = {
            'unique_constraints': [],
            'check_constraints': [],
            'not_null_constraints': []
        }
        
        # Find columns that should be NOT NULL
        for col_name, schema in column_schemas.items():
            if schema['null_rate'] == 0 and len(df) > 10:  # Only if we have enough data
                constraints['not_null_constraints'].append(col_name)
        
        # Find potential unique constraints (combinations of columns)
        # This is computationally expensive, so we limit it
        if len(df.columns) <= 10 and len(df) <= 10000:
            for col_name, schema in column_schemas.items():
                if schema['cardinality_rate'] == 1.0:  # Perfectly unique
                    constraints['unique_constraints'].append([col_name])
        
        return constraints
    
    def _discover_relationships(self, df: pd.DataFrame, column_schemas: Dict[str, Any]) -> Dict[str, Any]:
        """Discover potential relationships between columns"""
        relationships = {
            'potential_foreign_keys': [],
            'hierarchical_relationships': [],
            'functional_dependencies': []
        }
        
        # Find potential foreign key relationships
        for col_name, schema in column_schemas.items():
            if schema['foreign_key_candidate']['is_candidate']:
                relationships['potential_foreign_keys'].append({
                    'column': col_name,
                    'confidence': schema['foreign_key_candidate']['confidence'],
                    'reasons': schema['foreign_key_candidate']['reasons']
                })
        
        # Look for hierarchical relationships (e.g., country -> state -> city)
        hierarchical_candidates = [
            col for col, schema in column_schemas.items()
            if schema['python_type'] == 'categorical' and 5 <= schema['unique_count'] <= 100
        ]
        
        if len(hierarchical_candidates) >= 2:
            relationships['hierarchical_relationships'] = hierarchical_candidates
        
        return relationships
    
    def _analyze_domain_patterns(self, df: pd.DataFrame, column_schemas: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze domain-specific patterns"""
        insights = {
            'healthcare_indicators': [],
            'financial_indicators': [],
            'temporal_patterns': [],
            'geographic_indicators': []
        }
        
        for col_name, schema in column_schemas.items():
            patterns = schema.get('special_patterns', {})
            
            # Healthcare patterns
            if any(pattern in patterns for pattern in ['icd_10', 'cpt_code', 'npi', 'mrn']):
                insights['healthcare_indicators'].append(col_name)
            
            # Financial patterns
            if schema['python_type'] == 'currency' or 'likely_percentage' in schema.get('constraints', {}):
                insights['financial_indicators'].append(col_name)
            
            # Temporal patterns
            if schema['python_type'] == 'date' or patterns.get('likely_temporal'):
                insights['temporal_patterns'].append(col_name)
            
            # Geographic patterns
            if any(keyword in col_name.lower() for keyword in ['zip', 'state', 'country', 'city', 'address']):
                insights['geographic_indicators'].append(col_name)
        
        return insights


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function for schema inference
    """
    inferrer = SchemaInferrer()
    return inferrer.infer_schema(df)


def export_schema_to_sql(schema: Dict[str, Any], table_name: str = None) -> str:
    """
    Export inferred schema to SQL CREATE TABLE statement
    """
    if table_name:
        schema['table_name'] = table_name
    
    sql_lines = [f"CREATE TABLE {schema['table_name']} ("]
    
    column_definitions = []
    for col_name, col_schema in schema['columns'].items():
        col_def = f"    {col_name} {col_schema['inferred_type']}"
        
        if not col_schema['nullable']:
            col_def += " NOT NULL"
        
        column_definitions.append(col_def)
    
    sql_lines.append(",\n".join(column_definitions))
    
    # Add constraints
    constraints = []
    
    # Primary key candidates
    pk_candidates = [
        col_name for col_name, col_schema in schema['columns'].items()
        if col_schema['primary_key_candidate']['is_candidate']
    ]
    if pk_candidates:
        constraints.append(f"    PRIMARY KEY ({pk_candidates[0]})")
    
    # Unique constraints
    for unique_cols in schema['constraints'].get('unique_constraints', []):
        if len(unique_cols) == 1:
            constraints.append(f"    UNIQUE ({unique_cols[0]})")
    
    if constraints:
        sql_lines.append(",")
        sql_lines.append(",\n".join(constraints))
    
    sql_lines.append(");")
    
    return "\n".join(sql_lines)


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        from .ingestion import ingest_data
        result = ingest_data(sys.argv[1])
        schema = infer_schema(result['dataframe'])
        print(f"Schema inferred for {schema['num_columns']} columns")
        print(f"Quality Score: {schema['quality_assessment']['avg_quality_score']:.2f}")
        
        # Export to SQL
        sql = export_schema_to_sql(schema)
        print("\nSQL Schema:")
        print(sql)
