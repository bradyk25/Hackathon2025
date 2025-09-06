"""
Basic PII Detection
Regex-based detection of personally identifiable information
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json


class PIIDetector:
    """
    Regex-based PII detection for common patterns
    """
    
    def __init__(self):
        # Common first names (top 100 most common)
        self.common_first_names = {
            'james', 'robert', 'john', 'michael', 'david', 'william', 'richard', 'charles',
            'joseph', 'thomas', 'christopher', 'daniel', 'paul', 'mark', 'donald', 'george',
            'kenneth', 'steven', 'edward', 'brian', 'ronald', 'anthony', 'kevin', 'jason',
            'matthew', 'gary', 'timothy', 'jose', 'larry', 'jeffrey', 'frank', 'scott',
            'eric', 'stephen', 'andrew', 'raymond', 'gregory', 'joshua', 'jerry', 'dennis',
            'walter', 'patrick', 'peter', 'harold', 'douglas', 'henry', 'carl', 'arthur',
            'ryan', 'roger', 'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara',
            'susan', 'jessica', 'sarah', 'karen', 'nancy', 'lisa', 'betty', 'helen',
            'sandra', 'donna', 'carol', 'ruth', 'sharon', 'michelle', 'laura', 'sarah',
            'kimberly', 'deborah', 'dorothy', 'lisa', 'nancy', 'karen', 'betty', 'helen',
            'sandra', 'donna', 'carol', 'ruth', 'sharon', 'michelle', 'laura', 'emily',
            'kimberly', 'deborah', 'dorothy', 'amy', 'angela', 'ashley', 'brenda', 'emma',
            'olivia', 'cynthia', 'marie', 'janet', 'catherine', 'frances', 'christine'
        }
        
        # Common last names (top 100 most common)
        self.common_last_names = {
            'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller', 'davis',
            'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez', 'wilson', 'anderson',
            'thomas', 'taylor', 'moore', 'jackson', 'martin', 'lee', 'perez', 'thompson',
            'white', 'harris', 'sanchez', 'clark', 'ramirez', 'lewis', 'robinson', 'walker',
            'young', 'allen', 'king', 'wright', 'scott', 'torres', 'nguyen', 'hill',
            'flores', 'green', 'adams', 'nelson', 'baker', 'hall', 'rivera', 'campbell',
            'mitchell', 'carter', 'roberts', 'gomez', 'phillips', 'evans', 'turner',
            'diaz', 'parker', 'cruz', 'edwards', 'collins', 'reyes', 'stewart', 'morris',
            'morales', 'murphy', 'cook', 'rogers', 'gutierrez', 'ortiz', 'morgan', 'cooper',
            'peterson', 'bailey', 'reed', 'kelly', 'howard', 'ramos', 'kim', 'cox',
            'ward', 'richardson', 'watson', 'brooks', 'chavez', 'wood', 'james', 'bennett',
            'gray', 'mendoza', 'ruiz', 'hughes', 'price', 'alvarez', 'castillo', 'sanders',
            'patel', 'myers', 'long', 'ross', 'foster', 'jimenez'
        }
        
        # PII detection patterns
        self.pii_patterns = {
            'ssn': {
                'patterns': [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
                    r'\b\d{3}\s\d{2}\s\d{4}\b',  # XXX XX XXXX
                    r'\b\d{9}\b'  # XXXXXXXXX (9 consecutive digits)
                ],
                'confidence_threshold': 0.8,
                'description': 'Social Security Number'
            },
            'phone': {
                'patterns': [
                    r'\b\d{3}-\d{3}-\d{4}\b',  # XXX-XXX-XXXX
                    r'\b\(\d{3}\)\s?\d{3}-\d{4}\b',  # (XXX) XXX-XXXX
                    r'\b\d{3}\.\d{3}\.\d{4}\b',  # XXX.XXX.XXXX
                    r'\b\d{3}\s\d{3}\s\d{4}\b',  # XXX XXX XXXX
                    r'\b1?\s?\d{3}\s?\d{3}\s?\d{4}\b'  # Various formats with optional 1
                ],
                'confidence_threshold': 0.7,
                'description': 'Phone Number'
            },
            'email': {
                'patterns': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                ],
                'confidence_threshold': 0.9,
                'description': 'Email Address'
            },
            'credit_card': {
                'patterns': [
                    r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Visa
                    r'\b5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # MasterCard
                    r'\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b',  # American Express
                    r'\b6011[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'  # Discover
                ],
                'confidence_threshold': 0.8,
                'description': 'Credit Card Number'
            },
            'drivers_license': {
                'patterns': [
                    r'\b[A-Z]{1,2}\d{6,8}\b',  # Common DL format
                    r'\b\d{8,9}\b'  # Numeric DL
                ],
                'confidence_threshold': 0.6,
                'description': 'Driver\'s License'
            },
            'passport': {
                'patterns': [
                    r'\b[A-Z]{2}\d{7}\b',  # US Passport format
                    r'\b\d{9}\b'  # 9-digit passport
                ],
                'confidence_threshold': 0.6,
                'description': 'Passport Number'
            },
            'ip_address': {
                'patterns': [
                    r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
                ],
                'confidence_threshold': 0.9,
                'description': 'IP Address'
            },
            'address': {
                'patterns': [
                    r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
                    r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\.?\s*,?\s*[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}(-\d{4})?\b'
                ],
                'confidence_threshold': 0.7,
                'description': 'Street Address'
            },
            'zip_code': {
                'patterns': [
                    r'\b\d{5}(-\d{4})?\b'
                ],
                'confidence_threshold': 0.8,
                'description': 'ZIP Code'
            },
            'date_of_birth': {
                'patterns': [
                    r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(19|20)\d{2}\b',  # MM/DD/YYYY
                    r'\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b'  # YYYY-MM-DD
                ],
                'confidence_threshold': 0.6,
                'description': 'Date of Birth'
            }
        }
        
        # Medical/Healthcare specific PII patterns
        self.medical_patterns = {
            'mrn': {
                'patterns': [
                    r'\b[A-Z0-9]{6,12}\b'  # Medical Record Number (varies by institution)
                ],
                'confidence_threshold': 0.5,
                'description': 'Medical Record Number'
            },
            'npi': {
                'patterns': [
                    r'\b\d{10}\b'  # National Provider Identifier
                ],
                'confidence_threshold': 0.7,
                'description': 'National Provider Identifier'
            },
            'insurance_id': {
                'patterns': [
                    r'\b[A-Z]{2,3}\d{6,9}\b',  # Common insurance ID format
                    r'\b\d{9,12}\b'  # Numeric insurance ID
                ],
                'confidence_threshold': 0.5,
                'description': 'Insurance ID'
            }
        }
    
    def detect_pii_in_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Detect PII patterns in a single column
        """
        results = {
            'column_name': column_name,
            'total_values': len(series),
            'non_null_values': series.notna().sum(),
            'detected_patterns': {},
            'overall_risk_score': 0.0,
            'recommendations': []
        }
        
        # Convert to string for pattern matching
        str_series = series.astype(str).fillna('')
        
        # Check all PII patterns
        all_patterns = {**self.pii_patterns, **self.medical_patterns}
        
        for pattern_name, pattern_config in all_patterns.items():
            pattern_results = self._check_pattern(str_series, pattern_config, pattern_name)
            if pattern_results['match_count'] > 0:
                results['detected_patterns'][pattern_name] = pattern_results
        
        # Check for name patterns
        name_results = self._check_names(str_series, column_name)
        if name_results['confidence'] > 0.5:
            results['detected_patterns']['names'] = name_results
        
        # Calculate overall risk score
        results['overall_risk_score'] = self._calculate_risk_score(results['detected_patterns'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_pii_recommendations(results)
        
        return results
    
    def _check_pattern(self, str_series: pd.Series, pattern_config: Dict[str, Any], pattern_name: str) -> Dict[str, Any]:
        """
        Check a specific PII pattern against the series
        """
        total_matches = 0
        sample_matches = []
        
        for pattern in pattern_config['patterns']:
            matches = str_series.str.contains(pattern, regex=True, na=False)
            match_count = matches.sum()
            total_matches += match_count
            
            if match_count > 0 and len(sample_matches) < 5:
                # Get sample matches (first few)
                matched_values = str_series[matches].head(5 - len(sample_matches)).tolist()
                sample_matches.extend(matched_values)
        
        match_rate = total_matches / len(str_series) if len(str_series) > 0 else 0
        confidence = min(match_rate / pattern_config['confidence_threshold'], 1.0)
        
        return {
            'pattern_type': pattern_name,
            'description': pattern_config['description'],
            'match_count': int(total_matches),
            'match_rate': float(match_rate),
            'confidence': float(confidence),
            'sample_matches': sample_matches[:5],  # Limit to 5 samples
            'risk_level': self._assess_risk_level(confidence, match_rate)
        }
    
    def _check_names(self, str_series: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Check for personal names using common name lists
        """
        # Check column name for hints
        column_hints = any(keyword in column_name.lower() 
                          for keyword in ['name', 'first', 'last', 'fname', 'lname', 'patient', 'client'])
        
        # Check values against name lists
        first_name_matches = 0
        last_name_matches = 0
        
        for value in str_series.str.lower():
            if value in self.common_first_names:
                first_name_matches += 1
            if value in self.common_last_names:
                last_name_matches += 1
        
        total_values = len(str_series)
        first_name_rate = first_name_matches / total_values if total_values > 0 else 0
        last_name_rate = last_name_matches / total_values if total_values > 0 else 0
        
        # Calculate confidence
        confidence = 0.0
        if column_hints:
            confidence += 0.3
        if first_name_rate > 0.3:
            confidence += 0.4
        if last_name_rate > 0.3:
            confidence += 0.4
        
        name_type = 'unknown'
        if first_name_rate > last_name_rate and first_name_rate > 0.3:
            name_type = 'first_name'
        elif last_name_rate > first_name_rate and last_name_rate > 0.3:
            name_type = 'last_name'
        elif first_name_rate > 0.2 and last_name_rate > 0.2:
            name_type = 'full_name'
        
        return {
            'pattern_type': 'names',
            'description': 'Personal Names',
            'name_type': name_type,
            'first_name_matches': int(first_name_matches),
            'last_name_matches': int(last_name_matches),
            'first_name_rate': float(first_name_rate),
            'last_name_rate': float(last_name_rate),
            'confidence': float(min(confidence, 1.0)),
            'column_hints': column_hints,
            'risk_level': self._assess_risk_level(confidence, max(first_name_rate, last_name_rate))
        }
    
    def _assess_risk_level(self, confidence: float, match_rate: float) -> str:
        """
        Assess risk level based on confidence and match rate
        """
        if confidence >= 0.8 and match_rate >= 0.7:
            return 'HIGH'
        elif confidence >= 0.6 and match_rate >= 0.5:
            return 'MEDIUM'
        elif confidence >= 0.3 and match_rate >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _calculate_risk_score(self, detected_patterns: Dict[str, Any]) -> float:
        """
        Calculate overall risk score for the column
        """
        if not detected_patterns:
            return 0.0
        
        # Weight different PII types
        pii_weights = {
            'ssn': 1.0,
            'credit_card': 1.0,
            'email': 0.8,
            'phone': 0.7,
            'address': 0.8,
            'names': 0.6,
            'drivers_license': 0.7,
            'passport': 0.9,
            'date_of_birth': 0.8,
            'mrn': 0.9,
            'npi': 0.7,
            'insurance_id': 0.8
        }
        
        total_score = 0.0
        max_possible = 0.0
        
        for pattern_name, pattern_result in detected_patterns.items():
            weight = pii_weights.get(pattern_name, 0.5)
            confidence = pattern_result.get('confidence', 0.0)
            score = weight * confidence
            total_score += score
            max_possible += weight
        
        return min(total_score / max_possible if max_possible > 0 else 0.0, 1.0)
    
    def _generate_pii_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on PII detection results
        """
        recommendations = []
        risk_score = results['overall_risk_score']
        detected_patterns = results['detected_patterns']
        
        if risk_score >= 0.8:
            recommendations.append("HIGH RISK: This column contains sensitive PII and should be masked or encrypted")
        elif risk_score >= 0.6:
            recommendations.append("MEDIUM RISK: Consider masking or tokenizing this column")
        elif risk_score >= 0.3:
            recommendations.append("LOW RISK: Monitor this column for potential PII exposure")
        
        # Specific recommendations by pattern type
        high_risk_patterns = ['ssn', 'credit_card', 'passport', 'mrn']
        for pattern in high_risk_patterns:
            if pattern in detected_patterns:
                pattern_result = detected_patterns[pattern]
                if pattern_result['confidence'] > 0.7:
                    recommendations.append(f"Detected {pattern_result['description']} - requires immediate masking")
        
        if 'names' in detected_patterns:
            name_result = detected_patterns['names']
            if name_result['confidence'] > 0.7:
                recommendations.append("Personal names detected - consider pseudonymization")
        
        return recommendations
    
    def detect_pii_in_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect PII across entire dataset
        """
        dataset_results = {
            'dataset_summary': {
                'total_columns': len(df.columns),
                'total_rows': len(df),
                'columns_with_pii': 0,
                'high_risk_columns': 0,
                'medium_risk_columns': 0,
                'low_risk_columns': 0
            },
            'column_results': {},
            'overall_risk_assessment': 'MINIMAL',
            'priority_actions': [],
            'compliance_notes': []
        }
        
        # Analyze each column
        for column in df.columns:
            column_result = self.detect_pii_in_column(df[column], column)
            dataset_results['column_results'][column] = column_result
            
            # Update summary statistics
            if column_result['detected_patterns']:
                dataset_results['dataset_summary']['columns_with_pii'] += 1
                
                risk_score = column_result['overall_risk_score']
                if risk_score >= 0.8:
                    dataset_results['dataset_summary']['high_risk_columns'] += 1
                elif risk_score >= 0.6:
                    dataset_results['dataset_summary']['medium_risk_columns'] += 1
                elif risk_score >= 0.3:
                    dataset_results['dataset_summary']['low_risk_columns'] += 1
        
        # Determine overall risk assessment
        high_risk = dataset_results['dataset_summary']['high_risk_columns']
        medium_risk = dataset_results['dataset_summary']['medium_risk_columns']
        
        if high_risk > 0:
            dataset_results['overall_risk_assessment'] = 'HIGH'
        elif medium_risk > 0:
            dataset_results['overall_risk_assessment'] = 'MEDIUM'
        elif dataset_results['dataset_summary']['columns_with_pii'] > 0:
            dataset_results['overall_risk_assessment'] = 'LOW'
        
        # Generate priority actions
        dataset_results['priority_actions'] = self._generate_dataset_recommendations(dataset_results)
        
        # Add compliance notes
        dataset_results['compliance_notes'] = self._generate_compliance_notes(dataset_results)
        
        return dataset_results
    
    def _generate_dataset_recommendations(self, dataset_results: Dict[str, Any]) -> List[str]:
        """
        Generate dataset-level recommendations
        """
        recommendations = []
        summary = dataset_results['dataset_summary']
        
        if summary['high_risk_columns'] > 0:
            recommendations.append(f"URGENT: {summary['high_risk_columns']} columns contain high-risk PII requiring immediate attention")
        
        if summary['medium_risk_columns'] > 0:
            recommendations.append(f"IMPORTANT: {summary['medium_risk_columns']} columns contain medium-risk PII requiring masking")
        
        if summary['columns_with_pii'] > summary['total_columns'] * 0.5:
            recommendations.append("Dataset contains significant PII - consider comprehensive data anonymization")
        
        # Specific high-risk column recommendations
        high_risk_columns = []
        for col_name, col_result in dataset_results['column_results'].items():
            if col_result['overall_risk_score'] >= 0.8:
                high_risk_columns.append(col_name)
        
        if high_risk_columns:
            recommendations.append(f"Priority columns for masking: {', '.join(high_risk_columns[:5])}")
        
        return recommendations
    
    def _generate_compliance_notes(self, dataset_results: Dict[str, Any]) -> List[str]:
        """
        Generate compliance-related notes
        """
        notes = []
        
        # Check for HIPAA-relevant patterns
        hipaa_patterns = ['mrn', 'npi', 'insurance_id', 'ssn', 'names', 'date_of_birth', 'address']
        hipaa_detected = []
        
        for col_name, col_result in dataset_results['column_results'].items():
            for pattern in hipaa_patterns:
                if pattern in col_result['detected_patterns']:
                    if col_result['detected_patterns'][pattern]['confidence'] > 0.6:
                        hipaa_detected.append(pattern)
        
        if hipaa_detected:
            notes.append(f"HIPAA Compliance: Detected healthcare identifiers ({', '.join(set(hipaa_detected))})")
            notes.append("Consider HIPAA Safe Harbor de-identification requirements")
        
        # Check for financial data
        financial_patterns = ['credit_card', 'ssn']
        financial_detected = any(
            pattern in col_result['detected_patterns'] 
            for col_result in dataset_results['column_results'].values()
            for pattern in financial_patterns
        )
        
        if financial_detected:
            notes.append("PCI DSS Compliance: Financial data detected - ensure proper encryption and access controls")
        
        return notes


def detect_pii(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function for PII detection
    """
    detector = PIIDetector()
    return detector.detect_pii_in_dataset(df)


def generate_privacy_report(df: pd.DataFrame, output_file: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive privacy assessment report
    """
    detector = PIIDetector()
    pii_results = detector.detect_pii_in_dataset(df)
    
    report = {
        'executive_summary': {
            'dataset_name': 'analyzed_dataset',
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'pii_columns_found': pii_results['dataset_summary']['columns_with_pii'],
            'overall_risk': pii_results['overall_risk_assessment'],
            'immediate_actions_required': len(pii_results['priority_actions'])
        },
        'detailed_findings': pii_results,
        'next_steps': [
            'Review high-risk columns identified in the priority actions',
            'Implement recommended masking strategies',
            'Establish data governance policies',
            'Regular PII scanning and monitoring'
        ]
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        from .ingestion import ingest_data
        result = ingest_data(sys.argv[1])
        pii_results = detect_pii(result['dataframe'])
        print(f"PII Detection Complete:")
        print(f"Overall Risk: {pii_results['overall_risk_assessment']}")
        print(f"Columns with PII: {pii_results['dataset_summary']['columns_with_pii']}")
        print(f"High Risk Columns: {pii_results['dataset_summary']['high_risk_columns']}")
