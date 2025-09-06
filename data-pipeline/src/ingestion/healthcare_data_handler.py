"""
Specialized handler for healthcare claims and patient data.
Handles the specific format and structure of the hackathon dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class HealthcareDataHandler:
    """
    Specialized handler for healthcare claims and patient data.
    Handles mixed data formats, missing values, and healthcare-specific patterns.
    """
    
    def __init__(self):
        # Healthcare-specific field mappings
        self.claims_fields = {
            'CLAIM_ID', 'MEMBER_ID', 'CLAIM_TYPE', 'ADMISSION_TYPE', 
            'PROVIDER_ZIP', 'PROVIDER_SPECIALTY', 'DIAGNOSIS_CODE', 
            'PROCEDURE_CODE', 'TOTAL_CHARGE', 'COPAY', 'COINSURANCE', 
            'ADJUDICATION_STATUS', 'CLAIM_COST'
        }
        
        self.patient_fields = {
            'ID', 'LAST_NAME', 'AGE', 'GENDER', 'DRG_CODE', 
            'ICD9_PROCEDURE', 'STATUS'
        }
        
        # Data quality issues to handle
        self.known_issues = {
            'typos': {'Emergny': 'Emergency', 'Urgent': 'Urgent'},
            'missing_values': ['N/A', '', 'NULL', 'null'],
            'mixed_formats': True
        }
    
    async def process_healthcare_data(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process healthcare data with specific handling for the hackathon dataset.
        
        Args:
            raw_data: Raw data from file parsing
            
        Returns:
            Processed data with separated claims and patient records
        """
        
        if not raw_data:
            return {
                'claims_data': [],
                'patient_data': [],
                'processing_summary': {
                    'total_records': 0,
                    'claims_records': 0,
                    'patient_records': 0,
                    'issues_found': []
                }
            }
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(raw_data)
        
        # Clean and standardize the data
        df_cleaned = await self._clean_healthcare_data(df)
        
        # Separate claims and patient data
        claims_data, patient_data = await self._separate_data_types(df_cleaned)
        
        # Generate processing summary
        processing_summary = self._generate_processing_summary(df, df_cleaned, claims_data, patient_data)
        
        return {
            'claims_data': claims_data.to_dict('records') if not claims_data.empty else [],
            'patient_data': patient_data.to_dict('records') if not patient_data.empty else [],
            'processing_summary': processing_summary
        }
    
    async def _clean_healthcare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize healthcare data"""
        
        df_clean = df.copy()
        
        # Handle missing values
        for missing_val in self.known_issues['missing_values']:
            df_clean = df_clean.replace(missing_val, np.nan)
        
        # Fix known typos
        for typo, correction in self.known_issues['typos'].items():
            df_clean = df_clean.replace(typo, correction)
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.strip().str.upper()
        
        # Handle numeric fields with proper conversion
        numeric_fields = ['TOTAL_CHARGE', 'COPAY', 'COINSURANCE', 'CLAIM_COST', 
                         'AGE', 'DRG_CODE', 'ICD9_PROCEDURE', 'PROVIDER_ZIP']
        
        for field in numeric_fields:
            if field in df_clean.columns:
                df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
        
        # Handle categorical fields
        categorical_fields = ['CLAIM_TYPE', 'ADMISSION_TYPE', 'PROVIDER_SPECIALTY', 
                            'ADJUDICATION_STATUS', 'GENDER', 'STATUS']
        
        for field in categorical_fields:
            if field in df_clean.columns:
                df_clean[field] = df_clean[field].astype('category')
        
        # Validate healthcare-specific patterns
        df_clean = await self._validate_healthcare_patterns(df_clean)
        
        return df_clean
    
    async def _validate_healthcare_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean healthcare-specific patterns"""
        
        df_validated = df.copy()
        
        # Validate claim IDs (should start with specific prefixes)
        if 'CLAIM_ID' in df_validated.columns:
            # Mark invalid claim IDs
            valid_prefixes = ['IQV', 'CMS']
            mask = df_validated['CLAIM_ID'].str.startswith(tuple(valid_prefixes), na=False)
            df_validated.loc[~mask, 'CLAIM_ID'] = np.nan
        
        # Validate member IDs
        if 'MEMBER_ID' in df_validated.columns:
            # Should start with 'M' or 'PT'
            valid_member_prefixes = ['M', 'PT']
            mask = df_validated['MEMBER_ID'].str.startswith(tuple(valid_member_prefixes), na=False)
            df_validated.loc[~mask, 'MEMBER_ID'] = np.nan
        
        # Validate diagnosis codes (ICD format)
        if 'DIAGNOSIS_CODE' in df_validated.columns:
            # Basic ICD pattern validation
            icd_pattern = r'^[A-Z]\d{2}(\.\d{1,2})?$'
            mask = df_validated['DIAGNOSIS_CODE'].str.match(icd_pattern, na=False)
            # Don't nullify, but flag for review
            df_validated.loc[~mask, 'DIAGNOSIS_CODE_VALID'] = False
            df_validated.loc[mask, 'DIAGNOSIS_CODE_VALID'] = True
        
        # Validate age ranges
        if 'AGE' in df_validated.columns:
            # Age should be between 0 and 120
            mask = (df_validated['AGE'] >= 0) & (df_validated['AGE'] <= 120)
            df_validated.loc[~mask, 'AGE'] = np.nan
        
        # Validate gender codes
        if 'GENDER' in df_validated.columns:
            valid_genders = ['M', 'F', 'Male', 'Female']
            mask = df_validated['GENDER'].isin(valid_genders)
            df_validated.loc[~mask, 'GENDER'] = np.nan
        
        # Validate monetary amounts (should be positive)
        money_fields = ['TOTAL_CHARGE', 'COPAY', 'CLAIM_COST']
        for field in money_fields:
            if field in df_validated.columns:
                mask = df_validated[field] >= 0
                df_validated.loc[~mask, field] = np.nan
        
        # Validate coinsurance (should be between 0 and 1)
        if 'COINSURANCE' in df_validated.columns:
            mask = (df_validated['COINSURANCE'] >= 0) & (df_validated['COINSURANCE'] <= 1)
            df_validated.loc[~mask, 'COINSURANCE'] = np.nan
        
        return df_validated
    
    async def _separate_data_types(self, df: pd.DataFrame) -> tuple:
        """Separate claims data from patient data based on field patterns"""
        
        # Identify records with claims data vs patient data
        claims_mask = pd.Series([False] * len(df))
        patient_mask = pd.Series([False] * len(df))
        
        # Check for claims-specific fields
        for field in self.claims_fields:
            if field in df.columns:
                claims_mask |= df[field].notna()
        
        # Check for patient-specific fields
        for field in self.patient_fields:
            if field in df.columns:
                patient_mask |= df[field].notna()
        
        # Handle mixed records (records with both claims and patient data)
        mixed_mask = claims_mask & patient_mask
        
        if mixed_mask.any():
            logger.info(f"Found {mixed_mask.sum()} mixed records with both claims and patient data")
            # For mixed records, duplicate them into both datasets
            claims_df = df[claims_mask].copy()
            patient_df = df[patient_mask].copy()
        else:
            # Clean separation
            claims_df = df[claims_mask & ~patient_mask].copy()
            patient_df = df[patient_mask & ~claims_mask].copy()
        
        # Clean up columns for each dataset
        if not claims_df.empty:
            # Keep only relevant columns for claims
            claims_columns = [col for col in claims_df.columns if col in self.claims_fields or col.endswith('_VALID')]
            claims_df = claims_df[claims_columns]
        
        if not patient_df.empty:
            # Keep only relevant columns for patients
            patient_columns = [col for col in patient_df.columns if col in self.patient_fields or col.endswith('_VALID')]
            patient_df = patient_df[patient_columns]
        
        return claims_df, patient_df
    
    def _generate_processing_summary(
        self, 
        original_df: pd.DataFrame, 
        cleaned_df: pd.DataFrame, 
        claims_df: pd.DataFrame, 
        patient_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate summary of data processing"""
        
        issues_found = []
        
        # Check for data quality issues
        if original_df.isnull().sum().sum() != cleaned_df.isnull().sum().sum():
            issues_found.append("Missing values standardized")
        
        # Check for duplicates
        duplicates = original_df.duplicated().sum()
        if duplicates > 0:
            issues_found.append(f"{duplicates} duplicate records found")
        
        # Check for invalid patterns
        if 'DIAGNOSIS_CODE_VALID' in cleaned_df.columns:
            invalid_diagnoses = (~cleaned_df['DIAGNOSIS_CODE_VALID']).sum()
            if invalid_diagnoses > 0:
                issues_found.append(f"{invalid_diagnoses} invalid diagnosis codes")
        
        # Check for out-of-range values
        if 'AGE' in cleaned_df.columns:
            invalid_ages = cleaned_df['AGE'].isnull().sum() - original_df['AGE'].isnull().sum()
            if invalid_ages > 0:
                issues_found.append(f"{invalid_ages} invalid age values")
        
        return {
            'total_records': len(original_df),
            'claims_records': len(claims_df),
            'patient_records': len(patient_df),
            'duplicate_records': duplicates,
            'issues_found': issues_found,
            'data_quality_score': self._calculate_data_quality_score(cleaned_df),
            'field_completeness': self._calculate_field_completeness(cleaned_df)
        }
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        
        if df.empty:
            return 0.0
        
        # Completeness score
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        # Validity score (based on validation flags)
        validity_columns = [col for col in df.columns if col.endswith('_VALID')]
        if validity_columns:
            validity = df[validity_columns].mean().mean()
        else:
            validity = 1.0  # Assume valid if no validation flags
        
        # Consistency score (simplified)
        consistency = 0.9  # Placeholder - would need more complex logic
        
        overall_score = (completeness * 0.4 + validity * 0.4 + consistency * 0.2)
        return round(overall_score, 3)
    
    def _calculate_field_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate completeness for each field"""
        
        if df.empty:
            return {}
        
        completeness = {}
        for column in df.columns:
            if not column.endswith('_VALID'):
                complete_ratio = 1 - (df[column].isnull().sum() / len(df))
                completeness[column] = round(complete_ratio, 3)
        
        return completeness
    
    def get_healthcare_schema_hints(self) -> Dict[str, Dict[str, Any]]:
        """Provide schema hints specific to healthcare data"""
        
        return {
            'CLAIM_ID': {
                'type': 'string',
                'pattern': r'^(IQV|CMS)\d+$',
                'description': 'Unique claim identifier',
                'healthcare_category': 'identifier'
            },
            'MEMBER_ID': {
                'type': 'string',
                'pattern': r'^(M|PT)\d+$',
                'description': 'Patient/member identifier',
                'healthcare_category': 'identifier'
            },
            'CLAIM_TYPE': {
                'type': 'categorical',
                'categories': ['Inpatient', 'Outpatient', 'Pharmacy'],
                'description': 'Type of healthcare claim',
                'healthcare_category': 'administrative'
            },
            'ADMISSION_TYPE': {
                'type': 'categorical',
                'categories': ['Emergency', 'Elective', 'Urgent', 'Newborn'],
                'description': 'Type of hospital admission',
                'healthcare_category': 'clinical'
            },
            'PROVIDER_SPECIALTY': {
                'type': 'categorical',
                'categories': ['Oncology', 'Orthopedics', 'Pediatrics', 'Cardiology', 'General Practice'],
                'description': 'Medical specialty of provider',
                'healthcare_category': 'provider'
            },
            'DIAGNOSIS_CODE': {
                'type': 'string',
                'pattern': r'^[A-Z]\d{2}(\.\d{1,2})?$',
                'description': 'ICD diagnosis code',
                'healthcare_category': 'clinical'
            },
            'TOTAL_CHARGE': {
                'type': 'float',
                'min': 0,
                'max': 100000,
                'description': 'Total charge amount',
                'healthcare_category': 'financial'
            },
            'AGE': {
                'type': 'integer',
                'min': 0,
                'max': 120,
                'description': 'Patient age',
                'healthcare_category': 'demographics'
            },
            'GENDER': {
                'type': 'categorical',
                'categories': ['M', 'F'],
                'description': 'Patient gender',
                'healthcare_category': 'demographics'
            }
        }
