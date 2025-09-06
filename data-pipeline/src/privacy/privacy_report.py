"""
Privacy analysis and PII/PHI detection for healthcare data.
This module identifies and masks sensitive information in healthcare datasets.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import asyncio

from ..utils.config import get_settings

logger = logging.getLogger(__name__)

class PrivacyAnalyzer:
    """
    Analyzes healthcare data for privacy risks and sensitive information.
    Detects PII (Personally Identifiable Information) and PHI (Protected Health Information).
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.pii_threshold = self.settings.pii_detection_threshold
        self.phi_threshold = self.settings.phi_detection_threshold
        
        # PII Detection Patterns
        self.pii_patterns = {
            "ssn": [
                r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
                r'\b\d{9}\b',              # XXXXXXXXX
                r'\b\d{3}\s\d{2}\s\d{4}\b' # XXX XX XXXX
            ],
            "email": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "phone": [
                r'\b\d{3}-\d{3}-\d{4}\b',     # XXX-XXX-XXXX
                r'\b\(\d{3}\)\s\d{3}-\d{4}\b', # (XXX) XXX-XXXX
                r'\b\d{10}\b'                  # XXXXXXXXXX
            ],
            "credit_card": [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            ],
            "drivers_license": [
                r'\b[A-Z]{1,2}\d{6,8}\b'
            ]
        }
        
        # PHI Detection Patterns and Keywords
        self.phi_keywords = {
            "medical_record_number": ["mrn", "medical_record", "patient_id", "chart_number"],
            "diagnosis": ["diagnosis", "condition", "disease", "disorder", "syndrome"],
            "medication": ["medication", "drug", "prescription", "rx", "medicine"],
            "procedure": ["procedure", "surgery", "operation", "treatment"],
            "provider": ["doctor", "physician", "nurse", "provider", "clinician"],
            "insurance": ["insurance", "policy", "member_id", "subscriber"],
            "facility": ["hospital", "clinic", "facility", "department"]
        }
        
        # Common name patterns (simplified)
        self.name_patterns = [
            r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+,\s[A-Z][a-z]+\b'  # Last, First
        ]
    
    async def analyze_privacy(
        self,
        data: List[Dict[str, Any]],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze data for privacy risks and generate privacy report.
        
        Args:
            data: Raw data to analyze
            schema: Schema information
            
        Returns:
            Privacy analysis report with cleaned data
        """
        
        if not data:
            return {
                "pii_detected": {"count": 0, "fields": [], "confidence": []},
                "phi_detected": {"count": 0, "fields": [], "confidence": []},
                "masking_applied": {"pii": [], "phi": []},
                "risk_score": "NONE",
                "cleaned_data": []
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)
        
        # Detect PII
        pii_results = await self._detect_pii(df)
        
        # Detect PHI
        phi_results = await self._detect_phi(df, schema)
        
        # Apply masking
        cleaned_df = await self._apply_masking(df, pii_results, phi_results)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(pii_results, phi_results)
        
        # Generate report
        privacy_report = {
            "pii_detected": {
                "count": len(pii_results["fields"]),
                "fields": pii_results["fields"],
                "confidence": pii_results["confidence"]
            },
            "phi_detected": {
                "count": len(phi_results["fields"]),
                "fields": phi_results["fields"],
                "confidence": phi_results["confidence"]
            },
            "masking_applied": {
                "pii": pii_results["fields"],
                "phi": phi_results["fields"]
            },
            "risk_score": risk_score,
            "cleaned_data": cleaned_df.to_dict('records'),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "total_records_processed": len(data),
            "fields_analyzed": list(df.columns)
        }
        
        return privacy_report
    
    async def _detect_pii(self, df: pd.DataFrame) -> Dict[str, List]:
        """Detect PII in the dataset"""
        
        detected_fields = []
        confidence_scores = []
        
        for column in df.columns:
            column_lower = column.lower()
            max_confidence = 0.0
            
            # Check column name for PII indicators
            if any(keyword in column_lower for keyword in ["name", "ssn", "social", "email", "phone"]):
                max_confidence = max(max_confidence, 0.9)
            
            # Check data patterns
            if df[column].dtype == 'object':  # String columns
                sample_values = df[column].dropna().astype(str).head(100)
                
                for pii_type, patterns in self.pii_patterns.items():
                    pattern_matches = 0
                    total_checked = 0
                    
                    for value in sample_values:
                        total_checked += 1
                        for pattern in patterns:
                            if re.search(pattern, value):
                                pattern_matches += 1
                                break
                    
                    if total_checked > 0:
                        pattern_confidence = pattern_matches / total_checked
                        if pattern_confidence > 0.1:  # At least 10% match
                            max_confidence = max(max_confidence, pattern_confidence)
            
            # Check for name patterns
            if "name" in column_lower:
                sample_values = df[column].dropna().astype(str).head(50)
                name_matches = sum(1 for value in sample_values 
                                 if any(re.search(pattern, value) for pattern in self.name_patterns))
                if len(sample_values) > 0:
                    name_confidence = name_matches / len(sample_values)
                    max_confidence = max(max_confidence, name_confidence)
            
            # If confidence exceeds threshold, mark as PII
            if max_confidence >= self.pii_threshold:
                detected_fields.append(column)
                confidence_scores.append(round(max_confidence, 3))
        
        return {
            "fields": detected_fields,
            "confidence": confidence_scores
        }
    
    async def _detect_phi(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, List]:
        """Detect PHI in the dataset"""
        
        detected_fields = []
        confidence_scores = []
        
        for column in df.columns:
            column_lower = column.lower()
            max_confidence = 0.0
            
            # Check against PHI keywords
            for phi_category, keywords in self.phi_keywords.items():
                if any(keyword in column_lower for keyword in keywords):
                    max_confidence = max(max_confidence, 0.85)
                    break
            
            # Check schema information for PHI indicators
            field_info = schema.get("fields", {}).get(column, {})
            field_description = field_info.get("description", "").lower()
            
            if any(keyword in field_description for keywords in self.phi_keywords.values() for keyword in keywords):
                max_confidence = max(max_confidence, 0.8)
            
            # Healthcare-specific patterns
            if df[column].dtype == 'object':
                sample_values = df[column].dropna().astype(str).head(100)
                
                # Medical record number patterns
                mrn_pattern = r'\b[A-Z]{0,3}\d{6,10}\b'
                mrn_matches = sum(1 for value in sample_values if re.search(mrn_pattern, value))
                if len(sample_values) > 0 and mrn_matches / len(sample_values) > 0.3:
                    max_confidence = max(max_confidence, 0.9)
                
                # ICD codes
                icd_pattern = r'\b[A-Z]\d{2}(\.\d{1,2})?\b'
                icd_matches = sum(1 for value in sample_values if re.search(icd_pattern, value))
                if len(sample_values) > 0 and icd_matches / len(sample_values) > 0.2:
                    max_confidence = max(max_confidence, 0.85)
            
            # Date fields in healthcare context
            if "date" in column_lower and any(term in column_lower for term in ["birth", "visit", "admission", "discharge"]):
                max_confidence = max(max_confidence, 0.8)
            
            # If confidence exceeds threshold, mark as PHI
            if max_confidence >= self.phi_threshold:
                detected_fields.append(column)
                confidence_scores.append(round(max_confidence, 3))
        
        return {
            "fields": detected_fields,
            "confidence": confidence_scores
        }
    
    async def _apply_masking(
        self,
        df: pd.DataFrame,
        pii_results: Dict[str, List],
        phi_results: Dict[str, List]
    ) -> pd.DataFrame:
        """Apply masking to sensitive fields"""
        
        masked_df = df.copy()
        
        # Mask PII fields
        for field in pii_results["fields"]:
            if field in masked_df.columns:
                masked_df[field] = self._mask_pii_field(masked_df[field], field)
        
        # Mask PHI fields
        for field in phi_results["fields"]:
            if field in masked_df.columns:
                masked_df[field] = self._mask_phi_field(masked_df[field], field)
        
        return masked_df
    
    def _mask_pii_field(self, series: pd.Series, field_name: str) -> pd.Series:
        """Mask PII field values"""
        
        field_lower = field_name.lower()
        
        if "ssn" in field_lower or "social" in field_lower:
            return series.apply(lambda x: "XXX-XX-XXXX" if pd.notna(x) else x)
        elif "email" in field_lower:
            return series.apply(lambda x: "masked@email.com" if pd.notna(x) else x)
        elif "phone" in field_lower:
            return series.apply(lambda x: "XXX-XXX-XXXX" if pd.notna(x) else x)
        elif "name" in field_lower:
            return series.apply(lambda x: "MASKED_NAME" if pd.notna(x) else x)
        else:
            # Generic masking
            return series.apply(lambda x: "MASKED_PII" if pd.notna(x) else x)
    
    def _mask_phi_field(self, series: pd.Series, field_name: str) -> pd.Series:
        """Mask PHI field values"""
        
        field_lower = field_name.lower()
        
        if any(term in field_lower for term in ["diagnosis", "condition"]):
            return series.apply(lambda x: "MASKED_DIAGNOSIS" if pd.notna(x) else x)
        elif any(term in field_lower for term in ["medication", "drug"]):
            return series.apply(lambda x: "MASKED_MEDICATION" if pd.notna(x) else x)
        elif any(term in field_lower for term in ["provider", "doctor", "physician"]):
            return series.apply(lambda x: "MASKED_PROVIDER" if pd.notna(x) else x)
        elif "mrn" in field_lower or "medical_record" in field_lower:
            return series.apply(lambda x: "MRN_MASKED" if pd.notna(x) else x)
        else:
            # Generic PHI masking
            return series.apply(lambda x: "MASKED_PHI" if pd.notna(x) else x)
    
    def _calculate_risk_score(
        self,
        pii_results: Dict[str, List],
        phi_results: Dict[str, List]
    ) -> str:
        """Calculate overall privacy risk score"""
        
        pii_count = len(pii_results["fields"])
        phi_count = len(phi_results["fields"])
        
        # Calculate weighted risk
        total_sensitive_fields = pii_count + phi_count
        
        if total_sensitive_fields == 0:
            return "NONE"
        elif total_sensitive_fields <= 2:
            return "LOW"
        elif total_sensitive_fields <= 5:
            return "MEDIUM"
        elif total_sensitive_fields <= 10:
            return "HIGH"
        else:
            return "CRITICAL"
