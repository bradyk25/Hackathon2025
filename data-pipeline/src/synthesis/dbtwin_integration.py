"""
DBTwin API integration for synthetic data generation.
This module handles communication with the DBTwin API to generate synthetic healthcare data.
"""

import asyncio
import httpx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

from ..utils.config import get_settings

logger = logging.getLogger(__name__)

class DBTwinSynthesizer:
    """
    Synthetic data generator using DBTwin API.
    Handles the generation of privacy-safe synthetic healthcare data.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.dbtwin_api_key
        self.api_url = api_url or self.settings.dbtwin_api_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
        
        if not self.api_key:
            logger.warning("DBTwin API key not provided. Using fallback synthetic generation.")
    
    async def generate_synthetic_data(
        self,
        clean_data: List[Dict[str, Any]],
        schema: Dict[str, Any],
        multiplier: int = 10,
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic data using DBTwin API or fallback method.
        
        Args:
            clean_data: Cleaned original data
            schema: Inferred schema information
            multiplier: Number of synthetic records to generate per original record
            random_seed: Random seed for reproducibility
            
        Returns:
            List of synthetic data records
        """
        
        if self.api_key and self.api_url:
            try:
                return await self._generate_with_dbtwin_api(clean_data, schema, multiplier, random_seed)
            except Exception as e:
                logger.error(f"DBTwin API failed: {e}. Falling back to local generation.")
                return await self._generate_with_fallback(clean_data, schema, multiplier, random_seed)
        else:
            logger.info("Using fallback synthetic data generation")
            return await self._generate_with_fallback(clean_data, schema, multiplier, random_seed)
    
    async def _generate_with_dbtwin_api(
        self,
        clean_data: List[Dict[str, Any]],
        schema: Dict[str, Any],
        multiplier: int,
        random_seed: int
    ) -> List[Dict[str, Any]]:
        """Generate synthetic data using DBTwin API"""
        
        # Prepare API request
        request_payload = {
            "data": clean_data,
            "schema": schema,
            "generation_config": {
                "multiplier": multiplier,
                "random_seed": random_seed,
                "preserve_distributions": True,
                "preserve_correlations": True,
                "privacy_level": "high"
            },
            "output_format": "json"
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Make API request
        response = await self.client.post(
            f"{self.api_url}/generate",
            json=request_payload,
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"DBTwin API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if "synthetic_data" not in result:
            raise Exception("Invalid response format from DBTwin API")
        
        return result["synthetic_data"]
    
    async def _generate_with_fallback(
        self,
        clean_data: List[Dict[str, Any]],
        schema: Dict[str, Any],
        multiplier: int,
        random_seed: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback synthetic data generation using statistical methods.
        This is used when DBTwin API is not available.
        """
        
        if not clean_data:
            return []
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(clean_data)
        
        # Calculate target number of records
        target_records = len(clean_data) * multiplier
        
        synthetic_records = []
        
        # Generate synthetic records
        for _ in range(target_records):
            synthetic_record = {}
            
            for field_name, field_info in schema.get("fields", {}).items():
                if field_name not in df.columns:
                    continue
                
                field_type = field_info.get("type", "string")
                
                if field_type == "integer":
                    synthetic_record[field_name] = self._generate_synthetic_integer(
                        df[field_name], field_info
                    )
                elif field_type == "float":
                    synthetic_record[field_name] = self._generate_synthetic_float(
                        df[field_name], field_info
                    )
                elif field_type == "categorical":
                    synthetic_record[field_name] = self._generate_synthetic_categorical(
                        df[field_name], field_info
                    )
                elif field_type == "datetime":
                    synthetic_record[field_name] = self._generate_synthetic_datetime(
                        df[field_name], field_info
                    )
                else:  # string or other
                    synthetic_record[field_name] = self._generate_synthetic_string(
                        df[field_name], field_info
                    )
            
            synthetic_records.append(synthetic_record)
        
        return synthetic_records
    
    def _generate_synthetic_integer(self, series: pd.Series, field_info: Dict) -> int:
        """Generate synthetic integer value"""
        
        # Remove null values for statistics
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return 0
        
        # Use field constraints if available
        min_val = field_info.get("min", clean_series.min())
        max_val = field_info.get("max", clean_series.max())
        
        # Generate based on distribution
        distribution = field_info.get("distribution", "uniform")
        
        if distribution == "normal":
            mean = clean_series.mean()
            std = clean_series.std()
            value = np.random.normal(mean, std)
            return int(np.clip(value, min_val, max_val))
        else:
            return np.random.randint(min_val, max_val + 1)
    
    def _generate_synthetic_float(self, series: pd.Series, field_info: Dict) -> float:
        """Generate synthetic float value"""
        
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return 0.0
        
        min_val = field_info.get("min", clean_series.min())
        max_val = field_info.get("max", clean_series.max())
        
        distribution = field_info.get("distribution", "uniform")
        
        if distribution == "normal":
            mean = clean_series.mean()
            std = clean_series.std()
            value = np.random.normal(mean, std)
            return float(np.clip(value, min_val, max_val))
        else:
            return np.random.uniform(min_val, max_val)
    
    def _generate_synthetic_categorical(self, series: pd.Series, field_info: Dict) -> str:
        """Generate synthetic categorical value"""
        
        # Get categories and their frequencies
        categories = field_info.get("categories", series.unique().tolist())
        frequencies = field_info.get("frequencies")
        
        if frequencies and len(frequencies) == len(categories):
            # Use provided frequencies
            return np.random.choice(categories, p=frequencies)
        else:
            # Use observed frequencies
            value_counts = series.value_counts(normalize=True)
            available_categories = [cat for cat in categories if cat in value_counts.index]
            
            if available_categories:
                probabilities = [value_counts[cat] for cat in available_categories]
                return np.random.choice(available_categories, p=probabilities)
            else:
                return np.random.choice(categories) if categories else "unknown"
    
    def _generate_synthetic_datetime(self, series: pd.Series, field_info: Dict) -> str:
        """Generate synthetic datetime value"""
        
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return datetime.now().isoformat()
        
        # Convert to datetime if not already
        try:
            dt_series = pd.to_datetime(clean_series)
            min_date = dt_series.min()
            max_date = dt_series.max()
            
            # Generate random datetime between min and max
            time_range = (max_date - min_date).total_seconds()
            random_seconds = np.random.uniform(0, time_range)
            synthetic_date = min_date + pd.Timedelta(seconds=random_seconds)
            
            return synthetic_date.isoformat()
        except:
            return datetime.now().isoformat()
    
    def _generate_synthetic_string(self, series: pd.Series, field_info: Dict) -> str:
        """Generate synthetic string value"""
        
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return "synthetic_value"
        
        # Check if there's a pattern
        pattern = field_info.get("pattern")
        
        if pattern:
            # Generate based on pattern (simplified)
            return self._generate_from_pattern(pattern)
        else:
            # Sample from existing values with some modification
            sample_value = np.random.choice(clean_series.tolist())
            return f"synthetic_{sample_value}"
    
    def _generate_from_pattern(self, pattern: str) -> str:
        """Generate string based on regex pattern (simplified)"""
        
        # This is a simplified pattern generator
        # In a full implementation, you'd use a proper regex generator
        
        if "^P[0-9]{6}$" in pattern:
            # Patient ID pattern
            return f"P{np.random.randint(100000, 999999)}"
        elif "[0-9]" in pattern:
            # Contains numbers
            return f"GEN{np.random.randint(1000, 9999)}"
        else:
            return f"synthetic_{np.random.randint(1000, 9999)}"
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
