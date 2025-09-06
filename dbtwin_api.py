#!/usr/bin/env python3
"""
DBTwin API integration for synthetic data generation
Based on the official DBTwin API documentation
"""

import requests
import pandas as pd
import tempfile
import os
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DBTwinAPI:
    """
    DBTwin API client for generating synthetic healthcare data
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.dbtwin.com"
        
    def check_health(self) -> bool:
        """Check if DBTwin API is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200 and response.json().get("status") == "ok"
        except Exception as e:
            logger.error(f"DBTwin health check failed: {e}")
            return False
    
    def generate_synthetic_data(
        self, 
        data: List[Dict[str, Any]], 
        num_rows: int, 
        algo: str = "flagship"
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
        """
        Generate synthetic data using DBTwin API
        
        Args:
            data: List of dictionaries containing the original data
            num_rows: Number of synthetic rows to generate
            algo: Algorithm to use ("core" or "flagship")
            
        Returns:
            Tuple of (synthetic_dataframe, quality_metrics) or (None, None) if failed
        """
        
        if not self.api_key:
            logger.error("DBTwin API key not provided")
            return None, None
            
        if not data:
            logger.error("No data provided for synthesis")
            return None, None
            
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Validate minimum requirements
            if len(df) <= 40:
                logger.error("DBTwin requires more than 40 rows of data")
                return None, None
                
            if len(df.columns) <= 3:
                logger.error("DBTwin requires more than 3 columns of data")
                return None, None
            
            # Create temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_file:
                df.to_csv(temp_file, index=False)
                temp_file_path = temp_file.name
            
            try:
                # Prepare the request
                headers = {
                    "api-key": self.api_key,
                    "rows": str(num_rows),
                    "algo": algo
                }
                
                # Open file for upload
                with open(temp_file_path, 'rb') as file:
                    files = {"file": ("data.csv", file, "text/csv")}
                    
                    # Make API request
                    logger.info(f"Generating {num_rows} synthetic rows using {algo} algorithm")
                    response = requests.post(
                        f"{self.base_url}/generate",
                        headers=headers,
                        files=files,
                        timeout=300  # 5 minute timeout
                    )
                
                if response.status_code == 200:
                    # Parse synthetic data
                    synthetic_df = pd.read_csv(BytesIO(response.content))
                    
                    # Extract quality metrics from response headers
                    quality_metrics = {
                        "distribution_similarity_error": response.headers.get('distribution-similarity-error', 'N/A'),
                        "association_similarity": response.headers.get('association-similarity', 'N/A'),
                        "rows_generated": len(synthetic_df),
                        "algorithm_used": algo
                    }
                    
                    logger.info(f"Successfully generated {len(synthetic_df)} synthetic rows")
                    logger.info(f"Quality metrics: {quality_metrics}")
                    
                    return synthetic_df, quality_metrics
                    
                else:
                    # Handle API errors
                    try:
                        error_info = response.json()
                        logger.error(f"DBTwin API error {response.status_code}: {error_info}")
                    except:
                        logger.error(f"DBTwin API error {response.status_code}: {response.text}")
                    return None, None
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return None, None
    
    def generate_synthetic_claims_and_demographics(
        self,
        claims_data: List[Dict[str, Any]],
        demographics_data: List[Dict[str, Any]],
        num_rows: int,
        algo: str = "flagship"
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Generate synthetic data for both claims and demographics separately
        
        Returns:
            Tuple of (synthetic_claims_df, synthetic_demographics_df, combined_quality_metrics)
        """
        
        synthetic_claims = None
        synthetic_demographics = None
        combined_metrics = {}
        
        # Generate synthetic claims data
        if claims_data:
            logger.info("Generating synthetic claims data...")
            synthetic_claims, claims_metrics = self.generate_synthetic_data(
                claims_data, num_rows, algo
            )
            if claims_metrics:
                combined_metrics["claims"] = claims_metrics
        
        # Generate synthetic demographics data
        if demographics_data:
            logger.info("Generating synthetic demographics data...")
            synthetic_demographics, demo_metrics = self.generate_synthetic_data(
                demographics_data, num_rows, algo
            )
            if demo_metrics:
                combined_metrics["demographics"] = demo_metrics
        
        return synthetic_claims, synthetic_demographics, combined_metrics

def save_synthetic_data(df: pd.DataFrame, filename: str):
    """Save synthetic DataFrame to CSV file"""
    if df is not None and not df.empty:
        df.to_csv(filename, index=False)
        logger.info(f"Saved synthetic data to {filename}")
    else:
        logger.warning(f"No data to save to {filename}")
