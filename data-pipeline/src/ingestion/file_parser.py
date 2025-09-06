"""
File parsing and data extraction for healthcare datasets.
Handles various file formats including CSV, Excel, JSON, and ZIP archives.
"""

import os
import zipfile
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Union
import tempfile
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class FileParser:
    """
    Parses various file formats and extracts healthcare data.
    Supports CSV, Excel, JSON, and ZIP archives containing multiple files.
    """
    
    def __init__(self):
        self.supported_extensions = {'.csv', '.xlsx', '.xls', '.json', '.zip'}
        self.max_sample_size = 10000  # Maximum records to sample for analysis
    
    async def parse_zip_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a ZIP file containing healthcare data files.
        
        Args:
            file_path: Path to the ZIP file
            
        Returns:
            List of parsed data records
        """
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        all_data = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process all extracted files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path_full = os.path.join(root, file)
                    file_ext = Path(file).suffix.lower()
                    
                    if file_ext in self.supported_extensions and file_ext != '.zip':
                        try:
                            logger.info(f"Processing file: {file}")
                            file_data = await self._parse_single_file(file_path_full, file_ext)
                            
                            if file_data:
                                # Add metadata about source file
                                for record in file_data:
                                    record['_source_file'] = file
                                
                                all_data.extend(file_data)
                                logger.info(f"Extracted {len(file_data)} records from {file}")
                        
                        except Exception as e:
                            logger.error(f"Error processing file {file}: {str(e)}")
                            continue
        
        logger.info(f"Total records extracted: {len(all_data)}")
        return all_data
    
    async def _parse_single_file(self, file_path: str, file_ext: str) -> List[Dict[str, Any]]:
        """Parse a single file based on its extension"""
        
        try:
            if file_ext == '.csv':
                return await self._parse_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return await self._parse_excel(file_path)
            elif file_ext == '.json':
                return await self._parse_json(file_path)
            else:
                logger.warning(f"Unsupported file extension: {file_ext}")
                return []
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return []
    
    async def _parse_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse CSV file"""
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Could not decode CSV file with any supported encoding")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle common data issues
            df = self._clean_dataframe(df)
            
            # Sample if too large
            if len(df) > self.max_sample_size:
                df = df.sample(n=self.max_sample_size, random_state=42)
                logger.info(f"Sampled {self.max_sample_size} records from large CSV")
            
            return df.to_dict('records')
        
        except Exception as e:
            logger.error(f"Error parsing CSV {file_path}: {str(e)}")
            return []
    
    async def _parse_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse Excel file"""
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            all_data = []
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Clean column names
                    df.columns = df.columns.str.strip()
                    
                    # Handle common data issues
                    df = self._clean_dataframe(df)
                    
                    # Add sheet information
                    sheet_data = df.to_dict('records')
                    for record in sheet_data:
                        record['_sheet_name'] = sheet_name
                    
                    all_data.extend(sheet_data)
                    logger.info(f"Extracted {len(sheet_data)} records from sheet: {sheet_name}")
                
                except Exception as e:
                    logger.error(f"Error reading sheet {sheet_name}: {str(e)}")
                    continue
            
            # Sample if too large
            if len(all_data) > self.max_sample_size:
                import random
                all_data = random.sample(all_data, self.max_sample_size)
                logger.info(f"Sampled {self.max_sample_size} records from large Excel file")
            
            return all_data
        
        except Exception as e:
            logger.error(f"Error parsing Excel {file_path}: {str(e)}")
            return []
    
    async def _parse_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse JSON file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                json_data = data
            elif isinstance(data, dict):
                # Single object or nested structure
                if 'data' in data and isinstance(data['data'], list):
                    json_data = data['data']
                elif 'records' in data and isinstance(data['records'], list):
                    json_data = data['records']
                else:
                    # Single record
                    json_data = [data]
            else:
                logger.warning(f"Unexpected JSON structure in {file_path}")
                return []
            
            # Flatten nested objects if necessary
            flattened_data = []
            for record in json_data:
                if isinstance(record, dict):
                    flattened_record = self._flatten_dict(record)
                    flattened_data.append(flattened_record)
            
            # Sample if too large
            if len(flattened_data) > self.max_sample_size:
                import random
                flattened_data = random.sample(flattened_data, self.max_sample_size)
                logger.info(f"Sampled {self.max_sample_size} records from large JSON file")
            
            return flattened_data
        
        except Exception as e:
            logger.error(f"Error parsing JSON {file_path}: {str(e)}")
            return []
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean common data issues in DataFrame"""
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [str(col).strip().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Handle duplicate column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup 
                                                             for i in range(sum(cols == dup))]
        df.columns = cols
        
        # Convert object columns to string and clean
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace 'nan' strings with actual NaN
            df[col] = df[col].replace(['nan', 'NaN', 'NULL', 'null', ''], pd.NA)
        
        return df
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # Handle list of dictionaries by taking the first item
                items.extend(self._flatten_dict(v[0], new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic information about a file"""
        
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        file_stats = os.stat(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        info = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_extension": file_ext,
            "file_size_bytes": file_stats.st_size,
            "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "supported": file_ext in self.supported_extensions,
            "modified_time": file_stats.st_mtime
        }
        
        return info
