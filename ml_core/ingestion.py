"""
Data Ingestion Pipeline
Handles CSV/JSON parsing, normalization, and cleaning for messy datasets
"""

import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import chardet
import re


class DataIngester:
    """
    Robust data ingestion that handles messy real-world datasets
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.jsonl', '.tsv']
        self.common_null_values = [
            '', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', 'NA', 'na',
            'NaN', 'nan', '#N/A', '#NULL!', 'NIL', 'nil', '-', '--', '?',
            'missing', 'MISSING', 'unknown', 'UNKNOWN'
        ]
    
    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding using chardet
        """
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    def detect_delimiter(self, file_path: str, encoding: str = 'utf-8') -> str:
        """
        Auto-detect CSV delimiter by analyzing first few lines
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(1024)
                
            # Try common delimiters
            delimiters = [',', '\t', ';', '|', ':']
            delimiter_counts = {}
            
            for delimiter in delimiters:
                count = sample.count(delimiter)
                if count > 0:
                    delimiter_counts[delimiter] = count
            
            if delimiter_counts:
                return max(delimiter_counts, key=delimiter_counts.get)
            return ','  # Default fallback
            
        except Exception:
            return ','
    
    def normalize_headers(self, headers: List[str]) -> List[str]:
        """
        Clean and normalize column headers
        """
        normalized = []
        for header in headers:
            if header is None:
                header = 'unnamed_column'
            
            # Convert to string and strip whitespace
            header = str(header).strip()
            
            # Replace spaces and special chars with underscores
            header = re.sub(r'[^\w]', '_', header)
            
            # Remove multiple underscores
            header = re.sub(r'_+', '_', header)
            
            # Remove leading/trailing underscores
            header = header.strip('_')
            
            # Convert to lowercase
            header = header.lower()
            
            # Handle empty headers
            if not header:
                header = 'unnamed_column'
                
            normalized.append(header)
        
        # Handle duplicate headers
        seen = {}
        final_headers = []
        for header in normalized:
            if header in seen:
                seen[header] += 1
                final_headers.append(f"{header}_{seen[header]}")
            else:
                seen[header] = 0
                final_headers.append(header)
        
        return final_headers
    
    def ingest_csv(self, file_path: str) -> pd.DataFrame:
        """
        Robust CSV ingestion with auto-detection of encoding and delimiter
        """
        # Detect encoding
        encoding = self.detect_encoding(file_path)
        
        # Detect delimiter
        delimiter = self.detect_delimiter(file_path, encoding)
        
        try:
            # Read CSV with detected parameters
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                na_values=self.common_null_values,
                keep_default_na=True,
                dtype=str,  # Read everything as string initially
                low_memory=False
            )
            
            # Normalize headers
            df.columns = self.normalize_headers(df.columns.tolist())
            
            return df
            
        except Exception as e:
            # Fallback: try with different parameters
            try:
                df = pd.read_csv(
                    file_path,
                    encoding='latin-1',
                    delimiter=',',
                    na_values=self.common_null_values,
                    dtype=str,
                    low_memory=False,
                    on_bad_lines='skip'
                )
                df.columns = self.normalize_headers(df.columns.tolist())
                return df
            except Exception as e2:
                raise Exception(f"Failed to read CSV file: {e2}")
    
    def ingest_json(self, file_path: str) -> pd.DataFrame:
        """
        Ingest JSON/JSONL files and convert to DataFrame
        """
        encoding = self.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Try to load as regular JSON first
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        df = pd.json_normalize(data)
                    elif isinstance(data, dict):
                        df = pd.json_normalize([data])
                    else:
                        raise ValueError("JSON must contain list or dict")
                        
                except json.JSONDecodeError:
                    # Try as JSONL (newline-delimited JSON)
                    f.seek(0)
                    records = []
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    
                    if not records:
                        raise ValueError("No valid JSON records found")
                    
                    df = pd.json_normalize(records)
            
            # Normalize headers
            df.columns = self.normalize_headers(df.columns.tolist())
            
            # Replace pandas null values with our standard nulls
            df = df.replace({np.nan: None})
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to read JSON file: {e}")
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Main ingestion method that auto-detects format and processes file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Ingest based on file type
        if file_ext in ['.csv', '.tsv']:
            df = self.ingest_csv(str(file_path))
        elif file_ext in ['.json', '.jsonl']:
            df = self.ingest_json(str(file_path))
        else:
            raise ValueError(f"Handler not implemented for: {file_ext}")
        
        # Basic cleaning
        df = self.basic_cleaning(df)
        
        # Return metadata along with dataframe
        return {
            'dataframe': df,
            'file_path': str(file_path),
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict()
        }
    
    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning operations
        """
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Strip whitespace from string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                # Convert back to None for empty strings
                df[col] = df[col].replace('', None)
                df[col] = df[col].replace('nan', None)
        
        return df


def ingest_data(file_path: str) -> Dict[str, Any]:
    """
    Convenience function for data ingestion
    """
    ingester = DataIngester()
    return ingester.ingest_file(file_path)


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        result = ingest_data(sys.argv[1])
        print(f"Ingested {result['num_rows']} rows, {result['num_columns']} columns")
        print(f"Columns: {result['columns']}")
