# Sample Data Handling Guide

## Overview
This document explains how the Ghostwriter system handles the provided hackathon dataset, including specific data quality issues and processing strategies.

## Sample Dataset Analysis

### Data Structure
The hackathon dataset (`hackathon dataset.xlsx`) contains mixed healthcare data with the following characteristics:

**Claims Data Fields:**
- `CLAIM_ID`: Unique claim identifiers (IQV200000, CMS300000 formats)
- `MEMBER_ID`: Patient/member identifiers (M2000, PT1001 formats)
- `CLAIM_TYPE`: Inpatient, Outpatient, Pharmacy
- `ADMISSION_TYPE`: Emergency, Elective, Urgent, Newborn
- `PROVIDER_ZIP`: Provider ZIP codes
- `PROVIDER_SPECIALTY`: Oncology, Orthopedics, Pediatrics, Cardiology, General Practice
- `DIAGNOSIS_CODE`: ICD codes (C50.9, M16.0, J45.909, etc.)
- `PROCEDURE_CODE`: Medical procedure codes
- `TOTAL_CHARGE`: Claim amounts
- `COPAY`: Patient copayment amounts
- `COINSURANCE`: Coinsurance percentages (0.0-1.0)
- `ADJUDICATION_STATUS`: Pending, Paid, Denied

**Patient Data Fields:**
- `ID`: Patient identifiers (PT1001, PT1002, etc.)
- `LAST_NAME`: Patient surnames
- `AGE`: Patient ages (67-90)
- `GENDER`: M/F gender codes
- `DRG_CODE`: Diagnosis Related Group codes (291, 292, 293)
- `ICD9_PROCEDURE`: Procedure amounts
- `CLAIM_COST`: Total claim costs
- `STATUS`: PAID/DENIED status

### Data Quality Issues Identified

#### 1. Mixed Data Types
- **Issue**: Single dataset contains both claims and patient records
- **Solution**: `HealthcareDataHandler` separates data types based on field patterns
- **Implementation**: 
  ```python
  claims_data, patient_data = await self._separate_data_types(df_cleaned)
  ```

#### 2. Typos and Inconsistencies
- **Issue**: "Emergny" instead of "Emergency"
- **Solution**: Automated typo correction
- **Implementation**:
  ```python
  self.known_issues = {
      'typos': {'Emergny': 'Emergency', 'Urgent': 'Urgent'}
  }
  ```

#### 3. Missing Values
- **Issue**: "N/A" strings, empty cells
- **Solution**: Standardized missing value handling
- **Implementation**:
  ```python
  'missing_values': ['N/A', '', 'NULL', 'null']
  ```

#### 4. Duplicate Records
- **Issue**: Record IQV200005 appears twice
- **Solution**: Duplicate detection and flagging
- **Implementation**: Pandas duplicate detection with reporting

#### 5. Invalid Data Ranges
- **Issue**: Some monetary values may be negative or unrealistic
- **Solution**: Range validation with healthcare-specific constraints
- **Implementation**:
  ```python
  # Validate monetary amounts (should be positive)
  money_fields = ['TOTAL_CHARGE', 'COPAY', 'CLAIM_COST']
  for field in money_fields:
      if field in df_validated.columns:
          mask = df_validated[field] >= 0
          df_validated.loc[~mask, field] = np.nan
  ```

## Processing Pipeline for Sample Data

### Step 1: File Parsing
```python
# Excel file parsing with multiple encodings
parser = FileParser()
extracted_data = await parser.parse_zip_file(file_path)
```

**Handles:**
- Excel format (.xlsx)
- Mixed data types in single sheet
- Column name standardization

### Step 2: Healthcare Data Processing
```python
healthcare_handler = HealthcareDataHandler()
processed_data = await healthcare_handler.process_healthcare_data(extracted_data)
```

**Processes:**
- Data type separation (claims vs. patient records)
- Field validation and cleaning
- Healthcare-specific pattern recognition
- Data quality scoring

### Step 3: Schema Inference
```python
schema_inferrer = SchemaInferrer()
schema_report = await schema_inferrer.infer_schema(combined_data)

# Enhanced with healthcare hints
healthcare_schema_hints = healthcare_handler.get_healthcare_schema_hints()
```

**Infers:**
- Data types (integer, float, categorical, string)
- Value ranges and constraints
- Healthcare field categories
- Statistical distributions

### Step 4: Privacy Analysis
```python
privacy_analyzer = PrivacyAnalyzer()
privacy_report = await privacy_analyzer.analyze_privacy(combined_data, schema_report)
```

**Detects:**
- PII: Patient names, member IDs
- PHI: Diagnosis codes, medical procedures, provider information
- Risk assessment and masking strategies

### Step 5: Synthetic Data Generation
```python
synthesizer = DBTwinSynthesizer()
synthetic_data = await synthesizer.generate_synthetic_data(
    clean_data=privacy_report["cleaned_data"],
    schema=schema_report,
    multiplier=10
)
```

**Generates:**
- 10x synthetic records (500+ from ~50 original)
- Preserves statistical distributions
- Maintains field correlations
- Ensures privacy compliance

### Step 6: Quality Validation
```python
validator = QualityValidator()
validation_report = await validator.validate_synthetic_data(
    original_data=privacy_report["cleaned_data"],
    synthetic_data=synthetic_data,
    schema=schema_report
)
```

**Validates:**
- Distribution similarity (KS-test, Chi-square)
- Correlation preservation
- Constraint compliance
- Privacy preservation

## Expected Processing Results

### Data Separation
- **Claims Records**: ~30 records with claim-specific fields
- **Patient Records**: ~20 records with patient-specific fields
- **Mixed Records**: Handled by duplication to both datasets

### Data Quality Improvements
- **Typo Correction**: "Emergny" → "Emergency"
- **Missing Value Standardization**: "N/A" → null
- **Duplicate Detection**: 1 duplicate record flagged
- **Range Validation**: Invalid ages/amounts nullified

### Privacy Analysis Results
**Expected PII Detection:**
- `LAST_NAME`: High confidence (0.95+)
- `MEMBER_ID`: Medium confidence (0.8+)
- `ID`: Medium confidence (0.8+)

**Expected PHI Detection:**
- `DIAGNOSIS_CODE`: High confidence (0.9+)
- `PROCEDURE_CODE`: High confidence (0.9+)
- `PROVIDER_SPECIALTY`: High confidence (0.85+)
- `CLAIM_TYPE`: Medium confidence (0.8+)

### Synthetic Data Output
**Generated Records:** 500+ synthetic records
**File Formats:** CSV and JSON in ZIP archive
**Privacy Compliance:** No direct copying of sensitive fields
**Statistical Fidelity:** Distribution preservation validated

## Testing the System

### 1. Prepare Test Data
```bash
# Place the hackathon dataset.xlsx in uploads directory
mkdir -p uploads
cp "hackathon dataset.xlsx" uploads/test_data.zip
```

### 2. Start the Pipeline
```bash
cd data-pipeline
python -m uvicorn main:app --reload --port 8001
```

### 3. Submit Processing Job
```bash
curl -X POST "http://localhost:8001/process" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "uploads/test_data.zip",
    "options": {
      "synthetic_multiplier": 10,
      "random_seed": 42
    }
  }'
```

### 4. Monitor Progress
```bash
# Use returned job_id
curl "http://localhost:8001/status/{job_id}"
```

### 5. Retrieve Results
```bash
curl "http://localhost:8001/results/{job_id}"
```

## Expected Output Structure

```json
{
  "privacy_report": {
    "pii_detected": {
      "count": 3,
      "fields": ["LAST_NAME", "MEMBER_ID", "ID"],
      "confidence": [0.95, 0.85, 0.82]
    },
    "phi_detected": {
      "count": 6,
      "fields": ["DIAGNOSIS_CODE", "PROCEDURE_CODE", "PROVIDER_SPECIALTY", "CLAIM_TYPE", "DRG_CODE", "ICD9_PROCEDURE"],
      "confidence": [0.92, 0.90, 0.87, 0.83, 0.81, 0.79]
    },
    "risk_score": "MEDIUM"
  },
  "healthcare_processing_summary": {
    "total_records": 50,
    "claims_records": 30,
    "patient_records": 20,
    "duplicate_records": 1,
    "data_quality_score": 0.85,
    "issues_found": [
      "Missing values standardized",
      "1 duplicate records found",
      "Typos corrected"
    ]
  },
  "synthetic_data_info": {
    "record_count": 500,
    "original_count": 50,
    "claims_records": 30,
    "patient_records": 20,
    "multiplier": 10.0
  }
}
```

## Troubleshooting

### Common Issues

#### 1. File Format Problems
**Symptom**: "Unsupported file format" error
**Solution**: Ensure file is in ZIP format or supported Excel format

#### 2. Missing Healthcare Fields
**Symptom**: Low data quality scores
**Solution**: Verify column names match expected healthcare fields

#### 3. Privacy Detection Issues
**Symptom**: No PII/PHI detected
**Solution**: Check field naming conventions and data patterns

#### 4. Synthetic Data Generation Fails
**Symptom**: Empty synthetic dataset
**Solution**: Verify cleaned data has sufficient records and valid schema

### Performance Considerations

- **File Size**: Optimized for files up to 100MB
- **Record Count**: Efficient processing up to 10,000 records
- **Processing Time**: ~2-5 minutes for sample dataset
- **Memory Usage**: ~500MB peak for sample dataset

## Validation Metrics

The system provides comprehensive validation metrics:

- **Distribution Similarity**: KS-test p-values > 0.05
- **Correlation Preservation**: Difference < 0.1
- **Constraint Compliance**: 100% adherence to schema constraints
- **Privacy Preservation**: < 10% exact value matches
- **Overall Quality Score**: Target > 0.7 (70%)

This comprehensive handling ensures the hackathon dataset is processed accurately while maintaining privacy and generating high-quality synthetic data suitable for healthcare analytics and research.
