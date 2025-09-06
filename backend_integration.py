#!/usr/bin/env python3
"""
Backend integration for Ghostwriter UI
Handles file uploads and processing
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
import zipfile
import pandas as pd
import tempfile
import uuid
from datetime import datetime
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Import our processing functions
import sys
sys.path.append('.')
from process_and_clean import detect_and_clean_healthcare_data, save_cleaned_data

# Storage for processed files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Serve the main HTML interface"""
    try:
        return send_file('ghostwriter_ui_connected.html')
    except FileNotFoundError:
        return jsonify({
            "service": "Ghostwriter Backend",
            "status": "running",
            "version": "1.0.0",
            "error": "HTML interface not found"
        })

@app.route('/api/status')
def api_status():
    """API endpoint for status check"""
    return jsonify({
        "service": "Ghostwriter Backend",
        "status": "running",
        "version": "1.0.0"
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file (use basename to avoid path issues)
        base_filename = os.path.basename(file.filename)
        filename = f"{job_id}_{base_filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process the file
        try:
            print(f"🔄 Processing new file: {base_filename}")
            print(f"📁 Job ID: {job_id}")
            
            # Extract and read data
            data = extract_and_read_data(filepath)
            print(f"📊 Extracted {len(data)} records from uploaded file")
            
            # Process with our privacy protection
            results = detect_and_clean_healthcare_data(data)
            print(f"✅ Privacy processing complete for job {job_id}")
            
            # Separate and save claims and demographics data
            claims_data, demographics_data = separate_claims_and_demographics(results['cleaned_data'])
            
            # Save separated files
            claims_csv = os.path.join(OUTPUT_FOLDER, f"{job_id}_claims.csv")
            demographics_csv = os.path.join(OUTPUT_FOLDER, f"{job_id}_demographics.csv")
            
            save_cleaned_data(claims_data, claims_csv)
            save_cleaned_data(demographics_data, demographics_csv)
            
            # Save privacy report
            report_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_report.json")
            with open(report_file, 'w') as f:
                json.dump({
                    'pii_detected': results['pii_detected'],
                    'phi_detected': results['phi_detected'],
                    'processing_summary': {
                        'job_id': job_id,
                        'original_records': len(results['original_data']),
                        'cleaned_records': len(results['cleaned_data']),
                        'claims_count': results['claims_count'],
                        'patient_count': results['patient_count'],
                        'processed_at': datetime.utcnow().isoformat()
                    }
                }, f, indent=2)
            
            # Clean data for JSON serialization (handle NaN, N/A, etc.)
            cleaned_preview = clean_data_for_json(results['cleaned_data'][:5])
            
            # Return processing results
            return jsonify({
                'job_id': job_id,
                'status': 'completed',
                'recordCount': len(results['cleaned_data']),
                'piiFieldsProtected': len(results['pii_detected']),
                'phiFieldsIdentified': len(results['phi_detected']),
                'cleanedData': cleaned_preview,
                'privacyReport': {
                    'pii_detected': results['pii_detected'],
                    'phi_detected': results['phi_detected']
                }
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/download/<job_id>/claims')
def download_claims(job_id):
    """Download insurance claims CSV file"""
    try:
        claims_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_claims.csv")
        if os.path.exists(claims_file):
            return send_file(claims_file, 
                           as_attachment=True, 
                           download_name='insurance_claims.csv',
                           mimetype='text/csv')
        else:
            return jsonify({'error': 'Claims file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<job_id>/demographics')
def download_demographics(job_id):
    """Download patient demographics CSV file"""
    try:
        demographics_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_demographics.csv")
        if os.path.exists(demographics_file):
            return send_file(demographics_file, 
                           as_attachment=True, 
                           download_name='patient_demographics.csv',
                           mimetype='text/csv')
        else:
            return jsonify({'error': 'Demographics file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<job_id>/csv')
def download_csv(job_id):
    """Download cleaned CSV file (legacy endpoint)"""
    try:
        csv_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_cleaned.csv")
        if os.path.exists(csv_file):
            return send_file(csv_file, 
                           as_attachment=True, 
                           download_name='cleaned_healthcare_data.csv',
                           mimetype='text/csv')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<job_id>/report')
def download_report(job_id):
    """Download privacy report"""
    try:
        report_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_report.json")
        if os.path.exists(report_file):
            return send_file(report_file, 
                           as_attachment=True, 
                           download_name='privacy_report.json',
                           mimetype='application/json')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<job_id>')
def show_results(job_id):
    """Serve results page with analytics and expanded privacy report"""
    try:
        # Check if job exists by looking for report file
        report_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_report.json")
        if not os.path.exists(report_file):
            return jsonify({'error': 'Job not found'}), 404
        
        # Load the HTML template and inject the job ID only
        try:
            with open('ghostwriter_ui_connected.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Inject job ID and auto-load script
            html_content = html_content.replace(
                '<script>',
                f'''<script>
                // Auto-load results for this job
                window.jobId = "{job_id}";
                
                // Auto-show results when page loads
                document.addEventListener('DOMContentLoaded', function() {{
                    if (window.jobId) {{
                        // Fetch results data from API and display
                        fetch('/api/results/' + window.jobId)
                            .then(response => response.json())
                            .then(data => {{
                                if (data.error) {{
                                    console.error('Error loading results:', data.error);
                                }} else {{
                                    showResults(data);
                                }}
                            }})
                            .catch(error => {{
                                console.error('Error fetching results:', error);
                            }});
                    }}
                }});
                '''
            )
            
            return html_content
            
        except FileNotFoundError:
            return jsonify({
                'job_id': job_id,
                'error': 'HTML template not found'
            })
            
    except Exception as e:
        return jsonify({'error': f'Failed to load results: {str(e)}'}), 500

@app.route('/api/results/<job_id>')
def get_results_data(job_id):
    """API endpoint to get results data for a specific job"""
    try:
        # Load report data
        report_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_report.json")
        if not os.path.exists(report_file):
            return jsonify({'error': 'Job not found'}), 404
        
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        # Load sample cleaned data for preview
        claims_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_claims.csv")
        demographics_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_demographics.csv")
        
        cleaned_data = []
        if os.path.exists(claims_file):
            df = pd.read_csv(claims_file)
            cleaned_data.extend(df.head(5).to_dict('records'))
        if os.path.exists(demographics_file):
            df = pd.read_csv(demographics_file)
            cleaned_data.extend(df.head(5).to_dict('records'))
        
        # Clean data for JSON serialization
        cleaned_preview = clean_data_for_json(cleaned_data)
        
        # Generate mock analytics data for demonstration
        analytics_data = {
            'data_quality': {
                'completeness': 85,
                'accuracy': 92,
                'consistency': 88,
                'validity': 90
            },
            'ml_insights': {
                'claims_distribution': {'high_cost': 25, 'medium_cost': 45, 'low_cost': 30},
                'risk_categories': {'high_risk': 15, 'medium_risk': 35, 'low_risk': 50},
                'provider_analysis': {'specialty_a': 40, 'specialty_b': 35, 'specialty_c': 25}
            }
        }
        
        return jsonify({
            'job_id': job_id,
            'status': 'completed',
            'recordCount': report_data.get('processing_summary', {}).get('cleaned_records', 0),
            'piiFieldsProtected': len(report_data.get('pii_detected', [])),
            'phiFieldsIdentified': len(report_data.get('phi_detected', [])),
            'cleanedData': cleaned_preview,
            'privacyReport': {
                'pii_detected': report_data.get('pii_detected', []),
                'phi_detected': report_data.get('phi_detected', [])
            },
            'analytics': analytics_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

@app.route('/download/<job_id>/zip')
def download_zip(job_id):
    """Download complete package as ZIP file"""
    try:
        import zipfile
        from io import BytesIO
        
        # Create a BytesIO object to store the ZIP file in memory
        zip_buffer = BytesIO()
        
        # Create ZIP file
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add claims file if it exists
            claims_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_claims.csv")
            if os.path.exists(claims_file):
                zip_file.write(claims_file, 'insurance_claims.csv')
            
            # Add demographics file if it exists
            demographics_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_demographics.csv")
            if os.path.exists(demographics_file):
                zip_file.write(demographics_file, 'patient_demographics.csv')
            
            # Add privacy report if it exists
            report_file = os.path.join(OUTPUT_FOLDER, f"{job_id}_report.json")
            if os.path.exists(report_file):
                zip_file.write(report_file, 'privacy_report.json')
            
            # Add README file with instructions
            readme_content = f"""# Ghostwriter Healthcare Data Package

This ZIP file contains your processed healthcare data with privacy protection applied.

## Files Included:

1. **insurance_claims.csv** - Insurance claims data with PII/PHI protection
2. **patient_demographics.csv** - Patient demographic data with privacy safeguards
3. **privacy_report.json** - Detailed privacy analysis and compliance report

## Privacy Protection Applied:

- PII (Personally Identifiable Information) fields have been protected
- PHI (Protected Health Information) fields have been identified and secured
- Original CLAIM_ID and Patient ID preserved for data linking
- HIPAA compliance maintained throughout processing

## Data Usage:

- Claims and demographics files can be linked using CLAIM_ID
- All privacy-sensitive fields have been appropriately masked or anonymized
- Data maintains statistical utility for analysis while ensuring privacy protection

Generated by Ghostwriter Healthcare Data Privacy Processor
Processing Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
Job ID: {job_id}
"""
            zip_file.writestr('README.txt', readme_content)
        
        # Prepare the ZIP file for download
        zip_buffer.seek(0)
        
        return send_file(
            BytesIO(zip_buffer.read()),
            as_attachment=True,
            download_name=f'ghostwriter_healthcare_data_{job_id[:8]}.zip',
            mimetype='application/zip'
        )
        
    except Exception as e:
        return jsonify({'error': f'ZIP creation failed: {str(e)}'}), 500

def separate_claims_and_demographics(data):
    """Separate mixed healthcare data into claims and demographics files"""
    claims_data = []
    demographics_data = []
    
    # Define fields for each file type with proper ordering
    # Claims: CLAIM_ID first, MEMBER_ID second, payment info at end with ADJUDICATION_STATUS last
    claims_fields = [
        'CLAIM_ID', 'MEMBER_ID', 'CLAIM_TYPE', 'ADMISSION_TYPE', 'PROVIDER_SPECIALTY',
        'PROVIDER_ZIP', 'DIAGNOSIS_CODE', 'PROCEDURE_CODE', 'COPAY', 'COINSURANCE', 
        'TOTAL_CHARGE', 'ADJUDICATION_STATUS'
    ]
    
    # Demographics: CLAIM_ID first, ID (Patient ID) second
    demographics_fields = [
        'CLAIM_ID', 'ID', 'LAST_NAME', 'AGE', 'GENDER', 'DRG_CODE',
        'ICD9_PROCEDURE', 'CLAIM_COST', 'STATUS'
    ]
    
    # Process records in original order to maintain row integrity
    for record in data:
        # Check if this is a claims record (has MEMBER_ID) or demographics record (has ID)
        if record.get('MEMBER_ID') and not pd.isna(record.get('MEMBER_ID')):
            # This is a claims record - preserve ALL original data in the row
            claims_record = {}
            for field in claims_fields:
                if field in record:
                    claims_record[field] = record[field]
                else:
                    claims_record[field] = None
            claims_data.append(claims_record)
            
        elif record.get('ID') and not pd.isna(record.get('ID')):
            # This is a demographics record - preserve ALL original data in the row
            demographics_record = {}
            for field in demographics_fields:
                if field in record:
                    demographics_record[field] = record[field]
                else:
                    demographics_record[field] = None
            demographics_data.append(demographics_record)
    
    return claims_data, demographics_data

def clean_data_for_json(data):
    """Clean data for JSON serialization by handling NaN, N/A, None values"""
    import math
    
    cleaned_data = []
    for record in data:
        cleaned_record = {}
        for key, value in record.items():
            # Handle various problematic values
            if pd.isna(value) or value is None:
                cleaned_record[key] = None
            elif isinstance(value, float) and math.isnan(value):
                cleaned_record[key] = None
            elif str(value).upper() in ['N/A', 'NA', 'NULL', 'NONE', '']:
                cleaned_record[key] = None
            else:
                cleaned_record[key] = value
        cleaned_data.append(cleaned_record)
    
    return cleaned_data

def extract_and_read_data(filepath):
    """Extract data from uploaded file"""
    data = []
    
    if filepath.endswith('.zip'):
        # Handle ZIP files
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                # Skip directories and hidden files
                if file_info.is_dir() or file_info.filename.startswith('.') or file_info.filename.startswith('__MACOSX'):
                    continue
                    
                if file_info.filename.endswith('.csv'):
                    try:
                        with zip_ref.open(file_info.filename) as csv_file:
                            df = pd.read_csv(csv_file)
                            if not df.empty:
                                data.extend(df.to_dict('records'))
                    except Exception as e:
                        print(f"Error reading {file_info.filename}: {e}")
                        continue
                        
                elif file_info.filename.endswith('.xlsx'):
                    try:
                        with zip_ref.open(file_info.filename) as excel_file:
                            df = pd.read_excel(excel_file)
                            if not df.empty:
                                data.extend(df.to_dict('records'))
                    except Exception as e:
                        print(f"Error reading {file_info.filename}: {e}")
                        continue
                        
    elif filepath.endswith('.csv'):
        # Handle direct CSV files with error handling for malformed data
        try:
            df = pd.read_csv(filepath)
        except pd.errors.ParserError:
            # Try with error_bad_lines=False for pandas < 1.3 or on_bad_lines='skip' for newer versions
            try:
                df = pd.read_csv(filepath, on_bad_lines='skip')
            except TypeError:
                # Fallback for older pandas versions
                df = pd.read_csv(filepath, error_bad_lines=False, warn_bad_lines=True)
        data = df.to_dict('records')
        
    elif filepath.endswith('.xlsx'):
        # Handle Excel files
        df = pd.read_excel(filepath)
        data = df.to_dict('records')
    else:
        raise ValueError("Unsupported file format. Please upload CSV, XLSX, or ZIP files.")
    
    if not data:
        raise ValueError("No data found in uploaded file. Please ensure your ZIP contains CSV or XLSX files with data.")
    
    return data

if __name__ == '__main__':
    print("🏥 Starting Ghostwriter Backend Server...")
    print("📡 Server will be available at: http://localhost:5000")
    print("🔗 Upload endpoint: http://localhost:5000/upload")
    app.run(debug=True, host='0.0.0.0', port=5000)
