"""
Main FastAPI application for the Ghostwriter data processing pipeline.
Handles file processing, privacy analysis, schema inference, and synthetic data generation.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import os
import json
import asyncio
from datetime import datetime

from src.ingestion.file_parser import FileParser
from src.ingestion.schema_inferrer import SchemaInferrer
from src.ingestion.healthcare_data_handler import HealthcareDataHandler
from src.privacy.privacy_report import PrivacyAnalyzer
from src.synthesis.dbtwin_integration import DBTwinSynthesizer
from src.validation.quality_metrics import QualityValidator
from src.utils.config import get_settings

# Initialize FastAPI app
app = FastAPI(
    title="Ghostwriter Data Pipeline",
    description="Privacy-safe synthetic healthcare data generation pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings
settings = get_settings()

# In-memory job storage (in production, use Redis or database)
jobs: Dict[str, Dict[str, Any]] = {}

# Request/Response Models
class ProcessingRequest(BaseModel):
    file_path: str
    options: Optional[Dict[str, Any]] = {}

class ProcessingResponse(BaseModel):
    job_id: str
    status: str
    steps: List[str]

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    current_step: str
    logs: List[Dict[str, Any]]
    results: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Ghostwriter Data Pipeline",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/process", response_model=ProcessingResponse)
async def start_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Start processing a healthcare data file"""
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    jobs[job_id] = {
        "id": job_id,
        "status": "started",
        "progress": 0,
        "current_step": "initialization",
        "logs": [],
        "file_path": request.file_path,
        "options": request.options,
        "created_at": datetime.utcnow().isoformat(),
        "results": {}
    }
    
    # Define processing steps
    steps = [
        "file_extraction",
        "schema_inference", 
        "privacy_analysis",
        "data_synthesis",
        "validation",
        "report_generation"
    ]
    
    # Start background processing
    background_tasks.add_task(process_data_pipeline, job_id, request.file_path, request.options)
    
    return ProcessingResponse(
        job_id=job_id,
        status="started",
        steps=steps
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get processing job status"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_step=job["current_step"],
        logs=job["logs"],
        results=job.get("results")
    )

@app.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get processing results"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "results": job["results"],
        "download_links": {
            "synthetic_data": f"/download/{job_id}/synthetic_data.zip",
            "privacy_report": f"/download/{job_id}/privacy_report.json",
            "schema_report": f"/download/{job_id}/schema_report.json",
            "validation_report": f"/download/{job_id}/validation_report.json"
        }
    }

async def process_data_pipeline(job_id: str, file_path: str, options: Dict[str, Any]):
    """Main data processing pipeline"""
    
    try:
        # Update job status
        update_job_status(job_id, "processing", 10, "file_extraction", "Starting file extraction...")
        
        # Step 1: File Extraction and Parsing
        parser = FileParser()
        extracted_data = await parser.parse_zip_file(file_path)
        
        update_job_status(job_id, "processing", 20, "healthcare_processing", "Processing healthcare data...")
        
        # Step 1.5: Healthcare-specific data processing
        healthcare_handler = HealthcareDataHandler()
        processed_data = await healthcare_handler.process_healthcare_data(extracted_data)
        
        # Use the combined data for further processing
        combined_data = processed_data['claims_data'] + processed_data['patient_data']
        
        update_job_status(job_id, "processing", 30, "schema_inference", "Inferring data schema...")
        
        # Step 2: Schema Inference with healthcare hints
        schema_inferrer = SchemaInferrer()
        schema_report = await schema_inferrer.infer_schema(combined_data)
        
        # Enhance schema with healthcare-specific hints
        healthcare_schema_hints = healthcare_handler.get_healthcare_schema_hints()
        for field_name, field_info in schema_report.get("fields", {}).items():
            if field_name in healthcare_schema_hints:
                field_info.update(healthcare_schema_hints[field_name])
        
        update_job_status(job_id, "processing", 45, "privacy_analysis", "Analyzing privacy risks...")
        
        # Step 3: Privacy Analysis
        privacy_analyzer = PrivacyAnalyzer()
        privacy_report = await privacy_analyzer.analyze_privacy(combined_data, schema_report)
        
        update_job_status(job_id, "processing", 60, "data_synthesis", "Generating synthetic data...")
        
        # Step 4: Synthetic Data Generation
        synthesizer = DBTwinSynthesizer(
            api_key=settings.dbtwin_api_key,
            api_url=settings.dbtwin_api_url
        )
        
        synthetic_data = await synthesizer.generate_synthetic_data(
            clean_data=privacy_report["cleaned_data"],
            schema=schema_report,
            multiplier=options.get("synthetic_multiplier", 10),
            random_seed=options.get("random_seed", 42)
        )
        
        update_job_status(job_id, "processing", 80, "validation", "Validating synthetic data...")
        
        # Step 5: Validation
        validator = QualityValidator()
        validation_report = await validator.validate_synthetic_data(
            original_data=privacy_report["cleaned_data"],
            synthetic_data=synthetic_data,
            schema=schema_report
        )
        
        update_job_status(job_id, "processing", 95, "report_generation", "Generating final reports...")
        
        # Step 6: Generate Final Results
        results = {
            "privacy_report": privacy_report,
            "schema_report": schema_report,
            "validation_report": validation_report,
            "healthcare_processing_summary": processed_data['processing_summary'],
            "synthetic_data_info": {
                "record_count": len(synthetic_data),
                "original_count": len(combined_data),
                "claims_records": len(processed_data['claims_data']),
                "patient_records": len(processed_data['patient_data']),
                "multiplier": len(synthetic_data) / len(combined_data) if len(combined_data) > 0 else 0
            }
        }
        
        # Save synthetic data to file
        output_path = f"outputs/{job_id}_synthetic_data.zip"
        await save_synthetic_data(synthetic_data, output_path)
        
        # Complete job
        jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "completed",
            "results": results,
            "completed_at": datetime.utcnow().isoformat()
        })
        
        add_job_log(job_id, "info", "Processing completed successfully")
        
    except Exception as e:
        # Handle errors
        jobs[job_id].update({
            "status": "failed",
            "current_step": "error",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        })
        
        add_job_log(job_id, "error", f"Processing failed: {str(e)}")

def update_job_status(job_id: str, status: str, progress: int, step: str, message: str):
    """Update job status and add log entry"""
    if job_id in jobs:
        jobs[job_id].update({
            "status": status,
            "progress": progress,
            "current_step": step
        })
        add_job_log(job_id, "info", message)

def add_job_log(job_id: str, level: str, message: str):
    """Add log entry to job"""
    if job_id in jobs:
        jobs[job_id]["logs"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message
        })

async def save_synthetic_data(data: List[Dict], output_path: str):
    """Save synthetic data to zip file"""
    import pandas as pd
    import zipfile
    import io
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data)
    
    # Create zip file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save as CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        zipf.writestr("synthetic_data.csv", csv_buffer.getvalue())
        
        # Save as JSON
        json_buffer = json.dumps(data, indent=2)
        zipf.writestr("synthetic_data.json", json_buffer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
