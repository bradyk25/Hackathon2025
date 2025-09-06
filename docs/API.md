# API Documentation

## Overview
The Ghostwriter API provides endpoints for processing messy healthcare data and generating synthetic datasets through an agentic workflow.

## Base URLs
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:3001`
- Python Data Pipeline: `http://localhost:8001`

## Authentication
All API endpoints require session-based authentication. Sessions are created automatically when users start a new conversation.

## Endpoints

### Chat Interface

#### POST /api/chat/message
Send a message to the chatbot interface.

**Request:**
```json
{
  "message": "Hello, I want to process some healthcare data",
  "sessionId": "session-uuid",
  "context": {
    "currentStep": "initial",
    "uploadedFiles": []
  }
}
```

**Response:**
```json
{
  "response": "Hello! I'm here to help you process your healthcare data safely. Please upload your zip file to get started.",
  "sessionId": "session-uuid",
  "nextActions": ["file_upload"],
  "timestamp": "2025-01-05T21:30:00Z"
}
```

#### GET /api/chat/history/:sessionId
Retrieve chat history for a session.

**Response:**
```json
{
  "messages": [
    {
      "id": "msg-1",
      "type": "user",
      "content": "Hello, I want to process some healthcare data",
      "timestamp": "2025-01-05T21:30:00Z"
    },
    {
      "id": "msg-2",
      "type": "assistant",
      "content": "Hello! I'm here to help...",
      "timestamp": "2025-01-05T21:30:01Z"
    }
  ],
  "sessionId": "session-uuid"
}
```

### File Management

#### POST /api/files/upload
Upload a zip file containing messy healthcare data.

**Request:** Multipart form data
- `file`: ZIP file (max 100MB)
- `sessionId`: Session identifier

**Response:**
```json
{
  "fileId": "file-uuid",
  "filename": "healthcare-data.zip",
  "size": 1024000,
  "uploadedAt": "2025-01-05T21:30:00Z",
  "status": "uploaded",
  "sessionId": "session-uuid"
}
```

#### GET /api/files/download/:fileId
Download processed synthetic data as a zip file.

**Response:** Binary zip file with synthetic dataset

#### GET /api/files/status/:fileId
Check processing status of uploaded file.

**Response:**
```json
{
  "fileId": "file-uuid",
  "status": "processing",
  "progress": 65,
  "currentStep": "privacy_analysis",
  "estimatedTimeRemaining": "2 minutes"
}
```

### Data Processing

#### POST /api/processing/start
Start processing pipeline for uploaded data.

**Request:**
```json
{
  "fileId": "file-uuid",
  "sessionId": "session-uuid",
  "options": {
    "syntheticMultiplier": 10,
    "enableDifferentialPrivacy": false,
    "privacyThreshold": 0.8,
    "randomSeed": 42
  }
}
```

**Response:**
```json
{
  "jobId": "job-uuid",
  "status": "started",
  "estimatedDuration": "5-10 minutes",
  "steps": [
    "file_extraction",
    "schema_inference",
    "privacy_analysis",
    "data_synthesis",
    "validation",
    "report_generation"
  ]
}
```

#### GET /api/processing/status/:jobId
Get processing job status and progress.

**Response:**
```json
{
  "jobId": "job-uuid",
  "status": "processing",
  "currentStep": "privacy_analysis",
  "progress": 40,
  "completedSteps": ["file_extraction", "schema_inference"],
  "logs": [
    {
      "timestamp": "2025-01-05T21:30:00Z",
      "level": "info",
      "message": "Schema inference completed: 15 fields detected"
    }
  ]
}
```

### Reports

#### GET /api/reports/privacy/:jobId
Get privacy analysis report.

**Response:**
```json
{
  "jobId": "job-uuid",
  "privacyReport": {
    "piiDetected": {
      "count": 3,
      "fields": ["patient_name", "ssn", "email"],
      "confidence": [0.95, 0.98, 0.87]
    },
    "phiDetected": {
      "count": 5,
      "fields": ["diagnosis", "medication", "provider_id", "visit_date", "insurance_id"],
      "confidence": [0.92, 0.89, 0.94, 0.85, 0.91]
    },
    "maskingApplied": {
      "pii": ["patient_name", "ssn", "email"],
      "phi": ["diagnosis", "medication", "provider_id"]
    },
    "riskScore": "LOW"
  }
}
```

#### GET /api/reports/schema/:jobId
Get inferred schema report.

**Response:**
```json
{
  "jobId": "job-uuid",
  "schemaReport": {
    "fields": [
      {
        "name": "patient_id",
        "type": "string",
        "nullable": false,
        "unique": true,
        "pattern": "^P[0-9]{6}$"
      },
      {
        "name": "age",
        "type": "integer",
        "nullable": false,
        "min": 0,
        "max": 120,
        "distribution": "normal"
      },
      {
        "name": "diagnosis_code",
        "type": "categorical",
        "nullable": true,
        "categories": ["A01", "B02", "C03"],
        "frequencies": [0.4, 0.35, 0.25]
      }
    ],
    "constraints": [
      {
        "type": "foreign_key",
        "field": "provider_id",
        "references": "providers.id"
      }
    ],
    "recordCount": 1000,
    "qualityScore": 0.85
  }
}
```

#### GET /api/reports/validation/:jobId
Get validation report comparing original and synthetic data.

**Response:**
```json
{
  "jobId": "job-uuid",
  "validationReport": {
    "distributionComparison": {
      "age": {
        "ksTest": 0.023,
        "pValue": 0.89,
        "passed": true
      },
      "diagnosis_code": {
        "chiSquare": 2.45,
        "pValue": 0.65,
        "passed": true
      }
    },
    "correlationPreservation": {
      "age_diagnosis": {
        "original": 0.34,
        "synthetic": 0.32,
        "difference": 0.02,
        "passed": true
      }
    },
    "constraintViolations": {
      "count": 0,
      "violations": []
    },
    "overallScore": 0.92,
    "passed": true
  }
}
```

### Sessions

#### POST /api/sessions/create
Create a new session.

**Response:**
```json
{
  "sessionId": "session-uuid",
  "createdAt": "2025-01-05T21:30:00Z",
  "expiresAt": "2025-01-06T21:30:00Z"
}
```

#### GET /api/sessions/:sessionId
Get session information.

**Response:**
```json
{
  "sessionId": "session-uuid",
  "createdAt": "2025-01-05T21:30:00Z",
  "lastActivity": "2025-01-05T21:35:00Z",
  "status": "active",
  "uploadedFiles": ["file-uuid-1"],
  "processingJobs": ["job-uuid-1"]
}
```

## Python Data Pipeline API

### POST /process
Main processing endpoint for the Python pipeline.

**Request:**
```json
{
  "file_path": "/uploads/healthcare-data.zip",
  "options": {
    "synthetic_multiplier": 10,
    "privacy_threshold": 0.8,
    "random_seed": 42
  }
}
```

**Response:**
```json
{
  "job_id": "job-uuid",
  "status": "processing",
  "steps": [
    "extraction",
    "inference",
    "privacy_check",
    "synthesis",
    "validation"
  ]
}
```

### GET /status/{job_id}
Get processing status from Python pipeline.

### GET /results/{job_id}
Get processing results and download links.

## Error Responses

All endpoints return errors in the following format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "File size exceeds maximum limit",
    "details": {
      "maxSize": "100MB",
      "actualSize": "150MB"
    }
  },
  "timestamp": "2025-01-05T21:30:00Z"
}
```

## Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `413` - Payload Too Large
- `422` - Unprocessable Entity
- `500` - Internal Server Error

## Rate Limiting
- Chat messages: 60 requests per minute
- File uploads: 5 requests per minute
- Processing jobs: 3 concurrent jobs per session
