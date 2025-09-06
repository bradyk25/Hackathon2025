# Project Structure

```
ghostwriter-hackathon/
├── README.md
├── package.json
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
│
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── API.md
│   ├── DEVELOPMENT.md
│   └── DEPLOYMENT.md
│
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface/
│   │   │   │   ├── ChatInterface.tsx
│   │   │   │   ├── MessageBubble.tsx
│   │   │   │   └── ChatInterface.css
│   │   │   ├── FileUpload/
│   │   │   │   ├── FileUpload.tsx
│   │   │   │   └── FileUpload.css
│   │   │   ├── Reports/
│   │   │   │   ├── PrivacyReport.tsx
│   │   │   │   ├── SchemaReport.tsx
│   │   │   │   ├── ValidationReport.tsx
│   │   │   │   └── Reports.css
│   │   │   └── Layout/
│   │   │       ├── Header.tsx
│   │   │       ├── Sidebar.tsx
│   │   │       └── Layout.css
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useFileUpload.ts
│   │   │   └── useAPI.ts
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── websocket.ts
│   │   │   └── fileService.ts
│   │   ├── types/
│   │   │   ├── api.ts
│   │   │   ├── chat.ts
│   │   │   └── reports.ts
│   │   ├── utils/
│   │   │   ├── formatters.ts
│   │   │   └── validators.ts
│   │   ├── App.tsx
│   │   ├── App.css
│   │   ├── index.tsx
│   │   └── index.css
│   ├── package.json
│   └── tsconfig.json
│
├── backend/
│   ├── src/
│   │   ├── controllers/
│   │   │   ├── chatController.js
│   │   │   ├── fileController.js
│   │   │   ├── processingController.js
│   │   │   └── reportController.js
│   │   ├── middleware/
│   │   │   ├── auth.js
│   │   │   ├── fileValidation.js
│   │   │   ├── errorHandler.js
│   │   │   └── rateLimiter.js
│   │   ├── routes/
│   │   │   ├── chat.js
│   │   │   ├── files.js
│   │   │   ├── processing.js
│   │   │   └── reports.js
│   │   ├── services/
│   │   │   ├── listeningPostService.js
│   │   │   ├── dbtwinService.js
│   │   │   ├── fileService.js
│   │   │   └── sessionService.js
│   │   ├── models/
│   │   │   ├── Session.js
│   │   │   ├── ProcessingJob.js
│   │   │   └── Report.js
│   │   ├── utils/
│   │   │   ├── logger.js
│   │   │   ├── validators.js
│   │   │   └── helpers.js
│   │   ├── config/
│   │   │   ├── database.js
│   │   │   ├── apis.js
│   │   │   └── environment.js
│   │   └── app.js
│   ├── uploads/
│   ├── outputs/
│   ├── temp/
│   ├── package.json
│   └── server.js
│
├── data-pipeline/
│   ├── src/
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── file_parser.py
│   │   │   ├── schema_inferrer.py
│   │   │   └── data_normalizer.py
│   │   ├── privacy/
│   │   │   ├── __init__.py
│   │   │   ├── pii_detector.py
│   │   │   ├── phi_detector.py
│   │   │   ├── masking.py
│   │   │   └── privacy_report.py
│   │   ├── synthesis/
│   │   │   ├── __init__.py
│   │   │   ├── base_synthesizer.py
│   │   │   ├── statistical_synthesizer.py
│   │   │   ├── ml_synthesizer.py
│   │   │   └── dbtwin_integration.py
│   │   ├── validation/
│   │   │   ├── __init__.py
│   │   │   ├── statistical_tests.py
│   │   │   ├── constraint_validator.py
│   │   │   └── quality_metrics.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── file_utils.py
│   │   │   ├── data_utils.py
│   │   │   └── config.py
│   │   └── main.py
│   ├── tests/
│   │   ├── test_ingestion.py
│   │   ├── test_privacy.py
│   │   ├── test_synthesis.py
│   │   └── test_validation.py
│   ├── requirements.txt
│   └── setup.py
│
├── scripts/
│   ├── setup.sh
│   ├── start-dev.sh
│   ├── build.sh
│   └── deploy.sh
│
├── tests/
│   ├── integration/
│   ├── unit/
│   └── e2e/
│
└── sample-data/
    ├── messy-healthcare-data.zip
    ├── test-cases/
    └── expected-outputs/
```

## Key Directories Explained

### `/frontend`
React TypeScript application providing the user interface:
- **Components**: Reusable UI components organized by feature
- **Hooks**: Custom React hooks for state management and API calls
- **Services**: API communication and external service integrations
- **Types**: TypeScript type definitions

### `/backend`
Node.js Express server handling API requests:
- **Controllers**: Request handling logic
- **Routes**: API endpoint definitions
- **Services**: Business logic and external API integrations
- **Models**: Data models and database schemas
- **Middleware**: Authentication, validation, error handling

### `/data-pipeline`
Python-based data processing engine:
- **Ingestion**: File parsing, schema inference, data normalization
- **Privacy**: PII/PHI detection, masking, privacy reporting
- **Synthesis**: Synthetic data generation algorithms
- **Validation**: Statistical tests and quality metrics

### `/docs`
Project documentation:
- API specifications
- Development guidelines
- Deployment instructions

### `/scripts`
Automation scripts for development and deployment

### `/tests`
Comprehensive test suites for all components

### `/sample-data`
Test datasets and expected outputs for validation
