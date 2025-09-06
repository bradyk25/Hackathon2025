# Ghostwriter: Synthetic Healthcare Data Generator

## Mission
Build a chatbot-style service that learns the structure and distributions of messy healthcare datasets and generates larger, privacy-safe synthetic datasets without leaking sensitive information.

## Project Overview
This hackathon project transforms messy, inconsistent healthcare data into trustworthy, privacy-safe synthetic datasets through an agentic experience.

## Tech Stack
- **Frontend**: React.js with TypeScript
- **Backend**: Node.js with Express
- **Chatbot**: Listening Post integration
- **Synthetic Data Generation**: dbtwin API
- **Data Processing**: Python with pandas, numpy
- **Privacy Detection**: Custom PII/PHI detection algorithms
- **File Handling**: multer for zip uploads/downloads
- **Database**: SQLite for session management
- **Validation**: Custom statistical validation suite

## Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │  Data Pipeline  │
│   (React)       │◄──►│   (Node.js)      │◄──►│   (Python)      │
│                 │    │                  │    │                 │
│ - Chat Interface│    │ - API Routes     │    │ - Schema Infer  │
│ - File Upload   │    │ - File Handling  │    │ - PII Detection │
│ - Reports View  │    │ - Session Mgmt   │    │ - Synthesis     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  External APIs   │
                       │                  │
                       │ - Listening Post │
                       │ - dbtwin API     │
                       └──────────────────┘
```

## Development Timeline (Weekend Hackathon)

### Friday Evening
- [x] Project setup and structure
- [ ] Basic frontend scaffolding
- [ ] Backend API foundation
- [ ] Data ingestion pipeline setup

### Saturday Morning
- [ ] Schema inference implementation
- [ ] PII/PHI detection system
- [ ] Basic synthetic data generation
- [ ] File processing pipeline

### Saturday Evening
- [ ] Validation and metrics system
- [ ] Report generation
- [ ] Frontend-backend integration
- [ ] Basic chatbot integration

### Sunday Morning
- [ ] Testing and debugging
- [ ] UI/UX improvements
- [ ] Documentation
- [ ] Demo preparation

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.9+
- npm or yarn

### Installation
```bash
# Clone repository
git clone <repository-url>
cd ghostwriter-hackathon

# Install dependencies
npm install
pip install -r requirements.txt

# Start development servers
npm run dev
```

## Core Features

### Must-Haves (MVP)
- ✅ Live demo of agentic workflow
- ✅ Zip file upload/download
- ✅ Schema inference with JSON output
- ✅ PII/PHI detection and masking
- ✅ Distribution matching
- ✅ Deterministic reproducibility

### Nice-to-Haves (Stretch)
- [ ] Differential privacy
- [ ] Anomaly simulator
- [ ] Synthetic free-text generation
- [ ] Active schema editor

## Project Structure
See `docs/PROJECT_STRUCTURE.md` for detailed file organization.

## API Documentation
See `docs/API.md` for endpoint specifications.

## Contributing
This is a hackathon project. See `CONTRIBUTING.md` for development guidelines.

## License
MIT License - see LICENSE file for details.
