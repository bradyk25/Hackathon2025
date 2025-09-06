# Development Guide

## Overview
This guide provides comprehensive instructions for setting up, developing, and deploying the Ghostwriter synthetic healthcare data generation system.

## Prerequisites

### System Requirements
- **Node.js**: 18.0.0 or higher
- **Python**: 3.9 or higher
- **npm**: 8.0.0 or higher
- **Git**: Latest version

### Development Tools (Recommended)
- **VS Code** with extensions:
  - Python
  - TypeScript and JavaScript Language Features
  - React snippets
  - Prettier
  - ESLint

## Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd ghostwriter-hackathon

# Copy environment file
cp .env.example .env

# Install all dependencies
npm run setup
```

### 2. Configure Environment
Edit `.env` file with your API keys and settings:
```bash
# External APIs
LISTENING_POST_API_KEY=your_listening_post_api_key_here
DBTWIN_API_KEY=your_dbtwin_api_key_here

# Privacy Settings
PII_DETECTION_THRESHOLD=0.8
PHI_DETECTION_THRESHOLD=0.9

# Development
DEBUG=true
```

### 3. Start Development Servers
```bash
# Start all services (frontend, backend, python pipeline)
npm run dev

# Or start individually:
npm run frontend:dev    # React app on :3000
npm run backend:dev     # Node.js API on :3001
npm run python:dev      # Python pipeline on :8001
```

## Project Architecture

### Frontend (React + TypeScript)
- **Location**: `/frontend`
- **Port**: 3000
- **Tech Stack**: React 18, TypeScript, Tailwind CSS, Axios
- **Key Features**:
  - Chat interface for user interaction
  - File upload with drag-and-drop
  - Real-time processing status
  - Interactive reports and visualizations

### Backend (Node.js + Express)
- **Location**: `/backend`
- **Port**: 3001
- **Tech Stack**: Express, SQLite, Socket.IO, Multer
- **Key Features**:
  - RESTful API endpoints
  - File upload handling
  - Session management
  - WebSocket for real-time updates

### Data Pipeline (Python + FastAPI)
- **Location**: `/data-pipeline`
- **Port**: 8001
- **Tech Stack**: FastAPI, Pandas, Scikit-learn, Presidio
- **Key Features**:
  - File parsing and data extraction
  - Schema inference
  - Privacy analysis (PII/PHI detection)
  - Synthetic data generation
  - Quality validation

## Development Workflow

### Frontend Development

#### File Structure
```
frontend/src/
├── components/          # Reusable UI components
│   ├── ChatInterface/   # Chat functionality
│   ├── FileUpload/      # File upload components
│   ├── Reports/         # Report visualization
│   └── Layout/          # Layout components
├── hooks/               # Custom React hooks
├── services/            # API communication
├── types/               # TypeScript definitions
└── utils/               # Utility functions
```

#### Key Commands
```bash
cd frontend

# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build

# Lint code
npm run lint
```

#### Adding New Components
1. Create component directory in `src/components/`
2. Add TypeScript interface in `src/types/`
3. Export from component's `index.ts`
4. Add to main app routing if needed

### Backend Development

#### File Structure
```
backend/src/
├── controllers/         # Request handlers
├── middleware/          # Express middleware
├── routes/             # API route definitions
├── services/           # Business logic
├── models/             # Data models
├── utils/              # Utility functions
└── config/             # Configuration
```

#### Key Commands
```bash
cd backend

# Start development server
npm run dev

# Run tests
npm test

# Lint code
npm run lint
```

#### Adding New API Endpoints
1. Define route in `src/routes/`
2. Create controller in `src/controllers/`
3. Add middleware if needed
4. Update API documentation in `docs/API.md`

### Python Pipeline Development

#### File Structure
```
data-pipeline/src/
├── ingestion/          # Data parsing and schema inference
├── privacy/            # PII/PHI detection and masking
├── synthesis/          # Synthetic data generation
├── validation/         # Quality metrics and validation
└── utils/              # Configuration and utilities
```

#### Key Commands
```bash
cd data-pipeline

# Start FastAPI server
python -m uvicorn main:app --reload --port 8001

# Run tests
python -m pytest

# Format code
black src/

# Lint code
flake8 src/
```

#### Adding New Processing Modules
1. Create module in appropriate directory
2. Add to `__init__.py` exports
3. Update main pipeline in `main.py`
4. Add tests in `tests/`

## API Integration

### Listening Post Integration
The chatbot functionality integrates with Listening Post API:

```python
# Example usage in backend
const listeningPost = new ListeningPostService(
    process.env.LISTENING_POST_API_KEY
);

const response = await listeningPost.sendMessage(
    userMessage,
    sessionContext
);
```

### DBTwin Integration
Synthetic data generation uses DBTwin API:

```python
# Example usage in Python pipeline
synthesizer = DBTwinSynthesizer(
    api_key=settings.dbtwin_api_key,
    api_url=settings.dbtwin_api_url
)

synthetic_data = await synthesizer.generate_synthetic_data(
    clean_data=cleaned_data,
    schema=schema_info,
    multiplier=10
)
```

## Testing

### Frontend Testing
```bash
cd frontend
npm test                    # Run all tests
npm test -- --coverage     # Run with coverage
npm test -- --watch        # Watch mode
```

### Backend Testing
```bash
cd backend
npm test                    # Run all tests
npm run test:coverage       # Run with coverage
npm run test:watch          # Watch mode
```

### Python Testing
```bash
cd data-pipeline
python -m pytest                    # Run all tests
python -m pytest --cov=src         # Run with coverage
python -m pytest -v                # Verbose output
```

### Integration Testing
```bash
# Run full integration tests
npm run test

# Test specific workflows
npm run test:e2e
```

## Code Quality

### Linting and Formatting

#### Frontend (TypeScript/React)
```bash
cd frontend
npm run lint        # ESLint
npm run lint:fix    # Auto-fix issues
```

#### Backend (JavaScript)
```bash
cd backend
npm run lint        # ESLint
npm run lint:fix    # Auto-fix issues
```

#### Python
```bash
cd data-pipeline
black src/          # Format code
flake8 src/         # Lint code
mypy src/           # Type checking
```

### Pre-commit Hooks
Set up pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Debugging

### Frontend Debugging
- Use React Developer Tools browser extension
- Enable debug mode: `DEBUG=true` in `.env`
- Check browser console for errors
- Use VS Code debugger with launch configuration

### Backend Debugging
- Use VS Code debugger with Node.js configuration
- Enable debug logging: `LOG_LEVEL=debug`
- Check server logs in `logs/app.log`
- Use Postman/Insomnia for API testing

### Python Debugging
- Use VS Code Python debugger
- Add breakpoints in code
- Use `pdb` for command-line debugging
- Check FastAPI automatic docs at `http://localhost:8001/docs`

## Performance Optimization

### Frontend
- Use React.memo for expensive components
- Implement virtual scrolling for large lists
- Optimize bundle size with code splitting
- Use service workers for caching

### Backend
- Implement request caching
- Use database connection pooling
- Add request rate limiting
- Optimize file upload handling

### Python Pipeline
- Use async/await for I/O operations
- Implement data streaming for large files
- Cache expensive computations
- Use multiprocessing for CPU-intensive tasks

## Deployment

### Development Deployment
```bash
# Build all components
npm run build

# Start production servers
npm start
```

### Production Deployment
1. Set production environment variables
2. Build optimized bundles
3. Configure reverse proxy (nginx)
4. Set up SSL certificates
5. Configure monitoring and logging

### Docker Deployment
```bash
# Build Docker images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Kill process on port
npx kill-port 3000
npx kill-port 3001
npx kill-port 8001
```

#### Python Dependencies
```bash
# Reinstall Python dependencies
cd data-pipeline
pip install -r requirements.txt --force-reinstall
```

#### Node Dependencies
```bash
# Clear npm cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

#### Database Issues
```bash
# Reset SQLite database
rm -f backend/data/ghostwriter.db
# Restart backend to recreate
```

### Getting Help
1. Check the logs in `logs/` directory
2. Review API documentation in `docs/API.md`
3. Check GitHub issues
4. Contact team members

## Contributing

### Code Style
- Follow existing code patterns
- Write meaningful commit messages
- Add tests for new features
- Update documentation

### Pull Request Process
1. Create feature branch from `main`
2. Make changes with tests
3. Update documentation
4. Submit pull request
5. Address review feedback

### Hackathon Guidelines
- Focus on MVP features first
- Document assumptions and limitations
- Prioritize working demo over perfect code
- Collaborate effectively with team members
