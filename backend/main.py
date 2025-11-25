"""
PLEASe Framework - FastAPI Main Application
Research Automation Pipeline API
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from database import init_db
from api.routes import router


# ============================================================================
# LIFESPAN HANDLER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler for startup and shutdown events
    """
    # Startup
    print("🚀 Starting PLEASe Framework API...")
    print("   Initializing database...")
    init_db()
    print("   ✅ Database ready")
    print("   ✅ API ready at http://localhost:8000")
    print("   📖 Docs at http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    print("👋 Shutting down PLEASe Framework API...")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="PLEASe Framework API",
    description="""
## PLEASe Framework - Research Automation Pipeline

A multi-agent framework for automating ML research workflows.

### Features:
- **Submit Spec Sheets**: Define research problems and goals
- **Agent Pipeline**: Automated Planner → Learner → Executor → Assessor → PM
- **Reports & Dashboard**: View results, A.G.E. scores, and recommendations

### Agents:
- 🤖 **Planner**: Creates task breakdown and timeline
- 🧠 **Learner**: Gathers domain knowledge and resources
- ⚙️ **Executor**: Generates and runs ML code
- 📊 **Assessor**: Evaluates results with NIH scoring
- 📋 **PM**: Generates comprehensive reports with A.G.E. scores
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# CORS MIDDLEWARE
# ============================================================================

# Allow frontend origins
origins = [
    "http://localhost:3000",      # React/Next.js default
    "http://localhost:5173",      # Vite default
    "http://localhost:5174",      # Vite alternate
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# INCLUDE ROUTERS
# ============================================================================

app.include_router(router)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "PLEASe Framework API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "dashboard": "/api/dashboard",
            "projects": "/api/projects",
            "spec_sheets": "/api/spec-sheets",
            "health": "/api/health"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("PLEASe Framework API Server")
    print("="*60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
