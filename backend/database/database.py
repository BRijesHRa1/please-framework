"""
Database configuration and session management
Uses SQLite with SQLAlchemy ORM
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment or use default
# Use absolute path to ensure DB is always in backend/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "please.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Create engine
# For SQLite, we use StaticPool to handle threading better
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
else:
    # For PostgreSQL or other databases
    engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session
    Use with FastAPI Depends()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables
    Call this on application startup
    """
    from .models import Project, Cycle, AgentOutput, Task, Artifact, Report
    
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully")


def drop_all_tables():
    """
    Drop all tables - USE WITH CAUTION
    Only for testing/development
    """
    Base.metadata.drop_all(bind=engine)
    print("⚠️  All tables dropped")


def reset_db():
    """
    Reset database - drop and recreate all tables
    Only for testing/development
    """
    drop_all_tables()
    init_db()
    print("🔄 Database reset complete")

