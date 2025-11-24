"""
Database models for PLEASe Framework
Uses SQLAlchemy ORM with SQLite
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, 
    ForeignKey, JSON, BigInteger, Boolean
)
from sqlalchemy.orm import relationship
from .database import Base


def generate_uuid():
    """Generate UUID string for SQLite compatibility"""
    return str(uuid.uuid4())


class Project(Base):
    """
    Main project container
    Each project can have multiple cycles (iterations)
    """
    __tablename__ = "projects"
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Basic Info
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Specification Sheet (original input)
    spec_sheet = Column(JSON, nullable=False)
    
    # Status Tracking
    status = Column(String(50), nullable=False, default='initialized', index=True)
    # Status values: 'initialized', 'running', 'completed', 'failed', 'paused'
    
    current_cycle_number = Column(Integer, default=1)
    total_cycles_planned = Column(Integer, default=1)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Metadata
    tags = Column(JSON, default=list)  # Array of tags as JSON
    
    # Relationships
    cycles = relationship("Cycle", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name={self.name}, status={self.status})>"


class Cycle(Base):
    """
    PLEASe cycle/iteration tracking
    Each project can have multiple cycles
    """
    __tablename__ = "cycles"
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Foreign Key
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Cycle Info
    cycle_number = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False, default='initialized', index=True)
    # Status: 'initialized', 'planning', 'learning', 'executing', 'assessing', 'reporting', 'completed', 'failed'
    
    # State Progress (0.00 to 1.00)
    current_state = Column(Float, default=0.00, index=True)
    # 0.00 = Start, 0.10 = Planner, 0.20 = Learner, 0.30 = Executor, 0.40 = Assessor, 1.00 = Complete
    
    current_agent = Column(String(50), nullable=True)  # 'planner', 'learner', 'executor', 'assessor', 'pm'
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Duration tracking (in seconds)
    total_duration_seconds = Column(Integer, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="cycles")
    agent_outputs = relationship("AgentOutput", back_populates="cycle", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="cycle", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="cycle", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Cycle(id={self.id}, project_id={self.project_id}, cycle_number={self.cycle_number}, state={self.current_state})>"


class AgentOutput(Base):
    """
    Store each agent's output with version control
    Tracks: Planner, Learner, Executor, Assessor, PM outputs
    """
    __tablename__ = "agent_outputs"
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Foreign Key
    cycle_id = Column(String(36), ForeignKey("cycles.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Agent Info
    agent_name = Column(String(50), nullable=False, index=True)
    # Values: 'planner', 'learner', 'executor', 'assessor', 'pm'
    
    state_value = Column(Float, nullable=False, index=True)
    # 0.1 (planner), 0.2 (learner), 0.3 (executor), 0.4 (assessor), 1.0 (pm)
    
    # Version Control
    version = Column(Integer, default=1, nullable=False)
    is_current = Column(Boolean, default=True, nullable=False, index=True)
    
    # Output Data
    output_data = Column(JSON, nullable=False)
    summary = Column(Text, nullable=True)  # Brief summary (≤500 tokens)
    
    # Status
    status = Column(String(50), default='completed', nullable=False)
    # Values: 'in_progress', 'completed', 'failed'
    
    # Performance Metrics
    execution_time_seconds = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    api_calls_made = Column(Integer, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    cycle = relationship("Cycle", back_populates="agent_outputs")
    tasks = relationship("Task", back_populates="agent_output", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<AgentOutput(id={self.id}, agent={self.agent_name}, version={self.version}, current={self.is_current})>"


class Task(Base):
    """
    Individual tasks from Planner agent
    Tracks execution status and results
    """
    __tablename__ = "tasks"
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Foreign Key
    agent_output_id = Column(String(36), ForeignKey("agent_outputs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Task Info
    task_id = Column(String(50), nullable=False, index=True)  # T1, T2, T3, etc.
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Dependencies (stored as JSON array)
    dependencies = Column(JSON, default=list)  # ['T1', 'T2']
    
    # Resources
    gpu_hours = Column(Float, default=0)
    priority = Column(String(20), nullable=True)  # 'high', 'medium', 'low'
    
    # Status
    status = Column(String(50), default='pending', nullable=False, index=True)
    # Values: 'pending', 'in_progress', 'completed', 'failed', 'skipped'
    
    # Results (filled in by Executor)
    result_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    agent_output = relationship("AgentOutput", back_populates="tasks")
    
    def __repr__(self):
        return f"<Task(id={self.id}, task_id={self.task_id}, name={self.name}, status={self.status})>"


class Artifact(Base):
    """
    Files/models/data generated during execution
    Stores file paths (not binary data)
    """
    __tablename__ = "artifacts"
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Foreign Keys
    cycle_id = Column(String(36), ForeignKey("cycles.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_output_id = Column(String(36), ForeignKey("agent_outputs.id", ondelete="SET NULL"), nullable=True)
    
    # Artifact Info
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False, index=True)
    # Types: 'model', 'dataset', 'plot', 'log', 'report', 'code', 'other'
    
    # Storage (filesystem path)
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    mime_type = Column(String(100), nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    meta_data = Column(JSON, nullable=True)  # Additional metadata (renamed from metadata to avoid SQLAlchemy conflict)
    
    # Source tracking
    generated_by_agent = Column(String(50), nullable=True, index=True)
    generated_at_state = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    cycle = relationship("Cycle", back_populates="artifacts")
    
    def __repr__(self):
        return f"<Artifact(id={self.id}, name={self.name}, type={self.type})>"


class Report(Base):
    """
    Final comprehensive reports from PM agent
    """
    __tablename__ = "reports"
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # Foreign Key
    cycle_id = Column(String(36), ForeignKey("cycles.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Report Content
    content = Column(Text, nullable=False)  # Full markdown report
    format = Column(String(20), default='markdown')  # 'markdown', 'html', 'pdf', 'json'
    
    # Scores
    final_nih_score = Column(Integer, nullable=True, index=True)  # 1-9 NIH scoring
    final_bimodal_score = Column(Integer, nullable=True)  # Bimodal scoring
    
    # A.G.E. Scores (stored as JSON)
    age_scores = Column(JSON, nullable=True)
    # Example: {"planner": {"achievement": 6.5, "effort": 7}, ...}
    
    # Report Metadata
    executive_summary = Column(Text, nullable=True)
    recommendations = Column(JSON, nullable=True)  # Array of recommendation strings
    
    # Version tracking (if report is regenerated)
    version = Column(Integer, default=1)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    cycle = relationship("Cycle", back_populates="reports")
    
    def __repr__(self):
        return f"<Report(id={self.id}, cycle_id={self.cycle_id}, nih_score={self.final_nih_score})>"
