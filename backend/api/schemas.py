"""
Pydantic schemas for API request/response models
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class ProjectMetadata(BaseModel):
    """Project metadata in spec sheet"""
    project_name: str
    owner: Optional[str] = None
    created: Optional[str] = None


class ResearchProblem(BaseModel):
    """Research problem definition"""
    problem_statement: str
    success_metrics: List[str] = []
    goal_metrics: Optional[Dict[str, float]] = None


class BudgetConstraints(BaseModel):
    """Budget constraints for the project"""
    max_financial_cost: Optional[float] = None
    max_time_hours: Optional[float] = None
    max_iterations: int = 2
    reporting_period_hours: Optional[float] = None


class SpecSheetRequest(BaseModel):
    """Request schema for creating a new project with spec sheet"""
    project_metadata: ProjectMetadata
    research_problem: ResearchProblem
    data_sources: List[str] = []
    budget_constraints: Optional[BudgetConstraints] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "project_metadata": {
                    "project_name": "Survival Outcome Classification",
                    "owner": "research_lab@example.com"
                },
                "research_problem": {
                    "problem_statement": "Classify patient survival outcome from gene expression profiles",
                    "success_metrics": ["accuracy", "f1", "roc_auc"],
                    "goal_metrics": {"accuracy": 0.85, "f1": 0.80, "roc_auc": 0.75}
                },
                "data_sources": ["tcga_brca_500samples_expr_survival.csv"],
                "budget_constraints": {
                    "max_iterations": 2
                }
            }
        }
    }


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class ProjectResponse(BaseModel):
    """Response schema for project information"""
    id: str
    name: str
    description: Optional[str] = None
    status: str
    current_cycle_number: int
    total_cycles_planned: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    model_config = {"from_attributes": True}


class ProjectListResponse(BaseModel):
    """Response schema for listing projects"""
    total: int
    projects: List[ProjectResponse]


class CycleResponse(BaseModel):
    """Response schema for cycle information"""
    id: str
    cycle_number: int
    status: str
    current_state: float
    current_agent: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    model_config = {"from_attributes": True}


class AgentOutputResponse(BaseModel):
    """Response schema for agent output"""
    id: str
    agent_name: str
    state_value: float
    summary: Optional[str] = None
    status: str
    execution_time_seconds: Optional[float] = None
    output_data: Dict[str, Any] = {}
    
    model_config = {"from_attributes": True}


class GapAnalysisItem(BaseModel):
    """Single gap analysis item"""
    metric: str
    goal: float
    achieved: float
    gap: float
    status: str


class AGEScoreItem(BaseModel):
    """A.G.E. score for an agent"""
    agent_name: str
    score: float
    criteria: Dict[str, float]
    justification: str


class ReportResponse(BaseModel):
    """Response schema for project report"""
    id: str
    project_id: str
    project_name: str
    content: str  # Markdown content
    format: str
    final_nih_score: Optional[int] = None
    final_bimodal_score: Optional[int] = None
    age_scores: Optional[Dict[str, Any]] = None
    executive_summary: Optional[str] = None
    recommendations: Optional[List[str]] = None
    created_at: datetime
    
    model_config = {"from_attributes": True}


class DashboardResponse(BaseModel):
    """Response schema for dashboard overview"""
    total_projects: int
    completed_projects: int
    running_projects: int
    failed_projects: int
    recent_projects: List[ProjectResponse]
    avg_age_score: Optional[float] = None
    avg_nih_score: Optional[float] = None


class ProjectDetailResponse(BaseModel):
    """Detailed project response with all data"""
    project: ProjectResponse
    cycles: List[CycleResponse]
    agent_outputs: Dict[str, List[AgentOutputResponse]]
    report: Optional[ReportResponse] = None
    gap_analysis: Optional[List[GapAnalysisItem]] = None
    age_scores: Optional[Dict[str, AGEScoreItem]] = None


class ProjectStatusResponse(BaseModel):
    """Response for project status check"""
    project_id: str
    status: str
    current_cycle: int
    current_state: float
    current_agent: Optional[str] = None
    progress_percent: float
    message: str


class SubmitProjectResponse(BaseModel):
    """Response after submitting a new project"""
    project_id: str
    message: str
    status: str

