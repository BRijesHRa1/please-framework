# PLEASe Framework - Complete Technical Documentation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Backend Components](#backend-components)
   - [FastAPI Application](#fastapi-application)
   - [Database Layer](#database-layer)
   - [Agent System](#agent-system)
   - [API Routes & Schemas](#api-routes--schemas)
6. [Frontend Components](#frontend-components)
7. [Multi-Agent Pipeline](#multi-agent-pipeline)
8. [Database Schema](#database-schema)
9. [API Reference](#api-reference)
10. [Configuration & Setup](#configuration--setup)
11. [Workflow Diagrams](#workflow-diagrams)
12. [Troubleshooting](#troubleshooting)

---

## Executive Summary

### What is PLEASe Framework?

**PLEASe** (Plan, Learn, Execute, Assess, Share) is a **stateful multi-agent AI framework** designed to automate biomedical research workflows. It leverages a pipeline of 5 specialized AI agents that work sequentially to:

1. **Plan** - Decompose research problems into executable tasks
2. **Learn** - Gather domain knowledge and resources using BioContext MCP
3. **Execute** - Run ML experiments with dynamically generated code
4. **Assess** - Evaluate results against goals with NIH-style scoring
5. **PM (Project Manager)** - Generate comprehensive reports with A.G.E. scores

### Core Value Proposition

- **Automated Research Pipeline**: Submit a specification sheet → Get a complete research report
- **Multi-Agent Orchestration**: 5 specialized agents with distinct responsibilities
- **Biomedical Focus**: Integration with BioContext MCP for gene/pathway information
- **Iterative Improvement**: Support for multiple cycles with recommendations
- **Comprehensive Evaluation**: NIH 1-9 scoring and A.G.E. (Agent Grading Evaluation)

### Example Use Case

```
Input: "Classify patient survival outcome from gene expression profiles with 85% accuracy using TCGA dataset"

Output: 
- Trained ML model achieving target metrics
- Gap analysis comparing goals vs achieved
- NIH score evaluation (1-9)
- A.G.E. scores for each agent
- Comprehensive Markdown research report
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        REACT FRONTEND                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ SpecSheet   │  │  Dashboard  │  │    Project Detail       │  │
│  │   Upload    │  │   Overview  │  │  (Agent Outputs/Report) │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┴────────────────┴─────────────────────┴────────────────┘
                           │ HTTP REST API
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    API Routes Layer                       │   │
│  │  /api/projects  /api/dashboard  /api/projects/{id}/report│   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────┴─────────────────────────────────┐   │
│  │                 Background Task Runner                    │   │
│  │              (run_pipeline function)                      │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────┴─────────────────────────────────┐   │
│  │                  AGENT PIPELINE                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐    │   │
│  │  │ PLANNER │→ │ LEARNER │→ │ EXECUTOR │→ │ ASSESSOR │    │   │
│  │  └────┬────┘  └────┬────┘  └────┬─────┘  └────┬─────┘    │   │
│  │       │            │            │             │           │   │
│  │       │            │            │             ▼           │   │
│  │       │            │            │       ┌──────────┐      │   │
│  │       │            │            │       │    PM    │      │   │
│  │       │            │            │       └────┬─────┘      │   │
│  │       │            │            │            │            │   │
│  │       ▼            ▼            ▼            ▼            │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │                   Ollama LLM                        │ │   │
│  │  │              (llama3.2:3b via LangChain)            │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────┴─────────────────────────────────┐   │
│  │                   DATABASE LAYER                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌───────────┐   │   │
│  │  │Projects │  │ Cycles  │  │Agent     │  │  Reports  │   │   │
│  │  │         │  │         │  │Outputs   │  │           │   │   │
│  │  └─────────┘  └─────────┘  └──────────┘  └───────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────────┐   ┌──────────┐
   │  SQLite  │     │ BioContext   │   │ Artifacts│
   │ Database │     │   MCP Server │   │Directory │
   └──────────┘     └──────────────┘   └──────────┘
```

---

## Technology Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core runtime |
| **FastAPI** | ≥0.104.1 | Web framework with async support |
| **SQLAlchemy** | ≥2.0.23 | ORM for database operations |
| **Pydantic** | ≥2.5.0 | Data validation and serialization |
| **LangChain** | ≥0.1.0 | LLM orchestration framework |
| **LangChain-Ollama** | ≥0.1.0 | Ollama integration for local LLMs |
| **Ollama** | Latest | Local LLM runtime (llama3.2:3b) |
| **MCP** | ≥1.0.0 | Model Context Protocol for BioContext |
| **Pandas** | ≥2.0.0 | Data manipulation |
| **Scikit-learn** | ≥1.3.0 | ML model training |
| **Imbalanced-learn** | ≥0.11.0 | SMOTE for class imbalance |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18+ | UI framework |
| **Vite** | Latest | Build tool and dev server |
| **TailwindCSS** | Latest | Utility-first CSS |
| **Lucide React** | Latest | Icon library |
| **React Hot Toast** | Latest | Toast notifications |
| **Axios** | Latest | HTTP client |

### Infrastructure

| Component | Description |
|-----------|-------------|
| **SQLite** | Embedded database (please.db) |
| **Ollama** | Local LLM server on port 11434 |
| **BioContext MCP** | External biomedical knowledge service |

---

## Project Structure

```
please-framework/
│
├── backend/                          # Python FastAPI Backend
│   ├── main.py                       # FastAPI application entry point
│   ├── workflow.py                   # LangGraph workflow (placeholder)
│   ├── requirements.txt              # Python dependencies
│   ├── please.db                     # SQLite database file
│   │
│   ├── agents/                       # Multi-Agent System
│   │   ├── __init__.py
│   │   ├── planner.py               # Planner Agent - Task decomposition
│   │   ├── learner.py               # Learner Agent - Resource gathering
│   │   ├── executor.py              # Executor Agent - Code generation & execution
│   │   ├── assessor.py              # Assessor Agent - Evaluation & scoring
│   │   └── pm.py                    # PM Agent - Report generation & A.G.E. scoring
│   │
│   ├── api/                          # API Layer
│   │   ├── __init__.py
│   │   ├── routes.py                # FastAPI route handlers
│   │   └── schemas.py               # Pydantic request/response models
│   │
│   ├── database/                     # Database Layer
│   │   ├── __init__.py              # Exports services
│   │   ├── database.py              # SQLAlchemy engine & session
│   │   ├── models.py                # ORM models (Project, Cycle, etc.)
│   │   ├── db_utils.py              # Database service classes
│   │   └── README.md                # Database documentation
│   │
│   ├── artifacts/                    # Generated Artifacts Storage
│   │   └── {project_id}/            # Per-project artifacts
│   │       └── {cycle_id}/          # Per-cycle artifacts
│   │           ├── code/            # Generated Python scripts
│   │           ├── data/            # Preprocessed data files
│   │           ├── models/          # Trained model files
│   │           └── results/         # Evaluation results
│   │
│   ├── dataset/                      # Input Datasets
│   │   └── tcga_brca_500samples_expr_survival.csv
│   │
│   ├── spec_sheets/                  # Spec Sheet Templates
│   │   ├── survival_classification.json
│   │   └── survival_regression.json
│   │
│   └── seeders/                      # Database Seeders
│       ├── __init__.py
│       └── clean_db.py
│
├── frontend/                         # React Frontend
│   ├── index.html                   # HTML entry point
│   ├── package.json                 # NPM dependencies
│   ├── vite.config.js               # Vite configuration
│   │
│   └── src/
│       ├── main.jsx                 # React entry point
│       ├── App.jsx                  # Main application component
│       ├── App.css                  # Global styles
│       ├── index.css                # Base styles
│       │
│       ├── api/
│       │   └── client.js            # Axios API client
│       │
│       └── components/
│           ├── Header.jsx           # Navigation header
│           ├── Dashboard.jsx        # Project dashboard view
│           ├── ProjectDetail.jsx    # Detailed project view
│           ├── ProjectStatus.jsx    # Real-time status tracking
│           ├── SpecSheetUpload.jsx  # Spec sheet form
│           └── MarkdownReport.jsx   # Markdown renderer
│
└── please_framework_spec.txt         # Original specification document
```

---

## Backend Components

### FastAPI Application

**File**: `backend/main.py`

The main FastAPI application serves as the entry point for all HTTP requests.

#### Key Features

```python
app = FastAPI(
    title="PLEASe Framework API",
    description="Multi-agent framework for automating ML research workflows",
    version="1.0.0",
    lifespan=lifespan  # Handles startup/shutdown events
)
```

#### Lifespan Handler

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    print("🚀 Starting PLEASe Framework API...")
    init_db()
    yield
    # Shutdown
    print("👋 Shutting down...")
```

#### CORS Configuration

```python
origins = [
    "http://localhost:3000",      # React default
    "http://localhost:5173",      # Vite default
    "http://localhost:5174",      # Vite alternate
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### Database Layer

**Location**: `backend/database/`

The database layer uses SQLAlchemy ORM with SQLite.

#### Configuration (`database.py`)

```python
# Database path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "please.db")
DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"

# Engine configuration for SQLite threading
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

#### Key Functions

| Function | Purpose |
|----------|---------|
| `get_db()` | Dependency injection for FastAPI routes |
| `init_db()` | Create all tables on startup |
| `reset_db()` | Drop and recreate tables (dev only) |

---

### Agent System

The heart of PLEASe is its 5-agent pipeline, each with specialized responsibilities.

---

#### 1. Planner Agent

**File**: `backend/agents/planner.py`

**Role**: Decomposes research problems into executable tasks with dependencies.

##### Input
```python
spec_sheet = {
    "research_problem": {
        "problem_statement": "Classify patient survival...",
        "goal_metric": "accuracy >= 0.85"
    },
    "resources": {"gpu_budget_hours": 4}
}
```

##### Output Structure
```python
class PlannerOutput(BaseModel):
    summary: str                      # "4 tasks identified..."
    tasks: List[Task]                 # List of executable tasks
    total_gpu_estimate: float         # Total GPU hours needed
    estimated_duration: str           # "4-6 hours"
    risk_factors: List[str]           # Potential challenges
    mcp_query: str                    # Query for BioContext MCP
```

##### Task Structure
```python
class Task(BaseModel):
    task_id: str          # "T1", "T2", etc.
    name: str             # "Data Preparation"
    description: str      # What needs to be done
    dependencies: List[str]  # ["T1"] - must complete first
    gpu_hours: float      # Estimated GPU time
    priority: str         # "high", "medium", "low"
```

##### LLM Prompting Strategy

```python
system_prompt = """You are a research planner. Create an execution plan in JSON format.
REQUIRED output structure:
{
  "summary": "overview",
  "tasks": [...],
  "total_gpu_estimate": 0.0,
  "estimated_duration": "X-Y hours",
  "risk_factors": ["risk1", "risk2"],
  "mcp_query": "What genes are responsible for [disease]?"
}"""
```

##### Key Method
```python
def plan(self, spec_sheet: Dict[str, Any]) -> Dict[str, Any]:
    """Create execution plan from specification sheet"""
    messages = [
        SystemMessage(content=self.create_system_prompt()),
        HumanMessage(content=self.create_user_prompt(spec_sheet))
    ]
    response = self.llm.invoke(messages)
    return PlannerOutput(**json.loads(response.content)).model_dump()
```

---

#### 2. Learner Agent

**File**: `backend/agents/learner.py`

**Role**: Gathers domain knowledge and research resources using BioContext MCP.

##### Input
- Specification sheet
- Planner output (including `mcp_query`)

##### Output Structure
```python
class LearnerOutput(BaseModel):
    summary: str                      # "Identified 10 key genes..."
    key_genes: List[str]              # ["BRCA1", "BRCA2", "TP53"]
    datasets: List[str]               # ["TCGA"]
    tools: List[Dict[str, str]]       # [{"name": "scikit-learn", "purpose": "ML"}]
    preprocessing_notes: str          # "Normalize data, train/test split"
    model_suggestions: List[str]      # ["RandomForest", "LogisticRegression"]
    references: List[str]             # ["Feature selection", "Cox regression"]
```

##### BioContext MCP Integration

```python
async def query_mcp():
    server_params = StdioServerParameters(
        command="uvx", 
        args=["biocontext_kb@latest"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Validate genes via KEGG
            result = await session.call_tool(
                "bc_get_kegg_id_by_gene_symbol",
                arguments={"gene_symbol": "BRCA1", "organism_code": "9606"}
            )
            
            # Get STRING interactions
            result = await session.call_tool(
                "bc_get_string_interactions",
                arguments={"protein_symbol": "BRCA1", "species": "9606", "min_score": 900}
            )
```

##### Fallback Genes
```python
def _fallback_genes(self) -> Dict[str, Any]:
    return {
        "genes": ["BRCA1", "BRCA2", "TP53", "PIK3CA", "ERBB2", 
                  "ESR1", "PGR", "PTEN", "CDH1", "ATM"],
        "pathways": ["DNA repair", "Cell cycle", "PI3K-AKT"]
    }
```

---

#### 3. Executor Agent

**File**: `backend/agents/executor.py`

**Role**: Generates and executes ML code dynamically based on Learner suggestions.

##### Key Features

1. **Template-Based Code Generation**: Uses proven code templates for reliability
2. **Dynamic Model Selection**: Uses models suggested by Learner
3. **Cycle Support**: Improved templates for Cycle 2+ with feature engineering
4. **Artifact Management**: Saves data, models, and results

##### Model Templates
```python
MODEL_TEMPLATES = {
    "RandomForestClassifier": {
        "import": "from sklearn.ensemble import RandomForestClassifier",
        "init": "RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)",
        "params": {...}
    },
    "GradientBoostingClassifier": {...},
    "LogisticRegression": {...},
    "SVC": {...},
    "KNeighborsClassifier": {...},
    "DecisionTreeClassifier": {...}
}
```

##### Code Templates

**Preprocessing Template**:
```python
CODE_TEMPLATES["preprocessing"] = '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(DATASET_PATH)
X = df[available_genes].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

np.save(data_dir / "X_train.npy", X_train)
np.save(data_dir / "X_test.npy", X_test)
...
'''
```

**Model Training Template**:
```python
CODE_TEMPLATES["model_training"] = '''
from sklearn.metrics import accuracy_score, f1_score, ...
{model_import}

model = {model_init}
model.fit(X_train_res, y_train_res)

metrics = {
    "model": "{model_name}",
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred, average='weighted')),
    ...
}
'''
```

##### Cycle 2 Improvements
- **StandardScaler**: Feature normalization
- **SelectKBest**: Feature selection (top 15 features)
- **Cross-Validation**: 5-fold stratified CV
- **SMOTE**: Class imbalance handling

##### Artifact Directory Structure
```
artifacts/
└── {project_id}/
    └── {cycle_id}/
        ├── code/
        │   ├── T1_cycle1_generated.py
        │   ├── T2_cycle1_generated.py
        │   └── ...
        ├── data/
        │   ├── X_train.npy
        │   ├── X_test.npy
        │   ├── y_train.npy
        │   ├── y_test.npy
        │   ├── feature_names.json
        │   ├── scaler.pkl
        │   └── selector.pkl
        ├── models/
        │   ├── model.pkl
        │   └── results.json
        └── results/
```

##### Output Structure
```python
class ExecutorOutput(BaseModel):
    summary: str                      # "Executed 3/4 tasks (Cycle 1)"
    cycle: int                        # Current cycle number
    tasks_completed: List[str]        # ["T1", "T2", "T3"]
    baseline_results: Dict[str, Any]  # {"accuracy": 0.82, "f1": 0.79}
    model_results: Dict[str, Any]     # Best model metrics
    artifacts_generated: List[str]    # File paths
    total_execution_time: float       # Seconds
```

---

#### 4. Assessor Agent

**File**: `backend/agents/assessor.py`

**Role**: Evaluates execution quality against goals using NIH and satisfaction scoring.

##### Scoring Scales

**NIH Scale (1-9)** - Lower is better:
| Score | Description |
|-------|-------------|
| 1 | Exceptional - Outstanding, innovative approach |
| 2 | Outstanding - Excellent execution |
| 3 | Excellent - Very good results |
| 4 | Very Good - Sound approach |
| 5 | Good - Solid work |
| 6 | Satisfactory - Acceptable |
| 7 | Fair - Below expectations |
| 8 | Marginal - Poor results |
| 9 | Poor - Fundamentally flawed |

**Satisfaction Scale (1-5)** - Higher is better:
| Score | Description |
|-------|-------------|
| 1 | Very Dissatisfied |
| 2 | Dissatisfied |
| 3 | Neutral |
| 4 | Satisfied |
| 5 | Very Satisfied |

##### Gap Analysis
```python
class GapAnalysis(BaseModel):
    metric: str           # "accuracy"
    goal: float           # 0.85
    achieved: float       # 0.87
    gap: float            # +0.02
    status: str           # "EXCEEDED", "MET", "MISSED"
    percentage_of_goal: float  # 102.35%
```

##### Status Determination
```python
def calculate_gap(self, goal, achieved, metric):
    gap = achieved - goal
    if gap >= 0.02:      # Exceeded by 2%+
        status = "EXCEEDED"
    elif gap >= -0.02:   # Within 2% of goal
        status = "MET"
    else:
        status = "MISSED"
```

##### Output Structure
```python
class AssessorOutput(BaseModel):
    satisfaction_score: int           # 1-5
    nih_score: int                    # 1-9
    gap_analysis: List[GapAnalysis]   # Per-metric analysis
    overall_status: str               # "EXCEEDED", "MET", "PARTIALLY_MET", "MISSED"
    strengths: List[str]              # What worked well
    weaknesses: List[str]             # What needs improvement
    recommendations: List[str]        # Actionable suggestions
    summary: str                      # Brief assessment
    should_continue: bool             # Whether to run more cycles
```

---

#### 5. PM Agent (Project Manager)

**File**: `backend/agents/pm.py`

**Role**: Generates comprehensive research reports with A.G.E. (Agent Grading Evaluation) scores.

##### A.G.E. Scoring Criteria

**Planner Criteria**:
- `task_clarity`: Clear and actionable task definitions
- `resource_estimation`: Accurate GPU/time estimates
- `dependency_logic`: Logical task dependencies
- `risk_identification`: Identified relevant risks

**Learner Criteria**:
- `domain_knowledge`: Relevant biomedical context gathered
- `tool_selection`: Appropriate tools/methods suggested
- `gene_relevance`: Relevant genes identified
- `preprocessing_guidance`: Useful preprocessing recommendations

**Executor Criteria**:
- `task_completion`: Tasks completed successfully
- `code_quality`: Code executed without errors
- `metric_achievement`: Achieved target metrics
- `artifact_generation`: Generated useful artifacts

**Assessor Criteria**:
- `gap_analysis`: Accurate gap analysis
- `scoring_accuracy`: Appropriate NIH/satisfaction scores
- `recommendations`: Actionable recommendations
- `improvement_tracking`: Tracked improvements across cycles

##### A.G.E. Score Structure
```python
class AGEScore(BaseModel):
    agent_name: str           # "planner", "learner", etc.
    score: float              # 1-10 scale
    criteria: Dict[str, float]  # Individual criteria scores
    justification: str        # Brief explanation
```

##### Report Generation

The PM generates a comprehensive Markdown report including:
1. Executive Summary
2. Project Overview
3. Methodology
4. Results & Analysis
5. Performance Metrics (with tables)
6. A.G.E. Agent Evaluation
7. Conclusions
8. Future Work

##### Output Structure
```python
class PMOutput(BaseModel):
    age_scores: Dict[str, AGEScore]   # Per-agent scores
    overall_age_score: float          # Average A.G.E.
    report_markdown: str              # Full Markdown report
    report_summary: str               # Executive summary
    final_nih_score: int              # From assessor
    final_satisfaction: int           # From assessor
    project_status: str               # "SUCCESS", "PARTIAL", "FAILED"
    total_cycles: int                 # Number of cycles run
    total_execution_time: float       # Total time in seconds
```

---

### API Routes & Schemas

**Location**: `backend/api/`

#### Route Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/dashboard` | GET | Dashboard statistics |
| `/api/projects` | POST | Create new project |
| `/api/projects` | GET | List all projects |
| `/api/projects/{id}` | GET | Get project details |
| `/api/projects/{id}/status` | GET | Get project status |
| `/api/projects/{id}/report` | GET | Get project report |
| `/api/projects/{id}` | DELETE | Delete project |
| `/api/spec-sheets` | GET | List spec sheet templates |

#### Request Schemas

##### SpecSheetRequest
```python
class SpecSheetRequest(BaseModel):
    project_metadata: ProjectMetadata
    research_problem: ResearchProblem
    data_sources: List[str] = []
    budget_constraints: Optional[BudgetConstraints] = None

class ProjectMetadata(BaseModel):
    project_name: str
    owner: Optional[str] = None

class ResearchProblem(BaseModel):
    problem_statement: str
    success_metrics: List[str] = []
    goal_metrics: Optional[Dict[str, float]] = None

class BudgetConstraints(BaseModel):
    max_iterations: int = 2
    max_time_hours: Optional[float] = None
```

#### Response Schemas

##### ProjectResponse
```python
class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str                    # 'initialized', 'running', 'completed', 'failed'
    current_cycle_number: int
    total_cycles_planned: int
    created_at: datetime
    updated_at: datetime
```

##### DashboardResponse
```python
class DashboardResponse(BaseModel):
    total_projects: int
    completed_projects: int
    running_projects: int
    failed_projects: int
    recent_projects: List[ProjectResponse]
    avg_age_score: Optional[float]
    avg_nih_score: Optional[float]
```

##### ProjectDetailResponse
```python
class ProjectDetailResponse(BaseModel):
    project: ProjectResponse
    cycles: List[CycleResponse]
    agent_outputs: Dict[str, List[AgentOutputResponse]]
    report: Optional[ReportResponse]
    gap_analysis: Optional[List[GapAnalysisItem]]
    age_scores: Optional[Dict[str, AGEScoreItem]]
```

---

## Frontend Components

### Main Application (`App.jsx`)

The root component managing application state and routing.

```jsx
function App() {
  const [currentProject, setCurrentProject] = useState(null);
  const [showStatus, setShowStatus] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [viewingProjectId, setViewingProjectId] = useState(null);

  return (
    <div className="min-h-screen grid-pattern">
      <Header onDashboardClick={() => setShowDashboard(true)} />
      
      {showDashboard && !viewingProjectId && (
        <Dashboard onViewProject={handleViewProject} onClose={...} />
      )}
      
      {viewingProjectId && (
        <ProjectDetail projectId={viewingProjectId} onBack={...} />
      )}
      
      <main>
        {!showStatus ? (
          <SpecSheetUpload onProjectStarted={handleProjectStarted} />
        ) : (
          <ProjectStatus project={currentProject} onComplete={...} />
        )}
      </main>
    </div>
  );
}
```

### Component Hierarchy

```
App
├── Header                    # Navigation bar
├── Dashboard                 # Project list & statistics
│   └── StatCard             # Statistics display cards
├── ProjectDetail            # Full project details
│   ├── StatusBadge          # Status indicators
│   ├── ScoreDisplay         # Score visualizations
│   └── MarkdownReport       # Report renderer
├── SpecSheetUpload          # Spec sheet form
└── ProjectStatus            # Real-time status tracking
```

### API Client (`client.js`)

```javascript
const API_BASE = 'http://localhost:8000';

export const apiClient = {
  getDashboard: () => axios.get(`${API_BASE}/api/dashboard`),
  getProjects: (status) => axios.get(`${API_BASE}/api/projects`, { params: { status }}),
  createProject: (data) => axios.post(`${API_BASE}/api/projects`, data),
  getProject: (id) => axios.get(`${API_BASE}/api/projects/${id}`),
  getProjectStatus: (id) => axios.get(`${API_BASE}/api/projects/${id}/status`),
  getProjectReport: (id) => axios.get(`${API_BASE}/api/projects/${id}/report`),
  deleteProject: (id) => axios.delete(`${API_BASE}/api/projects/${id}`),
  getSpecSheets: () => axios.get(`${API_BASE}/api/spec-sheets`)
};
```

---

## Multi-Agent Pipeline

### Pipeline Flow

```
                    ┌─────────────────────────────────────────────────┐
                    │              POST /api/projects                  │
                    │        (SpecSheet submission)                    │
                    └─────────────────────┬───────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKGROUND TASK: run_pipeline()                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        CYCLE 1 (Baseline)                            │    │
│  │                                                                      │    │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐    │    │
│  │  │ PLANNER  │ ──▶ │ LEARNER  │ ──▶ │ EXECUTOR │ ──▶ │ ASSESSOR │    │    │
│  │  │          │     │          │     │          │     │          │    │    │
│  │  │ State:   │     │ State:   │     │ State:   │     │ State:   │    │    │
│  │  │ 0.0→0.1  │     │ 0.1→0.2  │     │ 0.2→0.3  │     │ 0.3→0.4  │    │    │
│  │  └──────────┘     └──────────┘     └──────────┘     └──────────┘    │    │
│  │                                                            │         │    │
│  │                                                            ▼         │    │
│  │                                               ┌────────────────────┐ │    │
│  │                                               │ should_continue?   │ │    │
│  │                                               └─────────┬──────────┘ │    │
│  └─────────────────────────────────────────────────────────┼────────────┘    │
│                                                            │                  │
│                                      ┌─────────────────────┴───────┐         │
│                                      │                             │         │
│                              YES (continue)               NO (stop)          │
│                                      │                             │         │
│                                      ▼                             │         │
│  ┌─────────────────────────────────────────────────────┐           │         │
│  │                  CYCLE 2 (Improved)                  │           │         │
│  │                                                      │           │         │
│  │  ┌──────────────────────────┐     ┌──────────────────────────┐  │         │
│  │  │  EXECUTOR (with recs)    │ ──▶ │     ASSESSOR             │  │         │
│  │  │  - Feature engineering   │     │     - Re-evaluate        │  │         │
│  │  │  - Cross-validation      │     │     - Update scores      │  │         │
│  │  └──────────────────────────┘     └──────────────────────────┘  │         │
│  │                                                                  │         │
│  └──────────────────────────────────────────────────────────────────┘         │
│                                      │                             │         │
│                                      └──────────────┬──────────────┘         │
│                                                     │                         │
│                                                     ▼                         │
│                                    ┌─────────────────────────────────────┐   │
│                                    │              PM AGENT                │   │
│                                    │  - Calculate A.G.E. scores          │   │
│                                    │  - Generate Markdown report         │   │
│                                    │  - Save to database                 │   │
│                                    │  - State: 0.4 → 1.0                 │   │
│                                    └─────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │           Project Status: COMPLETED              │
                    │                                                  │
                    │  Available via:                                  │
                    │  - GET /api/projects/{id}                        │
                    │  - GET /api/projects/{id}/report                 │
                    └─────────────────────────────────────────────────┘
```

### State Progression

| State | Agent | Description |
|-------|-------|-------------|
| 0.0 | - | Initialized |
| 0.1 | Planner | Task planning complete |
| 0.2 | Learner | Resource gathering complete |
| 0.3 | Executor | Code execution complete |
| 0.4 | Assessor | Evaluation complete |
| 1.0 | PM | Report generated, cycle complete |

---

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────┐
│    projects     │
├─────────────────┤
│ id (PK)         │
│ name            │
│ description     │
│ spec_sheet      │──────────┐
│ status          │          │
│ current_cycle   │          │
│ created_at      │          │
│ updated_at      │          │
└────────┬────────┘          │
         │                    │
         │ 1:N               │
         ▼                    │
┌─────────────────┐          │
│     cycles      │          │
├─────────────────┤          │
│ id (PK)         │          │
│ project_id (FK) │──────────┘
│ cycle_number    │
│ status          │
│ current_state   │
│ current_agent   │
│ started_at      │
│ completed_at    │
└────────┬────────┘
         │
         │ 1:N
    ┌────┴────┬─────────────────────┐
    │         │                     │
    ▼         ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  agent_outputs  │  │   artifacts     │  │    reports      │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ id (PK)         │  │ id (PK)         │  │ id (PK)         │
│ cycle_id (FK)   │  │ cycle_id (FK)   │  │ cycle_id (FK)   │
│ agent_name      │  │ name            │  │ content         │
│ state_value     │  │ type            │  │ format          │
│ version         │  │ file_path       │  │ final_nih_score │
│ is_current      │  │ file_size_bytes │  │ final_bimodal   │
│ output_data     │  │ description     │  │ age_scores      │
│ summary         │  │ generated_by    │  │ executive_sum   │
│ status          │  │ created_at      │  │ recommendations │
│ exec_time       │  └─────────────────┘  │ created_at      │
│ started_at      │                       └─────────────────┘
│ completed_at    │
└────────┬────────┘
         │
         │ 1:N
         ▼
┌─────────────────┐
│     tasks       │
├─────────────────┤
│ id (PK)         │
│ agent_output_id │
│ task_id         │
│ name            │
│ description     │
│ dependencies    │
│ gpu_hours       │
│ priority        │
│ status          │
│ result_data     │
└─────────────────┘
```

### Model Definitions

#### Project
```python
class Project(Base):
    __tablename__ = "projects"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    spec_sheet = Column(JSON, nullable=False)
    status = Column(String(50), default='initialized', index=True)
    current_cycle_number = Column(Integer, default=1)
    total_cycles_planned = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cycles = relationship("Cycle", back_populates="project", cascade="all, delete-orphan")
```

#### Cycle
```python
class Cycle(Base):
    __tablename__ = "cycles"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"))
    cycle_number = Column(Integer, nullable=False)
    status = Column(String(50), default='initialized', index=True)
    current_state = Column(Float, default=0.00, index=True)
    current_agent = Column(String(50), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="cycles")
    agent_outputs = relationship("AgentOutput", back_populates="cycle", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="cycle", cascade="all, delete-orphan")
```

#### AgentOutput
```python
class AgentOutput(Base):
    __tablename__ = "agent_outputs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    cycle_id = Column(String(36), ForeignKey("cycles.id", ondelete="CASCADE"))
    agent_name = Column(String(50), nullable=False, index=True)  # 'planner', 'learner', etc.
    state_value = Column(Float, nullable=False, index=True)      # 0.1, 0.2, 0.3, 0.4, 1.0
    version = Column(Integer, default=1)
    is_current = Column(Boolean, default=True, index=True)
    output_data = Column(JSON, nullable=False)
    summary = Column(Text, nullable=True)
    status = Column(String(50), default='completed')
    execution_time_seconds = Column(Float, nullable=True)
```

#### Report
```python
class Report(Base):
    __tablename__ = "reports"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    cycle_id = Column(String(36), ForeignKey("cycles.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)               # Full Markdown report
    format = Column(String(20), default='markdown')
    final_nih_score = Column(Integer, nullable=True)     # 1-9
    final_bimodal_score = Column(Integer, nullable=True) # 1-5
    age_scores = Column(JSON, nullable=True)             # Per-agent scores
    executive_summary = Column(Text, nullable=True)
    recommendations = Column(JSON, nullable=True)
```

---

## API Reference

### Create Project

```http
POST /api/projects
Content-Type: application/json

{
  "project_metadata": {
    "project_name": "Survival Outcome Classification",
    "owner": "research@example.com"
  },
  "research_problem": {
    "problem_statement": "Classify patient survival outcome from gene expression profiles",
    "success_metrics": ["accuracy", "f1", "roc_auc"],
    "goal_metrics": {
      "accuracy": 0.85,
      "f1": 0.80
    }
  },
  "data_sources": ["tcga_brca_500samples_expr_survival.csv"],
  "budget_constraints": {
    "max_iterations": 2
  }
}
```

**Response** (201 Created):
```json
{
  "project_id": "550ab285-085d-4fee-8712-45f169937591",
  "message": "Project 'Survival Outcome Classification' created. Pipeline started in background.",
  "status": "running"
}
```

### Get Project Status

```http
GET /api/projects/{project_id}/status
```

**Response**:
```json
{
  "project_id": "550ab285-085d-4fee-8712-45f169937591",
  "status": "running",
  "current_cycle": 1,
  "current_state": 0.3,
  "current_agent": "executor",
  "progress_percent": 30.0,
  "message": "Executing ML pipeline..."
}
```

### Get Project Details

```http
GET /api/projects/{project_id}
```

**Response**:
```json
{
  "project": {
    "id": "550ab285-...",
    "name": "Survival Outcome Classification",
    "status": "completed",
    "current_cycle_number": 2,
    "created_at": "2025-11-25T10:30:00Z"
  },
  "cycles": [...],
  "agent_outputs": {
    "cycle_1": [...],
    "cycle_2": [...]
  },
  "report": {
    "content": "# Research Report...",
    "final_nih_score": 3,
    "final_bimodal_score": 4,
    "age_scores": {...}
  },
  "gap_analysis": [
    {
      "metric": "accuracy",
      "goal": 0.85,
      "achieved": 0.87,
      "gap": 0.02,
      "status": "EXCEEDED"
    }
  ]
}
```

### Get Dashboard

```http
GET /api/dashboard
```

**Response**:
```json
{
  "total_projects": 10,
  "completed_projects": 7,
  "running_projects": 2,
  "failed_projects": 1,
  "recent_projects": [...],
  "avg_age_score": 7.2,
  "avg_nih_score": 3.5
}
```

---

## Configuration & Setup

### Prerequisites

1. **Python 3.10+**
2. **Node.js 18+**
3. **Ollama** with `llama3.2:3b` model

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in separate terminal)
ollama serve
ollama pull llama3.2:3b

# Run the server
python main.py
# Or with uvicorn:
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Environment Variables

Create `backend/.env`:
```env
DATABASE_URL=sqlite:///./please.db
OLLAMA_HOST=http://localhost:11434
```

### Verify Installation

1. **Backend Health**: `curl http://localhost:8000/api/health`
2. **Frontend**: Open `http://localhost:5173`
3. **API Docs**: `http://localhost:8000/docs`

---

## Workflow Diagrams

### Spec Sheet Processing

```
┌──────────────────────────────────────────────────────────────┐
│                    SPEC SHEET STRUCTURE                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ project_metadata:                                    │    │
│  │   project_name: "Survival Outcome Classification"   │    │
│  │   owner: "research@lab.edu"                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ research_problem:                                    │    │
│  │   problem_statement: "Classify survival from..."    │    │
│  │   success_metrics: ["accuracy", "f1", "roc_auc"]    │    │
│  │   goal_metrics:                                     │    │
│  │     accuracy: 0.85                                  │    │
│  │     f1: 0.80                                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ data_sources:                                        │    │
│  │   - "tcga_brca_500samples_expr_survival.csv"        │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ budget_constraints:                                  │    │
│  │   max_iterations: 2                                 │    │
│  │   max_time_hours: 4                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### A.G.E. Scoring Process

```
┌──────────────────────────────────────────────────────────────┐
│                    A.G.E. SCORING FLOW                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Planner Output  │    │ Learner Output  │                 │
│  │                 │    │                 │                 │
│  │ - tasks         │    │ - key_genes     │                 │
│  │ - dependencies  │    │ - tools         │                 │
│  │ - estimates     │    │ - preprocessing │                 │
│  └────────┬────────┘    └────────┬────────┘                 │
│           │                      │                           │
│           ▼                      ▼                           │
│      ┌─────────┐           ┌─────────┐                      │
│      │ Score:  │           │ Score:  │                      │
│      │ 7.5/10  │           │ 8.0/10  │                      │
│      └─────────┘           └─────────┘                      │
│           │                      │                           │
│           └──────────┬───────────┘                          │
│                      │                                       │
│  ┌─────────────────┐ │ ┌─────────────────┐                  │
│  │ Executor Output │ │ │ Assessor Output │                  │
│  │                 │ │ │                 │                  │
│  │ - tasks done    │ │ │ - gap analysis  │                  │
│  │ - accuracy      │ │ │ - scores        │                  │
│  │ - artifacts     │ │ │ - recommendations│                 │
│  └────────┬────────┘ │ └────────┬────────┘                  │
│           │          │          │                            │
│           ▼          │          ▼                            │
│      ┌─────────┐     │     ┌─────────┐                      │
│      │ Score:  │     │     │ Score:  │                      │
│      │ 6.5/10  │     │     │ 7.0/10  │                      │
│      └─────────┘     │     └─────────┘                      │
│           │          │          │                            │
│           └──────────┼──────────┘                           │
│                      │                                       │
│                      ▼                                       │
│           ┌─────────────────────┐                           │
│           │  OVERALL A.G.E.     │                           │
│           │    SCORE: 7.25/10   │                           │
│           └─────────────────────┘                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Common Issues

#### 1. Ollama Not Responding

```bash
# Check if Ollama is running
curl http://localhost:11434/api/generate -d '{"model": "llama3.2:3b", "prompt": "test"}'

# If not running, start it
ollama serve

# Pull the model if needed
ollama pull llama3.2:3b
```

#### 2. Database Locked Error

```bash
# Remove lock files
rm backend/please.db-shm backend/please.db-wal

# Or reset database
cd backend
python -c "from database import reset_db; reset_db()"
```

#### 3. CORS Errors

Ensure frontend URL is in `backend/main.py`:
```python
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",  # Add your port
]
```

#### 4. MCP Connection Failed

The Learner will automatically fall back to default genes if MCP fails:
```python
# Fallback genes used:
["BRCA1", "BRCA2", "TP53", "PIK3CA", "ERBB2", "ESR1", "PGR", "PTEN", "CDH1", "ATM"]
```

#### 5. Model Training Fails

Check data directory for preprocessed files:
```bash
ls backend/artifacts/{project_id}/{cycle_id}/data/
# Should contain: X_train.npy, X_test.npy, y_train.npy, y_test.npy
```

### Log Locations

- **Backend logs**: Console output from `uvicorn`
- **Agent outputs**: Stored in database `agent_outputs` table
- **Generated code**: `backend/artifacts/{project_id}/{cycle_id}/code/`

### Debug Mode

Enable SQL debugging:
```python
# In backend/database/database.py
engine = create_engine(DATABASE_URL, echo=True)  # Set echo=True
```

---

## Summary

The PLEASe Framework provides a complete, automated solution for biomedical ML research:

| Component | Purpose |
|-----------|---------|
| **5 Agents** | Specialized AI agents for each research phase |
| **BioContext MCP** | Real-time biomedical knowledge retrieval |
| **Iterative Cycles** | Multiple improvement iterations with recommendations |
| **Comprehensive Scoring** | NIH 1-9, Satisfaction 1-5, A.G.E. 1-10 |
| **Full Reports** | Markdown reports with metrics, analysis, and recommendations |
| **Modern UI** | React dashboard with real-time status tracking |

**Total Lines of Code**: ~4000 (Backend) + ~1500 (Frontend)

---

*Documentation generated for PLEASe Framework v1.0.0*
*© 2025 PLEASe Framework - Built by BRIJESH*

