# Database Module

This folder contains all database-related code for the PLEASe Framework.

## Files

### Core Files
- **`database.py`** - SQLAlchemy engine, session management, initialization
- **`models.py`** - 6 SQLAlchemy models (Project, Cycle, AgentOutput, Task, Artifact, Report)
- **`db_utils.py`** - Service layer with CRUD operations
- **`__init__.py`** - Package exports

### Documentation
- **`DB_SCHEMA_DESIGN.md`** - Detailed schema specification
- **`DB_SCHEMA_VISUAL.md`** - Visual reference and diagrams
- **`DB_IMPLEMENTATION_SUMMARY.md`** - Implementation guide

## Quick Start

```python
from database import (
    init_db, SessionLocal,
    ProjectService, CycleService, AgentOutputService
)

# Initialize database (first time)
init_db()

# Create session
db = SessionLocal()

# Use services
project = ProjectService.create_project(
    db, name="My Project", spec_sheet={...}
)

db.close()
```

## Testing

```bash
cd tests
python test_database.py
```

## Schema

### 6 Tables
1. **projects** - Main project container (UUID, spec_sheet)
2. **cycles** - PLEASe iterations (state 0.0→1.0)
3. **agent_outputs** - Agent results with version control
4. **tasks** - Task list from planner
5. **artifacts** - Generated files (filesystem paths)
6. **reports** - Final PM reports (markdown + scores)

See documentation files for detailed schema information.

