"""
API Routes for PLEASe Framework
"""

import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from database import (
    SessionLocal, get_db, init_db,
    ProjectService, CycleService, AgentOutputService, ReportService
)
from database.models import Project, Cycle, AgentOutput, Report

from .schemas import (
    SpecSheetRequest, 
    ProjectResponse, ProjectListResponse, ProjectDetailResponse,
    ProjectStatusResponse, SubmitProjectResponse,
    CycleResponse, AgentOutputResponse, ReportResponse,
    DashboardResponse, GapAnalysisItem, AGEScoreItem
)

# Create router
router = APIRouter(prefix="/api", tags=["PLEASe API"])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_pipeline(project_id: str, spec_sheet: dict):
    """
    Run the full agent pipeline in the background
    This function is called as a background task
    """
    import time
    from agents.planner import PlannerAgent
    from agents.learner import LearnerAgent
    from agents.executor import ExecutorAgent
    from agents.assessor import AssessorAgent
    from agents.pm import PMAgent
    
    db = SessionLocal()
    
    try:
        # Get project
        project = ProjectService.get_project(db, project_id)
        if not project:
            print(f"❌ Project {project_id} not found")
            return
        
        # Update status
        ProjectService.update_project_status(db, project_id, 'running')
        
        # Create cycle 1
        cycle = CycleService.create_cycle(db, project_id, 1)
        
        # ===== PLANNER =====
        print(f"🤖 [{project_id}] Running Planner Agent...")
        CycleService.update_cycle_state(db, cycle.id, 0.0, 'planner', 'planning')
        
        planner = PlannerAgent(model="llama3.2:3b")
        start_time = time.time()
        plan = planner.plan(spec_sheet)
        exec_time = time.time() - start_time
        
        AgentOutputService.create_agent_output(
            db, cycle.id, 'planner', 0.1, plan,
            summary=plan.get('summary', ''),
            execution_time_seconds=exec_time
        )
        
        # ===== LEARNER =====
        print(f"🧠 [{project_id}] Running Learner Agent...")
        CycleService.update_cycle_state(db, cycle.id, 0.1, 'learner', 'learning')
        
        learner = LearnerAgent(model="llama3.2:3b")
        start_time = time.time()
        resources = learner.learn(spec_sheet, plan)
        exec_time = time.time() - start_time
        
        AgentOutputService.create_agent_output(
            db, cycle.id, 'learner', 0.2, resources,
            summary=resources.get('summary', ''),
            execution_time_seconds=exec_time
        )
        
        # ===== EXECUTOR (CYCLE 1) =====
        print(f"⚙️  [{project_id}] Running Executor Agent (Cycle 1)...")
        CycleService.update_cycle_state(db, cycle.id, 0.2, 'executor', 'executing')
        
        executor = ExecutorAgent(model="llama3.2:3b")
        start_time = time.time()
        execution_results = executor.execute(
            spec_sheet, plan, resources,
            project_id=project_id,
            cycle_id=str(cycle.id)
        )
        exec_time = time.time() - start_time
        
        AgentOutputService.create_agent_output(
            db, cycle.id, 'executor', 0.3, execution_results,
            summary=execution_results.get('summary', ''),
            execution_time_seconds=exec_time
        )
        
        # ===== ASSESSOR (CYCLE 1) =====
        print(f"📊 [{project_id}] Running Assessor Agent (Cycle 1)...")
        CycleService.update_cycle_state(db, cycle.id, 0.3, 'assessor', 'assessing')
        
        assessor = AssessorAgent(model="llama3.2:3b")
        start_time = time.time()
        assessment = assessor.assess(spec_sheet, plan, resources, execution_results)
        exec_time = time.time() - start_time
        
        AgentOutputService.create_agent_output(
            db, cycle.id, 'assessor', 0.4, assessment,
            summary=assessment.get('summary', ''),
            execution_time_seconds=exec_time
        )
        
        # Store cycle 1 results
        cycle1_assessment = assessment
        cycle1_results = execution_results
        current_cycle = 1
        
        # ===== CYCLE 2 (always run) =====
        # Always run 2 cycles before concluding to apply improvements
        max_iterations = spec_sheet.get('budget_constraints', {}).get('max_iterations', 2)
        
        if current_cycle < max_iterations:
            print(f"🔄 [{project_id}] Starting Cycle 2 (applying improvements from Cycle 1)...")
            
            # Mark cycle 1 as completed before starting cycle 2
            CycleService.update_cycle_state(db, cycle.id, 1.0, 'assessor', 'completed')
            
            cycle2 = CycleService.create_cycle(db, project_id, 2)
            
            # Update project's current cycle number
            ProjectService.update_current_cycle(db, project_id, 2)
            recommendations = assessment.get('recommendations', [])
            current_cycle = 2
            
            # Executor Cycle 2
            CycleService.update_cycle_state(db, cycle2.id, 0.2, 'executor', 'executing')
            
            start_time = time.time()
            execution_results_2 = executor.execute(
                spec_sheet, plan, resources,
                project_id=project_id,
                cycle_id=str(cycle2.id),
                cycle=2,
                recommendations=recommendations
            )
            exec_time = time.time() - start_time
            
            AgentOutputService.create_agent_output(
                db, cycle2.id, 'executor', 0.3, execution_results_2,
                summary=execution_results_2.get('summary', ''),
                execution_time_seconds=exec_time
            )
            
            # Assessor Cycle 2
            CycleService.update_cycle_state(db, cycle2.id, 0.3, 'assessor', 'assessing')
            
            start_time = time.time()
            assessment_2 = assessor.assess(spec_sheet, plan, resources, execution_results_2)
            exec_time = time.time() - start_time
            
            AgentOutputService.create_agent_output(
                db, cycle2.id, 'assessor', 0.4, assessment_2,
                summary=assessment_2.get('summary', ''),
                execution_time_seconds=exec_time
            )
            
            CycleService.update_cycle_state(db, cycle2.id, 0.4, 'pm', 'completed')
            
            # Update for PM
            assessment = assessment_2
            cycle = cycle2
            executor_outputs = [execution_results, execution_results_2]
            assessor_outputs = [cycle1_assessment, assessment_2]
        else:
            executor_outputs = [execution_results]
            assessor_outputs = [cycle1_assessment]
        
        # ===== PM AGENT =====
        print(f"📋 [{project_id}] Running PM Agent...")
        CycleService.update_cycle_state(db, cycle.id, 0.4, 'pm', 'reporting')
        
        pm = PMAgent(model="llama3.2:3b")
        start_time = time.time()
        pm_output = pm.manage(
            spec_sheet=spec_sheet,
            planner_output=plan,
            learner_output=resources,
            executor_outputs=executor_outputs,
            assessor_outputs=assessor_outputs,
            total_execution_time=exec_time
        )
        exec_time = time.time() - start_time
        
        # Save PM output
        AgentOutputService.create_agent_output(
            db, cycle.id, 'pm', 1.0, pm_output,
            summary=pm_output.get('report_summary', ''),
            execution_time_seconds=exec_time
        )
        
        # Save report
        ReportService.create_report(
            db,
            cycle_id=cycle.id,
            content=pm_output['report_markdown'],
            final_nih_score=pm_output['final_nih_score'],
            final_bimodal_score=pm_output['final_satisfaction'],
            age_scores=pm_output['age_scores'],
            executive_summary=pm_output['report_summary'],
            recommendations=assessment.get('recommendations', [])
        )
        
        # Update final state
        CycleService.update_cycle_state(db, cycle.id, 1.0, 'pm', 'completed')
        ProjectService.update_project_status(db, project_id, 'completed')
        
        print(f"✅ [{project_id}] Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ [{project_id}] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        ProjectService.update_project_status(db, project_id, 'failed')
    finally:
        db.close()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PLEASe Framework API"}


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(db: Session = Depends(get_db)):
    """Get dashboard overview with statistics"""
    
    all_projects = ProjectService.get_all_projects(db)
    completed = [p for p in all_projects if p.status == 'completed']
    running = [p for p in all_projects if p.status == 'running']
    failed = [p for p in all_projects if p.status == 'failed']
    
    # Get recent projects (last 5)
    recent = sorted(all_projects, key=lambda p: p.created_at, reverse=True)[:5]
    
    # Calculate average scores from reports
    avg_age = None
    avg_nih = None
    
    if completed:
        age_scores = []
        nih_scores = []
        for p in completed:
            for cycle in p.cycles:
                report = ReportService.get_report_by_cycle(db, cycle.id)
                if report:
                    if report.age_scores:
                        # Calculate overall A.G.E. from individual scores
                        scores = [s.get('score', 0) for s in report.age_scores.values() if isinstance(s, dict)]
                        if scores:
                            age_scores.append(sum(scores) / len(scores))
                    if report.final_nih_score:
                        nih_scores.append(report.final_nih_score)
        
        if age_scores:
            avg_age = round(sum(age_scores) / len(age_scores), 1)
        if nih_scores:
            avg_nih = round(sum(nih_scores) / len(nih_scores), 1)
    
    return DashboardResponse(
        total_projects=len(all_projects),
        completed_projects=len(completed),
        running_projects=len(running),
        failed_projects=len(failed),
        recent_projects=[ProjectResponse.model_validate(p) for p in recent],
        avg_age_score=avg_age,
        avg_nih_score=avg_nih
    )


@router.post("/projects", response_model=SubmitProjectResponse)
async def create_project(
    spec_sheet: SpecSheetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Submit a new project with spec sheet
    Starts the agent pipeline in the background
    """
    
    # Convert to dict
    spec_dict = spec_sheet.model_dump()
    
    # Create project
    project_name = spec_sheet.project_metadata.project_name
    description = spec_sheet.research_problem.problem_statement
    
    project = ProjectService.create_project(
        db,
        name=project_name,
        spec_sheet=spec_dict,
        description=description,
        total_cycles_planned=spec_sheet.budget_constraints.max_iterations if spec_sheet.budget_constraints else 2
    )
    
    # Start pipeline in background
    background_tasks.add_task(run_pipeline, project.id, spec_dict)
    
    return SubmitProjectResponse(
        project_id=project.id,
        message=f"Project '{project_name}' created. Pipeline started in background.",
        status="running"
    )


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all projects, optionally filtered by status"""
    
    projects = ProjectService.get_all_projects(db, status=status)
    
    return ProjectListResponse(
        total=len(projects),
        projects=[ProjectResponse.model_validate(p) for p in projects]
    )


@router.get("/projects/{project_id}", response_model=ProjectDetailResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get detailed project information"""
    
    project = ProjectService.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get cycles
    cycles = project.cycles
    
    # Get agent outputs organized by cycle
    agent_outputs = {}
    for cycle in cycles:
        cycle_outputs = AgentOutputService.get_all_agent_outputs(db, cycle.id)
        agent_outputs[f"cycle_{cycle.cycle_number}"] = [
            AgentOutputResponse.model_validate(ao) for ao in cycle_outputs
        ]
    
    # Get report from latest cycle
    report = None
    gap_analysis = None
    age_scores = None
    
    if cycles:
        latest_cycle = max(cycles, key=lambda c: c.cycle_number)
        report_db = ReportService.get_report_by_cycle(db, latest_cycle.id)
        
        if report_db:
            report = ReportResponse(
                id=report_db.id,
                project_id=project_id,
                project_name=project.name,
                content=report_db.content,
                format=report_db.format,
                final_nih_score=report_db.final_nih_score,
                final_bimodal_score=report_db.final_bimodal_score,
                age_scores=report_db.age_scores,
                executive_summary=report_db.executive_summary,
                recommendations=report_db.recommendations,
                created_at=report_db.created_at
            )
            
            # Extract A.G.E. scores
            if report_db.age_scores:
                age_scores = {
                    name: AGEScoreItem(
                        agent_name=data.get('agent_name', name),
                        score=data.get('score', 0),
                        criteria=data.get('criteria', {}),
                        justification=data.get('justification', '')
                    )
                    for name, data in report_db.age_scores.items()
                    if isinstance(data, dict)
                }
        
        # Get gap analysis from assessor output
        assessor_output = AgentOutputService.get_current_agent_output(db, latest_cycle.id, 'assessor')
        if assessor_output and assessor_output.output_data:
            gap_data = assessor_output.output_data.get('gap_analysis', [])
            gap_analysis = [
                GapAnalysisItem(
                    metric=g.get('metric', ''),
                    goal=g.get('goal', 0),
                    achieved=g.get('achieved', 0),
                    gap=g.get('gap', 0),
                    status=g.get('status', '')
                )
                for g in gap_data
            ]
    
    return ProjectDetailResponse(
        project=ProjectResponse.model_validate(project),
        cycles=[CycleResponse.model_validate(c) for c in cycles],
        agent_outputs=agent_outputs,
        report=report,
        gap_analysis=gap_analysis,
        age_scores=age_scores
    )


@router.get("/projects/{project_id}/status", response_model=ProjectStatusResponse)
async def get_project_status(project_id: str, db: Session = Depends(get_db)):
    """Get current status of a project"""
    
    project = ProjectService.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get current cycle
    current_cycle = CycleService.get_current_cycle(db, project_id)
    
    current_state = 0.0
    current_agent = None
    
    if current_cycle:
        current_state = current_cycle.current_state
        current_agent = current_cycle.current_agent
    
    # Calculate progress percentage
    # State 1.0 = 100%, considering multiple cycles
    progress = (current_state / 1.0) * 100
    
    # Generate message
    if project.status == 'completed':
        message = "Pipeline completed successfully"
    elif project.status == 'failed':
        message = "Pipeline failed"
    elif project.status == 'running':
        agent_names = {
            'planner': 'Planning tasks...',
            'learner': 'Gathering research resources...',
            'executor': 'Executing ML pipeline...',
            'assessor': 'Evaluating results...',
            'pm': 'Generating final report...'
        }
        message = agent_names.get(current_agent, 'Processing...')
    else:
        message = "Waiting to start"
    
    return ProjectStatusResponse(
        project_id=project_id,
        status=project.status,
        current_cycle=current_cycle.cycle_number if current_cycle else 0,
        current_state=current_state,
        current_agent=current_agent,
        progress_percent=round(progress, 1),
        message=message
    )


@router.get("/projects/{project_id}/report", response_model=ReportResponse)
async def get_project_report(project_id: str, db: Session = Depends(get_db)):
    """Get the final report for a project"""
    
    project = ProjectService.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get report from latest cycle
    report = ReportService.get_report_by_project(db, project_id)
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found. Project may still be running.")
    
    return ReportResponse(
        id=report.id,
        project_id=project_id,
        project_name=project.name,
        content=report.content,
        format=report.format,
        final_nih_score=report.final_nih_score,
        final_bimodal_score=report.final_bimodal_score,
        age_scores=report.age_scores,
        executive_summary=report.executive_summary,
        recommendations=report.recommendations,
        created_at=report.created_at
    )


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a project and all its data"""
    
    project = ProjectService.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Delete project (cascade will handle related records)
    db.delete(project)
    db.commit()
    
    return {"message": f"Project {project_id} deleted successfully"}


@router.get("/spec-sheets")
async def list_spec_sheets():
    """List available spec sheet templates"""
    import os
    import json
    from pathlib import Path
    
    backend_dir = Path(__file__).resolve().parent.parent
    spec_sheets_dir = backend_dir / "spec_sheets"
    
    templates = []
    
    if spec_sheets_dir.exists():
        for spec_file in sorted(spec_sheets_dir.glob("*.json")):
            try:
                with open(spec_file, 'r') as f:
                    spec = json.load(f)
                
                templates.append({
                    "filename": spec_file.name,
                    "project_name": spec.get('project_metadata', {}).get('project_name', 'Unnamed'),
                    "problem_statement": spec.get('research_problem', {}).get('problem_statement', '')[:100],
                    "spec_sheet": spec
                })
            except:
                pass
    
    return {"templates": templates}


@router.get("/documentation")
async def get_documentation():
    """Get the framework documentation"""
    from pathlib import Path
    
    # Look for DOCUMENTATION.md in project root
    project_root = Path(__file__).resolve().parent.parent.parent
    doc_path = project_root / "DOCUMENTATION.md"
    
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Documentation file not found")
    
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "content": content,
            "filename": "DOCUMENTATION.md",
            "size": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading documentation: {str(e)}")

