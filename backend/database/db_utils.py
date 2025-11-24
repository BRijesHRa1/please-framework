"""
Database utility functions for common operations
"""

from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime
from .models import Project, Cycle, AgentOutput, Task, Artifact, Report


class ProjectService:
    """Service for project-related database operations"""
    
    @staticmethod
    def create_project(
        db: Session,
        name: str,
        spec_sheet: Dict[str, Any],
        description: Optional[str] = None,
        total_cycles_planned: int = 1
    ) -> Project:
        """Create a new project"""
        project = Project(
            name=name,
            description=description,
            spec_sheet=spec_sheet,
            status='initialized',
            total_cycles_planned=total_cycles_planned
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        return project
    
    @staticmethod
    def get_project(db: Session, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        return db.query(Project).filter(Project.id == project_id).first()
    
    @staticmethod
    def get_all_projects(db: Session, status: Optional[str] = None) -> List[Project]:
        """Get all projects, optionally filtered by status"""
        query = db.query(Project)
        if status:
            query = query.filter(Project.status == status)
        return query.order_by(Project.created_at.desc()).all()
    
    @staticmethod
    def update_project_status(db: Session, project_id: str, status: str) -> Optional[Project]:
        """Update project status"""
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = status
            project.updated_at = datetime.utcnow()
            if status == 'completed':
                project.completed_at = datetime.utcnow()
            db.commit()
            db.refresh(project)
        return project


class CycleService:
    """Service for cycle-related database operations"""
    
    @staticmethod
    def create_cycle(
        db: Session,
        project_id: str,
        cycle_number: int
    ) -> Cycle:
        """Create a new cycle for a project"""
        cycle = Cycle(
            project_id=project_id,
            cycle_number=cycle_number,
            status='initialized',
            current_state=0.00
        )
        db.add(cycle)
        db.commit()
        db.refresh(cycle)
        return cycle
    
    @staticmethod
    def get_cycle(db: Session, cycle_id: str) -> Optional[Cycle]:
        """Get cycle by ID"""
        return db.query(Cycle).filter(Cycle.id == cycle_id).first()
    
    @staticmethod
    def get_current_cycle(db: Session, project_id: str) -> Optional[Cycle]:
        """Get the current/latest cycle for a project"""
        return db.query(Cycle).filter(
            Cycle.project_id == project_id
        ).order_by(Cycle.cycle_number.desc()).first()
    
    @staticmethod
    def update_cycle_state(
        db: Session,
        cycle_id: str,
        state: float,
        agent: str,
        status: Optional[str] = None
    ) -> Optional[Cycle]:
        """Update cycle state and current agent"""
        cycle = db.query(Cycle).filter(Cycle.id == cycle_id).first()
        if cycle:
            cycle.current_state = state
            cycle.current_agent = agent
            if status:
                cycle.status = status
            if state == 1.0 and status != 'failed':
                cycle.status = 'completed'
                cycle.completed_at = datetime.utcnow()
                # Calculate duration
                if cycle.started_at:
                    duration = (cycle.completed_at - cycle.started_at).total_seconds()
                    cycle.total_duration_seconds = int(duration)
            db.commit()
            db.refresh(cycle)
        return cycle


class AgentOutputService:
    """Service for agent output operations"""
    
    @staticmethod
    def create_agent_output(
        db: Session,
        cycle_id: str,
        agent_name: str,
        state_value: float,
        output_data: Dict[str, Any],
        summary: Optional[str] = None,
        execution_time_seconds: Optional[float] = None
    ) -> AgentOutput:
        """Create a new agent output"""
        
        # Mark previous outputs for this agent as not current (version control)
        db.query(AgentOutput).filter(
            AgentOutput.cycle_id == cycle_id,
            AgentOutput.agent_name == agent_name
        ).update({"is_current": False})
        
        # Determine version number
        last_version = db.query(AgentOutput).filter(
            AgentOutput.cycle_id == cycle_id,
            AgentOutput.agent_name == agent_name
        ).order_by(AgentOutput.version.desc()).first()
        
        version = (last_version.version + 1) if last_version else 1
        
        # Create new output
        agent_output = AgentOutput(
            cycle_id=cycle_id,
            agent_name=agent_name,
            state_value=state_value,
            output_data=output_data,
            summary=summary,
            version=version,
            is_current=True,
            status='completed',
            execution_time_seconds=execution_time_seconds,
            completed_at=datetime.utcnow()
        )
        db.add(agent_output)
        db.commit()
        db.refresh(agent_output)
        return agent_output
    
    @staticmethod
    def get_current_agent_output(
        db: Session,
        cycle_id: str,
        agent_name: str
    ) -> Optional[AgentOutput]:
        """Get the current version of an agent's output"""
        return db.query(AgentOutput).filter(
            AgentOutput.cycle_id == cycle_id,
            AgentOutput.agent_name == agent_name,
            AgentOutput.is_current == True
        ).first()
    
    @staticmethod
    def get_all_agent_outputs(db: Session, cycle_id: str) -> List[AgentOutput]:
        """Get all current agent outputs for a cycle"""
        return db.query(AgentOutput).filter(
            AgentOutput.cycle_id == cycle_id,
            AgentOutput.is_current == True
        ).order_by(AgentOutput.state_value).all()


class TaskService:
    """Service for task operations"""
    
    @staticmethod
    def create_tasks_from_planner(
        db: Session,
        agent_output_id: str,
        tasks_data: List[Dict[str, Any]]
    ) -> List[Task]:
        """Create tasks from planner output"""
        tasks = []
        for task_data in tasks_data:
            task = Task(
                agent_output_id=agent_output_id,
                task_id=task_data.get('task_id'),
                name=task_data.get('name'),
                description=task_data.get('description'),
                dependencies=task_data.get('dependencies', []),
                gpu_hours=task_data.get('gpu_hours', 0),
                priority=task_data.get('priority', 'medium'),
                status='pending'
            )
            db.add(task)
            tasks.append(task)
        
        db.commit()
        for task in tasks:
            db.refresh(task)
        return tasks
    
    @staticmethod
    def get_tasks_by_cycle(db: Session, cycle_id: str) -> List[Task]:
        """Get all tasks for a cycle"""
        # Join through agent_output to get cycle's tasks
        return db.query(Task).join(AgentOutput).filter(
            AgentOutput.cycle_id == cycle_id,
            AgentOutput.agent_name == 'planner',
            AgentOutput.is_current == True
        ).order_by(Task.task_id).all()
    
    @staticmethod
    def update_task_status(
        db: Session,
        task_id: str,
        status: str,
        result_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        """Update task status and results"""
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = status
            if status == 'in_progress':
                task.started_at = datetime.utcnow()
            elif status in ['completed', 'failed']:
                task.completed_at = datetime.utcnow()
            if result_data:
                task.result_data = result_data
            db.commit()
            db.refresh(task)
        return task


class ArtifactService:
    """Service for artifact operations"""
    
    @staticmethod
    def create_artifact(
        db: Session,
        cycle_id: str,
        name: str,
        type: str,
        file_path: str,
        generated_by_agent: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        agent_output_id: Optional[str] = None
    ) -> Artifact:
        """Create a new artifact record"""
        import os
        
        artifact = Artifact(
            cycle_id=cycle_id,
            agent_output_id=agent_output_id,
            name=name,
            type=type,
            file_path=file_path,
            generated_by_agent=generated_by_agent,
            meta_data=meta_data
        )
        
        # Get file size if file exists
        if os.path.exists(file_path):
            artifact.file_size_bytes = os.path.getsize(file_path)
        
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        return artifact
    
    @staticmethod
    def get_artifacts_by_cycle(db: Session, cycle_id: str) -> List[Artifact]:
        """Get all artifacts for a cycle"""
        return db.query(Artifact).filter(
            Artifact.cycle_id == cycle_id
        ).order_by(Artifact.created_at).all()


class ReportService:
    """Service for report operations"""
    
    @staticmethod
    def create_report(
        db: Session,
        cycle_id: str,
        content: str,
        final_nih_score: Optional[int] = None,
        final_bimodal_score: Optional[int] = None,
        age_scores: Optional[Dict[str, Any]] = None,
        executive_summary: Optional[str] = None,
        recommendations: Optional[List[str]] = None
    ) -> Report:
        """Create a final report"""
        report = Report(
            cycle_id=cycle_id,
            content=content,
            final_nih_score=final_nih_score,
            final_bimodal_score=final_bimodal_score,
            age_scores=age_scores,
            executive_summary=executive_summary,
            recommendations=recommendations,
            format='markdown'
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        return report
    
    @staticmethod
    def get_report_by_cycle(db: Session, cycle_id: str) -> Optional[Report]:
        """Get report for a cycle"""
        return db.query(Report).filter(
            Report.cycle_id == cycle_id
        ).order_by(Report.created_at.desc()).first()
    
    @staticmethod
    def get_report_by_project(db: Session, project_id: str) -> Optional[Report]:
        """Get latest report for a project"""
        return db.query(Report).join(Cycle).filter(
            Cycle.project_id == project_id
        ).order_by(Report.created_at.desc()).first()

