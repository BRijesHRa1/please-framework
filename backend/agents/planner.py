"""
Planner Agent - Decomposes research problem into executable tasks
Uses Ollama with llama3.2:3b locally
"""

import json
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class Task(BaseModel):
    """Model for a single task"""
    task_id: str = Field(description="Unique identifier for the task (e.g., T1, T2)")
    name: str = Field(description="Name of the task")
    description: str = Field(description="Detailed description of what needs to be done")
    dependencies: List[str] = Field(description="List of task IDs that must complete before this task")
    gpu_hours: float = Field(description="Estimated GPU hours needed (0 if CPU only)")
    priority: str = Field(description="Priority level: high, medium, or low")


class PlannerOutput(BaseModel):
    """Model for planner agent output"""
    summary: str = Field(description="Brief summary of the plan")
    tasks: List[Task] = Field(description="List of tasks to execute")
    total_gpu_estimate: float = Field(description="Total estimated GPU hours")
    estimated_duration: str = Field(description="Estimated total duration (e.g., '4-6 hours')")
    risk_factors: List[str] = Field(description="Potential risks or challenges")


class PlannerAgent:
    """Planner Agent that decomposes research problems into tasks using Ollama"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize the Planner Agent
        
        Args:
            model: Ollama model to use (default: llama3.2:3b)
        """
        # Initialize Ollama LLM via LangChain
        self.llm = ChatOllama(
            model=model,
            temperature=0.3,  # Lower temperature for more consistent planning
            format="json",  # Request JSON output from Ollama
            num_predict=500  # Max 500 tokens output
        )
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for the planner agent"""
        return """You are a research planner. Create a CONCISE execution plan in JSON format (max 500 tokens).

REQUIRED output structure:
{
  "summary": "Brief one-line overview",
  "tasks": [
    {
      "task_id": "T1",
      "name": "Short task name",
      "description": "One brief sentence",
      "dependencies": [],
      "gpu_hours": 0.0,
      "priority": "high"
    }
  ],
  "total_gpu_estimate": 0.0,
  "estimated_duration": "X-Y hours",
  "risk_factors": ["risk1", "risk2", "risk3"]
}

MUST include 2-3 risk_factors. Keep all text brief. Return ONLY valid JSON."""
    
    def create_user_prompt(self, spec_sheet: Dict[str, Any]) -> str:
        """Create the user prompt with specification details"""
        
        problem = spec_sheet.get('research_problem', {})
        resources = spec_sheet.get('resources', {})
        
        problem_statement = problem.get('problem_statement', 'Not specified')
        dataset = problem.get('dataset', 'Not specified')
        goal_metric = problem.get('goal_metric', 'Not specified')
        baseline = problem.get('baseline', 'Not specified')
        gpu_budget = resources.get('gpu_budget_hours', 4)
        
        return f"""Create execution plan for:
Problem: {problem_statement}
Dataset: {dataset}
Goal: {goal_metric}
Baseline: {baseline}
GPU Budget: {gpu_budget}h

Create 3-4 brief tasks: data prep, baseline, evaluation.
Keep all descriptions to ONE sentence. Return valid JSON only."""
    
    def plan(self, spec_sheet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an execution plan from a specification sheet
        
        Args:
            spec_sheet: Dictionary containing research specification
            
        Returns:
            Dictionary containing the execution plan with tasks and estimates
        """
        
        messages = [
            SystemMessage(content=self.create_system_prompt()),
            HumanMessage(content=self.create_user_prompt(spec_sheet))
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Clean up response - remove markdown if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Parse and validate JSON
            plan_data = json.loads(content)
            validated_plan = PlannerOutput(**plan_data)
            
            return validated_plan.model_dump()
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response: {response.content if 'response' in locals() else 'No response'}")
            return self._create_fallback_plan(spec_sheet)
        
        except Exception as e:
            print(f"Planner error: {e}")
            return self._create_fallback_plan(spec_sheet)
    
    def _create_fallback_plan(self, spec_sheet: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback plan if AI generation fails"""
        
        print("   ⚠️  Planner fallback invoked – returning default plan.")
        
        problem = spec_sheet.get('research_problem', {})
        data_sources = spec_sheet.get('data_sources', [])
        budget = spec_sheet.get('budget_constraints', {})
        
        problem_statement = problem.get('problem_statement', 'research task')
        dataset = data_sources[0] if data_sources else "provided dataset"
        max_time = budget.get('max_time_hours', budget.get('max_time', 12))
        gpu_budget = max_time * 0.2  # simple heuristic
        
        fallback = {
            "summary": "Standard ML pipeline: data prep, baseline, evaluation",
            "tasks": [
                {
                    "task_id": "T1",
                    "name": "Data Preparation",
                    "description": "Load, clean, and preprocess the dataset. Perform train/test split.",
                    "dependencies": [],
                    "gpu_hours": 0,
                    "priority": "high"
                },
                {
                    "task_id": "T2",
                    "name": "Baseline Model",
                    "description": "Implement and train baseline model (Random Forest or similar)",
                    "dependencies": ["T1"],
                    "gpu_hours": 0,
                    "priority": "high"
                },
                {
                    "task_id": "T3",
                    "name": "Model Evaluation",
                    "description": "Evaluate model",
                    "dependencies": ["T1"],
                    "gpu_hours": 0,
                    "priority": "high"
                }
            ],
            "total_gpu_estimate": gpu_budget * 0.7,
            "estimated_duration": f"{int(gpu_budget * 1.5)}-{int(gpu_budget * 2)} hours",
            "risk_factors": [
                "Dataset may require additional cleaning",
                "GPU availability on cluster",
                "Model convergence issues"
            ]
        }
        
        return fallback


