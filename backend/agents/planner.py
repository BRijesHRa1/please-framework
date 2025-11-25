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
    mcp_query: str = Field(description="Query to pass to MCP server for biomedical knowledge retrieval")


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
            num_predict=2000  # Max tokens output
        )
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for the planner agent"""
        return """You are a research planner. Create an execution plan in JSON format.

REQUIRED output structure:
{
  "summary": "overview",
  "tasks": [
    {
      "task_id": "T1",
      "name": "task name",
      "description": "brief sentences",
      "dependencies": [],
      "gpu_hours": 0.0,
      "priority": "high"
    }
  ],
  "total_gpu_estimate": 0.0,
  "estimated_duration": "X-Y hours",
  "risk_factors": ["risk1", "risk2", "risk3"],
  "mcp_query": "What genes are responsible for [disease/condition] and their role in [outcome]?"
}

CRITICAL: mcp_query should be a specific biomedical question based on the problem statement.
Examples:
- "What genes are responsible for breast cancer survival and prognosis?"
- "Which genes regulate tumor progression and patient outcomes in lung cancer?"
- "What are the key biomarkers for predicting survival in colon cancer patients?"

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
        
        # Extract disease/condition from problem statement for MCP query hint
        disease_hint = ""
        problem_lower = problem_statement.lower()
        if 'breast' in problem_lower:
            disease_hint = "breast cancer"
        elif 'lung' in problem_lower:
            disease_hint = "lung cancer"
        elif 'colon' in problem_lower:
            disease_hint = "colon cancer"
        elif 'survival' in problem_lower:
            disease_hint = "cancer survival"
        elif 'tumor' in problem_lower:
            disease_hint = "tumor"
        
        return f"""Create execution plan for:
Problem: {problem_statement}
Dataset: {dataset}
Goal: {goal_metric}
Baseline: {baseline}
GPU Budget: {gpu_budget}h

Create 3-4 brief tasks: data prep, baseline, evaluation.
Keep all descriptions to ONE sentence.

IMPORTANT: Generate an mcp_query - a biomedical question asking what genes/biomarkers are responsible for the condition in the problem statement.
{f'Focus on: {disease_hint}' if disease_hint else ''}

Return valid JSON only."""
    
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
            
            # Ensure mcp_query exists
            if 'mcp_query' not in plan_data or not plan_data['mcp_query']:
                plan_data['mcp_query'] = self._generate_mcp_query(spec_sheet)
            
            validated_plan = PlannerOutput(**plan_data)
            
            return validated_plan.model_dump()
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response: {response.content if 'response' in locals() else 'No response'}")
            return self._create_fallback_plan(spec_sheet)
        
        except Exception as e:
            print(f"Planner error: {e}")
            return self._create_fallback_plan(spec_sheet)
    
    def _generate_mcp_query(self, spec_sheet: Dict[str, Any]) -> str:
        """Generate an MCP query based on the problem statement"""
        problem = spec_sheet.get('research_problem', {})
        problem_statement = problem.get('problem_statement', '').lower()
        
        # Extract disease type
        if 'breast' in problem_statement:
            disease = "breast cancer"
        elif 'lung' in problem_statement:
            disease = "lung cancer"
        elif 'colon' in problem_statement:
            disease = "colon cancer"
        elif 'cancer' in problem_statement:
            disease = "cancer"
        elif 'tumor' in problem_statement:
            disease = "tumor"
        else:
            disease = "the disease"
        
        # Extract outcome type
        if 'survival' in problem_statement:
            outcome = "survival prediction and prognosis"
        elif 'classification' in problem_statement:
            outcome = "classification and diagnosis"
        elif 'outcome' in problem_statement:
            outcome = "patient outcomes"
        else:
            outcome = "patient outcomes and prognosis"
        
        return f"What genes and biomarkers are responsible for {disease} {outcome}? Which genetic factors influence {disease} progression?"
    
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
        
        # Generate MCP query for fallback
        mcp_query = self._generate_mcp_query(spec_sheet)
        
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
            ],
            "mcp_query": mcp_query
        }
        
        return fallback
