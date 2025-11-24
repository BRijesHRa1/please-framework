"""
PM (Project Manager) Agent - Generates comprehensive research reports
Calculates A.G.E. (Agent Grading Evaluation) scores and creates final reports
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class AGEScore(BaseModel):
    """Model for Agent Grading Evaluation score"""
    agent_name: str = Field(description="Agent name (planner, learner, executor, assessor)")
    score: float = Field(description="A.G.E. score (1-10)")
    criteria: Dict[str, float] = Field(description="Individual criteria scores")
    justification: str = Field(description="Brief justification for score")


class PMOutput(BaseModel):
    """Model for PM agent output"""
    model_config = {"protected_namespaces": ()}
    
    # A.G.E. Scores
    age_scores: Dict[str, AGEScore] = Field(description="A.G.E. scores for each agent")
    overall_age_score: float = Field(description="Overall A.G.E. score (average)")
    
    # Report
    report_markdown: str = Field(description="Full research report in Markdown")
    report_summary: str = Field(description="Executive summary")
    
    # Final Metrics
    final_nih_score: int = Field(description="Final NIH score (1-9)")
    final_satisfaction: int = Field(description="Final satisfaction score (1-5)")
    project_status: str = Field(description="Project status: SUCCESS, PARTIAL, FAILED")
    
    # Metadata
    total_cycles: int = Field(description="Total number of cycles completed")
    total_execution_time: float = Field(description="Total execution time in seconds")


class PMAgent:
    """PM Agent that generates comprehensive research reports with A.G.E. scoring"""
    
    # A.G.E. Scoring Criteria
    AGE_CRITERIA = {
        "planner": {
            "task_clarity": "Clear and actionable task definitions",
            "resource_estimation": "Accurate GPU/time estimates",
            "dependency_logic": "Logical task dependencies",
            "risk_identification": "Identified relevant risks"
        },
        "learner": {
            "domain_knowledge": "Relevant biomedical context gathered",
            "tool_selection": "Appropriate tools/methods suggested",
            "gene_relevance": "Relevant genes identified",
            "preprocessing_guidance": "Useful preprocessing recommendations"
        },
        "executor": {
            "task_completion": "Tasks completed successfully",
            "code_quality": "Code executed without errors",
            "metric_achievement": "Achieved target metrics",
            "artifact_generation": "Generated useful artifacts"
        },
        "assessor": {
            "gap_analysis": "Accurate gap analysis",
            "scoring_accuracy": "Appropriate NIH/satisfaction scores",
            "recommendations": "Actionable recommendations",
            "improvement_tracking": "Tracked improvements across cycles"
        }
    }
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize the PM Agent
        
        Args:
            model: Ollama model to use (default: llama3.2:3b)
        """
        self.llm = ChatOllama(
            model=model,
            temperature=0.4,
            num_predict=3000  # Allow longer report generation
        )
    
    def calculate_age_score(self, agent_name: str, 
                            agent_output: Dict[str, Any],
                            spec_sheet: Dict[str, Any] = None) -> AGEScore:
        """
        Calculate A.G.E. score for a specific agent
        
        Args:
            agent_name: Name of the agent (planner, learner, executor, assessor)
            agent_output: Output from the agent
            spec_sheet: Research specification (for context)
            
        Returns:
            AGEScore object
        """
        criteria = self.AGE_CRITERIA.get(agent_name, {})
        scores = {}
        
        if agent_name == "planner":
            # Evaluate planner output
            tasks = agent_output.get('tasks', [])
            scores['task_clarity'] = min(10, 5 + len(tasks) * 1.5) if tasks else 3
            scores['resource_estimation'] = 7.0 if agent_output.get('total_gpu_estimate') else 5.0
            scores['dependency_logic'] = 8.0 if any(t.get('dependencies') for t in tasks) else 6.0
            scores['risk_identification'] = min(10, 5 + len(agent_output.get('risk_factors', [])) * 1.5)
            
        elif agent_name == "learner":
            # Evaluate learner output
            genes = agent_output.get('key_genes', [])
            tools = agent_output.get('tools', [])
            scores['domain_knowledge'] = min(10, 5 + len(genes) / 50)
            scores['tool_selection'] = min(10, 5 + len(tools) * 2)
            scores['gene_relevance'] = 7.5 if len(genes) >= 10 else 5.0
            scores['preprocessing_guidance'] = 7.0 if agent_output.get('preprocessing_notes') else 4.0
            
        elif agent_name == "executor":
            # Evaluate executor output
            tasks_completed = agent_output.get('tasks_completed', [])
            total_tasks = len(tasks_completed) + 1  # Avoid division by zero
            baseline = agent_output.get('baseline_results', {})
            
            completion_rate = len(tasks_completed) / max(total_tasks, 1)
            scores['task_completion'] = min(10, completion_rate * 10)
            scores['code_quality'] = 8.0 if completion_rate > 0.8 else 5.0
            
            # Check metric achievement
            accuracy = baseline.get('accuracy', 0)
            scores['metric_achievement'] = min(10, accuracy * 12)  # Scale accuracy to score
            
            artifacts = agent_output.get('artifacts_generated', [])
            scores['artifact_generation'] = min(10, 5 + len(artifacts) * 0.5)
            
        elif agent_name == "assessor":
            # Evaluate assessor output
            gap_analysis = agent_output.get('gap_analysis', [])
            recommendations = agent_output.get('recommendations', [])
            
            scores['gap_analysis'] = min(10, 5 + len(gap_analysis) * 1.5)
            scores['scoring_accuracy'] = 7.5  # Default reasonable score
            scores['recommendations'] = min(10, 5 + len(recommendations) * 1.5)
            scores['improvement_tracking'] = 7.0 if agent_output.get('overall_status') else 5.0
        
        # Calculate average score
        avg_score = sum(scores.values()) / len(scores) if scores else 5.0
        
        # Generate justification
        if avg_score >= 8:
            justification = f"Excellent performance across all criteria"
        elif avg_score >= 6:
            justification = f"Good performance with room for improvement"
        else:
            justification = f"Below expectations, needs significant improvement"
        
        return AGEScore(
            agent_name=agent_name,
            score=round(avg_score, 1),
            criteria=scores,
            justification=justification
        )
    
    def generate_report_prompt(self, spec_sheet: Dict[str, Any],
                               planner_output: Dict[str, Any],
                               learner_output: Dict[str, Any],
                               executor_outputs: List[Dict[str, Any]],
                               assessor_outputs: List[Dict[str, Any]],
                               age_scores: Dict[str, AGEScore]) -> str:
        """Generate prompt for comprehensive report"""
        
        project_name = spec_sheet.get('project_metadata', {}).get('project_name', 'Research Project')
        problem = spec_sheet.get('research_problem', {}).get('problem_statement', '')
        goals = spec_sheet.get('goals', {})
        
        # Get final metrics from last cycle
        final_executor = executor_outputs[-1] if executor_outputs else {}
        final_assessor = assessor_outputs[-1] if assessor_outputs else {}
        
        baseline = final_executor.get('baseline_results', {})
        gap_analysis = final_assessor.get('gap_analysis', [])
        
        # Build gap summary
        gap_summary = "\n".join([
            f"- {g.get('metric', 'metric')}: Goal={g.get('goal', 0):.2f}, Achieved={g.get('achieved', 0):.4f}, Status={g.get('status', 'UNKNOWN')}"
            for g in gap_analysis
        ])
        
        # Build A.G.E. score summary
        age_summary = "\n".join([
            f"- {name.capitalize()}: {score.score}/10"
            for name, score in age_scores.items()
        ])
        
        return f"""Generate a comprehensive research report in Markdown format.

PROJECT: {project_name}
PROBLEM: {problem}

GOALS:
- Primary Metric: {goals.get('primary_metric', 'accuracy')}
- Target: {goals.get('target_value', 0.80)}

FINAL RESULTS (after {len(executor_outputs)} cycle(s)):
- Accuracy: {baseline.get('accuracy', 'N/A')}
- F1 Score: {baseline.get('f1', 'N/A')}
- ROC-AUC: {baseline.get('roc_auc', 'N/A')}
- CV Mean (if available): {baseline.get('cv_mean', 'N/A')}

GAP ANALYSIS:
{gap_summary}

A.G.E. SCORES:
{age_summary}

NIH SCORE: {final_assessor.get('nih_score', 'N/A')}/9
SATISFACTION: {final_assessor.get('satisfaction_score', 'N/A')}/5
STATUS: {final_assessor.get('overall_status', 'UNKNOWN')}

RECOMMENDATIONS FROM ASSESSOR:
{chr(10).join(['- ' + r for r in final_assessor.get('recommendations', [])])}

Generate a professional research report with these sections:
1. Executive Summary (2-3 sentences)
2. Project Overview
3. Methodology
4. Results & Analysis
5. Performance Metrics
6. A.G.E. Agent Evaluation
7. Conclusions
8. Future Work

Use proper Markdown formatting with headers (##), bullet points, tables where appropriate.
Keep the report concise but comprehensive (500-800 words)."""
    
    def generate_report(self, spec_sheet: Dict[str, Any],
                        planner_output: Dict[str, Any],
                        learner_output: Dict[str, Any],
                        executor_outputs: List[Dict[str, Any]],
                        assessor_outputs: List[Dict[str, Any]],
                        age_scores: Dict[str, AGEScore]) -> str:
        """Generate comprehensive Markdown report using LLM"""
        
        print("   📝 Generating comprehensive report with Ollama...")
        
        prompt = self.generate_report_prompt(
            spec_sheet, planner_output, learner_output,
            executor_outputs, assessor_outputs, age_scores
        )
        
        messages = [
            SystemMessage(content="""You are a research report writer. Generate professional, 
well-structured Markdown reports for ML research projects. Include proper sections, 
metrics tables, and actionable conclusions. Be concise but comprehensive."""),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            report = response.content.strip()
            
            # Clean up if needed
            if report.startswith("```markdown"):
                report = report[11:]
            elif report.startswith("```"):
                report = report[3:]
            if report.endswith("```"):
                report = report[:-3]
            
            return report.strip()
            
        except Exception as e:
            print(f"   ⚠️  Report generation failed: {e}")
            return self._generate_fallback_report(
                spec_sheet, planner_output, learner_output,
                executor_outputs, assessor_outputs, age_scores
            )
    
    def _generate_fallback_report(self, spec_sheet: Dict[str, Any],
                                   planner_output: Dict[str, Any],
                                   learner_output: Dict[str, Any],
                                   executor_outputs: List[Dict[str, Any]],
                                   assessor_outputs: List[Dict[str, Any]],
                                   age_scores: Dict[str, AGEScore]) -> str:
        """Generate fallback report if LLM fails"""
        
        project_name = spec_sheet.get('project_metadata', {}).get('project_name', 'Research Project')
        problem = spec_sheet.get('research_problem', {}).get('problem_statement', '')
        
        final_executor = executor_outputs[-1] if executor_outputs else {}
        final_assessor = assessor_outputs[-1] if assessor_outputs else {}
        baseline = final_executor.get('baseline_results', {})
        
        overall_age = sum(s.score for s in age_scores.values()) / len(age_scores) if age_scores else 0
        
        return f"""# {project_name} - Research Report

## Executive Summary

This report presents the results of an automated ML research project focused on: {problem}

After {len(executor_outputs)} iteration(s), the project achieved an accuracy of {baseline.get('accuracy', 'N/A')} 
with an overall A.G.E. score of {overall_age:.1f}/10.

## Project Overview

- **Problem**: {problem}
- **Dataset**: TCGA breast cancer gene expression
- **Target**: Survival classification
- **Cycles Completed**: {len(executor_outputs)}

## Results

| Metric | Value |
|--------|-------|
| Accuracy | {baseline.get('accuracy', 'N/A')} |
| F1 Score | {baseline.get('f1', 'N/A')} |
| ROC-AUC | {baseline.get('roc_auc', 'N/A')} |
| Precision | {baseline.get('precision', 'N/A')} |
| Recall | {baseline.get('recall', 'N/A')} |

## A.G.E. Scores

| Agent | Score |
|-------|-------|
""" + "\n".join([f"| {name.capitalize()} | {score.score}/10 |" for name, score in age_scores.items()]) + f"""

**Overall A.G.E. Score: {overall_age:.1f}/10**

## Assessment

- **NIH Score**: {final_assessor.get('nih_score', 'N/A')}/9
- **Satisfaction**: {final_assessor.get('satisfaction_score', 'N/A')}/5
- **Status**: {final_assessor.get('overall_status', 'UNKNOWN')}

## Recommendations

""" + "\n".join([f"- {r}" for r in final_assessor.get('recommendations', ['Continue iterating to improve results'])]) + """

## Conclusion

The automated research pipeline successfully completed the analysis with measurable improvements 
across iterations. Further optimization of feature engineering and model selection may yield 
additional performance gains.

---
*Report generated by PLEASe Framework PM Agent*
"""
    
    def manage(self, spec_sheet: Dict[str, Any],
               planner_output: Dict[str, Any],
               learner_output: Dict[str, Any],
               executor_outputs: List[Dict[str, Any]],
               assessor_outputs: List[Dict[str, Any]],
               total_execution_time: float = 0) -> Dict[str, Any]:
        """
        Generate comprehensive project report with A.G.E. scoring
        
        Args:
            spec_sheet: Research specification
            planner_output: Output from planner agent
            learner_output: Output from learner agent
            executor_outputs: List of outputs from executor (one per cycle)
            assessor_outputs: List of outputs from assessor (one per cycle)
            total_execution_time: Total time for all cycles
            
        Returns:
            Dictionary containing PM output with report
        """
        print("\n" + "="*80)
        print("📋 PM AGENT - Generating Final Report")
        print("="*80)
        
        # Step 1: Calculate A.G.E. scores for each agent
        print("\n   📊 Calculating A.G.E. (Agent Grading Evaluation) Scores...")
        
        age_scores = {}
        
        # Planner score
        age_scores['planner'] = self.calculate_age_score('planner', planner_output, spec_sheet)
        print(f"      • Planner: {age_scores['planner'].score}/10")
        
        # Learner score
        age_scores['learner'] = self.calculate_age_score('learner', learner_output, spec_sheet)
        print(f"      • Learner: {age_scores['learner'].score}/10")
        
        # Executor score (use best/last cycle)
        final_executor = executor_outputs[-1] if executor_outputs else {}
        age_scores['executor'] = self.calculate_age_score('executor', final_executor, spec_sheet)
        print(f"      • Executor: {age_scores['executor'].score}/10")
        
        # Assessor score (use best/last cycle)
        final_assessor = assessor_outputs[-1] if assessor_outputs else {}
        age_scores['assessor'] = self.calculate_age_score('assessor', final_assessor, spec_sheet)
        print(f"      • Assessor: {age_scores['assessor'].score}/10")
        
        # Calculate overall A.G.E. score
        overall_age = sum(s.score for s in age_scores.values()) / len(age_scores)
        print(f"\n   🎯 Overall A.G.E. Score: {overall_age:.1f}/10")
        
        # Step 2: Generate comprehensive report
        print("\n   📝 Generating Report...")
        report = self.generate_report(
            spec_sheet, planner_output, learner_output,
            executor_outputs, assessor_outputs, age_scores
        )
        
        # Step 3: Extract final metrics
        final_nih = final_assessor.get('nih_score', 5)
        final_satisfaction = final_assessor.get('satisfaction_score', 3)
        overall_status = final_assessor.get('overall_status', 'UNKNOWN')
        
        # Determine project status
        if overall_status == "EXCEEDED":
            project_status = "SUCCESS"
        elif overall_status in ["MET", "PARTIALLY_MET"]:
            project_status = "PARTIAL"
        else:
            project_status = "FAILED"
        
        # Create executive summary
        project_name = spec_sheet.get('project_metadata', {}).get('project_name', 'Research Project')
        baseline = final_executor.get('baseline_results', {})
        
        exec_summary = (
            f"The {project_name} completed {len(executor_outputs)} iteration(s) "
            f"achieving {baseline.get('accuracy', 0):.1%} accuracy. "
            f"Final NIH score: {final_nih}/9, A.G.E. score: {overall_age:.1f}/10."
        )
        
        # Step 4: Display report preview
        print("\n   📄 REPORT PREVIEW:")
        print("   " + "-"*60)
        preview_lines = report.split('\n')[:15]
        for line in preview_lines:
            print(f"   {line}")
        if len(report.split('\n')) > 15:
            print(f"   ... ({len(report.split(chr(10))) - 15} more lines)")
        print("   " + "-"*60)
        
        # Compile output
        pm_output = {
            "age_scores": {name: score.model_dump() for name, score in age_scores.items()},
            "overall_age_score": round(overall_age, 1),
            "report_markdown": report,
            "report_summary": exec_summary,
            "final_nih_score": final_nih,
            "final_satisfaction": final_satisfaction,
            "project_status": project_status,
            "total_cycles": len(executor_outputs),
            "total_execution_time": total_execution_time
        }
        
        print("\n" + "="*80)
        print("✅ PM REPORT COMPLETE")
        print(f"   Project Status: {project_status}")
        print(f"   A.G.E. Score: {overall_age:.1f}/10")
        print(f"   NIH Score: {final_nih}/9")
        print(f"   Total Cycles: {len(executor_outputs)}")
        print("="*80)
        
        return pm_output
