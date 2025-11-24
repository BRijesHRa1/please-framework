"""
Assessor Agent - Evaluates execution quality and provides recommendations
Uses bimodal satisfaction score (1-5) and NIH-style 1-9 scale
"""

import json
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class GapAnalysis(BaseModel):
    """Model for gap analysis between goal and achieved"""
    metric: str = Field(description="Metric name (e.g., accuracy)")
    goal: float = Field(description="Target goal value")
    achieved: float = Field(description="Achieved value")
    gap: float = Field(description="Difference (achieved - goal)")
    status: str = Field(description="EXCEEDED, MET, or MISSED")
    percentage_of_goal: float = Field(description="Achieved as percentage of goal")


class AssessorOutput(BaseModel):
    """Model for assessor agent output"""
    model_config = {"protected_namespaces": ()}
    
    # Scores
    satisfaction_score: int = Field(description="Bimodal satisfaction score (1-5)")
    nih_score: int = Field(description="NIH-style score (1-9)")
    
    # Gap Analysis
    gap_analysis: List[GapAnalysis] = Field(description="Gap analysis for each metric")
    overall_status: str = Field(description="Overall status: EXCEEDED, MET, PARTIALLY_MET, MISSED")
    
    # Qualitative Assessment
    strengths: List[str] = Field(description="Identified strengths")
    weaknesses: List[str] = Field(description="Identified weaknesses")
    recommendations: List[str] = Field(description="Recommendations for next iteration")
    
    # Summary
    summary: str = Field(description="Brief summary of assessment")
    should_continue: bool = Field(description="Whether to continue with more iterations")


class AssessorAgent:
    """Assessor Agent that evaluates execution quality using multiple scales"""
    
    # NIH Score Descriptions
    NIH_SCALE = {
        1: "Exceptional - Outstanding, innovative approach with significant impact",
        2: "Outstanding - Excellent execution with minor improvements possible",
        3: "Excellent - Very good results, well-designed methodology",
        4: "Very Good - Sound approach with good results",
        5: "Good - Solid work meeting basic requirements",
        6: "Satisfactory - Acceptable but with notable weaknesses",
        7: "Fair - Below expectations, significant improvements needed",
        8: "Marginal - Poor results, major revisions required",
        9: "Poor - Fundamentally flawed, restart recommended"
    }
    
    # Satisfaction Score Descriptions  
    SATISFACTION_SCALE = {
        1: "Very Dissatisfied - Major issues, goals not met",
        2: "Dissatisfied - Below expectations",
        3: "Neutral - Met minimum requirements",
        4: "Satisfied - Good results, minor improvements possible",
        5: "Very Satisfied - Exceeded expectations"
    }
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize the Assessor Agent
        
        Args:
            model: Ollama model to use (default: llama3.2:3b)
        """
        self.llm = ChatOllama(
            model=model,
            temperature=0.3,
            format="json",
            num_predict=1000
        )
    
    def calculate_gap(self, goal: float, achieved: float, metric: str) -> GapAnalysis:
        """
        Calculate the gap between goal and achieved value
        
        Args:
            goal: Target value
            achieved: Achieved value
            metric: Metric name
            
        Returns:
            GapAnalysis object
        """
        gap = achieved - goal
        percentage = (achieved / goal * 100) if goal > 0 else 0
        
        # Determine status
        if gap >= 0.02:  # Exceeded by 2% or more
            status = "EXCEEDED"
        elif gap >= -0.02:  # Within 2% of goal
            status = "MET"
        else:
            status = "MISSED"
        
        return GapAnalysis(
            metric=metric,
            goal=goal,
            achieved=achieved,
            gap=round(gap, 4),
            status=status,
            percentage_of_goal=round(percentage, 2)
        )
    
    def analyze_gaps(self, spec_sheet: Dict[str, Any], 
                     executor_output: Dict[str, Any]) -> List[GapAnalysis]:
        """
        Analyze gaps for all metrics
        
        Args:
            spec_sheet: Research specification with goals
            executor_output: Results from executor
            
        Returns:
            List of GapAnalysis objects
        """
        gaps = []
        goals = spec_sheet.get('goals', {})
        
        # Get achieved metrics from executor output
        baseline_results = executor_output.get('baseline_results', {})
        model_results = executor_output.get('model_results', {})
        
        # Use best available results
        achieved_metrics = model_results if model_results else baseline_results
        
        # Primary metric
        primary_metric = goals.get('primary_metric', 'accuracy')
        target_value = goals.get('target_value', 0.80)
        achieved_value = achieved_metrics.get(primary_metric, 0)
        
        if achieved_value:
            gaps.append(self.calculate_gap(target_value, achieved_value, primary_metric))
        
        # Secondary goals
        secondary_goals = goals.get('secondary_goals', {})
        for metric, goal_value in secondary_goals.items():
            achieved = achieved_metrics.get(metric, 0)
            if achieved:
                gaps.append(self.calculate_gap(goal_value, achieved, metric))
        
        return gaps
    
    def determine_overall_status(self, gaps: List[GapAnalysis]) -> str:
        """Determine overall status based on all gap analyses"""
        if not gaps:
            return "UNKNOWN"
        
        statuses = [g.status for g in gaps]
        
        if all(s == "EXCEEDED" for s in statuses):
            return "EXCEEDED"
        elif all(s in ["EXCEEDED", "MET"] for s in statuses):
            return "MET"
        elif any(s in ["EXCEEDED", "MET"] for s in statuses):
            return "PARTIALLY_MET"
        else:
            return "MISSED"
    
    def calculate_scores(self, gaps: List[GapAnalysis], 
                        overall_status: str) -> tuple:
        """
        Calculate satisfaction and NIH scores based on gap analysis
        
        Returns:
            Tuple of (satisfaction_score, nih_score)
        """
        if not gaps:
            return 3, 5  # Neutral defaults
        
        # Calculate average percentage of goal achieved
        avg_percentage = sum(g.percentage_of_goal for g in gaps) / len(gaps)
        
        # Satisfaction Score (1-5)
        if avg_percentage >= 105:
            satisfaction = 5  # Very Satisfied - Exceeded
        elif avg_percentage >= 95:
            satisfaction = 4  # Satisfied - Met
        elif avg_percentage >= 85:
            satisfaction = 3  # Neutral - Close
        elif avg_percentage >= 70:
            satisfaction = 2  # Dissatisfied - Below
        else:
            satisfaction = 1  # Very Dissatisfied - Far below
        
        # NIH Score (1-9, lower is better)
        if avg_percentage >= 110:
            nih = 1  # Exceptional
        elif avg_percentage >= 105:
            nih = 2  # Outstanding
        elif avg_percentage >= 100:
            nih = 3  # Excellent
        elif avg_percentage >= 95:
            nih = 4  # Very Good
        elif avg_percentage >= 90:
            nih = 5  # Good
        elif avg_percentage >= 85:
            nih = 6  # Satisfactory
        elif avg_percentage >= 75:
            nih = 7  # Fair
        elif avg_percentage >= 60:
            nih = 8  # Marginal
        else:
            nih = 9  # Poor
        
        return satisfaction, nih
    
    def create_assessment_prompt(self, spec_sheet: Dict[str, Any],
                                  executor_output: Dict[str, Any],
                                  gaps: List[GapAnalysis],
                                  satisfaction_score: int,
                                  nih_score: int) -> str:
        """Create prompt for LLM to generate qualitative assessment"""
        
        goals = spec_sheet.get('goals', {})
        problem = spec_sheet.get('research_problem', {}).get('problem_statement', '')
        
        gap_summary = "\n".join([
            f"  - {g.metric}: Goal={g.goal:.2f}, Achieved={g.achieved:.4f}, Gap={g.gap:+.4f} ({g.status})"
            for g in gaps
        ])
        
        tasks_completed = executor_output.get('tasks_completed', [])
        total_tasks = len(executor_output.get('artifacts_generated', [])) > 0
        
        return f"""Evaluate this ML research execution and provide assessment.

RESEARCH PROBLEM: {problem}

GOALS:
- Primary: {goals.get('primary_metric', 'accuracy')} >= {goals.get('target_value', 0.80)}
- Description: {goals.get('description', 'Meet target accuracy')}

GAP ANALYSIS:
{gap_summary}

EXECUTION SUMMARY:
- Tasks Completed: {len(tasks_completed)}
- Artifacts Generated: {len(executor_output.get('artifacts_generated', []))}
- Execution Time: {executor_output.get('total_execution_time', 0):.1f}s

PRE-CALCULATED SCORES:
- Satisfaction Score: {satisfaction_score}/5 ({self.SATISFACTION_SCALE.get(satisfaction_score, '')})
- NIH Score: {nih_score}/9 ({self.NIH_SCALE.get(nih_score, '')})

Provide your assessment in this JSON format:
{{
  "strengths": ["strength1", "strength2", "strength3"],
  "weaknesses": ["weakness1", "weakness2"],
  "recommendations": ["recommendation1", "recommendation2", "recommendation3"],
  "summary": "Brief 1-2 sentence summary",
  "should_continue": true/false
}}

Focus on:
1. What worked well (strengths)
2. What needs improvement (weaknesses)  
3. Specific actionable recommendations for next iteration
4. Whether more iterations would be beneficial

Return ONLY valid JSON."""
    
    def assess(self, spec_sheet: Dict[str, Any],
               planner_output: Dict[str, Any],
               learner_output: Dict[str, Any],
               executor_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the execution quality
        
        Args:
            spec_sheet: Research specification with goals
            planner_output: Output from planner agent
            learner_output: Output from learner agent
            executor_output: Output from executor agent
            
        Returns:
            Dictionary containing assessment results
        """
        print("\n" + "="*80)
        print("📊 ASSESSOR AGENT - Evaluating Execution Quality")
        print("="*80)
        
        # Step 1: Analyze gaps
        print("\n   📐 Analyzing gaps between goals and achieved metrics...")
        gaps = self.analyze_gaps(spec_sheet, executor_output)
        
        # Display gap analysis
        print("\n   📊 GAP ANALYSIS:")
        for gap in gaps:
            status_emoji = "✅" if gap.status == "EXCEEDED" else ("🟡" if gap.status == "MET" else "❌")
            print(f"      {status_emoji} {gap.metric}: Goal={gap.goal:.2f} | Achieved={gap.achieved:.4f} | Gap={gap.gap:+.4f} ({gap.status})")
        
        # Step 2: Determine overall status
        overall_status = self.determine_overall_status(gaps)
        print(f"\n   🎯 Overall Status: {overall_status}")
        
        # Step 3: Calculate scores
        satisfaction_score, nih_score = self.calculate_scores(gaps, overall_status)
        
        print(f"\n   📈 SCORES:")
        print(f"      Satisfaction: {satisfaction_score}/5 - {self.SATISFACTION_SCALE.get(satisfaction_score, '')}")
        print(f"      NIH Score: {nih_score}/9 - {self.NIH_SCALE.get(nih_score, '')}")
        
        # Step 4: Get qualitative assessment from LLM
        print("\n   🤖 Generating qualitative assessment with Ollama...")
        
        prompt = self.create_assessment_prompt(
            spec_sheet, executor_output, gaps, satisfaction_score, nih_score
        )
        
        messages = [
            SystemMessage(content="""You are a research quality assessor. Evaluate ML research execution.
Provide constructive feedback with specific, actionable recommendations.
Return ONLY valid JSON with strengths, weaknesses, recommendations, summary, and should_continue."""),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Clean up response
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            llm_assessment = json.loads(content)
            
        except Exception as e:
            print(f"   ⚠️  LLM assessment failed: {e}")
            llm_assessment = self._create_fallback_assessment(gaps, overall_status)
        
        # Step 5: Compile final output
        print(f"\n   💡 STRENGTHS:")
        for s in llm_assessment.get('strengths', []):
            print(f"      ✓ {s}")
        
        print(f"\n   ⚠️  WEAKNESSES:")
        for w in llm_assessment.get('weaknesses', []):
            print(f"      • {w}")
        
        print(f"\n   📝 RECOMMENDATIONS:")
        for r in llm_assessment.get('recommendations', []):
            print(f"      → {r}")
        
        should_continue = llm_assessment.get('should_continue', True)
        print(f"\n   🔄 Continue Iterations: {'Yes' if should_continue else 'No'}")
        
        # Build output
        assessor_output = {
            "satisfaction_score": satisfaction_score,
            "nih_score": nih_score,
            "gap_analysis": [g.model_dump() for g in gaps],
            "overall_status": overall_status,
            "strengths": llm_assessment.get('strengths', []),
            "weaknesses": llm_assessment.get('weaknesses', []),
            "recommendations": llm_assessment.get('recommendations', []),
            "summary": llm_assessment.get('summary', f"Execution {overall_status.lower()} goals"),
            "should_continue": should_continue
        }
        
        print("\n" + "="*80)
        print(f"✅ ASSESSMENT COMPLETE")
        print(f"   Final Score: {nih_score}/9 (NIH) | {satisfaction_score}/5 (Satisfaction)")
        print("="*80)
        
        return assessor_output
    
    def _create_fallback_assessment(self, gaps: List[GapAnalysis], 
                                    overall_status: str) -> Dict[str, Any]:
        """Create fallback assessment if LLM fails"""
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        for gap in gaps:
            if gap.status == "EXCEEDED":
                strengths.append(f"Exceeded {gap.metric} goal by {abs(gap.gap):.2%}")
            elif gap.status == "MET":
                strengths.append(f"Successfully met {gap.metric} target")
            else:
                weaknesses.append(f"Did not meet {gap.metric} target (gap: {gap.gap:.2%})")
                recommendations.append(f"Focus on improving {gap.metric} in next iteration")
        
        if not strengths:
            strengths = ["Completed execution pipeline", "Generated model artifacts"]
        
        if not weaknesses:
            weaknesses = ["No hyperparameter tuning performed", "Single model tested"]
        
        if not recommendations:
            recommendations = [
                "Add cross-validation for more robust evaluation",
                "Try multiple model architectures",
                "Perform hyperparameter tuning"
            ]
        
        return {
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
            "recommendations": recommendations[:3],
            "summary": f"Execution {overall_status.lower()} with room for improvement",
            "should_continue": overall_status in ["PARTIALLY_MET", "MISSED"]
        }
