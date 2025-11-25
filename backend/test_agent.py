"""
Simple test script to run agent and store in database
Run: python test_agent.py [spec_number]
     python test_agent.py 1  # Run with first spec sheet
     python test_agent.py 2  # Run with second spec sheet
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from agents.planner import PlannerAgent
from agents.learner import LearnerAgent
from agents.executor import ExecutorAgent
from agents.assessor import AssessorAgent
from agents.pm import PMAgent
from database import (
    SessionLocal,
    ProjectService, CycleService, AgentOutputService, TaskService, ReportService,
    init_db
)


def select_spec_sheet(spec_number: int = None) -> dict:
    """
    Display available spec sheets and let user select one.
    If spec_number is provided, use it directly (for non-interactive mode).
    Returns the loaded spec sheet dictionary.
    """
    backend_dir = Path(__file__).resolve().parent
    spec_sheets_dir = backend_dir / "spec_sheets"
    
    # Find all spec sheet JSON files
    spec_files = []
    
    # Check spec_sheets directory
    if spec_sheets_dir.exists():
        spec_files.extend(sorted(list(spec_sheets_dir.glob("*.json"))))
    
    # Also check for legacy spec_sheet.json in backend root
    legacy_spec = backend_dir / "spec_sheet.json"
    if legacy_spec.exists() and legacy_spec not in spec_files:
        spec_files.append(legacy_spec)
    
    if not spec_files:
        print("❌ No spec sheets found!")
        print(f"   Please add JSON files to: {spec_sheets_dir}")
        sys.exit(1)
    
    # Load all spec info
    spec_info = []
    for i, spec_file in enumerate(spec_files, 1):
        try:
            with open(spec_file, 'r') as f:
                spec = json.load(f)
            
            project_name = spec.get('project_metadata', {}).get('project_name', 'Unnamed')
            problem = spec.get('research_problem', {}).get('problem_statement', 'No description')[:50]
            
            spec_info.append({
                'path': spec_file,
                'name': project_name,
                'problem': problem,
                'spec': spec
            })
        except json.JSONDecodeError:
            spec_info.append(None)
    
    # If spec_number provided via command line, use it directly
    if spec_number is not None:
        idx = spec_number - 1
        if 0 <= idx < len(spec_files) and spec_info[idx] is not None:
            selected = spec_info[idx]
            print(f"\n📄 Using spec sheet: {selected['name']}")
            print(f"   📁 {selected['path'].name}")
            return selected['spec']
        else:
            print(f"❌ Invalid spec number: {spec_number}. Available: 1-{len(spec_files)}")
            sys.exit(1)
    
    # Interactive mode - display menu
    print("\n" + "="*60)
    print("📋 AVAILABLE SPEC SHEETS")
    print("="*60)
    
    for i, info in enumerate(spec_info, 1):
        if info:
            print(f"\n  [{i}] {info['name']}")
            print(f"      📁 {info['path'].name}")
            print(f"      📝 {info['problem']}...")
        else:
            print(f"\n  [{i}] ⚠️  Invalid JSON")
    
    print("\n" + "-"*60)
    
    # Get user selection
    while True:
        try:
            choice = input(f"\n🔢 Select spec sheet (1-{len(spec_files)}): ").strip()
            idx = int(choice) - 1
            
            if 0 <= idx < len(spec_files) and spec_info[idx] is not None:
                selected = spec_info[idx]
                print(f"\n✅ Selected: {selected['name']}")
                return selected['spec']
            else:
                print(f"   ⚠️  Invalid choice. Enter 1-{len(spec_files)}")
        except ValueError:
            print(f"   ⚠️  Please enter a number (1-{len(spec_files)})")
        except KeyboardInterrupt:
            print("\n\n👋 Cancelled.")
            sys.exit(0)


def test_agents():
    """Run planner and learner agents and save to database"""
    
    print("="*80)
    print("RUNNING PLEASE AGENTS (Ollama - llama3.2:3b)")
    print("="*80)
    
    # Check for command line argument
    spec_number = None
    if len(sys.argv) > 1:
        try:
            spec_number = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid argument: {sys.argv[1]}. Use a number (1 or 2).")
            sys.exit(1)
    
    # Select spec sheet (interactive or via argument)
    spec_sheet = select_spec_sheet(spec_number)
    
    print("\n📄 Spec sheet loaded successfully!")
    
    # Ensure database tables exist
    init_db()
    
    db = SessionLocal()
    
    try:
        # Create project
        print("📁 Creating project...")
        project_meta = spec_sheet.get("project_metadata", {})
        project_name = project_meta.get("project_name", spec_sheet.get("project", {}).get("name", "Untitled Project"))
        project_desc = project_meta.get("description") or spec_sheet.get("research_problem", {}).get("problem_statement")
        project = ProjectService.create_project(
            db, 
            name=project_name,
            spec_sheet=spec_sheet,
            description=project_desc
        )
        print(f"   Project ID: {project.id}")
        
        # Create cycle
        print("🔄 Creating cycle...")
        cycle = CycleService.create_cycle(db, project.id, 1)
        print(f"   Cycle ID: {cycle.id}")
        
        # Update status
        ProjectService.update_project_status(db, project.id, 'running')
        CycleService.update_cycle_state(db, cycle.id, 0.0, 'planner', 'planning')
        
        # Run planner
        print("🤖 Running Planner Agent (Ollama)...")
        planner = PlannerAgent(model="llama3.2:3b")
        
        start_time = time.time()
        plan = planner.plan(spec_sheet)
        exec_time = time.time() - start_time
        
        print(f"   ✅ Complete ({exec_time:.1f}s)")
        print(f"\n📋 PLAN OUTPUT:")
        print(f"   Summary: {plan['summary']}")
        print(f"   Duration: {plan['estimated_duration']}")
        print(f"   Total GPU: {plan['total_gpu_estimate']}h")
        
        if plan.get('mcp_query'):
            print(f"\n🔍 MCP QUERY (for Learner):")
            print(f"   \"{plan['mcp_query']}\"")
        
        print(f"\n📝 TASKS ({len(plan['tasks'])}):")
        for i, task in enumerate(plan['tasks'], 1):
            print(f"\n   {i}. [{task['task_id']}] {task['name']}")
            print(f"      Description: {task['description']}")
            print(f"      Dependencies: {', '.join(task['dependencies']) if task['dependencies'] else 'None'}")
            print(f"      GPU Hours: {task['gpu_hours']}")
            print(f"      Priority: {task['priority']}")
        
        if plan.get('risk_factors'):
            print(f"\n⚠️  RISK FACTORS:")
            for risk in plan['risk_factors']:
                print(f"   - {risk}")
        
        # Save to database
        print(f"\n{'='*80}")
        print("💾 Saving to database...")
        planner_output = AgentOutputService.create_agent_output(
            db, cycle.id, 'planner', 0.1, plan,
            summary=plan['summary'],
            execution_time_seconds=exec_time
        )
        
        # Create tasks
        tasks = TaskService.create_tasks_from_planner(
            db, planner_output.id, plan['tasks']
        )
        print(f"   ✅ Saved plan and {len(tasks)} tasks to database")
        
        # Update cycle to learner state
        CycleService.update_cycle_state(db, cycle.id, 0.1, 'learner', 'learning')
        
        # ============================================================================
        # LEARNER AGENT
        # ============================================================================
        print(f"\n{'='*80}")
        print("🧠 RUNNING LEARNER AGENT")
        print("="*80)
        
        learner = LearnerAgent(model="llama3.2:3b")
        
        start_time = time.time()
        resources = learner.learn(spec_sheet, plan)
        exec_time = time.time() - start_time
        
        print(f"   ✅ Complete ({exec_time:.1f}s)")
        print(f"\n📚 LEARNER OUTPUT:")
        print(f"   Summary: {resources['summary']}")
        print(f"   Genes: {', '.join(resources['key_genes'][:5])}")
        print(f"   Datasets: {', '.join(resources['datasets'])}")
        
        print(f"\n🔧 TOOLS ({len(resources['tools'])}):")
        for i, tool in enumerate(resources['tools'], 1):
            print(f"   {i}. {tool['name']}: {tool['purpose']}")
        
        print(f"\n💡 PREPROCESSING:")
        print(f"   {resources['preprocessing_notes']}")
        
        print(f"\n🏗️  MODEL SUGGESTIONS:")
        for i, suggestion in enumerate(resources['model_suggestions'], 1):
            print(f"   {i}. {suggestion}")
        
        print(f"\n📖 REFERENCES:")
        for i, ref in enumerate(resources['references'], 1):
            print(f"   {i}. {ref}")
        
        # Save learner output to database
        print(f"\n{'='*80}")
        print("💾 Saving learner output to database...")
        learner_output = AgentOutputService.create_agent_output(
            db, cycle.id, 'learner', 0.2, resources,
            summary=resources['summary'],
            execution_time_seconds=exec_time
        )
        print(f"   ✅ Saved learner output to database")
        
        # Update cycle to executor state
        CycleService.update_cycle_state(db, cycle.id, 0.2, 'executor', 'executing')
        
        # ============================================================================
        # EXECUTOR AGENT
        # ============================================================================
        print(f"\n{'='*80}")
        print("⚙️  RUNNING EXECUTOR AGENT")
        print("="*80)
        
        executor = ExecutorAgent(model="llama3.2:3b")
        
        start_time = time.time()
        execution_results = executor.execute(spec_sheet, plan, resources, 
                                            project_id=str(project.id), 
                                            cycle_id=str(cycle.id))
        exec_time = time.time() - start_time
        
        print(f"   ✅ Complete ({exec_time:.1f}s)")
        print(f"\n🎯 EXECUTOR OUTPUT:")
        print(f"   Summary: {execution_results['summary']}")
        print(f"   Tasks Completed: {len(execution_results['tasks_completed'])}/{len(plan['tasks'])}")
        
        print(f"\n📊 BASELINE RESULTS:")
        baseline = execution_results['baseline_results']
        for key, value in baseline.items():
            print(f"   {key}: {value}")
        
        print(f"\n🚀 MODEL RESULTS:")
        model_res = execution_results['model_results']
        for key, value in model_res.items():
            print(f"   {key}: {value}")
        
        print(f"\n📦 ARTIFACTS ({len(execution_results['artifacts_generated'])}):")
        for i, artifact in enumerate(execution_results['artifacts_generated'][:10], 1):  # Show first 10
            # Show just the filename for readability
            artifact_name = artifact.split('/')[-1] if '/' in artifact else artifact
            print(f"   {i}. {artifact_name}")
        if len(execution_results['artifacts_generated']) > 10:
            print(f"   ... and {len(execution_results['artifacts_generated']) - 10} more")
        
        # Save executor output to database
        print(f"\n{'='*80}")
        print("💾 Saving executor output to database...")
        executor_output_db = AgentOutputService.create_agent_output(
            db, cycle.id, 'executor', 0.3, execution_results,
            summary=execution_results['summary'],
            execution_time_seconds=exec_time
        )
        print(f"   ✅ Saved executor output to database")
        
        # Update cycle to assessor state
        CycleService.update_cycle_state(db, cycle.id, 0.3, 'assessor', 'assessing')
        
        # ============================================================================
        # ASSESSOR AGENT
        # ============================================================================
        print(f"\n{'='*80}")
        print("📊 RUNNING ASSESSOR AGENT")
        print("="*80)
        
        assessor = AssessorAgent(model="llama3.2:3b")
        
        start_time = time.time()
        assessment = assessor.assess(spec_sheet, plan, resources, execution_results)
        exec_time = time.time() - start_time
        
        print(f"   ✅ Complete ({exec_time:.1f}s)")
        
        # Display assessment summary
        print(f"\n📋 ASSESSMENT SUMMARY:")
        print(f"   {assessment['summary']}")
        
        print(f"\n📊 SCORES:")
        print(f"   Satisfaction: {assessment['satisfaction_score']}/5")
        print(f"   NIH Score: {assessment['nih_score']}/9")
        print(f"   Overall Status: {assessment['overall_status']}")
        
        print(f"\n📐 GAP ANALYSIS:")
        for gap in assessment['gap_analysis']:
            status_emoji = "✅" if gap['status'] == "EXCEEDED" else ("🟡" if gap['status'] == "MET" else "❌")
            print(f"   {status_emoji} {gap['metric']}: Goal={gap['goal']:.2f} → Achieved={gap['achieved']:.4f} (Gap: {gap['gap']:+.4f})")
        
        print(f"\n💡 STRENGTHS:")
        for s in assessment['strengths']:
            print(f"   ✓ {s}")
        
        print(f"\n⚠️  WEAKNESSES:")
        for w in assessment['weaknesses']:
            print(f"   • {w}")
        
        print(f"\n📝 RECOMMENDATIONS FOR NEXT ITERATION:")
        for r in assessment['recommendations']:
            print(f"   → {r}")
        
        print(f"\n🔄 Should Continue: {'Yes' if assessment['should_continue'] else 'No - Goals achieved'}")
        
        # Save assessor output to database
        print(f"\n{'='*80}")
        print("💾 Saving assessor output to database...")
        assessor_output_db = AgentOutputService.create_agent_output(
            db, cycle.id, 'assessor', 0.4, assessment,
            summary=assessment['summary'],
            execution_time_seconds=exec_time
        )
        print(f"   ✅ Saved assessor output to database")
        
        # Store Cycle 1 results
        cycle1_assessment = assessment
        cycle1_results = execution_results
        
        # ============================================================================
        # CYCLE 2 - IMPROVEMENT ITERATION (always run to apply improvements)
        # ============================================================================
        MAX_ITERATIONS = 2
        current_cycle = 1
        
        # Always run Cycle 2 to apply improvements before concluding
        if current_cycle < MAX_ITERATIONS:
            print("\n" + "="*80)
            print("🔄 STARTING CYCLE 2 - APPLYING IMPROVEMENTS FROM CYCLE 1")
            print("="*80)
            
            # Mark cycle 1 as completed before starting cycle 2
            CycleService.update_cycle_state(db, cycle.id, 1.0, 'assessor', 'completed')
            print(f"   ✅ Cycle 1 marked as completed")
            
            current_cycle = 2
            recommendations = assessment['recommendations']
            
            print(f"\n📝 RECOMMENDATIONS TO APPLY:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
            
            # Create new cycle
            print("\n🔄 Creating Cycle 2...")
            cycle2 = CycleService.create_cycle(db, project.id, 2)
            
            # Update project's current cycle number
            ProjectService.update_current_cycle(db, project.id, 2)
            print(f"   Cycle 2 ID: {cycle2.id}")
            
            # Update status
            CycleService.update_cycle_state(db, cycle2.id, 0.2, 'executor', 'executing')
            
            # ============================================================================
            # EXECUTOR AGENT - CYCLE 2 (with improvements)
            # ============================================================================
            print(f"\n{'='*80}")
            print("⚙️  RUNNING EXECUTOR AGENT - CYCLE 2 (IMPROVED)")
            print("="*80)
            
            start_time = time.time()
            execution_results_2 = executor.execute(
                spec_sheet, plan, resources, 
                project_id=str(project.id), 
                cycle_id=str(cycle2.id),
                cycle=2,
                recommendations=recommendations
            )
            exec_time = time.time() - start_time
            
            print(f"   ✅ Complete ({exec_time:.1f}s)")
            print(f"\n🎯 EXECUTOR OUTPUT (CYCLE 2):")
            print(f"   Summary: {execution_results_2['summary']}")
            print(f"   Tasks Completed: {len(execution_results_2['tasks_completed'])}/{len(plan['tasks'])}")
            
            print(f"\n📊 IMPROVED RESULTS:")
            improved = execution_results_2['baseline_results']
            for key, value in improved.items():
                print(f"   {key}: {value}")
            
            # Save executor output
            print(f"\n{'='*80}")
            print("💾 Saving executor output (Cycle 2) to database...")
            executor_output_db_2 = AgentOutputService.create_agent_output(
                db, cycle2.id, 'executor', 0.3, execution_results_2,
                summary=execution_results_2['summary'],
                execution_time_seconds=exec_time
            )
            print(f"   ✅ Saved executor output to database")
            
            # Update cycle to assessor state
            CycleService.update_cycle_state(db, cycle2.id, 0.3, 'assessor', 'assessing')
            
            # ============================================================================
            # ASSESSOR AGENT - CYCLE 2
            # ============================================================================
            print(f"\n{'='*80}")
            print("📊 RUNNING ASSESSOR AGENT - CYCLE 2")
            print("="*80)
            
            start_time = time.time()
            assessment_2 = assessor.assess(spec_sheet, plan, resources, execution_results_2)
            exec_time = time.time() - start_time
            
            print(f"   ✅ Complete ({exec_time:.1f}s)")
            
            # Save assessor output
            print(f"\n{'='*80}")
            print("💾 Saving assessor output (Cycle 2) to database...")
            assessor_output_db_2 = AgentOutputService.create_agent_output(
                db, cycle2.id, 'assessor', 0.4, assessment_2,
                summary=assessment_2['summary'],
                execution_time_seconds=exec_time
            )
            print(f"   ✅ Saved assessor output to database")
            
            # Update cycle to completed
            CycleService.update_cycle_state(db, cycle2.id, 0.4, 'pm', 'completed')
            
            # ============================================================================
            # COMPARISON: CYCLE 1 vs CYCLE 2
            # ============================================================================
            print("\n" + "="*80)
            print("📊 IMPROVEMENT COMPARISON: CYCLE 1 vs CYCLE 2")
            print("="*80)
            
            # Get metrics from both cycles
            c1_metrics = cycle1_results.get('baseline_results', {})
            c2_metrics = execution_results_2.get('baseline_results', {})
            
            print(f"\n{'Metric':<15} {'Cycle 1':<12} {'Cycle 2':<12} {'Change':<12}")
            print("-" * 55)
            
            for metric in ['accuracy', 'f1', 'roc_auc']:
                c1_val = c1_metrics.get(metric, 0)
                c2_val = c2_metrics.get(metric, 0)
                if isinstance(c1_val, (int, float)) and isinstance(c2_val, (int, float)):
                    change = c2_val - c1_val
                    change_str = f"{change:+.4f}" if change != 0 else "0.0000"
                    emoji = "📈" if change > 0 else ("📉" if change < 0 else "➡️")
                    print(f"{metric:<15} {c1_val:<12.4f} {c2_val:<12.4f} {emoji} {change_str}")
            
            print(f"\n{'Score':<15} {'Cycle 1':<12} {'Cycle 2':<12} {'Change':<12}")
            print("-" * 55)
            print(f"{'NIH Score':<15} {cycle1_assessment['nih_score']:<12} {assessment_2['nih_score']:<12} {assessment_2['nih_score'] - cycle1_assessment['nih_score']:+d}")
            print(f"{'Satisfaction':<15} {cycle1_assessment['satisfaction_score']:<12} {assessment_2['satisfaction_score']:<12} {assessment_2['satisfaction_score'] - cycle1_assessment['satisfaction_score']:+d}")
            
            # Final assessment
            assessment = assessment_2
            cycle = cycle2
        
        # ============================================================================
        # PM AGENT - FINAL REPORT GENERATION
        # ============================================================================
        print(f"\n{'='*80}")
        print("📋 RUNNING PM AGENT - FINAL REPORT")
        print("="*80)
        
        # Collect all outputs for PM
        executor_outputs = [execution_results]
        assessor_outputs = [cycle1_assessment]
        
        # If cycle 2 ran, add those outputs
        if current_cycle == 2 and 'execution_results_2' in locals():
            executor_outputs.append(execution_results_2)
            assessor_outputs.append(assessment_2)
        
        # Calculate total execution time (approximate)
        total_exec_time = exec_time  # Last exec_time from assessor
        
        pm = PMAgent(model="llama3.2:3b")
        
        start_time = time.time()
        pm_output = pm.manage(
            spec_sheet=spec_sheet,
            planner_output=plan,
            learner_output=resources,
            executor_outputs=executor_outputs,
            assessor_outputs=assessor_outputs,
            total_execution_time=total_exec_time
        )
        pm_exec_time = time.time() - start_time
        
        print(f"\n   ✅ PM Agent complete ({pm_exec_time:.1f}s)")
        
        # Display A.G.E. Scores
        print(f"\n📊 A.G.E. SCORES:")
        for agent_name, score_data in pm_output['age_scores'].items():
            print(f"   • {agent_name.capitalize()}: {score_data['score']}/10")
        print(f"\n   🎯 Overall A.G.E. Score: {pm_output['overall_age_score']}/10")
        
        # Display Report Summary
        print(f"\n📝 REPORT SUMMARY:")
        print(f"   {pm_output['report_summary']}")
        
        # Save PM output to database
        print(f"\n{'='*80}")
        print("💾 Saving PM output and report to database...")
        
        # Save to agent_outputs
        pm_output_db = AgentOutputService.create_agent_output(
            db, cycle.id, 'pm', 1.0, pm_output,
            summary=pm_output['report_summary'],
            execution_time_seconds=pm_exec_time
        )
        print(f"   ✅ Saved PM output to agent_outputs")
        
        # Save report to reports table
        report_db = ReportService.create_report(
            db, 
            cycle_id=cycle.id,
            content=pm_output['report_markdown'],
            final_nih_score=pm_output['final_nih_score'],
            final_bimodal_score=pm_output['final_satisfaction'],
            age_scores=pm_output['age_scores'],
            executive_summary=pm_output['report_summary'],
            recommendations=assessment['recommendations']
        )
        print(f"   ✅ Saved report to reports table (ID: {report_db.id})")
        
        # Update cycle to fully complete (state 1.0)
        CycleService.update_cycle_state(db, cycle.id, 1.0, 'pm', 'completed')
        
        # Update project status
        ProjectService.update_project_status(db, project.id, 'completed')
        
        # ============================================================================
        # FINAL SUMMARY
        # ============================================================================
        print("\n" + "="*80)
        print("✅ ALL AGENTS COMPLETE!")
        print("="*80)
        print(f"\nProject: {project.id}")
        print(f"Total Cycles: {current_cycle}")
        print(f"State: 1.0 (Full pipeline complete)")
        print(f"\n📊 FINAL SCORES:")
        print(f"   NIH Score: {pm_output['final_nih_score']}/9")
        print(f"   Satisfaction: {pm_output['final_satisfaction']}/5")
        print(f"   A.G.E. Score: {pm_output['overall_age_score']}/10")
        print(f"\n📈 Status: {pm_output['project_status']}")
        print(f"\n📄 Report saved to database (reports table)")
        print(f"Database: please.db")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    test_agents()
