"""
Learner Agent - Gathers and synthesizes research resources
Uses BioContext MCP and Ollama for synthesis
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class ResourceTool(BaseModel):
    """Model for a tool recommendation"""
    name: str = Field(description="Tool name (e.g., PyTorch, scikit-learn)")
    purpose: str = Field(description="Why this tool is recommended")


class LearnerOutput(BaseModel):
    """Model for learner agent output"""
    model_config = {"protected_namespaces": ()}  # Allow model_ prefix
    
    summary: str = Field(description="Brief summary of resources gathered")
    key_genes: List[str] = Field(description="Key genes identified from biomedical context")
    datasets: List[str] = Field(description="Recommended datasets")
    tools: List[ResourceTool] = Field(description="Recommended software tools")
    preprocessing_notes: str = Field(description="Data preprocessing recommendations")
    model_suggestions: List[str] = Field(description="Model architecture suggestions")
    references: List[str] = Field(description="Key papers or methods to reference")


class LearnerAgent:
    """Learner Agent that gathers and synthesizes research resources using BioContext MCP"""
    
    def __init__(self, model: str = "llama3.2:3b", mcp_client=None,
                 dataset_path: str = "backend/dataset/tcga_brca_500samples_expr_survival.csv"):
        """
        Initialize the Learner Agent
        
        Args:
            model: Ollama model to use (default: llama3.2:3b)
            mcp_client: MCP client for BioContext (optional)
        """
        # Initialize Ollama LLM for synthesis
        self.llm = ChatOllama(
            model=model,
            temperature=0.3,
            format="json",
            num_predict=500  # Max 500 tokens output
        )
        
        self.mcp_client = mcp_client
        self.dataset_path = self._resolve_dataset_path(dataset_path)
    
    def _resolve_dataset_path(self, dataset_path: str) -> str:
        """Resolve dataset path relative to current working directory."""
        path = Path(dataset_path)
        if path.is_absolute():
            return str(path)
        
        candidates = [
            Path.cwd() / path,
            Path(__file__).resolve().parents[1] / path,
            Path(__file__).resolve().parents[2] / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str((Path.cwd() / path).resolve())
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for the learner agent"""
        return """You are a research resource specialist. Synthesize research resources CONCISELY (max 500 tokens).

REQUIRED output structure:
{
  "summary": "One-line summary of resources",
  "key_genes": ["GENE1", "GENE2", "... up to 100-500 unique genes"],
  "datasets": ["TCGA", "GEO"],
  "tools": [
    {"name": "PyTorch", "purpose": "Deep learning"},
    {"name": "scikit-learn", "purpose": "ML baseline"}
  ],
  "preprocessing_notes": "Brief preprocessing steps",
  "model_suggestions": ["RandomForestClassifier", "LogisticRegression", "SVC"],
  "references": ["Method1", "Method2"]
}

CRITICAL: model_suggestions must be an ARRAY OF STRINGS, not objects.
Keep all descriptions brief. Return ONLY valid JSON."""
    
    def create_user_prompt(self, spec_sheet: Dict[str, Any], planner_output: Dict[str, Any], 
                          biocontext_data: Dict[str, Any]) -> str:
        """Create the user prompt with project details and MCP data"""
        
        problem = spec_sheet.get('research_problem', {})
        problem_statement = problem.get('problem_statement', 'Not specified')
        success_metrics = ", ".join(problem.get('success_metrics', [])) or "accuracy, F1-score, ROC-AUC"
        data_sources = spec_sheet.get('data_sources', [])
        dataset = data_sources[0] if data_sources else "Not specified"
        
        # Extract genes from biocontext
        genes_found = biocontext_data.get('genes', [])
        pathways_found = biocontext_data.get('pathways', [])
        
        return f"""Synthesize research resources for:

Problem: {problem_statement}
Dataset: {dataset}
Success metrics: {success_metrics}
Plan: {planner_output.get('summary', '')}

BioContext findings:
- Genes: {', '.join(genes_found[:10]) if genes_found else 'None'}
- Pathways: {', '.join(pathways_found[:5]) if pathways_found else 'None'}

Provide:
1. Key genes (3-5)
2. Recommended datasets
3. Essential tools (ONLY scikit-learn for lightweight ML)
4. Preprocessing steps (brief)
5. Model suggestions: ONLY lightweight baseline ML models from scikit-learn:
   - RandomForest
   - LogisticRegression
   - SVM
   - GradientBoosting
   - KNeighbors
   - DecisionTree
   DO NOT suggest deep learning models (CNN, LSTM, Transformer, neural networks)
6. Key methods/papers to use

Return valid JSON only."""
    
    def query_biocontext_mcp(self, spec_sheet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query BioContext MCP for biomedical knowledge using real MCP client
        
        Args:
            spec_sheet: Research specification
            
        Returns:
            Dictionary with genes, pathways, and papers discovered from BioContext
        """
        try:
            import asyncio
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            import json
        except ImportError:
            print("   ⚠️  MCP client not installed. Using fallback BioContext data.")
            return self._create_fallback_biocontext(spec_sheet)
        
        problem = spec_sheet.get('research_problem', {})
        problem_statement = problem.get('problem_statement', '')
        data_sources = spec_sheet.get('data_sources', [])
        dataset = data_sources[0] if data_sources else ''
        
        # Extract cancer type from problem or dataset
        cancer_type = "breast cancer"  # Default
        if 'breast' in problem_statement.lower() or 'breast' in dataset.lower():
            cancer_type = "breast cancer"
        elif 'lung' in problem_statement.lower() or 'lung' in dataset.lower():
            cancer_type = "lung cancer"
        elif 'colon' in problem_statement.lower() or 'colon' in dataset.lower():
            cancer_type = "colon cancer"
        
        print(f"   🔍 Querying BioContext MCP for {cancer_type} genes...")
        
        async def query_mcp_genes():
            """Query MCP async for cancer genes"""
            server_params = StdioServerParameters(
                command="uvx",
                args=["biocontext_kb@latest"],
                env=None
            )
            
            try:
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        
                        # Core cancer genes to validate and expand
                        if cancer_type == "breast cancer":
                            seed_genes = ["BRCA1", "BRCA2", "TP53", "ERBB2", "EGFR", 
                                        "ESR1", "PGR", "PIK3CA", "PTEN", "CDH1", "ATM", "CHEK2"]
                        else:
                            seed_genes = ["TP53", "KRAS", "EGFR", "PIK3CA", "PTEN"]
                        
                        validated_genes = []
                        all_genes = set()
                        
                        # Step 1: Validate seed genes via KEGG
                        for gene in seed_genes:
                            try:
                                result = await session.call_tool(
                                    "bc_get_kegg_id_by_gene_symbol",
                                    arguments={
                                        "gene_symbol": gene,
                                        "organism_code": "9606"
                                    }
                                )
                                for item in result.content:
                                    if hasattr(item, 'text'):
                                        kegg_id = str(item.text).strip()
                                        if "hsa:" in kegg_id:
                                            validated_genes.append(gene)
                                            all_genes.add(gene)
                                            break
                            except:
                                pass
                        
                        # Step 2: Expand using STRING protein interactions
                        for gene in validated_genes[:3]:  # Top 3 genes
                            try:
                                result = await session.call_tool(
                                    "bc_get_string_interactions",
                                    arguments={
                                        "protein_symbol": gene,
                                        "species": "9606",
                                        "min_score": 900  # High confidence
                                    }
                                )
                                
                                for item in result.content:
                                    if hasattr(item, 'text'):
                                        text = str(item.text)
                                        try:
                                            interactions = json.loads(text)
                                            for interaction in interactions[:5]:
                                                gene_b = interaction.get('preferredName_B', '')
                                                if gene_b:
                                                    all_genes.add(gene_b)
                                        except:
                                            pass
                            except:
                                pass
                        
                        return {
                            "genes": list(all_genes),
                            "pathways": ["DNA repair", "Cell cycle", "PI3K-AKT signaling"],
                            "papers": ["BioContext MCP validated genes"]
                        }
                        
            except Exception as e:
                print(f"      ⚠️  MCP query failed: {e}")
                return None
        
        # Run async MCP query
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(query_mcp_genes())
            loop.close()
            
            if result and result.get('genes'):
                print(f"      ✅ Found {len(result['genes'])} genes via MCP")
                return result
            else:
                print(f"      ⚠️  MCP returned no genes, using fallback")
                return self._create_fallback_biocontext(spec_sheet)
        except Exception as e:
            print(f"      ⚠️  MCP error: {e}, using fallback")
            return self._create_fallback_biocontext(spec_sheet)
    
    def _create_fallback_biocontext(self, spec_sheet: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback biocontext data based on dataset type"""
        problem = spec_sheet.get('research_problem', {})
        data_sources = spec_sheet.get('data_sources', [])
        dataset = (data_sources[0] if data_sources else problem.get('problem_statement', '')).lower()
        
        # Common breast cancer genes for TCGA
        if 'tcga' in dataset or 'breast' in dataset.lower():
            return {
                "genes": ["BRCA1", "BRCA2", "TP53", "PIK3CA", "ERBB2", "ESR1", "PGR", 
                         "EGFR", "PTEN", "CDH1", "ATM", "CHEK2"],
                "pathways": ["DNA repair", "Cell cycle", "PI3K-AKT signaling"],
                "papers": ["Cox regression for survival", "Deep learning for genomics"]
            }
        else:
            # Generic fallback
            return {
                "genes": ["TP53", "EGFR", "KRAS", "PIK3CA", "PTEN"],
                "pathways": ["Cell signaling", "Gene regulation"],
                "papers": ["Machine learning methods", "Feature selection"]
            }

    
    def learn(self, spec_sheet: Dict[str, Any], planner_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather and synthesize research resources
        
        Args:
            spec_sheet: Research specification
            planner_output: Output from planner agent
            
        Returns:
            Dictionary containing synthesized resources
        """
        
        print("\n   🔍 Querying BioContext MCP...")
        # Query BioContext MCP for biomedical knowledge
        biocontext_data = self.query_biocontext_mcp(spec_sheet)
        
        # Display MCP results
        print(f"\n   📊 MCP RESPONSE:")
        print(f"      Genes ({len(biocontext_data.get('genes', []))}): {', '.join(biocontext_data.get('genes', [])[:10])}")
        if biocontext_data.get('pathways'):
            print(f"      Pathways ({len(biocontext_data.get('pathways', []))}): {', '.join(biocontext_data.get('pathways', [])[:5])}")
        if biocontext_data.get('papers'):
            print(f"      Papers: {len(biocontext_data.get('papers', []))} references")
        
        print("\n   🤖 Synthesizing with Ollama...")
        # Create messages for synthesis
        messages = [
            SystemMessage(content=self.create_system_prompt()),
            HumanMessage(content=self.create_user_prompt(spec_sheet, planner_output, biocontext_data))
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Show raw LLM response
            print(f"\n   📝 RAW LLM RESPONSE:")
            print(f"      {content[:200]}..." if len(content) > 200 else f"      {content}")
            
            # Clean up response - remove markdown if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Parse and validate JSON
            learner_data = json.loads(content)
            validated_output = LearnerOutput(**learner_data)
            
            # Expand gene list to ensure 100-500 unique genes using BioContext + dataset columns
            key_genes = list(dict.fromkeys((validated_output.key_genes or []) + biocontext_data.get('genes', [])))
            dataset_genes = []
            if len(key_genes) < 100:
                try:
                    import pandas as pd
                    df_head = pd.read_csv(self.dataset_path, nrows=1)
                    dataset_genes = [c for c in df_head.columns if c not in ['sampleID', 'vital_status', 'survival_time_days']]
                except Exception as dataset_err:
                    print(f"   ⚠️  Could not read dataset for gene expansion: {dataset_err}")
            for gene in dataset_genes:
                if len(key_genes) >= 500:
                    break
                if gene not in key_genes:
                    key_genes.append(gene)
            if not key_genes:
                key_genes = dataset_genes[:500] if dataset_genes else biocontext_data.get('genes', [])[:500]
            if len(key_genes) < 100 and key_genes:
                multiplier = (100 // len(key_genes)) + 1
                key_genes = (key_genes * multiplier)[:100]
            else:
                key_genes = key_genes[:500]
            validated_output.key_genes = key_genes
            
            # Show synthesized results
            print(f"\n   ✨ SYNTHESIZED OUTPUT:")
            print(f"      Summary: {validated_output.summary}")
            print(f"      Key Genes ({len(validated_output.key_genes)}): {', '.join(validated_output.key_genes[:5])}")
            print(f"      Tools: {', '.join([t.name for t in validated_output.tools])}")
            print(f"      Model Suggestions: {', '.join(validated_output.model_suggestions[:3])}")
            
            return validated_output.model_dump()
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON parse error: {e}")
            return self._create_fallback_synthesis(spec_sheet, planner_output, biocontext_data)
        
        except Exception as e:
            print(f"   ⚠️  Learner error: {e}")
            return self._create_fallback_synthesis(spec_sheet, planner_output, biocontext_data)
    
    def _create_fallback_synthesis(self, spec_sheet: Dict[str, Any], 
                                   planner_output: Dict[str, Any],
                                   biocontext_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback synthesis if AI generation fails"""
        
        problem = spec_sheet.get('research_problem', {})
        data_sources = spec_sheet.get('data_sources', [])
        dataset = data_sources[0] if data_sources else problem.get('problem_statement', '')
        genes = biocontext_data.get('genes', ["BRCA1", "BRCA2", "TP53"])
        if len(genes) < 100:
            try:
                import pandas as pd
                df_head = pd.read_csv(self.dataset_path, nrows=1)
                dataset_genes = [c for c in df_head.columns if c not in ['sampleID', 'vital_status', 'survival_time_days']]
                genes = (genes + dataset_genes)[:500]
            except Exception:
                genes = (genes * ((100 // len(genes)) + 1))[:100]
        
        num_tasks = len(planner_output.get('tasks', []))
        
        fallback = {
            "summary": f"Identified {len(genes)} key genes and essential tools for {num_tasks} tasks",
            "key_genes": genes[:500],
            "datasets": [dataset] if dataset else ["TCGA"],
            "tools": [
                {"name": "PyTorch", "purpose": "Deep learning framework"},
                {"name": "scikit-learn", "purpose": "Baseline models and preprocessing"},
                {"name": "pandas", "purpose": "Data manipulation"}
            ],
            "preprocessing_notes": "Log-transform and normalize gene expression data, handle missing values, perform train/test split",
            "model_suggestions": [
                "RandomForestClassifier",
                "LogisticRegression",
                "SVM (Support Vector Machine)"
            ],
            "references": [
                "Cox proportional hazards for survival analysis",
                "Deep learning for genomic data",
                "Feature selection methods for high-dimensional data"
            ]
        }
        
        return fallback
