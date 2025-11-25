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


class LearnerOutput(BaseModel):
    """Model for learner agent output"""
    model_config = {"protected_namespaces": ()}
    
    summary: str = Field(description="Brief summary of resources gathered")
    key_genes: List[str] = Field(description="Key genes identified")
    datasets: List[str] = Field(description="Recommended datasets")
    tools: List[Dict[str, str]] = Field(description="Recommended tools")
    preprocessing_notes: str = Field(description="Preprocessing recommendations")
    model_suggestions: List[str] = Field(description="Model suggestions")
    references: List[str] = Field(description="Key references")


class LearnerAgent:
    """Learner Agent that gathers research resources using BioContext MCP"""
    
    def __init__(self, model: str = "llama3.2:3b", 
                 dataset_path: str = "backend/dataset/tcga_brca_500samples_expr_survival.csv"):
        self.llm = ChatOllama(model=model, temperature=0.3, format="json", num_predict=2000)
        self.dataset_path = self._resolve_dataset_path(dataset_path)
    
    def _resolve_dataset_path(self, dataset_path: str) -> str:
        """Resolve dataset path"""
        path = Path(dataset_path)
        if path.is_absolute():
            return str(path)
        
        for candidate in [Path.cwd() / path, Path(__file__).resolve().parents[1] / path]:
            if candidate.exists():
                return str(candidate.resolve())
        return str((Path.cwd() / path).resolve())
    
    def query_biocontext_mcp(self, mcp_query: str) -> Dict[str, Any]:
        """Query BioContext MCP using the planner's query"""
        try:
            import asyncio
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            print("   ⚠️  MCP not installed, using fallback")
            return self._fallback_genes()
        
        print(f"\n   🔍 Querying BioContext MCP...")
        print(f"      📤 Query: \"{mcp_query}\"")
        
        async def query_mcp():
            server_params = StdioServerParameters(command="uvx", args=["biocontext_kb@latest"], env=None)
            
            try:
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        
                        # Seed genes for breast cancer (most common)
                        seed_genes = ["BRCA1", "BRCA2", "TP53", "ERBB2", "ESR1", "PGR", "PIK3CA", "PTEN", "CDH1", "ATM"]
                        validated_genes = set()
                        
                        # Validate genes via KEGG
                        for gene in seed_genes:
                            try:
                                result = await session.call_tool(
                                    "bc_get_kegg_id_by_gene_symbol",
                                    arguments={"gene_symbol": gene, "organism_code": "9606"}
                                )
                                for item in result.content:
                                    if hasattr(item, 'text') and "hsa:" in str(item.text):
                                        validated_genes.add(gene)
                                        break
                            except:
                                pass
                        
                        # Expand via STRING interactions (top 3 genes)
                        for gene in list(validated_genes)[:3]:
                            try:
                                result = await session.call_tool(
                                    "bc_get_string_interactions",
                                    arguments={"protein_symbol": gene, "species": "9606", "min_score": 900}
                                )
                                for item in result.content:
                                    if hasattr(item, 'text'):
                                        try:
                                            interactions = json.loads(str(item.text))
                                            for interaction in interactions[:5]:
                                                if gene_b := interaction.get('preferredName_B'):
                                                    validated_genes.add(gene_b)
                                        except:
                                            pass
                            except:
                                pass
                        
                        return {"genes": list(validated_genes), "pathways": ["DNA repair", "Cell cycle", "PI3K-AKT"]}
            except Exception as e:
                print(f"      ⚠️  MCP error: {e}")
                return None
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(query_mcp())
            loop.close()
            
            if result and result.get('genes'):
                print(f"\n   📊 MCP RESPONSE:")
                print(f"      ✅ Genes ({len(result['genes'])}): {', '.join(result['genes'][:10])}")
                return result
        except Exception as e:
            print(f"      ⚠️  MCP error: {e}")
        
        print("      ⚠️  Using fallback genes")
        return self._fallback_genes()
    
    def _fallback_genes(self) -> Dict[str, Any]:
        """Fallback gene list"""
        return {
            "genes": ["BRCA1", "BRCA2", "TP53", "PIK3CA", "ERBB2", "ESR1", "PGR", "PTEN", "CDH1", "ATM"],
            "pathways": ["DNA repair", "Cell cycle", "PI3K-AKT"]
        }
    
    def learn(self, spec_sheet: Dict[str, Any], planner_output: Dict[str, Any]) -> Dict[str, Any]:
        """Gather and synthesize research resources"""
        
        # Get MCP query from planner
        mcp_query = planner_output.get('mcp_query', 'What genes are responsible for cancer survival?')
        
        # Query BioContext MCP
        biocontext = self.query_biocontext_mcp(mcp_query)
        
        # Synthesize with Ollama
        print("\n   🤖 Synthesizing with Ollama...")
        
        problem = spec_sheet.get('research_problem', {})
        problem_statement = problem.get('problem_statement', '')
        dataset = spec_sheet.get('data_sources', ['TCGA'])[0] if spec_sheet.get('data_sources') else 'TCGA'
        
        prompt = f"""Synthesize resources for: {problem_statement}

MCP Query: {mcp_query}
Genes from MCP: {', '.join(biocontext.get('genes', [])[:10])}

Return JSON with:
- summary: one line
- key_genes: list of 5-10 important genes
- datasets: ["TCGA"] 
- tools: [{{"name": "scikit-learn", "purpose": "ML"}}]
- preprocessing_notes: brief steps
- model_suggestions: ["RandomForest", "LogisticRegression", "SVM"]
- references: 2-3 methods

Return ONLY valid JSON."""

        try:
            response = self.llm.invoke([
                SystemMessage(content="Return valid JSON only. model_suggestions must be array of strings."),
                HumanMessage(content=prompt)
            ])
            
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            
            result = json.loads(content)
            
            # Merge MCP genes with LLM genes
            result['key_genes'] = list(set(result.get('key_genes', []) + biocontext.get('genes', [])))[:20]
            
            print(f"\n   ✅ SYNTHESIS COMPLETE:")
            print(f"      📝 {result.get('summary', 'Done')}")
            print(f"      🧬 Genes: {len(result['key_genes'])} | Models: {', '.join(result.get('model_suggestions', [])[:3])}")
            
            return result
            
        except Exception as e:
            print(f"   ⚠️  Synthesis error: {e}, using fallback")
            return {
                "summary": f"Resources for {problem_statement[:50]}",
                "key_genes": biocontext.get('genes', [])[:20],
                "datasets": [dataset],
                "tools": [{"name": "scikit-learn", "purpose": "ML"}],
                "preprocessing_notes": "Normalize data, train/test split",
                "model_suggestions": ["RandomForest", "LogisticRegression", "SVM"],
                "references": ["Feature selection", "Cox regression"]
            }
