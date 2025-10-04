from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import ollama
import pandas as pd
import pdfplumber
import subprocess
import sys
from pathlib import Path
import importlib.util
import re

class AgentState(TypedDict):
    target_bank: str
    pdf_path: str
    csv_path: str 
    plan: dict
    generated_code: str
    test_results: dict
    attempt: int
    max_attempts: int
    status: Literal["planning", "generating", "testing", "refining", "success", "failed"]

class OpenSourceCodingAgent:
    def __init__(self, model_name: str = "mistral:latest"):
        self.model_name = model_name
        self.graph = self._create_agent_graph()
    
    def _create_agent_graph(self):
        """Create the LangGraph agent workflow"""
        builder = StateGraph(AgentState)
        
        builder.add_node("plan", self.plan_node)
        builder.add_node("generate", self.generate_node)
        builder.add_node("test", self.test_node)
        builder.add_node("refine", self.refine_node)
        
        builder.set_entry_point("plan")
        builder.add_edge("plan", "generate")
        builder.add_edge("generate", "test")
        
        builder.add_conditional_edges(
            "test",
            self.should_continue,
            {
                "continue": "refine", 
                "end": END
            }
        )
        builder.add_edge("refine", "generate")
        
        return builder.compile()
    
    def plan_node(self, state: AgentState) -> AgentState:
        """Analyze PDF structure and CSV schema to create a detailed plan"""
        print(" Analyzing ICICI bank statement structure...")
        
        pdf_analysis = self.analyze_pdf_structure(state["pdf_path"])
        csv_schema = self.analyze_csv_schema(state["csv_path"])
        
        # Extract specific patterns from the PDF
        patterns = self.extract_transaction_patterns(state["pdf_path"])
        
        plan = {
            "pdf_analysis": pdf_analysis,
            "csv_schema": csv_schema,
            "transaction_patterns": patterns,
            "target_bank": state["target_bank"]
        }
        
        # Safe printing with error handling
        total_pages = pdf_analysis.get('total_pages', 'unknown')
        sample_tx_count = len(patterns.get('sample_transactions', []))
        print(f" Analysis complete: {total_pages} pages, {sample_tx_count} sample transactions found")
        
        return {**state, "plan": plan, "status": "planning"}
    
    def analyze_pdf_structure(self, pdf_path: str) -> dict:
        """Extract detailed structure from PDF"""
        try:
            # Check if file exists
            if not Path(pdf_path).exists():
                return {"error": f"PDF file not found: {pdf_path}", "total_pages": 0}
                
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""
                transaction_sections = []
                
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    all_text += f"\\n--- Page {i+1} ---\\n{text}"
                    
                    # Look for transaction tables/patterns
                    if any(keyword in text.lower() for keyword in ["date", "transaction", "amount", "balance", "withdrawal", "deposit"]):
                        transaction_sections.append({
                            "page": i+1,
                            "text": text[:1000]  # First 1000 chars
                        })
                
                return {
                    "total_pages": len(pdf.pages),
                    "sample_text": all_text[:3000],  # First 3000 chars
                    "transaction_sections": transaction_sections[:3],  # First 3 sections
                    "file_path": pdf_path,
                    "success": True
                }
        except Exception as e:
            return {"error": f"PDF analysis failed: {str(e)}", "total_pages": 0, "success": False}
    
    def extract_transaction_patterns(self, pdf_path: str) -> dict:
        """Extract specific transaction patterns from the PDF"""
        try:
            if not Path(pdf_path).exists():
                return {"error": f"PDF file not found: {pdf_path}"}
                
            with pdfplumber.open(pdf_path) as pdf:
                patterns = {
                    "date_patterns": set(),
                    "amount_patterns": set(),
                    "description_patterns": set(),
                    "sample_transactions": []
                }
                
                # Analyze first few pages for patterns
                for page in pdf.pages[:3]:
                    text = page.extract_text() or ""
                    lines = text.split('\\n')
                    
                    # Look for date patterns
                    date_matches = re.findall(r'\\d{1,2}/\\d{1,2}/\\d{2,4}', text)
                    patterns["date_patterns"].update(date_matches)
                    
                    # Look for amount patterns
                    amount_matches = re.findall(r'\\d{1,3}(?:,\\d{3})*\\.\\d{2}', text)
                    patterns["amount_patterns"].update(amount_matches)
                    
                    # Sample lines that might be transactions
                    for line in lines:
                        if any(keyword in line.lower() for keyword in ["payment", "transfer", "withdrawal", "deposit", "charge"]):
                            patterns["sample_transactions"].append(line.strip()[:200])
                            if len(patterns["sample_transactions"]) > 10:
                                break
                
                # Convert sets to lists for JSON serialization
                patterns["date_patterns"] = list(patterns["date_patterns"])[:5]
                patterns["amount_patterns"] = list(patterns["amount_patterns"])[:5]
                patterns["sample_transactions"] = patterns["sample_transactions"][:5]
                patterns["success"] = True
                
                return patterns
                
        except Exception as e:
            return {"error": f"Pattern extraction failed: {str(e)}", "success": False}
    
    def analyze_csv_schema(self, csv_path: str) -> dict:
        """Understand the expected output format in detail"""
        try:
            if not Path(csv_path).exists():
                return {"error": f"CSV file not found: {csv_path}"}
                
            df = pd.read_csv(csv_path)
            return {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_data": df.head(3).fillna("").to_dict('records'),
                "row_count": len(df),
                "date_format": self.infer_date_format(df) if 'Date' in df.columns else "unknown",
                "file_path": csv_path,
                "success": True
            }
        except Exception as e:
            return {"error": f"CSV analysis failed: {str(e)}", "success": False}
    
    def infer_date_format(self, df: pd.DataFrame) -> str:
        """Try to infer the date format from the DataFrame"""
        if 'Date' not in df.columns:
            return "unknown"
        
        sample_dates = df['Date'].dropna().head(3)
        if len(sample_dates) > 0:
            return f"Sample: {sample_dates.iloc[0]}"
        return "unknown"
    
    def generate_node(self, state: AgentState) -> AgentState:
        """Generate ICICI-specific parser code using Ollama"""
        print(f" Generating ICICI parser code (Attempt {state['attempt'] + 1}/{state['max_attempts']})...")
        
        prompt = self._create_icici_specific_prompt(state["plan"])
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system', 
                        'content': '''You are an expert Python developer specializing in ICICI bank statement parsing. 
                        Create a SPECIFIC parser for ICICI bank statements. 
                        Provide ONLY valid Python code without any explanations or markdown formatting.'''
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.1}  # Lower temperature for more consistent code
            )
            
            generated_code = response['message']['content']
            
            # Clean the code more aggressively
            generated_code = self._clean_generated_code(generated_code)
            
            print(f" Generated {len(generated_code)} characters of code")
            return {**state, "generated_code": generated_code, "status": "generating"}
            
        except Exception as e:
            print(f" Code generation failed: {e}")
            # Return placeholder code to continue the flow
            placeholder_code = '''
import pandas as pd
import pdfplumber

def parse(pdf_path: str) -> pd.DataFrame:
    """ICICI Bank Statement Parser"""
    # Placeholder - generation failed
    return pd.DataFrame()
'''
            return {**state, "generated_code": placeholder_code, "status": "generating"}
    
    def _clean_generated_code(self, code: str) -> str:
        """Remove markdown and non-code elements"""
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        # Remove any remaining markdown
        code = re.sub(r'^```.*$', '', code, flags=re.MULTILINE)
        
        return code.strip()
    
    def _create_icici_specific_prompt(self, plan: dict) -> str:
        """Create a highly specific prompt for ICICI bank parsing"""
        
        # Safely extract plan components with defaults
        pdf_analysis = plan.get("pdf_analysis", {})
        csv_schema = plan.get("csv_schema", {})
        patterns = plan.get("transaction_patterns", {})
        
        # Safe text extraction with length limits
        sample_text = str(pdf_analysis.get("sample_text", ""))[:800]
        date_patterns = patterns.get("date_patterns", [])
        amount_patterns = patterns.get("amount_patterns", [])
        sample_transactions = patterns.get("sample_transactions", [])
        
        return f'''
Create a Python parser SPECIFICALLY for ICICI bank statements.

TARGET: ICICI Bank Statement Parser

PDF STRUCTURE ANALYSIS:
- Total pages: {pdf_analysis.get("total_pages", "unknown")}
- Transaction sections found: {len(pdf_analysis.get("transaction_sections", []))}
- Sample text: {sample_text}

TRANSACTION PATTERNS FOUND:
- Date patterns: {date_patterns}
- Amount patterns: {amount_patterns}
- Sample transactions: {sample_transactions}

EXPECTED OUTPUT (from CSV):
- Columns: {csv_schema.get("columns", [])}
- Data types: {csv_schema.get("dtypes", {})}
- Sample data: {csv_schema.get("sample_data", [])}
- Total rows expected: {csv_schema.get("row_count", 0)}

REQUIREMENTS:
1. Function must be: parse(pdf_path: str) -> pd.DataFrame
2. Output must match the CSV schema exactly: {csv_schema.get("columns", [])}
3. Use pdfplumber for text extraction (NOT PyPDF2 or tabula)
4. Focus on ICICI-specific format patterns
5. Handle date parsing, numeric extraction, text cleaning
6. Return empty DataFrame if no transactions found
7. Include proper error handling

IMPORTANT: Create a SPECIFIC parser for ICICI bank statements, not a generic bank parser.
Look for ICICI-specific patterns in the transaction data.

Provide only the Python code without any explanations, comments, or markdown formatting.
'''

    def test_node(self, state: AgentState) -> AgentState:
        """Test the generated parser against expected CSV"""
        print(" Testing generated ICICI parser...")
        
        # Save the generated code
        parser_path = f'custom_parsers/{state["target_bank"]}_parser.py'
        with open(parser_path, 'w', encoding='utf-8') as f:
            f.write(state["generated_code"])
        
        # Test the parser
        test_results = self._test_parser(parser_path, state["pdf_path"], state["csv_path"])
        
        if test_results["success"]:
            print(" Parser test PASSED!")
        else:
            print(" Parser test FAILED")
            if test_results.get("error"):
                print(f"   Error: {test_results['error']}")
            print(f"   Columns: {test_results.get('result_columns', [])} vs expected: {test_results.get('expected_columns', [])}")
        
        return {**state, "test_results": test_results, "status": "testing"}
    
    def _test_parser(self, parser_path: str, pdf_path: str, csv_path: str) -> dict:
        """Run the parser and compare with expected CSV"""
        try:
            # Clear any cached modules
            if "generated_parser" in sys.modules:
                del sys.modules["generated_parser"]
            
            # Dynamically import the generated parser
            spec = importlib.util.spec_from_file_location("generated_parser", parser_path)
            parser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parser_module)
            
            # Run the parse function
            result_df = parser_module.parse(pdf_path)
            
            # Load expected result
            expected_df = pd.read_csv(csv_path)
            
            # Compare results more thoroughly
            columns_match = list(result_df.columns) == list(expected_df.columns)
            shape_match = result_df.shape == expected_df.shape
            
            # Basic data validation
            data_valid = False
            if columns_match and shape_match and len(result_df) > 0:
                # Check if we have at least some non-null data
                data_valid = not result_df.isnull().all().all()
            
            return {
                "success": columns_match and shape_match and data_valid,
                "columns_match": columns_match,
                "shape_match": shape_match,
                "data_valid": data_valid,
                "result_columns": list(result_df.columns),
                "expected_columns": list(expected_df.columns),
                "result_shape": result_df.shape,
                "expected_shape": expected_df.shape,
                "result_sample": result_df.head(2).fillna("").to_dict('records') if len(result_df) > 0 else [],
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "columns_match": False,
                "shape_match": False,
                "data_valid": False
            }
    
    def refine_node(self, state: AgentState) -> AgentState:
        """Analyze test failures and prepare for next generation attempt"""
        print(" Analyzing test results and refining approach...")
        
        # Add error information to plan for next attempt
        if state["test_results"].get("error"):
            state["plan"]["previous_error"] = state["test_results"]["error"]
        if not state["test_results"].get("columns_match"):
            state["plan"]["column_mismatch"] = {
                "expected": state["test_results"].get("expected_columns", []),
                "actual": state["test_results"].get("result_columns", [])
            }
        
        # Increment attempt counter
        new_attempt = state["attempt"] + 1
        print(f" Refinement round {new_attempt}/{state['max_attempts']}")
        
        return {**state, "attempt": new_attempt, "status": "refining"}
    
    def should_continue(self, state: AgentState) -> str:
        """Decide whether to continue refining or end"""
        if state["test_results"].get("success", False):
            print(" Parser successfully created and tested!")
            return "end"
        elif state["attempt"] >= state["max_attempts"] - 1:
            print(f" Max attempts reached ({state['max_attempts']})")
            return "end"
        else:
            print(" Test failed, attempting refinement...")
            return "continue"
    
    def run(self, target_bank: str):
        """Run the agent for a specific bank"""
        print(f" Starting AI agent for {target_bank.upper()} bank statement parser...")
        print("=" * 60)
        
        # Verify files exist first
        pdf_path = f"data/{target_bank}/{target_bank}_sample.pdf"
        csv_path = f"data/{target_bank}/result.csv"
        
        if not Path(pdf_path).exists():
            print(f" PDF file not found: {pdf_path}")
            return {"status": "failed", "error": "PDF file not found"}
        
        if not Path(csv_path).exists():
            print(f" CSV file not found: {csv_path}")
            return {"status": "failed", "error": "CSV file not found"}
        
        initial_state = {
            "target_bank": target_bank,
            "pdf_path": pdf_path,
            "csv_path": csv_path, 
            "plan": {},
            "generated_code": "",
            "test_results": {},
            "attempt": 0,
            "max_attempts": 3,
            "status": "planning"
        }
        
        # Ensure custom_parsers directory exists
        Path("custom_parsers").mkdir(exist_ok=True)
        
        # Run the agent
        final_state = self.graph.invoke(initial_state)
        
        print("=" * 60)
        if final_state["test_results"].get("success"):
            print(f" SUCCESS: Parser created at custom_parsers/{target_bank}_parser.py")
        else:
            print(f" FAILED: Could not create working parser after {final_state['attempt'] + 1} attempts")
            if final_state["test_results"].get("error"):
                print(f"   Last error: {final_state['test_results']['error']}")
        
        return final_state

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Coding Agent for Bank Statement Parsers")
    parser.add_argument("--target", required=True, help="Target bank name (e.g., icici)")
    args = parser.parse_args()
    
    agent = OpenSourceCodingAgent()
    result = agent.run(args.target)
