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

class FixedCodingAgent:
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
        """Analyze PDF structure and CSV schema"""
        print(" Analyzing ICICI bank statement structure...")
        
        pdf_analysis = self.analyze_pdf_structure(state["pdf_path"])
        csv_schema = self.analyze_csv_schema(state["csv_path"])
        
        plan = {
            "pdf_analysis": pdf_analysis,
            "csv_schema": csv_schema,
            "target_bank": state["target_bank"]
        }
        
        print(f" Found: {pdf_analysis.get('total_pages', 0)} pages, CSV has {csv_schema.get('row_count', 0)} rows")
        return {**state, "plan": plan, "status": "planning"}
    
    def analyze_pdf_structure(self, pdf_path: str) -> dict:
        """Extract structure from PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                sample_text = ""
                for i, page in enumerate(pdf.pages[:2]):
                    text = page.extract_text() or ""
                    sample_text += f"\\n--- Page {i+1} ---\\n{text[:500]}"
                
                return {
                    "total_pages": len(pdf.pages),
                    "sample_text": sample_text,
                    "file_path": pdf_path,
                    "success": True
                }
        except Exception as e:
            return {"error": f"PDF analysis failed: {str(e)}", "success": False}
    
    def analyze_csv_schema(self, csv_path: str) -> dict:
        """Understand the expected output format"""
        try:
            df = pd.read_csv(csv_path)
            return {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_data": df.head(2).fillna("").to_dict('records'),
                "row_count": len(df),
                "file_path": csv_path,
                "success": True
            }
        except Exception as e:
            return {"error": f"CSV analysis failed: {str(e)}", "success": False}
    
    def generate_node(self, state: AgentState) -> AgentState:
        """Generate parser code using Ollama with better guidance"""
        print(f" Generating parser code (Attempt {state['attempt'] + 1}/3)...")
        
        prompt = self._create_smart_prompt(state["plan"], state.get("test_results", {}))
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system', 
                        'content': '''You are an expert Python developer. Create bank statement parsers using pdfplumber.
                        CRITICAL: Never use .seek() on strings. pdf.extract_text() returns a string, not a file object.
                        Return ONLY valid Python code without explanations.'''
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.1}
            )
            
            generated_code = self._clean_generated_code(response['message']['content'])
            print(f" Generated {len(generated_code)} characters")
            return {**state, "generated_code": generated_code, "status": "generating"}
            
        except Exception as e:
            print(f" Generation failed: {e}")
            # Fallback to working code template
            fallback_code = self._get_fallback_code()
            return {**state, "generated_code": fallback_code, "status": "generating"}
    
    def _create_smart_prompt(self, plan: dict, previous_results: dict) -> str:
        """Create a smart prompt that learns from previous errors"""
        
        error_context = ""
        if previous_results.get("error"):
            error_context = f'''
PREVIOUS ERROR: {previous_results["error"]}
FIX THIS: The error was caused by calling .seek() on a string. pdfplumber.open() returns a PDF object, not a file.
'''
        
        return f'''
Create a Python parser for ICICI bank statements using pdfplumber.

{error_context}

PDF INFO:
- Pages: {plan["pdf_analysis"].get("total_pages", "unknown")}
- Sample: {plan["pdf_analysis"].get("sample_text", "")[:800]}

EXPECTED OUTPUT:
- Columns: {plan["csv_schema"].get("columns", [])}
- Sample: {plan["csv_schema"].get("sample_data", [])}

CRITICAL REQUIREMENTS:
1. Function: parse(pdf_path: str) -> pd.DataFrame
2. Use pdfplumber correctly - NEVER call .seek() on strings
3. Extract tables with page.extract_tables()
4. Handle different page layouts
5. Return DataFrame with columns: {plan["csv_schema"].get("columns", [])}

CORRECT pdfplumber pattern:
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()  #  Correct
        text = page.extract_text()     #  Returns string

INCORRECT pattern:
text = pdf.extract_text()
text.seek(0)  #  WRONG: text is string, not file

Return ONLY the Python code:
'''
    
    def _get_fallback_code(self) -> str:
        """Return a working fallback code template"""
        return '''import pdfplumber
import pandas as pd
import re

def parse(pdf_path: str) -> pd.DataFrame:
    """Parse ICICI bank statement PDF"""
    all_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract tables from the page
                tables = page.extract_tables()
                
                for table in tables:
                    if table and len(table) > 1:
                        # Use first row as header
                        header = []
                        for cell in table[0]:
                            if cell and str(cell).strip():
                                header.append(str(cell).strip())
                        
                        # Process data rows
                        for row in table[1:]:
                            if any(cell and str(cell).strip() for cell in row):
                                cleaned_row = []
                                for cell in row:
                                    if cell and str(cell).strip():
                                        cleaned_row.append(str(cell).strip())
                                    else:
                                        cleaned_row.append(None)
                                
                                if len(cleaned_row) == len(header):
                                    all_data.append(cleaned_row)
        
        if all_data:
            df = pd.DataFrame(all_data, columns=header)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return pd.DataFrame()
'''
    
    def _clean_generated_code(self, code: str) -> str:
        """Remove markdown and fix common issues"""
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        # Remove .seek() calls that cause the error
        code = re.sub(r'\\.seek\\([^)]*\\)', '', code)
        
        return code.strip()
    
    def test_node(self, state: AgentState) -> AgentState:
        """Test the generated parser"""
        print(" Testing parser...")
        
        # Save the code
        parser_path = f'custom_parsers/{state["target_bank"]}_parser.py'
        with open(parser_path, 'w', encoding='utf-8') as f:
            f.write(state["generated_code"])
        
        test_results = self._test_parser(parser_path, state["pdf_path"], state["csv_path"])
        
        if test_results["success"]:
            print(" Parser test PASSED!")
        else:
            print(f" Parser test FAILED: {test_results.get('error', 'Unknown error')}")
        
        return {**state, "test_results": test_results, "status": "testing"}
    
    def _test_parser(self, parser_path: str, pdf_path: str, csv_path: str) -> dict:
        """Test the parser with better error handling"""
        try:
            # Clear module cache
            module_name = f"parser_{hash(parser_path)}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Import the generated parser
            spec = importlib.util.spec_from_file_location(module_name, parser_path)
            parser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parser_module)
            
            # Run the parser
            result_df = parser_module.parse(pdf_path)
            
            # Load expected result
            expected_df = pd.read_csv(csv_path)
            
            # Compare
            columns_match = list(result_df.columns) == list(expected_df.columns)
            shape_match = result_df.shape == expected_df.shape
            
            return {
                "success": columns_match and shape_match,
                "columns_match": columns_match,
                "shape_match": shape_match,
                "result_columns": list(result_df.columns),
                "expected_columns": list(expected_df.columns),
                "result_shape": result_df.shape,
                "expected_shape": expected_df.shape,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "columns_match": False,
                "shape_match": False
            }
    
    def refine_node(self, state: AgentState) -> AgentState:
        """Prepare for next attempt"""
        print(" Refining based on errors...")
        new_attempt = state["attempt"] + 1
        return {**state, "attempt": new_attempt, "status": "refining"}
    
    def should_continue(self, state: AgentState) -> str:
        """Decide whether to continue"""
        if state["test_results"].get("success", False):
            return "end"
        elif state["attempt"] < 2:  # 0,1,2 = 3 attempts total
            return "continue"
        else:
            return "end"
    
    def run(self, target_bank: str):
        """Run the agent"""
        print(f" Starting Fixed Agent for {target_bank.upper()}...")
        print("=" * 50)
        
        pdf_path = f"data/{target_bank}/icici sample.pdf"  # Note the space in filename
        csv_path = f"data/{target_bank}/result.csv"
        
        if not Path(pdf_path).exists():
            print(f" PDF not found: {pdf_path}")
            return
        
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
        
        Path("custom_parsers").mkdir(exist_ok=True)
        final_state = self.graph.invoke(initial_state)
        
        print("=" * 50)
        if final_state["test_results"].get("success"):
            print(f" SUCCESS! Parser: custom_parsers/{target_bank}_parser.py")
        else:
            print(" Failed after 3 attempts")
            print(" Try running with the working hardcoded parser instead")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    
    agent = FixedCodingAgent()
    agent.run(args.target)
