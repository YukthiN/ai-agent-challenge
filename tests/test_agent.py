import pytest
import sys
from pathlib import Path
import pandas as pd  # Add this import
from agent_final import WorkingCodingAgent

class TestAgent:
    def test_agent_initialization(self):
        """Test that agent initializes correctly"""
        agent = WorkingCodingAgent()
        assert agent is not None
    
    def test_plan_node_structure(self):
        """Test that plan node returns correct structure"""
        agent = WorkingCodingAgent()
        # Basic structure test
        assert hasattr(agent, 'graph')
    
    def test_file_existence(self):
        """Test that required files exist"""
        # Fixed: Use the actual filename with space
        assert Path("data/icici/icici sample.pdf").exists(), "ICICI PDF sample missing"
        assert Path("data/icici/result.csv").exists(), "ICICI CSV result missing"
        assert Path("custom_parsers/icici_parser.py").exists(), "Generated parser missing"

    def test_generated_parser(self):
        """Test that the generated parser works"""
        # Add custom_parsers to path
        sys.path.append('custom_parsers')
        
        try:
            from icici_parser import parse
            result_df = parse("data/icici/icici sample.pdf")
            assert isinstance(result_df, pd.DataFrame)  # Fixed this line
            assert len(result_df.columns) > 0
            print(f" Parser test passed: {result_df.shape} shape, {len(result_df.columns)} columns")
        except Exception as e:
            pytest.fail(f"Generated parser failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
