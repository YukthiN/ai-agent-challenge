import pytest
import pandas as pd
from pathlib import Path

class TestParsers:
    def test_parser_contract(self):
        """Test that parsers follow the required contract"""
        # Basic contract test
        assert Path("custom_parsers/icici_parser.py").exists()
    
    def test_csv_schema(self):
        """Test that expected CSV has correct schema"""
        if Path("data/icici/result.csv").exists():
            df = pd.read_csv("data/icici/result.csv")
            assert len(df.columns) == 5, "CSV should have 5 columns"
            expected_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
            assert list(df.columns) == expected_columns, f"Columns should be {expected_columns}"
            assert len(df) == 100, "CSV should have 100 rows"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
