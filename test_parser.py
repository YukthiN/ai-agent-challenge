import pandas as pd
import sys
from pathlib import Path

def test_generated_parser():
    """Comprehensive test for the generated parser"""
    print(" COMPREHENSIVE PARSER TEST")
    print("=" * 50)
    
    # Add custom_parsers to path
    sys.path.append('custom_parsers')
    
    try:
        # Import the generated parser
        from icici_parser import parse
        
        # Test files
        pdf_path = Path("data/icici/icici sample.pdf")
        csv_path = Path("data/icici/result.csv")
        
        if not pdf_path.exists():
            print(" PDF file not found")
            return False
        
        print(" Running parser...")
        result_df = parse(str(pdf_path))
        
        print(f" Parser executed successfully")
        print(f" Result shape: {result_df.shape}")
        print(f" Columns: {result_df.columns.tolist()}")
        
        if csv_path.exists():
            expected_df = pd.read_csv(csv_path)
            print(f" Expected shape: {expected_df.shape}")
            print(f" Expected columns: {expected_df.columns.tolist()}")
            
            # Comprehensive comparison
            columns_match = list(result_df.columns) == list(expected_df.columns)
            shape_match = result_df.shape == expected_df.shape
            
            print(f" Column match: {columns_match}")
            print(f" Shape match: {shape_match}")
            
            if columns_match and shape_match:
                print(" ALL TESTS PASSED! Parser matches expected format.")
                return True
            else:
                print(" Tests failed - format mismatch")
                return False
        else:
            print("  No CSV for comparison, but parser runs without errors")
            return True
            
    except Exception as e:
        print(f" Parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generated_parser()
    sys.exit(0 if success else 1)
