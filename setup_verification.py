import os
import sys
from pathlib import Path

def is_venv_active():
    """Check if virtual environment is active"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def verify_structure():
    """Verify the project structure meets requirements"""
    
    print(" Verifying project setup...")
    
    # Check virtual environment
    if not is_venv_active():
        print("  Virtual environment not active. Please run:")
        print("   .\\venv\\Scripts\\Activate")
        # Don't return False, just warn - they might be in different shell
    
    required_dirs = ['data', 'custom_parsers', 'tests']
    required_files = ['agent.py', 'requirements.txt']
    
    # Check directories
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f" Missing directory: {dir_name}")
            return False
        print(f" Directory found: {dir_name}")
    
    # Check files
    for file_name in required_files:
        if not os.path.exists(file_name):
            print(f" Missing file: {file_name}")
            return False
        print(f" File found: {file_name}")
    
    # Check data directory contents
    data_path = Path('data')
    if data_path.exists():
        bank_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        if not bank_dirs:
            print("  No bank directories found in data/ - you'll need to add test data")
        else:
            for bank_dir in bank_dirs:
                print(f" Bank data found: {bank_dir.name}")
                pdf_files = list(bank_dir.glob('*.pdf'))
                csv_files = list(bank_dir.glob('*.csv'))
                print(f"   - PDF files: {len(pdf_files)}")
                print(f"   - CSV files: {len(csv_files)}")
    else:
        print(" data/ directory not found")
    
    print(" Project structure verification complete!")
    if is_venv_active():
        print(" Virtual environment is active")
    else:
        print("  Remember to activate virtual environment: .\\venv\\Scripts\\Activate")
    return True

if __name__ == "__main__":
    verify_structure()
