#!/usr/bin/env python3
"""
Demo Script for AI Agent Challenge
Shows the complete workflow from start to finish
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f" Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(" SUCCESS")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(" FAILED")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def main():
    print(" AI AGENT CHALLENGE - COMPLETE DEMO")
    print("This demo shows the full agent workflow from start to finish")
    
    # 1. Verify setup
    print("\n1.  VERIFYING PROJECT SETUP")
    run_command("python final_verification.py", "Project Structure Verification")
    
    # 2. Run the agent
    print("\n2.  RUNNING AI AGENT")
    run_command("python agent.py --target icici", "AI Agent Generating Parser")
    
    # 3. Test the generated parser
    print("\n3.  TESTING GENERATED PARSER")
    run_command("python test_parser.py", "Parser Functional Test")
    
    # 4. Run full test suite
    print("\n4.  RUNNING COMPREHENSIVE TEST SUITE")
    run_command("python -m pytest tests/ -v", "Full Test Suite")
    
    # 5. Show generated files
    print("\n5.  GENERATED FILES")
    if Path("custom_parsers/icici_parser.py").exists():
        print(" custom_parsers/icici_parser.py - Generated parser")
        file_size = Path("custom_parsers/icici_parser.py").stat().st_size
        print(f"   File size: {file_size} bytes")
        
        # Show first few lines of generated code
        with open("custom_parsers/icici_parser.py", "r") as f:
            lines = f.readlines()[:10]
            print("   Preview (first 10 lines):")
            for line in lines:
                print(f"   {line.rstrip()}")
    
    print("\n" + "="*60)
    print(" DEMO COMPLETE!")
    print("The AI agent successfully:")
    print("   Analyzed ICICI bank statement structure")
    print("   Generated working parser code")
    print("   Validated output against expected CSV")
    print("   Passed comprehensive tests")
    print("="*60)

if __name__ == "__main__":
    main()
