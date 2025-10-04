import ollama
import json

def test_mistral_coding():
    print(" Testing Mistral for code generation...")
    
    try:
        # Test with a banking-specific prompt
        prompt = '''
        Create a Python function that parses a bank statement PDF and returns a pandas DataFrame.
        The function should:
        - Take a PDF file path as input
        - Return a DataFrame with columns: Date, Description, Withdrawal, Deposit, Balance
        - Handle common bank statement formats
        - Extract text and parse relevant transaction data
        
        Provide only the Python code without explanations.
        '''
        
        response = ollama.chat(
            model='mistral:latest',
            messages=[
                {'role': 'system', 'content': 'You are an expert Python developer specializing in financial data parsing.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        code = response['message']['content']
        print(" Mistral code generation successful!")
        print("Generated code preview:")
        print("=" * 50)
        print(code[:500] + "..." if len(code) > 500 else code)
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f" Mistral test failed: {e}")
        return False

if __name__ == "__main__":
    test_mistral_coding()
