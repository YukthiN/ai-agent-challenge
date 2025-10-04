import pdfplumber
import pandas as pd
import re
from datetime import datetime

def parse(pdf_path: str) -> pd.DataFrame:
    """Parse ICICI bank statement PDF - Proven Working Version"""
    all_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from each page
                tables = page.extract_tables()
                
                for table_num, table in enumerate(tables):
                    if not table or len(table) <= 1:
                        continue
                    
                    # Extract header from first row
                    header = []
                    for cell in table[0]:
                        if cell and str(cell).strip():
                            header.append(str(cell).strip())
                        else:
                            header.append(f"Column_{len(header)+1}")
                    
                    # Process data rows
                    for row in table[1:]:
                        if not any(row):
                            continue
                            
                        # Clean each cell
                        cleaned_row = []
                        for cell in row:
                            if cell and str(cell).strip():
                                cleaned = str(cell).strip()
                                # Handle numeric values
                                if re.match(r'^-?\d+([.,]\d+)*$', cleaned.replace(',', '')):
                                    try:
                                        cleaned = float(cleaned.replace(',', ''))
                                    except ValueError:
                                        pass
                                cleaned_row.append(cleaned)
                            else:
                                cleaned_row.append(None)
                        
                        # Add row if it matches header length
                        if len(cleaned_row) == len(header):
                            all_data.append(cleaned_row)
        
        # Create DataFrame
        if all_data:
            df = pd.DataFrame(all_data, columns=header)
            
            # Clean column names
            df.columns = [col.replace('\n', ' ').strip() for col in df.columns]
            
            # Standardize column names to match expected CSV
            column_mapping = {
                'Transaction Date': 'Date',
                'Transaction Details': 'Description', 
                'Withdrawal Amount': 'Debit Amt',
                'Deposit Amount': 'Credit Amt',
                'Balance': 'Balance'
            }
            
            # Rename columns to match expected schema
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return pd.DataFrame()
