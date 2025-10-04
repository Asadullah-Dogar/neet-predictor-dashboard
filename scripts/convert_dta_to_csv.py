"""
Convert LFS2020-21.dta (Stata format) to CSV

This script reads the original Stata file and converts it to CSV format
for easier processing in the project.
"""

import pandas as pd
import os
import sys

print("="*60)
print("CONVERTING LFS 2020-21 STATA FILE TO CSV")
print("="*60)

# Check if file exists
input_file = 'data/raw/LFS2020-21.dta'
output_file = 'data/raw/LFS2020-21.csv'

if not os.path.exists(input_file):
    print(f"\nERROR: File not found: {input_file}")
    print("\nPlease place LFS2020-21.dta in the data/raw/ directory")
    sys.exit(1)

print(f"\nReading Stata file: {input_file}")
print("This may take a few moments...")

try:
    # Read Stata file
    df = pd.read_stata(input_file)
    
    print(f"\n✓ Successfully loaded Stata file")
    print(f"  Shape: {df.shape}")
    print(f"  Rows: {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}")
    
    # Display column names
    print(f"\n  First 20 columns:")
    for i, col in enumerate(df.columns[:20], 1):
        print(f"    {i:2d}. {col}")
    if len(df.columns) > 20:
        print(f"    ... and {len(df.columns) - 20} more columns")
    
    # Save to CSV
    print(f"\nSaving to CSV: {output_file}")
    df.to_csv(output_file, index=False)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ Successfully saved CSV file")
    print(f"  File size: {file_size:.2f} MB")
    
    # Display sample data
    print(f"\n" + "="*60)
    print("SAMPLE DATA (First 5 rows, first 10 columns)")
    print("="*60)
    print(df.iloc[:5, :10].to_string())
    
    print(f"\n" + "="*60)
    print("✓ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nOriginal file: {input_file}")
    print(f"CSV file: {output_file}")
    print(f"\nNext step: Run notebooks/02_Preprocessing_Labeling.ipynb")
    
except Exception as e:
    print(f"\nERROR: Failed to convert file")
    print(f"Error message: {str(e)}")
    sys.exit(1)
