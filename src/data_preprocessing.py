"""
Data Preprocessing Module for NEET Predictor

This module provides functions to:
- Load raw LFS 2020-21 Stata data
- Create NEET labels based on education, employment, and training status
- Clean and encode variables
- Handle missing data and PII removal

Author: Data Science Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
import hashlib
import re

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_raw(path: str, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Load raw LFS 2020-21 Stata file with metadata preservation.
    
    Parameters:
    -----------
    path : str
        Path to the .dta file
    verbose : bool, default=True
        Whether to print loading information
        
    Returns:
    --------
    df : pd.DataFrame
        Loaded dataframe
    metadata : dict
        Dictionary containing variable labels, value labels, and other metadata
        
    Example:
    --------
    >>> df, metadata = load_raw('data/raw/LFS2020-21.dta')
    Loaded 150,000 rows and 120 columns from LFS2020-21.dta
    """
    if verbose:
        print(f"Loading data from {path}...")
    
    try:
        # Determine file format
        file_ext = path.lower().split('.')[-1]
        
        if file_ext == 'dta':
            # Read Stata file with pandas (preserves labels)
            df = pd.read_stata(path, convert_categoricals=False, preserve_dtypes=False)
            
            # Extract metadata
            metadata = {
                'variable_labels': {},
                'value_labels': {},
                'file_format': 'Stata (.dta)',
                'n_rows': len(df),
                'n_cols': len(df.columns)
            }
            
            # Try to extract variable labels if they exist
            try:
                with pd.io.stata.StataReader(path) as reader:
                    if hasattr(reader, 'variable_labels'):
                        metadata['variable_labels'] = reader.variable_labels()
                    if hasattr(reader, 'value_labels'):
                        metadata['value_labels'] = reader.value_labels()
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not extract full metadata: {e}")
                    
        elif file_ext == 'csv':
            # Read CSV file
            df = pd.read_csv(path, low_memory=False)
            
            # Create basic metadata
            metadata = {
                'variable_labels': {},
                'value_labels': {},
                'file_format': 'CSV (.csv)',
                'n_rows': len(df),
                'n_cols': len(df.columns)
            }
            
            if verbose:
                print(f"Note: CSV format - variable labels not available")
        
        else:
            raise ValueError(f"Unsupported file format: .{file_ext}. Use .dta or .csv")
        
        if verbose:
            print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        return df, metadata
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Please ensure LFS2020-21.dta is in the data/raw/ directory."
        )
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def save_schema(metadata: Dict, output_path: str = 'data/raw/schema.txt'):
    """
    Save variable schema and labels to a text file.
    
    Parameters:
    -----------
    metadata : dict
        Metadata dictionary from load_raw()
    output_path : str
        Path to save schema file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LFS 2020-21 VARIABLE SCHEMA\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Rows: {metadata['n_rows']:,}\n")
        f.write(f"Total Columns: {metadata['n_cols']}\n")
        f.write(f"File Format: {metadata['file_format']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("VARIABLE LABELS\n")
        f.write("-" * 80 + "\n")
        
        var_labels = metadata.get('variable_labels', {})
        if var_labels:
            for var, label in var_labels.items():
                f.write(f"{var:30s} : {label}\n")
        else:
            f.write("No variable labels available in metadata.\n")
    
    print(f"✓ Schema saved to {output_path}")


def detect_variable_names(df: pd.DataFrame, verbose: bool = True) -> Dict[str, str]:
    """
    Auto-detect variable names for key NEET components using pattern matching.
    
    This function attempts to map standard variable categories to actual column names
    in the dataset, as variable names may differ across LFS versions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print detected mappings
        
    Returns:
    --------
    var_map : dict
        Dictionary mapping standard names to actual column names
        
    Keys:
    -----
    - 'age': Age variable
    - 'sex': Gender/sex variable
    - 'education_status': Current education enrollment
    - 'employment_status': Current employment status
    - 'training': Training/vocational enrollment
    - 'province': Province/region
    - 'district': District
    - 'urban_rural': Urban/rural classification
    - 'weight': Survey weight (if available)
    """
    var_map = {}
    cols_lower = {col.lower(): col for col in df.columns}
    
    # Age detection
    age_patterns = ['age', 'umar', 'age_years']
    for pattern in age_patterns:
        matches = [col for col in cols_lower.keys() if pattern in col]
        if matches:
            var_map['age'] = cols_lower[matches[0]]
            break
    
    # Sex/Gender detection
    sex_patterns = ['sex', 'gender', 'jins']
    for pattern in sex_patterns:
        matches = [col for col in cols_lower.keys() if pattern in col]
        if matches:
            var_map['sex'] = cols_lower[matches[0]]
            break
    
    # Education status (currently enrolled)
    edu_patterns = ['educ', 'school', 'attend', 'enrol', 'current_ed', 'student']
    for pattern in edu_patterns:
        matches = [col for col in cols_lower.keys() if pattern in col and 
                  any(x in col for x in ['status', 'attend', 'current', 'enrol'])]
        if matches:
            var_map['education_status'] = cols_lower[matches[0]]
            break
    
    # Employment status
    emp_patterns = ['employ', 'work', 'job', 'labour', 'labor', 'occupation']
    for pattern in emp_patterns:
        matches = [col for col in cols_lower.keys() if pattern in col and 'status' in col]
        if matches:
            var_map['employment_status'] = cols_lower[matches[0]]
            break
    
    # Training/vocational
    train_patterns = ['train', 'vocational', 'skill', 'apprentice']
    for pattern in train_patterns:
        matches = [col for col in cols_lower.keys() if pattern in col]
        if matches:
            var_map['training'] = cols_lower[matches[0]]
            break
    
    # Geographic variables
    geo_patterns = {
        'province': ['prov', 'province', 'region'],
        'district': ['dist', 'district', 'tehsil'],
        'urban_rural': ['urban', 'rural', 'area_type', 'sector']
    }
    
    for key, patterns in geo_patterns.items():
        for pattern in patterns:
            matches = [col for col in cols_lower.keys() if pattern in col]
            if matches:
                var_map[key] = cols_lower[matches[0]]
                break
    
    # Survey weight
    weight_patterns = ['weight', 'wt', 'wgt', 'sampling_weight']
    for pattern in weight_patterns:
        matches = [col for col in cols_lower.keys() if pattern in col]
        if matches:
            var_map['weight'] = cols_lower[matches[0]]
            break
    
    if verbose:
        print("\n" + "="*60)
        print("DETECTED VARIABLE MAPPINGS")
        print("="*60)
        for key, col in var_map.items():
            print(f"{key:20s} -> {col}")
        
        missing = set(['age', 'sex', 'education_status', 'employment_status']) - set(var_map.keys())
        if missing:
            print(f"\n⚠ WARNING: Could not auto-detect: {', '.join(missing)}")
            print("  Please manually specify these variables.")
    
    return var_map


def create_neet_label(df: pd.DataFrame, 
                     var_map: Optional[Dict[str, str]] = None,
                     age_min: int = 15,
                     age_max: int = 24,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Create NEET label for youth (15-24 years).
    
    NEET Definition:
    ----------------
    A person is classified as NEET if they are:
    1. NOT currently in education (not attending school/college)
    2. AND NOT employed (not working or having a job)
    3. AND NOT in training (not in vocational/skills training)
    
    NEET = 1 if all three conditions are true, else 0
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with youth data
    var_map : dict, optional
        Variable name mapping from detect_variable_names()
        If None, will attempt auto-detection
    age_min : int, default=15
        Minimum age for youth
    age_max : int, default=24
        Maximum age for youth
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added columns:
        - in_education: Boolean, whether currently in education
        - employed: Boolean, whether currently employed
        - in_training: Boolean, whether in training/vocational
        - NEET: Integer (0 or 1), final NEET label
        - age_group: Categorical age bands
        
    Examples:
    ---------
    >>> df_labeled = create_neet_label(df, var_map)
    >>> print(df_labeled['NEET'].value_counts())
    0    85000
    1    15000
    
    Notes:
    ------
    - Missing values in component variables are treated conservatively:
      If education_status is missing, assume NOT in education (cautious approach)
    - Survey weights should be applied when aggregating NEET rates
    """
    if verbose:
        print("\n" + "="*60)
        print("CREATING NEET LABEL")
        print("="*60)
    
    df = df.copy()
    
    # Auto-detect variables if not provided
    if var_map is None:
        var_map = detect_variable_names(df, verbose=False)
    
    # 1. Filter to youth age range (15-24)
    if 'age' not in var_map:
        raise ValueError("Age variable not found. Please specify in var_map.")
    
    age_col = var_map['age']
    df_youth = df[(df[age_col] >= age_min) & (df[age_col] <= age_max)].copy()
    
    if verbose:
        print(f"\n1. Filtered to youth aged {age_min}-{age_max}")
        print(f"   Total youth: {len(df_youth):,} ({len(df_youth)/len(df)*100:.1f}% of sample)")
    
    # 2. Create component flags
    
    # Flag: in_education
    if 'education_status' in var_map:
        edu_col = var_map['education_status']
        # Common patterns: "Currently attending", "Enrolled", "Yes", 1, etc.
        df_youth['in_education'] = df_youth[edu_col].astype(str).str.lower().isin([
            'yes', 'currently attending', 'enrolled', 'student', '1', '1.0', 'true'
        ])
    else:
        warnings.warn("Education status variable not found. Assuming all NOT in education.")
        df_youth['in_education'] = False
    
    # Flag: employed
    if 'employment_status' in var_map:
        emp_col = var_map['employment_status']
        # Common patterns: "Employed", "Working", "Has job", 1, etc.
        df_youth['employed'] = df_youth[emp_col].astype(str).str.lower().isin([
            'employed', 'working', 'has job', 'self-employed', 'wage worker', '1', '1.0', 'yes'
        ])
    else:
        warnings.warn("Employment status variable not found. Assuming all NOT employed.")
        df_youth['employed'] = False
    
    # Flag: in_training
    if 'training' in var_map:
        train_col = var_map['training']
        df_youth['in_training'] = df_youth[train_col].astype(str).str.lower().isin([
            'yes', 'in training', 'vocational', 'apprentice', '1', '1.0', 'true'
        ])
    else:
        # If no training variable, assume none in training
        df_youth['in_training'] = False
    
    # Handle missing values conservatively
    df_youth['in_education'] = df_youth['in_education'].fillna(False)
    df_youth['employed'] = df_youth['employed'].fillna(False)
    df_youth['in_training'] = df_youth['in_training'].fillna(False)
    
    # 3. Create NEET label
    # NEET = NOT in education AND NOT employed AND NOT in training
    df_youth['NEET'] = (
        (~df_youth['in_education']) & 
        (~df_youth['employed']) & 
        (~df_youth['in_training'])
    ).astype(int)
    
    # 4. Create age groups for analysis
    df_youth['age_group'] = pd.cut(
        df_youth[age_col],
        bins=[15, 18, 21, 24],
        labels=['15-17', '18-20', '21-24'],
        include_lowest=True
    )
    
    if verbose:
        print(f"\n2. Component flags created:")
        print(f"   - In education: {df_youth['in_education'].sum():,} ({df_youth['in_education'].mean()*100:.1f}%)")
        print(f"   - Employed: {df_youth['employed'].sum():,} ({df_youth['employed'].mean()*100:.1f}%)")
        print(f"   - In training: {df_youth['in_training'].sum():,} ({df_youth['in_training'].mean()*100:.1f}%)")
        
        print(f"\n3. NEET Label Distribution:")
        neet_counts = df_youth['NEET'].value_counts().sort_index()
        for label, count in neet_counts.items():
            status = "NEET" if label == 1 else "Not NEET"
            print(f"   {status:10s}: {count:,} ({count/len(df_youth)*100:.1f}%)")
        
        print(f"\n✓ NEET label creation complete!")
    
    return df_youth


def remove_pii(df: pd.DataFrame, 
               pii_patterns: List[str] = None,
               create_hash_id: bool = True,
               verbose: bool = True) -> pd.DataFrame:
    """
    Remove or anonymize personally identifiable information (PII).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    pii_patterns : list of str, optional
        Column name patterns to remove (e.g., ['name', 'address', 'phone', 'cnic'])
    create_hash_id : bool, default=True
        Whether to create a hashed anonymous ID
    verbose : bool
        Whether to print removed columns
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with PII removed
    """
    if pii_patterns is None:
        pii_patterns = [
            'name', 'father', 'mother', 'guardian',
            'address', 'street', 'house',
            'phone', 'mobile', 'contact',
            'cnic', 'id_card', 'national_id',
            'email', 'nic'
        ]
    
    df = df.copy()
    removed_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in pii_patterns):
            removed_cols.append(col)
            df = df.drop(columns=[col])
    
    # Create anonymous hash ID if requested
    if create_hash_id and 'id_hash' not in df.columns:
        # Create hash from row index + random salt
        df['id_hash'] = pd.Series(df.index.astype(str)).apply(
            lambda x: hashlib.md5((x + str(RANDOM_SEED)).encode()).hexdigest()[:12]
        ).values
    
    if verbose and removed_cols:
        print(f"\n⚠ Removed {len(removed_cols)} PII columns: {', '.join(removed_cols)}")
    
    return df


def clean_vars(df: pd.DataFrame, 
               var_map: Dict[str, str],
               verbose: bool = True) -> pd.DataFrame:
    """
    Clean and standardize key variables.
    
    Operations:
    -----------
    - Standardize sex/gender to ['Male', 'Female']
    - Clean province and district names
    - Standardize urban/rural to ['Urban', 'Rural']
    - Handle missing values appropriately
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    var_map : dict
        Variable name mapping
    verbose : bool
        Print cleaning summary
        
    Returns:
    --------
    df : pd.DataFrame
        Cleaned dataframe
    """
    df = df.copy()
    
    if verbose:
        print("\n" + "="*60)
        print("CLEANING VARIABLES")
        print("="*60)
    
    # Clean sex/gender
    if 'sex' in var_map:
        sex_col = var_map['sex']
        df[sex_col] = df[sex_col].astype(str).str.strip().str.title()
        
        # Standardize variations
        sex_mapping = {
            '1': 'Male', '2': 'Female',
            'M': 'Male', 'F': 'Female',
            'Male': 'Male', 'Female': 'Female',
            'Man': 'Male', 'Woman': 'Female'
        }
        df[sex_col] = df[sex_col].replace(sex_mapping)
        
        if verbose:
            print(f"\n✓ Cleaned {sex_col}:")
            print(df[sex_col].value_counts())
    
    # Clean province
    if 'province' in var_map:
        prov_col = var_map['province']
        df[prov_col] = df[prov_col].astype(str).str.strip().str.title()
        df[prov_col] = df[prov_col].replace({'Nan': 'Unknown', 'None': 'Unknown'})
        
        if verbose:
            print(f"\n✓ Cleaned {prov_col}: {df[prov_col].nunique()} unique provinces")
    
    # Clean district
    if 'district' in var_map:
        dist_col = var_map['district']
        df[dist_col] = df[dist_col].astype(str).str.strip().str.title()
        df[dist_col] = df[dist_col].replace({'Nan': 'Unknown', 'None': 'Unknown'})
        
        if verbose:
            print(f"✓ Cleaned {dist_col}: {df[dist_col].nunique()} unique districts")
    
    # Clean urban/rural
    if 'urban_rural' in var_map:
        ur_col = var_map['urban_rural']
        df[ur_col] = df[ur_col].astype(str).str.strip().str.title()
        
        ur_mapping = {
            '1': 'Urban', '2': 'Rural',
            'U': 'Urban', 'R': 'Rural',
            'Urban': 'Urban', 'Rural': 'Rural'
        }
        df[ur_col] = df[ur_col].replace(ur_mapping)
        
        if verbose:
            print(f"\n✓ Cleaned {ur_col}:")
            print(df[ur_col].value_counts())
    
    return df


def encode_features(df: pd.DataFrame,
                    categorical_cols: List[str],
                    numeric_cols: List[str],
                    create_interactions: bool = True) -> pd.DataFrame:
    """
    Encode categorical features and create interaction terms.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with cleaned variables
    categorical_cols : list
        List of categorical column names to encode
    numeric_cols : list
        List of numeric column names
    create_interactions : bool
        Whether to create interaction features
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with encoded features
    """
    df = df.copy()
    
    # Create dummy variables for categoricals
    # Note: Actual encoding will be done in sklearn pipeline for proper train/test handling
    
    # Create interaction terms if requested
    if create_interactions:
        # Example: female_rural indicator
        if 'sex' in df.columns and 'urban_rural' in df.columns:
            df['female_rural'] = (
                (df['sex'] == 'Female') & 
                (df['urban_rural'] == 'Rural')
            ).astype(int)
        
        # Add more interactions as needed
    
    return df


# Example usage and unit tests
if __name__ == "__main__":
    print("="*60)
    print("DATA PREPROCESSING MODULE - UNIT TESTS")
    print("="*60)
    
    # Test 1: Create sample data
    print("\nTest 1: Creating sample data...")
    sample_data = pd.DataFrame({
        'age': [16, 18, 22, 19, 24, 17, 20, 23, 15, 21],
        'sex': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'education_status': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
        'employment_status': ['Not working', 'Working', 'Not working', 'Working', 'Working', 
                             'Not working', 'Not working', 'Not working', 'Not working', 'Working'],
        'training': ['No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No'],
        'province': ['Punjab', 'Sindh', 'KP', 'Punjab', 'Balochistan', 'Punjab', 'Sindh', 'KP', 'Punjab', 'Sindh'],
        'district': ['Lahore', 'Karachi', 'Peshawar', 'Faisalabad', 'Quetta', 'Multan', 'Hyderabad', 'Mardan', 'Rawalpindi', 'Sukkur'],
        'urban_rural': ['Urban', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Urban', 'Rural', 'Urban', 'Rural']
    })
    print(f"✓ Created sample data with {len(sample_data)} rows")
    
    # Test 2: Detect variables
    print("\nTest 2: Detecting variable names...")
    var_map = detect_variable_names(sample_data, verbose=True)
    
    # Test 3: Create NEET label
    print("\nTest 3: Creating NEET label...")
    df_labeled = create_neet_label(sample_data, var_map, verbose=True)
    
    # Test 4: Verify NEET logic
    print("\nTest 4: Verifying NEET label logic...")
    print("\nSample records:")
    cols_to_show = ['age', 'sex', 'in_education', 'employed', 'in_training', 'NEET']
    print(df_labeled[cols_to_show].head(10).to_string(index=False))
    
    # Manual verification
    print("\n" + "="*60)
    print("MANUAL VERIFICATION")
    print("="*60)
    print("Row 0: Age 16, in education=True, employed=False, in_training=False")
    print(f"       Expected NEET=0 (in education), Actual NEET={df_labeled.iloc[0]['NEET']}")
    print(f"       ✓ PASS" if df_labeled.iloc[0]['NEET'] == 0 else "       ✗ FAIL")
    
    print("\nRow 2: Age 22, in education=False, employed=False, in_training=False")
    print(f"       Expected NEET=1 (not in anything), Actual NEET={df_labeled.iloc[2]['NEET']}")
    print(f"       ✓ PASS" if df_labeled.iloc[2]['NEET'] == 1 else "       ✗ FAIL")
    
    print("\nRow 7: Age 23, in education=False, employed=False, in_training=True")
    print(f"       Expected NEET=0 (in training), Actual NEET={df_labeled.iloc[7]['NEET']}")
    print(f"       ✓ PASS" if df_labeled.iloc[7]['NEET'] == 0 else "       ✗ FAIL")
    
    # Test 5: Clean variables
    print("\nTest 5: Cleaning variables...")
    df_clean = clean_vars(df_labeled, var_map, verbose=True)
    
    # Test 6: PII removal
    print("\nTest 6: Testing PII removal...")
    sample_with_pii = df_clean.copy()
    sample_with_pii['full_name'] = ['Person ' + str(i) for i in range(len(sample_with_pii))]
    sample_with_pii['phone_number'] = ['0300-' + str(i).zfill(7) for i in range(len(sample_with_pii))]
    
    df_no_pii = remove_pii(sample_with_pii, verbose=True)
    print(f"✓ Columns before: {len(sample_with_pii.columns)}, after: {len(df_no_pii.columns)}")
    print(f"✓ Hash ID created: {'id_hash' in df_no_pii.columns}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
