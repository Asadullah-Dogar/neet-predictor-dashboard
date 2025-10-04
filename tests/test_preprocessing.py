"""
Unit Tests for NEET Predictor Data Preprocessing

Tests for:
- NEET label creation logic
- Data loading
- Variable detection
- Data cleaning

Author: Data Science Team
Date: October 2025
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import (
    create_neet_label,
    detect_variable_names,
    clean_vars,
    remove_pii
)


class TestNEETLabelCreation:
    """Test suite for NEET label creation logic."""
    
    def setup_method(self):
        """Create sample data for testing."""
        self.sample_data = pd.DataFrame({
            'age': [16, 18, 22, 19, 24, 17, 20, 23, 15, 21],
            'sex': ['Male', 'Female', 'Male', 'Female', 'Male', 
                   'Female', 'Male', 'Female', 'Male', 'Female'],
            'education_status': ['Yes', 'No', 'No', 'Yes', 'No', 
                                'Yes', 'No', 'No', 'Yes', 'No'],
            'employment_status': ['Not working', 'Working', 'Not working', 
                                 'Working', 'Working', 'Not working', 
                                 'Not working', 'Not working', 'Not working', 'Working'],
            'training': ['No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No'],
            'province': ['Punjab', 'Sindh', 'KP', 'Punjab', 'Balochistan', 
                        'Punjab', 'Sindh', 'KP', 'Punjab', 'Sindh'],
            'urban_rural': ['Urban', 'Urban', 'Rural', 'Urban', 'Rural', 
                           'Urban', 'Urban', 'Rural', 'Urban', 'Rural']
        })
    
    def test_neet_label_in_education(self):
        """Test that students are NOT NEET."""
        var_map = detect_variable_names(self.sample_data, verbose=False)
        df_labeled = create_neet_label(self.sample_data, var_map, verbose=False)
        
        # Row 0: Age 16, in education, not employed, not in training
        # Should be NOT NEET (0) because in education
        assert df_labeled.iloc[0]['NEET'] == 0, "Student should not be NEET"
        assert df_labeled.iloc[0]['in_education'] == True
    
    def test_neet_label_employed(self):
        """Test that employed youth are NOT NEET."""
        var_map = detect_variable_names(self.sample_data, verbose=False)
        df_labeled = create_neet_label(self.sample_data, var_map, verbose=False)
        
        # Row 1: Age 18, not in education, employed, not in training
        # Should be NOT NEET (0) because employed
        assert df_labeled.iloc[1]['NEET'] == 0, "Employed person should not be NEET"
        assert df_labeled.iloc[1]['employed'] == True
    
    def test_neet_label_in_training(self):
        """Test that those in training are NOT NEET."""
        var_map = detect_variable_names(self.sample_data, verbose=False)
        df_labeled = create_neet_label(self.sample_data, var_map, verbose=False)
        
        # Row 7: Age 23, not in education, not employed, in training
        # Should be NOT NEET (0) because in training
        assert df_labeled.iloc[7]['NEET'] == 0, "Person in training should not be NEET"
        assert df_labeled.iloc[7]['in_training'] == True
    
    def test_neet_label_is_neet(self):
        """Test that inactive youth are NEET."""
        var_map = detect_variable_names(self.sample_data, verbose=False)
        df_labeled = create_neet_label(self.sample_data, var_map, verbose=False)
        
        # Row 2: Age 22, not in education, not employed, not in training
        # Should be NEET (1)
        assert df_labeled.iloc[2]['NEET'] == 1, "Inactive person should be NEET"
        assert df_labeled.iloc[2]['in_education'] == False
        assert df_labeled.iloc[2]['employed'] == False
        assert df_labeled.iloc[2]['in_training'] == False
    
    def test_age_filter(self):
        """Test that age filtering works correctly."""
        # Add some people outside age range
        extended_data = self.sample_data.copy()
        extended_data = pd.concat([
            extended_data,
            pd.DataFrame({
                'age': [14, 25, 30],
                'sex': ['Male', 'Female', 'Male'],
                'education_status': ['No', 'No', 'No'],
                'employment_status': ['Not working', 'Not working', 'Not working'],
                'training': ['No', 'No', 'No'],
                'province': ['Punjab', 'Punjab', 'Punjab'],
                'urban_rural': ['Urban', 'Urban', 'Urban']
            })
        ])
        
        var_map = detect_variable_names(extended_data, verbose=False)
        df_labeled = create_neet_label(extended_data, var_map, age_min=15, age_max=24, verbose=False)
        
        # Should only have 10 rows (ages 15-24)
        assert len(df_labeled) == 10, "Should filter to youth aged 15-24"
        assert df_labeled['age'].min() >= 15
        assert df_labeled['age'].max() <= 24
    
    def test_neet_distribution(self):
        """Test that NEET label has expected distribution."""
        var_map = detect_variable_names(self.sample_data, verbose=False)
        df_labeled = create_neet_label(self.sample_data, var_map, verbose=False)
        
        # Check that we have both NEET and non-NEET
        assert 0 in df_labeled['NEET'].values
        assert 1 in df_labeled['NEET'].values
        
        # Check that NEET is binary
        assert set(df_labeled['NEET'].unique()).issubset({0, 1})


class TestVariableDetection:
    """Test suite for automatic variable detection."""
    
    def test_detect_age_variable(self):
        """Test detection of age variable."""
        df = pd.DataFrame({'age': [20], 'name': ['test']})
        var_map = detect_variable_names(df, verbose=False)
        assert 'age' in var_map
    
    def test_detect_sex_variable(self):
        """Test detection of sex/gender variable."""
        df = pd.DataFrame({'sex': ['Male'], 'age': [20]})
        var_map = detect_variable_names(df, verbose=False)
        assert 'sex' in var_map
    
    def test_detect_education_variable(self):
        """Test detection of education status variable."""
        df = pd.DataFrame({'current_education_status': ['Yes'], 'age': [20]})
        var_map = detect_variable_names(df, verbose=False)
        assert 'education_status' in var_map


class TestDataCleaning:
    """Test suite for data cleaning functions."""
    
    def test_clean_sex_values(self):
        """Test standardization of sex/gender values."""
        df = pd.DataFrame({
            'sex': ['1', '2', 'M', 'F', 'Male', 'Female'],
            'age': [20, 21, 22, 23, 24, 25]
        })
        var_map = {'sex': 'sex', 'age': 'age'}
        
        df_clean = clean_vars(df, var_map, verbose=False)
        
        # All values should be standardized to Male/Female
        assert set(df_clean['sex'].unique()).issubset({'Male', 'Female'})
    
    def test_clean_urban_rural(self):
        """Test standardization of urban/rural values."""
        df = pd.DataFrame({
            'urban_rural': ['1', '2', 'U', 'R', 'Urban', 'Rural'],
            'age': [20, 21, 22, 23, 24, 25]
        })
        var_map = {'urban_rural': 'urban_rural', 'age': 'age'}
        
        df_clean = clean_vars(df, var_map, verbose=False)
        
        # All values should be standardized to Urban/Rural
        assert set(df_clean['urban_rural'].unique()).issubset({'Urban', 'Rural'})


class TestPIIRemoval:
    """Test suite for PII removal."""
    
    def test_remove_name_columns(self):
        """Test that name columns are removed."""
        df = pd.DataFrame({
            'age': [20, 21],
            'full_name': ['John Doe', 'Jane Smith'],
            'father_name': ['Father1', 'Father2']
        })
        
        df_clean = remove_pii(df, verbose=False)
        
        assert 'full_name' not in df_clean.columns
        assert 'father_name' not in df_clean.columns
        assert 'age' in df_clean.columns
    
    def test_hash_id_creation(self):
        """Test that hash ID is created."""
        df = pd.DataFrame({'age': [20, 21, 22]})
        
        df_clean = remove_pii(df, create_hash_id=True, verbose=False)
        
        assert 'id_hash' in df_clean.columns
        assert len(df_clean['id_hash'].unique()) == len(df_clean)
    
    def test_remove_phone_columns(self):
        """Test that phone number columns are removed."""
        df = pd.DataFrame({
            'age': [20],
            'phone_number': ['0300-1234567'],
            'mobile': ['0321-7654321']
        })
        
        df_clean = remove_pii(df, verbose=False)
        
        assert 'phone_number' not in df_clean.columns
        assert 'mobile' not in df_clean.columns


class TestEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_missing_values_in_neet_components(self):
        """Test handling of missing values in NEET components."""
        df = pd.DataFrame({
            'age': [20, 21, 22],
            'sex': ['Male', 'Female', 'Male'],
            'education_status': ['Yes', None, 'No'],
            'employment_status': ['Working', 'Not working', None],
            'training': ['No', 'No', 'No'],
            'province': ['Punjab', 'Sindh', 'KP'],
            'urban_rural': ['Urban', 'Rural', 'Urban']
        })
        
        var_map = detect_variable_names(df, verbose=False)
        
        # Should not raise an error
        df_labeled = create_neet_label(df, var_map, verbose=False)
        
        # NEET label should be created for all rows
        assert len(df_labeled) == len(df)
        assert 'NEET' in df_labeled.columns
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()
        
        # Should handle gracefully
        var_map = detect_variable_names(df, verbose=False)
        assert isinstance(var_map, dict)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
