"""
NEET Predictor - Interactive Dashboard
Streamlit Application

This dashboard allows users to:
- Explore NEET predictions by demographic filters
- View district-level risk maps
- Identify high-risk segments
- Download predictions for targeted interventions

Author: Data Science Team
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Set page config
st.set_page_config(
    page_title="NEET Predictor Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Cache data loading
@st.cache_data
def load_data():
    """Load cleaned youth data."""
    # Try multiple possible paths (local vs cloud deployment)
    possible_paths = [
        Path(__file__).parent.parent / 'data' / 'processed' / 'lfs_youth_cleaned.csv',
        Path('data/processed/lfs_youth_cleaned.csv'),
        Path('./data/processed/lfs_youth_cleaned.csv')
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            try:
                df = pd.read_csv(data_path)
                return df
            except Exception as e:
                st.warning(f"Error loading data from {data_path}: {e}")
                continue
    
    # If no path worked
    st.error("Data file not found in any expected location.")
    st.info("""
    **For local development:** 
    Run `notebooks/02_Preprocessing_Labeling.ipynb` to generate the cleaned data.
    
    **For deployment:**
    Ensure `data/processed/lfs_youth_cleaned.csv` is committed to the repository.
    """)
    st.stop()


@st.cache_resource
def load_model():
    """Load trained model if available."""
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent.parent / 'models' / 'model_logistic.pkl',
        Path('models/model_logistic.pkl'),
        Path('./models/model_logistic.pkl')
    ]
    
    for model_path in possible_paths:
        if model_path.exists():
            try:
                model_data = joblib.load(model_path)
                return model_data
            except Exception as e:
                st.warning(f"Error loading model from {model_path}: {e}")
                continue
    
    # Model not found - dashboard works in exploratory mode
    return None


def main():
    # Header
    st.markdown('<p class="main-header">🎯 NEET Predictor Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Labour Force Survey 2020-21 | Youth (15-24 years)</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
        model_data = load_model()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Province filter
    provinces = ['All'] + sorted(df['province'].dropna().unique().tolist()) if 'province' in df.columns else ['All']
    selected_province = st.sidebar.selectbox("Province", provinces)
    
    # District filter (depends on province)
    if 'district' in df.columns:
        if selected_province != 'All':
            districts = ['All'] + sorted(df[df['province'] == selected_province]['district'].dropna().unique().tolist())
        else:
            districts = ['All'] + sorted(df['district'].dropna().unique().tolist())
        selected_district = st.sidebar.selectbox("District", districts)
    else:
        selected_district = 'All'
    
    # Age group filter
    if 'age_group' in df.columns:
        age_groups = ['All'] + sorted(df['age_group'].dropna().unique().tolist())
        selected_age = st.sidebar.selectbox("Age Group", age_groups)
    else:
        selected_age = 'All'
    
    # Gender filter
    if 'sex' in df.columns:
        genders = ['All'] + sorted(df['sex'].dropna().unique().tolist())
        selected_gender = st.sidebar.selectbox("Gender", genders)
    else:
        selected_gender = 'All'
    
    # Urban/Rural filter
    if 'urban_rural' in df.columns:
        area_types = ['All'] + sorted(df['urban_rural'].dropna().unique().tolist())
        selected_area = st.sidebar.selectbox("Urban/Rural", area_types)
    else:
        selected_area = 'All'
    
    # Apply filters
    df_filtered = df.copy()
    
    if selected_province != 'All' and 'province' in df.columns:
        df_filtered = df_filtered[df_filtered['province'] == selected_province]
    
    if selected_district != 'All' and 'district' in df.columns:
        df_filtered = df_filtered[df_filtered['district'] == selected_district]
    
    if selected_age != 'All' and 'age_group' in df.columns:
        df_filtered = df_filtered[df_filtered['age_group'] == selected_age]
    
    if selected_gender != 'All' and 'sex' in df.columns:
        df_filtered = df_filtered[df_filtered['sex'] == selected_gender]
    
    if selected_area != 'All' and 'urban_rural' in df.columns:
        df_filtered = df_filtered[df_filtered['urban_rural'] == selected_area]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Sample**: {len(df_filtered):,} youth")
    
    # Main dashboard
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Youth", f"{len(df_filtered):,}")
    
    with col2:
        if 'NEET' in df_filtered.columns:
            neet_rate = df_filtered['NEET'].mean() * 100
            st.metric("NEET Rate", f"{neet_rate:.1f}%")
        else:
            st.metric("NEET Rate", "N/A")
    
    with col3:
        if 'sex' in df_filtered.columns:
            female_pct = (df_filtered['sex'] == 'Female').mean() * 100
            st.metric("Female %", f"{female_pct:.1f}%")
        else:
            st.metric("Female %", "N/A")
    
    with col4:
        if 'urban_rural' in df_filtered.columns:
            rural_pct = (df_filtered['urban_rural'] == 'Rural').mean() * 100
            st.metric("Rural %", f"{rural_pct:.1f}%")
        else:
            st.metric("Rural %", "N/A")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Geographic", "⚠️ High-Risk Segments", "📥 Download"])
    
    with tab1:
        st.subheader("NEET Distribution")
        
        if 'NEET' in df_filtered.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # NEET by age group
                if 'age_group' in df_filtered.columns:
                    neet_by_age = df_filtered.groupby('age_group')['NEET'].agg(['mean', 'count']).reset_index()
                    neet_by_age['mean'] = neet_by_age['mean'] * 100
                    
                    fig = px.bar(neet_by_age, x='age_group', y='mean', 
                                title='NEET Rate by Age Group',
                                labels={'mean': 'NEET Rate (%)', 'age_group': 'Age Group'},
                                text='mean')
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # NEET by gender
                if 'sex' in df_filtered.columns:
                    neet_by_gender = df_filtered.groupby('sex')['NEET'].agg(['mean', 'count']).reset_index()
                    neet_by_gender['mean'] = neet_by_gender['mean'] * 100
                    
                    fig = px.bar(neet_by_gender, x='sex', y='mean',
                                title='NEET Rate by Gender',
                                labels={'mean': 'NEET Rate (%)', 'sex': 'Gender'},
                                text='mean',
                                color='sex')
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("NEET labels not available in data. Please run preprocessing notebook.")
    
    with tab2:
        st.subheader("Geographic Distribution")
        
        if 'NEET' in df_filtered.columns and 'province' in df_filtered.columns:
            # NEET by province
            neet_by_province = df_filtered.groupby('province').agg({
                'NEET': ['mean', 'sum', 'count']
            }).reset_index()
            neet_by_province.columns = ['Province', 'NEET Rate', 'NEET Count', 'Total']
            neet_by_province['NEET Rate'] = neet_by_province['NEET Rate'] * 100
            neet_by_province = neet_by_province.sort_values('NEET Rate', ascending=False)
            
            fig = px.bar(neet_by_province, x='Province', y='NEET Rate',
                        title='NEET Rate by Province',
                        labels={'NEET Rate': 'NEET Rate (%)'},
                        text='NEET Rate',
                        color='NEET Rate',
                        color_continuous_scale='Reds')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # District table (top 10 by NEET count)
            if 'district' in df_filtered.columns:
                st.markdown("#### Top 10 Districts by NEET Count")
                neet_by_district = df_filtered.groupby('district').agg({
                    'NEET': ['mean', 'sum', 'count']
                }).reset_index()
                neet_by_district.columns = ['District', 'NEET Rate', 'NEET Count', 'Total']
                neet_by_district['NEET Rate'] = neet_by_district['NEET Rate'] * 100
                neet_by_district = neet_by_district.sort_values('NEET Count', ascending=False).head(10)
                
                st.dataframe(neet_by_district.style.format({
                    'NEET Rate': '{:.1f}%',
                    'NEET Count': '{:.0f}',
                    'Total': '{:.0f}'
                }), use_container_width=True)
    
    with tab3:
        st.subheader("High-Risk Segments")
        st.markdown("Identify demographic segments with highest NEET rates for targeted interventions.")
        
        if 'NEET' in df_filtered.columns:
            # Create segments
            segment_cols = []
            if 'sex' in df_filtered.columns:
                segment_cols.append('sex')
            if 'age_group' in df_filtered.columns:
                segment_cols.append('age_group')
            if 'urban_rural' in df_filtered.columns:
                segment_cols.append('urban_rural')
            if 'province' in df_filtered.columns:
                segment_cols.append('province')
            
            if len(segment_cols) >= 2:
                segments = df_filtered.groupby(segment_cols).agg({
                    'NEET': ['mean', 'sum', 'count']
                }).reset_index()
                segments.columns = segment_cols + ['NEET Rate', 'NEET Count', 'Total']
                segments['NEET Rate'] = segments['NEET Rate'] * 100
                
                # Filter segments with at least 30 individuals
                segments = segments[segments['Total'] >= 30]
                segments = segments.sort_values('NEET Count', ascending=False).head(20)
                
                st.markdown("**Top 20 High-Risk Segments** (minimum 30 individuals per segment)")
                st.dataframe(segments.style.format({
                    'NEET Rate': '{:.1f}%',
                    'NEET Count': '{:.0f}',
                    'Total': '{:.0f}'
                }).background_gradient(subset=['NEET Rate'], cmap='Reds'), 
                use_container_width=True, height=400)
            else:
                st.info("Need at least 2 demographic variables for segment analysis.")
    
    with tab4:
        st.subheader("Download Data")
        st.markdown("Download filtered data for further analysis or intervention planning.")
        
        # Prepare download data
        download_cols = []
        for col in ['id_hash', 'age', 'sex', 'province', 'district', 'urban_rural', 'NEET']:
            if col in df_filtered.columns:
                download_cols.append(col)
        
        if 'neet_prob' in df_filtered.columns:
            download_cols.append('neet_prob')
        
        df_download = df_filtered[download_cols].copy()
        
        # Summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows to Download", f"{len(df_download):,}")
        with col2:
            st.metric("Columns", len(df_download.columns))
        
        # Download button
        csv = df_download.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"neet_predictions_{selected_province}_{selected_district}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.markdown("**Note**: Data is anonymized. Use for statistical analysis and program planning only.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        <p><strong>NEET Predictor Dashboard</strong> | Labour Force Survey 2020-21</p>
        <p>Data Science Team | October 2025</p>
        <p>⚠️ Model predictions are for decision support only. Always involve human judgment.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
