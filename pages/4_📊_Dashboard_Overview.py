"""
Dashboard Overview Page for the Earthquake Data Dashboard
--------------------------------------------------------
Provides a high-level summary of the entire dashboard with navigation
and quick explanations of each section's purpose and features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Earthquake Dashboard Overview",
    page_icon="📊",
    layout="wide"
)

# Dashboard title with styling
st.markdown("""
    <style>
    .dashboard-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .overview-card {
        border: 1px solid #f0f0f0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .overview-card:hover {
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .section-icon {
        font-size: 30px;
        margin-bottom: 10px;
    }
    .nav-button {
        width: 100%;
        padding: 10px 0;
        border-radius: 5px;
        margin-top: 15px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        text-align: center;
        cursor: pointer;
    }
    .nav-button:hover {
        background-color: #45a049;
    }
    /* Text color styling */
    .text-primary {
        color: #2a5298;
    }
    .text-secondary {
        color: #4CAF50;
    }
    .text-accent {
        color: #FF5722;
    }
    .text-info {
        color: #03A9F4;
    }
    .text-warning {
        color: #FFC107;
    }
    .text-danger {
        color: #F44336;
    }
    h3 {
        color: #000000;
    }
    .overview-card p {
        color: #555555;
    }
    .overview-card ul li {
        color: #333333;
    }
    .overview-card b {
        color: #1e3c72;
    }
    .metric-label {
        color: #2a5298;
        font-weight: bold;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.2em;
    }
    </style>
    <div class="dashboard-header">
        <h1>📊 Earthquake Data Dashboard Overview</h1>
        <p>A comprehensive platform for monitoring and analyzing seismic activity</p>
    </div>
    """, unsafe_allow_html=True)

# Load data for overview statistics
@st.cache_data
def load_data():
    df = pd.read_csv("Earthquake_Data.csv")
    # Convert column names to uppercase for consistency
    df.columns = df.columns.str.upper()
    # Parse the DATE & TIME column using the new format: "31 January 2023 - 11:58 PM"
    df['DATETIME'] = pd.to_datetime(df['DATE & TIME'], format="%d %B %Y - %I:%M %p", errors='coerce')
    # Convert numeric columns
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df['MAGNITUDE'] = pd.to_numeric(df['MAGNITUDE'], errors='coerce')
    df['DEPTH (KM)'] = pd.to_numeric(df['DEPTH (KM)'], errors='coerce')
    return df

# Load and process data
try:
    df = load_data()
    
    # Display high-level summary statistics
    st.markdown("<h2 class='text-primary'>Dashboard Summary</h2>", unsafe_allow_html=True)
    
    # Calculate key statistics
    total_events = len(df)
    avg_magnitude = df["MAGNITUDE"].mean()
    max_magnitude = df["MAGNITUDE"].max()
    earliest_date = df["DATETIME"].min().date()
    latest_date = df["DATETIME"].max().date()
    date_range = (latest_date - earliest_date).days
    
    # Display summary metrics in nice columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<div class='metric-label'>Total Seismic Events</div><div class='metric-value'>{total_events:,}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Average Magnitude</div><div class='metric-value'>{avg_magnitude:.2f}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div class='metric-label'>Maximum Magnitude</div><div class='text-danger metric-value'>{max_magnitude:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Data Timespan</div><div class='metric-value'>{date_range} days</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"<div class='metric-label'>Date Range</div><div class='metric-value'>{earliest_date} to {latest_date}</div>", unsafe_allow_html=True)
        # Calculate events per day
        st.markdown(f"<div class='metric-label'>Events per Day (avg)</div><div class='metric-value'>{total_events / max(1, date_range):.1f}</div>", unsafe_allow_html=True)
    
    # Small summary visualization
    st.markdown("<h2 class='text-primary'>Quick Visualization</h2>", unsafe_allow_html=True)
    
    # Create a simple magnitude histogram
    fig = px.histogram(
        df, 
        x="MAGNITUDE", 
        nbins=30, 
        title="Distribution of Earthquake Magnitudes",
        color_discrete_sequence=["#3366CC"]
    )
    
    fig.update_layout(
        xaxis_title="Magnitude",
        yaxis_title="Number of Events",
        bargap=0.1,
        plot_bgcolor="white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Dashboard sections overview
    st.markdown("<h2 class='text-primary'>Dashboard Sections</h2>", unsafe_allow_html=True)
    
    # Create overview cards for each section
    col1, col2 = st.columns(2)
    
    with col1:
        # Seismic Activity Trends card
        st.markdown("""
        <div class="overview-card">
            <div class="section-icon">📈</div>
            <h3 style='color: #000000;'>Seismic Activity Trends</h3>
            <p>Visualize real-time earthquake data with dynamic animations showing magnitude changes over time. Includes interactive controls and filters for customized monitoring.</p>
            <p><b>Key Features:</b></p>
            <ul>
                <li><span class="text-info">Real-time data simulation</span></li>
                <li><span class="text-info">Color-coded magnitude representation</span></li>
                <li><span class="text-info">Interactive animation controls</span></li>
                <li><span class="text-info">Automatic aftershock detection</span></li>
            </ul>
            <div class="nav-button" onclick="window.location.href='/1_%F0%9F%93%88_Seismic_Activity_Trends'">Go to Seismic Activity Trends</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Regional Analysis card
        st.markdown("""
        <div class="overview-card">
            <div class="section-icon">📊</div>
            <h3 style='color: #000000;'>Regional Analysis</h3>
            <p>Analyze seismic activity patterns by geographic regions with interactive filters and animated visualizations showing cumulative event counts.</p>
            <p><b>Key Features:</b></p>
            <ul>
                <li><span class="text-secondary">Region-based filtering</span></li>
                <li><span class="text-secondary">Animated data visualization</span></li>
                <li><span class="text-secondary">Comparative regional activity</span></li>
                <li><span class="text-secondary">Statistical summaries</span></li>
            </ul>
            <div class="nav-button" onclick="window.location.href='/3_%F0%9F%93%8A_Regional_Analysis'">Go to Regional Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Map View card
        st.markdown("""
        <div class="overview-card">
            <div class="section-icon">🗺️</div>
            <h3 style='color: #000000;'>Map View</h3>
            <p>Explore seismic events on an interactive map with advanced animations showing shockwave propagation and epicenter locations. Includes detailed filters and camera controls.</p>
            <p><b>Key Features:</b></p>
            <ul>
                <li><span class="text-accent">Advanced 3D visualization</span></li>
                <li><span class="text-accent">Realistic shockwave effects</span></li>
                <li><span class="text-accent">Sequential earthquake playback</span></li>
                <li><span class="text-accent">Customizable visual parameters</span></li>
            </ul>
            <div class="nav-button" onclick="window.location.href='/2_%F0%9F%97%BA%EF%B8%8F_Map_View'">Go to Map View</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Intensity Scale card
        st.markdown("""
        <div class="overview-card">
            <div class="section-icon">ℹ️</div>
            <h3 style='color: #000000;'>Intensity Scale</h3>
            <p>Reference guide to the PHIVOLCS Earthquake Intensity Scale (PEIS) with detailed explanations, visual representations, and practical examples of intensity levels.</p>
            <p><b>Key Features:</b></p>
            <ul>
                <li><span class="text-warning">Complete PEIS scale explanation</span></li>
                <li><span class="text-warning">Visual intensity representations</span></li>
                <li><span class="text-warning">Radial propagation visualization</span></li>
                <li><span class="text-warning">Damage assessment reference</span></li>
            </ul>
            <div class="nav-button" onclick="window.location.href='/4_%E2%84%B9%EF%B8%8F_Intensity_Scale'">Go to Intensity Scale</div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading dashboard overview: {e}")
    st.warning("Please check the data source and reload the page.")

# Footer
st.markdown("---")
st.markdown("<div style='color: #777777;'>Dashboard last updated: " + datetime.now().strftime('%Y-%m-%d') + ". Earthquake data courtesy of PHIVOLCS.</div>", unsafe_allow_html=True)