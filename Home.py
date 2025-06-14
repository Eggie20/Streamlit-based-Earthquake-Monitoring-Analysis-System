"""
Earthquake Data Dashboard
-------------------------
An interactive Streamlit application for visualizing and analyzing seismic activity data.
This dashboard provides real-time monitoring of earthquake events with filtering capabilities,
interactive maps, and statistical analysis.
"""

# Standard library imports
from datetime import datetime, timedelta

# Third-party imports
import numpy as np
import pandas as pd

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk

# UI Framework
import streamlit as st

# Import utility functions
from utils import load_data, apply_custom_styling, create_sidebar_filters, apply_data_filters

# ============================
# APPLICATION CONFIGURATION
# ============================

st.set_page_config(
    page_title="Earthquake Data Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_styling()

# ============================
# HEADER SECTION
# ============================

st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🌍 Enhanced Earthquake Dashboard</h1>
    <p style='text-align: center; color: #888888;'>Real-time interactive monitoring and analysis of seismic activities</p>
    """, unsafe_allow_html=True)

# ============================
# DATA LOADING
# ============================

# Load the data
df = load_data()

if df.empty:
    st.error("No data available. Please check that 'Earthquake_Data.csv' exists and is properly formatted.")
    st.stop()

# ============================
# SIDEBAR FILTERS
# ============================

date_range, min_magnitude, province_filter = create_sidebar_filters(df)
filtered_df = apply_data_filters(df, date_range, min_magnitude, province_filter)

# ============================
# DISPLAY KEY METRICS
# ============================

def display_key_metrics(filtered_df):
    """
    Display key metrics about the earthquake data in a row of cards.
    
    Parameters:
        filtered_df (pandas.DataFrame): Filtered earthquake data
    """
    col1, col2, col3, col4 = st.columns(4)
    
    # Total earthquake events
    with col1:
        st.metric("Total Events", len(filtered_df), 
                 help="Total number of earthquake events in the selected filters")
    
    # Average magnitude
    with col2:
        avg_mag = filtered_df['Magnitude'].mean() if not filtered_df.empty else 0
        st.metric("Average Magnitude", f"{avg_mag:.2f}",
                 help="Average earthquake magnitude in the selected filters")
    
    # Maximum magnitude
    with col3:
        max_mag = filtered_df['Magnitude'].max() if not filtered_df.empty else 0
        st.metric("Max Magnitude", f"{max_mag:.2f}",
                 help="Strongest earthquake magnitude in the selected filters")
    
    # Recent events (last 24 hours)
    with col4:
        recent_count = len(filtered_df[filtered_df['DateTime'] > datetime.now() - timedelta(days=1)])
        st.metric("Last 24h Events", recent_count,
                 help="Number of earthquakes in the last 24 hours")

display_key_metrics(filtered_df)

# Display warning alerts for recent strong earthquakes
recent_strong = filtered_df[
    (filtered_df['Magnitude'] >= 4.0) & 
    (filtered_df['DateTime'] > datetime.now() - timedelta(days=7))
]

if not recent_strong.empty:
    st.warning(f"⚠️ {len(recent_strong)} strong earthquakes (M4.0+) detected in the last 7 days!")

# ============================
# MAIN CONTENT
# ============================

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Interactive Map", "Recent Events", "Statistics"])

with tab1:
    st.subheader("Earthquake Distribution Map")
    
    if not filtered_df.empty:
        # Create interactive map with PyDeck
        view_state = pdk.ViewState(
            latitude=filtered_df["Latitude"].mean(),
            longitude=filtered_df["Longitude"].mean(),
            zoom=5,
            pitch=0
        )
        
        # Create scatter layer for earthquake points
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            filtered_df,
            get_position=["Longitude", "Latitude"],
            get_radius=["Magnitude * 5000"],
            get_fill_color=["255 * (Magnitude / 10)", "140 * (1 - Magnitude / 10)", "0", "180"],
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=1,
            radius_min_pixels=5,
            radius_max_pixels=30
        )
        
        # Create deck
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=view_state,
            layers=[scatter_layer],
            tooltip={
                "html": "<b>Location:</b> {Location}<br><b>Magnitude:</b> {Magnitude}<br><b>Depth:</b> {Depth} km<br><b>Date:</b> {DateTime}",
                "style": {
                    "backgroundColor": "#0E1117",
                    "color": "white"
                }
            }
        )
        
        # Render the map
        st.pydeck_chart(deck)
    else:
        st.info("No earthquakes match your current filter criteria. Try adjusting the filters.")

with tab2:
    st.subheader("Recent Earthquake Events")
    
    if not filtered_df.empty:
        # Sort by date, most recent first
        recent_quakes = filtered_df.sort_values("DateTime", ascending=False).head(10)
        
        # Create a table with key information
        recent_quakes_display = recent_quakes[["DateTime", "Magnitude", "Depth", "Location", "Province"]].copy()
        recent_quakes_display.columns = ["Date & Time", "Magnitude", "Depth (km)", "Location", "Province"]
        
        # Format the date column
        recent_quakes_display["Date & Time"] = recent_quakes_display["Date & Time"].dt.strftime("%Y-%m-%d %H:%M")
        
        # Display the table
        st.dataframe(recent_quakes_display, use_container_width=True)
        
        # Add a detailed view of the most recent event
        if len(recent_quakes) > 0:
            st.subheader("Most Recent Event Details")
            
            latest = recent_quakes.iloc[0]
            
            # Create two columns for the details
            col1, col2 = st.columns(2)
            
            with col1:
                # Display key metrics
                st.metric("Magnitude", f"{latest['Magnitude']:.1f}")
                st.metric("Depth", f"{latest['Depth']:.1f} km")
                st.metric("Date & Time", latest['DateTime'].strftime("%Y-%m-%d %H:%M:%S"))
                
            with col2:
                # Create a mini map for the latest event (using scatter_map instead of scatter_mapbox)
                fig = px.scatter_map(
                    pd.DataFrame([latest]),
                    lat="Latitude",
                    lon="Longitude",
                    size_max=15,
                    zoom=8,
                    center={"lat": latest['Latitude'], "lon": latest['Longitude']},
                    height=300
                )
                
                fig.update_traces(marker=dict(size=15, color="#FF4B4B"))
                fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No earthquakes match your current filter criteria. Try adjusting the filters.")

with tab3:
    st.subheader("Earthquake Statistics")
    
    if not filtered_df.empty:
        # Create two columns for statistics charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Magnitude distribution histogram
            fig = px.histogram(
                filtered_df, 
                x="Magnitude",
                nbins=20,
                color_discrete_sequence=["#FF4B4B"],
                title="Magnitude Distribution",
                labels={"Magnitude": "Magnitude", "count": "Number of Earthquakes"}
            )
            
            fig.update_layout(
                xaxis_title="Magnitude",
                yaxis_title="Number of Earthquakes",
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Depth distribution
            fig = px.histogram(
                filtered_df, 
                x="Depth",
                nbins=20,
                color_discrete_sequence=["#4B4BFF"],
                title="Depth Distribution",
                labels={"Depth": "Depth (km)", "count": "Number of Earthquakes"}
            )
            
            fig.update_layout(
                xaxis_title="Depth (km)",
                yaxis_title="Number of Earthquakes",
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        st.subheader("Earthquake Activity Over Time")
        
        # Group by date
        time_data = filtered_df.groupby(filtered_df['DateTime'].dt.date).size().reset_index()
        time_data.columns = ['Date', 'Count']
        
        # Create time series chart
        fig = px.line(
            time_data,
            x="Date",
            y="Count",
            title="Daily Earthquake Activity",
            labels={"Date": "Date", "Count": "Number of Earthquakes"}
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Earthquakes",
            showlegend=False
        )
        
        # Add 7-day moving average
        time_data['MA7'] = time_data['Count'].rolling(window=7).mean()
        fig.add_scatter(
            x=time_data['Date'], 
            y=time_data['MA7'], 
            mode='lines', 
            name='7-day Moving Average',
            line=dict(color='#FF4B4B', width=2, dash='dash')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No earthquakes match your current filter criteria. Try adjusting the filters.")

# ============================
# FOOTER SECTION
# ============================

st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #888888;'>
    Earthquake Data Dashboard | Interactive Visualization and Analysis Tool<br>
    Use the navigation sidebar to explore different analysis views.
    </p>
""", unsafe_allow_html=True)
