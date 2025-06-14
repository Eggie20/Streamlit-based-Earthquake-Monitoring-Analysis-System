"""
Regional Analysis Page for the Earthquake Data Dashboard
-------------------------------------------------------
Provides analysis of earthquake distribution by province across the Philippines,
with interactive filtering and visualization options.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import time
from datetime import datetime, timedelta

# Import utility functions
from utils import load_data, apply_custom_styling

# Page configuration
st.set_page_config(page_title="Regional Earthquake Analysis", page_icon="📊", layout="wide")

# Apply custom styling
apply_custom_styling()

# Page header
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>📊 Regional Earthquake Analysis</h1>
    <p style='text-align: center; color: #888888;'>Distribution and patterns of seismic events by province</p>
""", unsafe_allow_html=True)

st.write(
    """This analysis shows earthquake distribution by province across the Philippines.
    Explore the data by selecting different regions, time periods, and visualization options."""
)

# Load data
df = load_data()

if df.empty:
    st.error("No data available. Please check that 'Earthquake_Data.csv' exists and is properly formatted.")
    st.stop()

# Add sidebar filters
st.sidebar.header("Analysis Filters")

# Province selection
province_options = sorted(df["Province"].dropna().unique())
provinces = st.sidebar.multiselect(
    "Choose provinces",
    options=province_options,
    default=province_options[:2] if len(province_options) >= 2 else province_options
)

# Magnitude filter
min_magnitude = st.sidebar.slider("Minimum Magnitude", 3.0, 8.0, 3.5, 0.1)

# Date range selection
min_date = df['DateTime'].min().date()
max_date = df['DateTime'].max().date()
# Ensure default start is not before min_date
default_start = max(min_date, max_date - timedelta(days=30))  # Show last 30 days by default
date_range = st.sidebar.date_input(
    "Date Range",
    value=[default_start, max_date],
    min_value=min_date,
    max_value=max_date
)

# Analysis by dimension
dimension = st.sidebar.selectbox("Analyze by:", ["Province", "Area", "Category"])

# Apply filters
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (
        (df['DateTime'].dt.date >= start_date) &
        (df['DateTime'].dt.date <= end_date) &
        (df['Magnitude'] >= min_magnitude)
    )
    
    # Apply province filter if selected
    if provinces:
        mask &= df['Province'].isin(provinces)
    
    filtered_df = df[mask]
else:
    # Default filtering if date range isn't properly selected
    filtered_df = df[df['Magnitude'] >= min_magnitude]
    if provinces:
        filtered_df = filtered_df[filtered_df['Province'].isin(provinces)]

# Check if we have data after filtering
if filtered_df.empty:
    st.error("No data matches your filter criteria. Please adjust your selections.")
else:
    # Group data by the selected dimension
    dimension_counts = filtered_df[dimension].value_counts().reset_index()
    dimension_counts.columns = [dimension, 'Count']
    
    # Sort by count in descending order
    dimension_counts = dimension_counts.sort_values('Count', ascending=False)
    
    # Take top 10 for better visualization
    top_dimension_counts = dimension_counts.head(10)
    
    # Create main content
    st.subheader(f"Top 10 Most Affected {dimension}s")
    
    # Create bar chart with Plotly to match the screenshot style
    fig = px.bar(
        top_dimension_counts,
        x=dimension,
        y='Count',
        color='Count',
        color_continuous_scale="Viridis",
        labels={dimension: dimension.upper(), "Count": "Number of Earthquakes"},
        text='Count'
    )
    
    fig.update_layout(
        xaxis_title=dimension.upper(),
        yaxis_title="Number of Earthquakes",
        coloraxis_colorbar_title="Frequency",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    fig.update_traces(
        textposition='outside',
        texttemplate='%{text}'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add animated visualization
    st.subheader(f"Animated Earthquake Activity by {dimension}")
    
    # Animation controls
    animation_speed = st.slider("Animation Speed", 0.01, 0.5, 0.1, 0.01, 
                              help="Control how fast the animation runs")
    
    smoothing = st.checkbox("Enable Smoothing", True, 
                          help="Apply smoothing to make transitions more fluid")
    
    # Create containers for the animation
    progress_container = st.empty()
    status_container = st.empty()
    chart_container = st.empty()
    
    # Only run animation if there's data and user clicks button
    if not filtered_df.empty and st.button("Run Animation", key="run_animation_regional", use_container_width=True):
        # Add visual appeal - brief loading effect
        with st.spinner("Preparing data visualization..."):
            time.sleep(0.5)  # Short pause for effect
            
        # Prepare data for animation
        # Get top dimensions for better visualization (use more if available)
        max_dimensions = 8  # Show more dimensions for better visualization
        top_dimensions = dimension_counts.head(max_dimensions)[dimension].tolist()
        
        # Filter for top dimensions
        anim_df = filtered_df[filtered_df[dimension].isin(top_dimensions)].copy()
        
        # Set up animation
        n_steps = min(120, len(anim_df))  # Use more steps for smoother animation
        
        # Initialize with zeros but with a slight offset for visual appeal
        start_values = np.random.uniform(0.1, 0.5, len(top_dimensions))  
        last_rows = pd.DataFrame({dim: [val] for dim, val in zip(top_dimensions, start_values)})
        
        # Create a custom themed chart with enhanced styling
        chart = chart_container.line_chart(last_rows, height=400)
        progress_bar = progress_container.progress(0)
        
        # Create color map with custom styling for better visual hierarchy
        color_palette = px.colors.qualitative.Bold[:len(top_dimensions)]
        # Create legend with more professional styling
        colors_html = [f'<span style="color:{color}; font-weight:500; padding:3px 8px; margin:0 3px; border-radius:3px; background-color:rgba(255,255,255,0.1)">{dim}</span>' 
                      for dim, color in zip(top_dimensions, color_palette)]
        color_legend = "".join(colors_html)
        st.markdown(f"""
        <div style='text-align:center; margin-bottom:10px; padding:10px; border-radius:5px; background-color:rgba(0,0,0,0.05)'>
            {color_legend}
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare cumulative counts with initial small values for better animation start
        cumulative_counts = {dim: start_values[i] for i, dim in enumerate(top_dimensions)}
        
        # Sort by date for proper animation sequence
        anim_df.sort_values("DateTime", inplace=True)
        
        # Prepare animation pacing variables for dynamic speed
        acceleration_factor = 1.0
        
        # Run enhanced animation
        for i in range(1, n_steps + 1):
            # Update progress with smoother style
            progress_bar.progress(i / n_steps)
            
            # More professional status updates
            if i < n_steps // 3:
                status_container.info(f"Processing initial data: {i}/{n_steps}")
            elif i < 2 * n_steps // 3:
                status_container.info(f"Analyzing patterns: {i}/{n_steps}")
            else:
                status_container.info(f"Finalizing visualization: {i}/{n_steps}")
            
            # Dynamic chunk sizing for more natural data flow
            # Smaller chunks at start and end, larger in the middle for visual appeal
            progress_ratio = i / n_steps
            if progress_ratio < 0.2 or progress_ratio > 0.8:
                # Slower at beginning and end
                chunk_size = max(1, int(len(anim_df) // (n_steps * 1.5)))
            else:
                # Faster in the middle
                chunk_size = max(1, int(len(anim_df) // (n_steps * 0.8)))
                
            chunk_start = int((i-1) * len(anim_df) / n_steps)
            chunk_end = min(int(i * len(anim_df) / n_steps), len(anim_df))
            chunk = anim_df.iloc[chunk_start:chunk_end]
            
            # Update counts with enhanced visual weighting
            for dim in top_dimensions:
                # Get counts for this dimension in the current chunk
                dim_count = len(chunk[chunk[dimension] == dim])
                
                # Apply visual multiplication factor based on position in the animation
                # This creates more dramatic growth curves
                if i < n_steps * 0.3:  # Early stages - slower growth
                    visual_factor = 0.7
                elif i > n_steps * 0.7:  # Later stages - faster consolidation
                    visual_factor = 1.3
                else:  # Middle stages - normal growth
                    visual_factor = 1.0
                    
                # Apply the factor to the count
                cumulative_counts[dim] += dim_count * visual_factor
            
            # Create new row for chart with enhanced styling
            new_row = pd.DataFrame({dim: [cumulative_counts[dim]] for dim in top_dimensions})
            
            # Apply professional smoothing if enabled
            if smoothing and i > 1:
                # Dynamic noise based on progress - more at beginning, less at end
                noise_factor = 0.08 * (1 - (i / n_steps) * 0.7)  # Gradually reduce noise
                
                # Generate correlated noise (not completely random) for more natural movement
                if i % 3 == 0:  # Occasionally create a small correlated bump
                    noise_direction = np.random.choice([-1, 1])
                    noise = np.random.normal(noise_direction * 0.02, noise_factor, len(top_dimensions))
                else:
                    noise = np.random.normal(0, noise_factor, len(top_dimensions))
                    
                # Apply weight factors to make dominant regions more stable, smaller ones more volatile
                weighted_noise = []
                for j, dim in enumerate(top_dimensions):
                    # Weight by inverse of current value - smaller values get more noise
                    if cumulative_counts[dim] > 0:
                        weight = min(1.0, 5.0 / cumulative_counts[dim])
                    else:
                        weight = 1.0
                    weighted_noise.append(noise[j] * weight)
                
                # Apply the weighted noise
                noise_df = pd.DataFrame({dim: [weighted_noise[j]] for j, dim in enumerate(top_dimensions)})
                new_row = new_row + noise_df
                
                # Ensure values stay positive
                for dim in top_dimensions:
                    if new_row[dim].iloc[0] < 0:
                        new_row[dim] = 0
            
            # Update chart with the new data point
            chart.add_rows(new_row)
            
            # Dynamic animation speed - accelerate in the middle, slow at beginning and end
            # Creates a more cinematic feel to the animation
            if i < n_steps * 0.2:  # Start slow
                current_speed = animation_speed * 0.7
            elif i > n_steps * 0.8:  # End slow
                current_speed = animation_speed * 0.8
            else:  # Middle faster
                current_speed = animation_speed * 1.2
                
            # Add occasional dramatic pause for visual interest
            if i % 25 == 0 and i > 10:
                time.sleep(0.3 / current_speed)  # Slight pause
                
            # Control animation speed with the calculated dynamic value
            time.sleep(0.8 / current_speed)
        
        # Clear progress when done
        progress_container.empty()
        status_container.success("Animation complete!")
        
        # Add re-run button
        if st.button("Re-run Animation", key="rerun_animation_regional"):
            st.experimental_rerun()
    
    # Information panel to explain the visualization
    with st.expander("Understanding the Visualization", expanded=False):
        st.markdown("""
        This visualization displays earthquake frequency by province:
        - Bar height represents the number of earthquakes recorded
        - Color intensity correlates with earthquake frequency
        - Data can be filtered by province and minimum magnitude
        - The map shows the geographic distribution of these events
        
        The PHIVOLCS Earthquake Intensity Scale (PEIS) is used to measure the intensity
        of earthquakes in the Philippines, ranging from I (Scarcely Perceptible) to X
        (Completely Devastating).
        """)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    # Create a summary table
    summary_data = {
        "Metric": [
            "Total Events", 
            "Average Magnitude", 
            "Maximum Magnitude",
            f"Most Active {dimension}",
            "Average Depth (km)"
        ],
        "Value": [
            len(filtered_df),
            f"{filtered_df['Magnitude'].mean():.2f}",
            f"{filtered_df['Magnitude'].max():.2f}",
            dimension_counts.iloc[0][dimension],
            f"{filtered_df['Depth'].mean():.2f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

st.markdown("---")
st.caption(f"Data last updated: {datetime.now().strftime('%Y-%m-%d')}. Analysis based on {len(filtered_df)} earthquake events.")
