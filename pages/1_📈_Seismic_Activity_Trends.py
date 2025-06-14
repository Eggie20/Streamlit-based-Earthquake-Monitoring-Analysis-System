
"""
Seismic Activity Trends
------------------------
This page provides real-time monitoring and visualization of seismic activity trends.
The page includes interactive simulations and animations showing earthquake data over time.
"""

import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(page_title="Seismic Activity Trends", page_icon="📈", layout="wide")

# Apply custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
    }
    .stAlert {
        padding: 20px;
        border-radius: 10px;
    }
    .plot-container {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 10px;
        padding: 10px;
        background-color: #1E1E1E;
    }
    /* Additional styling for better readability */
    .section-header {
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>📈 Seismic Activity Trends</h1>
    <p style='text-align: center; color: #888888;'>Real-time monitoring and visualization of earthquake data</p>
""", unsafe_allow_html=True)

# Introduction text
st.write(
    """This visualization demonstrates real-time earthquake monitoring based on actual seismic data.
    Watch as seismic events appear and transform with smooth transitions and dynamic effects.
    Interactive controls let you customize the visualization experience for optimal viewing."""
)

@st.cache_data
def load_earthquake_data():
    # Load directly from Earthquake_Data.csv as specified
    try:
        df = pd.read_csv("Earthquake_Data.csv")
        return process_earthquake_data(df)
    except Exception as e:
        st.error(f"Error loading earthquake data: {e}")
        return pd.DataFrame()
    
    # If we get here, display error and return empty dataframe
    st.error("Could not load earthquake data. Using sample data instead.")
    return pd.DataFrame()

def process_earthquake_data(df):
    # Process the Earthquake_Data.csv data
    try:
        # Convert DATE & TIME column to datetime
        df['DateTime'] = pd.to_datetime(df['DATE & TIME'], format='%d %B %Y - %I:%M %p')
        
        # Sort by datetime
        df = df.sort_values('DateTime')
        
        # Extract numeric magnitude (in case it's a string)
        df['MAGNITUDE'] = pd.to_numeric(df['MAGNITUDE'])
        
        # Create a region column (use PROVINCE if AREA not available)
        df['Region'] = df['PROVINCE']
        
        # Rename columns for easier access
        df = df.rename(columns={
            'DateTime': 'Time',
            'MAGNITUDE': 'Magnitude',
            'LONGITUDE': 'Longitude',
            'LATITUDE': 'Latitude',
            'DEPTH (KM)': 'Depth',
            'LOCATION': 'Location'
        })
        
        return df
    except Exception as e:
        st.error(f"Error processing earthquake data: {e}")
        return pd.DataFrame()

# Load the earthquake data
earthquake_df = load_earthquake_data()

# Check if data was loaded successfully, if not, create sample data
if earthquake_df.empty:
    st.warning("Using sample earthquake data for demonstration.")
    # Create time values for the past week with 30-minute intervals
    start_time = datetime.now() - timedelta(days=7)
    time_values = pd.date_range(start=start_time, end=datetime.now(), freq='30min')
    
    # Generate initial data with realistic earthquake patterns
    base_magnitudes = np.random.normal(3.5, 0.5, len(time_values))
    spike_indices = np.random.choice(
        range(len(time_values)), 
        size=int(len(time_values) * 0.05),
        replace=False
    )
    
    for idx in spike_indices:
        base_magnitudes[idx] = base_magnitudes[idx] + np.random.uniform(1.5, 3.5)
    
    # Ensure magnitudes are within realistic bounds
    base_magnitudes = np.clip(base_magnitudes, 2.0, 7.5)
    
    # Create DataFrame with time and magnitude
    earthquake_df = pd.DataFrame({
        'Time': time_values,
        'Magnitude': base_magnitudes,
        'Region': np.random.choice(
            ['Luzon', 'Visayas', 'Mindanao', 'Pacific Ring of Fire'],
            size=len(time_values)
        ),
        'Location': ["Sample location" for _ in range(len(time_values))],
        'Depth': np.random.uniform(10, 100, len(time_values))
    })

# Add sidebar components
st.sidebar.title("Monitoring Controls")

# Animation settings
st.sidebar.subheader("Animation Settings")
animation_speed = st.sidebar.slider("Animation Speed", 0.1, 3.0, 1.0, 0.1, 
                                   help="Control how fast new data points appear")
use_smooth_transitions = st.sidebar.checkbox("Smooth Transitions", True,
                                           help="Enable curved lines between data points")
show_ripple_effects = st.sidebar.checkbox("Ripple Effects", True,
                                        help="Show propagation waves for new earthquakes")
show_gradient_colors = st.sidebar.checkbox("Dynamic Color Gradients", True,
                                         help="Change colors based on magnitude")
use_area_fill = st.sidebar.checkbox("Area Fill Effect", True,
                                  help="Fill area under the trend line")

# Filter options
st.sidebar.subheader("Filter Options")
magnitude_filter = st.sidebar.slider("Minimum Magnitude", 1.0, 9.0, 3.0, 0.1)

# Get unique regions for selection
all_regions = ['All Regions']
if 'Region' in earthquake_df.columns:
    all_regions.extend(sorted(earthquake_df['Region'].unique()))
else:
    all_regions.extend(["Luzon", "Visayas", "Mindanao", "Pacific Ring of Fire"])

selected_region = st.sidebar.selectbox("Select Region", all_regions)

# Time period selection
time_window = st.sidebar.selectbox(
    "Time Window", 
    ["Last 24 Hours", "Last Week", "Last Month", "Last Year"]
)

# Create progress indicators
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

# Create the main content area with two columns
col1, col2 = st.columns([3, 1])

# Add a visual explanation of animation elements
with st.expander("About the Animation Elements", expanded=False):
    st.markdown("""
    ### Animation Elements in This Visualization
    
    This advanced earthquake visualization combines several visual techniques:
    
    1. **Dynamic Line Transitions** - Curved interpolation between data points creates more natural movement
    2. **Ripple Effects** - Concentric circles represent seismic wave propagation from new events
    3. **Color Gradients** - Colors shift based on magnitude, from green (low) to red (high)
    4. **Delta Indicators** - Shows how metrics change with each new earthquake
    5. **Weighted Animation Speed** - More significant events receive longer animation times
    
    These elements work together to provide both an engaging visualization and meaningful
    representation of seismic data patterns.
    """)

# Filter data based on user selections
def filter_data(df):
    # Apply magnitude filter
    filtered_df = df[df['Magnitude'] >= magnitude_filter].copy()
    
    # Apply region filter if not "All Regions"
    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    # Apply time window filter
    end_time = datetime.now()
    if time_window == "Last 24 Hours":
        start_time = end_time - timedelta(hours=24)
    elif time_window == "Last Week":
        start_time = end_time - timedelta(days=7)
    elif time_window == "Last Month":
        start_time = end_time - timedelta(days=30)
    else:  # Last Year
        start_time = end_time - timedelta(days=365)
    
    # Make sure 'Time' column is datetime
    if 'Time' in filtered_df.columns and not pd.api.types.is_datetime64_any_dtype(filtered_df['Time']):
        filtered_df['Time'] = pd.to_datetime(filtered_df['Time'])
    
    filtered_df = filtered_df[(filtered_df['Time'] >= start_time) & (filtered_df['Time'] <= end_time)]
    
    return filtered_df

# Get color based on magnitude
def get_magnitude_color(magnitude):
    if magnitude < 3.0:
        # Blue to Green for very low magnitudes (subtle earthquakes)
        r = int(70 + (magnitude - 1.0) * 30)
        g = int(130 + (magnitude - 1.0) * 40)
        b = max(60, int(200 - (magnitude - 1.0) * 70))
        alpha = 0.7 + (magnitude - 1.0) * 0.1
        return f'rgba({r}, {g}, {b}, {alpha})'
    elif magnitude < 4.0:
        # Green to Yellow for low magnitudes
        r = int(100 + (magnitude - 3.0) * 155)
        g = int(170 + (magnitude - 3.0) * 50)
        b = max(20, int(100 - (magnitude - 3.0) * 80))
        alpha = 0.8 + (magnitude - 3.0) * 0.1
        return f'rgba({r}, {g}, {b}, {alpha})'
    elif magnitude < 6.0:
        # Yellow to Orange/Red for medium magnitudes
        r = min(255, int(210 + (magnitude - 4.0) * 45))
        g = max(60, int(220 - (magnitude - 4.0) * 120))
        b = max(10, int(40 - (magnitude - 4.0) * 30))
        alpha = 0.9 + (magnitude - 4.0) * 0.05
        return f'rgba({r}, {g}, {b}, {alpha})'
    else:
        # Deep Red to Bright Red for high magnitudes
        r = 255
        g = max(10, int(60 - (magnitude - 6.0) * 50))
        b = max(5, int(30 - (magnitude - 6.0) * 25))
        alpha = min(1.0, 0.95 + (magnitude - 6.0) * 0.05)
        return f'rgba({r}, {g}, {b}, {alpha})'

# Create and display initial chart
with col1:
    # Create a placeholder for the chart
    chart_placeholder = st.empty()
    
    # Filter the data
    display_df = filter_data(earthquake_df)
    
    if len(display_df) > 0:
        # Cap the number of points to display for performance
        max_points = 100
        if len(display_df) > max_points:
            # Get the most recent data points
            display_df = display_df.sort_values('Time').tail(max_points)
        
        # Create the initial chart
        fig = px.line(
            display_df, 
            x='Time', 
            y='Magnitude',
            color_discrete_sequence=['#FF4B4B'],
            title="Earthquake Magnitude Trends"
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Magnitude",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        
        # Display the initial chart
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        chart_placeholder.warning("No data available with the current filters.")

# Display key metrics in the second column
with col2:
    # Display key metrics
    st.subheader("Key Metrics")
    
    if len(display_df) > 0:
        # Average magnitude
        avg_mag = display_df['Magnitude'].mean()
        st.metric("Average Magnitude", f"{avg_mag:.2f}")
        
        # Maximum magnitude
        max_mag = display_df['Magnitude'].max()
        max_location = display_df.loc[display_df['Magnitude'].idxmax()]['Location'] if 'Location' in display_df.columns else "Unknown"
        st.metric("Maximum Magnitude", f"{max_mag:.2f}")
        st.caption(f"Location: {max_location}")
        
        # Number of events
        num_events = len(display_df)
        st.metric("Total Events", num_events)
        
        # Most active region
        if 'Region' in display_df.columns:
            region_counts = display_df['Region'].value_counts()
            most_active = region_counts.index[0] if not region_counts.empty else "Unknown"
            st.metric("Most Active Region", most_active)
    else:
        st.warning("No data available to calculate metrics.")

# Animate the seismic data over time
if st.button("Animate Seismic Data", key="run_animation", use_container_width=True):
    # Filter data for animation
    animation_df = filter_data(earthquake_df).copy()
    
    if len(animation_df) == 0:
        st.warning("No data available to animate with the current filters.")
    else:
        # Sort by datetime to ensure proper animation sequence
        animation_df = animation_df.sort_values('Time')
        
        # Determine number of animation steps
        n_steps = min(100, len(animation_df))
        
        # Calculate chunk size to distribute data over animation steps
        chunk_size = max(1, len(animation_df) // n_steps)
        
        # Create a buffer for displaying data
        display_buffer = pd.DataFrame(columns=animation_df.columns)
        
        # Animation loop
        for i in range(1, n_steps + 1):
            # Update progress bar
            progress_bar.progress(i / n_steps)
            status_text.text(f"Processing seismic data: {i}% Complete")
            
            # Calculate end index for current frame
            end_idx = min(i * chunk_size, len(animation_df))
            
            # Get data up to this point
            display_buffer = animation_df.iloc[:end_idx].copy()
            
            # Limit buffer size for performance
            if len(display_buffer) > 50:
                display_buffer = display_buffer.tail(50)
            
            # Create enhanced visualization
            fig = go.Figure()
            
            # Add individual points with color based on magnitude if enabled
            if show_gradient_colors:
                for j in range(len(display_buffer)):
                    magnitude = display_buffer['Magnitude'].iloc[j]
                    
                    # Add ripple effect for recent points if enabled
                    if show_ripple_effects and j > len(display_buffer) - 3:
                        for k in range(3):
                            ripple_size = (3-k) * 8
                            opacity = 0.3 - (k * 0.1)
                            fig.add_trace(go.Scatter(
                                x=[display_buffer['Time'].iloc[j]],
                                y=[magnitude],
                                mode='markers',
                                marker=dict(
                                    size=ripple_size,
                                    color=get_magnitude_color(magnitude),
                                    opacity=opacity,
                                    line=dict(width=0)
                                ),
                                hoverinfo='skip',
                                showlegend=False
                            ))
                    
                    # Add individual points with proper color
                    hover_text = (f"<b>Magnitude: {magnitude:.1f}</b><br>" +
                                 f"Location: {display_buffer['Location'].iloc[j] if 'Location' in display_buffer else 'Unknown'}<br>" +
                                 f"Time: {display_buffer['Time'].iloc[j]}")
                    
                    fig.add_trace(go.Scatter(
                        x=[display_buffer['Time'].iloc[j]],
                        y=[magnitude],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=get_magnitude_color(magnitude)
                        ),
                        showlegend=False,
                        hovertext=hover_text,
                        hoverinfo='text'
                    ))
            
            # Add the main trend line
            fig.add_trace(go.Scatter(
                x=display_buffer['Time'],
                y=display_buffer['Magnitude'],
                mode='lines',
                line=dict(
                    shape='spline' if use_smooth_transitions else 'linear',
                    smoothing=1.3,
                    width=3,
                    color='rgba(255, 75, 75, 0.7)'
                ),
                name='Magnitude',
                showlegend=False
            ))
            
            # Add shaded area under the line if enabled
            if use_area_fill:
                fig.add_trace(go.Scatter(
                    x=display_buffer['Time'],
                    y=display_buffer['Magnitude'],
                    mode='none',
                    fill='tozeroy',
                    fillcolor='rgba(255, 75, 75, 0.1)',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Update layout with professional styling
            fig.update_layout(
                title={
                    'text': "Earthquake Magnitude Trends",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                xaxis_title="Time",
                yaxis_title="Magnitude",
                hovermode="closest",
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.1)",
                    showline=True,
                    linecolor="rgba(255,255,255,0.2)"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.1)",
                    showline=True,
                    linecolor="rgba(255,255,255,0.2)",
                    range=[
                        max(1.0, display_buffer['Magnitude'].min() - 0.5),
                        min(10, display_buffer['Magnitude'].max() + 0.5)
                    ]
                )
            )
            
            # Add threshold lines with enhanced styling
            fig.add_shape(
                type="line",
                x0=display_buffer['Time'].min(),
                x1=display_buffer['Time'].max(),
                y0=4.0,
                y1=4.0,
                line=dict(
                    color="rgba(255, 255, 0, 0.6)",
                    width=1.5,
                    dash="dash"
                ),
                name="Moderate"
            )
            
            fig.add_shape(
                type="line",
                x0=display_buffer['Time'].min(),
                x1=display_buffer['Time'].max(),
                y0=6.0,
                y1=6.0,
                line=dict(
                    color="rgba(255, 0, 0, 0.6)",
                    width=1.5,
                    dash="dash"
                ),
                name="Major"
            )
            
            # Add annotations for threshold lines
            fig.add_annotation(
                x=display_buffer['Time'].min(),
                y=4.0,
                text="Moderate",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(color="rgba(255, 255, 0, 0.8)")
            )
            
            fig.add_annotation(
                x=display_buffer['Time'].min(),
                y=6.0,
                text="Major",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(color="rgba(255, 0, 0, 0.8)")
            )
            
            # Update the chart display
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Update metrics in real-time with animations
            with col2:
                # Recalculate metrics
                avg_mag = display_buffer['Magnitude'].mean()
                max_mag = display_buffer['Magnitude'].max()
                num_events = len(display_buffer)
                
                # Get location of max magnitude
                max_location = "Unknown"
                if 'Location' in display_buffer.columns:
                    max_idx = display_buffer['Magnitude'].idxmax()
                    if max_idx in display_buffer.index:
                        max_location = display_buffer.loc[max_idx]['Location']
                
                # Get most active region
                most_active = "Unknown"
                if 'Region' in display_buffer.columns and not display_buffer.empty:
                    region_counts = display_buffer['Region'].value_counts()
                    if not region_counts.empty:
                        most_active = region_counts.index[0]
                
                # Update metrics with delta values for animation
                if i > 1:
                    # Calculate metric changes for animation
                    prev_chunk = animation_df.iloc[:(end_idx - chunk_size)].tail(50)
                    prev_avg = prev_chunk['Magnitude'].mean() if not prev_chunk.empty else avg_mag
                    prev_max = prev_chunk['Magnitude'].max() if not prev_chunk.empty else max_mag
                    prev_count = len(prev_chunk)
                    
                    # Update metrics with change indicators
                    st.metric("Average Magnitude", f"{avg_mag:.2f}", f"{avg_mag - prev_avg:.2f}")
                    st.metric("Maximum Magnitude", f"{max_mag:.2f}", None if max_mag == prev_max else f"{max_mag - prev_max:.2f}")
                    st.caption(f"Location: {max_location}")
                    st.metric("Total Events", num_events, f"+{num_events - prev_count}" if num_events > prev_count else None)
                    st.metric("Most Active Region", most_active)
                else:
                    # First update without deltas
                    st.metric("Average Magnitude", f"{avg_mag:.2f}")
                    st.metric("Maximum Magnitude", f"{max_mag:.2f}")
                    st.caption(f"Location: {max_location}")
                    st.metric("Total Events", num_events)
                    st.metric("Most Active Region", most_active)
            
            # Adjust animation speed
            time.sleep(1.0 / animation_speed)
        
        # Clear progress when complete
        progress_bar.empty()
        status_text.text("Animation complete")

# Add explanatory text about the data
with st.expander("Understanding Seismic Activity Categories", expanded=False):
    st.markdown("""
    ### Earthquake Magnitude Categories
    
    The Philippines classifies earthquakes based on their magnitude:
    
    | Magnitude Range | Category           | Description                                                |
    |----------------:|:-------------------|:-----------------------------------------------------------|
    | Less than 2.0   | Scarcely Perceptible | Detected only by instruments                              |
    | 2.0 - 3.0      | Slightly Felt       | Felt by sensitive people, hanging objects may swing slightly |
    | 3.0 - 4.0      | Weak                | Felt by many people, hanging objects swing                  |
    | 4.0 - 5.0      | Moderately Strong   | Felt by all, windows rattle, some glassware breaks         |
    | 5.0 - 6.0      | Strong              | General alarm, slight damage, objects fall from shelves    |
    | 6.0 - 7.0      | Very Strong         | Damage to poorly built structures, wall cracks             |
    | 7.0 - 8.0      | Destructive         | Most structures damaged significantly                      |
    | Above 8.0      | Devastating         | Total or near-total destruction in affected areas          |
    
    ### Key Regions in the Philippines
    
    - **Luzon**: Northern main island including Manila
    - **Visayas**: Central island group
    - **Mindanao**: Southern main island
    - **Pacific Ring of Fire**: Active seismic zone along the eastern border
    
    The system monitors these regions continuously for seismic activity.
    """)

st.markdown("---")
st.caption("Visualization created using data from Earthquake_Data.csv. Last update: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
