"""
Intensity Scale Reference Page for the Earthquake Data Dashboard
--------------------------------------------------------------
Provides detailed information about the PHIVOLCS Earthquake Intensity Scale (PEIS)
with color-coded visualizations and explanatory text.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import utility functions
from utils import apply_custom_styling, get_intensity_scale_descriptions

# Page configuration
st.set_page_config(
    page_title="Earthquake Intensity Scale",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom styling
apply_custom_styling()

# Header
st.title("ℹ️ PHIVOLCS Earthquake Intensity Scale (PEIS)")

# Introduction
st.markdown("""
    The PHIVOLCS Earthquake Intensity Scale (PEIS) is a scale used in the Philippines to measure the 
    intensity of an earthquake. Unlike magnitude scales (like the Richter scale) which measure the 
    energy released by an earthquake, intensity scales measure the effects of an earthquake at a 
    specific location.
    
    Intensity varies depending on:
    - Distance from the epicenter
    - Local geological conditions
    - Building structures and quality
    - Population density
    
    This reference guide explains the different intensity levels and their corresponding effects.
""")

# Get intensity scale data
intensity_scale = get_intensity_scale_descriptions()

# Create tabs for different views
tab1, tab2 = st.tabs(["Detailed Reference", "Visual Comparison"])

with tab1:
    # Create detailed reference table
    st.header("Intensity Scale Reference")
    
    # Create a container for each intensity level
    for level, data in intensity_scale.items():
        with st.container():
            # Use columns for better layout
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Create a color box to represent the intensity
                st.markdown(f"""
                <div style="
                    background-color: {data['color']}; 
                    padding: 30px; 
                    border-radius: 10px; 
                    text-align: center;
                    color: {'black' if level in ['I', 'II', 'III', 'IV', 'V'] else 'white'};
                    font-weight: bold;
                    font-size: 24px;
                    ">
                    {level}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Display the intensity title and description
                st.markdown(f"""
                <h3 style="margin-top: 0;">{data['title']}</h3>
                <p>{data['description']}</p>
                """, unsafe_allow_html=True)
            
            # Add a separator
            st.markdown("<hr>", unsafe_allow_html=True)

with tab2:
    # Create visual comparison
    st.header("Visual Comparison of Intensity Levels")
    
    # Create data for visualization
    intensity_data = []
    for level, data in intensity_scale.items():
        intensity_data.append({
            "Level": level,
            "Title": data["title"],
            "Description": data["description"],
            "Color": data["color"],
            # Convert roman numerals to integers for proper ordering
            "Order": {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, 
                      "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}[level]
        })
    
    intensity_df = pd.DataFrame(intensity_data)
    
    # Sort by order
    intensity_df = intensity_df.sort_values("Order")
    
    # Create a bar chart visualization
    fig = px.bar(
        intensity_df,
        x="Level",
        y="Order",
        color="Level",
        text="Title",
        color_discrete_map={row["Level"]: row["Color"] for _, row in intensity_df.iterrows()},
        height=600,
        title="PHIVOLCS Earthquake Intensity Scale (PEIS) - Visual Comparison"
    )
    
    fig.update_traces(
        textposition="inside",
        textfont_color=["black" if i < 5 else "white" for i in intensity_df["Order"]],
        marker_line_width=0,
        width=0.7,
        hovertemplate="<b>Level %{x}: %{text}</b><extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title="Intensity Level",
        yaxis_title="",
        showlegend=False,
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional information about interpreting intensity
    st.subheader("How to Interpret Intensity Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            ### Intensity vs. Magnitude
            
            **Intensity (PEIS)** measures:
            - The effects at a specific location
            - Subjective perception by people
            - Damage to structures
            - Visual observations
            
            **Magnitude** measures:
            - Total energy released at the source
            - Measured by seismograph instruments
            - Single value for each earthquake
            - Independent of location
        """)
    
    with col2:
        st.markdown("""
            ### Factors Affecting Intensity
            
            The same earthquake can have different intensity values depending on:
            
            1. **Distance from epicenter**
               - Intensity generally decreases with distance
            
            2. **Soil and Geological Conditions**
               - Soft soil can amplify shaking
               - Solid bedrock reduces shaking
            
            3. **Building Construction**
               - Well-designed structures show less damage
               - Older buildings may show higher intensity effects
            
            4. **Population Density**
               - More observations in populated areas
        """)
    
    # Add a visual example of how intensity varies with distance
    st.subheader("Intensity Variation with Distance from Epicenter")
    
    # Create sample data for visualization
    distance = list(range(0, 101, 10))
    intensity_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
    
    # Create a proper radial visualization with matched array lengths
    # Create 36 angles (0 to 350 degrees)
    angles = list(range(0, 360, 10))  # 36 angles
    
    # We need exactly 36 r-values to match our 36 angles
    # Calculate how many points per intensity value
    points_per_value = len(angles) // len(intensity_values)
    remainder = len(angles) % len(intensity_values)
    
    # Create properly sized r_values array
    r_values = []
    for i, val in enumerate(intensity_values):
        # Add extra point to early values if we have a remainder
        extra = 1 if i < remainder else 0
        r_values.extend([val] * (points_per_value + extra))
    
    # Verify lengths match
    assert len(r_values) == len(angles), "Array lengths must match"
    
    # Create a radial visualization
    fig = px.line_polar(
        r=r_values,
        theta=angles,
        line_close=True
    )
    
    fig.update_traces(
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.3)'
    )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=list(range(1, 11)),
                ticktext=["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"],
                tickmode="array"
            ),
            angularaxis=dict(
                visible=True,
                tickvals=[0, 90, 180, 270],
                ticktext=["North", "East", "South", "West"]
            )
        ),
        showlegend=False,
        height=500,
        title="Example: How Intensity Decreases with Distance from Epicenter"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
        Note: This is a simplified visualization. Actual intensity patterns are influenced by many factors
        including geological formations, fault orientation, and directivity effects.
    """)
