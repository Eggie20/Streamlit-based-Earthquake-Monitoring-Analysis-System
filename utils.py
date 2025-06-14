"""
Utility functions for the Earthquake Dashboard application.
Provides data loading, preprocessing, and other common functionalities.
"""

import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta

@st.cache_data
def load_data():
    """
    Load and preprocess earthquake data from CSV file.
    
    Returns:
        pandas.DataFrame: Cleaned and processed earthquake data
    """
    try:
        df = pd.read_csv("Earthquake_Data.csv")
        
        # Standardize column names
        column_mapping = {
            'DATE & TIME': 'Date & Time',
            'LATITUDE': 'Latitude',
            'LONGITUDE': 'Longitude',
            'DEPTH (KM)': 'Depth',
            'MAGNITUDE': 'Magnitude',
            'LOCATION': 'Location',
            'DATE': 'Date',
            'TIME': 'Time',
            'CATEGORY': 'Category',
            'AREA': 'Area',
            'PROVINCE': 'Province'
        }
        
        # Apply column renaming if the columns exist
        df = df.rename(columns={col: column_mapping[col] for col in df.columns if col in column_mapping})

        # Convert date & time to datetime with specific format
        df['DateTime'] = pd.to_datetime(df['Date & Time'], format='%d %B %Y - %I:%M %p', errors='coerce')

        # Convert necessary columns to proper data types
        numeric_columns = ["Latitude", "Longitude", "Depth", "Magnitude"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where critical numeric values are missing
        df = df.dropna(subset=numeric_columns)

        # Add month and year columns for time-based analysis
        df['Month'] = df['DateTime'].dt.month
        df['Year'] = df['DateTime'].dt.year
        
        # Ensure Date column exists
        if 'Date' not in df.columns:
            df['Date'] = df['DateTime'].dt.date

        # Drop rows with NaN in key columns
        return df.dropna(subset=['DateTime', 'Latitude', 'Longitude', 'Magnitude'])
        
    except FileNotFoundError:
        st.error("Error: Data file not found. Please ensure 'Earthquake_Data.csv' exists in the application directory.")
        return pd.DataFrame()  # Return empty DataFrame
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame

def apply_custom_styling():
    """
    Apply custom CSS styling to enhance the visual appearance of the dashboard.
    Includes styling for metrics, alerts, and plot containers.
    """
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
        .data-table {
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

def create_sidebar_filters(df):
    """
    Create and display sidebar filtering options for the earthquake data.
    
    Parameters:
        df (pandas.DataFrame): The earthquake dataset
        
    Returns:
        tuple: Contains filter parameters (date_range, min_magnitude, province_filter)
    """
    st.sidebar.title("Dashboard Controls")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df['DateTime'].min().date(), df['DateTime'].max().date()],
        help="Filter earthquakes by date range"
    )
    
    # Magnitude filter
    min_magnitude = st.sidebar.slider(
        "Minimum Magnitude",
        float(df['Magnitude'].min()),
        float(df['Magnitude'].max()),
        3.0,
        help="Filter earthquakes by minimum magnitude"
    )
    
    # Province filter
    province_options = sorted(df['Province'].dropna().unique())
    province_filter = st.sidebar.multiselect(
        "Select Provinces",
        options=province_options,
        default=[],
        key="main_province_filter",
        help="Filter earthquakes by province"
    )
    
    return date_range, min_magnitude, province_filter

def apply_data_filters(df, date_range, min_magnitude, province_filter):
    """
    Apply the selected filters to the earthquake dataset.
    
    Parameters:
        df (pandas.DataFrame): The earthquake dataset
        date_range (list): Start and end dates for filtering
        min_magnitude (float): Minimum earthquake magnitude to include
        province_filter (list): List of provinces to include
        
    Returns:
        pandas.DataFrame: Filtered earthquake data
    """
    # Apply date and magnitude filters
    mask = (
        (df['DateTime'].dt.date >= date_range[0]) &
        (df['DateTime'].dt.date <= date_range[1]) &
        (df['Magnitude'] >= min_magnitude)
    )
    
    # Apply province filter if selected
    if province_filter:
        mask &= df['Province'].isin(province_filter)
    
    # Return filtered data
    return df[mask]

def get_intensity_scale_descriptions():
    """
    Get the PHIVOLCS Earthquake Intensity Scale (PEIS) descriptions and color mappings.
    
    Returns:
        dict: Dictionary containing intensity scales, descriptions, and color mappings
    """
    intensity_scale = {
        "I": {
            "title": "SCARCELY PERCEPTIBLE",
            "description": "Perceptible to people under favorable circumstances. Delicately balanced objects may swing.",
            "color": "#E8F5E9"
        },
        "II": {
            "title": "SLIGHTLY FELT",
            "description": "Felt by few individuals at rest indoors. Hanging objects may swing slightly. Still water in containers may be slightly disturbed.",
            "color": "#C8E6C9"
        },
        "III": {
            "title": "WEAK",
            "description": "Felt by many people indoors especially in upper floors of buildings. Vibration is felt like one passing of a light truck. Dizziness and nausea may be experienced. Hanging objects swing moderately. Still water in containers oscillates moderately.",
            "color": "#A5D6A7"
        },
        "IV": {
            "title": "MODERATELY STRONG",
            "description": "Felt generally by people indoors and by some people outdoors. Light sleepers are awakened. Vibration is felt like a passing of heavy truck. Hanging objects swing considerably. Dinner, plates, glasses, windows and doors rattle. Floors and walls creak. Furniture shakes visibly. Liquids in containers are slightly disturbed. Water in containers oscillate strongly. Standing motor cars may rock slightly.",
            "color": "#4CAF50"
        },
        "V": {
            "title": "STRONG",
            "description": "Generally felt by most people indoors and outdoors. Many sleeping people are awakened. Some are frightened, some run outdoors. Strong shaking and rocking felt throughout building. Hanging objects swing violently. Dining utensils clatter and clink; some are broken. Small, light and unstable objects may fall or overturn. Liquids spill from filled open containers. Standing vehicles rock noticeably. Shaking of leaves and twigs of trees are noticeable.",
            "color": "#FFEB3B"
        },
        "VI": {
            "title": "VERY STRONG",
            "description": "Many people are frightened; many run outdoors. Some people lose their balance. Motorists feel like driving in flat tires. Heavy objects or furniture move or may be shifted. Small church bells may ring. Wall plaster may crack. Very old or poorly built houses and man-made structures are slightly damaged though well-built structures are not affected. Limited rockfalls and rolling boulders occur in hilly to mountainous areas and escarpments. Trees are noticeably shaken.",
            "color": "#FFA000"
        },
        "VII": {
            "title": "DESTRUCTIVE",
            "description": "Most people are frightened and run outdoors. People find it difficult to stand in upper floors. Heavy objects and furniture overturn or topple. Big church bells may ring. Old or poorly-built structures suffer considerably damage. Some well-built structures are slightly damaged. Some cracks may appear on dikes, fish ponds, road surface, or concrete hollow block walls. Limited liquefaction, lateral spreading and landslides are observed. Trees are shaken strongly. (Liquefaction is a process by which loose saturated soil loses strength during an earthquake and behaves as fluid).",
            "color": "#FF5722"
        },
        "VIII": {
            "title": "VERY DESTRUCTIVE",
            "description": "People panic. People find it difficult to stand even outdoors. Many well-built buildings are considerably damaged. Concrete dikes and foundation of bridges are destroyed by ground settling or toppling. Railway tracks are bent or broken. Tombstones may be displaced, twisted or overturned. Utility posts, towers and monuments may tilt or topple. Water and sewer pipes may be bent, twisted or broken. Liquefaction and lateral spreading cause man-made structure to sink, tilt or topple. Numerous landslides and rockfalls occur in mountainous and hilly areas. Boulders are thrown out from their positions particularly near the epicenter. Fissures and faults rapture may be observed. Trees are violently shaken. Water splash or slop over dikes or banks of rivers.",
            "color": "#D32F2F"
        },
        "IX": {
            "title": "DEVASTATING",
            "description": "People are forcibly thrown to ground. Many cry and shake with fear. Most buildings are totally damaged. Bridges and elevated concrete structures are toppled or destroyed. Numerous utility posts, towers and monument are tilted, toppled or broken. Water sewer pipes are bent, twisted or broken. Landslides and liquefaction with lateral spreadings and sandboils are widespread. The ground is distorted into undulations. Trees are shaken very violently with some toppled or broken. Boulders are commonly thrown out. River water splashes violently on slops over dikes and banks.",
            "color": "#B71C1C"
        },
        "X": {
            "title": "COMPLETELY DEVASTATING",
            "description": "Practically all man-made structures are destroyed. Massive landslides and liquefaction, large scale subsidence and uplifting of land forms and many ground fissures are observed. Changes in river courses and destructive seiches in lakes occur. Many trees are toppled, broken and uprooted.",
            "color": "#7B1FA2"
        }
    }
    
    return intensity_scale
