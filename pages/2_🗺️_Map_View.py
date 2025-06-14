
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time
from datetime import datetime
import colorsys

st.set_page_config(page_title="Advanced Seismic Wave Visualization", layout="wide")
st.title("🗺️ Advanced Seismic Wave Visualization")

# ------------------------------------------------
# Data Loading and Preprocessing
# ------------------------------------------------
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
    # Ensure CATEGORY is uppercase
    df["CATEGORY"] = df["CATEGORY"].str.upper()

    # Handle potential NaN values in string columns
    string_columns = ['PROVINCE', 'AREA', 'CATEGORY', 'LOCATION']
    for col in string_columns:
        if col in df.columns:
            # Replace NaN with "Unknown" and ensure all values are strings
            df[col] = df[col].fillna("Unknown").astype(str)

    return df

try:
    df = load_data()

    # Information panel to explain the visualization
    with st.expander("Understanding the Visualization", expanded=False):
        st.markdown("""
        ### How to Read This Visualization

        This advanced earthquake visualization uses several visual elements to represent seismic activity:

        1. **Initial Burst**: The sudden, powerful energy release at the epicenter
        2. **Expanding Ripples**: Concentric rings showing how seismic waves propagate outward
        3. **Color Intensity**: Indicates the earthquake intensity category
        4. **Particle Effects**: Debris-like elements showing energy dispersion
        5. **Ripple Speed**: Faster-expanding ripples indicate more powerful seismic events
        6. **Depth Effect**: Deeper earthquakes produce different wave patterns

        The visualization has three modes:
        - **Static View**: Shows all earthquakes for the selected date
        - **Simultaneous Animation Mode**: Visualizes all earthquakes together 
        - **Sequential Animation Mode**: Visualizes each earthquake one by one with transition arrows

        ### Advanced Features & Critical Considerations

        #### Bound-to-Bound Transition Arrows
        The arrows connecting sequential events are a visual aid to follow chronological order, not an indication of causal relationships between earthquakes. Consider that:
        - Adjacent events in time may be entirely independent seismic phenomena
        - The smooth transitions prioritize visual continuity over representing actual geological processes
        - Alternative interpretation: These could represent how seismic monitoring attention shifted between events

        #### Sequential Event Playback
        The sequential animation treats all events equally in terms of timing, which raises questions:
        - Should more significant earthquakes be given longer focus?
        - Is chronological order always the most informative sequence?
        - Are small events given disproportionate visual importance compared to their actual impact?

        #### Camera & Visual Effects
        The dynamic camera movements create engaging visuals but:
        - Motion effects may exaggerate perceived intensity of smaller events
        - Concentric ripples are a simplified abstraction of complex wave propagation
        - Perfect circles don't reflect how actual seismic waves interact with varied terrain

        #### Performance Considerations
        The visualization is resource-intensive:
        - Complex animations may run differently across devices
        - High earthquake counts may affect performance
        - Consider using simpler visual settings on less powerful hardware

        ### Key Questions to Ask While Using This Tool

        1. **Scientific Accuracy vs. Visual Appeal**: Does this visualization prioritize education or aesthetics?

        2. **Pattern Recognition**: Are there spatial or temporal patterns visible that warrant further investigation?

        3. **Scale Perception**: Does the visualization adequately convey the vast differences in energy release between magnitude levels?

        4. **Accessibility**: How might this data be interpreted by users with different visual abilities?

        Customize the visualization using the controls in the sidebar to explore different visual representations of seismic data, and maintain a critical perspective on how visual choices influence data interpretation.
        """)

    # ------------------------------------------------
    # Enhanced Color Mapping with Intensity Levels
    # ------------------------------------------------
    # Define a function to create a gradient of colors for each intensity level
    def create_color_gradient(base_color, num_steps=5):
        """Create a gradient of colors from the base color to a lighter version"""
        r, g, b, a = base_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        colors = []
        for i in range(num_steps):
            # Decrease saturation and increase value for a "fading" effect
            new_s = max(0, s - (i * 0.15))
            new_v = min(1, v + (i * 0.15))
            new_r, new_g, new_b = colorsys.hsv_to_rgb(h, new_s, new_v)
            new_a = max(0, a - (i * 40))  # Gradually decrease opacity
            colors.append((int(new_r*255), int(new_g*255), int(new_b*255), int(new_a)))
        return colors

    intensity_base_colors = {
        "SCARCELY PERCEPTIBLE": (255, 255, 255, 200),    # White
        "SLIGHTLY FELT": (223, 230, 254, 200),            # GreenYellow
        "WEAK": (130, 249, 251, 200),                      # Yellow
        "MODERATELY STRONG": (130, 250, 224, 200),         # Gold
        "STRONG": (152, 247, 130, 200),                    # Orange
        "VERY STRONG": (247, 246, 80, 200),               # DarkOrange
        "DESTRUCTIVE": (252, 199, 67, 200),                # OrangeRed
        "VERY DESTRUCTIVE": (252, 109, 44, 200),            # Red
        "DEVASTATING": (232, 37, 29, 200),                 # DarkRed
        "COMPLETELY DEVASTATING": (196, 31, 24, 200),      # Maroon
        "UNKNOWN": (128, 128, 128, 200),                 # Gray
    }


    # Create gradients for each intensity category
    intensity_color_gradients = {category: create_color_gradient(color) 
                               for category, color in intensity_base_colors.items()}

    # Safely map base colors with a default for any unmapped categories
    df["COLOR"] = df["CATEGORY"].apply(lambda x: intensity_base_colors.get(x, (128, 128, 128, 200)))

    # ------------------------------------------------
    # Sidebar Filtering with "ALL" Options
    # ------------------------------------------------
    st.sidebar.header("Filters")

    # Province Filter: add "ALL" option
    # Fix: Convert all values to strings and handle NaN values before sorting
    all_provinces = sorted([p for p in df['PROVINCE'].unique() if p != "Unknown"])
    if "Unknown" in df['PROVINCE'].unique():
        all_provinces.append("Unknown")  # Add Unknown at the end if it exists

    selected_provinces = st.sidebar.multiselect("Select Province(s)", options=["ALL"] + all_provinces, default=["ALL"])
    if "ALL" in selected_provinces:
        selected_provinces = all_provinces

    # Intensity Category Filter: add "ALL" option
    all_categories = sorted([c for c in df["CATEGORY"].unique() if c != "Unknown"])
    if "Unknown" in df["CATEGORY"].unique():
        all_categories.append("Unknown")  # Add Unknown at the end if it exists

    selected_categories = st.sidebar.multiselect("Select Intensity Categories", options=["ALL"] + all_categories, default=["ALL"])
    if "ALL" in selected_categories:
        selected_categories = all_categories

    # Filter by date only (ignoring time)
    # Find the earliest and latest dates in the data
    valid_dates = df['DATETIME'].dropna()
    if len(valid_dates) > 0:
        min_date = valid_dates.dt.date.min()
        max_date = valid_dates.dt.date.max()
        default_date = max_date  # Default to the most recent date
    else:
        min_date = datetime.now().date()
        max_date = datetime.now().date()
        default_date = datetime.now().date()

    selected_date = st.sidebar.date_input("Select Date", value=default_date, min_value=min_date, max_value=max_date)

    # Apply filters
    filtered_df = df[
        df['PROVINCE'].isin(selected_provinces) &
        df["CATEGORY"].isin(selected_categories)
    ]

    # Safely filter by date
    filtered_df = filtered_df[filtered_df['DATETIME'].dt.date == selected_date]

    if filtered_df.empty:
        st.warning("No earthquake data for the selected filters.")
        st.stop()

    # Sort events in chronological order
    sorted_quakes = filtered_df.sort_values('DATETIME').reset_index(drop=True)

    # ------------------------------------------------
    # Enhanced Animation Settings
    # ------------------------------------------------
    st.sidebar.subheader("Advanced Animation Settings")

    # Animation Mode Selection
    st.sidebar.markdown("#### Animation Mode")
    animation_mode = st.sidebar.radio(
        "Select Animation Mode", 
        ["Simultaneous (All Events)", "Sequential (One by One)"], 
        index=0
    )

    # Shockwave Parameters
    st.sidebar.markdown("#### Shockwave Parameters")
    base_radius = st.sidebar.slider("Base Radius Multiplier", 1000, 15000, 5000, step=100)
    max_ripples = st.sidebar.slider("Number of Ripple Rings", 1, 8, 5, step=1)
    shockwave_speed = st.sidebar.slider("Shockwave Speed", 0.5, 10.0, 3.0, step=0.1)

    # Visual Effects
    st.sidebar.markdown("#### Visual Effects")
    pulse_amplitude = st.sidebar.slider("Pulse Amplitude", 0.1, 1.0, 0.5, step=0.05)
    pulse_frequency = st.sidebar.slider("Pulse Frequency", 0.05, 1.0, 0.2, step=0.05)
    motion_blur = st.sidebar.slider("Motion Blur Effect", 0.0, 1.0, 0.5, step=0.05)

    # Epicenter Effects
    st.sidebar.markdown("#### Epicenter Effects")
    epicenter_glow = st.sidebar.slider("Epicenter Glow Intensity", 0.1, 2.0, 1.0, step=0.1)
    initial_burst_size = st.sidebar.slider("Initial Burst Size", 0.5, 5.0, 2.0, step=0.1)
    use_burst_effect = st.sidebar.checkbox("Use Burst Effect (Instead of Needle)", False)
    burst_particles = st.sidebar.slider("Burst Particles", 0, 20, 0, step=1)
    burst_intensity = st.sidebar.slider("Burst Intensity", 0.1, 2.0, 1.0, step=0.1)

    # Sequential Animation Settings (NEW)
    if animation_mode == "Sequential (One by One)":
        st.sidebar.markdown("#### Sequential Animation Timing")
        event_duration = st.sidebar.slider("Event Duration (sec)", 3, 15, 8, step=1)
        transition_duration = st.sidebar.slider("Transition Duration (sec)", 1, 10, 3, step=1)
        pause_duration = st.sidebar.slider("Pause Between Events (sec)", 0.0, 5.0, 1.0, step=0.5)

        st.sidebar.markdown("#### Arrow Settings")
        arrow_thickness = st.sidebar.slider("Arrow Thickness", 1, 10, 3, step=1)
        arrow_head_size = st.sidebar.slider("Arrow Head Size", 1, 10, 4, step=1)
        arrow_color_picker = st.sidebar.color_picker("Arrow Color", "#FF5733")
        # Convert hex color to RGB tuple
        arrow_color = tuple(int(arrow_color_picker.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (200,)
    else:
        # Default values for simultaneous mode (to avoid errors)
        event_duration = st.sidebar.slider("Duration per Event (sec)", 3, 15, 8, step=1)
        transition_duration = 3
        pause_duration = 1.0
        arrow_thickness = 3
        arrow_head_size = 4
        arrow_color = (255, 87, 51, 200)  # Default orange with alpha

    # Camera Settings
    st.sidebar.markdown("#### Camera Settings")
    camera_options = ["Top-down", "Tilted View", "Dynamic Camera"]
    selected_camera = st.sidebar.selectbox("Camera Angle", camera_options, index=1)

    # Animation physics
    show_depth_effect = st.sidebar.checkbox("Show Depth Effect", True)
    enable_terrain_interaction = st.sidebar.checkbox("Enable Terrain Interaction", True)

    # ------------------------------------------------
    # Animation Setup
    # ------------------------------------------------
    map_container = st.empty()

    # Show different info message based on animation mode
    if animation_mode == "Sequential (One by One)":
        st.info("Click 'Start Animation' to visualize each earthquake sequentially with transition arrows between events.")
    else:
        st.info("Click 'Start Animation' to visualize the seismic waves for all earthquake events simultaneously.")

    # Display a summary of events for the selected date
    with st.expander("Events on Selected Date", expanded=True):
        st.subheader(f"Events on {selected_date}")
        summary_cols = ["DATETIME", "AREA", "PROVINCE", "MAGNITUDE", "CATEGORY", "DEPTH (KM)"]
        st.dataframe(sorted_quakes[summary_cols], use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        start_animation = st.button("Start Animation")
    with col2:
        stop_animation = st.button("Stop Animation")

    # Initialize animation state
    if "animation_running" not in st.session_state:
        st.session_state.animation_running = False

    # Update animation state based on button clicks
    if start_animation:
        st.session_state.animation_running = True
    if stop_animation:
        st.session_state.animation_running = False

    # ------------------------------------------------
    # Advanced Seismic Animation Functions
    # ------------------------------------------------
    def calculate_shockwave_parameters(magnitude, depth):
        """Calculate parameters for the shockwave based on earthquake properties"""
        # Base intensity affects the strength of the visual effect
        base_intensity = np.clip(magnitude / 10, 0.1, 1.0)

        # Depth affects how the waves propagate (deeper = slower, more spread out)
        depth_factor = 1.0 - min(depth, 100) / 150  # Normalize depth effect

        # Calculate wave speed based on magnitude and depth
        wave_speed = shockwave_speed * (0.5 + base_intensity) * (0.2 + depth_factor)

        # Number of visible ripples depends on magnitude
        ripple_count = min(int(magnitude), max_ripples)

        # Initial burst size is larger for bigger earthquakes
        burst_multiplier = initial_burst_size * (0.5 + base_intensity * 1.5)

        # Bursts particles count scaled by magnitude but limited by user setting
        particle_count = min(int(magnitude * 2), burst_particles)

        return {
            "intensity": base_intensity,
            "depth_factor": depth_factor,
            "wave_speed": wave_speed,
            "ripple_count": ripple_count,
            "burst_multiplier": burst_multiplier,
            "particle_count": particle_count
        }

    def generate_ripple_layers(quake_data, animation_time, params):
        """Generate multiple ripple layers with realistic wave physics"""
        layers = []

        # Extract earthquake properties
        lat = quake_data['LATITUDE']
        lon = quake_data['LONGITUDE']
        magnitude = quake_data['MAGNITUDE']
        depth = quake_data['DEPTH (KM)']
        category = quake_data['CATEGORY']

        # Get base color for this earthquake's intensity category
        base_color = intensity_base_colors.get(category, (128, 128, 128, 200))
        color_gradient = intensity_color_gradients.get(category, create_color_gradient((128, 128, 128, 200)))

        # Calculate time-dependent factors
        animation_progress = min(animation_time / event_duration, 1.0)

        # ------------------------------------------------
        # Initial Burst Effect (replaces needle/column)
        # ------------------------------------------------
        burst_phase = min(animation_time * 2, 1.0)  # Quick phase in
        burst_opacity = max(0, 1 - (animation_time / (event_duration * 0.3)))  # Quick fade out

        if burst_phase > 0 and burst_opacity > 0 and use_burst_effect:
            # Create expanding burst effect
            burst_pulse = 1 + 0.5 * np.sin(animation_time * 10)
            burst_radius = magnitude * base_radius * params["burst_multiplier"] * burst_pulse * burst_phase

            # More intense color for the burst
            burst_color = list(base_color)
            burst_color[3] = int(burst_color[3] * burst_opacity)  # Adjust opacity

            # Create concentric rings for the burst effect
            for i in range(3):  # Create multiple rings for the initial burst
                ring_scale = 0.4 + (i * 0.3)  # Each ring gets progressively larger
                ring_opacity = burst_opacity * (1.0 - (i * 0.2))  # Each ring gets slightly more transparent

                # Initial burst layer with concentric rings
                burst_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{
                        "position": [lon, lat],
                        "radius": burst_radius * ring_scale,
                        "color": [burst_color[0], burst_color[1], burst_color[2], int(burst_color[3] * ring_opacity)]
                    }],
                    get_position="position",
                    get_radius="radius",
                    get_fill_color="color",
                    opacity=ring_opacity,
                    pickable=False,
                    stroked=True,
                    filled=i == 0,  # Only fill the innermost ring
                    lineWidthMinPixels=3 - i,  # Thicker lines for inner rings
                    get_line_color=[255, 255, 255, int(200 * ring_opacity)]
                )
                layers.append(burst_layer)

            # Add a bright epicenter flash that fades quickly
            if animation_time < 2.0:
                flash_opacity = max(0, 1 - (animation_time / 1.0))
                flash_radius = magnitude * base_radius * 0.4 * (1 - animation_time/2.0) * burst_intensity

                flash_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{
                        "position": [lon, lat],
                        "radius": flash_radius,
                        "color": [255, 255, 255, int(255 * flash_opacity)]
                    }],
                    get_position="position",
                    get_radius="radius",
                    get_fill_color="color",
                    opacity=flash_opacity * epicenter_glow,
                    pickable=False,
                    stroked=False,
                    filled=True
                )
                layers.append(flash_layer)

            # Add particles that "explode" outward from the epicenter - simulating debris or energy
            if use_burst_effect and animation_time < 3.0 and params["particle_count"] > 0:
                particle_data = []

                # Create multiple particles originating from the epicenter
                for i in range(params["particle_count"]):
                    # Calculate particle position based on random angle and distance from epicenter
                    angle = (i / params["particle_count"]) * 2 * np.pi
                    distance = animation_time * magnitude * 10000 * burst_intensity

                    # Add some randomness to the particle motion
                    angle_jitter = np.random.uniform(-0.2, 0.2)
                    distance_jitter = np.random.uniform(0.7, 1.3)

                    # Calculate particle position
                    particle_lon = lon + np.cos(angle + angle_jitter) * distance * distance_jitter / 111000
                    particle_lat = lat + np.sin(angle + angle_jitter) * distance * distance_jitter / 111000

                    # Particle size and opacity decrease with time
                    particle_size = magnitude * 1000 * burst_intensity * (1 - animation_time/3.0)
                    particle_opacity = max(0, 1 - (animation_time / 2.0))

                    # Use a brightened version of the base color
                    particle_color = list(base_color)
                    particle_color[0] = min(255, particle_color[0] + 50)
                    particle_color[1] = min(255, particle_color[1] + 50)
                    particle_color[2] = min(255, particle_color[2] + 50)
                    particle_color[3] = int(200 * particle_opacity)

                    particle_data.append({
                        "position": [particle_lon, particle_lat],
                        "radius": particle_size,
                        "color": particle_color
                    })

                # Create particle layer
                if particle_data:
                    particle_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=particle_data,
                        get_position="position",
                        get_radius="radius",
                        get_fill_color="color",
                        opacity=0.7,
                        pickable=False,
                        stroked=False,
                        filled=True
                    )
                    layers.append(particle_layer)

        # ------------------------------------------------
        # Multiple Expanding Ripple Rings
        # ------------------------------------------------
        for i in range(params["ripple_count"]):
            # Each ripple has a different start time and speed
            ripple_delay = i * 0.5  # stagger the start times
            ripple_time = max(0, animation_time - ripple_delay)

            if ripple_time <= 0:
                continue  # Skip ripples that haven't started yet

            # Wave propagation physics - ripples expand outward with decreasing opacity
            ripple_progress = ripple_time / event_duration
            ripple_speed_factor = params["wave_speed"] * (1 + i * 0.2)  # Each successive ring moves slightly faster
            ripple_expansion = ripple_progress * ripple_speed_factor

            # Depth affects wave propagation - deeper earthquakes have more gradual expansion
            if show_depth_effect:
                depth_adjustment = 1 - (depth / 200)  # Normalize depth to 0-1 range
                ripple_expansion *= (0.7 + 0.3 * depth_adjustment)

            # Ripple radius calculation with oscillation for wave-like effect
            wave_oscillation = 1 + pulse_amplitude * np.sin(ripple_time * pulse_frequency * np.pi * 2)
            ripple_radius = magnitude * base_radius * ripple_expansion * wave_oscillation

            # Opacity fades as the ripple expands
            max_ripple_life = 0.8  # Ripples fade out after reaching 80% of their life
            ripple_opacity = max(0, 1 - (ripple_progress / max_ripple_life))

            # Get color from the gradient based on ripple index
            color_index = min(i, len(color_gradient) - 1)
            ripple_color = list(color_gradient[color_index])
            ripple_color[3] = int(ripple_color[3] * ripple_opacity)

            # Dynamic line width for better visibility of expanding rings
            line_width = max(1, 3 - ripple_progress * 2)

            # Create ripple layer
            ripple_layer = pdk.Layer(
                "ScatterplotLayer",
                data=[{
                    "position": [lon, lat],
                    "radius": ripple_radius,
                    "color": ripple_color,
                    "line_width": line_width
                }],
                get_position="position",
                get_radius="radius",
                get_fill_color="color",
                getFillColor="color",
                get_line_color=[255, 255, 255, int(100 * ripple_opacity)],
                get_line_width="line_width",
                opacity=ripple_opacity * (1 - motion_blur * 0.3),  # Apply motion blur effect
                pickable=False,
                stroked=True,
                filled=False,
                lineWidthMinPixels=1
            )
            layers.append(ripple_layer)

            # Add motion blur effect - create a slightly larger, more transparent ring
            if motion_blur > 0:
                blur_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{
                        "position": [lon, lat],
                        "radius": ripple_radius * (1 + 0.05 * motion_blur),
                        "color": [ripple_color[0], ripple_color[1], ripple_color[2], int(ripple_color[3] * 0.5)]
                    }],
                    get_position="position",
                    get_radius="radius",
                    get_fill_color="color",
                    opacity=ripple_opacity * motion_blur * 0.5,
                    pickable=False,
                    stroked=False,
                    filled=True
                )
                layers.append(blur_layer)

        # ------------------------------------------------
        # Epicenter Marker (Alternative to "needle")
        # ------------------------------------------------
        # Only show the traditional needle/column if burst effect is disabled
        if not use_burst_effect:
            epicenter_radius = magnitude * 2000 * (1 + 0.3 * np.sin(animation_time * 3))
            epicenter_elevation = magnitude * 20000 * (1 + 0.2 * np.sin(animation_time * 2))

            epicenter_layer = pdk.Layer(
                "ColumnLayer",
                data=[{
                    "position": [lon, lat],
                    "elevation": epicenter_elevation,
                    "color": base_color
                }],
                get_position="position",
                get_elevation="elevation",
                get_fill_color="color",
                radius=epicenter_radius,
                pickable=True,
                auto_highlight=True,
                elevationScale=1,
                extruded=True
            )
            layers.append(epicenter_layer)

        # Add a persistent glowing epicenter (always present regardless of needle/burst choice)
        glow_radius = magnitude * 3000 * (1 + 0.2 * np.sin(animation_time * 4))
        glow_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[{
                "position": [lon, lat],
                "radius": glow_radius,
                "color": [base_color[0], base_color[1], base_color[2], int(100 * epicenter_glow)]
            }],
            get_position="position",
            get_radius="radius",
            get_fill_color="color",
            opacity=0.7 * epicenter_glow,
            pickable=False,
            stroked=False,
            filled=True
        )
        layers.append(glow_layer)

        return layers

    # ------------------------------------------------
    # NEW: Transition Arrow Generation
    # ------------------------------------------------
    def generate_transition_arrow(source, target, progress):
        """Generate animated arrow layer connecting source and target points"""
        source_lon, source_lat = source['LONGITUDE'], source['LATITUDE']
        target_lon, target_lat = target['LONGITUDE'], target['LATITUDE']

        # Calculate the direction vector
        direction_lon = target_lon - source_lon
        direction_lat = target_lat - source_lat

        # Calculate the distance
        distance = np.sqrt(direction_lon**2 + direction_lat**2)

        # Calculate the unit direction vector
        if distance > 0:
            unit_lon = direction_lon / distance
            unit_lat = direction_lat / distance
        else:
            # Handle case when source and target are the same
            return []

        # Create points along the path for drawing the arrow
        num_segments = 30
        arrow_points = []

        # Easing function for smooth arrow growth
        # Use a cubic easing function: progress^3 for beginning slow, accelerate in middle
        eased_progress = progress**3 if progress < 0.5 else 1 - (1-progress)**3

        # Calculate how much of the path to draw based on progress
        path_length = distance * eased_progress

        # Create segments along the path up to the current progress point
        for i in range(num_segments):
            segment_progress = i / (num_segments - 1)
            if segment_progress <= eased_progress:
                point_lon = source_lon + unit_lon * distance * segment_progress
                point_lat = source_lat + unit_lat * distance * segment_progress
                arrow_points.append({
                    "position": [point_lon, point_lat],
                    "color": arrow_color
                })

        if not arrow_points:
            return []

        # Main path line layer
        arrow_line_layer = pdk.Layer(
            "PathLayer",
            data=[{
                "path": [[p["position"][0], p["position"][1]] for p in arrow_points],
                "color": arrow_color
            }],
            get_path="path",
            get_color="color",
            width_scale=arrow_thickness * 20,
            width_min_pixels=2,
            get_width=arrow_thickness,
            rounded=True,
            joint_rounded=True,
            cap_rounded=True,
            pickable=False
        )

        # Add arrow head if progress is at least 90%
        arrow_head_layers = []
        if eased_progress > 0.9 and len(arrow_points) > 1:
            # Use the last point for the arrowhead
            tip_position = arrow_points[-1]["position"]

            # Calculate arrow head angle
            angle = np.arctan2(unit_lat, unit_lon)

            # Calculate arrow head points
            arrow_head_size_factor = arrow_head_size * 5000

            # First side of arrowhead
            side1_lon = tip_position[0] - arrow_head_size_factor * np.cos(angle + np.pi/6) / 111000
            side1_lat = tip_position[1] - arrow_head_size_factor * np.sin(angle + np.pi/6) / 111000

            # Second side of arrowhead
            side2_lon = tip_position[0] - arrow_head_size_factor * np.cos(angle - np.pi/6) / 111000

            side2_lat = tip_position[1] - arrow_head_size_factor * np.sin(angle - np.pi/6) / 111000

            # Create arrow head as a polygon
            arrow_head_layer = pdk.Layer(
                "PolygonLayer",
                data=[{
                    "polygon": [[
                        tip_position, 
                        [side1_lon, side1_lat], 
                        [side2_lon, side2_lat]
                    ]],
                    "color": arrow_color
                }],
                get_polygon="polygon",
                get_fill_color="color",
                get_line_color=[0, 0, 0, 0],  # No border
                get_line_width=0,
                opacity=0.9,
                pickable=False,
                stroked=False,
                filled=True
            )
            arrow_head_layers.append(arrow_head_layer)

        # Combine path and arrow head if it exists
        layers = [arrow_line_layer] + arrow_head_layers

        return layers

    # ------------------------------------------------
    # Dynamic Camera Configuration
    # ------------------------------------------------
    def get_view_state(quake_data, animation_time=0, target_quake=None, camera_transition_progress=0):
        """Configure the camera view based on selected settings"""
        # For simultaneous animation or static view, use the center of all quakes
        if isinstance(quake_data, pd.DataFrame) and len(quake_data) > 0 and target_quake is None:
            # Calculate the center point of all earthquakes
            center_lat = quake_data['LATITUDE'].mean()
            center_lon = quake_data['LONGITUDE'].mean()

            # Calculate appropriate zoom level based on the spread of earthquakes
            lat_range = quake_data['LATITUDE'].max() - quake_data['LATITUDE'].min()
            lon_range = quake_data['LONGITUDE'].max() - quake_data['LONGITUDE'].min()

            # Adjust zoom based on the geographic spread (larger spread = lower zoom)
            max_range = max(lat_range, lon_range)
            if max_range < 0.5:  # Small area
                zoom_level = 8
            elif max_range < 2:  # Medium area
                zoom_level = 7
            elif max_range < 5:  # Large area
                zoom_level = 6
            else:  # Very large area
                zoom_level = 5
        else:
            # Single quake mode or specific target quake (for sequential animation)
            if target_quake is not None:
                center_lat = target_quake['LATITUDE']
                center_lon = target_quake['LONGITUDE']
                magnitude = target_quake['MAGNITUDE']
            else:
                center_lat = quake_data['LATITUDE']
                center_lon = quake_data['LONGITUDE']
                magnitude = quake_data['MAGNITUDE']

            zoom_level = max(4, 9 - magnitude * 0.5)

        # For sequential animation with camera transition between two earthquakes
        if target_quake is not None and camera_transition_progress > 0:
            # Get previous earthquake position (source of transition)
            source_lat = quake_data['LATITUDE']
            source_lon = quake_data['LONGITUDE']

            # Calculate interpolated position based on transition progress
            # Using easing function for smooth movement
            t = camera_transition_progress
            # Use cubic bezier easing: smooth start and end
            eased_t = 3 * t**2 - 2 * t**3

            # Interpolate between source and target
            center_lat = source_lat + (target_quake['LATITUDE'] - source_lat) * eased_t
            center_lon = source_lon + (target_quake['LONGITUDE'] - source_lon) * eased_t

            # Optionally adjust zoom during transition
            # Zoom out slightly in the middle of the transition for better context
            mid_transition_zoom_adjust = 0.7 * np.sin(eased_t * np.pi)
            zoom_level = zoom_level - mid_transition_zoom_adjust

        if selected_camera == "Top-down":
            return pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom_level,
                pitch=0,
                bearing=0
            )
        elif selected_camera == "Tilted View":
            return pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom_level,
                pitch=45,
                bearing=0
            )
        else:  # Dynamic Camera
            # Camera rotates slowly around the center
            camera_bearing = animation_time * 15  # degrees per second

            # Camera pitches up and down slightly for dramatic effect
            camera_pitch = 45 + 15 * np.sin(animation_time * 0.3)

            # Camera "breathes" in and out slightly
            camera_zoom = zoom_level + 0.2 * np.sin(animation_time * 0.5)

            return pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=camera_zoom,
                pitch=camera_pitch,
                bearing=camera_bearing
            )

    # ------------------------------------------------
    # Animation Loop Implementation
    # ------------------------------------------------
    if st.session_state.animation_running:
        # Display info about the animation
        event_info = st.empty()

        # Choose animation mode based on user selection
        if animation_mode == "Sequential (One by One)" and len(sorted_quakes) > 1:
            event_info.info(f"Visualizing {len(sorted_quakes)} earthquake events sequentially")

            # Total animation duration calculation
            total_duration = 0
            for i in range(len(sorted_quakes)):
                # Each event has its own duration
                total_duration += event_duration
                # Add transition time if not the last event
                if i < len(sorted_quakes) - 1:
                    total_duration += transition_duration
                # Add pause time if not the last event
                if i < len(sorted_quakes) - 1:
                    total_duration += pause_duration

            # Animation loop for sequential events
            current_time = 0
            stop_requested = False

            for current_event_idx in range(len(sorted_quakes)):
                if stop_requested or not st.session_state.animation_running:
                    break

                # Get current earthquake data
                current_quake = sorted_quakes.iloc[current_event_idx]

                # Display which event is currently being animated
                event_info.info(f"Event {current_event_idx + 1} of {len(sorted_quakes)}: {current_quake['AREA']}, {current_quake['PROVINCE']} - Magnitude {current_quake['MAGNITUDE']}")

                # Get the next earthquake for transition (if not the last one)
                next_quake = None
                if current_event_idx < len(sorted_quakes) - 1:
                    next_quake = sorted_quakes.iloc[current_event_idx + 1]

                # 1. Animate the current earthquake ripples
                quake_parameters = calculate_shockwave_parameters(
                    current_quake['MAGNITUDE'], 
                    current_quake['DEPTH (KM)']
                )

                # Animation steps for current earthquake
                ripple_steps = int(event_duration / 0.1)
                for step in range(ripple_steps):
                    if not st.session_state.animation_running:
                        stop_requested = True
                        break

                    animation_time = step * 0.1  # Local animation time for this earthquake
                    current_time += 0.1  # Global animation time

                    # Generate ripple layers for the current earthquake
                    all_layers = generate_ripple_layers(
                        current_quake, 
                        animation_time, 
                        quake_parameters
                    )

                    # Configure camera view focused on current earthquake
                    view_state = get_view_state(current_quake, animation_time)

                    # Render the visualization
                    deck = pdk.Deck(
                        map_style="mapbox://styles/mapbox/dark-v10",
                        initial_view_state=view_state,
                        layers=all_layers,
                        tooltip={
                            "html": "<b>Location:</b> {AREA}, {PROVINCE}<br>"
                                    "<b>Magnitude:</b> {MAGNITUDE}<br>"
                                    "<b>Category:</b> {CATEGORY}<br>"
                                    "<b>Depth:</b> {DEPTH (KM)} km<br>"
                                    "<b>Date & Time:</b> {DATETIME}",
                            "style": {"color": "white"}
                        }
                    )

                    map_container.pydeck_chart(deck, use_container_width=True)
                    time.sleep(0.1)

                # After ripple animation is complete, add pause if configured
                if pause_duration > 0 and next_quake is not None and not stop_requested:
                    pause_steps = int(pause_duration / 0.1)
                    for _ in range(pause_steps):
                        if not st.session_state.animation_running:
                            stop_requested = True
                            break
                        current_time += 0.1
                        time.sleep(0.1)

                # 2. Animate transition to next event with arrow if not the last one
                if next_quake is not None and not stop_requested:
                    # Transition animation steps
                    transition_steps = int(transition_duration / 0.1)

                    for step in range(transition_steps):
                        if not st.session_state.animation_running:
                            stop_requested = True
                            break

                        # Calculate progress through transition (0 to 1)
                        transition_progress = step / max(1, transition_steps - 1)
                        current_time += 0.1

                        # Generate the transition arrow with animation
                        arrow_layers = generate_transition_arrow(
                            current_quake,
                            next_quake,
                            transition_progress
                        )

                        # Continue showing a fading version of the last ripple
                        fade_opacity = max(0, 1 - transition_progress)
                        last_ripple_layers = []

                        if fade_opacity > 0:
                            # Show fading version of last earthquake's ripples
                            ripple_layers = generate_ripple_layers(
                                current_quake, 
                                event_duration, 
                                quake_parameters
                            )

                            # Reduce opacity of all ripple layers
                            for layer in ripple_layers:
                                layer.opacity = layer.opacity * fade_opacity if hasattr(layer, 'opacity') else fade_opacity
                                last_ripple_layers.append(layer)

                        # Combine all layers
                        all_layers = last_ripple_layers + arrow_layers

                        # Smoothly transition camera to next earthquake
                        view_state = get_view_state(
                            current_quake, 
                            0, 
                            next_quake, 
                            transition_progress
                        )

                        # Render the visualization
                        deck = pdk.Deck(
                            map_style="mapbox://styles/mapbox/dark-v10",
                            initial_view_state=view_state,
                            layers=all_layers,
                            tooltip={
                                "html": "<b>Transitioning to:</b> {AREA}, {PROVINCE}<br>"
                                        "<b>Magnitude:</b> {MAGNITUDE}<br>"
                                        "<b>Category:</b> {CATEGORY}<br>"
                                        "<b>Date & Time:</b> {DATETIME}",
                                "style": {"color": "white"}
                            }
                        )

                        map_container.pydeck_chart(deck, use_container_width=True)
                        time.sleep(0.1)

            # Animation complete
            if not stop_requested:
                event_info.success(f"Sequential animation complete for {len(sorted_quakes)} earthquake events")
            else:
                event_info.warning("Animation stopped")

        # Simultaneous animation mode (original implementation)
        else:
            # Show fallback message if using sequential mode with just one event
            if animation_mode == "Sequential (One by One)" and len(sorted_quakes) == 1:
                event_info.info("Only one earthquake detected - using simultaneous animation mode")
            else:
                event_info.info(f"Visualizing {len(sorted_quakes)} earthquake events simultaneously")

            # Animation timing
            animation_steps = int(event_duration / 0.1)

            for step in range(animation_steps):
                if not st.session_state.animation_running:
                    break

                animation_time = step * 0.1  # global animation time

                # Initialize an empty list to hold all layers from all earthquakes
                all_layers = []

                # Process each earthquake and generate its layers
                for _, quake in sorted_quakes.iterrows():
                    # Calculate parameters for this earthquake
                    quake_parameters = calculate_shockwave_parameters(
                        quake['MAGNITUDE'], 
                        quake['DEPTH (KM)']
                    )

                    # Generate ripple layers for this earthquake
                    quake_layers = generate_ripple_layers(
                        quake, 
                        animation_time, 
                        quake_parameters
                    )

                    # Add these layers to the combined list
                    all_layers.extend(quake_layers)

                # Configure the camera to show all earthquakes
                view_state = get_view_state(sorted_quakes, animation_time)

                # Render the visualization with all earthquake layers
                deck = pdk.Deck(
                    map_style="mapbox://styles/mapbox/dark-v10",
                    initial_view_state=view_state,
                    layers=all_layers,
                    tooltip={
                        "html": "<b>Location:</b> {AREA}, {PROVINCE}<br>"
                                "<b>Magnitude:</b> {MAGNITUDE}<br>"
                                "<b>Category:</b> {CATEGORY}<br>"
                                "<b>Depth:</b> {DEPTH (KM)} km<br>"
                                "<b>Date & Time:</b> {DATETIME}",
                        "style": {"color": "white"}
                    }
                )

                map_container.pydeck_chart(deck, use_container_width=True)
                time.sleep(0.1)

            # Animation complete
            event_info.success(f"Animation complete for {len(sorted_quakes)} earthquake events")

        # Reset animation state
        st.session_state.animation_running = False

    else:
        # Display a static map when not animating
        if len(sorted_quakes) > 0:
            # Use the center of all earthquakes for the static view
            static_view = get_view_state(sorted_quakes)

            # Create enhanced static visualization with multiple layers

            # 1. Base layer showing all quakes with size based on magnitude
            static_base_layer = pdk.Layer(
                "ScatterplotLayer",
                data=sorted_quakes,
                get_position=["LONGITUDE", "LATITUDE"],
                get_radius="MAGNITUDE * 5000",
                get_fill_color="COLOR",
                pickable=True,
                opacity=0.6,
                stroked=True,
                filled=True,
                get_line_color=[255, 255, 255],  # Fixed typo: removed duplicate 255
                line_width_min_pixels=1
            )

            # 2. Add a glow effect layer for visual enhancement
            static_glow_layer = pdk.Layer(
                "ScatterplotLayer",
                data=sorted_quakes,
                get_position=["LONGITUDE", "LATITUDE"],
                get_radius="MAGNITUDE * 8000",  # Larger radius for glow effect
                get_fill_color=[255, 255, 255, 50],  # White with low opacity for glow
                pickable=False,
                opacity=0.3,
                stroked=False,
                filled=True
            )

            # 3. Add text labels for significant earthquakes
            significant_quakes = sorted_quakes[sorted_quakes['MAGNITUDE'] >= 4.5].copy()
            if not significant_quakes.empty:
                significant_quakes['text'] = significant_quakes.apply(
                    lambda row: f"M{row['MAGNITUDE']}", axis=1
                )

                text_layer = pdk.Layer(
                    "TextLayer",
                    data=significant_quakes,
                    get_position=["LONGITUDE", "LATITUDE"],
                    get_text="text",
                    get_size=16,
                    get_color=[255, 255, 255],
                    get_angle=0,
                    text_anchor="middle",
                    text_baseline="center",
                    pickable=True
                )
            else:
                text_layer = None

            # 4. Add a heatmap layer to show concentration of events
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=sorted_quakes,
                get_position=["LONGITUDE", "LATITUDE"],
                get_weight="MAGNITUDE",
                aggregation="SUM",
                threshold=0.05,
                radiusPixels=60,
                intensity=1,
                opacity=0.4
            )

            # 5. Add chronological connections if in sequential mode
            if animation_mode == "Sequential (One by One)" and len(sorted_quakes) > 1:
                # Create path data for lines connecting events in chronological order
                path_data = []
                for i in range(len(sorted_quakes) - 1):
                    path_data.append({
                        "path": [
                            [sorted_quakes.iloc[i]['LONGITUDE'], sorted_quakes.iloc[i]['LATITUDE']],
                            [sorted_quakes.iloc[i+1]['LONGITUDE'], sorted_quakes.iloc[i+1]['LATITUDE']]
                        ],
                        "color": arrow_color
                    })

                # Static path layer showing chronological connections
                path_layer = pdk.Layer(
                    "PathLayer",
                    data=path_data,
                    get_path="path",
                    get_color="color",
                    width_scale=arrow_thickness * 10,
                    width_min_pixels=1,
                    get_width=arrow_thickness / 2,  # Thinner than animated version
                    rounded=True,
                    joint_rounded=True,
                    cap_rounded=True,
                    pickable=False,
                    opacity=0.4
                )

                # Add sequentially numbered markers
                marker_data = []
                for i, (_, quake) in enumerate(sorted_quakes.iterrows()):
                    marker_data.append({
                        "position": [quake['LONGITUDE'], quake['LATITUDE']],
                        "text": str(i + 1),
                        "color": [255, 255, 255, 200]
                    })

                marker_layer = pdk.Layer(
                    "TextLayer",
                    data=marker_data,
                    get_position="position",
                    get_text="text",
                    get_color="color",
                    get_size=14,
                    get_angle=0,
                    text_anchor="middle",
                    text_baseline="center",
                    background_color=[0, 0, 0, 100],
                    background_padding=3,
                    pickable=False
                )
            else:
                path_layer = None
                marker_layer = None

            # Combine all layers
            static_layers = [heatmap_layer, static_glow_layer, static_base_layer]
            if text_layer:
                static_layers.append(text_layer)
            if path_layer:
                static_layers.append(path_layer)
            if marker_layer:
                static_layers.append(marker_layer)

            # Create and display the static deck
            static_deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=static_view,
                layers=static_layers,
                tooltip={
                    "html": "<b>Location:</b> {AREA}, {PROVINCE}<br>"
                            "<b>Magnitude:</b> {MAGNITUDE}<br>"
                            "<b>Category:</b> {CATEGORY}<br>"
                            "<b>Depth:</b> {DEPTH (KM)} km<br>"
                            "<b>Date & Time:</b> {DATETIME}",
                    "style": {"color": "white"}
                }
            )

            map_container.pydeck_chart(static_deck, use_container_width=True)

            # Display a message about starting the animation
            if animation_mode == "Sequential (One by One)":
                st.info("Click 'Start Animation' to visualize each earthquake sequentially with transition arrows between events.")
            else:
                st.info("Click 'Start Animation' to visualize seismic waves for all earthquake events simultaneously.")
        else:
            st.warning("No earthquake data available for the selected filters.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
