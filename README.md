# Earthquake Monitoring Dashboard

A comprehensive, interactive Streamlit application for real-time earthquake monitoring and analysis. This dashboard provides advanced visualizations, statistical analysis, and filtering capabilities for seismic activity data.

## Features

- **Interactive Visualizations**: Dynamic, animated visualizations of earthquake data with smooth transitions and professional-looking effects
- **Multiple Analysis Views**: Specialized pages for different perspectives on seismic data
- **Advanced Filtering**: Filter by region, magnitude, date, and more
- **Real-time Simulation**: Simulate real-time data updates with realistic patterns and aftershocks
- **Responsive UI**: Optimized for desktop and tablet viewing
- **Educational Content**: Informative resources about seismic activity and intensity scales

## Dashboard Pages

1. **Dashboard Overview**: High-level summary of the entire dashboard with navigation and key statistics
2. **Seismic Activity Trends**: Real-time monitoring with animated updates and trend analysis
3. **Map View**: Advanced geographic visualization with shockwave animations and 3D effects
4. **Regional Analysis**: Comparative analysis of seismic activity across different regions
5. **Intensity Scale**: Reference guide to the PHIVOLCS Earthquake Intensity Scale (PEIS)

## Installation & Running Locally

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/earthquake-monitoring-dashboard.git
   cd earthquake-monitoring-dashboard
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Run the Streamlit application:
```bash
streamlit run Home.py
```

The application will open in your default web browser at `http://localhost:5000`.

## Data Sources

The dashboard uses earthquake data stored in CSV format. The data includes:
- Earthquake date and time
- Location (coordinates and region)
- Magnitude
- Depth
- Intensity category

## Technologies Used

- **Streamlit**: Main framework for the web application
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **PyDeck**: Advanced map visualizations
- **NumPy**: Numerical processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Philippine Institute of Volcanology and Seismology (PHIVOLCS) for the intensity scale reference
- Seismic data providers for the earthquake information