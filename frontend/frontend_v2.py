import sys
from pathlib import Path
import zipfile
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
import os


# Add parent directory to the Python path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)


from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Set the GDAL configuration to restore SHX files
# os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False

def create_taxi_map(shapefile_path, prediction_data, highlight_id=None):
    """
    Create an interactive choropleth map of NYC taxi zones with predicted rides.
    If highlight_id is provided, that taxi zone gets a thick black border.
    """
    nyc_zones = gpd.read_file(shapefile_path)
    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left"
    )
    nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
    nyc_zones = nyc_zones.to_crs(epsg=4326)

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
    colormap = LinearColormap(
        colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max()
    )
    colormap.add_to(m)

    def style_function(feature):
        predicted_demand = feature["properties"].get("predicted_demand", 0)
        try:
            zone_id = int(feature["properties"].get("LocationID", -1))
        except:
            zone_id = -1
        if highlight_id is not None and zone_id == highlight_id:
            return {
                "fillColor": colormap(float(predicted_demand)),
                "color": "black",
                "weight": 5,  # thicker border for the selected location
                "fillOpacity": 0.7
            }
        else:
            return {
                "fillColor": colormap(float(predicted_demand)),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.7
            }

    zones_json = nyc_zones.to_json()
    folium.GeoJson(
        zones_json,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "predicted_demand"],
            aliases=["Zone:", "Predicted Demand:"],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)

    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m

def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
    """
    Downloads, extracts, and loads a shapefile as a GeoDataFrame.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"

    if not zip_path.exists():
        if log:
            print(f"Downloading file from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(response.content)
            if log:
                print(f"File downloaded and saved to {zip_path}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {e}")
    else:
        if log:
            print(f"File already exists at {zip_path}, skipping download.")

    if not shapefile_path.exists():
        if log:
            print(f"Extracting files to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            if log:
                print(f"Files extracted to {extract_path}")
        except zipfile.BadZipFile as e:
            raise Exception(f"Failed to extract zip file {zip_path}: {e}")
    else:
        if log:
            print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

    if log:
        print(f"Loading shapefile from {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
        if log:
            print("Shapefile successfully loaded.")
        return gdf
    except Exception as e:
        raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")

# ---- Main App Code ----

# Set New York/EST time for current date and time.
current_date = pd.Timestamp.now(tz="America/New_York")
st.title("New York Yellow Taxi Cab Demand Next Hour")
st.header(current_date.strftime("%Y-%m-%d %H:%M:%S EST"))

progress_bar = st.sidebar.header("Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 4

with st.spinner("Download shape file for taxi zones"):
    geo_df = load_shape_data_file(DATA_DIR)
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1 / N_STEPS)

with st.spinner("Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write("Inference features fetched from the store")
    progress_bar.progress(2 / N_STEPS)

with st.spinner("Fetching predictions"):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Model was loaded from the registry")
    progress_bar.progress(3 / N_STEPS)

# Ensure that pickup_location_id is an integer.
predictions["pickup_location_id"] = predictions["pickup_location_id"].astype(int)

# Map taxi zone names from the shapefile to predictions.
if "LocationID" in geo_df.columns and "zone" in geo_df.columns:
    zone_mapping = geo_df.set_index("LocationID")["zone"].to_dict()
    predictions["zone_name"] = predictions["pickup_location_id"].map(zone_mapping)
    predictions["zone_name"] = predictions["zone_name"].fillna(predictions["pickup_location_id"].astype(str))
    predictions["zone_display"] = predictions["pickup_location_id"].astype(str) + " - " + predictions["zone_name"]

shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"


# Build dropdown options with "Top 10 Locations" as the default.
unique_zones = predictions[["pickup_location_id", "zone_display"]].drop_duplicates().sort_values("pickup_location_id")
dropdown_options = ["Top 10 Locations"] + unique_zones["zone_display"].tolist()

selected_option = st.sidebar.selectbox(
    "Select Taxi Zone for Detailed Prediction",
    options=dropdown_options,
    index=0  # default is "Top 10 Locations"
)

# Determine the highlight id; if a specific taxi zone is selected then parse its ID.
if selected_option == "Top 10 Locations":
    highlight_id = None
else:
    try:
        highlight_id = int(selected_option.split(" - ")[0])
    except Exception as e:
        st.error("Failed to parse the selected taxi zone.")
        highlight_id = None

# Recreate and display the map with (if applicable) the highlighted taxi zone.
st.subheader("NYC Taxi Zones Map")
map_obj = create_taxi_map(shapefile_path, predictions, highlight_id=highlight_id)
st_folium(map_obj, width=800, height=600, returned_objects=[])

# Add Top 10 Locations table
st.subheader("Top 10 Pickup Locations by Predicted Demand")
top10_df = predictions.sort_values("predicted_demand", ascending=False).head(10)
st.dataframe(top10_df[["pickup_location_id", "zone_display", "predicted_demand"]])

# Display prediction graphs based on the dropdown selection.
if selected_option == "Top 10 Locations":
    st.subheader("Prediction Details for Top 10 Locations")
    # Use the same top10_df for graphs
    for idx, row in top10_df.iterrows():
        loc_id = row["pickup_location_id"]
        display_label = row["zone_display"]
        st.markdown(f"### Taxi Zone: {display_label}")
        filtered_features = features[features["pickup_location_id"] == loc_id]
        filtered_predictions = predictions[predictions["pickup_location_id"] == loc_id]
        if not filtered_features.empty and not filtered_predictions.empty:
            fig = plot_prediction(filtered_features, filtered_predictions)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
else:
    st.subheader(f"Prediction Details for Taxi Zone: {selected_option}")
    filtered_features = features[features["pickup_location_id"] == highlight_id]
    filtered_predictions = predictions[predictions["pickup_location_id"] == highlight_id]
    if not filtered_features.empty and not filtered_predictions.empty:
        fig = plot_prediction(filtered_features, filtered_predictions)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
