from datetime import datetime, timedelta
import pandas as pd
from hsfs.client.exceptions import RestAPIError

import src.config as config
from src.inference import (
    get_feature_store,
    get_model_predictions,
    load_model_from_registry,
)
from src.data_utils import transform_ts_data_info_features

# Get current UTC time
current_date = pd.Timestamp.now(tz="Etc/UTC")
feature_store = get_feature_store()

# Define time window
fetch_data_to = current_date - timedelta(hours=1)
fetch_data_from = current_date - timedelta(days=29)
print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

# 1. Attempt Feature View retrieval — fallback to Feature Group on failure
try:
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
    )
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
    )
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    print("⚙️ Fetched data via Feature View")
except RestAPIError as e:
    print(f"⚠ Feature View unavailable, using Feature Group fallback: {e}")
    fg = feature_store.get_feature_group(
        name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION
    )
    ts_data = fg.read()
    ts_data = ts_data.query("pickup_hour >= @fetch_data_from and pickup_hour <= @fetch_data_to")
    print("ℹ Fetched data via Feature Group")

# Ensure proper sorting and formatting
ts_data = (
    ts_data.sort_values(["pickup_location_id", "pickup_hour"])
    .reset_index(drop=True)
)
ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)

# Feature transformation (sliding window)
features = transform_ts_data_info_features(
    ts_data, window_size=24 * 28, step_size=23
)

model = load_model_from_registry()
predictions = get_model_predictions(model, features)
predictions["pickup_hour"] = current_date.ceil("h")
print(predictions.head())

# Insert predictions
feature_group = feature_store.get_or_create_feature_group(
    name=config.FEATURE_GROUP_MODEL_PREDICTION,
    version=1,
    description="Predictions from LGBM Model",
    primary_key=["pickup_location_id", "pickup_hour"],
    event_time="pickup_hour",
)
feature_group.insert(predictions, write_options={"wait_for_job": False})
