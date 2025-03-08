# 🚖 sp25_taxi

Welcome to the **sp25_taxi** project! This repository contains everything you need to work with taxi data, from fetching and processing to modeling and inference. Buckle up and let's take a ride through the project! 🚕

## 📁 Project Structure

Here's a quick overview of the project's structure:

### 📂 Data

- **data/**: Contains all the data files, including raw, processed, and transformed data.
  - **taxi_zones.zip**: Compressed file with taxi zone data.
  - **processed/**: Processed data files.
  - **raw/**: Raw data files.
  - **taxi_zones/**: Taxi zone data files.
  - **transformed/**: Transformed data files.

### 🌐 Frontend

- **frontend/**: Contains the frontend code.
  - **frontend_monitor.py**: Script for monitoring the frontend.
  - **frontend_v2.py**: Version 2 of the frontend script.

### 🧠 Models

- **models/**: Contains the machine learning models.
  - **lgb_model.pkl**: Pre-trained LightGBM model.

### 📓 Notebooks

- **notebooks/**: Jupyter notebooks for data exploration and processing.
  - **01_fetch_data.ipynb**: Notebook for fetching data.
  - **02_validate_and_save.ipynb**: Notebook for validating and saving data.
  - **03_transform_processed_data_into_ts_data.ipynb**: Notebook for transforming processed data into time series data.

### 🚀 Pipelines

- **pipelines/**: Contains the pipeline scripts.
  - **inference_pipeline.py**: Script for running the inference pipeline.

### 🗂️ Source Code

- **src/**: Contains the source code for the project.

### 🧪 Tests

- **test/**: Contains the test scripts.

## 🛠️ Setup

To get started with the project, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/sp25_taxi.git
   cd sp25_taxi
   ```

2. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Run the notebooks or scripts as needed.

## 📜 License

This project is licensed under the GPLv2. See the [LICENSE](LICENSE) file for details.

## 🎉 Contributing

We welcome contributions! Feel free to open issues or submit pull requests.

## 📞 Contact

For any questions or inquiries, please open an issue in this repo.

Happy coding! 🚀
