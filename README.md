# 🌾 Crop Yield Prediction & Agricultural Decision Support System

## Project Overview

Crop Yield Prediction is a machine learning-powered agricultural decision support application designed to help farmers and agricultural stakeholders make more informed crop production decisions.

The project combines data analysis, machine learning and agricultural information to estimate crop yields based on selected environmental conditions and provide useful crop management guidance.

## Problem Statement

Agricultural productivity is influenced by several environmental factors, including rainfall, temperature and crop management practices. Understanding how these factors relate to crop yield can support better agricultural planning and decision-making.

This project explores the use of historical agricultural data and machine learning to generate crop yield estimates and provide additional agricultural guidance.

## Project Objectives

- Analyse agricultural data to understand factors influencing crop yield.
- Develop a machine learning model to predict crop yield.
- Build an interactive application for generating yield predictions.
- Provide crop-specific pest management information.
- Present planting calendars and yield reference information to support agricultural decisions.

## Key Features

### 1. Crop Yield Prediction
Users can select a crop and enter:
- Average rainfall (mm/year)
- Estimated pesticide usage (tonnes)
- Average temperature (°C)

The application processes the inputs and generates an estimated crop yield.

### 2. Crop Pest Guide
Provides information about common crop pests, symptoms, crop damage, prevention and management practices.

### 3. Planting Calendar
Provides planting-period guidance to support agricultural planning.

### 4. Yield Reference Guide
Displays crop-specific yield reference ranges and categorises predictions into low, average and high yield levels.

## Data Analysis and Machine Learning

The project involved working with agricultural datasets, preparing data for modelling and developing a predictive machine learning solution.

The trained model and preprocessing pipeline are saved and integrated into the Streamlit application for prediction.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Git and GitHub

## Project Structure

```text
Crop_Yield_Prediction/
│
├── Crop_Yield_Prediction.ipynb
├── app.py
├── crop_yield_model.pkl
├── preprocessor.pkl
├── yield_df.csv
├── yield_df_clean.csv
├── .gitattributes
└── README.md
```

## How to Run the Application Locally

1. Clone the repository:

```bash
git clone https://github.com/StephanieOlisa/crop-yield-prediction.git
```

2. Navigate into the project folder:

```bash
cd crop-yield-prediction
```

3. Install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

5. Open the local URL displayed in your terminal to access the application.

## Application Screenshots

# Crop Yield Prediction

<img width="959" height="445" alt="yield_prediction" src="https://github.com/user-attachments/assets/44bb4fc9-27f6-4184-8ccc-0ef0ba6a349e" />

# Crop Pest Guide

<img width="959" height="440" alt="crop_pest_guide" src="https://github.com/user-attachments/assets/f4a369ac-d5c3-4cde-b4b5-34d27adfb037" />

# Planting Calendar

<img width="959" height="442" alt="planting_calendar" src="https://github.com/user-attachments/assets/97d8adb9-9e52-4c25-a8e2-53cadb302245" />

# Yield Reference Guide

<img width="959" height="440" alt="yield_reference" src="https://github.com/user-attachments/assets/00ccb8be-55c7-41b0-84d0-e1b5f1bdb0c3" />


## Project Limitations

- Predictions are estimates and are not guaranteed actual harvest outcomes.
- Accuracy depends on the quality and coverage of the training data.
- Actual crop yield is also influenced by factors such as soil fertility, irrigation, crop variety, pests and farm management.
- Planting and pest management information should be validated against local agricultural conditions.

## Future Improvements

- Incorporate additional environmental variables such as soil properties and water availability.
- Improve model performance through further experimentation and evaluation.
- Expand crop coverage and location-specific agricultural recommendations.
- Deploy the application online for wider accessibility.

## Author

**Stephanie Olisa**

Data Analytics | Data Science | Agricultural Technology

[GitHub Repository](https://github.com/StephanieOlisa/crop-yield-prediction)
