# Polish Apartment Price Prediction

A machine learning web application that predicts apartment prices in Poland using May-June 2024 data. Built with scikit-learn and deployed via Flask on AWS Elastic Beanstalk.

## Live Demo
🌐 [Polish Apartment Price Predictor](http://polish-apartment-price-predictio-env.eba-rheknhrx.eu-north-1.elasticbeanstalk.com/predict_data)

---

## Project Overview

This project covers the full ML lifecycle — from exploratory data analysis to cloud deployment:

1. **EDA** — Data cleaning, correlation analysis, outlier detection
2. **Feature Engineering** — Encoding categorical variables, dropping correlated features
3. **Model Training** — Comparing Linear Regression, Decision Tree, and Random Forest
4. **Hyperparameter Tuning** — GridSearchCV to find optimal Random Forest parameters
5. **Pipeline** — Custom sklearn transformer wrapping all preprocessing steps
6. **Deployment** — Flask web app deployed on AWS Elastic Beanstalk

---

## Model Performance

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression | — | Low |
| Decision Tree | 0.9996 | 0.91 (overfitting) |
| **Random Forest (Final)** | **0.93** | **0.91** |

### Best Parameters (via GridSearchCV)
```
n_estimators=200, max_depth=10, max_features=0.5,
min_samples_split=10, cv=5, scoring='r2'
```

---

## Features Used

| Feature | Type |
|---|---|
| `type` | Categorical (OneHotEncoded) |
| `ownership` | Categorical (OneHotEncoded) |
| `hasParkingSpace`, `hasBalcony`, `hasElevator`, `hasSecurity`, `hasStorageRoom` | Binary (OrdinalEncoded) |
| `squareMeters`, `floorCount`, `buildYear`, `latitude`, `longitude` | Numerical |
| `centreDistance`, `poiCount`, `clinicDistance`, `restaurantDistance`, `collegeDistance` | Numerical |

Correlated features dropped: `rooms`, `floor`, `schoolDistance`, `postOfficeDistance`, `kindergartenDistance`, `pharmacyDistance`, `type_tenement`, `type_apartmentBuilding`, `ownership_cooperative`

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, Flask
- **ML** — RandomForestRegressor, GridSearchCV, Pipeline, ColumnTransformer
- **Deployment** — AWS Elastic Beanstalk, AWS CodePipeline (CI/CD)

---

## Project Structure

```
├── application.py          # Flask app
├── custom_classes.py       # Custom sklearn transformer
├── housing_model.pkl       # Serialized pipeline (preprocessing + model)
├── feature_names.json      # Valid input features for validation
├── requirements.txt
├── .ebextensions/
│   └── python.config       # Elastic Beanstalk configuration
├── templates/
│   └── home.html           # Input form and prediction result
├── ML_EDA_project.ipynb    # Exploratory Data Analysis
└── Model_pipeline.ipynb    # Model training and pipeline
```

---

## Pipeline Design

A custom `CustomePreprocess` transformer (inheriting `BaseEstimator`, `TransformerMixin`) wraps all preprocessing steps so the entire pipeline — encoding, feature dropping, and prediction — is serialized into a single `.pkl` file:

```
Raw Input → OrdinalEncoder (binary) → OneHotEncoder (categorical)
         → Drop correlated columns → RandomForestRegressor → Price
```

---

## CI/CD

Connected to GitHub via AWS CodePipeline — every push to `main` branch automatically redeploys the app on Elastic Beanstalk.

---

## Dataset

Polish apartment listings — May/June 2024. Cleaned version used for training (`Cleaned_data_Polish_housing.csv`).
