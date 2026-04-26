# Airbnb Price Prediction Project Report

## 1. Project Title
Airbnb Rental Price Prediction

## 2. Business Task
The objective of this project is to develop a regression model to predict the rental price of Airbnb listings. By analyzing property features, host profiles, and market dynamics, the model aims to provide accurate pricing estimates to help hosts optimize their revenue and guests understand market value.

## 3. Dataset Overview
- **Domain:** Short-term real estate rental market (Airbnb).
- **Total Rows:** 15,000
- **Total Columns:** 36
- **Row Meaning:** Each row represents a unique rental listing (apartment, house, or room).

## 4. Target Column and Task Type
- **Target Column:** `price`
- **Task Type:** Regression

## 5. Feature Groups
- **Numeric:** accommodates, bathrooms, bedrooms, beds, host_listings_count, host_total_listings_count, latitude, longitude, availability_30, availability_60, availability_90, availability_365, number_of_reviews, number_of_reviews_ltm, number_of_reviews_l30d, review_scores_rating, review_scores_cleanliness, review_scores_location, review_scores_value, reviews_per_month.
- **Categorical:** property_type, room_type, host_location, host_response_time, host_response_rate, host_acceptance_rate, city.
- **Text:** description, amenities.
- **Datetime:** host_since, first_review, last_review.
- **Boolean-like:** host_is_superhost, has_availability.
- **ID:** id.

## 6. Data Quality Summary
- **Missing Values:** 31,176 total missing values across the dataset.
- **Duplicate Rows:** 0
- **Data Preparation Status:** Success. All missing values were handled during the preparation stage.

## 7. Data Preparation Steps
1. **Feature Type Casting:** Ensuring columns are in the correct format (numeric, categorical, etc.).
2. **Feature Engineering:** Creating new derived features to capture complex relationships.
3. **Column Cleaning:** Removing or fixing inconsistent data.
4. **Missing Value Imputation:** Handling null values to ensure model compatibility.
5. **Encoding and Scaling:** Transforming categorical variables and normalizing numeric features.
6. **Final Preparation:** Finalizing the dataset for training (Final shape: 205 rows, 66 features).

## 8. Created Features
- **beds_per_bedroom:** Indicates density of sleeping arrangements relative to rooms.
- **amenities_count:** Total number of amenities offered.
- **host_tenure_days:** Duration the host has been on the platform.
- **availability_30_ratio:** Normalized short-term availability indicator.
- **review_span_days:** Duration the listing has been active and receiving reviews.
- **description_length:** Length of the property description.

## 9. Models Tested
- Ridge Regression
- RandomForestRegressor

## 10. Best Model
- **Model Name:** RandomForestRegressor

## 11. Best Metrics
- **MAE (Mean Absolute Error):** 80.24
- **RMSE (Root Mean Squared Error):** 128.72
- **R2 Score:** 0.5942

## 12. Hyperparameter Tuning Result
Tuning was performed on the RandomForestRegressor.
- **Best Parameters:** `max_depth: null`, `min_samples_split: 5`, `n_estimators: 100`.
- **Improvement:** R2 score improved from 0.5514 (baseline) to 0.5942 (tuned).

## 13. Business Interpretation
The RandomForestRegressor model explains approximately 59.4% of the variance in Airbnb prices. With a Mean Absolute Error of $80.24, the model provides a solid foundation for automated pricing suggestions. The inclusion of engineered features like `amenities_count` and `beds_per_bedroom` suggests that property utility and service offerings are significant drivers of price.

## 14. Final Artifacts
- **Best Model:** `artifacts/modeling/best_model.joblib`
- **Prepared Dataset:** `artifacts/data_preparation/prepared_dataset.csv`
- **EDA Summary:** `artifacts/data_description/eda_artifacts.json`
- **Final Metrics:** `artifacts/modeling/final_metrics.json`
