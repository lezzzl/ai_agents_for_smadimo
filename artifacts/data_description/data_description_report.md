# Data Description Report

## Summary
Dataset contains 15000 rows and 36 columns. Task type: regression.

## Domain
The dataset pertains to the short-term real estate rental market, specifically Airbnb listings. It contains detailed information about property features, host profiles, geographic locations, availability, and guest reviews, with the primary target variable being the rental price.

## Row meaning
Each row represents a unique rental listing (an apartment, house, or room) on the Airbnb platform.

## Target
- Target column: price
- Task type: regression

## Dataset shape
- Rows: 15000
- Columns: 36

## Data quality summary
- Total missing values: 31176
- Duplicate rows: 0
- Constant columns count: 0
- High-cardinality columns count: 8

## Schema
```json
{
  "numeric": [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "host_listings_count",
    "host_total_listings_count",
    "latitude",
    "longitude",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "reviews_per_month"
  ],
  "categorical": [
    "property_type",
    "room_type",
    "host_location",
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "city"
  ],
  "text": [
    "description",
    "amenities"
  ],
  "datetime": [
    "host_since",
    "first_review",
    "last_review"
  ],
  "boolean_like": [
    "host_is_superhost",
    "has_availability"
  ],
  "id": [
    "id"
  ],
  "target": [
    "price"
  ]
}
```

## Artifacts
- Summary: artifacts/data_description/data_description_summary.json
- EDA artifacts: artifacts/data_description/eda_artifacts.json
- Statistics: artifacts/data_description/stages/statistics.json
- Stage artifacts dir: artifacts/data_description/stages
