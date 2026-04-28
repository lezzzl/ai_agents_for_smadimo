# Data Description Report

## Summary
Dataset contains 15000 rows and 36 columns. Task type: regression.

## Domain
Airbnb rental property listings with host information, availability, reviews, and pricing data.

## Row meaning
One row represents a single Airbnb property listing with its characteristics, host details, availability metrics, review scores, and nightly price.

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
- High-cardinality columns count: 5

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
    "city",
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "has_availability"
  ],
  "text": [
    "description",
    "amenities",
    "host_location"
  ],
  "datetime": [
    "host_since",
    "first_review",
    "last_review"
  ],
  "boolean_like": [
    "host_is_superhost"
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
