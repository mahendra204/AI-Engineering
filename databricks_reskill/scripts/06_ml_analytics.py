# Script 6: Advanced Analytics & Machine Learning
# ================================================

from pyspark.sql.functions import col, when, sum as spark_sum, avg, count
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import math

print(\"=== Advanced Analytics & ML Demo ===\n\")

# 1. CREATE SAMPLE DATA (customer behavior)
print(\"Step 1: Creating sample customer data...\")

customer_data = [
    (1, 100, 5, 120),    # (id, total_spent, num_purchases, days_active)
    (2, 250, 12, 180),
    (3, 75, 2, 30),
    (4, 500, 25, 365),
    (5, 150, 8, 90),
    (6, 1000, 50, 365),
    (7, 50, 1, 15),
    (8, 300, 15, 200),
    (9, 400, 20, 250),
    (10, 600, 30, 330),
]

df_customers = spark.createDataFrame(
    customer_data,
    [\"customer_id\", \"total_spent\", \"num_purchases\", \"days_active\"]
)

print(\"Sample customer data created:\")
df_customers.show()
print()

# 2. FEATURE ENGINEERING
print(\"Step 2: Feature engineering...\")

df_features = df_customers.select(
    \"customer_id\",
    col(\"total_spent\"),
    col(\"num_purchases\"),
    col(\"days_active\"),
    (col(\"total_spent\") / col(\"days_active\")).alias(\"daily_spend\"),
    (col(\"num_purchases\") / col(\"days_active\")).alias(\"daily_purchases\"),
    (col(\"total_spent\") / col(\"num_purchases\")).alias(\"avg_purchase_value\"),
    when(col(\"total_spent\") > 300, 1).otherwise(0).alias(\"high_value_customer\")
)

print(\"Features created:\")
df_features.show()
print()

# 3. PREPARE FOR ML (assemble features)
print(\"Step 3: Preparing features for ML...\")

assembler = VectorAssembler(
    inputCols=[\"num_purchases\", \"days_active\"],
    outputCol=\"raw_features\"
)

df_assembled = assembler.transform(df_features)

print(\"Features assembled:\")
df_assembled.select(\"customer_id\", \"raw_features\", \"total_spent\").show()
print()

# 4. SCALE FEATURES
print(\"Step 4: Scaling features...\")

scaler = StandardScaler(
    inputCol=\"raw_features\",
    outputCol=\"scaled_features\",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

print(\"Features scaled:\")
df_scaled.select(\"customer_id\", \"scaled_features\", \"total_spent\").show()
print()

# 5. TRAIN MODEL (Linear Regression)
print(\"Step 5: Training Linear Regression model...\")

lr = LinearRegression(
    featuresCol=\"scaled_features\",
    labelCol=\"total_spent\",
    maxIter=10,
    regParam=0.0
)

model_lr = lr.fit(df_scaled)

print(f\"Model trained!\")
print(f\"Coefficients: {model_lr.coefficients}\")
print(f\"Intercept: {model_lr.intercept}\")
print()

# 6. MAKE PREDICTIONS
print(\"Step 6: Making predictions...\")

df_predictions = model_lr.transform(df_scaled)

print(\"Predictions vs Actual:\")
df_predictions.select(
    \"customer_id\",
    \"total_spent\",
    \"prediction\"
).show()
print()

# 7. EVALUATE MODEL
print(\"Step 7: Evaluating model...\")

evaluator = RegressionEvaluator(
    predictionCol=\"prediction\",
    labelCol=\"total_spent\",
    metricName=\"rmse\"
)

rmse = evaluator.evaluate(df_predictions)

evaluator_r2 = RegressionEvaluator(
    predictionCol=\"prediction\",
    labelCol=\"total_spent\",
    metricName=\"r2\"
)

r2 = evaluator_r2.evaluate(df_predictions)

evaluator_mae = RegressionEvaluator(
    predictionCol=\"prediction\",
    labelCol=\"total_spent\",
    metricName=\"mae\"
)

mae = evaluator_mae.evaluate(df_predictions)

print(f\"RMSE: {rmse:.2f}\")
print(f\"R² Score: {r2:.4f}\")
print(f\"MAE: {mae:.2f}\")
print()

# 8. SEGMENTATION ANALYSIS
print(\"Step 8: Customer segmentation analysis...\")

df_segments = df_features.select(
    \"customer_id\",
    \"total_spent\",
    \"num_purchases\",
    \"high_value_customer\",
    when(col(\"num_purchases\") > 20, \"Active\")
        .when(col(\"num_purchases\") > 10, \"Regular\")
        .otherwise(\"Inactive\")
        .alias(\"activity_level\")
)

print(\"Customer segments:\")
df_segments.show()

# Segment statistics
print(\"\\nSegment statistics:\")
segment_stats = df_segments.groupBy(\"activity_level\") \
    .agg(
        count(\"customer_id\").alias(\"count\"),
        avg(\"total_spent\").alias(\"avg_spent\")
    )
segment_stats.show()
print()

# 9. COHORT ANALYSIS
print(\"Step 9: Cohort analysis...\")

cohort_analysis = df_features.select(
    \"customer_id\",
    \"total_spent\",
    \"high_value_customer\",
    when(col(\"days_active\") < 100, \"New\")
        .when(col(\"days_active\") < 200, \"Returning\")
        .otherwise(\"Loyal\")
        .alias(\"cohort\")
)

print(\"Cohort-based metrics:\")
cohort_stats = cohort_analysis.groupBy(\"cohort\") \
    .agg(
        count(\"customer_id\").alias(\"customers\"),
        avg(\"total_spent\").alias(\"avg_spent\"),
        sum(\"total_spent\").alias(\"total_spent\")
    )

cohort_stats.show()
print()

# 10. SAVE MODEL
print(\"Step 10: Saving model...\")
model_lr.write().overwrite().save(\"/tmp/customer_value_model/\")
print(\"Model saved to /tmp/customer_value_model/\")
