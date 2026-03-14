# Script 2: Data Transformation Operations
# =========================================

from pyspark.sql.functions import (
    col, when, sum as spark_sum, avg, max, min, count, 
    upper, lower, substring, concat, lit, datediff, 
    year, month, to_date, current_date, row_number
)
from pyspark.sql.window import Window

# 1. CREATE SAMPLE DATA
print("=== Creating Sample Sales Data ===")
sales_data = [
    (1, "Alice", "US", 150.50, "2023-01-15"),
    (2, "Bob", "CA", 200.75, "2023-01-16"),
    (3, "Charlie", "US", 175.25, "2023-01-17"),
    (1, "Alice", "US", 220.00, "2023-01-20"),
    (4, "Diana", "MX", 125.50, "2023-01-21"),
    (2, "Bob", "CA", 300.00, "2023-01-22"),
]

df = spark.createDataFrame(sales_data, ["customer_id", "name", "region", "amount", "date"])
print("Sample data created")
df.show()
print()

# 2. FILTERING DATA
print("=== Filtering Operations ===")

# Filter: amount > 150
df_filtered = df.filter(col("amount") > 150)
print("Filter: amount > 150")
df_filtered.show()
print()

# Filter: multiple conditions
df_filtered2 = df.filter((col("region") == "US") & (col("amount") > 160))
print("Filter: region='US' AND amount > 160")
df_filtered2.show()
print()

# 3. COLUMN OPERATIONS
print("=== Column Operations ===")

df_transformed = df.select(
    col("customer_id"),
    upper(col("name")).alias("name_upper"),
    col("region"),
    (col("amount") * 1.1).alias("amount_with_tax"),
    to_date(col("date")).alias("order_date")
)

print("Apply transformations: uppercase name, add tax, convert date")
df_transformed.show()
print()

# 4. CONDITIONAL LOGIC
print("=== Conditional Logic (CASE WHEN) ===")

df_conditional = df.select(
    "*",
    when(col("amount") < 150, "Small")
        .when(col("amount") < 250, "Medium")
        .otherwise("Large")
        .alias("order_size")
)

print("Add order_size column based on amount")
df_conditional.show()
print()

# 5. AGGREGATION
print("=== Aggregation Operations ===")

# Sum by region
df_region_sum = df.groupBy("region").agg(
    spark_sum("amount").alias("total_sales"),
    count("customer_id").alias("num_orders"),
    avg("amount").alias("avg_sale")
)

print("Aggregation by region:")
df_region_sum.show()
print()

# 6. WINDOW FUNCTIONS
print("=== Window Functions ===")

window_spec = Window.partitionBy("region").orderBy(col("amount").desc())

df_ranked = df.select(
    "*",
    row_number().over(window_spec).alias("rank_in_region")
)

print("Rank customers by amount within each region:")
df_ranked.show()
print()

# 7. DATE OPERATIONS
print("=== Date Operations ===")

df_dates = df.select(
    "customer_id",
    "name",
    col("date"),
    to_date(col("date")).alias("date_parsed"),
    year(to_date(col("date"))).alias("year"),
    month(to_date(col("date"))).alias("month"),
    datediff(current_date(), to_date(col("date"))).alias("days_ago")
)

print("Date transformations:")
df_dates.show()
print()

# 8. DEDUPLICATION
print("=== Deduplication ===")

print("Original data:")
df.show()
print(f"Total rows: {df.count()}")

df_dedup = df.dropDuplicates(["customer_id"])
print("After deduplication by customer_id:")
df_dedup.show()
print(f"Total rows: {df_dedup.count()}")
print()

# 9. NULL HANDLING
print("=== NULL Handling ===")

# Add some nulls
df_nulls = spark.createDataFrame([
    (1, "Alice", "US", None),
    (2, None, "CA", 200.75),
    (3, "Charlie", None, 175.25),
], ["customer_id", "name", "region", "amount"])

print("Data with NULLs:")
df_nulls.show()

# Remove rows with any NULL
df_no_nulls = df_nulls.dropna()
print("After dropping NULLs:")
df_no_nulls.show()

# Fill NULLs with default
df_filled = df_nulls.fillna({"name": "Unknown", "region": "Other", "amount": 0})
print("After filling NULLs:")
df_filled.show()
print()

# 10. SAVE TRANSFORMED DATA
print("=== Saving Transformed Data ===")
df_transformed.write.mode("overwrite").format("delta").save("/dbfs/learning/data/transformed/")
print("Transformed data saved!")
