# Script 1: Basic Databricks Setup & Cluster Management
# ======================================================

# 1. CHECK SPARK VERSION AND CONTEXT
print("=== Spark Environment ===")
print(f"Spark Version: {spark.version}")
print(f"Python Version: {sc.pythonVer}")
print(f"App Name: {sc.appName}")
print(f"Master: {sc.master}")
print()

# 2. CHECK CLUSTER CONFIGURATION
print("=== Cluster Configuration ===")
conf = spark.sparkContext.getConf()
print(f"Executor Memory: {conf.get('spark.executor.memory')}")
print(f"Driver Memory: {conf.get('spark.driver.memory')}")
print(f"Shuffle Partitions: {conf.get('spark.sql.shuffle.partitions')}")
print(f"Default Parallelism: {sc.defaultParallelism}")
print()

# 3. CREATE SAMPLE DATA
print("=== Creating Sample Data ===")
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType

schema = StructType([
    StructField("customer_id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("order_date", DateType(), True)
])

sample_data = [
    (1, "Alice Johnson", "alice@email.com", 150.50, "2023-01-15"),
    (2, "Bob Smith", "bob@email.com", 200.75, "2023-01-16"),
    (3, "Charlie Brown", "charlie@email.com", 175.25, "2023-01-17"),
    (1, "Alice Johnson", "alice@email.com", 220.00, "2023-01-20"),
    (4, "Diana Davis", "diana@email.com", 315.80, "2023-01-21"),
]

df = spark.createDataFrame(sample_data, schema)
print("Sample DataFrame created with 5 rows")
print()

# 4. DISPLAY DATA
print("=== Display Data ===")
df.show()
print()

# 5. DATA EXPLORATION
print("=== Data Exploration ===")
print(f"Total Rows: {df.count()}")
print(f"Total Columns: {len(df.columns)}")
print(f"Columns: {df.columns}")
print()

# 6. SCHEMA INFORMATION
print("=== Schema ===")
df.printSchema()
print()

# 7. DATA TYPES
print("=== Data Types ===")
for col_name, col_type in df.dtypes:
    print(f"{col_name}: {col_type}")
print()

# 8. BASIC STATISTICS
print("=== Statistics ===")
df.describe().show()
print()

# 9. SAVE TO DBFS
print("=== Saving to DBFS ===")
# Create folder if not exists
dbutils.fs.mkdirs("/dbfs/learning/data/", exist_ok=True)

# Save as CSV
df.write.mode("overwrite").option("header", "true").csv("/dbfs/learning/data/sample.csv")
print("Data saved to /dbfs/learning/data/sample.csv")

# Save as Delta
df.write.mode("overwrite").format("delta").save("/dbfs/learning/data/sample_delta/")
print("Data saved as Delta format")
print()

# 10. VERIFY SAVED FILES
print("=== Verify Saved Files ===")
files = dbutils.fs.ls("/dbfs/learning/data/")
for file in files:
    print(f"  {file.name} - Size: {file.size} bytes")
