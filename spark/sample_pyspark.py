"""
Sample PySpark Script - DataFrame Operations
Demonstrates basic PySpark functionality with local execution on Windows
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, split, explode, lower
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import sys
import os

# Set environment variables for Windows compatibility
# Point PySpark to the correct Python executable
python_exec = sys.executable
os.environ['PYSPARK_PYTHON'] = python_exec
os.environ['PYSPARK_DRIVER_PYTHON'] = python_exec

# Suppress Hadoop warnings on Windows
os.environ['HADOOP_HOME'] = os.path.join(os.path.expanduser('~'), '.hadoop')
os.path.exists(os.environ['HADOOP_HOME']) or os.makedirs(os.environ['HADOOP_HOME'], exist_ok=True)

# Create a SparkSession (local mode with 1 thread)
spark = SparkSession.builder \
    .appName("SamplePySparkApp") \
    .master("local[1]") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.sql.shuffle.partitions", "1") \
    .getOrCreate()

# Set log level to reduce verbose output
spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("PYSPARK SAMPLE SCRIPT - DATAFRAME OPERATIONS")
print("=" * 60)

# ============================================
# 1. Create DataFrame from Python data
# ============================================
print("\n1. DataFrame Operations:")
print("-" * 60)

# Create sample data
data = [
    ("Alice", 25, 70000),
    ("Bob", 30, 85000),
    ("Charlie", 35, 95000),
    ("David", 28, 75000),
    ("Eve", 32, 90000)
]

columns = ["Name", "Age", "Salary"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

print("\nOriginal DataFrame:")
df.show()

# Filter and transform
print("\nEmployees with Salary > 80000:")
high_earners = df.filter(col("Salary") > 80000)
high_earners.show()

# Group by and aggregate
print("\nAverage Salary by Age Group:")
age_groups = df.withColumn("AgeGroup", 
    (col("Age") // 5 * 5).cast("int")) \
    .groupBy("AgeGroup") \
    .agg({"Salary": "avg"}) \
    .withColumnRenamed("avg(Salary)", "AvgSalary") \
    .orderBy("AgeGroup")
age_groups.show()

# ============================================
# 2. SQL Operations on DataFrame
# ============================================
print("\n2. SQL Operations:")
print("-" * 60)

# Register DataFrame as temporary view
df.createOrReplaceTempView("employees")

# Execute SQL query
result = spark.sql("""
    SELECT 
        Name,
        Age,
        Salary,
        CASE 
            WHEN Salary >= 90000 THEN 'High'
            WHEN Salary >= 80000 THEN 'Medium'
            ELSE 'Low'
        END as SalaryLevel
    FROM employees
    ORDER BY Salary DESC
""")

print("\nSalary Levels:")
result.show()

# ============================================
# 3. DataFrame Statistics
# ============================================
print("\n3. DataFrame Statistics:")
print("-" * 60)

print("\nDataFrame Schema:")
df.printSchema()

print("\nBasic Statistics:")
df.describe().show()

# ============================================
# 4. Text Processing Example (Simple Version)
# ============================================
print("\n4. Simple String Processing:")
print("-" * 60)

# Create a simple list and show processing
text_list = [
    ("Welcome to PySpark",),
    ("PySpark is awesome",),
    ("Data processing made easy",)
]

text_df = spark.createDataFrame(text_list, ["text"])

# Show sample text data
print("\nSample Text Data:")
text_df.show(truncate=False)

# Count words manually
print("\nWord Analysis:")
words_in_first = text_list[0][0].split()
print(f"  Sentence 1 has {len(words_in_first)} words: {words_in_first}")
words_in_second = text_list[1][0].split()
print(f"  Sentence 2 has {len(words_in_second)} words: {words_in_second}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SAMPLE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nKey Concepts Demonstrated:")
print("  ✓ SparkSession creation")
print("  ✓ DataFrame creation from Python data")
print("  ✓ Filtering and selection")
print("  ✓ Aggregation operations")
print("  ✓ SQL queries on DataFrames")
print("  ✓ String processing")
print("  ✓ Data statistics and schema inspection")
print("\n")

# Stop Spark Session
spark.stop()
print("SparkSession stopped. PySpark is ready for use!")
