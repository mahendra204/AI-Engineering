# 📊 Module 3: Data Fundamentals - Ingestion, Transformation & Storage

## Table of Contents
1. Data Ingestion Patterns
2. Ingesting from Common Sources
3. Data Transformation Basics
4. Delta Lake as Storage
5. Data Quality & Validation
6. Practical ETL Pattern

---

## 1. Data Ingestion Patterns

### ETL vs ELT vs Streaming

#### ETL (Extract → Transform → Load)
```
Source DB → Extract → Transform → Load → Warehouse
           (clean here)

When to use: Fixed schedules, batch data
Speed: Hours
Example: Weekly customer export → Transform → Load
```

**Pros:**
- ✅ Data quality guaranteed before load
- ✅ Smaller storage needed
- ✅ Predictable performance

**Cons:**
- ❌ Inflexible (hard to change transformation)
- ❌ Slow (transform before load)

#### ELT (Extract → Load → Transform)
```
Source → Load (raw) → Transform → Analytic tables
         (as-is)       (in warehouse)

When to use: Modern data lakes, rapid changes
Speed: Minutes
Example: Raw data → Store → Transform on-demand
```

**Pros:**
- ✅ Flexible (change transformations anytime)
- ✅ Fast (load immediately)
- ✅ Raw data available for audit

**Cons:**
- ❌ More storage needed
- ❌ Need transformation layer

#### Streaming (Continuous ingestion)
```
Event Stream → Continuous processing → Real-time insights
              (no batching)

When to use: Real-time apps, IoT, dashboards
Latency: Seconds/milliseconds
Example: Click stream → Process → Update Dashboard in real-time
```

### Data Ingestion Workflow (General)

```
Step 1: Connect to source
  ├─ Credentials
  ├─ Connection test
  └─ Table/file discovery

Step 2: Extract metadata
  ├─ Column names/types
  ├─ Row count estimate
  └─ Data quality metrics

Step 3: Perform extraction
  ├─ Full load (all data)
  ├─ Incremental load (only new/changed)
  └─ CDC (change data capture)

Step 4: Land data
  ├─ BRONZE (raw)
  ├─ SILVER (cleaned)
  └─ GOLD (business-ready)

Step 5: Validate
  ├─ Row count check
  ├─ Schema validation
  ├─ Data quality rules
  └─ Alert on failure
```

---

## 2. Ingesting from Common Sources

### 2.1 CSV Files (Local/Cloud Storage)

#### From Local Files
```python
# Method 1: Direct read
df = spark.read.csv("/dbfs/path/file.csv", header=True, inferSchema=True)

# Method 2: With options
df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("encoding", "UTF-8") \
    .csv("/path/file.csv")

# Method 3: Multiple files
df = spark.read.csv("/path/data/*.csv", header=True, inferSchema=True)

df.show()
```

#### From Cloud Storage

```python
# AWS S3
df = spark.read.csv("s3://my-bucket/data.csv", header=True)

# Azure Blob Storage
df = spark.read.csv("wasbs://container@storage.blob.core.windows.net/data.csv")

# Google Cloud Storage
df = spark.read.csv("gs://my-bucket/data.csv")
```

### 2.2 Databases

#### SQL Server
```python
# Read from SQL Server
df = spark.read \
    .format("sqlserver") \
    .option("host", "server.database.windows.net") \
    .option("port", "1433") \
    .option("database", "MyDB") \
    .option("user", "username") \
    .option("password", dbutils.secrets.get(scope="db", key="password")) \
    .option("query", "SELECT * FROM users") \
    .load()

df.show()
```

#### MySQL/PostgreSQL
```python
# PostgreSQL (using JDBC)
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "users") \
    .option("user", "postgres") \
    .option("password", password) \
    .load()
```

#### Delta Table (from another workspace)
```python
# Read existing Delta table
df = spark.read.table("catalog.schema.table_name")

# Or direct path
df = spark.read.format("delta").load("/mnt/data/my_table")
```

### 2.3 APIs

#### REST API
```python
import requests
import json

# GET data from API
response = requests.get(
    "https://api.example.com/data",
    headers={"Authorization": "Bearer token"}
)

data = response.json()

# Create DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("id", IntegerType()),
    StructField("name", StringType()),
    StructField("email", StringType())
])

df = spark.createDataFrame(data["records"], schema=schema)
df.show()
```

### 2.4 JSON Files

```python
# Single JSON file
df = spark.read.json("/path/file.json")

# Multiple JSON files
df = spark.read.json("/path/data/*.json")

# JSON with nested structure
df = spark.read \
    .option("multiline", True) \
    .json("/path/nested.json")

df.show()
```

### 2.5 Parquet Files

```python
# Parquet (most efficient format)
df = spark.read.parquet("/path/file.parquet")

# Multiple parquet files
df = spark.read.parquet("/path/data/*.parquet")

# Most efficient for lake storage
df.write.mode("overwrite").parquet("/path/output")
```

---

## 3. Data Transformation Basics

### 3.1 Column Operations

#### Select Columns
```python
# Select specific columns
df_selected = df.select("id", "name", "email")

# Rename column
df_renamed = df.withColumnRenamed("old_name", "new_name")

# Select with alias
df = df.select(col("customer_id").alias("cust_id"))
```

#### Add Columns
```python
from pyspark.sql.functions import col, when, lit

# Add constant column
df = df.withColumn("ingestion_date", lit("2025-03-17"))

# Add calculated column
df = df.withColumn("total_amount", col("quantity") * col("price"))

# Add conditional column
df = df.withColumn(
    "status",
    when(col("amount") > 1000, "high")
    .when(col("amount") > 100, "medium")
    .otherwise("low")
)

df.show()
```

#### Remove Columns
```python
# Remove single column
df = df.drop("unnecessary_column")

# Remove multiple columns
df = df.drop("col1", "col2", "col3")
```

### 3.2 Filtering

```python
from pyspark.sql.functions import col

# Simple filter
df_filtered = df.filter(col("age") > 18)

# Multiple conditions (AND)
df_filtered = df.filter(
    (col("age") > 18) &
    (col("country") == "USA") &
    (col("active") == True)
)

# OR condition
df_filtered = df.filter(
    (col("status") == "active") |
    (col("status") == "pending")
)

# Using SQL string
df_filtered = df.filter("salary > 50000 AND department = 'Sales'")

df_filtered.show()
```

### 3.3 Aggregation

```python
from pyspark.sql.functions import sum, avg, count, max, min

# Group by and aggregate
df_agg = df \
    .groupBy("region", "product_category") \
    .agg(
        count("*").alias("transaction_count"),
        sum("amount").alias("total_sales"),
        avg("amount").alias("average_sale"),
        max("amount").alias("max_sale"),
        min("amount").alias("min_sale")
    )

df_agg.show()
```

### 3.4 Joins

```python
# Inner join (only matching rows)
df_result = df1.join(df2, on="customer_id", how="inner")

# Left join (all from left, matching from right)
df_result = df1.join(df2, on="customer_id", how="left")

# Left anti-join (in df1 but NOT in df2)
df_result = df1.join(df2, on="customer_id", how="left_anti")

df_result.show()
```

### 3.5 Distinct & Deduplication

```python
# Get unique rows
df_unique = df.distinct()

# Drop duplicates based on columns
df_dedup = df.dropDuplicates(["id", "email"])

# Remove rows where all columns are identical
df_dedup = df.dropDuplicates()

print(f"Before: {df.count()}, After: {df_dedup.count()}")
```

### 3.6 Sorting

```python
from pyspark.sql.functions import col, desc

# Sort ascending
df_sorted = df.orderBy(col("date"))

# Sort descending
df_sorted = df.orderBy(desc("amount"))

# Multiple sort columns
df_sorted = df.orderBy(col("region"), desc("sales"))

df_sorted.show()
```

### 3.7 Union & Combine

```python
# Combine two DataFrames (same schema)
df_combined = df1.union(df2)

# Union all (includes duplicates)
df_combined = df1.unionByName(df2)  # By column names

# Append rows
df_final = df_combined.filter(...).select(...)
```

---

## 4. Delta Lake as Storage

### What is Delta Lake?

**Delta Lake** = Modern data lake format
- ACID transactions (like database)
- Time travel (query any version)
- Schema validation
- 100x better performance

### Writing to Delta Lake

#### Mode Options
```python
# Overwrite (replace entire table)
df.write.mode("overwrite").format("delta").save("/mnt/data/users")

# Append (add rows to existing table)
df.write.mode("append").format("delta").save("/mnt/data/users")

# Error if exists
df.write.mode("error").format("delta").save("/mnt/data/users")

# Ignore if exists
df.write.mode("ignore").format("delta").save("/mnt/data/users")
```

#### Create Delta Table
```python
# Using DataFrame write
df.write.format("delta").mode("overwrite").save("/mnt/tables/customers")

# SQL command
spark.sql("""
    CREATE OR REPLACE TABLE customers AS
    SELECT * FROM my_dataframe
""")

# External table (file not in warehouse)
spark.sql("""
    CREATE TABLE IF NOT EXISTS customers
    USING DELTA
    LOCATION '/mnt/data/customers'
""")
```

### Reading from Delta

```python
# Read entire table
df = spark.read.format("delta").load("/mnt/tables/customers")

# Or use table name
df = spark.read.table("customers")

# Read specific version (time travel!)
df = spark.read.option("versionAsOf", 5).format("delta").load("/path")

# Read at specific timestamp
df = spark.read \
    .option("timestampAsOf", "2025-03-16 10:00:00") \
    .format("delta") \
    .load("/path")
```

### Delta Optimization

```python
# Optimize table (improve read performance)
spark.sql("OPTIMIZE customers")

# With Z-order (for columns used in filter)
spark.sql("""
    OPTIMIZE customers
    ZORDER BY region, product_category
""")

# Vacuum (remove old versions, save space)
spark.sql("VACUUM customers RETAIN 7 DAYS")
```

### Delta Table History

```python
# View all versions
spark.sql("SELECT * FROM table_changes('customers')").show()

# Restore to previous version
spark.sql("RESTORE TABLE customers TO VERSION AS OF 5")

# Show Delta history
spark.sql("DESC HISTORY customers").show()
```

---

## 5. Data Quality & Validation

### Data Quality Checks

```python
from pyspark.sql.functions import col, count, when, isnan, isnull

# Check null values
null_counts = df.select([
    count(when(isnull(col(c)), c)).alias(c)
    for c in df.columns
]).show()

# Check for negative values
df_bad = df.filter(col("amount") < 0)
print(f"Negative amounts: {df_bad.count()}")

# Check for duplicates
df_dups = df.groupBy("id").count().filter(col("count") > 1)
print(f"Duplicate IDs: {df_dups.count()}")

# Check data types
print(df.dtypes)

# Data quality summary
df.describe().show()
```

### Data Validation Framework

```python
def validate_data(df, validations):
    """
    Validate DataFrame against rules
    
    validations = {
        'id': {'nullable': False, 'min': 1},
        'amount': {'nullable': False, 'min': 0},
        'status': {'in': ['active', 'inactive']}
    }
    """
    errors = []
    
    for column, rules in validations.items():
        if rules.get('nullable') is False:
            null_count = df.filter(col(column).isNull()).count()
            if null_count > 0:
                errors.append(f"{column}: {null_count} null values")
        
        if 'min' in rules:
            bad_count = df.filter(col(column) < rules['min']).count()
            if bad_count > 0:
                errors.append(f"{column}: {bad_count} below minimum")
    
    if errors:
        raise Exception(f"Validation failed: {errors}")
    else:
        print("✅ All validations passed!")

# Usage
validations = {
    'id': {'nullable': False},
    'amount': {'nullable': False, 'min': 0},
}
validate_data(df, validations)
```

---

## 6. Practical ETL Pattern (Bronze-Silver-Gold)

### Architecture

```
┌──────────────┐
│ Raw Sources  │ (API, DB, Files)
└──────┬───────┘
       │
       ↓
┌──────────────────┐
│ BRONZE Layer     │ (Raw data, minimal transformation)
│ (Fast ingest)    │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│ SILVER Layer     │ (Cleaned, deduplicated, validated)
│ (Clean & join)   │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│ GOLD Layer       │ (Aggregated, business-ready)
│ (Analytics)      │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│ Consumers        │ (BI, ML, Applications)
└──────────────────┘
```

### Implementation Example

```python
from pyspark.sql.functions import col, current_timestamp, sha2

# 1. BRONZE: Ingest raw data
df_bronze = spark.read.csv("s3://raw-data/customers.csv", header=True)

# Add metadata
df_bronze = df_bronze.withColumn("ingestion_date", current_timestamp())
df_bronze = df_bronze.withColumn("source_file", lit("customers.csv"))

# Save to Bronze
df_bronze.write.mode("append").format("delta").save("/mnt/bronze/customers")

# 2. SILVER: Clean and transform
df_silver = spark.read.format("delta").load("/mnt/bronze/customers") \
    .filter(col("email").contains("@")) \
    .dropDuplicates(["customer_id"]) \
    .withColumn("name", upper(trim(col("name")))) \
    .withColumn("email", lower(col("email"))) \
    .withColumn("hash_ssn", sha2(col("ssn"), 256)) \
    .drop("ssn")  # Remove sensitive column

# Validate and save
df_silver_count = df_silver.count()
print(f"Records in Silver: {df_silver_count}")
df_silver.write.mode("overwrite").format("delta").save("/mnt/silver/customers")

# 3. GOLD: Aggregate for analytics
df_gold = spark.sql("""
    SELECT
        region,
        COUNT(*) as customer_count,
        AVG(lifetime_value) as avg_ltv,
        MAX(last_purchase_date) as last_activity
    FROM delta.`/mnt/silver/customers`
    GROUP BY region
""")

df_gold.write.mode("overwrite").format("delta").save("/mnt/gold/customer_summary")

print("✅ ETL Pipeline Complete!")
```

---

## 7. Error Handling in Data Pipelines

```python
from pyspark.sql.utils import AnalysisException, ParseException

try:
    # Try to read data
    df = spark.read.csv("s3://bucket/data.csv", header=True)
    
    # Try transformation
    df_processed = df.filter(col("amount") > 0).groupBy("region").count()
    
    # Save
    df_processed.write.mode("overwrite").format("delta").save("/mnt/output")
    
    print("✅ Pipeline successful!")
    
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    # Send alert
    
except AnalysisException as e:
    print(f"❌ Schema error: {e}")
    # Log error
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    # Alert team
    raise
```

---

## 📚 Complete Ingestion Script Example

```python
# See: 02_data_transformation.py
# Full script with error handling, logging, and validation
```

---

## 💡 Best Practices

1. ✅ Always validate data after ingestion
2. ✅ Use Delta Lake format (not Parquet alone)
3. ✅ Implement Bronze-Silver-Gold pattern
4. ✅ Add ingestion timestamp to all data
5. ✅ Track data lineage
6. ✅ Use parameterized queries (prevent SQL injection)
7. ✅ Implement error handling and alerts
8. ✅ Monitor pipeline performance
9. ✅ Keep raw data (BRONZE) unchanged
10. ✅ Document transformations

---

## 🎯 Key Takeaways

1. ✅ ETL vs ELT: Choose based on needs
2. ✅ Delta Lake provides ACID + time travel
3. ✅ Spark transformations are lazy evaluated
4. ✅ Bronze-Silver-Gold pattern = scalable design
5. ✅ Validation essential for data quality
6. ✅ Use parameterized queries for security

---

## 📚 Next Step

**→ Move to Module 4: Spark Basics**

Learn about:
- Spark architecture and execution
- RDD vs DataFrame
- Lazy evaluation & DAG
- Performance fundamentals

---

## 🔗 Resources

- **Spark SQL Docs:** https://spark.apache.org/docs/latest/sql-programming-guide.html
- **Delta Lake Guide:** https://docs.delta.io/
- **Data Quality:** https://github.com/bartosz25/dbt-framework/tree/master/data-quality

---

**Module 3 Complete! ✅**

**Time to read:** ~25 minutes  
**Time to practice:** ~45 minutes (run scripts)  
**Next:** 03_spark_basics.md

