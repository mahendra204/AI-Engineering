# Module 5.1: Delta Lake Advanced & Streaming

## 🌊 Streaming with Databricks

Process real-time data streams (Kafka, IoT, Event Hubs, Kinesis).

```
Real-time Sources → Databricks Streaming → Delta Tables/Dashboards
├── Kafka
├── Azure Event Hubs
├── AWS Kinesis
├── Files (directory monitoring)
└── Socket/HTTP
```

### Structured Streaming

```python
from pyspark.sql.functions import col, window, sum, count

# Read streaming data from Kafka
df_stream = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker1:9092,broker2:9092") \
    .option("subscribe", "events_topic") \
    .load()

# Parse JSON from Kafka
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("event_id", IntegerType()),
    StructField("customer_id", IntegerType()),
    StructField("amount", IntegerType()),
    StructField("timestamp", StringType())
])

df_parsed = df_stream.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Apply transformations
df_processed = df_parsed.select(
    "event_id",
    "customer_id",
    col("amount").cast("double"),
    to_timestamp(col("timestamp")).alias("event_time")
)

# Aggregation with 1-minute tumbling window
df_agg = df_processed \
    .groupBy(
        window(col("event_time"), "1 minute"),
        "customer_id"
    ) \
    .agg(
        count("event_id").alias("events"),
        sum("amount").alias("total_amount")
    )

# Write to Delta table (append mode)
query = df_agg \
    .writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/mnt/checkpoints/events/") \
    .table("events_aggregated")

# Start stream
query.start()

# Monitor
query.status
query.lastProgress

# Stop stream
query.stop()
```

### Output Modes

```python
# Append Mode: Only new rows written
# ✅ For aggregations and updates
df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .table("events")

# Complete Mode: All rows written
# ✅ For complete aggregate state
df_agg.writeStream \
    .format("delta") \
    .outputMode("complete") \
    .table("agg_state")

# Update Mode: Only updated rows
# ✅ For stateful operations
df_updated.writeStream \
    .format("delta") \
    .outputMode("update") \
    .table("changes")
```

---

## 🎬 Change Data Capture (CDC)

Track data changes for incremental processing:

```python
from delta.tables import DeltaTable

# Enable CDC on table
spark.sql("""
    CREATE TABLE orders (
        order_id INT,
        customer_id INT,
        amount DOUBLE,
        status STRING
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
""")

# Read changes
df_changes = spark.read \
    .format("delta") \
    .option("readChangeData", "true") \
    .option("startingVersion", 0) \
    .table("orders")

# _change_type:
# • insert: New row
# • update_preimage: Before update
# • update_postimage: After update
# • delete: Deleted row
```

---

## 📊 Advanced Analytics with ML

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression

# Feature engineering
assembler = VectorAssembler(
    inputCols=["amount", "frequency", "days_since"],
    outputCol="features"
)
df_features = assembler.transform(df)

# Scaling
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
df_scaled = scaler.fit(df_features).transform(df_features)

# Clustering (K-Means)
kmeans = KMeans(k=5, seed=42)
model = kmeans.fit(df_scaled)
df_clustered = model.transform(df_scaled)

# Regression
lr = LinearRegression(
    featuresCol="features",
    labelCol="target",
    maxIter=10
)
model_lr = lr.fit(df_train)
predictions = model_lr.transform(df_test)

# Save model
model_lr.write().overwrite().save("/mnt/models/linear_regression/")
```

---

## 🔐 Data Governance & Security

```python
# Column-level security
spark.sql("""
    CREATE TABLE customers_secure (
        customer_id INT,
        name STRING COMMENT 'PII',
        email STRING COMMENT 'PII',
        region STRING
    )
    USING DELTA
""")

# Row-level security with views
spark.sql("""
    CREATE VIEW us_customers AS
    SELECT * FROM customers WHERE region = 'US'
""")

# Grant permissions
spark.sql("GRANT SELECT ON TABLE customers TO user@company.com")

# Audit logging
spark.sql("""
    SELECT
        *
    FROM system.access.audit_logs
    WHERE object_name = 'sensitive_table'
    ORDER BY event_time DESC
""")
```

---

## 🆕 Recent Updates (2024-2025)

### Unity Catalog
```python
# Unified governance across clouds
spark.sql("""
    CREATE TABLE main.analytics.customers (
        id INT,
        name STRING
    )
    USING DELTA
""")

# Cross-workspace sharing
spark.sql("""
    GRANT SELECT ON TABLE catalog.schema.table
    TO `account-id`
""")
```

### Databricks SQL Warehouses
```python
# Optimized for SQL
# • Serverless compute
# • Multi-user concurrency
# • Photon engine enabled
# • Fast query times
```

---

**Duration:** 2 hours | **Difficulty:** Advanced | **Last Updated:** 2025

