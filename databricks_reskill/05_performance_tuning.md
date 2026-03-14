# Module 4.1: Performance Tuning & Optimization

## ⚡ Query Optimization Fundamentals

Query optimization is the process of improving query execution speed and reducing resource usage.

```
Optimization Layers:
┌──────────────────────────┐
│  SQL Query               │ (What you write)
└────────┬─────────────────┘
         │
┌────────▼──────────────────────────────┐
│  Catalyst Optimizer                   │ (What Spark optimizes)
│  • Predicate pushdown                │
│  • Constant folding                  │
│  • Dead code elimination             │
│  • Expression simplification         │
└────────┬──────────────────────────────┘
         │
┌────────▼──────────────────────────────┐
│  Adaptive Query Execution (AQE)       │ (Runtime optimization)
│  • Partition coalescing              │
│  • Join strategy switching           │
│  • Broadcast threshold adjustment    │
└────────┬──────────────────────────────┘
         │
┌────────▼──────────────┐
│  Code Generation      │ (Optimized execution)
│  (Project Tungsten)  │
└──────────────────────┘
```

---

## 🔍 EXPLAIN & Query Plans

```python
# View query execution plan
df = spark.read.delta("/mnt/delta/orders/")
df_plan = df.filter(col("amount") > 100) \
    .groupBy("region") \
    .agg(sum("amount"))

# Simple EXPLAIN
df_plan.explain()

# Extended EXPLAIN (with statistics)
df_plan.explain(extended=True)

# Shows:
# ├── Parsed Logical Plan
# ├── Analyzed Logical Plan
# ├── Optimized Logical Plan
# └── Physical Plan

# SQL version
spark.sql("""
    EXPLAIN EXTENDED
    SELECT region, SUM(amount)
    FROM orders
    WHERE amount > 100
    GROUP BY region
""").show()
```

### Reading Execution Plans

```
Physical Plan:
├── Aggregate [region], [sum(amount)]
│  └── Filter (amount > 100)
│     └── Scan ParquetTable (orders) [Partitions: 200, Filters: (amount > 100)]

Key metrics:
✅ Filters pushed down (good - happens before scan)
✅ Partition pruning (good - skips partitions)
❌ Full table scan needed (inefficient)
❌ Shuffle required (expensive operation)
```

---

## 🎯 Key Optimization Techniques

### 1. **Predicate Pushdown**

```python
# ❌ BAD: Filter after join (processes all rows)
df_result = df_large.join(df_small, "key") \
    .filter(col("amount") > 100)

# ✅ GOOD: Filter before join (processes fewer rows)
df_large_filtered = df_large.filter(col("amount") > 100)
df_result = df_large_filtered.join(df_small, "key")

# Spark usually does this automatically, but be explicit
```

### 2. **Column Pruning**

```python
# ❌ BAD: Select all columns
df.select("*").filter(col("customer_id") == 5).select("name")

# ✅ GOOD: Select only needed columns
df.select("customer_id", "name") \
    .filter(col("customer_id") == 5)
```

### 3. **Partition Pruning**

```python
# Table partitioned by year/month
# ❌ BAD: Full scan
df_2023 = spark.read.delta("/mnt/orders/")
df_filtered = df_2023.filter(col("year") == 2023)

# ✅ GOOD: Scan only 2023 partition
df_2023 = spark.read.delta("/mnt/orders/year=2023/")

# ✅ ALSO GOOD: Partition pruning in WHERE
df_filtered = spark.read.delta("/mnt/orders/") \
    .filter((col("year") == 2023) & (col("month") == 12))

# Limit to specific partitions
df_filtered = spark.read \
    .format("delta") \
    .load("/mnt/orders/") \
    .where("year = 2023 AND month BETWEEN 1 AND 6")
```

### 4. **Join Operations Optimization**

```python
# Broadcast Join (small table to workers)
from pyspark.sql.functions import broadcast

# Automatic if small enough
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "104857600")  # 100MB

# Manual broadcast (for tables < 2GB)
df_result = df_large.join(
    broadcast(df_small),
    "key"
)

# Sort-Merge Join (for large tables)
# Good for frequently joined columns
# Expensive but necessary for large table joins

# Shuffle Join (default for non-broadcast)
# Redistribute data by join key
# Should avoid if possible
```

### 5. **Shuffle Optimization**

```python
# ⚠️ Shuffle = expensive operation (data movement across network)

# Causes of shuffle:
# • GROUP BY
# • JOIN (non-broadcast)
# • DISTINCT
# • Repartition

# Optimize shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "200")  # Default

# For small datasets
spark.conf.set("spark.sql.shuffle.partitions", "50")

# For large datasets
spark.conf.set("spark.sql.shuffle.partitions", "500")

# Adaptive Query Execution (automatic)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

### 6. **Caching Strategy**

```python
from pyspark import StorageLevel

# Cache intermediate results used multiple times
df_clean = df.filter(col("amount") > 0) \
    .filter(col("status") == "active") \
    .cache()

# First action: computes and caches
count1 = df_clean.count()

# Second action: uses cache (much faster!)
grouped = df_clean.groupBy("region").sum()

# When done
df_clean.unpersist()

# Cache strategy
# ├── Cache after expensive filtering
# ├── Cache before reuse
# ├── Cache intermediate aggregations
# └── Don't cache one-time operations
```

### 7. **Materialization**

```python
# Save intermediate results
df_clean = df.filter(col("amount") > 0) \
    .withColumn("tax", col("amount") * 0.1)

df_clean.write.mode("overwrite") \
    .format("delta") \
    .save("/mnt/processed/orders_clean/")

# Reuse from storage
df_clean = spark.read.delta("/mnt/processed/orders_clean/")
result = df_clean.groupBy("region").sum()

# When to use:
# • Complex transformations
# • Used by multiple jobs
# • Multiple runs of same pipeline
```

---

## 💾 Memory Management

```python
# Check cluster memory settings
spark.sparkContext._jsc.hadoopConfiguration()

# Memory configuration
spark.conf.set("spark.memory.fraction", "0.6")           # 60% for Spark
spark.conf.set("spark.memory.storageFraction", "0.5")    # 50% for cache

# Total Memory = (Driver/Executor Memory) × num_executors

# Monitor memory
print(spark.sparkContext.appName)
print(f"Executors: {sc.getConf().get('spark.executor.instances')}")
print(f"Memory per executor: {sc.getConf().get('spark.executor.memory')}")
```

---

## 🎯 Optimization Checklist

```python
# Before running query:
□ Check data size and distribution
□ Review EXPLAIN plan
□ Identify shuffle operations
□ Plan join strategy
□ Partition by join key
□ Filter early in pipeline
□ Select needed columns
□ Use appropriate data types
□ Check for skewed data
□ Enable AQE

# After query runs:
□ Check execution time
□ Review statistics
□ Monitor Spark UI
□ Check GC time
□ Identify bottlenecks
```

---

## 🆕 Photon Engine (2023+)

Vectorized query engine for 10-100x speedup:

```python
# Enable Photon (Premium/Enterprise)
# Automatic on SQL Warehouses
# Optional on All-purpose clusters

spark.conf.set("spark.databricks.photon.enabled", "true")

# Works with:
# • Delta queries
# • Aggregations
# • Joins
# • Sorts
# • Window functions
```

---

## 🧪 Hands-on Lab

```python
# Lab: Query Optimization

# Setup
df = spark.range(1, 100000)
df = df.select(
    col("id").alias("customer_id"),
    (col("id") % 100).alias("region"),
    (col("id") * 2.5).alias("amount")
)

# ❌ SLOW Query
print("SLOW version:")
result_slow = df.select("*") \
    .filter(col("amount") > 50000) \
    .groupBy("region") \
    .agg(sum("amount"))

result_slow.explain(extended=True)
result_slow.collect()

# ✅ FAST Query
print("\nFAST version:")
result_fast = df.select("customer_id", "region", "amount") \
    .filter(col("amount") > 50000) \
    .cache() \
    .groupBy("region") \
    .agg(sum("amount"))

result_fast.explain(extended=True)
result_fast.collect()
```

---

**Duration:** 1.5 hours | **Difficulty:** Advanced | **Last Updated:** 2025

