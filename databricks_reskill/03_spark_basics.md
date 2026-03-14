# Module 3.1: Apache Spark Fundamentals

## ⚡ What is Apache Spark?

Apache Spark is a **distributed computing framework** for large-scale data processing:

```
Spark Architecture:
┌──────────────────────────────────────────┐
│         Spark Application                │
│  (Python/Scala/SQL/R code)              │
└────────────────┬─────────────────────────┘
                 │
        ┌────────▼────────┐
        │  Spark Context  │
        │  (Driver)       │
        └────────┬────────┘
                 │
    ┌────────────┴────────────┐
    │   Cluster Manager       │
    │ (Standalone/YARN/K8s)   │
    └────────────┬────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐   ┌───▼────┐   ┌──▼────┐
│Worker1│   │Worker2 │   │Worker3│
│Executor   │Executor│   │Executor
└────────┘   └────────┘   └───────┘
```

---

## 🎯 Spark Core Concepts

### 1. **RDD (Resilient Distributed Dataset)**

Lowest-level abstraction - immutable collection distributed across cluster.

```python
# Create RDD
rdd1 = sc.parallelize([1, 2, 3, 4, 5])
rdd2 = sc.textFile("/mnt/data/file.txt")

# RDD Operations
rdd_mapped = rdd1.map(lambda x: x * 2)
rdd_filtered = rdd1.filter(lambda x: x > 2)
rdd_reduced = rdd1.reduce(lambda x, y: x + y)

# Cause evaluation (default: lazy)
result = rdd_mapped.collect()

# RDD Actions
count = rdd1.count()
first_five = rdd1.take(5)
sample = rdd1.takeSample(False, 3)
```

### 2. **DataFrame - High-Level API**

Structured data with schema (SQL-like operations).

```python
# Create DataFrame
df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "name"])

# From RDD
df = rdd.toDF(["id", "name"])

# From Pandas
import pandas as pd
pdf = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
df = spark.createDataFrame(pdf)

# Operations
df_filtered = df.filter(col("id") > 1)
df_selected = df.select("id")
df_aggregated = df.groupBy("name").agg({"id": "count"})

# SQL queries on DataFrame
df.createOrReplaceTempView("people")
result = spark.sql("SELECT * FROM people WHERE id > 1")
```

### 3. **Dataset (Scala) vs DataFrame (Python)**

Python uses only DataFrames (similar to R data.frame or Pandas).

```python
# DataFrame in Python = RDD[Row] + Schema
df.rdd  # Convert back to RDD
```

---

## 🔄 Lazy Evaluation

Spark uses **lazy evaluation** - computation happens only at action.

```python
# Transformations (lazy - not executed)
df1 = df.filter(col("age") > 30)           # Not executed yet
df2 = df1.select("name", "salary")         # Not executed yet
df3 = df2.groupBy("salary").count()        # Not executed yet

# Action (triggers execution!)
result = df3.collect()                      # NOW everything executes

# Execution plan
df3.explain()  # See query plan
```

### Query Optimization

```python
# Catalyst Optimizer automatically optimizes queries
df3.explain(extended=True)  # See optimizations

# Example optimization:
# Original: SELECT * FROM table WHERE age > 30 ORDER BY salary
# Optimized: Push filter down, then sort (more efficient)
```

---

## 📊 RDD vs DataFrame vs Dataset

| Feature | RDD | DataFrame | Dataset |
|---------|-----|-----------|---------|
| API | Low-level | High-level | Type-safe |
| Performance | Slow | Fast | Very Fast |
| Optimization | Manual | Automatic | Automatic |
| Type Safety | None | Weak | Strong |
| SQL Support | No | Yes | Yes |
| Language | All | All | Scala/Java |
| Memory | More | Less | Least |

**When to use:**
- **RDD**: Unstructured data, low-level control
- **DataFrame**: Most cases, SQL queries
- **Dataset**: Scala/Java, compile-time safety

---

## 💾 Persistence & Caching

```python
# Cache in memory
df.cache()  # or df.persist()
result1 = df.filter(...).count()  # First: computes and caches
result2 = df.filter(...).count()  # Second: uses cache (fast!)

# Explicit removal
df.unpersist()

# Different storage levels
from pyspark import StorageLevel

df.persist(StorageLevel.MEMORY_ONLY)
df.persist(StorageLevel.MEMORY_AND_DISK)  # Spillover to disk if needed
df.persist(StorageLevel.DISK_ONLY)
df.persist(StorageLevel.MEMORY_ONLY_2)    # Replicate twice

# Example: Cache intermediate results
df_clean = df.filter(col("amount") > 0).cache()
result1 = df_clean.groupBy("region").sum()
result2 = df_clean.filter(col("status") == "active").count()
```

---

## 🎯 Partitioning

Data is split across partitions for parallel processing.

```python
# Check number of partitions
print(df.rdd.getNumPartitions())

# Repartition (reshuffle data)
df_repartitioned = df.repartition(50)

# Coalesce (reduce partitions, faster)
df_coalesced = df.coalesce(1)

# Partition by column (for skewed data)
df_partitioned = df.repartition("region")

# ⚠️ Be careful with partition size
# Too many: overhead
# Too few: not enough parallelism

# Optimal partitions = num_cores × 2-4
```

---

## ⚙️ Spark Configuration

```python
# Set configuration
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "52428800")  # 50MB

# Get configuration
shuffle_partitions = spark.conf.get("spark.sql.shuffle.partitions")

# Common configs
{
    "spark.sql.adaptive.enabled": "true",           # AQE
    "spark.sql.adaptive.skewJoin.enabled": "true",  # Handle skew
    "spark.sql.shuffle.partitions": "200",
    "spark.memory.fraction": "0.6",
    "spark.memory.storageFraction": "0.5",
    "spark.sql.files.maxPartitionBytes": "128MB"
}
```

---

## 🧮 Performance Metrics

```python
# Spark UI available at: http://driver-ip:4040

# Or in Databricks notebook:
# Clusters → <cluster> → Spark UI

# Check execution time
from time import time

start = time()
result = df.count()
duration = time() - start
print(f"Execution time: {duration:.2f} seconds")

# Check memory usage
df.show()  # Look at Spark UI during execution
```

---

## 🆕 Adaptive Query Execution (AQE) - 2024 Update

```python
# Enable Adaptive Query Execution
spark.conf.set("spark.sql.adaptive.enabled", "true")

# AQE optimizations:
# 1. Dynamically coalesces shuffle partitions
# 2. Handles skewed data automatically
# 3. Converts sort-merge join to broadcast join if beneficial

# Example: Skew handling
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Queries with skewed joins now automatically optimize:
df_large.join(df_small, "key")  # Detects skew and optimizes
```

---

## 🎯 Learning Objectives

After this module, you should understand:

✅ RDD, DataFrame, Dataset concepts  
✅ Lazy evaluation and action  
✅ Query optimization  
✅ Partitioning strategies  
✅ Caching and performance  
✅ Spark configuration  

---

## 🧪 Hands-on Lab

```python
# Lab: RDD vs DataFrame Performance

from pyspark.sql.functions import col
import time

# Create large dataset
data = spark.range(1, 1000000)

# Method 1: RDD (slow)
start = time.time()
result_rdd = data.rdd \
    .map(lambda x: x.value * 2) \
    .filter(lambda x: x > 1000000) \
    .collect()
rdd_time = time.time() - start
print(f"RDD time: {rdd_time:.3f}s")

# Method 2: DataFrame (fast)
start = time.time()
result_df = data \
    .select((col("id") * 2).alias("doubled")) \
    .filter(col("doubled") > 1000000) \
    .collect()
df_time = time.time() - start
print(f"DataFrame time: {df_time:.3f}s")

print(f"Speedup: {rdd_time / df_time:.1f}x faster with DataFrame")
```

---

**Duration:** 1.5 hours | **Difficulty:** Intermediate | **Last Updated:** 2025

