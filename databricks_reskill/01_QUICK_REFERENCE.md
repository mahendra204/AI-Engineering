# Databricks Course - Quick Reference Guide
# ==========================================

## 📚 Course Overview

This comprehensive Databricks course covers:
- **Module 1:** Basics (Platform, Workspace, Clusters)
- **Module 2:** Data Engineering (Ingestion, Transformation, Storage)
- **Module 3:** Spark Fundamentals (RDD, DataFrames, SQL)
- **Module 4:** Advanced Optimization (Performance Tuning)
- **Module 5:** Advanced Topics (Delta Lake, Streaming, ML)
- **Module 6:** Production Ready (Security, Monitoring, CI/CD)

---

## 🚀 Getting Started

### 1. Create a Cluster
```
Compute → Create Cluster
- Name: learning-cluster
- Runtime: 14.3 LTS
- Workers: 2
- Auto-terminate: 30 min
```

### 2. Create Your First Notebook
```python
# Cell 1: Test
spark.version
sc.defaultParallelism

# Cell 2: Create Data
df = spark.range(1, 100)
df.show()
```

### 3. Install Required Packages
```python
%pip install pandas-profiling plotly
```

---

## 💻 Essential Commands

### Python DataFrame Operations
```python
# Read data
df = spark.read.csv("/path/to/file.csv", header=True)
df = spark.read.format("delta").load("/path/to/table/")

# Basic operations
df.show()
df.count()
df.columns
df.schema

# Filter & Transform
df.filter(col("amount") > 100)
df.select("id", "name")
df.withColumn("new_col", col("amount") * 1.1)

# Group & Aggregate
df.groupBy("region").agg(sum("amount"))

# Write
df.write.format("delta").mode("overwrite").save("/path/")
```

### SQL Queries
```sql
-- Create table
CREATE TABLE orders (
    order_id INT,
    amount DOUBLE
)
USING DELTA;

-- Query
SELECT region, SUM(amount) 
FROM orders 
GROUP BY region;

-- Window function
SELECT 
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM orders;
```

---

## 🎯 Key Concepts

### 1. Lazy Evaluation
- Transformations are not executed until an action
- Actions: `.show()`, `.count()`, `.collect()`, `.write()`

### 2. Partitioning
- Splits data across executors
- Improves parallelism
- Key for large datasets

### 3. Delta Lake
- ACID transactions
- Schema enforcement
- Time travel
- Unified storage format

### 4. Caching
```python
df.cache()  # Load into memory
df.unpersist()  # Remove from cache
```

### 5. Joins
```python
# Broadcast join (small table)
df_large.join(broadcast(df_small), "key")

# Default join (shuffle)
df1.join(df2, "key")
```

---

## ⚡ Performance Tips

✅ **DO's**
- Use Delta format
- Partition large tables
- Select only needed columns
- Filter early
- Use broadcast joins for small tables
- Cache reused dataframes
- Enable Adaptive Query Execution

❌ **DON'Ts**
- Don't read all data at once
- Don't use wide transformations without need
- Don't cache one-time operations
- Don't ignore NULL values
- Don't use CSV for production

---

## 🔐 Security Essentials

### Secrets Management
```python
# Store passwords in Databricks Secrets
password = dbutils.secrets.get(scope="my-scope", key="password")

# Create scope
databricks secrets create-scope --scope my-scope
```

### Access Control
```sql
GRANT SELECT ON TABLE sensitive_data TO user@company.com;
REVOKE SELECT ON TABLE sensitive_data FROM user@company.com;
```

---

## 📊 Common Patterns

### ETL Pipeline
```python
# Extract
df_raw = spark.read.csv("/raw/data.csv")

# Transform
df_clean = df_raw \
    .filter(col("amount") > 0) \
    .withColumn("date", to_date(col("date")))

# Load
df_clean.write.mode("overwrite").format("delta").save("/processed/")
```

### Master Data Management
```python
# Slowly Changing Dimension (SCD)
from delta.tables import DeltaTable

dt = DeltaTable.forPath(spark, "/path/to/table/")
dt.merge(...).whenMatched...whenNotMatched...execute()
```

### Real-time Aggregation
```python
# Streaming aggregation
df_stream.groupBy(window("timestamp", "1 minute"), "category") \
    .agg(sum("amount")) \
    .writeStream \
    .format("delta") \
    .table("aggregates")
```

---

## 🧪 Testing Data

```python
# Small test dataset
test_df = spark.createDataFrame([
    (1, "test1", 100.0),
    (2, "test2", 200.0)
], ["id", "name", "amount"])

# Validate results
assert test_df.count() == 2
assert test_df.filter(col("amount") > 150).count() == 1
```

---

## 📈 Monitoring & Debugging

### Check Query Plan
```python
df.explain(extended=True)
```

### Monitor Execution
```
Clusters → Spark UI → Jobs/Stages/Storage
```

### Check Cluster
```bash
databricks clusters get --cluster-id <id>
```

---

## 🎓 Learning Path

**Week 1:**
- Modules 1-2 (Basics, Data Engineering)
- Complete scripts 1-2

**Week 2:**
- Module 3 (Spark Fundamentals)
- Complete scripts 3-4

**Week 3:**
- Module 4 (Optimization)
- Complete scripts 5-6

**Week 4:**
- Modules 5-6 (Advanced, Production)
- Complete scripts 7

---

## 📚 Resources

- [Databricks Documentation](https://docs.databricks.com)
- [Apache Spark Docs](https://spark.apache.org/docs/latest/)
- [Delta Lake Docs](https://docs.delta.io/)
- [PySpark API](https://spark.apache.org/docs/latest/api/python/)

---

## ✅ Certification Prep

After completing this course, you're ready for:
- Databricks Associate Cloud Engineer
- Databricks Professional Data Engineer
- Apache Spark Certification

**Practice Areas:**
- Data ingestion patterns
- Transformation optimization
- Delta Lake operations
- Performance tuning
- Production deployment

---

## 🤝 Getting Help

1. Check Databricks documentation
2. Review course materials
3. Search Stack Overflow
4. Ask in Databricks Community
5. Contact Databricks support (Premium)

---

**Total Course Duration:** 20-26 hours  
**Difficulty:** Beginner → Advanced  
**Last Updated:** 2025

