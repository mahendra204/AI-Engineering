# 🎯 Module 1: Databricks Introduction

## Table of Contents
1. What is Databricks?
2. Lakehouse Architecture
3. Databricks Editions
4. Key Concepts
5. Getting Started

---

## 1. What is Databricks?

### Definition
**Databricks** is a cloud-based unified data analytics platform that combines:
- **Data Warehouse** (structured, SQL)
- **Data Lake** (unstructured, big data)
- **Machine Learning** (ML/AI)
- **Real-Time Streaming** (event processing)

### Why Databricks? (5 Key Advantages)

| Advantage | Benefit |
|-----------|---------|
| **Lakehouse** | SQL + unstructured data in one platform |
| **Built on Spark** | Industry-standard big data processing |
| **Collaborative** | Notebooks, teams, version control |
| **Serverless** | Pay only for what you use |
| **Unified** | ETL, analytics, ML in single tool |

### The Databricks Story
- **Founded:** 2013 by Apache Spark creators
- **HQ:** San Francisco
- **Backed by:** A16Z, Sequoia, Microsoft
- **IPO:** 2023
- **Users:** 10,000+ organizations globally

---

## 2. Lakehouse Architecture

### Traditional Data Architecture (Problems)

```
Data Source
    ↓
[Data Lake] ← unstructured, unreliable, slow queries
    ↓
[Data Warehouse] ← structured, fast, expensive
    ↓
[BI/Analytics]
    ↓
[ML]
```

**Problems:**
❌ Data duplication across systems  
❌ Expensive to maintain multiple systems  
❌ Data consistency issues  
❌ Slow data movement  

### Lakehouse Architecture (Solution)

```
Data Source
    ↓
[Delta Lake]
├── Raw data + metadata
├── SQL queries (fast)
├── ML pipelines (scalable)
├── Real-time streaming
└── Version control + time travel
    ↓
All analytics & ML in one place
```

**Benefits:**
✅ Single source of truth  
✅ Reduced costs  
✅ Unified analytics  
✅ Real-time capabilities  

### Key Components

#### 1. **Delta Lake**
Brings ACID transactions to data lake:
```sql
-- ACID guarantee (even with concurrent writes)
INSERT INTO sales_data VALUES (...)
UPDATE sales_data SET price = price * 1.1
DELETE FROM sales_data WHERE region = 'OLD'
-- Time travel available!
SELECT * FROM sales_data VERSION AS OF 0
```

#### 2. **Apache Spark**
Distributed computing engine:
- 100x faster than Hadoop MapReduce
- Unified data processing (batch + streaming)
- Multi-language (Python, Scala, SQL, R)

#### 3. **Unity Catalog**
Governance layer (added 2023):
```sql
-- Govern access across clusters, workspaces
GRANT SELECT ON TABLE prod.sales TO user@company.com
MASK COLUMN salary USING NULL WHERE NOT is_admin
```

#### 4. **Photon**
GPU-accelerated query engine:
- 2-10x faster queries
- C++ implementation for Spark
- No code changes needed

---

## 3. Databricks Editions

### Community Edition (FREE)
- ✅ Single user
- ✅ Up to 15 GB data
- ✅ Perfect for learning
- ❌ No collaboration
- ❌ No scheduling/jobs

### Standard Edition
- ✅ SQL + notebooks
- ✅ Collaborative workspaces
- ✅ Job scheduling
- ✅ Up to 500 users
- ❌ No Advanced governance
- ❌ No Unity Catalog

### Premium Edition
- ✅ Everything in Standard +
- ✅ Unity Catalog
- ✅ SCIM sync
- ✅ Advanced admin controls
- ✅ Enterprise support

### Enterprise Edition
- ✅ Everything + custom SLA
- ✅ Dedicated support
- ✅ Custom integrations
- ✅ Highest security

**Bottom line:** Start FREE (Community), upgrade as you scale.

---

## 4. Key Concepts

### 4.1 Notebooks
Interactive documents combining code + documentation:
```python
# Python cell
data = spark.read.csv("/data/sales.csv", header=True)
data.show()

# Cell markdown
# ## My Analysis
# This shows sales by region
```

### 4.2 Clusters
Virtual computers to run code:
- **Driver:** Organizes work (1 per cluster)
- **Workers:** Execute tasks (many)
- **Auto-scaling:** Grows/shrinks by demand

```
┌─────────────────┐
│    Driver       │ (1 machine)
│  - Coordinates  │
│  - Executes     │
└────────┬────────┘
         │
    ┌────┴────┬────────┬─────────┐
    ↓         ↓        ↓         ↓
 Worker 1  Worker 2 Worker 3  Worker 4
 (4 core)  (4 core) (4 core)  (4 core)
```

### 4.3 Spark Execution

1. **Lazy Evaluation** - Define operations, execute on action
2. **DAG (Directed Acyclic Graph)** - Optimize before running
3. **Partitions** - Data split across workers for parallel processing

```python
# Lazy - no execution yet
df = spark.read.csv("sales.csv")
df_filtered = df.filter(df.amount > 100)
df_grouped = df_filtered.groupBy("region").sum()

# Eager - now it executes!
result = df_grouped.collect()  # ← Action triggers execution
```

### 4.4 Delta Lake Features

#### ACID Transactions
```sql
-- All 3 queries are atomic (all or nothing)
BEGIN TRANSACTION
  INSERT INTO customer VALUES (1, "Alice");
  UPDATE sales SET status = "completed" WHERE id = 1;
  DELETE FROM temp WHERE date < '2023-01-01';
COMMIT;
```

#### Time Travel
```sql
-- Query data from yesterday
SELECT * FROM sales TIMESTAMP AS OF "2025-03-16 00:00:00"

-- Query from specific version
SELECT * FROM sales VERSION AS OF 5

-- Restore to previous version
RESTORE TABLE sales TO VERSION AS OF 5
```

#### Schema Evolution
```python
# Add column automatically (no schema migration needed!)
df = spark.read.csv("file.csv")
df_new = df.withColumn("new_col", lit("value"))
df_new.write.mode("overwrite").save("output")
```

### 4.5 Jobs & Workflows
Automate repeating tasks:

```python
# Schedule notebook to run daily
dbutils.jobs.taskRunNow(task_id=1, notebook_params={
    "date": "2025-03-17",
    "region": "NA"
})
```

---

## 5. Architecture Layers (Bottom to Top)

```
┌────────────────────────────────────────┐
│   Applications Layer                   │
│  (BI Tools, Custom Apps, ML Models)    │
└────────────────────────────────────────┘
              ↑
┌────────────────────────────────────────┐
│   Data Services Layer                  │
│  (SQL, Python, Streaming, ML APIs)     │
└────────────────────────────────────────┘
              ↑
┌────────────────────────────────────────┐
│   Compute Layer                        │
│  (Spark Engine, Photon, Jobs)          │
└────────────────────────────────────────┘
              ↑
┌────────────────────────────────────────┐
│   Storage Layer                        │
│  (Delta Lake, S3/Azure/GCS)            │
└────────────────────────────────────────┘
              ↑
┌────────────────────────────────────────┐
│   Cloud Infrastructure                 │
│  (AWS/Azure/GCP + Networking)          │
└────────────────────────────────────────┘
```

---

## 6. Real-World Use Cases

### 1. **Real-Time Analytics**
```
Clickstream → Kafka → Spark Streaming → Dashboard
(Update every 10 seconds)
```

### 2. **ETL Pipeline**
```
Sales DB → Extract → Transform → Load → Data Warehouse
(Nightly job)
```

### 3. **ML Model Training**
```
Customer Data → Feature Engineering → Training → Model
(Weekly retraining)
```

### 4. **Data Governance**
```
All Data → Unity Catalog → Access Control → Audit Logs
(Real-time enforcement)
```

---

## 7. Databricks Workspace Overview

```
Workspace
├── Notebooks
│   ├── Shared (team access)
│   └── Users (personal)
├── Repos (GitHub integration)
├── Data (tables, external data)
├── Workflows (jobs, pipelines)
└── Clusters (compute resources)
```

---

## 8. Pricing Model (as of 2025)

### Community Edition
**FREE** ⭐
- Great for learning
- Single user only

### Standard Edition
- **$0.30 per DBU/hour**
- DBU = compute unit
- Auto-scaling available
- ~$200-500/month typical

### Premium Edition
- **$0.55 per DBU/hour**
- Advanced features
- Higher limits
- ~$500-2000/month typical

### How to Estimate Costs
```
Monthly Cost = (Cluster Size in DBU) × (Hours/month) × (Price/DBU)

Example:
- 8-core cluster (8 DBU)
- Running 40 hours/month
- Standard edition ($0.30/DBU)

Cost = 8 × 40 × $0.30 = $96/month
```

---

## 9. Databricks vs. Alternatives

| Feature | Databricks | AWS EMR | Snowflake | BigQuery |
|---------|-----------|---------|-----------|----------|
| SQL | ✅ | ✅ | ✅ | ✅ |
| Python/Spark | ✅ | ✅ | ❌ | Limited |
| Streaming | ✅ | ✅ | Limited | ✅ |
| ML | ✅ | Limited | ❌ | ✅ |
| Governance | ✅ | Limited | ✅ | ✅ |
| Price | $$ | $$$ | $$$$ | $$ |
| Learning Curve | Medium | High | Easy | Easy |

**Verdict:** Databricks best for complete AI/ML/Analytics stack.

---

## 10. Getting Started Roadmap

```
Week 1: Concepts (this module)
     ↓
Week 2: Workspace Setup & Clusters
     ↓
Week 3: Data Ingestion & Transformation
     ↓
Week 4: Spark SQL
     ↓
Week 5: Optimization
     ↓
Week 6+: Advanced Topics & Production
```

---

## 🎯 Key Takeaways

1. ✅ Databricks = Lakehouse (data warehouse + lake combined)
2. ✅ Built on Spark, adds SQL + governance
3. ✅ Delta Lake provides ACID guarantees
4. ✅ Better than separate warehouse + lake
5. ✅ Scaling is automatic (pay per use)
6. ✅ Great for AI/ML + Analytics
7. ✅ Free tier available for learning

---

## 📚 Next Step

**→ Move to Module 2: Workspace Setup & Clusters**

Learn how to:
- Create a workspace
- Set up clusters
- Create your first notebook
- Run your first Spark code

---

## 🔗 External Resources

- **Databricks Docs:** https://docs.databricks.com
- **Apache Spark Guide:** https://spark.apache.org/docs/latest/
- **Learning Paths:** https://databricks.com/learn
- **Community Edition:** https://community.cloud.databricks.com

---

## ✅ Quiz (Self-Check)

1. What is the main advantage of Lakehouse?
2. Name 3 Databricks editions
3. What is Delta Lake used for?
4. What is lazy evaluation?
5. What is a DBU?

**Answers at end of course materials**

---

**Module 1 Complete! ✅**

**Time to read:** ~30 minutes  
**Time to understand:** ~1 hour  
**Next:** 02_Workspace_Setup.md

