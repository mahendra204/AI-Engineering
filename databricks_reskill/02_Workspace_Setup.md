# 🏢 Module 2: Workspace Setup & Clusters

## Table of Contents
1. Creating Your First Workspace
2. Workspace Navigation
3. Cluster Basics & Types
4. Creating & Managing Clusters
5. Notebooks: Getting Started
6. Compute Resources & Optimization

---

## 1. Creating Your First Workspace

### Option 1: Free Community Edition (Quickest)

**Step 1:** Visit https://community.cloud.databricks.com  
**Step 2:** Sign up with email or OAuth  
**Step 3:** Accept terms  
**Step 4:** Instant access! ✅

**Limitations:**
- Single user only
- 15 GB storage limit
- No scheduling/jobs
- Perfect for learning though!

### Option 2: Paid Cloud (AWS/Azure/GCP)

**AWS Setup:**
```
1. Go to https://databricks.com/try
2. Choose AWS
3. Sign in to your AWS account
4. Authorize Databricks
5. Create workspace (5-10 min setup)
6. Select region (closest = lowest latency)
```

**Key Decisions:**
- **Cloud:** AWS (most options), Azure, or GCP
- **Region:** us-west-2 (fast), eu-west-1 (Europe)
- **Edition:** Standard (easiest), Premium (advanced)

### Workspace URL

Once created, you'll have a URL like:
```
https://adb-1234567890.5.cloud.databricks.com/
```

🔖 Bookmark this! You'll use it daily.

---

## 2. Workspace Navigation

### Home Screen Layout

```
┌─────────────────────────────────────────────┐
│  🏠 Databricks  ☰ [Workspace] [Account] [?] │
└─────────────────────────────────────────────┘
         ↓ (Click sidebar menu)
  ┌──────────────────────────────────┐
  │ 📁 Workspace                     │
  │ 📊 Workflows                     │
  │ 📚 Repos                         │
  │ 🛠️  Compute                       │
  │ 📋 Jobs                          │
  │ 🗂️  Catalog                       │
  │ 👥 Admin Console                 │
  └──────────────────────────────────┘
```

### Key Navigation Areas

#### 📁 **Workspace** (Home of notebooks)
```
Workspace/
├── Shared/
│   ├── team_projects/
│   └── shared_notebooks/
├── Users/
│   └── your_email@company.com/
│       ├── my_analysis/
│       └── experiments/
└── Repos/
    └── github_projects/
```

**Path format during coding:**
```python
# Reference notebooks in workspace
%run /Users/your_email@company.com/my_utilities
%run /Shared/team_projects/shared_functions
```

#### 🛠️ **Compute** (Cluster management)
```
Clusters
├── Interactive Clusters (for notebooks)
│   └── data-analysis-cluster
├── Job Clusters (auto-created for jobs)
└── SQL Warehouses (for BI/SQL)
```

#### 📋 **Jobs** (Scheduled workflows)
```
New Job
├── Notebook job (run notebook)
├── Python job (run .py file)
├── Spark Submit (run JAR)
└── dbt job (run dbt model)
```

#### 🗂️ **Catalog** (Data governance)
```
Catalogs
├── hive_metastore (default)
├── my_analytics
└── prod_data
    ├── sales_db
    ├── customer_db
    └── product_db
```

---

## 3. Cluster Basics & Types

### What is a Cluster?

A **cluster** is virtual computers(s) that:
1. **Run your code** (Python, SQL, Scala)
2. **Process data** (distributed)
3. **Scale automatically** (or manually)
4. **Cost money** (pay per minute running)

### Cluster Architecture

```
                   ┌──────────────┐
                   │   Databricks │
                   │   (Your IDE) │
                   └──────┬───────┘
                          │
        ┌─────────────────┴────────────────┐
        │                                  │
        ↓                                  ↓
   ┌─────────────┐              ┌──────────────────┐
   │   Driver    │              │   Workers        │
   │             │              │                  │
   │ - Master    │─────────────→│ Executor 1       │
   │ - Schedules │              │ Executor 2       │
   │ - Collects  │              │ Executor 3       │
   │   results   │              │ Executor 4...    │
   └─────────────┘              └──────────────────┘
```

### Cluster Types

#### 1. **All-Purpose Cluster** (Most Common)
- For: Interactive development, notebooks
- Lifetime: Manual start/stop
- Cost: Pay per minute (while running)
- Size: 2-1000 worker nodes
- Use case: Data science, analysis

```python
Create Cluster:
- Name: data-analysis
- Spark: 14.3 LTS
- Node type: i3.xlarge
- Workers: 2-8 (auto-scale)
- Cost: ~$0.50-2.00/hour
```

#### 2. **Job Cluster** (Automated)
- For: Running scheduled jobs
- Lifetime: Auto-created, auto-destroyed
- Cost: Pay only when job runs
- Size: As needed for job
- Use case: ETL, daily reports

#### 3. **SQL Warehouse** (Dashboard queries)
- For: SQL queries, BI tools
- Lifetime: Always on (cost per hour)
- Concurrency: Multiple users
- Cost: $3-50/hour + compute
- Use case: Dashboards, analytics

---

## 4. Creating & Managing Clusters

### Create All-Purpose Cluster (Step-by-step)

**Navigate to:** Compute → Create Cluster

```
┌─────────────────────────────────────┐
│ Cluster Name:  my-learning-cluster  │
│ Cluster Mode: Single node/Multi-node│
│                                     │
│ Databricks Runtime:                 │
│ └─ 14.3 LTS (Recommended)          │
│                                     │
│ Node Type:                          │
│ └─ General: i3.xlarge              │
│    (4 cores, 30 GB RAM)            │
│                                     │
│ Worker Nodes:                       │
│ └─ Min: 2, Max: 8 (auto-scale)     │
│                                     │
│ Driver Node:                        │
│ └─ Same as worker (i3.xlarge)      │
│                                     │
│ [Create Cluster]                    │
└─────────────────────────────────────┘
```

### Configuration Options Explained

| Setting | What it means | Impact |
|---------|---------------|--------|
| **Cluster Mode** | Single vs Multiple | Single = cheaper, multi = scalable |
| **Runtime** | Spark + libraries | LTS = stable, Latest = newest features |
| **Node Type** | CPU/RAM per machine | Larger = faster but costlier |
| **Min Workers** | Minimum cluster size | Too small = slow, too large = expensive |
| **Max Workers** | Auto-scale limit | Safety cap for costs |

### Monitor Cluster Status

```
Status indicators:
🟡 PENDING → Starting (1-2 min)
🟢 RUNNING → Ready to use
🔴 TERMINATED → Stopped (not using resources)
⚠️  ERROR → Problem occurred
```

### Terminate Cluster (Save Money!)

```
When to terminate:
❌ After daily work
❌ Before vacation
❌ When not using for >30 min

How to terminate:
1. Compute → Select cluster
2. Click "Terminate"
3. Confirm

Benefit: Don't pay if not running!
```

### Cluster Auto-Termination

```python
Enable auto-termination to save money:

Settings → Edit → Auto-terminate after:
└─ 15 minutes (recommended)

= Automatically stops cluster if idle
= Saves ~$5-20/day
```

---

## 5. Notebooks: Getting Started

### Create New Notebook

```
Workspace → Create new → Notebook
├─ Name: MyFirstNotebook
├─ Language: Python (or SQL, Scala, R)
├─ Cluster: my-learning-cluster
└─ [Create]
```

### Notebook Components

```
┌─────────────────────────────────────────────┐
│ MyFirstNotebook  [Detach]  [▶ Run All]     │
├─────────────────────────────────────────────┤
│ [ %md ]                                     │
│ # Welcome to Databricks                     │
│ This is markdown documentation              │
│ [Run Cell ▶]                               │
├─────────────────────────────────────────────┤
│ [ Python ]                                  │
│ print("Hello Databricks!")                  │
│ [Run Cell ▶]                               │
├─────────────────────────────────────────────┤
│ [ Output ]                                  │
│ Hello Databricks!                           │
└─────────────────────────────────────────────┘
```

### Cell Types

#### 1. **Python/SQL/Scala** (Execute code)
```python
# Python cell
data = spark.read.csv("/data/file.csv", header=True)
print(f"Rows: {data.count()}")
```

#### 2. **Markdown** (Documentation)
```markdown
%md
# This is a heading
- Bullet points
- More points

**Bold** and *italic* text
```

#### 3. **Shell** (Run bash commands)
```python
%sh
ls -la /mnt/data/
wget https://example.com/file.zip
```

#### 4. **Magic Commands** (Utilities)
```python
%run /path/to/other_notebook    # Include another notebook

%fs ls /data/                    # File system operations

%sql SELECT * FROM table_name    # SQL query

%pip install package_name        # Install Python packages
```

### Keyboard Shortcuts

```
Ctrl + Enter       → Run current cell
Shift + Enter      → Run cell + jump to next
Ctrl + Shift + P   → Command palette
Ctrl + A           → Select all
Ctrl + /           → Comment/uncomment
```

---

## 6. Compute Resources & Optimization

### DBU (Databricks Unit) Explained

**DBU** = compute billing unit
- 1 DBU = 1 core × 1 hour
- Always charged per minute (1/60 DBU)

**Cost Calculate:**
```
Example: 8-core cluster for 3 hours
= 8 cores × 3 hours = 24 DBU
= 24 × $0.40 (per DBU) = $9.60
```

### Cost Optimization Tips

#### 1. **Right-Size Clusters**
```python
# Too large (wastes money)
🔴 32-core cluster for 5GB data

# Right-sized
🟢 4-core cluster processes same data 10% slower
   But costs 87% less!
```

#### 2. **Use Auto-Termination**
```
Saves: $15-50/month per cluster
How: Settings → Auto-terminate after 15 min
```

#### 3. **Use Single-Node for Small Data**
```python
# Multi-node (overkill for small data)
🔴 Cluster: 1 driver + 4 workers
   Cost: 5 nodes = 5× driver cost

# Single-node (perfect for < 10GB)
🟢 Cost: 1/5 compared to multi-node
   Still process 100GB/hour!
```

#### 4. **Schedule Jobs Off-Peak**
```python
# Run at off-peak times (if available)
- Peak hours: 9am-5pm ($0.50/DBU)
- Off-peak: 6pm-8am ($0.30/DBU)
- Savings: 40% for overnight jobs
```

#### 5. **Monitor Cluster Idle Time**
```
Problem: Cluster running but not used
Solution: Set auto-terminate to 15 min
Result: Save 60-70% on compute

Real case:
- Cluster running 8 hours/day
- Actually using: 2 hours/day
- Wasted: $200/month
- Solution: Auto-terminate = Save $140/month
```

### View Cluster Metrics

```
Cluster → Metrics → Monitor:
├─ Active tasks executors
├─ Memory usage (%)
├─ CPU usage (%)
├─ Disk I/O rates
└─ Unbilled executors (idle)
```

---

## 7. Attaching Notebooks to Clusters

### When Creating Notebook
```
Cluster: Select from dropdown
└─ my-learning-cluster
```

### Change Cluster of Running Notebook
```
Top right corner: [Cluster] dropdown
├─ Detach
└─ Attach to another cluster
```

### Detached Notebooks
```python
# Notebook runs but not attached to cluster
# Code won't execute
# Shows: "No clusters available"

Solution: Click [Attach to cluster]
```

---

## 8. Workspace Best Practices

### Organize Notebooks

```
Workspace Structure:
├── Shared/
│   ├── 01_ETL/
│   │   ├── daily_extract.py
│   │   └── transform_rules.py
│   ├── 02_Analytics/
│   │   └── sales_dashboard.sql
│   └── 03_ML/
│       └── recommendation_model.py
└── Users/
    └── my_email@company.com/
        ├── experiments/
        └── learning/
```

### Naming Conventions

```python
✅ Good naming:
2025_03_daily_etl_customer_raw_to_silver
customer_sales_analysis_by_region
ml_feature_engineering_v2

❌ Bad naming:
notebook1
test
my_stuff
analysis (which one?)
```

### Version Control

```
Link Repos for version control:

Repos → Clone Repo → Paste GitHub URL
└─ Commits tracked in GitHub
└─ Roll back to any version
```

---

## 💡 Quick Checklist

- [ ] Created Databricks workspace
- [ ] Bookmarked workspace URL
- [ ] Created all-purpose cluster
- [ ] Created first notebook
- [ ] Ran at least one code cell
- [ ] Understand DBU/pricing
- [ ] Set auto-terminate
- [ ] Organized workspace folders

---

## 🎯 Key Takeaways

1. ✅ Clusters = compute resources
2. ✅ Notebooks = development environment
3. ✅ All-purpose clusters for development
4. ✅ Set auto-terminate to save money
5. ✅ Size clusters based on data
6. ✅ Use right node types
7. ✅ Monitor costs via DBU

---

## 📚 Next Step

**→ Move to Module 3: Data Fundamentals**

Learn how to:
- Ingest data from various sources
- Transform and clean data
- Store data in Delta Lake
- Run first real data pipeline

---

## 🔗 Resource

- **Cluster Configuration:** https://docs.databricks.com/clusters
- **Pricing Calculator:** https://databricks.com/product/pricing
- **Notebook Guide:** https://docs.databricks.com/notebooks

---

**Module 2 Complete! ✅**

**Time to read:** ~20 minutes  
**Time to practice:** ~30 minutes  
**Next:** 03_Data_Fundamentals.md

