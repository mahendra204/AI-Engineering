# Module 6.1: Production Best Practices

## 🚀 Production Readiness Checklist

```python
# Before deploying to production:
□ Code quality (unit tests, code review)
□ Error handling (try-catch, retry logic)
□ Logging and monitoring
□ Performance testing
□ Security review (secrets management)
□ Documentation complete
□ Rollback strategy
□ Disaster recovery plan
```

---

## 🧪 Testing & Quality

```python
# Unit testing with pytest
def test_data_transformation():
    # Arrange
    input_df = spark.createDataFrame([(1, "test")], ["id", "name"])
    expected = spark.createDataFrame([(1, "TEST")], ["id", "name"])
    
    # Act
    actual = input_df.select(col("id"), upper(col("name")).alias("name"))
    
    # Assert
    assert actual.collect() == expected.collect()

# Data validation
def validate_data(df):
    """Validate data quality before processing"""
    errors = []
    
    # Check for nulls
    null_count = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    if null_count.collect()[0][0] > 0:
        errors.append("NULL values found")
    
    # Check data types
    for col_name in df.columns:
        col_type = dict(df.dtypes)[col_name]
        if col_type != "double" and col_name == "amount":
            errors.append(f"Column {col_name} has wrong type: {col_type}")
    
    if errors:
        raise ValueError(f"Data validation failed: {errors}")
    
    return True

# Run validation
validate_data(df_input)
```

---

## 📝 Logging & Monitoring

```python
import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def process_data_with_logging(df):
    """Process data with comprehensive logging"""
    
    try:
        logger.info(f"Starting data processing at {datetime.now()}")
        logger.info(f"Input records: {df.count()}")
        
        # Process
        df_clean = df.filter(col("amount") > 0)
        logger.info(f"After filtering: {df_clean.count()} records")
        
        # Aggregate
        result = df_clean.groupBy("region").agg(sum("amount"))
        logger.info(f"Aggregation complete. Result rows: {result.count()}")
        
        # Save
        result.write.mode("overwrite").format("delta").save("/mnt/output/")
        logger.info("Data saved successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise

# Databricks-specific logging
dbutils.notebook.run("parent_notebook", timeout_seconds=3600)
```

---

## 🛡️ Error Handling & Retry Logic

```python
from retry import retry
from functools import wraps
import time

# Retry decorator
@retry(tries=3, delay=5, backoff=2)
def read_external_api():
    """Automatically retry failed operations"""
    response = requests.get("https://api.example.com/data")
    return response.json()

# Manual retry logic
def load_data_with_retry(path, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = spark.read.format("delta").load(path)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

# Try-catch in PySpark
try:
    df = spark.read.csv("/mnt/data/file.csv")
    df.write.mode("overwrite").format("delta").save("/mnt/output/")
except FileNotFoundError:
    logger.error("Input file not found")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

---

## 💰 Cost Optimization

```python
# 1. Use job clusters (vs all-purpose)
# Job clusters: $1.50 DBU/hour
# All-purpose: $3.00 DBU/hour
# Savings: 50%

# 2. Auto-termination
cluster_config = {
    "idle_in_minutes": 30,
    "auto_terminate": True
}

# 3. Autoscaling
cluster_config = {
    "autoscale": {
        "min_workers": 1,
        "max_workers": 10
    }
}

# 4. Right-size VMs
# Small: dev/testing
# Medium: production (sweet spot)
# Large: big data processing

# 5. Data partitioning (skip unnecessary scans)
# Partition by: date, region, customer_segment
# Saves: 60-80% query cost

# 6. Use Delta optimization
spark.sql("OPTIMIZE /mnt/delta/table/ ZORDER BY key_column")

# Cost tracking
spark.sql("""
    SELECT
        cluster_name,
        SUM(dbus) as total_dbus,
        SUM(dbus) * 0.40 as estimated_cost_usd
    FROM cluster_billing
    GROUP BY cluster_name
""")
```

---

## 🔄 CI/CD Pipeline

```yaml
# .github/workflows/databricks-deploy.yml
name: Deploy to Databricks

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install Databricks CLI
        run: pip install databricks-cli
      
      - name: Configure Databricks
        run: |
          echo "[DEFAULT]" > ~/.databrickscfg
          echo "host = ${{ secrets.DATABRICKS_HOST }}" >> ~/.databrickscfg
          echo "token = ${{ secrets.DATABRICKS_TOKEN }}" >> ~/.databrickscfg
      
      - name: Upload notebook
        run: |
          databricks workspace import_dir ./notebooks /Shared/deployed
          
      - name: Deploy job
        run: |
          databricks jobs reset --job-config jobs.json
      
      - name: Run tests
        run: |
          python -m pytest tests/
```

---

## 🔒 Security Best Practices

```python
# 1. Secrets Management (Never hardcode!)
password = dbutils.secrets.get(scope="databricks", key="db_password")
api_key = dbutils.secrets.get(scope="external_api", key="api_key")

# 2. Network security
# • Use private endpoints (AWS PrivateLink, Azure)
# • VPC security groups / Network ACLs
# • IP whitelist for cluster access

# 3. Data encryption
# • At rest: Enable DBFS encryption
# • In transit: TLS for all connections
# • Column-level: Encrypt sensitive columns

# 4. Access control
spark.sql("GRANT SELECT ON TABLE sensitive_data TO `audit_group`")

# 5. Audit logging
spark.sql("""
    SELECT
        user_identity.email as user,
        action_type,
        object_name,
        timestamp
    FROM system.access.audit_logs
    WHERE object_name = 'sensitive_table'
""")

# 6. IP allowlist
cluster_config = {
    "security_configuration": "encryption-enabled",
    "init_scripts": [{
        "dbfs": {"destination": "dbfs:/init/security.sh"}
    }]
}
```

---

## 📊 Monitoring & Alerting

```python
# Set up alerts for job failures
from databricks_sdk import WorkspaceClient

client = WorkspaceClient()

# Monitor job runs
jobs = client.jobs.list()
for job in jobs:
    latest_run = client.jobs.get_run(job.job_id)
    if latest_run.state == "FAILED":
        print(f"ALERT: Job {job.job_id} failed!")
        # Send email/Slack notification

# Check table sizes
spark.sql("""
    SELECT
        table_name,
        size_in_bytes / (1024 * 1024 * 1024) as size_gb,
        num_files,
        table_version
    FROM information_schema.tables
    WHERE table_size > 100  -- 100GB
""")

# Performance monitoring
spark.sql("""
    SELECT
        *
    FROM system.compute.clusters
    WHERE state NOT IN ('RUNNING', 'PENDING')
""")
```

---

## 📋 Documentation & Handoff

```python
# Document your code
def process_orders(df: DataFrame) -> DataFrame:
    """
    Process raw orders data and prepare for analytics.
    
    Args:
        df (DataFrame): Raw orders DataFrame with columns:
            - order_id (INT): Unique order identifier
            - amount (DOUBLE): Order amount in USD
            - date (STRING): Order date in YYYY-MM-DD format
    
    Returns:
        DataFrame: Processed orders with:
            - order_id (INT)
            - amount (DOUBLE)
            - amount_with_tax (DOUBLE): Amount including 10% tax
            - year (INT)
            - month (INT)
    
    Raises:
        ValueError: If amount < 0 or date is invalid
    
    Example:
        >>> df = spark.read.csv('/mnt/data/orders.csv')
        >>> result = process_orders(df)
    """
    return df.select(
        "order_id",
        col("amount"),
        (col("amount") * 1.1).alias("amount_with_tax"),
        year(to_date(col("date"))).alias("year"),
        month(to_date(col("date"))).alias("month")
    )
```

---

**Duration:** 2 hours | **Difficulty:** Advanced | **Last Updated:** 2025

