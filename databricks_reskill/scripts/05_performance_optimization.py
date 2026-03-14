# Script 5: Performance Optimization
# ===================================

from pyspark.sql.functions import col, row_number, sum as spark_sum, rand
from pyspark.sql.window import Window
import time

print("=== Performance Optimization Demo ===\n")

# 1. CREATE LARGE DATASET
print(\"Step 1: Creating large dataset (100K rows)...\")
df_large = spark.range(1, 100000) \
    .select(
        col(\"id\").alias(\"customer_id\"),
        (col(\"id\") % 100).alias(\"region\"),
        (col(\"id\") * 2.5).alias(\"amount\"),
        (col(\"id\") % 10).alias(\"category\")
    )

print(f\"Total rows: {df_large.count()}\n\")

# 2. ❌ SLOW QUERY (no optimization)
print(\"Step 2: SLOW query (unoptimized)...\")
start = time.time()

df_slow = df_large \
    .select(\"*\") \
    .filter(col(\"amount\") > 50000) \
    .filter(col(\"category\") < 5) \
    .groupBy(\"region\") \
    .agg(spark_sum(\"amount\").alias(\"total\"))

result_slow = df_slow.collect()
slow_time = time.time() - start

print(f\"Query time: {slow_time:.3f} seconds\")
print(f\"Result rows: {len(result_slow)}\n\")

# 3. ✅ FAST QUERY (optimized)
print(\"Step 3: FAST query (optimized)...\")
start = time.time()

df_fast = df_large \
    .select(\"region\", \"amount\") \
    .filter((col(\"amount\") > 50000) & (col(\"category\") < 5)) \
    .groupBy(\"region\") \
    .agg(spark_sum(\"amount\").alias(\"total\"))

result_fast = df_fast.collect()
fast_time = time.time() - start

print(f\"Query time: {fast_time:.3f} seconds\")
print(f\"Result rows: {len(result_fast)}\n\")

print(f\"Speedup: {slow_time / fast_time:.2f}x faster\n\")

# 4. EXPLAIN PLAN COMPARISON
print(\"Step 4: Query execution plans...\")
print(\"\\nSLOW query plan:\")
df_slow.explain(extended=False)

print(\"\\nFAST query plan:\")
df_fast.explain(extended=False)
print()

# 5. PARTITIONING OPTIMIZATION
print(\"Step 5: Partitioning optimization...\")

# Write partitioned data
df_large.write \
    .mode(\"overwrite\") \
    .partitionBy(\"region\") \
    .format(\"delta\") \
    .save(\"/tmp/sales_partitioned/\")

print(\"Data written with region partitioning\")

# Query with partition pruning
start = time.time()
df_partitioned = spark.read.delta(\"/tmp/sales_partitioned/\") \
    .filter(col(\"region\") == 5) \
    .filter(col(\"amount\") > 50000)

count_partitioned = df_partitioned.count()
partition_time = time.time() - start

print(f\"Query with partition pruning: {partition_time:.3f} seconds\")
print(f\"Rows returned: {count_partitioned}\n\")

# 6. CACHING OPTIMIZATION
print(\"Step 6: Caching optimization...\")

# Without cache
df_test = df_large.filter(col(\"amount\") > 50000)

start = time.time()
result1 = df_test.count()
time1 = time.time() - start

start = time.time()
result2 = df_test.groupBy(\"region\").count().collect()
time2 = time.time() - start

print(f\"Without cache - Count: {time1:.3f}s, GroupBy: {time2:.3f}s\")

# With cache
df_test.cache()

start = time.time()
result1 = df_test.count()
time1_cached = time.time() - start

start = time.time()
result2 = df_test.groupBy(\"region\").count().collect()
time2_cached = time.time() - start

print(f\"With cache - Count: {time1_cached:.3f}s, GroupBy: {time2_cached:.3f}s\")
print(f\"Cache speedup: {(time1 + time2) / (time1_cached + time2_cached):.2f}x faster\n\")

# 7. BROADCAST JOIN
print(\"Step 7: Broadcast join optimization...\")

# Create large table
df_orders = spark.range(1, 50000) \
    .select(
        col(\"id\").alias(\"order_id\"),
        (col(\"id\") % 1000).alias(\"product_id\"),
        (col(\"id\") * 1.5).alias(\"amount\")
    )

# Create small lookup table
df_products = spark.range(1, 1001) \
    .select(
        col(\"id\").alias(\"product_id\"),
        (col(\"id\") % 10).alias(\"category\")
    )

# ❌ Without broadcast
start = time.time()
result_no_broadcast = df_orders.join(df_products, \"product_id\").collect()
time_no_broadcast = time.time() - start

# ✅ With broadcast
from pyspark.sql.functions import broadcast

start = time.time()
result_broadcast = df_orders.join(broadcast(df_products), \"product_id\").collect()
time_broadcast = time.time() - start

print(f\"Without broadcast: {time_no_broadcast:.3f}s\")
print(f\"With broadcast: {time_broadcast:.3f}s\")
print(f\"Speedup: {time_no_broadcast / time_broadcast:.2f}x faster\n\")

# 8. CONFIGURATION OPTIMIZATION
print(\"Step 8: Configuration optimization...\")

# Check current settings
print(f\"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}\")
print(f\"Adaptive QE enabled: {spark.conf.get('spark.sql.adaptive.enabled')}\")

# Optimize settings
spark.conf.set(\"spark.sql.adaptive.enabled\", \"true\")
spark.conf.set(\"spark.sql.adaptive.skewJoin.enabled\", \"true\")

print(\"\\nOptimized settings:\")
print(f\"Adaptive QE enabled: {spark.conf.get('spark.sql.adaptive.enabled')}\")
print(f\"Skew join handling: {spark.conf.get('spark.sql.adaptive.skewJoin.enabled')}\")
