# Script 4: Delta Lake Operations
# ================================

from pyspark.sql.functions import col, current_timestamp, lit
from delta.tables import DeltaTable

print("=== Delta Lake Operations ===\n")

# 1. CREATE SAMPLE DATA
print("Step 1: Creating initial data...")
customers_data = [
    (1, "Alice", "alice@email.com", "US"),
    (2, "Bob", "bob@email.com", "CA"),
    (3, "Charlie", "charlie@email.com", "US"),
    (4, "Diana", "diana@email.com", "MX"),
]

df_customers = spark.createDataFrame(
    customers_data,
    ["customer_id", "name", "email", "country"]
)
print("Sample data created with 4 customers\n")

# 2. WRITE AS DELTA TABLE
print("Step 2: Writing initial Delta table...")
df_customers.write \
    .mode("overwrite") \
    .format("delta") \
    .save("/tmp/customers_delta/")

print("Delta table created at /tmp/customers_delta/\n")

# 3. READ DELTA TABLE
print("Step 3: Reading Delta table...")
df_read = spark.read.format("delta").load("/tmp/customers_delta/")
print("Data in Delta table:")
df_read.show()
print()

# 4. INSERT NEW RECORDS
print("Step 4: Inserting new records...")
new_customers = spark.createDataFrame(
    [(5, "Eve", "eve@email.com", "US")],
    ["customer_id", "name", "email", "country"]
)

new_customers.write \
    .mode("append") \
    .format("delta") \
    .save("/tmp/customers_delta/")

print("New record inserted")
df_updated = spark.read.format("delta").load("/tmp/customers_delta/")
print("Updated data:")
df_updated.show()
print()

# 5. UPDATE RECORDS (using DeltaTable API)
print("Step 5: Updating records...")
delta_table = DeltaTable.forPath(spark, "/tmp/customers_delta/")

delta_table.update(
    condition="customer_id = 1",
    set={"email": "'alice.new@email.com'", "country": "'UK'"}
)

print("Updated customer 1")
df_check = spark.read.format("delta").load("/tmp/customers_delta/")
print(df_check.filter(col("customer_id") == 1).show())
print()

# 6. DELETE RECORDS
print("Step 6: Deleting records...")
delta_table.delete(condition="customer_id = 2")
print("Deleted customer 2")

df_after_delete = spark.read.format("delta").load("/tmp/customers_delta/")
print(f"Remaining records: {df_after_delete.count()}")
df_after_delete.show()
print()

# 7. MERGE (UPSERT) OPERATION
print("Step 7: UPSERT with MERGE...")

# New/updated data
merge_data = [
    (3, "Charlie Brown", "charlie.brown@email.com", "US"),  # Update
    (6, "Frank", "frank@email.com", "CA"),  # Insert
]
df_merge = spark.createDataFrame(
    merge_data,
    ["customer_id", "name", "email", "country"]
)

delta_table.alias("existing") \
    .merge(
        df_merge.alias("updates"),
        "existing.customer_id = updates.customer_id"
    ) \
    .whenMatchedUpdate(set={"name": "updates.name", "email": "updates.email", "country": "updates.country"}) \
    .whenNotMatchedInsert(values={"customer_id": "updates.customer_id", "name": "updates.name", "email": "updates.email", "country": "updates.country"}) \
    .execute()

print("MERGE completed")
df_after_merge = spark.read.format("delta").load("/tmp/customers_delta/")
print("Data after merge:")
df_after_merge.show()
print()

# 8. VIEW TABLE HISTORY
print("Step 8: Viewing table history...")
history = spark.sql("DESCRIBE HISTORY delta.`/tmp/customers_delta/`")
print("Table version history:")
history.select("version", "operation", "operationParameters", "timestamp").show(truncate=False)
print()

# 9. TIME TRAVEL - Read previous version
print("Step 9: Time travel to previous version...")
df_v0 = spark.read \
    .format("delta") \
    .option("versionAsOf", 0) \
    .load("/tmp/customers_delta/")

print("Data at version 0 (initial):")
df_v0.show()
print()

# 10. RESTORE TO PREVIOUS VERSION
print("Step 10: Restore to version 0...")
spark.sql("RESTORE TABLE delta.`/tmp/customers_delta/` TO VERSION AS OF 0")
print("Restored to version 0")

df_restored = spark.read.format("delta").load("/tmp/customers_delta/")
print("Data after restore:")
df_restored.show()
print()

# 11. OPTIMIZE TABLE
print("Step 11: Optimizing table...")
spark.sql("OPTIMIZE delta.`/tmp/customers_delta/`")
print("Table optimized!\n")

# 12. VACUUM (cleanup old versions)
print("Step 12: Vacuuming (cleanup)...")
spark.sql("VACUUM delta.`/tmp/customers_delta/` RETAIN 0 HOURS")
print("Vacuumed old versions!\n")

# 13. CHECK TABLE DETAILS
print("Step 13: Table details...")
details = spark.sql("DESCRIBE DETAIL delta.`/tmp/customers_delta/`")
details.show(truncate=False)
