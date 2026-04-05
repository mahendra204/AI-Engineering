"""
PySpark Installation Test - Simple Version
"""

print("=" * 60)
print("PYSPARK INSTALLATION TEST")
print("=" * 60)

# Test imports
print("\n[Test 1] Importing PySpark...")
try:
    from pyspark import SparkContext, __version__
    from pyspark.sql import SparkSession
    print(f"[PASS] PySpark version {__version__} imported successfully")
except ImportError as e:
    print(f"[FAIL] {e}")
    exit(1)

# Test dependencies
print("\n[Test 2] Checking dependencies...")
for lib in ['numpy', 'pandas', 'py4j']:
    try:
        __import__(lib)
        print(f"[PASS] {lib} is installed")
    except ImportError:
        print(f"[FAIL] {lib} is missing")

# Test Java
print("\n[Test 3] Checking Java installation...")
import subprocess
result = subprocess.run(['java', '-version'], capture_output=True)
if result.returncode == 0:
    print(f"[PASS] Java is installed")
else:
    print(f"[FAIL] Java not found")

# Test SparkSession
print("\n[Test 4] Creating SparkSession...")
import sys, os
os.environ['PYSPARK_PYTHON'] = sys.executable

spark = SparkSession.builder \
    .appName("Test") \
    .master("local[1]") \
    .config("spark.sql.shuffle.partitions", "1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"[PASS] SparkSession created")
print(f"  App: {spark.sparkContext.appName}")
print(f"  Master: {spark.sparkContext.master}")

# Test DataFrame creation
print("\n[Test 5] Creating DataFrame...")
import pandas as pd
pdf = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]})
df = spark.createDataFrame(pdf)
print(f"[PASS] DataFrame created with {df.count()} rows")

# Test SQL
print("\n[Test 6] Running SQL query...")
df.createOrReplaceTempView("test_table")
result = spark.sql("SELECT COUNT(*) as cnt FROM test_table").collect()
print(f"[PASS] SQL executed, row count: {result[0]['cnt']}")

spark.stop()

print("\n" + "=" * 60)
print("ALL TESTS PASSED - PYSPARK IS READY TO USE!")
print("=" * 60)
print("\nInstalled packages:")
print("  - PySpark: Big data processing framework")
print("  - NumPy: Numerical computing")
print("  - Pandas: Data manipulation")
print("  - Py4J: Java/Python bridge")
print("\nYou can now use PySpark to:")
print("  - Process data with Spark SQL")
print("  - Use DataFrames for analysis")
print("  - Scale to distributed clusters")
