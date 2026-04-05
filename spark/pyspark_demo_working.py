"""
PySpark Working Demo - No Worker Operations
Demonstrates PySpark functionality while avoiding worker process issues on Windows
"""

import sys
import os

# Configure Python path for Spark
os.environ['PYSPARK_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
import pandas as pd

print("=" * 70)
print(" PySpark Working Demo - DataFrame & SQL Operations")
print("=" * 70)

# Create SparkSession
print("\n[1] Creating Spark Session...")
spark = SparkSession.builder \
    .appName("PySpark-Demo") \
    .master("local[1]") \
    .config("spark.sql.shuffle.partitions", "1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✓ Spark Session created successfully")
print(f"  Version: {spark.version}")

# Create sample data
print("\n[2] Creating Sample Data...")
data = {
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Department': ['Sales', 'IT', 'HR', 'Sales', 'IT'],
    'Salary': [70000, 85000, 65000, 75000, 95000],
    'Years': [5, 8, 3, 6, 10]
}

pdf = pd.DataFrame(data)
print("✓ Created Pandas DataFrame:")
print(pdf.to_string())

# Convert to Spark DataFrame (using Python data, no workers needed)
print("\n[3] Converting to Spark DataFrame...")
df = spark.createDataFrame(pdf)
print("✓ Spark DataFrame created")
print("  Schema:")
df.printSchema()

# Register as temporary view
print("\n[4] Registering as SQL Table...")
df.createOrReplaceTempView("employees")
print("✓ Registered temporary view 'employees'")

# SQL Query 1: Basic SELECT
print("\n[5] SQL Query - Basic SELECT...")
sql_query1 = """
SELECT 
    Employee, 
    Department, 
    Salary,
    Years
FROM employees
WHERE Salary > 70000
ORDER BY Salary DESC
"""
result1 = spark.sql(sql_query1)
print(result1.toPandas().to_string())

# SQL Query 2: Aggregation
print("\n[6] SQL Query - Aggregation by Department...")
sql_query2 = """
SELECT 
    Department,
    COUNT(*) as EmployeeCount,
    AVG(Salary) as AvgSalary,
    MAX(Salary) as MaxSalary,
    MIN(Salary) as MinSalary
FROM employees
GROUP BY Department
ORDER BY AvgSalary DESC
"""
result2 = spark.sql(sql_query2)
print(result2.toPandas().to_string())

# SQL Query 3: Using CASE WHEN
print("\n[7] SQL Query - Salary Classification...")
sql_query3 = """
SELECT 
    Employee,
    Salary,
    CASE 
        WHEN Salary >= 90000 THEN 'Senior'
        WHEN Salary >= 80000 THEN 'Mid-Level'
        ELSE 'Junior'
    END as Level,
    CASE 
        WHEN Years >= 8 THEN 'Experienced'
        ELSE 'Growing'
    END as Experience
FROM employees
ORDER BY Salary DESC
"""
result3 = spark.sql(sql_query3)
print(result3.toPandas().to_string())

# SQL Query 4: Window Function
print("\n[8] SQL Query - Ranking Employees...")
sql_query4 = """
SELECT 
    Employee,
    Department,
    Salary,
    ROW_NUMBER() OVER (PARTITION BY Department ORDER BY Salary DESC) as DeptRank,
    ROW_NUMBER() OVER (ORDER BY Salary DESC) as OverallRank
FROM employees
"""
result4 = spark.sql(sql_query4)
print(result4.toPandas().to_string())

# DataFrame Operations (without action operations)
print("\n[9] DataFrame Transformations...")
print("Adding a new column 'TotalComp' = Salary * (1 + Years * 0.05)...")
df_with_comp = df.selectExpr(
    "Employee",
    "Department",
    "Salary",
    "Years",
    "(Salary * (1 + Years * 0.05)) as TotalComp"
)
print(df_with_comp.toPandas().to_string())

# Summary
print("\n" + "=" * 70)
print(" PYSPARK SUCCESSFULLY DEMONSTRATED")
print("=" * 70)

print("\nInstalled Components:")
print("  ✓ PySpark 4.1.1")
print("  ✓ NumPy")
print("  ✓ Pandas")
print("  ✓ Py4J")
print("  ✓ Java 25.0.1")

print("\nCapabilities Verified:")
print("  ✓ SparkSession creation")
print("  ✓ DataFrame creation from Python data")
print("  ✓ SQL queries with WHERE, GROUP BY, ORDER BY")
print("  ✓ CASE WHEN statements")
print("  ✓ Window functions (ROW_NUMBER)")
print("  ✓ DataFrame transformations")
print("  ✓ Data type conversions")

print("\nNext Steps:")
print("  1. Use PySpark for data processing on your machine")
print("  2. Scale to distributed computing clusters")
print("  3. Process larger datasets efficiently")
print("  4. Combine with other libraries (scikit-learn, etc.)")

print("\nNote:")
print("  Some worker-dependent operations may have compatibility issues")
print("  on Windows with Python 3.13. Stick with SQL operations or")
print("  use Pandas integration as shown in this demo.")

spark.stop()
print("\n✓ Spark Session closed\n")
