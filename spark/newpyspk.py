from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder \
    .appName("PySpark-Example").getOrCreate()

employees_data = [
    ("Alice", "Sales", 70000, 5)]

df = spark.createDataFrame(
    employees_data,
    schema=["Employee", "Department", "Salary", "Years"] )

print("Dataset: Employee Information")