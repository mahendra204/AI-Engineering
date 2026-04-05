from pyspark.sql import SparkSession

# Create Spark session with Windows-compatible settings
spark = SparkSession.builder \
    .appName("DisplayDF") \
    .master("local[1]") \
    .config("spark.sql.shuffle.partitions", "1") \
    .config("spark.default.parallelism", "1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Create the same dataframe
df = spark.createDataFrame([(1, "Alice"), (2, "Bob"), (3, "Charlie")], schema=["id", "name"])

# Display the dataframe
print("Your DataFrame:")
print("=" * 30)
df.show()
print("=" * 30)
