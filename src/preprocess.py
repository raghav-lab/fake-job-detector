from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws

# ---------------------------------------
# Create Spark Session
# ---------------------------------------
spark = SparkSession.builder \
    .appName("Fake Job Detector") \
    .getOrCreate()

# ---------------------------------------
# Load Dataset (Correct way)
# ---------------------------------------
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("multiLine", "true") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .csv("data/fake_job_postings.csv")

# ---------------------------------------
# Dataset Information
# ---------------------------------------
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Total Rows    : {df.count()}")
print(f"Total Columns : {len(df.columns)}")

print("\nColumn Names:\n")
for column in df.columns:
    print(column)

# ---------------------------------------
# Target Variable Distribution
# ---------------------------------------
print("\n" + "=" * 60)
print("TARGET VARIABLE DISTRIBUTION")
print("=" * 60)

df.groupBy("fraudulent").count().show()

# ---------------------------------------
# Missing Values
# ---------------------------------------
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

for column in df.columns:
    null_count = df.filter(col(column).isNull()).count()
    print(f"{column:<25}: {null_count}")

# ---------------------------------------
# Replace Null Values
# ---------------------------------------
df = df.fillna("")

# ---------------------------------------
# Create Combined Text Column
# ---------------------------------------
df = df.withColumn(
    "text",
    concat_ws(
        " ",
        col("title"),
        col("company_profile"),
        col("description"),
        col("requirements"),
        col("benefits")
    )
)

# ---------------------------------------
# Preview
# ---------------------------------------
print("\n" + "=" * 60)
print("TEXT COLUMN CREATED SUCCESSFULLY")
print("=" * 60)

df.select("text", "fraudulent").show(5, truncate=100)

# ---------------------------------------
# Stop Spark
# ---------------------------------------
spark.stop()